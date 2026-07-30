"""Правка сущностей через ЗАМЕНЫ: модель предлагает пары «было → стало», применяет их код.

Почему так, а не свободным переписыванием текста — уроки, купленные живыми прогонами:
  • обрезанный ответ модели молча удалил 365 слов речи из реплики (ep11);
  • верно распознанный «Колодезев» вернулся гарблом после правки (ep11);
  • запрет переводить в промпте не сработал вовсе — 65 замен письма стали 69 (ep9).
Эвристики постфактум (объём ответа, смена письма) — гадание: LLM недетерминирована. Здесь модель
физически не может удалить речь или переписать прозу, потому что прозу у неё не берут.
"""
import pytest

from stages.final_round import apply_fixes
from stages.glossary import _HAS_WF


def test_fix_is_applied_verbatim():
    text = 'нам рассказал Василедец про разложение'

    out, applied, skipped = apply_fixes(text, [{'was': 'Василедец', 'now': 'Оселедец'}])

    assert out == 'нам рассказал Оселедец про разложение'
    assert (applied, skipped) == (1, 0)


def test_multiword_garble_is_supported():
    """Гарбл разъезжается на соседнее слово — ради этого `was` и допускает до трёх слов."""
    text = 'об этом говорил и Селедец на конференции'

    out, applied, _ = apply_fixes(text, [{'was': 'и Селедец', 'now': 'Иван Оселедец'}])

    assert out == 'об этом говорил Иван Оселедец на конференции'
    assert applied == 1


def test_text_outside_fixes_is_byte_for_byte():
    """Главная гарантия: речь вне замен измениться не может — её никто не переписывает."""
    text = 'первая строка\n\n  вторая  строка с Василедец  и хвостом'

    out, _, _ = apply_fixes(text, [{'was': 'Василедец', 'now': 'Оселедец'}])

    assert out == 'первая строка\n\n  вторая  строка с Оселедец  и хвостом'


def test_fix_not_found_in_text_is_dropped():
    """Модель выдумала подстроку — заменять нечего, текст не трогаем."""
    text = 'обычная реплика без сущностей'

    out, applied, skipped = apply_fixes(text, [{'was': 'ChatGPT', 'now': 'ChatGPT-4'}])

    assert out == text and (applied, skipped) == (0, 1)


def test_long_span_is_dropped():
    """Больше трёх слов — это уже не сущность, а попытка переписать фразу."""
    text = 'мы обсуждали это очень долго и подробно вчера'

    out, applied, skipped = apply_fixes(
        text, [{'was': 'это очень долго и подробно', 'now': 'это подробно'}])

    assert out == text and (applied, skipped) == (0, 1)


def test_known_term_cannot_be_destroyed():
    """«Колодезев» распознан верно — каноник мы знаем точно, ломать его нельзя."""
    text = 'со мной Дмитрий Колодезев сегодня'

    out, applied, skipped = apply_fixes(
        text, [{'was': 'Дмитрий Колодезев', 'now': 'Дмитрий Колодзев'}],
        always=['Дмитрий Колодезев'])

    assert out == text and (applied, skipped) == (0, 1)


@pytest.mark.skipif(not _HAS_WF, reason='нужен wordfreq (он в requirements.txt)')
@pytest.mark.parametrize('was, now', [
    ('агенты', 'agents'),
    ('умные очки', 'smart glasses'),
    ('финансирования', 'funding'),
])
def test_translation_is_dropped(was, now):
    """Перевод обычной речи — не канонизация. Промптом это не лечится, проверено прогоном."""
    text = f'тут про {was} речь'

    out, applied, skipped = apply_fixes(text, [{'was': was, 'now': now}])

    assert out == text and (applied, skipped) == (0, 1)


@pytest.mark.skipif(not _HAS_WF, reason='нужен wordfreq')
@pytest.mark.parametrize('was, now', [
    ('сам Артман', 'Sam Altman'),      # имя с заглавной, а не перевод
    ('аж двести', 'H200'),             # обозначение с цифрой
    ('джапити', 'ChatGPT'),            # редкий источник — гарбл
    ('перле', 'Perl'),
])
def test_legitimate_canonicalisation_passes(was, now):
    text = f'вот тут {was} упоминается'

    out, applied, _ = apply_fixes(text, [{'was': was, 'now': now}])

    assert now in out and applied == 1


@pytest.mark.skipif(not _HAS_WF, reason='нужен wordfreq')
def test_glossary_sanctioned_translation_passes():
    """Каноник пришёл из глоссария — замена санкционирована, а не самодеятельность модели."""
    text = 'там работают агенты постоянно'

    out, applied, _ = apply_fixes(text, [{'was': 'агенты', 'now': 'AI agents'}],
                                  canonicals=['AI agents'])

    assert 'AI agents' in out and applied == 1


def test_empty_and_noop_fixes_are_counted_as_skipped():
    text = 'реплика про NVIDIA'

    out, applied, skipped = apply_fixes(text, [
        {'was': '', 'now': 'X'}, {'was': 'NVIDIA', 'now': 'NVIDIA'}, {'was': 'NVIDIA', 'now': ''}])

    assert out == text and (applied, skipped) == (0, 3)


def test_no_fixes_leaves_text_alone():
    text = 'ничего править не надо'

    assert apply_fixes(text, []) == (text, 0, 0)
    assert apply_fixes(text, None) == (text, 0, 0)


def test_replacement_respects_word_boundaries():
    """«СМЛ»→«ASML» попала внутрь слова «АСМЛ» и дала «АASML» — поймано на живом прогоне ep11."""
    text = 'литографы АСМЛ поставляют'

    out, applied, skipped = apply_fixes(text, [{'was': 'СМЛ', 'now': 'ASML'}])

    assert out == text and (applied, skipped) == (0, 1)


def test_whole_word_is_still_replaced():
    text = 'литографы СМЛ поставляют, и ещё раз СМЛ'

    out, applied, _ = apply_fixes(text, [{'was': 'СМЛ', 'now': 'ASML'}])

    assert out == 'литографы ASML поставляют, и ещё раз ASML' and applied == 1


def test_replacement_with_regex_specials_is_literal():
    """`now` подставляется как текст: обратные слэши не должны трактоваться как группы."""
    text = 'модель си плюс плюс тут'

    out, applied, _ = apply_fixes(text, [{'was': 'си плюс плюс', 'now': 'C++'}])

    assert out == 'модель C++ тут' and applied == 1


def test_partial_fix_cannot_break_a_known_term():
    """Замена приходит на одну фамилию, а термин — полное имя: строкового пересечения нет.

    Поймано на ep13: «Колодезев»→«Колодзев» проскочило мимо проверки, сравнивавшей строки, и
    сломало верно распознанное имя. Проверка идёт ПО РЕЗУЛЬТАТУ применения.
    """
    text = 'с вами Валентин Малых и Дмитрий Колодезев сегодня'

    out, applied, skipped = apply_fixes(
        text, [{'was': 'Колодезев', 'now': 'Колодзев'}], always=['Дмитрий Колодезев'])

    assert out == text and (applied, skipped) == (0, 1)


def test_fix_elsewhere_still_applies_when_term_is_intact():
    """Барьер не должен запрещать всё подряд — только то, что рушит термин."""
    text = 'Дмитрий Колодезев рассказал про Визделе'

    out, applied, _ = apply_fixes(
        text, [{'was': 'Визделе', 'now': 'VHDL'}], always=['Дмитрий Колодезев'])

    assert out == 'Дмитрий Колодезев рассказал про VHDL' and applied == 1


def test_known_term_is_protected_in_any_case_form():
    """«спросили Колодезева» не содержит «Дмитрий Колодезев» — сравнения строк целиком мало.

    Считаем вхождения слов термина: подстрока снимает падежи даром, «Колодезева» содержит
    «Колодезев», а гарбл «Колодзева» — нет. Стеммер для этого не нужен.
    """
    text = 'вчера мы спросили Колодезева про это'

    out, applied, skipped = apply_fixes(
        text, [{'was': 'Колодезева', 'now': 'Колодзева'}], always=['Дмитрий Колодезев'])

    assert out == text and (applied, skipped) == (0, 1)


def test_unrelated_fix_passes_with_stemming_on():
    text = 'Колодезев рассказал про Визделе'

    out, applied, _ = apply_fixes(
        text, [{'was': 'Визделе', 'now': 'VHDL'}], always=['Дмитрий Колодезев'])

    assert out == 'Колодезев рассказал про VHDL' and applied == 1


@pytest.mark.parametrize('text, was, now', [
    ('переговоры между Roosevelt и Churchill в Casablanca', 'Roosevelt и Churchill', 'Roosevelt'),
    ('поляны от Visa и Mastercard вот', 'Visa и Mastercard', 'Visa'),
    ('ну конечно NVIDIA TSMC отпраздновали', 'NVIDIA TSMC', 'NVIDIA'),
    ('передавая её в Unicode в Promptaf например', 'Unicode в Promptaf', 'Unicode'),
])
def test_excision_is_dropped(text, was, now):
    """Замена ничего не принесла, только выкинула слова — это редактирование речи, не канонизация.

    Замерено на четырёх выпусках: 19 таких случаев, восемь меняют смысл, в четырёх исчезают
    названные вслух сущности. Плюс так модель выбрасывает то, чего не смогла починить.
    """
    out, applied, skipped = apply_fixes(text, [{'was': was, 'now': now}])

    assert out == text and (applied, skipped) == (0, 1)


@pytest.mark.parametrize('was, now', [
    ('аж двести', 'H200'),                 # сжатие, но каноник новый
    ('си плюс плюс', 'C++'),
    ('и Селедец', 'Иван Оселедец'),
    ('Клод Код', 'Claude Code'),
])
def test_compression_with_new_words_still_passes(was, now):
    """Канонизация вправе сжимать — лишь бы приносила то, чего в исходной фразе не было."""
    text = f'вот тут {was} упоминается'

    out, applied, _ = apply_fixes(text, [{'was': was, 'now': now}])

    assert now in out and applied == 1


@pytest.mark.parametrize('was, now', [
    ('H200', 'H100'),          # контекст соседей «нормализовал» верное обозначение (ep20)
    ('10.80', 'A100'),         # речь про GTX 1080, соседи про A100
    ('ADAS-128', 'ADAS-256'),
])
def test_number_change_is_dropped(was, now):
    """Число в обозначении решает акустика, не контекст — зеркало правила глоссария."""
    text = f'производительность четверть от {was} примерно'

    out, applied, skipped = apply_fixes(text, [{'was': was, 'now': now}])

    assert out == text and (applied, skipped) == (0, 1)


@pytest.mark.parametrize('was, now', [
    ('аж двести', 'H200'),     # слева цифр нет — класс-2 канонизация законна
    ('10.80', '10-80'),        # цифры те же, поменялось только письмо
    ('АДМС-128', 'ADAS-128'),
    ('один т.', '1 трлн.'),
])
def test_number_preserving_fixes_pass(was, now):
    text = f'там {was} стояло'

    out, applied, _ = apply_fixes(text, [{'was': was, 'now': now}])

    assert now in out and applied == 1

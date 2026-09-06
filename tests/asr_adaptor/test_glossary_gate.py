"""Гейт редкости в глоссарии: гарбл проходит, перевод обычного слова — нет.

Дыра, которую эти тесты закрывают: раньше ЛЮБОЙ латинский каноник проходил мимо гейта, и глоссарий
принимал «агенты»→agents, «умные очки»→smart glasses. На корпусе это дало 2069 замен кириллицы на
латиницу против 56 обратных. Промптом не лечится — проверено живым прогоном (65→69 замен).
"""
import pytest

from stages.glossary import _HAS_WF, _keep_one

pytestmark = pytest.mark.skipif(not _HAS_WF, reason='нужен wordfreq (он в requirements.txt)')


@pytest.mark.parametrize('heard, canonical', [
    ('джапити', 'ChatGPT'),
    ('стенгаус', 'Westinghouse'),
    ('василедец', 'Oseledets'),
    ('клоду', 'Claude'),
    ('перле', 'Perl'),
])
def test_garble_passes(heard, canonical):
    """Редкое слово — это гарбл ASR, ради него глоссарий и существует."""
    assert _keep_one(heard, canonical)


@pytest.mark.parametrize('heard, canonical', [
    ('агенты', 'agents'),
    ('способностями', 'capabilities'),
    ('умные очки', 'smart glasses'),
    ('холодного бэкапа', 'cold backup'),
])
def test_translation_of_common_word_rejected(heard, canonical):
    """Частотное русское слово — не гарбл. Каноник тут был бы переводом, а переводы запрещены."""
    assert not _keep_one(heard, canonical)


@pytest.mark.parametrize('heard, canonical', [
    ('аж двести', 'H200'),
    ('а сто', 'A100'),
    ('десять-восемьдесят', '1080'),
    ('джипити четыре', 'GPT-4'),
])
def test_model_numbers_survive_the_rarity_gate(heard, canonical):
    """Номера моделей проговаривают обычными числительными — гейт редкости их бы вырезал.

    Это флагман Класса-2 («услышал верно, не понял домен»): аудио «проигрывает аж двести» плюс
    каноник в подсказке → ASR пишет H200. Без исключения на цифру в канонике механизм умирает —
    поймано на живой проверке гейта.
    """
    assert _keep_one(heard, canonical)


def test_pure_latin_heard_passes():
    """ASR уже написал латиницей — канонизируем написание, а не переводим."""
    assert _keep_one('Chat GPT', 'ChatGPT')


def test_mixed_heard_is_gated_by_its_cyrillic_part():
    """Латиница рядом не выкупает частотное русское слово: «frontier способностями» — перевод.

    Через эту смешанную форму дыра и держалась после первой правки (49 замен из 65).
    """
    assert not _keep_one('frontier способностями', 'frontier capabilities')
    assert _keep_one('frontier кардреллы', 'frontier guardrails')  # редкое рядом с латиницей — гарбл


def test_cyrillic_canonical_for_rare_heard_passes():
    """Каноник кириллицей — тоже канонизация: «Василедец» → «Оселедец»."""
    assert _keep_one('василедец', 'Оселедец')


@pytest.mark.parametrize('heard, canonical', [
    ('слово', ''),
    ('слово', 'x' * 41),
    ('я' * 41, 'ChatGPT'),
    ('слово', 'канон\nс переводом строки'),
])
def test_guards(heard, canonical):
    """Мусор в каноник не пускаем: пустое, абзац, неправдоподобная длина."""
    assert not _keep_one(heard, canonical)


@pytest.mark.parametrize('heard, canonical', [
    ('институт Айри', 'AIRI'),          # обычный сосед не должен убивать имя
    ('фирма Вестингауз', 'Westinghouse'),
    ('модель Клод', 'Claude'),
    ('библиотека тензор трейн', 'Tensor Train'),  # у имени есть редкий токен, у перевода нет
])
def test_common_neighbour_does_not_kill_a_name(heard, canonical):
    """Промпт просит писать `heard` с соседним словом — из-за него запись вылетала.

    Поймано живым прогоном: AIRI ушёл из сырого текста (8 упоминаний → 1), потому что «институт Айри»
    отсекалось по слову «институт». Приговор выносится не по русской стороне, а по паре признаков.
    """
    assert _keep_one(heard, canonical)


@pytest.mark.parametrize('heard, canonical', [
    ('умные очки', 'smart glasses'),
    ('холодного бэкапа', 'cold backup'),
    ('много агентов', 'AI agents'),
])
def test_plain_english_canonical_is_still_rejected(heard, canonical):
    """Перевод — это когда обычное русское меняется на сплошь обычное английское."""
    assert not _keep_one(heard, canonical)


class _RouletteLLM:
    """Каждый вызов отдаёт разные термины — как настоящий провайдер при temperature=0."""

    def __init__(self):
        self.calls = 0

    async def complete_json(self, messages, **kw):
        self.calls += 1
        if self.calls == 1:
            raise ValueError('LLM returned invalid JSON')  # деген-петля первого вызова
        return {'terms': [{'heard': f'гарблик{self.calls}', 'canonicals': [f'Term{self.calls}']}]}


async def test_glossary_unions_passes_and_survives_failed_batch():
    """Ретраи вызова живут в RetryingLLM (config.py), не в стадии: упавший батч скипается с логом,
    второй проход страхует объединением (одиночный прогон терял до половины recall)."""
    from stages.glossary import build_glossary
    llm = _RouletteLLM()

    out = await build_glossary('Одно предложение про гарблик.', llm, passes=2)

    assert llm.calls == 2
    assert {t['heard'] for t in out} == {'гарблик2'}


class _DeadLLM:
    """Эндпоинт лежит: каждый вызов срывается после всех ретраев RetryingLLM."""

    def __init__(self):
        self.calls = 0

    async def complete_json(self, messages, **kw):
        self.calls += 1
        raise ConnectionError('endpoint unreachable')


class _SilentLLM:
    """Эндпоинт жив, но терминов в тексте не нашёл — это НЕ отказ."""

    async def complete_json(self, messages, **kw):
        return {'terms': []}


async def test_glossary_raises_when_every_call_failed():
    """Мёртвый эндпоинт обязан ронять джобу, а не отдавать пустой глоссарий.

    Пустой глоссарий неотличим от «терминов не нашлось», а последствия разные: без подсказок
    пасса-2 и каноников финал-раунда запись выходит заметно хуже, но по виду нормальной (на
    корпусе митапов глоссарий чинит 74% доменных гарблов). Упавшая джоба не оставляет `.json`,
    поэтому `run_folder.sh` возьмёт запись заново на следующем прогоне — без человека.
    """
    from stages.glossary import build_glossary
    llm = _DeadLLM()

    with pytest.raises(RuntimeError, match='все .* вызовов LLM сорвались'):
        await build_glossary('Одно предложение про гарблик.', llm, passes=2)
    assert llm.calls == 2  # оба прохода честно попробованы


async def test_glossary_survives_when_most_but_not_all_calls_failed():
    """Порога «сорвалось больше половины» нет намеренно: уцелевший вызов лучше, чем ничего."""
    from stages.glossary import build_glossary

    class _MostlyDead:
        def __init__(self):
            self.calls = 0

        async def complete_json(self, messages, **kw):
            self.calls += 1
            if self.calls < 4:
                raise ConnectionError('endpoint unreachable')
            return {'terms': [{'heard': 'гарблик', 'canonicals': ['Garblic']}]}

    llm = _MostlyDead()
    # Четыре прохода по одному батчу: три вызова мертвы, четвёртый жив — глоссарий из него.
    out = await build_glossary('Одно предложение про гарблик.', llm, passes=4)

    assert llm.calls == 4
    assert {t['heard'] for t in out} == {'гарблик'}


async def test_glossary_empty_result_is_not_a_failure():
    """Живой эндпоинт, не нашедший терминов, джобу ронять не должен: у короткой бытовой записи
    пустой глоссарий — законный исход."""
    from stages.glossary import build_glossary

    assert await build_glossary('Привет, как дела.', _SilentLLM(), passes=2) == []

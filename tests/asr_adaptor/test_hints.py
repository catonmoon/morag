"""Seed-проход глоссария: заведомо верные написания против черновика.

Канал ведёт ПРЯМО в подсказку пасса-2, а она — единственное место конвейера без последующей
проверки: что туда положили, то и повлияло на декодирование. Поэтому тесты здесь не про «нашлось
ли», а про то, что через валидацию НЕ пролезет.
"""
import pytest

from stages.glossary import _HAS_WF
from stages.hints import SIM_MIN, build_hints, hinted, merge, similar
from stages.prompt_budget import build_prompt

TERMS = ['Постгрес', 'Редис', 'LDAP', 'Графана']
NAMES = ['Кузнецова']


class _LLM:
    """Отдаёт заранее заданный ответ и запоминает, о чём его спросили."""

    def __init__(self, found):
        self.found, self.system, self.user = found, None, None

    async def complete_json(self, messages, **kw):
        self.system, self.user = messages[0]['content'], messages[1]['content']
        return {'found': self.found}


async def _run(found, text, terms=TERMS, names=NAMES):
    return await build_hints(text, _LLM(found), terms=terms, names=names, about='тест')


@pytest.mark.asyncio
async def test_confirmed_garble_becomes_a_glossary_entry():
    got = await _run([{'term': 'Постгрес', 'heard': 'пост грез'}], 'связь пост грез с ядром')
    assert got == [{'heard': 'пост грез', 'canonicals': ['Постгрес']}]


@pytest.mark.asyncio
async def test_term_outside_the_offered_list_is_dropped():
    """Несущее свойство: модель физически не может протащить термин, которого снаружи не давали."""
    assert await _run([{'term': 'agents', 'heard': 'агенты'}], 'агенты работают') == []


@pytest.mark.asyncio
async def test_garble_absent_from_the_draft_is_dropped():
    """Главный режим отказа — перечислить весь список с выдуманными искажениями."""
    assert await _run([{'term': 'LDAP', 'heard': 'ЭлДАП'}], 'разговор без термина вовсе') == []


@pytest.mark.asyncio
async def test_correctly_written_term_is_dropped():
    """Чинить нечего, а бюджет подсказки 200 токенов — место дорогое."""
    assert await _run([{'term': 'LDAP', 'heard': 'LDAP'}], 'сервис LDAP отвечает') == []


@pytest.mark.asyncio
async def test_surname_pinned_to_a_wrong_spot_is_dropped():
    """Под фамилией живой человек: приписать её чужому месту хуже, чем оставить гарбл.

    Замерено на живом прогоне: модель вернула фамилию, приписанную к чужому по звучанию месту
    (похожесть 0.47), рядом с верной привязкой той же формы. Порог режет первую, оставляя вторую.
    """
    assert await _run([{'term': 'Кузнецова', 'heard': 'Ковалёва'}], 'сказала Ковалёва нам') == []


@pytest.mark.asyncio
@pytest.mark.skipif(not _HAS_WF, reason='нужен wordfreq (он в requirements.txt)')
async def test_translation_is_dropped_by_the_rarity_gate():
    got = await _run([{'term': 'data', 'heard': 'данные'}], 'данные лежат тут', terms=['data'])
    assert got == []


@pytest.mark.asyncio
async def test_one_spot_keeps_both_offered_spellings():
    """Латиница и кириллица — законные соседи; какая форма легла в звук, решает акустика."""
    got = await _run([{'term': 'Grafana', 'heard': 'Grafanna'}, {'term': 'Графана', 'heard': 'Grafanna'}],
                     'сервис Grafanna отвечает', terms=['Графана', 'Grafana'])
    assert got == [{'heard': 'Grafanna', 'canonicals': ['Grafana', 'Графана']}]


@pytest.mark.asyncio
async def test_empty_input_costs_no_llm_call():
    """Нет знания о записи — нет и вызова: подкаст ходит прежним путём и не платит за него."""
    llm = _LLM([{'term': 'Постгрес', 'heard': 'пост грез'}])
    assert await build_hints('связь пост грез', llm, terms=(), names=()) == []
    assert llm.system is None


@pytest.mark.asyncio
async def test_llm_failure_does_not_raise():
    """Сорвавшийся seed оставляет запись такой, какой она выходила до нас, — ронять нечего."""
    class _Dead:
        async def complete_json(self, *a, **kw):
            raise RuntimeError('endpoint down')

    assert await build_hints('пост грез', _Dead(), terms=TERMS) == []


def test_measured_pairs_pass_the_similarity_threshold():
    """Порог не догадка: он замерен на подтверждённых правках корпуса, минимум там 0.57."""
    for heard, term in [('Grafanna', 'Grafana'), ('партишн', 'partition'), ('LEDAP', 'LDAP'),
                        ('Кознецова', 'Кузнецова'), ('Redisом', 'Редисом'),
                        ('пост грез', 'Постгрес'), ('Прокровский', 'Покровский')]:
        assert similar(heard, term) >= SIM_MIN, (heard, term)


def test_invented_pairs_fall_below_the_threshold():
    for heard, term in [('Ковалёва', 'Кузнецова'), ('SMTP', 'Редис'), ('XY', 'Ковалёв'),
                        ('CV', 'Grafana'), ('данных', 'Data')]:
        assert similar(heard, term) < SIM_MIN, (heard, term)


def test_seed_goes_first_and_wins_the_duplicate():
    """Порядок = приоритет в подсказке: подтверждённое знание раньше догадки по тому же черновику."""
    seed = [{'heard': 'пост грез', 'canonicals': ['Постгрес']}]
    gloss = [{'heard': 'пост грез', 'canonicals': ['Postgres']}, {'heard': 'кафку', 'canonicals': ['Kafka']}]
    assert merge(seed, gloss) == seed + [gloss[1]]


def test_merge_without_hints_keeps_the_glossary_byte_for_byte():
    gloss = [{'heard': 'кафку', 'canonicals': ['Kafka']}]
    assert merge([], gloss) == gloss


class _Counter:
    """Счёт по словам вместо whisper-токенайзера: тест не должен тянуть transformers."""

    def count(self, text):
        return len(text.split())

    def truncate(self, text, limit):
        return ' '.join(text.split()[:limit])


def test_confirmed_cyrillic_canonical_reaches_the_prompt():
    """Ради этого всё и затевалось: 79% наших правок кириллические, и фильтр латиницы их убивал."""
    seed = [{'heard': 'пост грез', 'canonicals': ['Постгрес']}]
    assert 'Постгрес' not in build_prompt(['Постгрес'], _Counter())
    assert 'Постгрес' in build_prompt(['Постгрес'], _Counter(), hinted=hinted(seed))


def test_unconfirmed_cyrillic_still_filtered_out():
    """Освобождение — только за четыре звена, а не всем подряд."""
    assert 'Постгрес' not in build_prompt(['Постгрес'], _Counter(), hinted=hinted([]))

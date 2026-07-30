"""Подсказка пасса-2: постоянные термины корпуса впереди канонико́в куска.

Зачем постоянные: глоссарий строится заново на каждом выпуске и на устойчивом гарбле срывается —
по корпусу «Колодзев» вместо «Колодезев» в 14 репликах, `MotorMost` вместо `Mattermost` в 24,
причём финал-раунд их не чинит. A/B на интро ep11 показал, что кириллическое имя в подсказке гарбл
снимает, а прежний фильтр «только латиница» этого не позволял.
"""
from stages.prompt_budget import PREFIX, build_prompt, fit_prompt


class FakeCounter:
    """Считает по словам — точный whisper-токенайзер тут не нужен, важен порядок и обрезка."""

    def count(self, text: str) -> int:
        return len(text.split())

    def truncate(self, text: str, limit: int) -> str:
        return ' '.join(text.split()[:limit])


def test_always_terms_come_first():
    out = build_prompt(['ChatGPT', 'NVIDIA'], FakeCounter(), 100, ['Дмитрий Колодезев'])

    assert out.startswith(PREFIX)
    assert out.index('Колодезев') < out.index('ChatGPT')


def test_cyrillic_always_term_survives_the_latin_filter():
    """Каноники куска фильтруются по латинице, постоянные термины — нет."""
    out = build_prompt(['Оселедец'], FakeCounter(), 100, ['Валентин Малых'])

    assert 'Валентин Малых' in out
    assert 'Оселедец' not in out  # кириллический каноник куска по-прежнему отсекается


def test_budget_sacrifices_chunk_canonicals_not_names():
    """Бюджет ≤200 токенов: терять хвост случайных канонико́в куска дешевле, чем фамилию ведущего."""
    canon = [f'Term{i}' for i in range(50)]

    out = build_prompt(canon, FakeCounter(), 12, ['Дмитрий Колодезев'])

    assert 'Колодезев' in out
    assert 'Term49' not in out


def test_duplicates_do_not_eat_budget():
    out = build_prompt(['Mattermost', 'ChatGPT'], FakeCounter(), 100, ['Mattermost'])

    assert out.count('Mattermost') == 1


def test_no_terms_gives_empty_prompt():
    assert build_prompt([], FakeCounter(), 100, []) == ''
    assert fit_prompt([], FakeCounter(), 100) == ''

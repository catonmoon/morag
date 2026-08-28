"""Описание корпуса подставляется в промпты правки и наминга (ASR_CORPUS_DESC).

Конвейер generic и жанр записи знать не должен, но подсказка модели о домене — доменная: на
записях рабочих встреч правка, настроенная на «подкаст про технологии», тянет канонизацию не туда.
Здесь проверяется, что строка доезжает до промпта, дефолт остаётся прежним, а плейсхолдер не течёт
наружу.
"""
import pytest

from stages import namer
from stages.final_round import DEFAULT_CORPUS, _CORPUS_SLOT, _CORRECT_SYS, correct

WORK = 'рабочей записи: встречи, митапа или обучающего материала'


class _LLM:
    """Ловит system-промпт, которым его позвали."""

    def __init__(self):
        self.system = None

    async def complete_json(self, messages, **kw):
        self.system = messages[0]['content']
        return {'fixes': []}

    async def complete(self, messages, **kw):
        self.system = messages[0]['content']
        return ''


@pytest.mark.asyncio
async def test_corpus_desc_reaches_correction_prompt():
    llm = _LLM()
    await correct('текст', 'сводка', '', [], llm, corpus_desc=WORK)
    assert WORK in llm.system
    assert DEFAULT_CORPUS not in llm.system


@pytest.mark.asyncio
async def test_default_keeps_previous_wording():
    """Без настройки поведение прежнее — иначе правка молча поехала бы на всём корпусе подкаста."""
    llm = _LLM()
    await correct('текст', 'сводка', '', [], llm)
    assert DEFAULT_CORPUS in llm.system


@pytest.mark.asyncio
async def test_naming_prompt_takes_corpus_desc():
    llm = _LLM()
    turns = [{'speaker': 'Speaker_0', 'text': 'всем привет'}]
    await namer.name_speakers(turns, {}, llm, corpus_desc=WORK)
    assert WORK in llm.system


def test_placeholder_never_leaks():
    """Плейсхолдер @CORPUS@ не должен доехать до модели ни в одном промпте."""
    for prompt in (_CORRECT_SYS, namer._SYS, namer._GUESTS_SYS):
        slot = _CORPUS_SLOT if prompt is _CORRECT_SYS else namer._CORPUS_SLOT
        assert slot not in prompt.replace(slot, WORK)


def test_correction_prompt_keeps_literal_json_braces():
    """Подстановка идёт .replace(), а не .format(): в промпте есть литеральные {"fixes": …},
    и format() упал бы на них KeyError."""
    assert '{"fixes"' in _CORRECT_SYS
    assert '{"fixes"' in _CORRECT_SYS.replace(_CORPUS_SLOT, WORK)

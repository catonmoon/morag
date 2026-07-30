"""Одно имя на двух метках — противоречие, а не два человека.

Поймано на ep18 корпусного прогона: LLM отдала «Пётр Ермаков» и ведущему (он представлял гостя),
и гостю; реестр обоих не подтвердил → used_intro дважды → выпуск с двумя Ермаковыми.
"""
from stages.namer import name_speakers


class IntroLLM:
    """Карта самоназваний как её вернула LLM; гостей не находит."""

    def __init__(self, mapping):
        self.mapping = mapping

    async def complete_json(self, messages, schema=None, schema_name='', **kw):
        if schema_name == 'guests':
            return {'guests': []}
        return {'speakers': [{'label': k, 'name': v} for k, v in self.mapping.items()]}


def _turns():
    return [{'speaker': 'Speaker_1', 'start': 0.0,
             'final': 'Всем привет! У нас сегодня в гостях Пётр Ермаков. Петя, привет!'},
            {'speaker': 'Speaker_4', 'start': 12.0,
             'final': 'Да, действительно, Петя Ермаков. Рад, что позвали'},
            {'speaker': 'Speaker_1', 'start': 25.0, 'final': 'Расскажи, чем занимаешься'}]


async def test_duplicate_name_goes_to_the_self_introducer():
    llm = IntroLLM({'Speaker_1': 'Пётр Ермаков', 'Speaker_4': 'Пётр Ермаков'})
    registry = {'Speaker_1': 'Дмитрий Колодезев', 'Speaker_4': 'Алексей Натёкин'}

    names, conflicts = await name_speakers(_turns(), registry, llm)

    assert names['Speaker_4'] == 'Пётр Ермаков'          # гость: отвечает на представление
    assert names['Speaker_1'] == 'Дмитрий Колодезев'     # ведущий: откат к реестру
    assert any(c['resolution'] == 'dropped_duplicate' for c in conflicts)


async def test_duplicate_without_registry_falls_back_to_label():
    """Реестра нет — метка остаётся Speaker_N: невыставленное имя дешевле свопа людей."""
    llm = IntroLLM({'Speaker_1': 'Пётр Ермаков', 'Speaker_4': 'Пётр Ермаков'})

    names, _ = await name_speakers(_turns(), {}, llm)

    assert names['Speaker_4'] == 'Пётр Ермаков'
    assert names['Speaker_1'] == 'Speaker_1'


async def test_distinct_names_are_untouched():
    llm = IntroLLM({'Speaker_1': 'Валентин Малых', 'Speaker_4': 'Пётр Ермаков'})

    names, conflicts = await name_speakers(_turns(), {}, llm)

    assert names == {'Speaker_1': 'Валентин Малых', 'Speaker_4': 'Пётр Ермаков'}
    assert not [c for c in conflicts if c['resolution'] == 'dropped_duplicate']

"""Наминг на случаях перепрогона «Капитанского мостика» (2026-10-13).

ep1-5: реестр пишет «Натёкин», интро — «Натекин»; без свёртки ё имя «не звучало» в интро, реестр не
подтверждался, и ведущий выпуска становился «Леша Натекин».
ep2-20: гость сматчен с чужим голосом на пороге, а интро-LLM выдала его метке имя другого ведущего.
Эта запись занимала метку, исключение «один гость — одна свободная метка» не срабатывало, и 3989
слов гостя оставались под чужим именем.
"""
from stages.namer import name_speakers


class LLM:
    def __init__(self, speakers, guests=()):
        self.speakers, self.guests = speakers, list(guests)

    async def complete_json(self, messages, schema=None, schema_name='', **kw):
        if schema_name == 'guests':
            return {'guests': self.guests}
        return {'speakers': [{'label': k, 'name': v} for k, v in self.speakers.items()]}


async def test_yo_does_not_break_registry_corroboration():
    turns = [{'speaker': 'Speaker_0', 'start': 0.0,
              'final': 'Меня зовут Валентин Малых, с нами сегодня Леша Натекин.'},
             {'speaker': 'Speaker_4', 'start': 14.0, 'final': 'Всем привет!'}]
    registry = {'Speaker_0': 'Валентин Малых', 'Speaker_4': 'Алексей Натёкин'}

    names, conflicts = await name_speakers(turns, registry, LLM({'Speaker_4': 'Леша Натекин'}))

    assert names['Speaker_4'] == 'Алексей Натёкин'
    assert conflicts[0]['reason'] == 'registry_corroborated'


async def test_host_name_on_guest_label_does_not_block_the_guest():
    turns = [{'speaker': 'Speaker_0', 'start': 0.0,
              'final': 'С вами Валентин Малых и Дмитрий Колодезев. Затащили в гости Валеру Бабушкина.'},
             {'speaker': 'Speaker_4', 'start': 88.0, 'final': 'Не первый раз он куда-то выходит.'},
             {'speaker': 'Speaker_1', 'start': 140.0, 'final': 'Ну, кстати, его спросили в Твиттере.'}]
    registry = {'Speaker_0': 'Валентин Малых', 'Speaker_1': 'Дмитрий Колодезев',
                'Speaker_4': 'Алексей Натёкин'}
    llm = LLM({'Speaker_0': 'Валентин Малых', 'Speaker_4': 'Дмитрий Колодезев'},
              guests=['Валерий Бабушкин'])

    names, conflicts = await name_speakers(turns, registry, llm)

    assert names == {'Speaker_0': 'Валентин Малых', 'Speaker_1': 'Дмитрий Колодезев',
                     'Speaker_4': 'Валерий Бабушкин'}
    assert [c['reason'] for c in conflicts] == ['label_swap', 'false_voice_match']

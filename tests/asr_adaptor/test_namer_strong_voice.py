"""Уверенно узнанный голос не перебивается именем, которое интро-LLM вычитала из упоминания.

Поймано на ep2-5: реестр узнал Колодезева (cos 0.83), а тот на пятой минуте говорит «два часа
назад Денис Дергачев рассказывал…». Своё имя в первых репликах не прозвучало — и ветка «ложный
матч голоса» (ep18, cos 0.55) отдала 1860 его слов чужому имени.
"""
from stages.namer import name_speakers


class IntroLLM:
    def __init__(self, mapping):
        self.mapping = mapping

    async def complete_json(self, messages, schema=None, schema_name='', **kw):
        if schema_name == 'guests':
            return {'guests': []}
        return {'speakers': [{'label': k, 'name': v} for k, v in self.mapping.items()]}


def _turns():
    return [{'speaker': 'Speaker_0', 'start': 0.0, 'final': 'У нас сегодня пятый выпуск, поехали.'},
            {'speaker': 'Speaker_1', 'start': 251.5,
             'final': 'Слушай, ну тут два часа назад Денис Дергачев рассказывал про коллайдер.'}]


REGISTRY = {'Speaker_0': 'Валентин Малых', 'Speaker_1': 'Дмитрий Колодезев'}
LLM = {'Speaker_1': 'Денис Дергачев'}


async def test_strong_match_keeps_registry_name():
    names, conflicts = await name_speakers(_turns(), REGISTRY, IntroLLM(LLM),
                                           voice_cos={'Speaker_1': 0.826}, trust_cos=0.7)

    assert names['Speaker_1'] == 'Дмитрий Колодезев'
    assert [c['reason'] for c in conflicts] == ['strong_voice_match']
    assert conflicts[0]['cos'] == 0.826


async def test_weak_match_still_yields_to_intro():
    """ep18: голос гостя пришит к чужому центроиду на пороге — интро по-прежнему исправляет."""
    names, conflicts = await name_speakers(_turns(), REGISTRY, IntroLLM(LLM),
                                           voice_cos={'Speaker_1': 0.56}, trust_cos=0.7)

    assert names['Speaker_1'] == 'Денис Дергачев'
    assert conflicts[0]['reason'] == 'false_voice_match'


async def test_off_by_default():
    names, _ = await name_speakers(_turns(), REGISTRY, IntroLLM(LLM), voice_cos={'Speaker_1': 0.9})

    assert names['Speaker_1'] == 'Денис Дергачев'


async def test_new_guest_named_by_intro_is_untouched():
    """Реестр голос не знает — сила матча ни при чём, имя из интро."""
    names, conflicts = await name_speakers(_turns(), {'Speaker_0': 'Валентин Малых'}, IntroLLM(LLM),
                                           voice_cos={}, trust_cos=0.7)

    assert names['Speaker_1'] == 'Денис Дергачев'
    assert not conflicts

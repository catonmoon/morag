"""Гость, представленный ведущим: имя извлекает LLM, метку назначает КОД методом исключения.

Регрессия смены модели: grok выводил гостя сам, deepseek строго отказывается спекулировать —
на ep2-20 трижды вернул null и потерял Бабушкина (выпуск ушёл под ложным voice-match Натёкина).
Промпт-разрешения не помогли; исключение — детерминированная логика, ей и место в коде.
"""
from stages.namer import name_speakers


class TwoStepLLM:
    """Первый вызов — карта самоназваний (гостя нет), второй — список представленных гостей."""

    def __init__(self, guests):
        self.guests = guests

    async def complete_json(self, messages, schema=None, schema_name='', **kw):
        if schema_name == 'guests':
            return {'guests': self.guests}
        return {'speakers': [{'label': 'Speaker_0', 'name': 'Валентин Малых'},
                             {'label': 'Speaker_4', 'name': None}]}


def _turns():
    return [{'speaker': 'Speaker_0', 'start': 0.0,
             'final': 'С вами Валентин Малых и Дмитрий Колодезев. В гостях Валера Бабушкин. Привет!'},
            {'speaker': 'Speaker_4', 'start': 9.0, 'final': 'Всем привет, рад быть тут'},
            {'speaker': 'Speaker_1', 'start': 15.0, 'final': 'И правда рады'}]


async def test_introduced_guest_is_assigned_by_elimination():
    registry = {'Speaker_0': 'Валентин Малых', 'Speaker_1': 'Дмитрий Колодезев',
                'Speaker_4': 'Алексей Натёкин'}  # ложный voice-match, как на ep2-20

    names, conflicts = await name_speakers(_turns(), registry, TwoStepLLM(['Валерий Бабушкин']))

    assert names['Speaker_4'] == 'Валерий Бабушкин'
    assert conflicts and conflicts[0]['resolution'] == 'used_intro'  # реестр не подтверждён интро


async def test_guest_name_must_sound_in_intro():
    """Защита от фантазии экстрактора: имя, которого нет в тексте интро, метку не получает."""
    registry = {'Speaker_0': 'Валентин Малых', 'Speaker_1': 'Дмитрий Колодезев'}

    names, _ = await name_speakers(_turns(), registry, TwoStepLLM(['Пётр Выдуманный']))

    assert names['Speaker_4'] == 'Speaker_4'


async def test_two_guests_do_not_guess():
    """Двое гостей на одну свободную метку — исключение не работает, лучше Speaker_N, чем своп."""
    registry = {'Speaker_0': 'Валентин Малых', 'Speaker_1': 'Дмитрий Колодезев'}

    names, _ = await name_speakers(_turns(), registry,
                                   TwoStepLLM(['Валерий Бабушкин', 'Иван Иванов']))

    assert names['Speaker_4'] == 'Speaker_4'


async def test_extraction_is_doubled_and_deduped_by_surname():
    """Один прогон экстрактора — лотерея (2 из 3 находили гостя): удваиваем и объединяем,
    «Валера/Валерий Бабушкин» схлопываются в полную форму."""
    class Flaky(TwoStepLLM):
        def __init__(self):
            super().__init__(None)
            self.n = 0

        async def complete_json(self, messages, schema=None, schema_name='', **kw):
            if schema_name == 'guests':
                self.n += 1
                return {'guests': [] if self.n == 1 else ['Валера Бабушкин', 'Валерий Бабушкин']}
            return await super().complete_json(messages, schema, schema_name, **kw)

    registry = {'Speaker_0': 'Валентин Малых', 'Speaker_1': 'Дмитрий Колодезев'}

    names, _ = await name_speakers(_turns(), registry, Flaky())

    assert names['Speaker_4'] == 'Валерий Бабушкин'

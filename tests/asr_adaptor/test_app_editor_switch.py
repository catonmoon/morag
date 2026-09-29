"""Редактор на прогон: env `ASR_EDITOR` и поле формы `editor=` у одного и того же конвейера."""
from __future__ import annotations

import config


def test_editor_is_off_by_default():
    assert config.Config().editor is False, 'без переменной — финал-раунд, как раньше'


def test_form_field_switches_the_editor_for_one_record():
    import app
    from conveyor.run import run_conveyor
    assert app._runner('') is run_conveyor, 'пусто — как в конфиге'
    assert app._runner('1').keywords == {'editor': True}
    assert app._runner('off').keywords == {'editor': False}
    assert app._runner('1').func is run_conveyor


async def test_retrying_llm_wraps_tool_calls_in_the_retry_policy():
    from morag.llm.retry import RetryPolicy

    class Flaky:
        n = 0

        async def complete_with_tools(self, messages, tools, **kw):
            self.n += 1
            if self.n == 1:
                raise RuntimeError('спайк')
            return {'choices': [{'message': {'role': 'assistant', 'content': ''}, 'finish_reason': 'stop'}]}

    llm = config.RetryingLLM(Flaky(), RetryPolicy(max_retries=1, delay=0.0, backoff=1.0))
    got = await llm.complete_with_tools([{'role': 'user', 'content': 'x'}], [])
    assert got['choices'][0]['finish_reason'] == 'stop'

"""Переключатель конвейера: env `ASR_PIPELINE` и поле формы `pipeline=` выбирают исполнителя."""
from __future__ import annotations

import config
import pipeline
from config import _pipeline_mode


def test_env_value_is_normalised_and_defaults_to_legacy():
    assert _pipeline_mode('') == 'legacy'
    assert _pipeline_mode('graph') == 'graph'
    assert _pipeline_mode(' GRAPH ') == 'graph'
    assert _pipeline_mode('something') == 'legacy'
    assert config.Config().pipeline == 'legacy', 'без переменной — прежний конвейер'


def test_runner_picks_the_implementation():
    import app
    from graph.run import run_graph
    assert app._runner('legacy') is pipeline.run_pipeline
    assert app._runner('graph') is run_graph
    assert app._runner('') is pipeline.run_pipeline


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

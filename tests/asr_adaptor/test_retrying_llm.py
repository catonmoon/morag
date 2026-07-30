"""RetryingLLM: прикладные ретраи поверх SDK — битый JSON деген-петли для SDK успех (HTTP 200).

Политика одна на все стадии (глоссарий, правка, наминг) и живёт в клиенте, а не россыпью циклов.
"""
import pytest

from config import RetryingLLM
from morag.llm.retry import RetryPolicy


class FlakyClient:
    context_window = 32768

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0

    async def complete_json(self, *a, **kw):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ValueError('LLM returned invalid JSON')
        return {'ok': True}

    async def complete(self, *a, **kw):
        return 'текст'


async def test_semantic_failure_is_retried():
    client = FlakyClient(fail_times=2)
    llm = RetryingLLM(client, RetryPolicy(max_retries=2, delay=0))

    assert await llm.complete_json([]) == {'ok': True}
    assert client.calls == 3


async def test_exhausted_attempts_raise():
    client = FlakyClient(fail_times=99)
    llm = RetryingLLM(client, RetryPolicy(max_retries=2, delay=0))

    with pytest.raises(ValueError):
        await llm.complete_json([])
    assert client.calls == 3


async def test_other_attributes_pass_through():
    llm = RetryingLLM(FlakyClient(0), RetryPolicy(max_retries=0, delay=0))

    assert llm.context_window == 32768
    assert await llm.complete([]) == 'текст'

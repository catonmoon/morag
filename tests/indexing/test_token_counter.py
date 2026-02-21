import pytest

from morag.indexing.token_counter import TiktokenCounter, TokenCounter


@pytest.fixture
def counter() -> TiktokenCounter:
    return TiktokenCounter()


def test_counter_is_token_counter(counter):
    assert isinstance(counter, TokenCounter)


def test_count_returns_positive_int(counter):
    result = counter.count('Привет, мир!')
    assert isinstance(result, int)
    assert result > 0


def test_count_empty_string_returns_zero(counter):
    assert counter.count('') == 0


def test_count_grows_with_text_length(counter):
    short = counter.count('Hello')
    long = counter.count('Hello world, this is a longer text with more tokens')
    assert long > short


def test_fits_when_text_within_limit(counter):
    assert counter.fits('short text', limit=100) is True


def test_fits_when_text_exceeds_limit(counter):
    long_text = 'слово ' * 500
    assert counter.fits(long_text, limit=10) is False


def test_fits_exactly_at_limit(counter):
    text = 'hello'
    tokens = counter.count(text)
    assert counter.fits(text, limit=tokens) is True
    assert counter.fits(text, limit=tokens - 1) is False

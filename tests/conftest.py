

def pytest_addoption(parser):
    # Снимки выхода конвейера на заглушках (tests/asr_adaptor/golden/): обновлять только осознанно.
    parser.addoption('--update-golden', action='store_true', default=False,
                     help='переписать снимки конвейера в tests/asr_adaptor/golden/')

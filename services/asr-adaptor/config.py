"""Конфиг asr-adaptor: env-driven настройки + сборка morag `LLMClient`.

Аудио (диаризация/ASR/CAM++) — на Маке по HTTP (Apple-Silicon-bound). LLM (облако) — Grok НАПРЯМУЮ
через xAI (`api.x.ai/v1`). Дефолт `grok-4.20-0309-non-reasoning` — быстрый non-reasoning SKU
(~1.5с/вызов vs ~8.5с у grok-4.3, качество коррекции идентично — замерено 2026-06). Для него
reasoning-флаг НЕ шлём (`ASR_LLM_ENABLE_THINKING` пусто → None; модель реджектит `reasoningEffort` → 400).
grok-4.3 (reasoning) — fallback: `ASR_LLM_MODEL=grok-4.3` + `ASR_LLM_ENABLE_THINKING=false` (reasoning
OFF обязателен, иначе loop на structured). Ключ — env `OR_KEY` (имя историческое, держит xAI-ключ).
Все значения берутся из env при импорте (env выставляется в launchd-plist / docker-compose).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from morag.llm.client import LLMClient
from morag.llm.retry import RetryPolicy


def _env(key: str, default: str = '') -> str:
    return os.environ.get(key, default)


def _flag(key: str, default: str = '1') -> bool:
    return _env(key, default) not in ('0', 'false', 'no', '')


def _prompt_glossary_mode(raw: str) -> str:
    v = (raw or '').strip().lower()
    if v in ('0', 'false', 'no', 'off', 'none'):
        return 'none'
    return 'clean' if v in ('clean', 'чисто') else 'all'


def _clean_ear_mode(raw: str) -> str:
    v = (raw or '').strip().lower()
    if v in ('demand', 'по требованию'):
        return 'demand'
    return 'always' if v in ('1', 'true', 'yes', 'on', 'always') else ''


def _enable_thinking() -> bool | None:
    """Reasoning-флаг для LLMClient (env `ASR_LLM_ENABLE_THINKING`). Пусто/none → None: НЕ слать
    reasoning-параметр (для non-reasoning моделей типа grok-4.20-non-reasoning он невалиден → 400).
    false → reasoning OFF (для reasoning-моделей типа grok-4.3, иначе loop на structured). true → ON."""
    v = _env('ASR_LLM_ENABLE_THINKING').strip().lower()
    if v in ('', 'none', 'null'):
        return None
    return v in ('1', 'true', 'yes', 'on')


@dataclass
class Config:
    # --- аудио-бэкенды на Маке (loopback при native-деплое; Caddy-URL при Docker/remote) ---
    diarizer_url: str = field(default_factory=lambda: _env('ASR_DIARIZER_URL', 'http://127.0.0.1:8090/diarize'))
    diarizer_key: str = field(default_factory=lambda: _env('ASR_DIARIZER_KEY'))
    asr_url: str = field(default_factory=lambda: _env('ASR_BACKEND_URL', 'http://127.0.0.1:8123/v1/audio/transcriptions'))
    asr_key: str = field(default_factory=lambda: _env('ASR_BACKEND_KEY'))
    asr_model: str = field(default_factory=lambda: _env('ASR_MODEL', 'whisper-podlodka-turbo'))
    # Времена слов ОТ ДЕКОДЕРА в сегментах пасса-2 (`segments[].words`, приблизительные, с
    # `probability`). Выключено — вызов бэкенда и артефакт байт в байт прежние. Нужно разбору
    # внутри прогона: без времён слов он не видит ни растянутых слов, ни дыр (ADR-0030, план п. 3).
    word_times: bool = field(default_factory=lambda: _flag('ASR_WORD_TIMES', '0'))
    campp_url: str = field(default_factory=lambda: _env('ASR_CAMPP_URL', 'http://127.0.0.1:8126/embed-centroids'))
    campp_key: str = field(default_factory=lambda: _env('ASR_CAMPP_KEY'))

    # --- LLM: облако = Grok через xAI. Дефолт grok-4.20-0309-non-reasoning (быстрый non-reasoning,
    #     ~1.5с/вызов, качество коррекции = grok-4.3 — замерено). reasoning-флаг для него НЕ шлём (None).
    #     grok-4.3/OpenRouter — fallback через ASR_LLM_MODEL/ASR_LLM_BASE_URL + ASR_LLM_ENABLE_THINKING=false.
    #     Mac-native без облака: oMLX-local qwen3.6-35b (ASR_LLM_BASE_URL=http://127.0.0.1:8000/v1). ---
    llm_base_url: str = field(default_factory=lambda: _env('ASR_LLM_BASE_URL', 'https://api.x.ai/v1'))
    llm_model: str = field(default_factory=lambda: _env('ASR_LLM_MODEL', 'grok-4.20-0309-non-reasoning'))
    llm_key: str = field(default_factory=lambda: _env('OR_KEY'))  # имя env историческое; держит xAI-ключ

    # --- реестр спикеров (локальный JSON, см. CLAUDE.md — НЕ Qdrant) ---
    registry_path: str = field(default_factory=lambda: _env(
        'ASR_REGISTRY_PATH',
        os.path.join(os.environ.get('ASR_STACK_HOME')
                     or os.path.expanduser('~/asr-stack'), 'state', 'speaker_registry.json')))
    match_threshold: float = field(default_factory=lambda: float(_env('ASR_MATCH_THRESHOLD', '0.55')))
    max_centroids: int = field(default_factory=lambda: int(_env('ASR_MAX_CENTROIDS', '8')))

    # --- прочее ---
    mode: str = field(default_factory=lambda: _env('ASR_MODE', 'async'))  # async | sync
    whisper_tokenizer: str = field(default_factory=lambda: _env('ASR_WHISPER_TOKENIZER', 'openai/whisper-large-v3'))
    # Форма подсказки пасса-2 (префикс перед списком написаний). Пусто — умолчание движка.
    prompt_prefix: str = field(default_factory=lambda: _env('ASR_PROMPT_PREFIX', ''))
    prompt_budget: int = field(default_factory=lambda: int(_env('ASR_PROMPT_BUDGET', '200')))
    # Свободный глоссарий (LLM по черновику пасса-1) в подсказке пасса-2 — латинские каноники куска.
    # ⚠️ Замерено на 522 местах гарблов эталона (ADR-0030, 2ж): глоссарий «канонизирует» гарбл в него
    # самого и сидит в подсказке рядом с верным написанием; с ним канон выходит в 39 % мест против
    # 47 % без подсказки вовсе (гарбл — 22 % против 6 %). Только внешнее знание — постоянные термины
    # и подтверждённые подсказки — 51 % при 5 % гарблов. Но живьём без глоссария теряются настоящие
    # латинские термины, которых снаружи никто не дал (GitLab, Big Data): 8.8 → 10.9 % WER окон.
    # Поэтому три режима: `all` (умолчание, прежнее байт в байт — подкаст не трогаем), `none`,
    # `clean` — каноник глоссария, ЗВУЧАЩИЙ как постоянный термин или подтверждённая подсказка, но
    # написанный иначе, в подсказку не идёт (яд), остальная латиница остаётся. ASR_PROMPT_GLOSSARY=1|0|clean.
    prompt_glossary: str = field(default_factory=lambda: _prompt_glossary_mode(_env('ASR_PROMPT_GLOSSARY', '1')))
    # Авто-наминг Speaker_N → имя (интро-LLM + реестр). off → транскрипт остаётся в Speaker_N.
    # ⚠️ ВЫКЛЮЧЕН ПО УМОЛЧАНИЮ. Стадия исходит из подкастового допущения «ведущий представляет
    # гостя в начале», и вне него подписывает не того: на записи митапа докладчицу благодарят по
    # имени в КОНЦЕ, и имя досталось тому, кто эти слова произнёс (15 секунд эфира), а настоящая
    # докладчица (701 секунда) осталась безымянной. Ошибка системная, не случайная. Имя живого
    # человека — не та вещь, которую движок вправе угадывать молча, поэтому по умолчанию имён не
    # ищем вовсе; кому наминг подходит — включает `ASR_ENABLE_NAMING=1` осознанно.
    enable_naming: bool = field(default_factory=lambda: _flag('ASR_ENABLE_NAMING', '0'))

    # --- покрытие звука расшифровкой (потери речи, см. stages/coverage.py) ---
    # hole_min_s — с какой длины считать непокрытый промежуток дырой (и добирать его чанками);
    # coverage_warn_s — расхождение «длительность реплики vs распознанное», с которого пишем WARN.
    hole_min_s: float = field(default_factory=lambda: float(_env('ASR_HOLE_MIN_S', '5')))
    coverage_warn_s: float = field(default_factory=lambda: float(_env('ASR_COVERAGE_WARN_S', '5')))
    recover_gaps: bool = field(default_factory=lambda: _flag('ASR_RECOVER_GAPS'))
    retry_empty: bool = field(default_factory=lambda: _flag('ASR_RETRY_EMPTY'))
    # Переслушивание подозрительных кусков (stages/relisten.py). По умолчанию ВЫКЛЮЧЕНО: стадия
    # меняет текст расшифровки, и включать её чужому корпусу молча нельзя. У кого включено —
    # петля вместо речи чинится на месте, а не доезжает до читателя.
    relisten: bool = field(default_factory=lambda: _flag('ASR_RELISTEN', '0'))
    # Арбитраж (stages/arbitrate.py, ADR-0030): каждый кусок пасса-2 слушается ещё раз ДРУГОЙ
    # моделью (`ASR_SECOND_MODEL` — её имя у бэкенда; пусто — стадии нет, поведение прежнее байт в
    # байт) и, если `ASR_CLEAN_EAR`, той же моделью без подсказки — третий голос для голосования.
    # Замерено: другая модель — 4.8 %, с голосованием — 4.4 % против 5.1 % без стадии; берётся
    # модель ДРУГОЙ ШКОЛЫ, не более крупная той же — та ошибается в тех же местах.
    # ⚠️ У второго уха НЕТ подсказки и температуры — намеренно, по замеру (ADR-0030, 2а): подсказка
    # на переслушивании у первой модели даёт ± 0, у второй — WER × 6; лесенка тождественна нулю.
    # Ручки `ASR_SECOND_PROMPT` / `ASR_SECOND_TEMPERATURE` не заводить без нового замера.
    second_model: str = field(default_factory=lambda: _env('ASR_SECOND_MODEL', ''))
    # Третий голос — та же модель без подсказки: `always` (каждый кусок), `demand` — только там, где
    # после второго уха остались споры, не решённые ни частотой, ни каноном (ADR-0030: эскалация по
    # куску, а не проход по записи); пусто/0 — без него (тогда голосования нет).
    clean_ear: str = field(default_factory=lambda: _clean_ear_mode(_env('ASR_CLEAN_EAR', '0')))
    # Окно третьего голоса — `chunk` (кусок как есть), `window30` (30 с вокруг спорного места по
    # временам слов декодера, поверх границ кусков), `neighbours` (кусок с соседями, ≤ 90 с).
    # ⚠️ Никогда не слово: окно по слову давало вдвое меньше попаданий (stages/arbitrate.ear_window).
    # Замерено на эталоне (ADR-0030, 2д): кусок — 5 починок / 0 порч, 15 с — 1/1, 30 с вокруг
    # спора — 4/2, 90 с — 4/0. Шире куска не лучше, центрировать по спору незачем.
    clean_ear_window: str = field(default_factory=lambda: _env('ASR_CLEAN_EAR_WINDOW', 'chunk').strip().lower() or 'chunk')
    # Ворота второго уха: `reader` — LLM-читатель решает, слушать ли кусок ещё раз (оценка
    # вменяемости слов с глоссарием записи); пусто — слушать все куски. Замерено: 4.4 % против 4.3 %
    # «везде» при ≈ 30 % экономии прохода; одна короткая LLM-проверка на кусок.
    arbitrate_gate: str = field(default_factory=lambda: _env('ASR_ARBITRATE_GATE', '').strip().lower())
    arbitrate_ratio: float = field(default_factory=lambda: float(_env('ASR_ARBITRATE_RATIO', '100')))

    # Параллельность финал-раунда: реплики независимы, а стадия занимала 8-12 мин из 15-18 на
    # выпуск. Потолок скромный — упирается не в нас, а в лимиты провайдера.
    round_concurrency: int = field(default_factory=lambda: int(_env('ASR_ROUND_CONCURRENCY', '6')))
    # Финал-раунд не вправе менять ИЗВЕСТНОЕ слово записи (подсказки, имена, постоянные термины) на
    # другое по звучанию — вето `known_term` (ADR-0030). Замерено звуком по корпусу: 64 такие замены
    # зря, 10 верно. Верные возвращает `final_ear`: отвергнутая замена переслушивается чистым ухом, и
    # если звук ближе к замене — она принимается. Оба выключены: прежнее поведение байт в байт.
    protect_known: bool = field(default_factory=lambda: _flag('ASR_PROTECT_KNOWN', '0'))
    final_ear: bool = field(default_factory=lambda: _flag('ASR_FINAL_EAR', '0'))
    # Контекст правки — ЦЕЛЫМИ репликами по n с каждой стороны (реплика = непрерывная речь одного
    # человека из диаризации, законченная мысль). Не символы и не токены: они рвут фразы и шумят.
    # По ОДНОЙ: с двумя контекст выходил крупнее самого фрагмента (реплика бывает и в 1000 слов),
    # модель теряла внимание к нему и переставала править вовсе — на ep20 ушли Three Mile Island,
    # Olkiluoto, FAW, Kaspersky, а раунд разбух с 48 до 172 секунд.
    context_turns: int = field(default_factory=lambda: int(_env('ASR_CONTEXT_TURNS', '1')))

    # --- постоянный словарь корпуса: имена ведущих и повторяющиеся продукты идут в подсказку
    #     пасса-2 ВСЕГДА. Глоссарий переоткрывает термины каждый выпуск и на устойчивом гарбле
    #     срывается (замерено: «Колодзев» ×14, `MotorMost` ×24 по корпусу). Список доменный —
    #     живёт в env деплоя, не в коде. Пример: ASR_ALWAYS_TERMS='Иван Иванов,Mattermost' ---
    always_terms: list[str] = field(default_factory=lambda: [
        t.strip() for t in _env('ASR_ALWAYS_TERMS').split(',') if t.strip()])

    # --- ЧТО ЭТО ЗА МАТЕРИАЛ. Одна строка в родительном падеже, подставляется в промпты правки
    #     терминов и наминга: «расшифровку фрагмента <corpus_desc>». Конвейер generic, а подсказка
    #     о домене — доменная: на записях рабочих встреч правка, настроенная на подкаст про
    #     технологии, тянет канонизацию не туда.
    #     Пример: ASR_CORPUS_DESC='рабочих записей: встреч, митапов и обучающих материалов по ML'
    #     ⚠️ Описывает КОРПУС, а не конкретную запись: в одной папке жанры мешаются, и знать
    #     заранее, встреча это или лекция, не нужно — стадиям это и не требуется. ---
    corpus_desc: str = field(default_factory=lambda: _env(
        'ASR_CORPUS_DESC', 'русскоязычного подкаста про технологии'))

    # --- сколько голосов ждать в записи. Дефолт 1-10 покрывает подкаст и небольшую встречу;
    #     на совещании с десятком участников потолок упрётся, и лишние голоса склеятся с чужими. ---
    min_speakers: int = field(default_factory=lambda: int(_env('ASR_MIN_SPEAKERS', '1')))
    max_speakers: int = field(default_factory=lambda: int(_env('ASR_MAX_SPEAKERS', '10')))

    # --- прикладные ретраи LLM: SDK повторяет ТРАНСПОРТ (429/5xx/обрывы), а битый JSON
    #     деген-петли для него успех (HTTP 200). Поверх — RetryPolicy ядра: 3 попытки с паузой
    #     10с (спайк сети/нагрузки живёт дольше секунды, немедленный повтор бьёт в него же) ---
    llm_attempts: int = field(default_factory=lambda: int(_env('ASR_LLM_ATTEMPTS', '3')))
    llm_retry_delay: float = field(default_factory=lambda: float(_env('ASR_LLM_RETRY_DELAY', '10')))
    llm_max_concurrent: int = field(default_factory=lambda: int(_env('ASR_LLM_MAX_CONCURRENT', '8')))

    # --- параллельные транскрибации: N выпусков в полёте, ресурсы гейтятся по отдельности.
    #     Выигрыш — конвейеризация СТАДИЙ: GPU-стадии выпуска B идут, пока у A работает LLM.
    #     Слоты по 1: у бэкендов один инстанс модели (pyannote thread-unsafe, mlx-очередь общая);
    #     whisper_slots=2 можно пробовать после замера. max_jobs=1 — прежнее поведение ---
    max_jobs: int = field(default_factory=lambda: int(_env('ASR_MAX_JOBS', '1')))
    diarize_slots: int = field(default_factory=lambda: int(_env('ASR_DIARIZE_SLOTS', '1')))
    whisper_slots: int = field(default_factory=lambda: int(_env('ASR_WHISPER_SLOTS', '1')))
    campp_slots: int = field(default_factory=lambda: int(_env('ASR_CAMPP_SLOTS', '1')))
    align_slots: int = field(default_factory=lambda: int(_env('ASR_ALIGN_SLOTS', '1')))

    # --- пословное выравнивание (stages/align.py; торч ставится отдельно, см. requirements-align.txt) ---
    enable_align: bool = field(default_factory=lambda: _flag('ASR_ENABLE_ALIGN'))
    align_device: str = field(default_factory=lambda: _env('ASR_ALIGN_DEVICE'))  # '' → mps|cuda|cpu

    # Ожидание перезагрузки модели на эндпоинте. Ядро умеет ждать (`_is_model_unavailable` ловит
    # «Model not found» и пустой ответ), но адаптер этим не пользовался: без этих полей клиент
    # получает 0 попыток и цикл ожидания выключен.
    # ⚠️ Куплено ночным прогоном: лекция на 199 минут погибла на ФИНАЛ-РАУНДЕ от 400 «Model not
    # found» — эндпоинт на секунду перезагружал модель. Тридцать четыре минуты работы (диаризация,
    # два прохода whisper, выравнивание) выброшены из-за паузы, которую достаточно переждать.
    # ⚠️ 400 не ретраится обычной политикой: по HTTP это ошибка клиента, а не транзиент.
    llm_model_wait_s: int = field(default_factory=lambda: int(_env('ASR_LLM_MODEL_WAIT_S', '20')))
    llm_model_wait_retries: int = field(
        default_factory=lambda: int(_env('ASR_LLM_MODEL_WAIT_RETRIES', '3')))

    # Штраф за повтор для стадий со structured output. Пусто → не шлём (поведение прежнее).
    # ⚠️ Не украшение: деген-петля — известный отказ этой схемы, и на КВАНТОВАННОЙ модели она
    # куда вероятнее. Замерено на одном и том же батче и промпте: 8-битная сборка отвечает за
    # 7.7 с, 4-битная той же модели уходит в цикл (один и тот же термин сотни раз) и рвёт JSON о
    # потолок в 8000 токенов. 1.05 не помогает, 1.15 снимает петлю начисто — 0.4 с.
    llm_repetition_penalty: float = field(
        default_factory=lambda: float(_env('ASR_LLM_REPETITION_PENALTY') or 0) or None)

    def build_llm(self) -> 'RetryingLLM':
        """morag LLMClient + прикладной RetryPolicy. SDK-ретраи (max_retries) закрывают транспорт,
        политика поверх — битый JSON деген-петель и спайки; max_concurrent — общий потолок
        одновременных вызовов через реестр семафоров ядра (важен при ASR_MAX_JOBS > 1)."""
        client = LLMClient(base_url=self.llm_base_url, model=self.llm_model, api_key=self.llm_key,
                           enable_thinking=_enable_thinking(), timeout=180, max_retries=4,
                           max_concurrent=self.llm_max_concurrent,
                           model_wait_seconds=self.llm_model_wait_s,
                           model_wait_retries=self.llm_model_wait_retries)
        policy = RetryPolicy(max_retries=max(0, self.llm_attempts - 1),
                             delay=self.llm_retry_delay, backoff=1.0)
        return RetryingLLM(client, policy, self.llm_repetition_penalty)


class RetryingLLM:
    """Прозрачная обёртка LLMClient: complete/complete_json идут через RetryPolicy ядра.

    Ретраи живут ЗДЕСЬ, а не россыпью по стадиям: глоссарий, правка, наминг получают их
    автоматически и одинаково. Остальные атрибуты (context_window и пр.) — сквозные."""

    def __init__(self, client: LLMClient, policy: RetryPolicy,
                 repetition_penalty: float | None = None) -> None:
        self._client = client
        self._policy = policy
        self._rep = repetition_penalty

    def _with_rep(self, kwargs: dict) -> dict:
        """Подмешать штраф за повтор, если стадия не задала параметры сама.

        Стадии зовут `complete_json` без `params`, и другого места приложить настройку корпуса
        нет: клиент ядра сливает с дефолтами только `enable_thinking` и `seed`."""
        if self._rep and kwargs.get('params') is None:
            from morag.llm.client import GenerationParams
            kwargs = dict(kwargs, params=GenerationParams(repetition_penalty=self._rep))
        return kwargs

    def __getattr__(self, name):
        return getattr(self._client, name)

    async def complete(self, *args, **kwargs):
        kwargs = self._with_rep(kwargs)
        return await self._policy.call(lambda: self._client.complete(*args, **kwargs), 'complete')

    async def complete_json(self, *args, **kwargs):
        kwargs = self._with_rep(kwargs)
        return await self._policy.call(
            lambda: self._client.complete_json(*args, **kwargs), 'complete_json')


CFG = Config()

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

import json
import os
from dataclasses import dataclass, field

from morag.llm.client import LLMClient
from morag.llm.retry import RetryPolicy


def _env(key: str, default: str = '') -> str:
    return os.environ.get(key, default)


def _asr_urls() -> list[str]:
    """Пул whisper: явный `ASR_BACKEND_URLS` (через запятую) или `ASR_WHISPER_INSTANCES=N` — те же
    порты, что поднимает `deploy/mac/stack.sh` (8123, 8133, 8143…). Иначе — один `ASR_BACKEND_URL`."""
    explicit = [u.strip() for u in _env('ASR_BACKEND_URLS').split(',') if u.strip()]
    if explicit:
        return explicit
    base = _env('ASR_BACKEND_URL', 'http://127.0.0.1:8123/v1/audio/transcriptions')
    n = max(1, int(_env('ASR_WHISPER_INSTANCES', '1') or 1))
    if n == 1 or ':8123/' not in base:
        return [base]
    return [base.replace(':8123/', f':{8123 + 10 * k}/') for k in range(n)]


# --- полосы: куда едет модель -------------------------------------------------------------------
# Полоса = имя (оно же имя семафора `pipeline._res`), класс устройства, слоты, адреса бэкендов.
# Зачем: модель другой школы (GigaAM, torch на CPU) в ОДНОМ процессе с mlx-whisper стоит в той же
# очереди инференса (у бэкенда один поток), и «CPU-модель» не выигрывает ничего — замерено 16.10 на
# пяти записях: корректор 598 с = очередь двух ушей, не LLM. Своя полоса = свой процесс бэкенда и
# свой семафор: CPU-ухо идёт рядом с GPU-ухом, а не за ним. Адрес может быть и удалённым (другая
# машина, видеокарта) — адаптер устройств не знает, он знает только, что полосы разные.
# Полоса по умолчанию `whisper` описана прежними ручками (`ASR_BACKEND_URL(S)`, `ASR_WHISPER_INSTANCES`,
# `ASR_WHISPER_SLOTS`) и здесь не перечисляется: без `ASR_LANES` всё байт в байт как было.
_LANE_DEVICES = ('gpu', 'cpu', 'remote')
_ASR_PATH = '/v1/audio/transcriptions'


@dataclass(frozen=True)
class Lane:
    name: str
    device: str
    slots: int
    urls: tuple


def _lane_url(u: str) -> str:
    """Адрес полосы можно дать базой (`http://127.0.0.1:8223`) — путь транскрипций дописывается."""
    u = u.strip().rstrip('/')
    return u if u.endswith(_ASR_PATH) else u + _ASR_PATH


def _lanes() -> dict[str, 'Lane']:
    """`ASR_LANES="cpu=cpu,1,http://127.0.0.1:8223;…"` → {имя: Lane}; форма `имя=устройство,слоты,url[,url…]`."""
    out: dict[str, Lane] = {}
    for spec in _env('ASR_LANES').split(';'):
        spec = spec.strip()
        if not spec:
            continue
        name, eq, rest = spec.partition('=')
        parts = [p.strip() for p in rest.split(',') if p.strip()]
        name = name.strip()
        if not eq or not name or len(parts) < 3:
            raise ValueError(f'ASR_LANES: «{spec}» — нужно имя=устройство,слоты,url[,url…]')
        if name == 'whisper':
            raise ValueError('ASR_LANES: полоса «whisper» задаётся прежними ручками (ASR_BACKEND_URL(S), ASR_WHISPER_SLOTS)')
        if parts[0] not in _LANE_DEVICES:
            raise ValueError(f'ASR_LANES: устройство «{parts[0]}» у полосы «{name}» — одно из {", ".join(_LANE_DEVICES)}')
        out[name] = Lane(name, parts[0], max(1, int(parts[1])), tuple(_lane_url(u) for u in parts[2:]))
    return out


def _model_lanes() -> dict[str, str]:
    """`ASR_MODEL_LANES="gigaam-v3-e2e-rnnt=cpu,другая=полоса"` → {модель: полоса}; полоса обязана быть описана."""
    lanes = _lanes()
    out: dict[str, str] = {}
    for pair in _env('ASR_MODEL_LANES').split(','):
        pair = pair.strip()
        if not pair:
            continue
        model, eq, lane = (x.strip() for x in pair.partition('='))
        if not eq or not model or not lane:
            raise ValueError(f'ASR_MODEL_LANES: «{pair}» — нужно модель=полоса')
        if lane not in lanes:
            raise ValueError(f'ASR_MODEL_LANES: полоса «{lane}» модели «{model}» не описана в ASR_LANES')
        out[model] = lane
    return out


def lane_for(cfg, model: str = '') -> 'Lane':
    """Полоса модели (пусто — модель профиля). Записи в карте нет — полоса `whisper` из прежних ручек.
    Поля берутся через getattr: cfg в тестах узлов — `SimpleNamespace` без полей полос."""
    model = model or str(getattr(cfg, 'asr_model', '') or '')
    name = (getattr(cfg, 'model_lanes', None) or {}).get(model, '')
    lanes = getattr(cfg, 'lanes', None) or {}
    if name:
        if name not in lanes:
            raise ValueError(f'полоса «{name}» модели «{model}» не описана в ASR_LANES')
        return lanes[name]
    return Lane('whisper', 'gpu', max(1, int(getattr(cfg, 'whisper_slots', 1) or 1)),
                tuple(getattr(cfg, 'asr_urls', ()) or ()))


def _flag(key: str, default: str = '1') -> bool:
    return _env(key, default) not in ('0', 'false', 'no', '')


def _prompt_glossary_mode(raw: str) -> str:
    v = (raw or '').strip().lower()
    if v in ('0', 'false', 'no', 'off', 'none'):
        return 'none'
    if v in ('substitute', 'подменять'):
        return 'substitute'
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
    # Пул процессов whisper; `asr_url` — первый из них (им пользуются health и старые вызовы).
    asr_urls: list = field(default_factory=_asr_urls)
    asr_key: str = field(default_factory=lambda: _env('ASR_BACKEND_KEY'))
    asr_model: str = field(default_factory=lambda: _env('ASR_MODEL', 'whisper-podlodka-turbo'))
    # Времена слов ОТ ДЕКОДЕРА в сегментах пасса-2 (`segments[].words`, приблизительные, с
    # `probability`). Выключено — вызов бэкенда и артефакт байт в байт прежние. Нужно разбору
    # внутри прогона: без времён слов он не видит ни растянутых слов, ни дыр (ADR-0030, план п. 3).
    word_times: bool = field(default_factory=lambda: _flag('ASR_WORD_TIMES', '0'))
    # Шов с соседями (stages/seams.py): окна С ЗАПАСОМ (повтор пустого куска, переслушивание)
    # отдают куску только слова, чья середина внутри него, и страховка снимает повтор конца
    # соседа. Выключено — поведение прежнее байт в байт.
    seam: bool = field(default_factory=lambda: _flag('ASR_SEAM', '0'))
    # Голоса по словам (stages/resplit.py): после выравнивания реплика режется там, где по
    # диаризации сменился говорящий, — короткая реплика второго голоса не тонет в чужой.
    # Выключено — реплики прежние байт в байт.
    resplit: bool = field(default_factory=lambda: _flag('ASR_RESPLIT', '0'))
    # Поддакивания (через запятую): только такое слово в наложении отдаётся короткому отрезку
    # перебившего; любое другое остаётся хозяину. Язык корпуса — забота профиля, движок слов не знает.
    resplit_backchannel: str = field(default_factory=lambda: _env('ASR_RESPLIT_BACKCHANNEL'))
    # Реестр: два кластера ОДНОЙ записи становятся одним голосом, только если похожи и друг на
    # друга (stages/registry.py). Выключено — прежнее «каждый кластер к ближайшему в реестре».
    record_guard: bool = field(default_factory=lambda: _flag('ASR_REGISTRY_RECORD_GUARD', '0'))
    # Центроид голоса без чужой речи (backends/campp: наложения и ±0.3 с у чужих отрезков).
    clean_centroids: bool = field(default_factory=lambda: _flag('ASR_CLEAN_CENTROIDS', '0'))
    # Склеенный диалог (stages/recover.py): диаризация дала ОДИН голос, а людей по знанию о
    # записи больше — голоса ищутся заново по фразам. 0 — узла нет.
    recover_speakers: bool = field(default_factory=lambda: _flag('ASR_RECOVER_SPEAKERS', '0'))
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
    # Файл переопределений промптов стадий (TOML, см. prompts.py). Пусто — встроенные тексты.
    prompts_file: str = field(default_factory=lambda: _env('ASR_PROMPTS', ''))
    # Редактор расшифровки (`conveyor/editor.py`): агент читает страницы черновика после голосов и
    # предлагает правки со свидетелем вместо финал-раунда. Выключено — финал-раунд, как раньше.
    # Поле формы `editor=1|0` у `POST /v1/audio/transcriptions` перебивает на одну запись.
    editor: bool = field(default_factory=lambda: _flag('ASR_EDITOR', '0'))
    # Бюджет шагов на одно место (кусок арбитража, реплика финал-раунда). Правилам хватает: арбитраж —
    # до пяти (ворота, два уха, правила, третий голос), реплика — два плюс три на каждую замену
    # известного слова.
    place_steps: int = field(default_factory=lambda: int(_env('ASR_PLACE_STEPS', '12')))
    # Каталог чекпойнтов конвейера: после каждого узла состояние пишется в JSON (0600, без центроидов
    # голосов), `run_conveyor(resume=<файл>)` продолжает с первого непройденного узла. Пусто — выключено.
    checkpoints: str = field(default_factory=lambda: _env('ASR_CHECKPOINTS', ''))
    # Свои события конвейера в ленте (`conveyor.node` со счётчиком после каждого узла). Выключено —
    # лента прежняя: её читает окно загрузки записи.
    node_events: bool = field(default_factory=lambda: _flag('ASR_NODE_EVENTS', '0'))
    # Наложение стадий одной записи (узел `opening`): после пасса-1 диаризация (GPU) идёт ВМЕСТЕ с
    # глоссарием (LLM), сводка записи для финал-раунда — фоном. GPU с GPU не накладываем: замер
    # 13.10 — оба вдвое медленнее. Выключено — порядок узлов прежний байт в байт (золотые снимки).
    overlap: bool = field(default_factory=lambda: _flag('ASR_OVERLAP', '0'))
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
    # написанный иначе, в подсказку не идёт (яд), остальная латиница остаётся; `substitute` — такой
    # каноник ПОДМЕНЯЕТСЯ известным написанием (постоянные термины → подсказки → spellings), а не
    # выбрасывается: живьём `clean` терял латинские термины, которых снаружи никто не дал (28.09,
    # пасс-2 −2.5 пункта), потому что убирал форму, не кладя ничего взамен. ASR_PROMPT_GLOSSARY=1|0|clean|substitute.
    prompt_glossary: str = field(default_factory=lambda: _prompt_glossary_mode(_env('ASR_PROMPT_GLOSSARY', '1')))
    # Сверка глоссария с известными написаниями (stages/glossary.reconcile, ADR-0030 2з): каноник,
    # звучащий как известное слово, подменяется им. Выключено — глоссарий прежний байт в байт;
    # действует на подсказку пасса-2, финал-раунд и мету разом.
    glossary_reconcile: bool = field(default_factory=lambda: _flag('ASR_GLOSSARY_RECONCILE', '0'))
    # A. Латинский каноник глоссария без свидетеля (известные написания, словарь домена
    # `hints.vocabulary`, английский словарь) в подсказку не идёт. B. Модель сама помечает пары
    # «знаю / догадка», в подсказку идут только известные. Оба выключены; оба — ADR-0030 (2и).
    glossary_witness: bool = field(default_factory=lambda: _flag('ASR_GLOSSARY_WITNESS', '0'))
    # Порог английского словаря для свидетеля A (zipf). 3.0 — обычные слова; замерено живьём, что при
    # нём уходит половина глоссария длинной записи (59 из 111): «kubernetes» 2.2, «jira» 2.1, «gitlab»
    # 1.9 по zipf — термины, а не обычные слова. 1.5 оставляет их, выбрасывая выдумки (0.0).
    glossary_witness_en: float = field(default_factory=lambda: float(_env('ASR_GLOSSARY_WITNESS_EN', '3.0')))
    glossary_selflabel: bool = field(default_factory=lambda: _flag('ASR_GLOSSARY_SELFLABEL', '0'))
    # Авто-наминг Speaker_N → имя (интро-LLM + реестр). off → транскрипт остаётся в Speaker_N.
    # ⚠️ ВЫКЛЮЧЕН ПО УМОЛЧАНИЮ. Стадия исходит из подкастового допущения «ведущий представляет
    # гостя в начале», и вне него подписывает не того: на записи митапа докладчицу благодарят по
    # имени в КОНЦЕ, и имя досталось тому, кто эти слова произнёс (15 секунд эфира), а настоящая
    # докладчица (701 секунда) осталась безымянной. Ошибка системная, не случайная. Имя живого
    # человека — не та вещь, которую движок вправе угадывать молча, поэтому по умолчанию имён не
    # ищем вовсе; кому наминг подходит — включает `ASR_ENABLE_NAMING=1` осознанно.
    enable_naming: bool = field(default_factory=lambda: _flag('ASR_ENABLE_NAMING', '0'))

    # --- корректор после распознавания (conveyor/proofread.py, ADR-0032): детектор по тексту + решатель
    # со звуком. По умолчанию ВЫКЛ. Уши решателя: clean (основная модель без подсказки), second (ASR_SECOND_MODEL).
    # min_ears — сколько ушей должны слышать замену (по слуху владельца Мостика: 1 ухо — 26/29 верных,
    # 2 уха — 13/13); латиница из известных написаний проходит без уха. pad — окно ±с вокруг места.
    # max_per_min — бюджет пометок на минуту звука (детектор даёт ~2/мин, половина шум).
    # suspect — корни подозрительных слов через запятую (брань в деловой записи): доменная настройка профиля.
    proofread: bool = field(default_factory=lambda: _flag('ASR_PROOFREAD', '0'))
    proofread_ears: str = field(default_factory=lambda: _env('ASR_PROOFREAD_EARS', 'clean,second'))
    proofread_min_ears: int = field(default_factory=lambda: int(_env('ASR_PROOFREAD_MIN_EARS', '1')))
    proofread_pad_s: float = field(default_factory=lambda: float(_env('ASR_PROOFREAD_PAD_S', '4')))
    proofread_max_per_min: float = field(default_factory=lambda: float(_env('ASR_PROOFREAD_MAX_PER_MIN', '4')))
    # Потолок токенов ответа детектора на страницу. ⚠️ 2500 обрезает JSON на страницах с десятками пометок
    # (Ornith, 16.10: «Unterminated string» на ~7400 знаках → повтор через 10 с, ~40 с на событие, 4 события на
    # три записи); 4000 — запас вдвое. Умолчание прежнее.
    proofread_detect_max_tokens: int = field(default_factory=lambda: int(_env('ASR_PROOFREAD_DETECT_MAX_TOKENS', '2500')))
    proofread_suspect: str = field(default_factory=lambda: _env('ASR_PROOFREAD_SUSPECT', ''))
    # Чистое ухо — та же модель, что основной проход, и ошибается заодно с ним: одно оно свидетелем
    # не считается (контроль молчания 14.10: только clean — 2 верных / 2 порчи, только второе ухо —
    # 27 / 0, оба — 4 / 0). Включить — для стека без второго уха, осознанно.
    proofread_clean_alone: bool = field(default_factory=lambda: _flag('ASR_PROOFREAD_CLEAN_ALONE', '0'))
    # Длина заменяемого участка (слов) у корректора. В движке 3 — как у финал-раунда (ADR-0020). Мостик
    # ставит 6: 40 правок по 4-6 слов с ухом другой школы — 23 верных, 0 порч по слуху владельца (14.10);
    # из 112 отказов too_long на стенде 72 проходили барьеры при лимите 6. now — до max_was + 2.
    proofread_max_was_words: int = field(default_factory=lambda: int(_env('ASR_PROOFREAD_MAX_WAS_WORDS', '3')))
    # Слушатель (ADR-0032, дополнение 15.10): аудио-LLM (OpenRouter, input_audio), которая сама слушает окно
    # ±listener_pad_s вокруг места вместо текстов ушей; её дословное `heard` — свидетель замены. Пусто — выкл.
    # Новую латиницу одно слово слушателя не подтверждает (страж: «слон → Qwen» и т.п.), только канон.
    # Мостик, 728 мест: qwen/qwen3.8-omni-flash принял 289 правок против 148 у ушей; на 59 правках «только у
    # слушателя» по слуху владельца 23 верных / 4 порчи, со стражем 20 / 1.
    proofread_listener: str = field(default_factory=lambda: _env('ASR_PROOFREAD_LISTENER', ''))
    proofread_listener_pad_s: float = field(default_factory=lambda: float(_env('ASR_PROOFREAD_LISTENER_PAD_S', '6')))
    # Маршрут по роду ошибки (`kind`): на каждое место параллельно и уши (ASR_PROOFREAD_EARS), и слушатель; правка
    # «смысла» (обычные слова) берётся у ушей, иначе у слушателя; «написания» (латиница, имена, слова глоссария) —
    # у слушателя, иначе у ушей лишь со вторым ухом. Мостик, слух владельца на 728 местах: уши 95%, слушатель со
    # стражем 94%, маршрут 111 верных / 4 порчи (97%) при 312 правках. Пусто — один решатель, как раньше.
    proofread_route: str = field(default_factory=lambda: _env('ASR_PROOFREAD_ROUTE', ''))
    # Ухо `omni` корректора (ASR_PROOFREAD_EARS=…,omni): аудио-LLM как распознаватель — дословная расшифровка окна
    # через OpenAI-совместимый чат со звуком (локальный oMLX и т.п.). Мостик, 15.10: Qwen3-Omni-30B 8bit третьим ухом
    # под решателем — 160 правок, 96% против 150, 95% у двух ушей. Адрес и ключ — свои (по умолчанию как у LLM стадий).
    proofread_omni_model: str = field(default_factory=lambda: _env('ASR_PROOFREAD_OMNI_MODEL', ''))
    proofread_omni_base_url: str = field(default_factory=lambda: _env('ASR_PROOFREAD_OMNI_BASE_URL', ''))
    proofread_omni_key: str = field(default_factory=lambda: _env('ASR_PROOFREAD_OMNI_KEY', ''))
    # С какого косинуса матча реестра имя голоса НЕ перебивается интро. Интро-LLM путает упоминание
    # с представлением (ep2-5: «два часа назад Денис Дергачев рассказывал» → чужое имя ведущему при
    # cos 0.83). 0 — выкл, прежнее «истина = интро». Разумно 0.7: разные люди у CAM++ 0.21-0.50.
    naming_trust_cos: float = field(default_factory=lambda: float(_env('ASR_NAMING_TRUST_COS', '0')))

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
    # Третий голос не звать, когда все споры куска — о форме одного слова («повышенные/повышенной»):
    # голосование такие и так почти не решает. Замер 14.10 на 2-5: 39 из 196 спорных кусков (20%),
    # 688 с звука из 3341, цена — одно решение голосованием («тем»→«теми»). По умолчанию выкл.
    clean_ear_skip_inflection: bool = field(default_factory=lambda: _flag('ASR_CLEAN_EAR_SKIP_INFLECTION', '0'))
    # Ворота второго уха: `reader` — LLM-читатель решает, слушать ли кусок ещё раз (оценка
    # вменяемости слов с глоссарием записи); пусто — слушать все куски. Замерено: 4.4 % против 4.3 %
    # «везде» при ≈ 30 % экономии прохода; одна короткая LLM-проверка на кусок.
    arbitrate_gate: str = field(default_factory=lambda: _env('ASR_ARBITRATE_GATE', '').strip().lower())
    arbitrate_ratio: float = field(default_factory=lambda: float(_env('ASR_ARBITRATE_RATIO', '100')))
    # Страж латиницы: правило арбитража не заменяет слово куска с латиницей словом без неё. Нужен
    # второму уху ДРУГОЙ школы, пишущему термины кириллицей (GigaAM, Parakeet): иначе «NovelAI»→«вел»,
    # «cloaked»→«Cloud». Замер 2026-10-14 на 410 местах «Мостика»: −1…−2 поломки при тех же починках.
    arbitrate_latin_guard: bool = field(default_factory=lambda: _flag('ASR_ARBITRATE_LATIN_GUARD', '0'))

    # Параллельность финал-раунда: реплики независимы, а стадия занимала 8-12 мин из 15-18 на
    # выпуск. Потолок скромный — упирается не в нас, а в лимиты провайдера.
    # Кусков пасса-2 в работе разом (узел `pass2`); 1 — по одному, как раньше.
    pass2_concurrency: int = field(default_factory=lambda: int(_env('ASR_PASS2_CONCURRENCY', '1')))
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
    # Не задано — по числу процессов пула: слот на процесс (у каждого один поток инференса).
    whisper_slots: int = field(default_factory=lambda: int(_env('ASR_WHISPER_SLOTS') or len(_asr_urls())))
    # Полосы (см. `Lane` выше): `ASR_LANES` — описания, `ASR_MODEL_LANES` — какая модель куда едет.
    # Пусто — одна полоса `whisper`, прежнее поведение.
    lanes: dict = field(default_factory=_lanes)
    model_lanes: dict = field(default_factory=_model_lanes)
    # --- черновик другой школой (16.10). Пасс-1 моделью `ASR_DRAFT_MODEL` вместо модели профиля (пусто —
    #     как было). Зачем: GigaAM на своей полосе CPU делает черновик записи целиком за 1.3 с на минуту
    #     звука (whisper-пасс-1 на GPU — ~6 с) и ПАРАЛЛЕЛЬНО диаризации (`ASR_OVERLAP`); слова с временами
    #     идут разрезу на куски. `ASR_SECOND_FROM_DRAFT=1` — второе ухо арбитража и корректора берёт слова
    #     черновика в своём окне, не переслушивая: та же модель без подсказки — то же ухо, другое окно;
    #     нет слов в окне — прежний запрос к бэкенду. ---
    draft_model: str = field(default_factory=lambda: _env('ASR_DRAFT_MODEL', ''))
    second_from_draft: bool = field(default_factory=lambda: _flag('ASR_SECOND_FROM_DRAFT', '0'))
    # --- ранние голоса (16.10, ADR-0033 §5): кластеры диаризации узнаются реестром СРАЗУ после диаризации,
    #     а не после правок. Имена узнанных (реестр + карта вызывающего `hints.speaker_names`) идут в `names`
    #     подсказки пасса-2 (seed-проход найдёт их в черновике) и в промпты финал-раунда и корректора вместо
    #     метки кластера, которая им ничего не говорит. Три уровня остаются в данных: кластер → голос → имя. ---
    voices_early: bool = field(default_factory=lambda: _flag('ASR_VOICES_EARLY', '0'))
    # --- очередь сервера (ADR-0033 §6): общий замок на всё, что на GPU (полосы gpu, диаризация, выравнивание),
    #     поверх семафоров ресурсов — две задачи (`ASR_MAX_JOBS=2`) не пересекаются на GPU (13.10: GPU на GPU —
    #     обе стадии вдвое медленнее), а CPU-полоса и LLM свободны. `ASR_GPU_SLOTS=2` — с пулом whisper. ---
    gpu_gate: bool = field(default_factory=lambda: _flag('ASR_GPU_GATE', '0'))
    gpu_slots: int = field(default_factory=lambda: int(_env('ASR_GPU_SLOTS', '1')))
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

    # Выбор провайдера OpenRouter для LLM-стадий — JSON объекта `provider` (`{"ignore": [...]}`,
    # `{"order": [...], "allow_fallbacks": true}`). Одна модель у разных провайдеров ведёт себя
    # по-разному. Замер 2026-10-14, deepseek-v4-flash, глоссарий черновика 1-6, по очереди:
    # любые провайдеры — 1225 с, 32 обрезанных ответа (петля одного термина до потолка токенов),
    # 4 потерянных батча; без StreamLake/AtlasCloud/OpenInference — 49 с, 0, 0. AtlasCloud к тому
    # же игнорирует выключенный reasoning. Ни штраф за частоту, ни температура петлю не снимали.
    # Свойство эндпоинта, а не корпуса: место ему в машинном env. Пусто — не шлём.
    llm_provider: dict = field(default_factory=lambda: json.loads(_env('ASR_LLM_PROVIDER') or '{}'))

    def build_llm(self, model: str | None = None, base_url: str | None = None,
                  api_key: str | None = None) -> 'RetryingLLM':
        """morag LLMClient + прикладной RetryPolicy. SDK-ретраи (max_retries) закрывают транспорт,
        политика поверх — битый JSON деген-петель и спайки; max_concurrent — общий потолок
        одновременных вызовов через реестр семафоров ядра (важен при ASR_MAX_JOBS > 1).

        `model` — другая модель на том же ключе (слушатель корректора): без рассуждений и без
        маршрутизации провайдеров основной модели (у неё свои провайдеры). `base_url`/`api_key` — другой сервер
        (локальное ухо omni)."""
        other = bool(model) and model != self.llm_model or bool(base_url)
        client = LLMClient(base_url=base_url or self.llm_base_url, model=model or self.llm_model,
                           api_key=api_key or self.llm_key,
                           enable_thinking=False if other else _enable_thinking(), timeout=180, max_retries=4,
                           max_concurrent=self.llm_max_concurrent,
                           model_wait_seconds=self.llm_model_wait_s,
                           model_wait_retries=self.llm_model_wait_retries,
                           **({'extra_body': {'provider': self.llm_provider}} if self.llm_provider and not other else {}))
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

    async def complete_with_tools(self, *args, **kwargs):
        """Function calling (редактор расшифровки) — под той же политикой ретраев.

        До этого метод уходил в клиент через `__getattr__` МИМО ретраев: спайк шлюза посреди
        цикла редактора ронял бы решение, а не ждал десять секунд, как остальные стадии."""
        kwargs = self._with_rep(kwargs)
        return await self._policy.call(
            lambda: self._client.complete_with_tools(*args, **kwargs), 'complete_with_tools')


CFG = Config()

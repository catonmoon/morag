"""Пословные тайм-коды: forced alignment готовой расшифровки по звуку.

Это НЕ повторное распознавание. Текст берётся как есть (он уже прошёл пасс-2 и финал-раунд),
акустическая модель лишь показывает, где какое слово прозвучало. Стадия generic для ASR — звук и
реплики уже в конвейере, поэтому и время слова считается здесь, а не у потребителя: `.json` несёт
время каждого слова, дыры в покрытии видны в момент транскрибации.

Порт `../morag-audio-web/tools/align_words.py` (обвязка продукта — фронт-маттер, скачивание mp3,
обход каталогов — отпала: звук и реплики приходят из конвейера). ~185× быстрее реального времени
на M3 Ultra: 85-минутный выпуск за 30 с, считается локально, денег не стоит.

Торч тяжёлый, Docker-образу адаптера не нужен и всё равно упирается в MPS: импорты ЛЕНИВЫЕ,
зависимости — `requirements-align.txt` (Mac-native venv). Нет торча → `pipeline` ловит ImportError,
пишет предупреждение и отдаёт транскрипт без слов.
"""
from __future__ import annotations

import difflib
import logging
import re
import time

from .coverage import turn_windows

log = logging.getLogger('asr')

SR = 16000
FORMAT = 'morag-words-v1'

# Звук режем только для расчёта акустики: сам поиск пути идёт по всей реплике (см. split_turn).
CHUNK = 30.0    # кусок для модели — больше не влезает в память внимания
OVERLAP = 2.0   # нахлёст: края куска модель считает без контекста и врёт

# Сколько секунд реплики уходит в ОДИН поиск пути. Цена растёт как «кадры × токены», причём на
# длинной реплике сверхлинейно: матрица пути перестаёт помещаться в кэш. Замерено на M4 Pro
# (MMS_FA, доклад одним говорящим): 240 с звука — 2.9 с, 480 — 6.0, 960 — 13.0, а на 1440 с стенд
# не закончил за 8.5 минуты; доклад целиком (реплика 1953 с) не закончил в конвейере за 68 минут.
# 240 с — там, где цена ещё линейна по длине, а матрица пути весит десятки мегабайт.
MAX_TURN = 240.0
# Короче — не пауза: в разрез должна укладываться погрешность границ сегментов ASR.
MIN_PAUSE = 0.4
# Как часто стадия говорит, что жива. Десять секунд — чтобы на 13-минутной записи (стадия
# идёт 10 с) строк не было вовсе, а на часовой они шли редко и по делу.
PROGRESS_EVERY = 10.0

_ALIGNER = None


def _spell_numbers(word: str) -> str:
    """«2019» → «две тысячи девятнадцать»: цифр в словаре модели нет.

    Составные («14.30», «0.30») читаем ПО ЧАСТЯМ, как читает человек: «четырнадцать тридцать».
    Дробь «ноль целых три десятых» звучит совсем не так, как сказано вслух — модель такого не
    находит и размазывает интервал на соседей (замерено: именно на этом ломался порядок слов).
    """
    from num2words import num2words

    def sub(m: re.Match) -> str:
        said = []
        for part in re.split(r'[.,:]', m.group(0)):
            if not part:
                continue
            try:
                said.append(num2words(int(part), lang='ru'))
            except (ValueError, OverflowError, NotImplementedError):
                return ' '
        return ' ' + ' '.join(said) + ' '

    return re.sub(r'\d+(?:[.,:]\d+)*', sub, word)


class _Text:
    """Слово расшифровки → латиница из словаря модели (романизация)."""

    def __init__(self) -> None:
        import uroman

        self._uroman = uroman.Uroman()

    def normalize(self, word: str) -> str:
        roman = self._uroman.romanize_string(_spell_numbers(word), lcode='rus')
        return re.sub(r"[^a-z']", '', roman.lower())


def _read_wav(path: str):
    """wav конвейера (16 кГц моно) → тензор. Пересчёт частоты оставлен страховкой."""
    import soundfile as sf
    import torch

    data, sr = sf.read(path, dtype='float32', always_2d=True)
    wave = torch.from_numpy(data.T).mean(0, keepdim=True)  # стерео → моно
    if sr != SR:
        import torchaudio
        wave = torchaudio.functional.resample(wave, sr, SR)
    return wave


class Aligner:
    def __init__(self, device: str) -> None:
        import torchaudio

        bundle = torchaudio.pipelines.MMS_FA
        self.model = bundle.get_model().to(device).eval()
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()
        self.device = device
        self.text = _Text()

    def emission(self, wave):
        """Акустика реплики целиком: считаем кусками, склеиваем в одну матрицу.

        Модель не переварит четырёхминутный кусок разом (память внимания растёт квадратично),
        поэтому режем звук — но ТОЛЬКО звук. Куски идут с нахлёстом, и края каждого отбрасываются:
        у границы модель не видит контекста и врёт.
        """
        import torch

        parts = []
        step = int((CHUNK - OVERLAP) * SR)
        span = int(CHUNK * SR)
        offsets = list(range(0, wave.size(1), step))
        for i, off in enumerate(offsets):
            seg = wave[:, off:off + span]
            if seg.size(1) < SR:  # огрызок короче секунды не несёт информации
                break
            with torch.inference_mode():
                em, _ = self.model(seg.to(self.device))
            em = em.cpu()[0]
            cut = max(1, round(OVERLAP / 2 * em.size(0) / (seg.size(1) / SR)))
            # Обрезаем ТОЛЬКО стыки: у первого куска нет левого соседа, у последнего — правого.
            # Срезать хвост последнего нельзя: кадров станет меньше, чем звука, и все времена
            # растянутся (ловили — начало реплики совпадало, а дальше уезжало всё сильнее).
            first, last = i == 0, off + span >= wave.size(1)
            parts.append(em[0 if first else cut:em.size(0) if last else em.size(0) - cut])
        if not parts:
            return None, 0.0
        stacked = torch.cat(parts)
        return stacked, wave.size(1) / stacked.size(0) / SR

    def turn(self, wave, t0: float, t1: float, words: list[str]) -> list[list]:
        """Слова куска реплики с временами — ОДНИМ проходом по всему куску.

        Порциями по окнам выравнивать нельзя: порция слов обязана лечь в своё окно, лишнего места
        нет, и на пропуске речи (см. stages/coverage.py) всё дальнейшее уезжает. Замерено на живом
        выпуске: сдвиг −36 с, карточка показывала текст из другого места разговора. Здесь путь
        ищется по всему куску сразу, поэтому пропуск просто поглощается (−36 с → доли секунды).
        Куском по умолчанию идёт вся реплика; длинную режет split_turn — строго по паузам.
        """
        a, b = int(max(0, t0) * SR), int(min(t1, wave.size(1) / SR) * SR)
        piece = wave[:, a:b]
        if piece.size(1) < SR:
            return []
        emission, ratio = self.emission(piece)
        if emission is None:
            return []

        # Пунктуация («—», «…») в звуке не звучит и в словарь модели не входит. Между словами
        # ставим «*» — «здесь звучит что-то, чего в тексте нет»: без него выравниватель ОБЯЗАН
        # растянуть переданные слова на весь кусок, и пропуск в расшифровке некуда деть.
        tokens, spoken = ['*'], []
        for i, word in enumerate(words):
            norm = self.text.normalize(word)
            if not norm:
                continue
            spoken.append(i)
            tokens += [norm, '*']

        if not spoken:
            return []
        # CTC не умеет уложить целей больше, чем кадров: на такой реплике (густой текст, короткий
        # звук) отступаем к выравниванию без звёздочек.
        if sum(len(t) for t in tokens) >= emission.size(0):
            tokens = [self.text.normalize(words[i]) for i in spoken]

        try:
            spans = self.aligner(emission, self.tokenizer(tokens))
        except RuntimeError as error:
            log.warning('alignment failed at %.1fs: %s', t0, error)
            return []
        if len(tokens) > len(spoken):  # были звёздочки — берём только слова
            spans = spans[1::2]

        timed = {}
        for i, group in zip(spoken, spans):
            if group:
                timed[i] = (t0 + group[0].start * ratio, t0 + group[-1].end * ratio)

        out: list[list] = []
        for i, word in enumerate(words):
            if i in timed:
                start, end = timed[i]
            elif out:  # пунктуация: прислоняем к предыдущему слову, длины не даём
                start = end = out[-1][2]
            else:
                continue
            out.append([word, round(start, 2), round(end, 2)])
        return out


def _raw_to_final(raw: list[str], final: list[str]) -> list[int]:
    """Индекс слова в СЫРОМ тексте → индекс в финальном.

    Паузы известны про сегменты ASR, то есть про сырой текст, а выравниваем мы финальный: его уже
    поправил финал-раунд («пост грез» → «Postgres», «кэш-то-кэш» → «C2C»). Правки локальные, и
    хватает словесного диффа; внутри изменённого куска индекс размазывается линейно, и промах в
    слово-другое стоит долей секунды у границы куска, а не сдвига всей реплики.
    """
    def key(words):
        return [re.sub(r'\W', '', w.lower()) for w in words]

    pos = [len(final)] * (len(raw) + 1)  # хвост по умолчанию — конец финального текста
    ops = difflib.SequenceMatcher(a=key(raw), b=key(final), autojunk=False).get_opcodes()
    for _, i1, i2, j1, j2 in ops:
        for i in range(i1, i2):
            pos[i] = j1 + (j2 - j1) * (i - i1) // max(1, i2 - i1)
    return pos


def _pauses(segments, t0: float, t1: float) -> list[tuple[float, float, int]]:
    """Внутренние паузы реплики: (середина, длина, сколько сырых слов сказано до неё).

    Пауза — промежуток между соседними сегментами ASR: речи там нет ни по одной версии событий.
    Дыра в покрытии (потерянная речь, stages/coverage.py) попадает сюда же — и резать по ней даже
    безопаснее прочего: слов в ней нет, а значит и увезти нечего.
    """
    out, said = [], 0
    for prev, nxt in zip(segments, segments[1:]):
        said += len(prev['text'].split())
        gap = float(nxt['start']) - float(prev['end'])
        mid = float(prev['end']) + gap / 2
        if gap >= MIN_PAUSE and t0 < mid < t1:
            out.append((mid, gap, said))
    return out


def split_turn(turn, t0: float, t1: float, words: list[str]) -> list[tuple[float, float, int, int]]:
    """Длинная реплика → куски (окно звука, слова [i:j]). Режем ТОЛЬКО по паузам.

    Порциями по окнам резать нельзя (см. Aligner.turn). По паузе — можно: ни одно слово не
    пересекает границу, потому что в паузе слов нет. Границу ставим в СЕРЕДИНУ паузы, чтобы
    погрешность времён сегментов ни у кого не отрезала начало или хвост слова.

    Из пауз в досягаемости берём САМУЮ ДЛИННУЮ, а не ближайшую к капу: длинная пауза — это конец
    мысли, там разрез дешевле всего. Реплики короче капа не трогаем вовсе: у подкаста медиана
    61 с, и его выравнивание обязано остаться прежним.
    """
    whole = [(t0, t1, 0, len(words))]
    if t1 - t0 <= MAX_TURN or not words:
        return whole
    segments = turn.get('segments') or []
    cuts = _pauses(segments, t0, t1)
    if not cuts:  # реплика без сегментов (чужой вызов) или сплошная речь — идём как раньше
        log.warning('align: реплика %.0f с на %.0f-й секунде, пауз для разреза нет — целиком',
                    t1 - t0, t0)
        return whole
    final = _raw_to_final([w for s in segments for w in s['text'].split()], words)

    pieces, a, i = [], t0, 0
    while t1 - a > MAX_TURN:
        # в окне капа пауз может не оказаться — тогда тянемся до ближайшей следующей: кусок выйдет
        # длиннее капа, но резать не по паузе нельзя ни при каких обстоятельствах
        # кусок короче эмиссионного (CHUNK) резать незачем, а короче секунды модель не примет
        # и слова остались бы без времён вовсе
        left = [c for c in cuts if c[0] > a + CHUNK]
        reach = [c for c in left if c[0] <= a + MAX_TURN] or left[:1]
        if not reach:
            log.warning('align: после %.0f-й секунды пауз нет — остаток %.0f с идёт целиком',
                        a, t1 - a)
            break
        mid, _, said = max(reach, key=lambda c: (c[1], c[0]))  # длиннейшая, при равных — поздняя
        j = min(max(final[said], i), len(words))
        pieces.append((a, mid, i, j))
        a, i = mid, j
    pieces.append((a, t1, i, len(words)))
    if len(pieces) > 1 and t1 - a < CHUNK:  # огрызок в хвосте — к предыдущему куску
        (a0, _, i0, _), (_, b1, _, j1) = pieces[-2], pieces[-1]
        pieces[-2:] = [(a0, b1, i0, j1)]
    pieces = [p for p in pieces if p[3] > p[2]]  # кусок без слов модели не отдаём
    log.info('align: реплика %.0f с (t=%.0f) разрезана по паузам на %d кусков',
             t1 - t0, t0, len(pieces))
    return pieces


def enforce_order(turns) -> int:
    """Порядок слов по времени — обязательство перед потребителем: караоке ищет текущее слово
    двоичным поиском и на неотсортированном врёт молча. Счётчик сквозной: собеседники перебивают
    друг друга, и разъехаться слова могут не только внутри реплики, но и на стыке двух."""
    fixed, prev_end = 0, 0.0
    for turn in turns:
        for word in turn['words']:
            if word[1] < prev_end:
                word[1] = prev_end
                word[2] = max(word[2], prev_end)
                fixed += 1
            prev_end = word[2]
    return fixed


def _device(explicit: str = '') -> str:
    import torch

    if explicit:
        return explicit
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _get_aligner(device: str) -> Aligner:
    """Модель грузится один раз на процесс (адаптер держит одну джобу in-flight)."""
    global _ALIGNER
    if _ALIGNER is None or _ALIGNER.device != device:
        _ALIGNER = Aligner(device)
    return _ALIGNER


def align_turns(wav_path: str, turns, audio_sec: float, *, episode: str = '',
                device: str = '') -> dict:
    """Реплики [{start, speaker, text}] + звук → готовый документ morag-words-v1.

    Блокирующая (torch/MPS) — конвейер зовёт через asyncio.to_thread.
    """
    dev = _device(device)
    aligner = _get_aligner(dev)
    wave = _read_wav(wav_path)
    started = last_said = time.monotonic()
    log.info('align: %d реплик, %.0f с звука, устройство %s', len(turns), audio_sec, dev)

    out = []
    for turn, (a, b) in zip(turns, turn_windows(turns, audio_sec)):
        words = (turn.get('text') or '').split()
        timed: list[list] = []
        for wa, wb, i, j in (split_turn(turn, a, b, words) if words else []):
            timed += aligner.turn(wave, wa, wb, words[i:j])
        out.append({'start': round(a, 2), 'end': round(b, 2), 'speaker': turn.get('speaker', ''),
                    'words': timed})
        # Прогресс: стадия идёт десятки секунд и до сих пор не говорила ни слова — в логе
        # она была неотличима от зависшей джобы, а именно так она однажды и провисела
        # 68 минут. Раз в PROGRESS_EVERY секунд, а не на каждой реплике: у длинного доклада
        # реплик тридцать, у планёрки — три сотни.
        now = time.monotonic()
        if now - last_said >= PROGRESS_EVERY:
            last_said = now
            log.info('align: %.0f%% (%.0f из %.0f с звука) за %.0f с',
                     100 * b / max(audio_sec, 1), b, audio_sec, now - started)
    fixed = enforce_order(out)
    log.info('align: готово за %.1f с, слов %d, порядок правился %d раз',
             time.monotonic() - started, sum(len(t['words']) for t in out), fixed)

    return {'format': FORMAT, 'episode': episode, 'duration_sec': round(audio_sec, 2),
            'words_total': sum(len(t['words']) for t in out), 'reordered': fixed,
            'device': dev, 'turns': out}

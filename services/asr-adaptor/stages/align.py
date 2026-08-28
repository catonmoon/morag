"""Пословные тайм-коды: forced alignment готовой расшифровки по звуку.

Это НЕ повторное распознавание. Текст берётся как есть (он уже прошёл пасс-2 и финал-раунд),
акустическая модель лишь показывает, где какое слово прозвучало. Стадия generic для ASR — звук и
реплики уже в конвейере, поэтому и время слова считается здесь, а не у потребителя: `.json` несёт
время каждого слова, дыры в покрытии видны в момент транскрибации.

Портирован из продуктовой обвязки (фронт-маттер, скачивание mp3,
обход каталогов — отпала: звук и реплики приходят из конвейера). ~185× быстрее реального времени
на большой Apple-Silicon-машине: 85-минутный выпуск за 30 с, считается локально, денег не стоит.

Торч тяжёлый, Docker-образу адаптера не нужен и всё равно упирается в MPS: импорты ЛЕНИВЫЕ,
зависимости — `requirements-align.txt` (Mac-native venv). Нет торча → `pipeline` ловит ImportError,
пишет предупреждение и отдаёт транскрипт без слов.
"""
from __future__ import annotations

import logging
import re

from .coverage import turn_windows

log = logging.getLogger('asr')

SR = 16000
FORMAT = 'morag-words-v1'

# Звук режем только для расчёта акустики: сам поиск пути идёт по всей реплике.
CHUNK = 30.0    # кусок для модели — больше не влезает в память внимания
OVERLAP = 2.0   # нахлёст: края куска модель считает без контекста и врёт

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
        """Слова одной реплики с временами — ОДНИМ проходом по всей реплике.

        Порциями по окнам выравнивать нельзя: порция слов обязана лечь в своё окно, лишнего места
        нет, и на пропуске речи (см. stages/coverage.py) всё дальнейшее уезжает. Замерено на живом
        выпуске: сдвиг −36 с, карточка показывала текст из другого места разговора. Здесь путь
        ищется по всей реплике сразу, поэтому пропуск просто поглощается (−36 с → доли секунды).
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

    out = []
    for turn, (a, b) in zip(turns, turn_windows(turns, audio_sec)):
        words = (turn.get('text') or '').split()
        out.append({'start': round(a, 2), 'end': round(b, 2), 'speaker': turn.get('speaker', ''),
                    'words': aligner.turn(wave, a, b, words) if words else []})
    fixed = enforce_order(out)

    return {'format': FORMAT, 'episode': episode, 'duration_sec': round(audio_sec, 2),
            'words_total': sum(len(t['words']) for t in out), 'reordered': fixed,
            'device': dev, 'turns': out}

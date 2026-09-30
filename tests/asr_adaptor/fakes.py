"""Заглушки для сквозных прогонов графа и конвейера: бэкенд, у которого есть что арбитрировать.

Три слова в куске; второе ухо слышит иначе; чистое ухо в середине записи соглашается со вторым
(голосование), в конце — петля (переслушивание); спорное слово звучит ОДИНАКОВО из любого окна,
накрывающего то же место (окно третьего голоса и ухо финал-раунда начинаются раньше куска).
Ставится поверх `pipeline.<имя>` — те же имена, что патчит `test_pipeline_recovery`, поэтому
одна установка гоняет обе реализации.
"""
from __future__ import annotations

import wave
from pathlib import Path

import pipeline

SR = 16000
AUDIO_S = 90.0
PASS1 = [{'start': 0.0, 'end': 20.0, 'text': 'начало разговора'},
         {'start': 50.0, 'end': 90.0, 'text': 'продолжение разговора'}]
SILENT_CHUNK = (0.0, 20.0)      # на этот кусок пасс-2 ответит пусто, пока его спрашивают с промптом
HINTS = {'terms': ['Postgres', 'Kubernetes'], 'names': ['Мария Кузнецова'], 'about': 'демо'}


def make_silence(path: Path, seconds: float = AUDIO_S) -> Path:
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b'\0' * (int(seconds * SR) * 2))
    return path


class RichBackend:
    def __init__(self) -> None:
        self.slices: dict[str, tuple[float, float]] = {}
        self.calls: list[dict] = []

    def cut(self, wav, a, b, dst):
        self.slices[dst] = (a, b)

    def asr(self, path: str, prompt: str = '', model: str = '', words: bool = False) -> dict:
        if path.endswith('in.wav'):
            return {'text': ' '.join(s['text'] for s in PASS1), 'segments': PASS1}
        a, b = self.slices[path]
        self.calls.append({'a': round(a, 2), 'b': round(b, 2), 'prompt': prompt, 'model': model, 'words': words})
        if SILENT_CHUNK[0] <= a < SILENT_CHUNK[1] and prompt:
            return {'text': '', 'segments': []}
        mid = (a + b) / 2
        if model or (not prompt and 20 <= mid < 50):
            text = f'начало печь-{int(mid // 15) * 15} конец'
        elif a >= 50 and prompt:
            text = 'ИИИИИИИИИИИИ петля'
        else:
            text = f'начало речь-{a:.0f} конец'
        seg = {'start': 0.0, 'end': b - a, 'text': text, 'avg_logprob': -0.3}
        if words:
            seg['words'] = [{'word': ' ' + w, 'start': 0.1 * k, 'end': 0.1 * k + 0.05, 'probability': 0.9}
                            for k, w in enumerate(text.split())]
        return {'text': text, 'segments': [seg]}


def install_rich(monkeypatch, wav: Path) -> RichBackend:
    fake = RichBackend()
    monkeypatch.setattr(pipeline, '_RES_SEMS', {})   # семафоры чужого цикла событий
    monkeypatch.setattr(pipeline, '_to_wav', lambda src, dst: Path(dst).write_bytes(wav.read_bytes()))
    monkeypatch.setattr(pipeline, '_slice', fake.cut)
    monkeypatch.setattr(pipeline.audio_clients, 'asr', fake.asr)
    monkeypatch.setattr(pipeline.audio_clients, 'diarize',
                        lambda p: [{'start': 0.0, 'end': AUDIO_S, 'speaker': 'SPEAKER_00'}])
    monkeypatch.setattr(pipeline.audio_clients, 'campp', lambda p, spans: ({}, {}))
    monkeypatch.setattr(pipeline.audio_clients, 'campp_spans', lambda p, spans: [None] * len(spans))
    monkeypatch.setattr(pipeline.registry, 'assign', lambda *a, **kw: {'SPEAKER_00': 'Speaker_0'})
    monkeypatch.setattr(pipeline.registry, 'names', lambda path: {})

    async def nothing(*a, **kw):
        return []

    async def empty_text(*a, **kw):
        return ''

    async def no_names(*a, **kw):
        return {}, []

    async def reader(llm, text, terms):
        return ['речь-20'] if 'речь-20' in text or 'печь-20' in text else []

    async def correct(text, dsum, ctx, canonicals, llm, always=(), recalled='', corpus_desc='',
                      fixes_out=None, protect=()):
        # Замена известного слова, отвергнутая сторожем: её и переслушивает `final_ear`.
        if fixes_out is not None and 'речь-20' in text:
            fixes_out.append({'was': 'речь-20', 'now': 'печь-20', 'ok': False, 'why': 'known_term'})
        return text

    monkeypatch.setattr(pipeline, 'build_glossary', nothing)
    monkeypatch.setattr(pipeline, 'build_hints', nothing)
    monkeypatch.setattr(pipeline, 'doc_summary', empty_text)
    monkeypatch.setattr(pipeline, 'recall_entities', empty_text)
    monkeypatch.setattr(pipeline, 'correct', correct)
    monkeypatch.setattr(pipeline, 'has_entity_signal', lambda raw, gloss: 'речь-20' in raw)
    monkeypatch.setattr(pipeline, 'name_speakers', no_names)
    monkeypatch.setattr(pipeline, 'WhisperTokenCounter', lambda model: None)
    monkeypatch.setattr(pipeline, 'build_prompt',
                        lambda terms, counter, budget, always=(), hinted=(): 'каноники')
    monkeypatch.setattr(pipeline.arbitrate_stage, 'reader_flags', reader)
    return fake

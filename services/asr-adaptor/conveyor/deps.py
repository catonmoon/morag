"""Зависимости узлов: помощники и стадии — ПОЗДНИМ связыванием через модуль `pipeline`.

⚠️⚠️ Узлы берут `pipeline._slice`, `pipeline.audio_clients.asr`, `pipeline.build_prompt` и прочее
атрибутом модуля В МОМЕНТ ВЫЗОВА, а не импортом имени. Причина — тесты: заглушки бэкендов и
LLM-стадий патчат именно `pipeline.<имя>` (см. `tests/asr_adaptor/test_pipeline_recovery.py`).
Вторая причина — семафоры GPU (`pipeline._res`) и билеты реестра голосов: они обязаны быть ОДНИМИ
на процесс, иначе при `ASR_MAX_JOBS>1` гейтинг раздвоится, а нумерация `Speaker_N` параллельных
выпусков поедет.
"""
from __future__ import annotations

# Всё, что узлы берут у модуля `pipeline`. Список — документация и тест (`Deps.check`): имя, которое
# исчезло из `pipeline.py`, всплывёт тестом, а не AttributeError посреди часового прогона.
NAMES = (
    # конфиг, лог, клиенты, стадии-модули
    'CFG', 'log', 'audio_clients', 'registry', 'coverage', 'align', 'arbitrate_stage', 'relisten_stage',
    # ресурсы и порядок параллельных выпусков
    '_res', '_take_ticket', '_wait_turn', '_release_turn',
    # звук
    '_to_wav', '_slice', '_decode', '_relisten_chunk', 'PAD_S',
    # знание о записи
    'build_glossary', 'build_hints', 'merge_hints', 'reconcile', 'witnessed', 'selflabelled',
    'hinted_canonicals', 'relevant', 'WhisperTokenCounter', 'build_prompt', '_neighbour_text',
    # нарезка
    'chunk_fn', 'gap_chunks', 'chunking_min_s',
    # финал-раунд и наминг
    'doc_summary', 'has_entity_signal', 'recall_entities', 'correct', 'apply_fixes',
    '_final_round', '_ear_prefers', '_around', '_window', 'name_speakers',
    # лента
    '_emit_spans', '_emit_draft', '_project',
    # отпечаток установки
    'stack_fingerprint', 'one_line',
)


class Deps:
    """Ручки к `pipeline.*` и клиент LLM. `d.<имя>` → `pipeline.<имя>` при каждом обращении."""

    def __init__(self, llm, pipeline_module=None) -> None:
        if pipeline_module is None:
            import pipeline as pipeline_module  # noqa: PLC0415 — поздно и намеренно
        self._p = pipeline_module
        self.llm = llm

    @property
    def cfg(self):
        return self._p.CFG

    def __getattr__(self, name: str):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self._p, name)

    def check(self) -> list[str]:
        """Имена из `NAMES`, которых у конвейера нет."""
        return [n for n in NAMES if not hasattr(self._p, n)]

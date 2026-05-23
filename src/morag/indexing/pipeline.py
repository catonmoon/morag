from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque

from morag.indexing.chunker import Chunker, PassthroughChunker
from morag.indexing.context import ContextGenerator, NoopContextGenerator
from morag.indexing.processors import ChunkProcessor, DocumentProcessor
from morag.indexing.splitter import (
    FixedSizeSplitter,
    MarkdownHeaderSplitter,
    RecursiveSplitter,
    TableRowSplitter,
    pack_blocks,
)
from morag.indexing.status_reporter import NullStatusReporter, StatusReporter
from morag.indexing.token_counter import TokenCounter, TiktokenCounter
from morag.run_context import RunContext
from morag.sources.base import Chunk, Document, Source
from morag.storage.repository import ChunkRepository, DocRepository

logger = logging.getLogger(__name__)

_DEFAULT_BLOCK_LIMIT = 2048  # токенов на один блок перед чанкованием


def _topological_levels(stubs: list) -> list[list]:
    """Разбить стабы по уровням BFS: родители всегда на уровень выше потомков.

    Учитываются только рёбра внутри текущего батча (parent_doc_ids, входящие в id_set).
    Документы с внешними или отсутствующими родителями попадают на уровень 0.
    """
    id_set = {s.id for s in stubs}
    id_to_stub = {s.id: s for s in stubs}

    in_degree: dict[str, int] = {
        s.id: sum(1 for p in s.parent_doc_ids if p in id_set) for s in stubs
    }
    children: dict[str, list[str]] = defaultdict(list)
    for s in stubs:
        for p in s.parent_doc_ids:
            if p in id_set:
                children[p].append(s.id)

    levels: list[list] = []
    queue: deque[str] = deque(sid for sid, deg in in_degree.items() if deg == 0)
    while queue:
        level_ids = list(queue)
        queue.clear()
        levels.append([id_to_stub[i] for i in level_ids])
        for sid in level_ids:
            for child_id in children[sid]:
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)

    return levels


class IndexingPipeline:
    """Оркестратор индексации документов.

    Полный цикл через run():
    1. Загружает документы из Source.
    2. Проверяет актуальность (idempotency).
    3. Прогоняет через DocumentProcessor-цепочку.
    4. Сохраняет документ в Qdrant (коллекция docs).
    5. Разбивает на блоки (RecursiveSplitter + pack_blocks).
    6. Чанкует каждый блок (Chunker).
    7. Генерирует контекстуальное суммари (ContextGenerator).
    8. Прогоняет чанки через ChunkProcessor-цепочку.
    9. Сохраняет чанки в Qdrant (коллекция chunks).
    """

    def __init__(
        self,
        doc_repo: DocRepository,
        chunk_repo: ChunkRepository,
        doc_processors: list[DocumentProcessor] | None = None,
        chunk_processors: list[ChunkProcessor] | None = None,
        chunker: Chunker | None = None,
        context_generator: ContextGenerator | None = None,
        token_counter: TokenCounter | None = None,
        block_limit: int = _DEFAULT_BLOCK_LIMIT,
        concurrency: int = 1,
        skip_presplit: bool = False,
        passthrough_threshold: int | None = None,
        embed_batch_size: int = 64,
        max_table_rows: int = 0,
        narrate_tables_enabled: bool = False,
        narrate_tables_min_rows: int = 5,
        status_reporter: StatusReporter | None = None,
        cancel_event: asyncio.Event | None = None,
        run_context: RunContext | None = None,
        embedder_fingerprint: str | None = None,
        reindex_floor: int | None = None,
    ) -> None:
        self._doc_repo = doc_repo
        self._chunk_repo = chunk_repo
        self._doc_processors = doc_processors or []
        self._chunk_processors = chunk_processors or []
        self._chunker = chunker or PassthroughChunker()
        self._context_generator = context_generator or NoopContextGenerator()
        self._token_counter = token_counter or TiktokenCounter()
        self._block_limit = block_limit
        self._concurrency = max(1, concurrency)
        self._skip_presplit = skip_presplit
        self._passthrough_threshold = passthrough_threshold
        self._embed_batch_size = embed_batch_size
        self._max_table_rows = max_table_rows
        self._narrate_tables_enabled = narrate_tables_enabled
        self._narrate_tables_min_rows = narrate_tables_min_rows
        self._status_reporter: StatusReporter = status_reporter or NullStatusReporter()
        self._cancel_event = cancel_event or asyncio.Event()
        # Run versioning + embedder fingerprint (см. ADR-0012). None для CLI-режимов
        # без RunContext / без явного fingerprint — payload-fields просто не
        # пишутся, fingerprint-mismatch не проверяется.
        self._run_context = run_context
        self._embedder_fingerprint = embedder_fingerprint
        # Реиндекс-эффорт (ADR-0014): если задан — актуальность документа
        # определяется по run_number (>= floor = уже обработан эффортом),
        # обычные updated_at/fingerprint/count проверки не применяются.
        self._reindex_floor = reindex_floor
        if not skip_presplit or passthrough_threshold:
            self._splitter = RecursiveSplitter(
                self._token_counter,
                self._block_limit,
                splitters=[
                    MarkdownHeaderSplitter(),
                    TableRowSplitter(self._token_counter, self._block_limit),
                    FixedSizeSplitter(self._token_counter, self._block_limit),
                ],
            )

    async def _predict_real_count(self, stubs: list[Document]) -> int:
        """Pre-pass: оценить сколько stubs реально пройдут обработку (не skip).

        Делает ОДИН batch-fetch payload'ов всех stubs из docs collection и
        сравнивает in-memory: updated_at + embedder_fingerprint. Это покрывает
        ~95% skip-кейсов (нормальный idempotency-flow). НЕ проверяет
        count(chunks) — это потребовало бы отдельных запросов в chunks
        collection per doc. Если документ имеет неполный набор чанков
        (предыдущий crash), pre-pass его посчитает как skip, а main-loop
        обработает как real — predicted_real_total окажется чуть занижен.
        Это приемлемо: реальный ETA получится слегка пессимистичным.

        Reindex-effort (`reindex_floor`) — все документы с run_number < floor
        считаются real. Если floor не задан — обычный idempotency-предикт.
        """
        if not stubs:
            return 0
        stored = await self._doc_repo.get_payloads_by_ids([s.id for s in stubs])
        real_count = 0
        for stub in stubs:
            payload = stored.get(stub.id)
            if payload is None:
                real_count += 1  # документа нет в сторе → новый, real
                continue
            if self._reindex_floor is not None:
                rn = payload.get('run_number')
                if not isinstance(rn, int) or rn < self._reindex_floor:
                    real_count += 1
                continue
            # Обычный flow: updated_at + embedder_fingerprint
            stored_updated = payload.get('updated_at')
            stub_updated_iso = stub.updated_at.isoformat() if stub.updated_at else None
            if stored_updated != stub_updated_iso:
                real_count += 1
                continue
            if self._embedder_fingerprint is not None:
                if payload.get('embedder_fingerprint') != self._embedder_fingerprint:
                    real_count += 1
                    continue
            # Иначе считаем skip (предположительно — финальная проверка
            # count(chunks) делается уже в main-loop).
        return real_count

    async def _is_up_to_date(self, stub: Document) -> bool:
        """Проверить актуальность документа по стабу метаданных без загрузки контента."""
        existing = await self._doc_repo.get_by_id(stub.id)
        if existing is None:
            return False

        # Реиндекс-эффорт (ADR-0014): актуальность определяется ТОЛЬКО прогоном.
        # Документ актуален ⇔ уже обработан эффортом — его точка коммита несёт
        # run_number >= floor. Резюм после краха так пропускает уже сделанное и
        # доделывает остальное. updated_at/fingerprint/чанки при эффорте не
        # смотрим: задача эффорта — пересобрать всё в scope.
        if self._reindex_floor is not None:
            run_number = existing.payload.get('run_number')
            return isinstance(run_number, int) and run_number >= self._reindex_floor

        if existing.updated_at != stub.updated_at:
            return False

        # Embedder fingerprint mismatch → переиндексировать (векторы stale).
        # Защита от silent staleness при смене embedder-модели.
        if self._embedder_fingerprint is not None:
            existing_fp = existing.payload.get('embedder_fingerprint')
            if existing_fp != self._embedder_fingerprint:
                logger.info(
                    'Embedder fingerprint changed for %s: %r → %r — reindexing',
                    stub.id, existing_fp, self._embedder_fingerprint,
                )
                return False

        # Структурные документы не имеют чанков — достаточно совпадения метаданных
        if existing.structural:
            return True

        status = await self._chunk_repo.get_index_status(stub.id)
        if status is None:
            return False
        count, total = status
        return count == total

    async def _prepare_document(self, document: Document, w: str = '') -> Document | None:
        """Проверить idempotency, прогнать процессоры, проштамповать payload.

        Возвращает обработанный документ или None если документ актуален.
        Документ здесь НЕ упсертится: точка документа сохраняется последней,
        после чанков — это точка коммита replace-not-delete (ADR-0014). Для
        не-структурных документов upsert делает _chunk_document, для
        структурных — run().
        """
        logger.info('%sPreparing document: %s (size=%d)', w, document.id, document.size)
        existing = await self._doc_repo.get_by_id(document.id)

        prev_version = 0
        content_changed = True
        if existing is not None:
            # Embedder fingerprint mismatch проверяется в _is_up_to_date;
            # здесь — fallback path для уже-загруженного документа.
            fingerprint_ok = (
                self._embedder_fingerprint is None
                or existing.payload.get('embedder_fingerprint') == self._embedder_fingerprint
            )
            content_changed = existing.updated_at != document.updated_at

            # При реиндекс-эффорте skip-проверка отключена — документ всегда
            # переобрабатывается (актуальность уже отфильтрована _is_up_to_date
            # по run_number).
            if self._reindex_floor is None and not content_changed and fingerprint_ok:
                status = await self._chunk_repo.get_index_status(document.id)
                if status is not None:
                    count, total = status
                    if count == total:
                        logger.info('%sDocument up to date, skipping: %s', w, document.id)
                        return None

            prev_version = int(existing.payload.get('version', 0))
            logger.info(
                '%sRe-indexing document: %s (prev_version=%d, content_changed=%s)',
                w, document.id, prev_version, content_changed,
            )
            # Контент изменился → attached-дети могли измениться, сносим их
            # (пере-деривируются в фазе вложений). При чистом реиндексе того же
            # контента дети не трогаются. Чанки и точку документа НЕ удаляем —
            # старое остаётся запрашиваемым до swap-delete (replace-not-delete).
            if content_changed:
                await self._doc_repo.delete_attached(document.id, self._chunk_repo)

        # Прогоняем через цепочку процессоров (LLM-операции живут здесь)
        for processor in self._doc_processors:
            document = await processor.process(document)

        # version — ось контента (ADR-0014): бампается только при изменении
        # самого документа; чистый реиндекс того же контента version не меняет.
        version = prev_version + 1 if content_changed else prev_version
        self._stamp_payload(document.payload, version=version)

        return document

    def _stamp_payload(self, payload: dict, version: int | None = None) -> None:
        """Добавить run_number/indexed_at/version/embedder_fingerprint в payload.

        Шарится между документами и чанками одного прогона: indexed_at заморожен
        при RunContext.begin(), run_number = одно значение на весь прогон.
        version — per-document, для чанков наследуется из родительского документа.
        """
        if self._run_context is not None:
            payload['run_number'] = self._run_context.run_number
            payload['indexed_at'] = self._run_context.indexed_at
        if version is not None:
            payload['version'] = version
        if self._embedder_fingerprint is not None:
            payload['embedder_fingerprint'] = self._embedder_fingerprint

    async def run(self, source: Source) -> None:
        """Полный цикл индексации с параллельной обработкой документов.

        Сначала загружаются метаданные всех документов и выполняется idempotency-проверка.
        Затем документы, требующие переиндексации, обрабатываются конкурентно — не более
        `concurrency` одновременно. Каждый документ: load_one → prepare → chunk → upsert.
        """
        stubs = await source.get_metadata()
        total = len(stubs)
        logger.info('Loaded metadata for %d document(s) from source (concurrency=%d)', total, self._concurrency)
        self._status_reporter.start_phase(f'indexing_{source.source_type}', total)

        # Full sync: удалить документы, которых больше нет в источнике.
        # Скоуп — конкретный инстанс (kind, name, source_type), иначе при
        # нескольких инстансах одного source_type (multi-Confluence) второй
        # снесёт документы первого как orphans.
        current_ids = {stub.id for stub in stubs}
        stored_ids = await self._doc_repo.get_ids_by_source_instance(
            source.source_type, source.kind, source.name
        )
        orphaned = stored_ids - current_ids
        if orphaned:
            logger.info(
                'Deleting %d orphaned document(s) for source %s:%s (source_type=%s)',
                len(orphaned), source.kind, source.name, source.source_type,
            )
            for doc_id in orphaned:
                await self._doc_repo.cascade_delete(doc_id, self._chunk_repo)

        # Pre-pass: предсказать сколько документов реально пройдут обработку
        # (не skipnut'ы по idempotency). Один batch Qdrant запрос вместо N
        # отдельных. Используется UI для точного ETA с самого старта прогона —
        # без extrapolation skip-ratio из rolling-окна.
        predicted_real = await self._predict_real_count(stubs)
        self._status_reporter.set_predicted_real_total(predicted_real)
        logger.info(
            'Pre-pass: predicted %d/%d documents will need real processing (~%d%% skip)',
            predicted_real, total, int((1 - predicted_real / max(total, 1)) * 100),
        )

        sem = asyncio.Semaphore(self._concurrency)

        async def process_one(i: int, stub: Document) -> bool:
            """Обработать один документ. Возвращает True если был проиндексирован."""
            w = f'[{i}/{total}] '
            async with sem:
                # Cancel может прийти после планирования gather, но до acquire
                # этого конкретного слота семафора — выходим сразу, не делая работу.
                if self._cancel_event.is_set():
                    logger.info('%sCancelled before start: %s', w, stub.id)
                    return False
                try:
                    logger.info('%sChecking [%d/%d]: %s', w, i, total, stub.id)
                    if await self._is_up_to_date(stub):
                        logger.info('%sDocument up to date, skipping: %s', w, stub.id)
                        self._status_reporter.document_done(stub.id)
                        return False

                    self._status_reporter.document_start(stub.id, stub.title, stub.url)
                    try:
                        document = await source.load_one(stub.id)
                        if document is None:
                            logger.warning('%sFailed to load document: %s', w, stub.id)
                            return False

                        prepared = await self._prepare_document(document, w=w)
                        if prepared is None:
                            return False

                        if prepared.structural:
                            # Структурные документы без чанков — точку документа
                            # упсертим здесь (для не-структурных это делает
                            # _chunk_document последним шагом).
                            logger.info('%sStructural document, skipping chunking: %s', w, prepared.id)
                            await self._doc_repo.upsert(prepared)
                        else:
                            await self._chunk_document(prepared, w=w)
                        return True
                    finally:
                        self._status_reporter.document_done(stub.id)
                except Exception as exc:
                    # replace-not-delete (ADR-0014): точку документа упсертим
                    # последним шагом, поэтому при падении она НЕ закоммичена —
                    # документ остаётся на прежней версии со старыми чанками,
                    # запрашиваемым. Недоделанные новые чанки (текущий run_number)
                    # сметёт swap-delete следующего реиндекса. Чистить не нужно.
                    logger.exception('%sDocument failed: %s', w, stub.id)
                    self._status_reporter.document_failed(stub.id, stub.title, exc)
                    return False

        levels = _topological_levels(stubs)
        logger.info('Processing order: %d level(s)', len(levels))

        results: list[bool] = []
        stub_idx = 0
        for level_num, level in enumerate(levels):
            if self._cancel_event.is_set():
                logger.info('Cancel requested — stopping before level %d', level_num)
                break
            logger.info('Level %d: %d document(s)', level_num, len(level))
            level_results = await asyncio.gather(
                *[process_one(stub_idx + i + 1, stub) for i, stub in enumerate(level)]
            )
            results.extend(level_results)
            stub_idx += len(level)

        indexed = sum(results)
        if self._cancel_event.is_set():
            logger.info('Indexing cancelled: %d indexed before stop, %d not processed', indexed, total - len(results))
        else:
            logger.info('Indexing complete: %d indexed, %d skipped', indexed, total - indexed)

    async def _presplit_and_chunk(self, document: Document, w: str = '') -> list[str]:
        """Pre-split на блоки + жадная упаковка + chunker для каждой пачки."""
        blocks = self._splitter.split(document.text)
        packs = pack_blocks(blocks, self._token_counter, self._block_limit)
        logger.info('%s  Pre-split: %d block(s) -> %d pack(s)', w, len(blocks), len(packs))

        chunk_texts: list[str] = []
        for i, pack in enumerate(packs):
            block_text = '\n\n'.join(pack)
            block_tokens = self._token_counter.count(block_text)
            logger.info(
                '%s  Chunking pack %d/%d (%d chars, ~%d tokens)...',
                w, i + 1, len(packs), len(block_text), block_tokens,
            )
            new_chunks = await self._chunker.chunk(block_text)
            logger.info('%s    -> %d chunk(s)', w, len(new_chunks))
            chunk_texts.extend(new_chunks)
        return chunk_texts

    async def _passthrough_chunk(self, document: Document, w: str = '') -> list[str]:
        """Pre-split на блоки + merge до min_tokens — без SemanticChunker."""
        blocks = self._splitter.split(document.text)
        min_tokens = 128
        max_tokens = self._chunker._max_tokens if hasattr(self._chunker, '_max_tokens') else 256

        # Склеиваем блоки: накапливаем пока < min_tokens,
        # сбрасываем когда >= min_tokens или следующий блок не влезет
        merged: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for block in blocks:
            if not block.strip():
                continue
            bt = self._token_counter.count(block)
            # Если добавление превысит max_tokens и уже набрали min_tokens — сбросить
            if current_tokens + bt > max_tokens and current_tokens >= min_tokens:
                merged.append('\n\n'.join(current))
                current = [block]
                current_tokens = bt
            else:
                # Добавляем: либо не превышает max, либо ещё не набрали min
                current.append(block)
                current_tokens += bt
        if current:
            if merged and current_tokens < min_tokens:
                # Последний кусок мелкий — приклеить к предыдущему
                merged[-1] += '\n\n' + '\n\n'.join(current)
            else:
                merged.append('\n\n'.join(current))

        logger.info(
            '%s  Passthrough: %d block(s) -> %d chunk(s) (min %d, max %d tok)',
            w, len(blocks), len(merged), min_tokens, max_tokens,
        )
        return merged

    async def _chunk_document(self, document: Document, w: str = '') -> None:
        """Разбить документ на чанки и сохранить в Qdrant.

        replace-not-delete (ADR-0014): новые чанки вставляются поверх старых,
        затем swap-delete сметает чанки прежних прогонов, и последним шагом
        упсертится точка документа (точка коммита). Корпус остаётся
        запрашиваемым всё время реиндекса.
        """
        logger.info('%sChunking: %s', w, document.id)

        # replace-not-delete опирается на run_number чанков для swap-delete.
        # Без RunContext run_number не штампуется — нечем строить swap-предикат,
        # fallback на delete-upfront (краткое окно неполного документа; путь
        # для тестов / голого CLI, где плавность не важна).
        if self._run_context is None:
            await self._chunk_repo.delete_by_doc_id(document.id)

        # Чанкинг: chunk_with_metadata (hybrid) или обычный chunk
        chunk_results: list | None = None  # list[ChunkResult] если hybrid

        if self._skip_presplit:
            doc_tokens = self._token_counter.count(document.text)
            if self._passthrough_threshold and doc_tokens > self._passthrough_threshold:
                logger.info(
                    '%s  Passthrough fallback (%d chars, ~%d tokens > %d threshold)...',
                    w, len(document.text), doc_tokens, self._passthrough_threshold,
                )
                chunk_texts = await self._passthrough_chunk(document, w)
            elif hasattr(self._chunker, 'chunk_with_metadata'):
                logger.info('%s  Hybrid chunking (%d chars, ~%d tokens)...', w, len(document.text), doc_tokens)
                chunk_results = await self._chunker.chunk_with_metadata(
                    document.text, paged=document.paged,
                )
                logger.info('%s  -> %d chunk(s)', w, len(chunk_results))
            else:
                logger.info('%s  Semantic chunking (%d chars, ~%d tokens)...', w, len(document.text), doc_tokens)
                chunk_texts = await self._chunker.chunk(document.text)
                logger.info('%s  -> %d chunk(s)', w, len(chunk_texts))
        else:
            chunk_texts = await self._presplit_and_chunk(document, w)

        # Нормализуем: если нет chunk_results, создаём из текстов
        if chunk_results is None:
            from morag.indexing.chunker import ChunkResult
            chunk_results = [ChunkResult(text=t) for t in chunk_texts]

        total = len(chunk_results)
        logger.info('%s  Total chunks: %d', w, total)
        self._status_reporter.document_set_chunks(document.id, total)

        # Чанки наследуют version от родительского документа (атомарность прогона)
        doc_version = document.payload.get('version')

        # Собираем Chunk-объекты с order/total, генерируем context
        chunks: list[Chunk] = []
        for order, cr in enumerate(chunk_results):
            chunk_tokens = self._token_counter.count(cr.text)
            logger.info(
                '%s  Processing chunk %d/%d (~%d tok): %s...',
                w, order + 1, total, chunk_tokens, repr(cr.text[:60]),
            )
            doc_summary = document.payload.get('doc_summary', '')
            context = await self._context_generator.generate(
                document.text, cr.text, doc_summary,
                char_offset=cr.char_offset, path=document.path,
            )

            chunk = Chunk(
                doc_id=document.id,
                path=document.path,
                source_type=document.source_type,
                order=order,
                total=total,
                text=cr.text,
                context=context,
                updated_at=document.updated_at,
            )
            chunk.payload['char_offset'] = cr.char_offset
            if cr.pages:
                chunk.payload['pages'] = cr.pages
            # Пропагандируем source_kind/source_name из документа в payload чанка —
            # нужно для retrieval-фильтров по kind на коллекции chunks. Без этого
            # фильтр `must_not source_kind=X` молча матчит ноль точек (см.
            # memory/chunks-payload-source-kind.md).
            for k in ('source_kind', 'source_name'):
                if k in document.payload:
                    chunk.payload[k] = document.payload[k]
            # Stamp run-versioning + fingerprint (наследуем version от документа)
            self._stamp_payload(chunk.payload, version=doc_version)
            chunks.append(chunk)
            self._status_reporter.document_chunk_done(document.id)

        # Post-chunk: разрезать чанки, содержащие большие markdown-таблицы.
        # Делаем ДО ChunkProcessors — embeddings/metadata считаются уже по
        # финальному набору sub-чанков. Перенумеровывает order/total.
        if self._max_table_rows > 0:
            from morag.indexing.chunk_splitter import split_table_chunks
            before = len(chunks)
            chunks = split_table_chunks(chunks, self._max_table_rows)
            total = len(chunks)
            if total != before:
                logger.info(
                    '%s  Table split: %d → %d chunks (max_table_rows=%d)',
                    w, before, total, self._max_table_rows,
                )

        # Дублирующее покрытие: для каждой таблицы (>= min_rows строк) добавляем
        # narrative-чанки (по 1 на строку). Используются retrieval'ом как
        # search-key + swap-to-parent. Не trogaet существующие чанки. См. ADR-0013.
        if self._narrate_tables_enabled:
            from morag.indexing.chunk_splitter import add_table_narratives
            before = len(chunks)
            chunks = add_table_narratives(chunks, self._narrate_tables_min_rows)
            added = len(chunks) - before
            if added > 0:
                logger.info(
                    '%s  Table narratives: +%d row-chunks (min_rows=%d)',
                    w, added, self._narrate_tables_min_rows,
                )
            # Stamp run-versioning + fingerprint у новых narratives
            for c in chunks[before:]:
                self._stamp_payload(c.payload, version=doc_version)

        # Применяем процессоры и сохраняем батчами.
        # process_batch — async (AsyncOpenAI / httpx.AsyncClient), параллелизм
        # при concurrency>1 достигается естественно через event loop.
        for batch_start in range(0, len(chunks), self._embed_batch_size):
            batch = chunks[batch_start:batch_start + self._embed_batch_size]
            for processor in self._chunk_processors:
                batch = await processor.process_batch(batch, document)
            for chunk in batch:
                vec_summary = ', '.join(
                    f"{k}:dense({len(v)})" if isinstance(v, list)
                    else f"{k}:sparse({len(v['indices'])})"
                    for k, v in chunk.vectors.items()
                )
                logger.info('%s    chunk %d/%d vectors: [%s]', w, chunk.order + 1, total, vec_summary)
            await self._chunk_repo.upsert_batch(batch)
            logger.info(
                '%s  Batch %d-%d/%d saved',
                w, batch_start + 1, batch_start + len(batch), len(chunks),
            )

        logger.info('%sChunks saved: %s (%d)', w, document.id, total)

        # Swap-delete: новые чанки уже в Qdrant с текущим run_number — сметаем
        # чанки прежних прогонов этого документа (replace-not-delete, ADR-0014).
        if self._run_context is not None:
            await self._chunk_repo.delete_stale_chunks(
                document.id, self._run_context.run_number,
            )

        # Точка коммита: точку документа упсертим последней, после чанков.
        # До этого момента doc point несёт прежний run_number — при крахе
        # документ для резюма считается недоделанным.
        await self._doc_repo.upsert(document)
        logger.info(
            '%sDocument committed: %s (version=%s)',
            w, document.id, document.payload.get('version'),
        )

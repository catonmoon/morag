#!/usr/bin/env python3
"""Конвертация PDF → Markdown через VisionPdfConverter.

Использование:
    python scripts/convert_pdf.py <path_to_pdf> [--output <path_to_md>]
    python scripts/convert_pdf.py <directory> --outdir <output_directory>

Настройки LLM берутся из config.yml (секция llm_vision).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from morag.config import PdfConfig, load_config
from morag.llm.client import GenerationParams, LLMClient
from morag.sources.pdf_converter import VisionPdfConverter
from morag.sources.pdf_postprocess import CodeFencePostProcessor, DeduplicatePostProcessor, PdfPostProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def _build_postprocessors(pdf_config: PdfConfig) -> list[PdfPostProcessor]:
    """Собрать цепочку постпроцессоров по конфигу."""
    processors: list[PdfPostProcessor] = []
    if pdf_config.postprocessing.strip_code_fences:
        processors.append(CodeFencePostProcessor())
    if pdf_config.postprocessing.dedup.enabled:
        processors.append(DeduplicatePostProcessor(
            threshold=pdf_config.postprocessing.dedup.threshold,
            window=pdf_config.postprocessing.dedup.window,
            min_phrase_len=pdf_config.postprocessing.dedup.min_phrase_len,
        ))
    return processors


def _build_converter(config) -> VisionPdfConverter:
    """Собрать VisionPdfConverter из конфига."""
    vision_client = LLMClient(
        base_url=config.llm_vision.base_url,
        model=config.llm_vision.model,
        api_key=config.llm_vision.api_key,
        timeout=config.llm_vision.timeout,
        max_retries=config.llm_vision.retry.max_retries,
        model_wait_seconds=config.llm_vision.model_wait_seconds,
        model_wait_retries=config.llm_vision.model_wait_retries,
    )
    pdf_config = config.pdf
    gen_params = GenerationParams(
        temperature=pdf_config.temperature if pdf_config else 0.0,
        repetition_penalty=pdf_config.repetition_penalty if pdf_config else None,
        frequency_penalty=pdf_config.frequency_penalty if pdf_config else 0.0,
        presence_penalty=pdf_config.presence_penalty if pdf_config else 0.0,
        seed=42,
        enable_thinking=config.llm_vision.enable_thinking if config.llm_vision else None,
    )
    postprocessors = _build_postprocessors(pdf_config) if pdf_config else []
    return VisionPdfConverter(
        vision_client=vision_client,
        max_tokens=pdf_config.page_max_tokens if pdf_config else 4096,
        dpi=pdf_config.dpi if pdf_config else 144,
        concurrency=pdf_config.concurrency if pdf_config else 1,
        generation_params=gen_params,
        context_tail_lines=pdf_config.context_tail_lines if pdf_config else 0,
        postprocessors=postprocessors,
    )


async def _convert_one(converter: VisionPdfConverter, pdf_path: Path, output_path: Path) -> bool:
    """Конвертировать один PDF. Возвращает True при успехе."""
    logger.info('Converting: %s (%d bytes)', pdf_path.name, pdf_path.stat().st_size)
    try:
        pdf_bytes = pdf_path.read_bytes()
        markdown = await converter.convert(pdf_bytes, pdf_path.name)
        if markdown:
            output_path.write_text(markdown, encoding='utf-8')
            logger.info('Done: %s → %s (%d chars)', pdf_path.name, output_path.name, len(markdown))
            return True
        logger.error('Failed (empty result): %s', pdf_path.name)
    except Exception as e:
        logger.error('Error converting %s: %s', pdf_path.name, e)
    return False


async def main() -> None:
    parser = argparse.ArgumentParser(description='Convert PDF to Markdown via Vision LLM')
    parser.add_argument('pdf', help='Path to PDF file or directory with PDFs')
    parser.add_argument('--output', '-o', help='Output markdown file (single file mode)')
    parser.add_argument('--outdir', '-d', help='Output directory (batch mode for directory input)')
    parser.add_argument('--suffix', '-s', help='Suffix before .md (e.g. "_v2" → file_v2.md)')
    parser.add_argument('--skip-existing', action='store_true', help='Skip already converted files')
    parser.add_argument('--config', default='config.yml', help='Path to config.yml')
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logger.error('File not found: %s', pdf_path)
        sys.exit(1)

    config = load_config(args.config)
    if not config.llm_vision:
        logger.error('llm_vision is not configured in %s', args.config)
        sys.exit(1)

    converter = _build_converter(config)

    # Batch mode: directory
    if pdf_path.is_dir():
        outdir = Path(args.outdir) if args.outdir else pdf_path
        outdir.mkdir(parents=True, exist_ok=True)
        pdfs = sorted(pdf_path.glob('*.pdf'))
        if not pdfs:
            logger.error('No PDF files found in %s', pdf_path)
            sys.exit(1)
        logger.info('Found %d PDFs in %s → %s', len(pdfs), pdf_path, outdir)
        ok, fail = 0, 0
        for i, p in enumerate(pdfs, 1):
            out = outdir / p.with_suffix('.md').name
            if args.skip_existing and out.exists():
                logger.info('[%d/%d] SKIP (exists): %s', i, len(pdfs), p.name)
                ok += 1
                continue
            logger.info('[%d/%d] Processing...', i, len(pdfs))
            if await _convert_one(converter, p, out):
                ok += 1
            else:
                fail += 1
        logger.info('Batch complete: %d ok, %d failed out of %d', ok, fail, len(pdfs))
        return

    # Single file mode
    pdf_bytes = pdf_path.read_bytes()
    logger.info('Reading %s (%d bytes)', pdf_path, len(pdf_bytes))

    markdown = await converter.convert(pdf_bytes, pdf_path.name)
    if markdown is None:
        logger.error('Conversion failed')
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    elif args.suffix:
        output_path = pdf_path.with_name(f'{pdf_path.stem}{args.suffix}.md')
    else:
        output_path = pdf_path.with_suffix('.md')
    output_path.write_text(markdown, encoding='utf-8')
    logger.info('Written %d chars to %s', len(markdown), output_path)


if __name__ == '__main__':
    asyncio.run(main())

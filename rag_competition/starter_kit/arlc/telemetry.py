"""
Helpers for telemetry calculation and normalization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time


@dataclass(frozen=True)
class TimingMetrics:
    """
    Timing metrics for answer generation
    """

    ttft_ms: int
    tpot_ms: int
    total_time_ms: int


@dataclass(frozen=True)
class UsageMetrics:
    """
    Token usage metrics
    """

    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class RetrievalRef:
    """
    Source reference with page numbers
    """

    doc_id: str
    page_numbers: list[int]


@dataclass(frozen=True)
class Telemetry:
    """
    Telemetry structure for a single answer
    """

    timing: TimingMetrics
    retrieval: list[RetrievalRef]
    usage: UsageMetrics
    model_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert telemetry into a submission.json-ready dict
        """
        return {
            "timing": {
                "ttft_ms": self.timing.ttft_ms,
                "tpot_ms": self.timing.tpot_ms,
                "total_time_ms": self.timing.total_time_ms,
            },
            "retrieval": {
                "retrieved_chunk_pages": [
                    {"doc_id": ref.doc_id, "page_numbers": ref.page_numbers} for ref in self.retrieval
                ]
            },
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
            },
            "model_name": self.model_name,
        }


class TelemetryTimer:
    """
    Timer for TTFT, TPOT, and total time
    """

    def __init__(self) -> None:
        self._start_time = time.perf_counter()
        self._token_timestamps: list[float] = []

    def mark_token(self) -> None:
        """
        Mark the arrival of a new token (including the first)
        """
        self._token_timestamps.append(time.perf_counter())

    def finish(self) -> TimingMetrics:
        """
        Finish timing and return metrics
        """
        end_time = time.perf_counter()
        total_time_ms = int((end_time - self._start_time) * 1000)
        if not self._token_timestamps:
            return TimingMetrics(ttft_ms=0, tpot_ms=0, total_time_ms=total_time_ms)

        ttft_ms = int((self._token_timestamps[0] - self._start_time) * 1000)
        if len(self._token_timestamps) < 2:
            return TimingMetrics(ttft_ms=ttft_ms, tpot_ms=0, total_time_ms=total_time_ms)

        diffs_ms = [
            (self._token_timestamps[index] - self._token_timestamps[index - 1]) * 1000
            for index in range(1, len(self._token_timestamps))
        ]
        tpot_ms = int(sum(diffs_ms) / len(diffs_ms)) if diffs_ms else 0
        return TimingMetrics(ttft_ms=ttft_ms, tpot_ms=tpot_ms, total_time_ms=total_time_ms)


def normalize_retrieved_pages(raw_refs: list[dict[str, Any] | RetrievalRef]) -> list[RetrievalRef]:
    """
    Normalize source references by merging duplicates and cleaning pages
    """
    by_doc: dict[str, set[int]] = {}
    for raw_ref in raw_refs:
        if isinstance(raw_ref, RetrievalRef):
            doc_id = raw_ref.doc_id.strip()
            pages = raw_ref.page_numbers
        else:
            doc_id = str(raw_ref.get("doc_id") or "").strip()
            pages = raw_ref.get("page_numbers", [])

        if not doc_id:
            continue
        normalized_pages = _parse_page_numbers(pages)
        if not normalized_pages:
            continue
        by_doc.setdefault(doc_id, set()).update(normalized_pages)

    return [
        RetrievalRef(doc_id=doc_id, page_numbers=sorted(page_numbers))
        for doc_id, page_numbers in sorted(by_doc.items())
    ]


def _parse_page_numbers(value: Any) -> list[int]:
    """
    Extract page numbers from int, string, or list
    """
    raw_values = value if isinstance(value, list) else [value]
    pages: set[int] = set()
    for raw_page in raw_values:
        if isinstance(raw_page, bool):
            continue
        if isinstance(raw_page, int):
            if raw_page > 0:
                pages.add(raw_page)
            continue
        if not isinstance(raw_page, str):
            continue
        for token in _tokenize_page_numbers(raw_page):
            if token.isdigit():
                page = int(token)
                if page > 0:
                    pages.add(page)
    return sorted(pages)


def _tokenize_page_numbers(raw_page: str) -> list[str]:
    """
    Split a page-number string into tokens
    """
    separators = [" ", ",", ";", "|", "_", ":", "[", "]", "(", ")"]
    buffer = raw_page
    for separator in separators:
        buffer = buffer.replace(separator, " ")
    return [token for token in buffer.split() if token]

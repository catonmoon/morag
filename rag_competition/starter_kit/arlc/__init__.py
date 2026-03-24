"""
Transparent framework for API access and submissions (AGENTIC RAG Legal Challenge).
"""

from arlc.client import EvaluationClient
from arlc.config import EnvConfig, get_config
from arlc.submission import SubmissionAnswer, SubmissionBuilder
from arlc.telemetry import (
    RetrievalRef,
    Telemetry,
    TelemetryTimer,
    TimingMetrics,
    UsageMetrics,
    normalize_retrieved_pages,
)

__all__ = [
    "EnvConfig",
    "get_config",
    "EvaluationClient",
    "SubmissionAnswer",
    "SubmissionBuilder",
    "RetrievalRef",
    "Telemetry",
    "TelemetryTimer",
    "TimingMetrics",
    "UsageMetrics",
    "normalize_retrieved_pages",
]

"""
Load settings from environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass
from dataclasses import dataclass


def _get(key: str, default: str = "") -> str:
    return (os.getenv(key) or default).strip()


@dataclass(frozen=True)
class EnvConfig:
    """Settings from env with defaults."""

    eval_api_key: str
    eval_base_url: str
    openai_api_key: str
    openrouter_api_key: str
    llm_api_base: str
    submission_path: Path
    code_archive_path: Path
    questions_path: Path
    docs_dir: str

    @classmethod
    def from_env(cls) -> "EnvConfig":
        """Load config from env. Defaults point to production platform."""
        eval_api_key = _get("EVAL_API_KEY")
        eval_base_url = _get(
            "EVAL_BASE_URL",
            "https://platform.agentic-challenge.ai/api/v1",
        )
        openai_api_key = _get("OPENAI_API_KEY")
        openrouter_api_key = _get("OPENROUTER_API_KEY")

        if openai_api_key:
            llm_api_base = _get("OPENAI_API_BASE", "https://api.openai.com/v1")
        else:
            llm_api_base = _get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

        submission_path = Path(_get("SUBMISSION_PATH", "submission.json"))
        code_archive_path = Path(_get("CODE_ARCHIVE_PATH", "code_archive.zip"))
        questions_path = Path(_get("QUESTIONS_PATH", "questions.json"))
        docs_dir = _get("DOCS_DIR", "docs_corpus")

        return cls(
            eval_api_key=eval_api_key,
            eval_base_url=eval_base_url.rstrip("/"),
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
            llm_api_base=llm_api_base,
            submission_path=submission_path,
            code_archive_path=code_archive_path,
            questions_path=questions_path,
            docs_dir=docs_dir,
        )

    def get_llm_api_key(self) -> str:
        """API key for LLM: OpenAI or OpenRouter."""
        return self.openai_api_key or self.openrouter_api_key

    def get_embedding_api_key(self) -> str:
        """API key for embeddings: OpenAI or OpenRouter."""
        return self.openai_api_key or self.openrouter_api_key


def get_config() -> EnvConfig:
    """Get current config (always fresh, no caching)."""
    return EnvConfig.from_env()

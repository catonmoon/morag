"""
Client for interacting with the competition API and resources.
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import zipfile

import requests


@dataclass(frozen=True)
class ApiConfig:
    """
    API access configuration.
    """

    api_key: str
    base_url: str


class EvaluationClient:
    """
    Thin client for API: questions, documents, submissions.
    """

    def __init__(self, api_key: str, base_url: str) -> None:
        self._config = ApiConfig(api_key=api_key, base_url=base_url.rstrip("/"))
        self._session = requests.Session()
        self._session.headers.update({"X-API-Key": self._config.api_key})

    @classmethod
    def from_env(
        cls,
        *,
        api_key_env: str = "EVAL_API_KEY",
        base_url_env: str = "EVAL_BASE_URL",
        default_base_url: str = "https://platform.agentic-challenge.ai/api/v1",
    ) -> "EvaluationClient":
        """
        Create a client from environment variables.
        """
        api_key = _read_env(api_key_env)
        base_url = _read_env(base_url_env, default=default_base_url)
        if not api_key:
            raise ValueError(f"API key is not set in {api_key_env}.")
        return cls(api_key=api_key, base_url=base_url)

    def download_questions(self, target_path: str | Path | None = None) -> list[dict[str, Any]]:
        """
        Download questions and optionally save to disk.
        """
        response = self._session.get(f"{self._config.base_url}/questions", timeout=60)
        response.raise_for_status()
        questions = response.json()
        if target_path:
            path = Path(target_path)
            path.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
        return questions

    def download_documents(self, target_dir: str | Path = "docs_corpus") -> Path:
        """
        Download the documents archive and extract it to the target directory
        """
        response = self._session.get(f"{self._config.base_url}/documents", timeout=120)
        response.raise_for_status()
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        zip_path = target_path / "documents.zip"
        zip_path.write_bytes(response.content)
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(target_path)
        return target_path

    def submit_submission(
        self,
        submission_path: str | Path,
        code_archive_path: str | Path,
    ) -> dict[str, Any]:
        """
        Submit submission.json with a required code archive and return the API response
        """
        path = Path(submission_path)
        archive_path = Path(code_archive_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"Code archive not found: {archive_path}")
        with path.open("rb") as file_handle, archive_path.open("rb") as archive_handle:
            response = self._session.post(
                f"{self._config.base_url}/submissions",
                files={
                    "file": (path.name, file_handle, "application/json"),
                    "code_archive": (archive_path.name, archive_handle, "application/zip"),
                },
                timeout=120,
            )
        response.raise_for_status()
        return response.json()

    def get_submission_status(self, submission_uuid: str) -> dict[str, Any]:
        """
        Get submission processing status and metrics.
        """
        response = self._session.get(
            f"{self._config.base_url}/submissions/{submission_uuid}/status",
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def create_code_archive(
        self,
        include_paths: list[str] | list[Path],
        archive_path: str | Path,
        root_dir: str | Path | None = None,
    ) -> Path:
        """
        Create a ZIP archive with the specified paths for submission.
        """
        archive_path = Path(archive_path)
        root = Path(root_dir) if root_dir else Path.cwd()
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for path_item in include_paths:
                path = Path(path_item)
                if not path.is_absolute():
                    path = root / path
                if not path.exists():
                    continue
                if path.is_dir():
                    for file_path in path.rglob("*"):
                        if file_path.is_file() and "__pycache__" not in file_path.parts:
                            try:
                                arcname = file_path.relative_to(root)
                            except ValueError:
                                arcname = file_path.name
                            zip_file.write(file_path, arcname)
                elif path.is_file():
                    try:
                        arcname = path.relative_to(root)
                    except ValueError:
                        arcname = path.name
                    zip_file.write(path, arcname)
        return archive_path


def _read_env(env_key: str, *, default: str | None = None) -> str:
    """
    Read an environment variable with a fallback value.
    """
    value = os.getenv(env_key, default)
    return value or ""

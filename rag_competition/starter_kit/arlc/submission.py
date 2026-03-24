"""
Build submission.json payloads for evaluation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arlc.telemetry import Telemetry


AnswerValue = str | int | float | bool | list[str] | None


@dataclass(frozen=True)
class SubmissionAnswer:
    """
    A single answer in a submission
    """

    question_id: str
    answer: AnswerValue
    telemetry: Telemetry

    def to_dict(self) -> dict[str, Any]:
        """
        Convert an answer into a JSON-ready dict
        """
        return {
            "question_id": self.question_id,
            "answer": self.answer,
            "telemetry": self.telemetry.to_dict(),
        }


@dataclass
class SubmissionBuilder:
    """
    Answer builder with convenient save logic.
    Supports context manager: with SubmissionBuilder(target_path="sub.json") as builder: ...
    On exit, save() is called automatically if target_path was provided.
    """

    architecture_summary: str | None = None
    answers: list[SubmissionAnswer] = field(default_factory=list)
    target_path: str | Path | None = None

    def __enter__(self) -> "SubmissionBuilder":
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object) -> None:
        if exc_type is None and self.target_path is not None:
            self.save(self.target_path)

    def add_answer(self, answer: SubmissionAnswer) -> None:
        """
        Add an answer to the builder
        """
        self.answers.append(answer)

    def build(self) -> dict[str, Any]:
        """
        Build the final submission.json payload
        """
        return {
            "architecture_summary": self.architecture_summary,
            "answers": [answer.to_dict() for answer in self.answers],
        }

    def save(self, target_path: str | Path) -> Path:
        """
        Save submission.json to disk
        """
        path = Path(target_path)
        payload = self.build()
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


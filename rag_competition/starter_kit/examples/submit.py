"""
Create submission.json and submit to the evaluation API (with code archive).
Unified script: creates example submission if needed, builds code archive, submits.
"""

from pathlib import Path
import sys
import zipfile

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from arlc import (
    EvaluationClient,
    RetrievalRef,
    SubmissionAnswer,
    SubmissionBuilder,
    Telemetry,
    TimingMetrics,
    UsageMetrics,
    get_config,
    normalize_retrieved_pages,
)


def create_example_submission(target_path: str | Path = "submission.json") -> Path:
    """
    Create an example submission.json with a single record.
    """
    retrieval_refs = normalize_retrieved_pages(
        [
            RetrievalRef(doc_id="document_01.pdf", page_numbers=[1, 3]),
        ]
    )
    telemetry = Telemetry(
        timing=TimingMetrics(ttft_ms=120, tpot_ms=50, total_time_ms=500),
        retrieval=retrieval_refs,
        usage=UsageMetrics(input_tokens=1024, output_tokens=256),
        model_name="gpt-4o",
    )

    builder = SubmissionBuilder(architecture_summary="Naive RAG with vector search")
    builder.add_answer(
        SubmissionAnswer(
            question_id="q1",
            answer="Your answer here",
            telemetry=telemetry,
        )
    )
    return builder.save(target_path)


def ensure_code_archive(archive_path: Path) -> Path:
    """Create code archive if it does not exist."""
    if archive_path.exists():
        return archive_path

    include_paths = [
        ROOT_DIR / "arlc",
        ROOT_DIR / "examples",
        ROOT_DIR / "README.md",
        ROOT_DIR / "API.md",
        ROOT_DIR / "EVALUATION.md",
        ROOT_DIR / "openapi.yaml",
    ]
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for path in include_paths:
            if path.is_dir():
                for file_path in path.rglob("*"):
                    if file_path.is_file() and "__pycache__" not in file_path.parts:
                        zip_file.write(file_path, file_path.relative_to(ROOT_DIR))
            elif path.is_file():
                zip_file.write(path, path.relative_to(ROOT_DIR))
    return archive_path


def create_and_submit(
    submission_path: str | Path = "submission.json",
    code_archive_path: str | Path = "code_archive.zip",
    create_if_missing: bool = True,
) -> dict:
    """
    Create submission (if missing) and submit to the API.
    """
    path = Path(submission_path)
    if create_if_missing and not path.exists():
        create_example_submission(path)
        print(f"Created {path}")

    archive_path = ensure_code_archive(Path(code_archive_path))
    print(f"Using code archive: {archive_path}")

    client = EvaluationClient.from_env()
    return client.submit_submission(path, archive_path)


if __name__ == "__main__":
    config = get_config()
    response = create_and_submit(config.submission_path, config.code_archive_path)
    print(response)

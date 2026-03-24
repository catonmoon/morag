"""
Example of telemetry calculation methodology for submission answers.

Uses TelemetryTimer for TTFT/TPOT measurement and normalize_retrieved_pages
for retrieval references. Run this to understand the exact calculation flow.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from arlc import (
    RetrievalRef,
    Telemetry,
    TelemetryTimer,
    TimingMetrics,
    UsageMetrics,
    normalize_retrieved_pages,
)


def main() -> None:
    """
    Demonstrate telemetry calculation methodology.
    """
    print("=== Telemetry calculation example ===\n")

    print("1. Timing (TTFT, TPOT, total_time_ms)")
    print("   - TelemetryTimer marks each token arrival via mark_token()")
    print("   - TTFT = time from start to first token")
    print("   - TPOT = mean time between consecutive tokens")
    print("   - total_time_ms = full generation time\n")

    timer = TelemetryTimer()
    timer.mark_token()
    timer.mark_token()
    timer.mark_token()
    timing = timer.finish()
    print(f"   Example: ttft_ms={timing.ttft_ms}, tpot_ms={timing.tpot_ms}, total_time_ms={timing.total_time_ms}\n")

    print("2. Retrieval references (page_numbers start from 1)")
    print("   - doc_id = PDF filename (SHA-like string from corpus)")
    print("   - page_numbers = list of 1-based page indices in the document\n")

    raw_refs = [
        RetrievalRef(doc_id="443e04bc1a78940b3fcd5438d24b6c5f182a276d354a3108e738b193675de032", page_numbers=[1, 2]),
        {"doc_id": "443e04bc1a78940b3fcd5438d24b6c5f182a276d354a3108e738b193675de032", "page_numbers": [2, 3]},
    ]
    merged = normalize_retrieved_pages(raw_refs)
    print(f"   normalize_retrieved_pages merges duplicates: {merged}\n")

    print("3. Assembling Telemetry")
    telemetry = Telemetry(
        timing=TimingMetrics(ttft_ms=320, tpot_ms=45, total_time_ms=1200),
        retrieval=merged,
        usage=UsageMetrics(input_tokens=512, output_tokens=128),
        model_name="gpt-4o-mini",
    )
    print(f"   to_dict(): {telemetry.to_dict()}\n")

    print("4. Malformed telemetry")
    print("   - Missing timing/usage/retrieval -> telemetry penalty (factor 0.9)")
    print("   - TTFT not provided -> max TTFT penalty (ttft_factor 0.85)")


if __name__ == "__main__":
    main()

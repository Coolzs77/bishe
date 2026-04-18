#!/usr/bin/env python3
"""Parse RV1126B runtime logs and summarize inference latency distribution."""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path
from typing import Iterable

PATTERNS = [
    re.compile(r"(?:平均推理|avg(?:erage)?\\s+inference|inference)\\s*[:：]\\s*([0-9]+(?:\\.[0-9]+)?)\\s*ms", re.IGNORECASE),
]


def extract_ms_values(lines: Iterable[str]) -> list[float]:
    values: list[float] = []
    for line in lines:
        for pattern in PATTERNS:
            match = pattern.search(line)
            if match:
                values.append(float(match.group(1)))
                break
    return values


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * (p / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]

    weight = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse inference latency from RV1126B logs")
    parser.add_argument("log_file", type=Path, help="Path to runtime log file")
    parser.add_argument("--target-ms", type=float, default=20.0, help="Target mean latency in ms")
    parser.add_argument("--show-samples", action="store_true", help="Print parsed raw samples")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.log_file.exists() or not args.log_file.is_file():
        print(f"ERROR: log file not found: {args.log_file}")
        return 1

    lines = args.log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    values = extract_ms_values(lines)

    if not values:
        print("ERROR: no inference latency values were found in the log")
        return 1

    sorted_values = sorted(values)
    mean_val = statistics.fmean(sorted_values)
    p50_val = percentile(sorted_values, 50)
    p90_val = percentile(sorted_values, 90)
    p95_val = percentile(sorted_values, 95)

    print("=== Latency Summary ===")
    print(f"samples: {len(sorted_values)}")
    print(f"mean_ms: {mean_val:.3f}")
    print(f"p50_ms: {p50_val:.3f}")
    print(f"p90_ms: {p90_val:.3f}")
    print(f"p95_ms: {p95_val:.3f}")

    if args.show_samples:
        sample_str = ", ".join(f"{value:.3f}" for value in sorted_values)
        print(f"samples_ms: [{sample_str}]")

    passed = mean_val <= args.target_ms
    print(f"target_ms: {args.target_ms:.3f}")
    print(f"result: {'PASS' if passed else 'FAIL'}")

    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

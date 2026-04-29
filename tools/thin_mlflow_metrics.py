#!/usr/bin/env python3
"""Thin out MLflow metric files using log-scale spacing.

Keeps every step for steps 0-99, then progressively thins out,
capping at 1 log per 100 steps. Always keeps the last line.

Usage:
    python scripts/thin_mlflow_metrics.py [mlruns_dir]

Default mlruns_dir is ./mlruns
"""

import sys
from pathlib import Path


def should_keep(step: int) -> bool:
    interval = min(100, max(1, step // 100))
    return step % interval == 0


def thin_file(path: Path) -> tuple[int, int]:
    lines = path.read_text().splitlines()
    if len(lines) <= 1:
        return len(lines), len(lines)

    kept = []
    for i, line in enumerate(lines):
        parts = line.split()
        step = int(parts[2])
        if should_keep(step) or i == len(lines) - 1:
            kept.append(line)

    original = len(lines)
    if len(kept) < original:
        path.write_text("\n".join(kept) + "\n")

    return original, len(kept)


def main():
    mlruns = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("mlruns")
    if not mlruns.exists():
        print(f"Directory not found: {mlruns}")
        sys.exit(1)

    total_original = 0
    total_kept = 0

    for metrics_dir in sorted(mlruns.rglob("metrics")):
        if not metrics_dir.is_dir():
            continue
        for metric_file in sorted(metrics_dir.rglob("*")):
            print(f"Processing {metric_file}...")
            if not metric_file.is_file():
                continue
            original, kept = thin_file(metric_file)
            if kept < original:
                rel = metric_file.relative_to(mlruns)
                print(f"{rel}: {original} -> {kept}")
            total_original += original
            total_kept += kept

    if total_original:
        print(f"\nTotal: {total_original} -> {total_kept} lines ({100 * total_kept / total_original:.0f}%)")
    else:
        print("No metric files found.")


if __name__ == "__main__":
    main()

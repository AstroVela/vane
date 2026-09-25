# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Summarize video actor JSONL files without model or framework dependencies."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def read_records(directory: Path):
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A force-stopped actor may leave only its last line partial.
                    if not line.endswith("\n"):
                        print(f"Ignoring truncated final record: {path}:{number}", file=sys.stderr)
                        continue
                    raise
                if record.get("schema_version") != 1:
                    raise ValueError(f"Unsupported profile schema: {path}:{number}")
                yield record


def distribution(values):
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "total_ms": sum(ordered),
        "mean_ms": statistics.mean(ordered),
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)],
    }


def summarize(records, skip_batches: int = 1):
    """Exclude the first N completed calls per actor; never infer pipeline FPS."""
    if skip_batches < 0:
        raise ValueError("skip_batches must be non-negative")
    groups = defaultdict(list)
    for record in records:
        if record["batch_index"] > skip_batches:
            groups[record["engine"]].append(record)
    output = {}
    for engine, batches in sorted(groups.items()):
        metrics = defaultdict(list)
        for batch in batches:
            for name in ("actor_body_ms", "actor_thread_cpu_ms", "actor_gap_ms", "previous_write_ms"):
                if batch.get(name) is not None:
                    metrics[name].append(batch[name])
            for name, phase in batch["phases"].items():
                for clock, value in phase.items():
                    metrics[f"phase.{name}.{clock}"].append(value)
            for name, value in (batch["model_ms"] or {}).items():
                metrics[f"model.{name}.wall_ms"].append(value)
        output[engine] = {
            "actors": len({batch["actor_id"] for batch in batches}),
            "completed_batches": len(batches),
            "inferred_frames": sum(batch["rows"] for batch in batches),
            "model_timing_batches": sum(batch["model_ms"] is not None for batch in batches),
            "skip_batches_per_actor": skip_batches,
            "metrics": {name: distribution(values) for name, values in sorted(metrics.items())},
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--skip-batches", type=int, default=1, help="Discard first N batches per actor (default: 1)")
    args = parser.parse_args()
    if args.skip_batches < 0:
        parser.error("--skip-batches must be non-negative")
    result = summarize(read_records(args.directory), args.skip_batches)
    if not result:
        parser.error("No completed batches remain; check the directory and --skip-batches")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

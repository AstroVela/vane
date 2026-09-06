#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0
"""Measure explicitly selected Python/native media execution on the same input.

Run with the installed Vane package. Input generation and extension loading
are outside the timed region. Repetitions alternate backend order after one
warmup per backend; this measures warm-cache execution, not cold remote I/O.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import time
from pathlib import Path

OPERATIONS = {
    "image_metadata": ("image", "sum((image_file_metadata(image_file(url))).width)"),
    "image_decode": ("image", "sum(octet_length((decode_image_file(image_file(url))).data))"),
    "audio_metadata": ("audio", "sum((audio_metadata(audio_file(url))).sample_rate)"),
    "audio_resample": ("audio", "sum(tensor_shape(resample(audio_file(url), 16000))[1])"),
    "video_metadata": ("video", "sum((video_metadata(video_file(url))).width)"),
    "video_frames": ("video", None),
}


def main() -> None:
    os.environ["VANE_RUNNER"] = "local-fast"
    import vane

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=OPERATIONS)
    parser.add_argument("input", type=Path)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--backend", choices=("both", "python", "native"), default="both")
    parser.add_argument("--start-time", type=float, default=0)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--allow-unsigned-development-artifact", action="store_true")
    args = parser.parse_args()
    if min(args.rows, args.repetitions, args.threads) < 1:
        parser.error("rows, repetitions, and threads must be positive")
    domain, expression = OPERATIONS[args.operation]
    source = args.input.resolve(strict=True)
    artifact = args.extension.resolve(strict=True)
    versions = {"vane": vane.__version__}
    for name in ("pillow", "soundfile", "soxr", "av"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    backends = ("python", "native") if args.backend == "both" else (args.backend,)
    timings: dict[str, list[float]] = {backend: [] for backend in backends}
    cpu_timings: dict[str, list[float]] = {backend: [] for backend in backends}
    results: dict[str, object] = {}
    with vane.connect(
        config={"allow_unsigned_extensions": str(args.allow_unsigned_development_artifact).lower()}
    ) as con:
        con.load_extension(str(artifact))
        con.execute("SET threads=?", [args.threads])
        con.execute("CREATE TEMP TABLE inputs AS SELECT ?::VARCHAR AS url FROM range(?)", [str(source), args.rows])
        engine = con.execute("PRAGMA version").fetchall()

        def run(backend: str) -> object:
            con.execute(f"SET {domain}_backend='{backend}'")
            if expression:
                return con.execute(f"SELECT {expression} FROM inputs").fetchone()
            from vane.datasource.video_reader import VideoFrameSource

            scan = VideoFrameSource(
                [str(source)] * args.rows, height=90, width=160, start_time=args.start_time, end_time=args.end_time
            )
            return con.from_datasource(scan).aggregate("count(*), sum(frame_index)").fetchone()

        for backend in timings:
            results[backend] = run(backend)
        for repetition in range(args.repetitions):
            order = backends if repetition % 2 == 0 else tuple(reversed(backends))
            for backend in order:
                usage = resource.getrusage(resource.RUSAGE_SELF)
                cpu_started = usage.ru_utime + usage.ru_stime
                started = time.perf_counter()
                results[backend] = run(backend)
                timings[backend].append(time.perf_counter() - started)
                usage = resource.getrusage(resource.RUSAGE_SELF)
                cpu_timings[backend].append(usage.ru_utime + usage.ru_stime - cpu_started)
    # Capture before hashing large inputs/artifacts. Use --backend in separate
    # processes to compare peaks; ru_maxrss cannot be reset between backends.
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "operation": args.operation,
        "input_name": source.name,
        "input_bytes": source.stat().st_size,
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "rows": args.rows,
        "threads": args.threads,
        "backends": backends,
        "repetitions": args.repetitions,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "package_versions": versions,
        "engine": engine,
        "cache": "warm; one warmup per backend; alternating measurement order",
        "seconds": timings,
        "cpu_seconds": cpu_timings,
        "process_peak_rss_bytes": peak_rss if platform.system() == "Darwin" else peak_rss * 1024,
        "peak_rss_scope": "whole process through query completion, including imports, loading, and warmups",
        "median_seconds": {key: statistics.median(value) for key, value in timings.items()},
        "results": results,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

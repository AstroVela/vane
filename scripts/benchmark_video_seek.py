#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare Python, native sequential, and native indexed video selection.

Run outside the checkout with the installed package. Index construction is
measured separately. Query measurements use one warmup and alternating order;
they do not claim cold operating-system caches. --http measures a local range
server and reports its actual response-body bytes for each query.
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
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@contextmanager
def _source(path, http):
    counters = {"bytes": 0, "requests": 0}
    if not http:
        yield str(path), counters
        return
    size = path.stat().st_size
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def serve(self, body):
            if self.path != "/video":
                self.send_error(404)
                return
            requested = self.headers.get("Range")
            start, end = 0, size - 1
            if requested:
                unit, _, limits = requested.partition("=")
                first, _, last = limits.partition("-")
                if unit != "bytes" or not first.isdigit() or (last and not last.isdigit()):
                    self.send_error(416)
                    return
                start, end = int(first), min(int(last), end) if last else end
                if start > end:
                    self.send_error(416)
                    return
            self.send_response(206 if requested else 200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("ETag", '"fixed-benchmark-input"')
            if requested:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if body:
                with path.open("rb") as source:
                    source.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        block = source.read(min(1024**2, remaining))
                        if not block:
                            raise OSError("benchmark source changed during its read")
                        with lock:
                            counters["bytes"] += len(block)
                        self.wfile.write(block)
                        remaining -= len(block)
                with lock:
                    counters["requests"] += 1

        def do_HEAD(self):
            self.serve(False)

        def do_GET(self):
            self.serve(True)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/video", counters
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--start-time", type=float, required=True)
    parser.add_argument("--end-time", type=float, required=True)
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--mode", choices=("all", "python", "sequential", "indexed"), default="all")
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--allow-unsigned-development-artifact", action="store_true")
    args = parser.parse_args()
    if min(args.height, args.width, args.threads, args.repetitions) < 1 or args.idx < 0:
        parser.error("dimensions, threads and repetitions must be positive; idx must be nonnegative")
    source, artifact = args.input.resolve(strict=True), args.extension.resolve(strict=True)
    os.environ["VANE_RUNNER"] = "local-fast"
    import vane

    timings = {}
    metrics = {}
    modes = ("python", "sequential", "indexed") if args.mode == "all" else (args.mode,)
    operations = ("index", "window", "keyframes", "stream")
    with (
        _source(source, args.http) as (url, counters),
        vane.connect(
            config={
                "allow_unsigned_extensions": str(args.allow_unsigned_development_artifact).lower(),
                "threads": args.threads,
            }
        ) as con,
    ):
        if args.mode != "python":
            con.load_extension(str(artifact))
        engine = con.execute("PRAGMA version").fetchall()
        file = vane.VideoFile(url)
        seek_index, build, info = None, None, None
        if "indexed" in modes:
            con.execute("SET video_backend='native'")
            started, cpu = time.perf_counter(), time.process_time()
            seek_index = con.execute("SELECT build_video_index($1)", [file]).fetchone()[0]
            build = {
                "seconds": time.perf_counter() - started,
                "cpu_seconds": time.process_time() - cpu,
                "http_bytes": counters["bytes"] if args.http else None,
            }
            info = con.execute("SELECT video_index_info($1)", [seek_index]).fetchone()[0]

        def run(mode, operation):
            con.execute("SET video_backend=?", ["python" if mode == "python" else "native"])
            index = seek_index if mode == "indexed" else None
            if operation == "index":
                return con.execute(
                    "SELECT octet_length((get_video_frame_by_idx($1, $2, index => $3)).data)", [file, args.idx, index]
                ).fetchone()
            if operation == "stream":
                return (
                    vane.read_video_frames(
                        file,
                        args.height,
                        args.width,
                        start_time=args.start_time,
                        end_time=args.end_time,
                        sample_interval_seconds=args.interval,
                        indexes=[index] if index else None,
                        connection=con,
                    )
                    .aggregate("count(*), sum(octet_length(data.data))")
                    .fetchone()
                )
            function = "video_frames" if operation == "window" else "video_keyframes"
            payload = "frame.data.data" if operation == "window" else "frame.data"
            return con.execute(
                f"SELECT count(*), sum(octet_length({payload})) FROM (SELECT unnest({function}("
                "$1, start_time => $2, end_time => $3, sample_interval_seconds => $4, "
                "height => $5, width => $6, index => $7)) AS frame)",
                [file, args.start_time, args.end_time, args.interval, args.height, args.width, index],
            ).fetchone()

        for operation in operations:
            for mode in modes:
                run(mode, operation)
                timings[f"{operation}/{mode}"] = []
            for repetition in range(args.repetitions):
                for mode in modes if repetition % 2 == 0 else tuple(reversed(modes)):
                    before = counters["bytes"]
                    started, cpu = time.perf_counter(), time.process_time()
                    result = run(mode, operation)
                    timings[f"{operation}/{mode}"].append(
                        {
                            "seconds": time.perf_counter() - started,
                            "cpu_seconds": time.process_time() - cpu,
                            "http_bytes": counters["bytes"] - before if args.http else None,
                            "result": result,
                        }
                    )
            for mode in modes:
                if mode == "python":
                    continue
                con.execute("SET video_backend='native'")
                index = seek_index if mode == "indexed" else None
                options = (
                    {"idx": args.idx}
                    if operation == "index"
                    else {
                        "start_time": args.start_time,
                        "end_time": args.end_time,
                        "sample_interval_seconds": args.interval,
                        "is_key_frame": True if operation == "keyframes" else None,
                    }
                )
                metrics[f"{operation}/{mode}"] = (
                    con.sql("SELECT $1 AS file", params=[file])
                    .select(vane.video_scan_stats(vane.col("file"), index=index, **options))
                    .fetchone()[0]
                )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "engine": engine,
        "vane": vane.__version__,
        "pyav": _version("av"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "threads": args.threads,
        "repetitions": args.repetitions,
        "modes": modes,
        "input_bytes": source.stat().st_size,
        "input_sha256": _digest(source),
        "artifact_sha256": _digest(artifact),
        "selection": {
            name: getattr(args, name) for name in ("start_time", "end_time", "idx", "interval", "height", "width")
        },
        "storage": "loopback HTTP range server" if args.http else "local file",
        "cache": "one warmup per mode/operation; alternating order; OS cache uncontrolled",
        "index_build": build,
        "index_info": info,
        "measurements": timings,
        "native_selection_metrics": metrics,
        "metrics_scope": "separate repeat of the same native cursor; excludes output pixel conversion",
        "median_seconds": {key: statistics.median(row["seconds"] for row in rows) for key, rows in timings.items()},
        "process_peak_rss_bytes": peak if platform.system() == "Darwin" else peak * 1024,
        "peak_scope": "whole process, including selected-mode imports, index construction when used, and warmups",
        "temporary_video_materialization_bytes": 0,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

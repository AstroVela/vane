#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0
"""Measure installed Python/native media operators with declared cost boundaries.

Queries alternate backend order after warmup. HTTP runs use a separate local
server process. Diagnostics run separately, after timing and peak-RSS capture.
Use separate --backend processes for RSS comparisons; caches are not cleared.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import platform
import resource
import statistics
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

OPERATIONS = {
    "image_metadata": "image",
    "image_decode": "image",
    "audio_metadata": "audio",
    "audio_resample": "audio",
    "video_metadata": "video",
    "video_frames": "video",
}


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _http_server(paths, delay, counters, control):
    class Handler(BaseHTTPRequestHandler):
        def serve(self, body):
            path = paths.get(self.path)
            if path is None:
                self.send_error(404)
                return
            size = path.stat().st_size
            start, end = 0, size - 1
            requested = self.headers.get("Range")
            if requested:
                unit, equal, bounds = requested.partition("=")
                first, dash, last = bounds.partition("-")
                if unit != "bytes" or not equal or not dash or not first.isdigit() or (last and not last.isdigit()):
                    self.send_error(416)
                    return
                start, end = int(first), min(int(last), end) if last else end
                if start > end:
                    self.send_error(416)
                    return
            with counters.get_lock():
                counters[1 if body else 2] += 1
                counters[3] += 1
            try:
                if delay:
                    time.sleep(delay)
                self.send_response(206 if requested else 200)
                self.send_header("Content-Length", str(end - start + 1))
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("ETag", '"immutable-benchmark-input"')
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
                            self.wfile.write(block)
                            with counters.get_lock():
                                counters[0] += len(block)
                            remaining -= len(block)
            except OSError:
                with counters.get_lock():
                    counters[4] += 1
            finally:
                with counters.get_lock():
                    counters[3] -= 1

        def do_HEAD(self):
            self.serve(False)

        def do_GET(self):
            self.serve(True)

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        control.send(server.server_port)
        try:
            control.recv()
        finally:
            server.shutdown()
            thread.join(timeout=5)
            control.close()


@contextmanager
def _sources(paths, transport, delay):
    if transport == "local":
        yield [str(path) for path in paths], lambda: None
        return
    context = multiprocessing.get_context("spawn")
    counters = context.Array("Q", 5)
    parent, child = context.Pipe()
    routes = {f"/input/{index}": path for index, path in enumerate(paths)}
    process = context.Process(target=_http_server, args=(routes, delay, counters, child))
    process.start()
    child.close()

    def snapshot():
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with counters.get_lock():
                values = list(counters)
            if not values[3]:
                return dict(zip(("body_bytes", "get_requests", "head_requests", "active_requests", "errors"), values))
            time.sleep(0.001)
        raise TimeoutError("HTTP responses did not finish after query completion")

    try:
        if not parent.poll(15):
            raise TimeoutError("benchmark HTTP server did not start")
        port = parent.recv()
        yield [f"http://127.0.0.1:{port}{route}" for route in routes], snapshot
    finally:
        try:
            if process.is_alive():
                parent.send("stop")
        except (BrokenPipeError, EOFError, OSError):
            pass
        parent.close()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


@contextmanager
def _temporary_writes():
    """Count Python TemporaryFile spool writes in a separate diagnostic pass."""
    original = tempfile.TemporaryFile
    counters = {"files": 0, "written_bytes": 0, "peak_live_bytes": 0}
    live_bytes = 0
    lock = threading.Lock()

    class TrackedFile:
        def __init__(self, file):
            self.file = file
            self.size = 0
            self.closed = False
            with lock:
                counters["files"] += 1

        def __getattr__(self, name):
            return getattr(self.file, name)

        def write(self, value):
            nonlocal live_bytes
            written = self.file.write(value)
            size = max(self.size, self.file.tell())
            with lock:
                live_bytes += size - self.size
                self.size = size
                counters["written_bytes"] += written
                counters["peak_live_bytes"] = max(counters["peak_live_bytes"], live_bytes)
            return written

        def close(self):
            nonlocal live_bytes
            if not self.closed:
                self.file.close()
                with lock:
                    live_bytes -= self.size
                self.closed = True

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    tempfile.TemporaryFile = lambda *args, **kwargs: TrackedFile(original(*args, **kwargs))
    try:
        yield counters
    finally:
        tempfile.TemporaryFile = original
        counters["live_bytes_at_end"] = live_bytes


def _file_expression(args):
    position = "NULL" if args.position is None else args.position
    size = "NULL" if args.size is None else args.size
    return f"{OPERATIONS[args.operation]}_file(file(url, NULL, {position}, {size}, NULL))"


def _expression(args):
    value = _file_expression(args)
    expressions = {
        "image_metadata": f"sum((image_file_metadata({value})).width)",
        "image_decode": f"sum(octet_length((decode_image_file({value}, '{args.image_mode}')).data))",
        "audio_metadata": f"sum((audio_metadata({value})).sample_rate)",
        "audio_resample": f"sum((audio_resample({value}, {args.sample_rate})).frames)",
        "video_metadata": f"sum((video_metadata({value})).width)",
    }
    return expressions.get(args.operation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=OPERATIONS)
    parser.add_argument("input", type=Path, nargs="+")
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--installed-provider", action="store_true")
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--backend", choices=("both", "python", "native"), default="both")
    parser.add_argument("--runner", choices=("local", "ray"), default="local")
    parser.add_argument("--ray-cpus", type=int, default=4)
    parser.add_argument("--transport", choices=("local", "http"), default="local")
    parser.add_argument("--http-delay-ms", type=float, default=0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--image-mode", choices=("L", "LA", "RGB", "RGBA"), default="RGB")
    parser.add_argument("--position", type=int)
    parser.add_argument("--size", type=int)
    parser.add_argument("--start-time", type=float, default=0)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--allow-unsigned-development-artifact", action="store_true")
    args = parser.parse_args()
    if min(args.rows, args.repetitions, args.threads, args.concurrency, args.ray_cpus, args.sample_rate) < 1:
        parser.error("rows, repetitions, threads, concurrency, CPUs, and sample rate must be positive")
    if args.http_delay_ms < 0 or not args.http_delay_ms < float("inf"):
        parser.error("HTTP delay must be finite and nonnegative")
    if (args.position is None) != (args.size is None) or (
        args.position is not None and min(args.position, args.size) < 0
    ):
        parser.error("position and size must be supplied together and be nonnegative")
    if args.backend != "python" and not (args.extension or args.installed_provider):
        parser.error("native execution requires --extension or --installed-provider")
    if args.extension and args.installed_provider:
        parser.error("choose an artifact or an installed provider")
    if args.runner == "ray" and args.backend != "python" and not args.installed_provider:
        parser.error("Ray native execution requires signed installed providers")
    if args.runner == "ray" and args.diagnostics:
        parser.error("spool and phase diagnostics are local; run Ray throughput separately")
    paths = [path.resolve(strict=True) for path in args.input]
    artifact = args.extension.resolve(strict=True) if args.extension else None
    identities = [{"name": path.name, "bytes": path.stat().st_size, "sha256": _digest(path)} for path in paths]
    domain = OPERATIONS[args.operation]
    backends = ("python", "native") if args.backend == "both" else (args.backend,)
    os.environ["VANE_RUNNER"] = "local-fast"
    timings = {backend: [] for backend in backends}
    diagnostics = {}
    descriptors = []
    with ExitStack() as stack:
        urls, http = stack.enter_context(_sources(paths, args.transport, args.http_delay_ms / 1000))
        if args.runner == "ray":
            # Ray Workers do not inherit the driver's -I flag. Start them away
            # from the source checkout so they import the installed package.
            original_directory = Path.cwd()
            run_directory = stack.enter_context(tempfile.TemporaryDirectory(prefix="vane-media-benchmark-"))
            os.chdir(run_directory)
            stack.callback(os.chdir, original_directory)
        import vane

        if args.runner == "ray":
            import ray

            ray.init(address="local", num_cpus=args.ray_cpus, include_dashboard=False, log_to_driver=False)
            stack.callback(ray.shutdown)
        groups = {}
        engine = None
        for backend in backends:
            group = []
            for _ in range(args.concurrency):
                con = stack.enter_context(
                    vane.connect(
                        config={
                            "allow_unsigned_extensions": str(args.allow_unsigned_development_artifact).lower(),
                            "threads": args.threads,
                        }
                    )
                )
                if backend == "native":
                    if args.installed_provider:
                        vane.load_installed_extension(domain, connection=con)
                        from vane.extensions import _capture_dynamic_extension_snapshot

                        captured = _capture_dynamic_extension_snapshot(con)
                        if descriptors and descriptors != captured:
                            raise RuntimeError("connections loaded different extension identities")
                        descriptors = captured
                    else:
                        con.load_extension(str(artifact))
                con.execute(f"SET {domain}_backend=?", [backend])
                engine = con.execute("PRAGMA version").fetchall()
                inputs = f"(SELECT ({vane.ConstantExpression(urls)})[(range % {len(urls)})+1] AS url FROM range({args.rows})) inputs"
                expression = _expression(args)
                query = f"SELECT {expression} FROM {inputs}" if expression else None
                runner = None
                if args.runner == "ray":
                    from vane.runners.ray.runner import RayRunner

                    runner = RayRunner(address=None, max_task_backlog=None)
                    stack.callback(runner.close)
                group.append((con, query, inputs, runner))
            groups[backend] = group
        # Drain query threads before closing the connections and Ray runners.
        executor = stack.enter_context(ThreadPoolExecutor(max_workers=args.concurrency))

        def run(item):
            con, query, _, runner = item
            if query:
                relation = con.sql(query)
            else:
                files = [
                    vane.VideoFile(urls[i % len(urls)], position=args.position, size=args.size)
                    for i in range(args.rows)
                ]
                relation = vane.read_video_frames(
                    files, 90, 160, start_time=args.start_time, end_time=args.end_time, connection=con
                ).aggregate("count(*), sum(frame_index)")
            if runner is None:
                return relation.fetchone()
            import pyarrow as pa

            parts = list(runner.run_iter_tables(relation))
            if not parts:
                raise RuntimeError("Ray benchmark aggregate returned no partitions")
            table = pa.concat_tables([part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts])
            if table.num_rows != 1:
                raise RuntimeError("benchmark aggregate did not return one row")
            values = [column[0].as_py() for column in table.columns]
            # Arrow represents HUGEINT count/sum results as Decimal. Preserve
            # their exact integer values in the JSON report.
            for index, value in enumerate(values):
                if isinstance(value, Decimal):
                    if not value.is_finite() or value != value.to_integral_value():
                        raise RuntimeError("benchmark aggregate returned a non-integral count")
                    values[index] = int(value)
            return tuple(values)

        def run_group(backend):
            try:
                return list(executor.map(run, groups[backend]))
            except BaseException:
                for con, _, _, _ in groups[backend]:
                    con.interrupt()
                raise

        results = {backend: run_group(backend) for backend in backends}
        for repetition in range(args.repetitions):
            for backend in backends if repetition % 2 == 0 else tuple(reversed(backends)):
                before = http()
                started, cpu = time.perf_counter(), time.process_time()
                result = run_group(backend)
                elapsed, cpu_elapsed = time.perf_counter() - started, time.process_time() - cpu
                after = http()
                if result != results[backend]:
                    raise RuntimeError("benchmark results changed between repetitions")
                traffic = None if before is None else {key: after[key] - before[key] for key in before}
                if traffic and traffic["errors"]:
                    raise RuntimeError("benchmark HTTP response failed")
                timings[backend].append(
                    {"seconds": elapsed, "cpu_seconds": cpu_elapsed, "http": traffic, "results": result}
                )

        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if args.diagnostics:
            for backend in backends:
                with _temporary_writes() as temporary:
                    result = run_group(backend)
                if result != results[backend]:
                    raise RuntimeError("spool instrumentation changed the benchmark result")
                if temporary["live_bytes_at_end"]:
                    raise RuntimeError("a Python temporary spool remained open after execution")
                diagnostics[backend] = {"python_temporary_files": temporary}
                if backend == "native" and args.operation == "audio_resample":
                    profiles = []
                    for con, _, inputs, _ in groups[backend]:
                        profiles.extend(
                            row[0]
                            for row in con.execute(
                                f"SELECT native_audio_resample_profile({_file_expression(args)}, {args.sample_rate}) FROM {inputs}"
                            ).fetchall()
                        )
                    diagnostics[backend]["audio_profiles"] = profiles
    versions = {"vane": vane.__version__}
    for name in ("numpy", "pillow", "soundfile", "soxr", "av", "ray"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {
        "operation": args.operation,
        "inputs": identities,
        "artifact_sha256": _digest(artifact) if artifact and args.backend != "python" else None,
        "installed_provider": args.installed_provider,
        "extension_descriptors": descriptors,
        "rows_per_query": args.rows,
        "threads_per_connection": args.threads,
        "concurrency": args.concurrency,
        "runner": args.runner,
        "ray_cpus": args.ray_cpus if args.runner == "ray" else None,
        "backends": backends,
        "repetitions": args.repetitions,
        "transport": args.transport,
        "http_delay_ms": args.http_delay_ms,
        "position": args.position,
        "size": args.size,
        "sample_rate": args.sample_rate if domain == "audio" else None,
        "image_mode": args.image_mode if args.operation == "image_decode" else None,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "package_versions": versions,
        "engine": engine,
        "cache": "uncontrolled OS cache; one warmup per backend; alternating query groups",
        "samples": timings,
        "median_seconds": {
            key: statistics.median(sample["seconds"] for sample in value) for key, value in timings.items()
        },
        "process_peak_rss_bytes": peak_rss if platform.system() == "Darwin" else peak_rss * 1024,
        "process_metrics_scope": "driver only; excludes HTTP server and Ray workers; includes imports, loading, warmups and timed queries; excludes subsequent diagnostics",
        "results": results,
        "aggregate_results_match": all(value == next(iter(results.values())) for value in results.values()),
        "diagnostics": diagnostics,
        "diagnostic_scope": "separate untimed executions; Python TemporaryFile writes exclude native filesystem writes and engine/Ray spill; audio profiles allocate real waveforms and retain the batch limits",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

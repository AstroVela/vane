#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0
"""Measure installed local, Ray pipelined and FTE execution with replayable inputs.

Run with python -I against a non-editable installation. Correctness and resource
diagnostics run outside latency samples. Cold means a new session/worker pool,
not a cold process, filesystem cache or Ray cluster. See EXECUTION_BENCHMARKS.md.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import secrets
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import vane

MODES = ("local", "pipelined", "fte")
PROFILES = ("default", "compact")
SCENARIOS = ("cold", "warm", "slow", "mixed", "recovery")
INTERFACES = ("runtime", "flight")


@dataclass(frozen=True)
class Configuration:
    output: Path
    rows: int = 100_000
    seed: int = 970
    repetitions: int = 3
    warmups: int = 1
    worker_count: int = 2
    worker_threads: int = 1
    partitions: int = 2
    batch_rows: int = 2048
    consumer_rows_per_second: float = 50_000
    deadline: float = 120
    modes: tuple[str, ...] = MODES
    profiles: tuple[str, ...] = PROFILES
    scenarios: tuple[str, ...] = SCENARIOS
    interface: str = "runtime"

    def __post_init__(self):
        if self.interface not in INTERFACES:
            raise ValueError(f"interface must be one of {INTERFACES}")
        for name in ("rows", "repetitions", "warmups", "worker_count", "worker_threads", "partitions", "batch_rows"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_rows > 2048:
            raise ValueError("batch_rows must be at most 2048")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2**31)")
        for name in ("consumer_rows_per_second", "deadline"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, allowed in (("modes", MODES), ("profiles", PROFILES), ("scenarios", SCENARIOS)):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values) or set(values) - set(allowed):
                raise ValueError(f"{name} must contain distinct choices from {allowed}")
        if "mixed" in self.scenarios and not {"pipelined", "fte"} <= set(self.modes):
            raise ValueError("mixed requires pipelined and fte modes")
        if "recovery" in self.scenarios and "fte" not in self.modes:
            raise ValueError("recovery requires fte mode")


@dataclass(frozen=True)
class Workload:
    name: str
    sql: str
    # Sorting is used only in the independent correctness pass, outside timing.
    keys: tuple[str, ...]
    ordered: bool = False
    streaming: bool = False


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def dataset(config):
    """Four deterministic files; bounded generation memory even for large runs."""
    import numpy as np

    directory = config.output / "inputs"
    directory.mkdir()
    schema = pa.schema([("id", pa.int64()), ("k", pa.int64()), ("v", pa.int64()), ("payload", pa.string())])
    paths = []
    for part in range(4):
        path = directory / f"part-{part}.parquet"
        start, end = config.rows * part // 4, config.rows * (part + 1) // 4
        with pq.ParquetWriter(path, schema, compression="zstd", use_dictionary=False) as writer:
            for offset in range(start, end, 65_536):
                ids = np.arange(offset, min(end, offset + 65_536), dtype=np.int64)
                table = pa.Table.from_arrays(
                    [
                        pa.array(ids),
                        pa.array((ids * 17 + config.seed) % 128),
                        pa.array((ids * 7919 + config.seed) % 10000 - 5000, mask=ids % 17 == 0),
                        pa.array([f"{int(i):016x}" for i in ids]),
                    ],
                    schema=schema,
                )
                writer.write_table(table, row_group_size=65_536)
        paths.append(path)
    files = [{"name": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in paths]
    write_json(directory / "manifest.json", {"generator": 1, "seed": config.seed, "rows": config.rows, "files": files})
    references = ",".join("'" + str(p).replace("'", "''") + "'" for p in paths)
    source = f"read_parquet([{references}])"
    workloads = [
        Workload("tiny", "select 42::bigint answer", ("answer",)),
        Workload("scan", f"select id,k,v,payload from {source} where id%7<>0", ("id",), streaming=True),
        Workload("aggregate", f"select k,count(*) n,sum(v)::bigint s from {source} group by k", ("k",)),
        Workload(
            "join_topn",
            f"select a.id,a.v,b.range k from {source} a join range(128) b on a.k=b.range "
            "where a.v>0 order by a.v desc,a.id limit 50",
            ("v", "id"),
            ordered=True,
        ),
    ]
    for workload in workloads:
        (directory / f"{workload.name}.sql").write_text(workload.sql + ";\n", encoding="utf-8")
    return workloads, files


def resources(config, profile):
    from vane.execution.direct_exchange import DirectExchangeLimits

    if profile == "local":
        return vane.QueryResources()
    limits = vane.RayResources(
        worker_count=config.worker_count,
        cpus_per_worker=config.worker_threads,
        partitions=config.partitions,
        exchange_stores=(vane.ExchangeStore("benchmark", str((config.output / "stores" / profile).resolve())),),
    )
    if profile == "compact":
        limits = replace(limits, exchange=DirectExchangeLimits(64 << 10, 16 << 10, 256, 4))
    return limits


def options(config, mode):
    target = (
        vane.LocalExecution()
        if mode == "local"
        else vane.RayExecution(mode, vane.FteOptions("benchmark", 3, 0.1) if mode == "fte" else None)
    )
    return vane.QueryExecutionOptions(target, config.deadline, config.deadline, config.deadline)


@contextmanager
def connect(config, profile):
    # Match the aggregate CPU and admitted operator-memory allowance of one Ray
    # query. Session delivery limits stay at their production defaults.
    limits = resources(config, profile)
    ray_limits = vane.RayResources()
    memory = config.worker_count * (ray_limits.operator_memory_bytes // ray_limits.max_active_queries)
    settings = {"threads": config.worker_count * config.worker_threads, "memory_limit": f"{memory}B"}
    if profile == "local":
        with vane.connect(backend="local", resources=limits, config=settings) as connection:
            yield connection
    elif config.interface == "flight":
        from vane.server import Server

        token = secrets.token_urlsafe(48)
        with (
            Server(
                token=token,
                port=0,
                resources=limits,
                config=settings,
            ) as server,
            FlightConnection(server, token) as connection,
        ):
            yield connection
    else:
        with vane.Runtime(limits) as application, application.connect(config=settings) as connection:
            yield connection


class FlightConnection:
    """Benchmark adapter; SQL and every result batch use the public client.

    The colocated server is observed only for identities, fault injection and
    post-sample accounting. Each cursor creates another remote session on the
    same service, so mixed-mode samples exercise cross-session sharing.
    """

    def __init__(self, server, token):
        from vane.client import Client

        self.server, self.token = server, token
        self.client = Client(server.location, token=token)
        self.session = server.service._sessions[self.client.identity["session_id"]]
        self.query_runtime = self.session.owner
        self.queries = {}

    def query(self, *args, **kwargs):
        result = self.client.query(*args, **kwargs)
        with self.server.service._condition:
            self.queries[result.query_id] = self.session.queries[int(result.query_id)]
        return result

    def execution_query_id(self, result):
        # Client sequence numbers are session-local and are not worker IDs.
        return self.queries[result.query_id].context.query_id

    def cursor(self):
        return FlightConnection(self.server, self.token)

    def interrupt(self):
        from vane.execution.server_session import SessionError

        with self.client._lock:
            queries = tuple(self.client._queries.values())
        for query in queries:
            try:
                query.cancel()
            except SessionError as error:
                # The reader can close a handle after the snapshot above.
                if error.code != "QUERY_RETIRED":
                    raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.client.close()


def close_result(result):
    from vane.execution.result_delivery import _ResultCleanupPending

    deadline = time.monotonic() + 10
    while True:
        try:
            result.close()
            return
        except _ResultCleanupPending:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def idle(connection):
    """Observe all ledgers after timing, including the workers' own reservations."""
    runtime = connection.query_runtime
    state = runtime.resource_snapshot()
    if state["queries"] or any(state["request_admission"][n] for n in ("active_requests", "queued_requests")):
        raise AssertionError(f"queries did not retire: {state}")
    if any(
        state["result_delivery"][n] for n in ("active_results", "usage_bytes", "buffers", "cleanup_pending_results")
    ):
        raise AssertionError(f"results did not retire: {state}")
    if runtime.backend == "ray":
        import ray

        workers = runtime.pool.admission.snapshot()
        if workers["reservations"] or workers["waiting"]:
            raise AssertionError(f"worker admission did not retire: {workers}")
        for worker in runtime.pool.workers:
            if ray.get(worker.resources_snapshot.remote(), timeout=5)["reservations"]:
                raise AssertionError("worker retained a query reservation")
        for store in runtime.stores.values():
            if store.snapshot() != {"queries": 0, "reserved_bytes": 0}:
                raise AssertionError("store retained a query reservation")
    if isinstance(connection, FlightConnection):
        delivery = connection.client.resource_snapshot()
        if any(delivery[n] for n in ("active_results", "usage_bytes", "buffers", "cleanup_pending_results")):
            raise AssertionError(f"client results did not retire: {delivery}")
        if connection.server.service.snapshot()["queries"] or connection.server._gateway.active_links:
            raise AssertionError("server retained query handles or gateway capabilities")
        service = connection.server.service.runtime.resource_snapshot()["service"]
        if service["result_service"]["active_contexts"] or service["result_delivery"]["active_results"]:
            raise AssertionError(f"server retained results: {service}")
        connection.queries.clear()
    return state


def compare(workload, expected, actual):
    actual.validate(full=True)
    if not actual.schema.equals(expected.schema):
        raise AssertionError(f"{workload.name}: schema differs: {actual.schema} != {expected.schema}")
    if not workload.ordered:
        keys = [(key, "ascending") for key in workload.keys]
        expected, actual = (table.take(pc.sort_indices(table, sort_keys=keys)) for table in (expected, actual))
    if not actual.equals(expected):
        raise AssertionError(f"{workload.name}: values, order or duplicate counts differ")


def consumer_pause(result, seconds, deadline, stop):
    """Retain the batch without hiding a failed query or delaying mixed teardown."""
    from vane.execution.result_delivery import ResultDeliveryTimeout

    until = time.monotonic() + seconds
    while True:
        result.check_preparation()
        if stop.is_set():
            raise InterruptedError("benchmark consumption canceled")
        now = time.monotonic()
        if now >= deadline:
            raise ResultDeliveryTimeout("benchmark result delivery deadline exceeded")
        if result.context is not None:
            result.context.check()
        remaining = until - time.monotonic()
        if remaining <= 0:
            return
        stop.wait(min(remaining, max(0, deadline - time.monotonic()), 0.05))


def measure(connection, workload, config, mode, expected, *, paced=False, ready=None, stop=None):
    """Time public query/read/close; no diagnostic RPCs or full-table comparison."""
    result, batch = None, None
    captured = []
    stop = threading.Event() if stop is None else stop
    start = time.perf_counter()
    try:
        result = connection.query(workload.sql, options=options(config, mode), rows_per_batch=config.batch_rows)
        returned = time.perf_counter()
        delivery_deadline = time.monotonic() + config.deadline
        first = None
        rows = byte_count = batches = 0
        pause_seconds = 0.0
        while True:
            try:
                batch = result.read_batch()
            except StopIteration:
                break
            if first is None:
                first = time.perf_counter()
                if ready is not None:
                    ready.set()
            if not batch.schema.equals(expected.schema):
                raise AssertionError("timed result schema differs from the checked reference")
            rows += batch.num_rows
            byte_count += batch.nbytes
            batches += 1
            # Small aggregate/TopN outputs are copied to Python so failure runs
            # can be checked exactly. Streaming scans only count rows/bytes;
            # their full values are checked in the independent validation pass.
            if not workload.streaming:
                captured.extend(batch.to_pylist())
            if paced:
                before_pause = time.perf_counter()
                consumer_pause(result, batch.num_rows / config.consumer_rows_per_second, delivery_deadline, stop)
                pause_seconds += time.perf_counter() - before_pause
            batch = None
        drained = time.perf_counter()
        close_result(result)
        ended = time.perf_counter()
        if rows != expected.num_rows:
            raise AssertionError(f"timed row count differs: {rows} != {expected.num_rows}")
        if not workload.streaming:
            compare(workload, expected, pa.Table.from_pylist(captured, schema=expected.schema))
        if result.completion_status not in {"ok", "empty"} or (
            result.context is not None and result.execution_state != "SUCCEEDED"
        ):
            raise AssertionError(f"query did not succeed: {result.execution_state}")
        seconds = ended - start
        return {
            "query_id": connection.execution_query_id(result)
            if isinstance(connection, FlightConnection)
            else result.query_id,
            "started_at_monotonic": start,
            "query_return_seconds": returned - start,
            "first_batch_seconds": first - start if first is not None else None,
            "drain_seconds": drained - start,
            "close_seconds": ended - drained,
            "total_seconds": seconds,
            "rows": rows,
            "arrow_bytes": byte_count,
            "batches": batches,
            "output_rows_per_second": rows / seconds,
            "output_arrow_bytes_per_second": byte_count / seconds,
            "consumer_pause_seconds": pause_seconds,
            "requested_consumer_pause_seconds": rows / config.consumer_rows_per_second if paced else 0,
            "delivery_timing": result.timing_snapshot(),
        }, result
    finally:
        batch = None
        if result is not None:
            close_result(result)


@contextmanager
def worker_occupancy(connection, intervals):
    """Observe charged worker capacity, not query submission or queue wait.

    Both hooks run under the admission condition, so another thread cannot
    release or reacquire a token between its ledger change and the timestamp.
    No worker RPCs or polling are added to the measured queries.
    """
    manager = connection.query_runtime.pool.admission
    acquire, release = manager.try_acquire, manager.release
    active = {}

    def observed_acquire(token, query, demands):
        with manager.condition:
            admitted = acquire(token, query, demands)
            if admitted and token not in active:
                value = {
                    "token": token,
                    "query_id": query,
                    "workers": {index: dict(demand) for index, demand in demands.items()},
                    "acquired_at_monotonic": time.perf_counter(),
                    "released_at_monotonic": None,
                }
                active[token] = value
                intervals.append(value)
            return admitted

    def observed_release(token):
        with manager.condition:
            if token in active:
                active.pop(token)["released_at_monotonic"] = time.perf_counter()
            release(token)

    with patch.object(manager, "try_acquire", observed_acquire), patch.object(manager, "release", observed_release):
        yield


def mixed_overlap(intervals, pipeline_query_id, fte_query_id):
    """Longest proven overlap of reservations on at least one shared worker."""
    by_query = [[v for v in intervals if v["query_id"] == query] for query in (pipeline_query_id, fte_query_id)]
    if any(not values for values in by_query) or any(v["released_at_monotonic"] is None for v in intervals):
        raise AssertionError("mixed worker reservation evidence is incomplete")
    overlap = max(
        (
            min(left["released_at_monotonic"], right["released_at_monotonic"])
            - max(left["acquired_at_monotonic"], right["acquired_at_monotonic"])
            for left in by_query[0]
            for right in by_query[1]
            if left["workers"].keys() & right["workers"].keys()
        ),
        default=0,
    )
    return max(0, overlap)


@contextmanager
def worker_loss(connection):
    """One real actor death at a declared pre-commit downstream attempt boundary."""
    import ray

    from vane.execution.recovery_runtime import RecoveryScheduler

    original = RecoveryScheduler._dispatch
    fault = {}

    def dispatch(owner, index, partition, binding, upstream):
        admitted = original(owner, index, partition, binding, upstream)
        if admitted and upstream and owner.pool is connection.query_runtime.pool and not fault:
            attempt = owner.active[index]
            fault.update(
                injected_at_monotonic=time.perf_counter(),
                task_id=attempt.reservation.token.task_id,
                scheduler=owner,
            )
            ray.kill(attempt.worker, no_restart=True)
        return admitted

    with patch.object(RecoveryScheduler, "_dispatch", dispatch):
        yield fault


def recovery_details(fault, sample):
    if not fault:
        raise AssertionError("the requested worker loss did not happen")
    attempts = [t for t in fault["scheduler"].history if t.task_id == fault["task_id"]]
    if len(attempts) != 2 or attempts[0].input_id != attempts[1].input_id or attempts[0].fence == attempts[1].fence:
        raise AssertionError("recovery did not retry fixed inputs with a fresh fence")
    since_start = fault["injected_at_monotonic"] - sample["started_at_monotonic"]
    return {
        "fault_kind": "kill_first_dispatched_downstream_worker_before_commit",
        "fault_after_seconds": since_start,
        "fault_to_completion_seconds": sample["total_seconds"] - since_start,
        "attempt_count": len(attempts),
        "same_input_id": True,
        "distinct_fences": True,
    }


def distribution(values):
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "max": ordered[-1],
    }


def summarize(samples):
    groups = defaultdict(list)
    for sample in samples:
        if sample["phase"] != "warmup":
            groups[(sample["profile"], sample["mode"], sample["phase"], sample["workload"])].append(sample)
    metrics = (
        "session_open_seconds",
        "session_first_query_seconds",
        "query_return_seconds",
        "first_batch_seconds",
        "total_seconds",
        "output_rows_per_second",
        "output_arrow_bytes_per_second",
        "consumer_pause_seconds",
        "fault_to_completion_seconds",
    )
    return [
        {
            **dict(zip(("profile", "mode", "phase", "workload"), key)),
            "samples": len(group),
            "metrics": {
                metric: distribution([s[metric] for s in group if s.get(metric) is not None])
                for metric in metrics
                if any(s.get(metric) is not None for s in group)
            },
        }
        for key, group in sorted(groups.items())
    ]


def recovery_pairs(samples):
    pairs = defaultdict(dict)
    for sample in samples:
        if sample["phase"] in {"recovery", "recovery_control"}:
            pairs[(sample["profile"], sample["iteration"])][sample["phase"]] = sample["total_seconds"]
    return [
        {
            "profile": profile,
            "iteration": iteration,
            **values,
            "additional_seconds": values["recovery"] - values["recovery_control"],
        }
        for (profile, iteration), values in sorted(pairs.items())
        if len(values) == 2
    ]


def metadata(config):
    from vane._native.execution_plan import engine_identity

    root = Path(__file__).resolve().parents[1]
    try:
        git_root, revision = (
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel", "HEAD"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            .strip()
            .splitlines()
        )
        revision = revision if Path(git_root).resolve() == root else None
        dirty = (
            bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True, timeout=5
                )
            )
            if revision
            else None
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        revision = None
        dirty = None
    native_path = Path(vane._native.__file__)
    if root / "vane" in Path(vane.__file__).resolve().parents:
        raise RuntimeError("benchmark requires the installed wheel, not the source package; run python -I")
    return {
        "configuration": asdict(config),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "versions": {p: importlib.metadata.version(p) for p in ("vane-ai", "ray", "pyarrow", "numpy")},
        "engine_identity": engine_identity(),
        "native_sha256": digest(native_path),
        "script_sha256": digest(Path(__file__)),
        "commit": revision,
        "git_dirty": dirty,
        "clock": "perf_counter; p95 uses nearest rank; warmups excluded from summaries",
        "cold_boundary": (
            "new server, Flight listeners, client session and worker pool"
            if config.interface == "flight"
            else "new Runtime, session and worker pool"
        )
        + "; Ray startup recorded separately; OS caches not cleared",
        "client_placement": "same process, real loopback Flight RPCs" if config.interface == "flight" else "embedded",
    }


class Recorder:
    def __init__(self, config):
        self.config = config
        self.samples = []
        self.lock = threading.Lock()

    def append(self, sample, **labels):
        with self.lock:
            record = {**labels, **sample, "sample_index": len(self.samples)}
            with (self.config.output / "samples.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, allow_nan=False) + "\n")
            self.samples.append(record)
        print(
            f"{labels['profile']}/{labels['mode']} {labels['phase']}/{labels['workload']}: {sample['total_seconds']:.4f}s",
            flush=True,
        )

    def sample(self, connection, workload, mode, profile, phase, iteration, expected, **kwargs):
        active = self.begin(workload, mode, profile, phase, iteration)
        value, result = measure(connection, workload, self.config, mode, expected, **kwargs)
        self.append(value, **active)
        return value, result

    def begin(self, workload, mode, profile, phase, iteration):
        active = {"profile": profile, "mode": mode, "phase": phase, "iteration": iteration, "workload": workload.name}
        # Parallel mixed queries have distinct active files, written before SQL.
        write_json(self.config.output / f"active-{profile}-{mode}.json", {**active, "sql": workload.sql})
        return active


def markdown_report(report):
    lines = [
        "# Execution benchmark",
        "",
        f"Completed: {report['complete']}",
        "",
        "Latency in milliseconds; throughput counts output Arrow bytes. Warmups are excluded.",
        "p95 is nearest rank and equals the maximum with fewer than 20 repetitions.",
        "",
        "| Profile | Mode | Scenario | SQL | n | First batch median ms | Total median ms | Output MiB/s median |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for group in report["summary"]:
        metrics = group["metrics"]
        first = metrics.get("first_batch_seconds", {}).get("median")
        first = "—" if first is None else f"{first * 1000:.2f}"
        lines.append(
            f"| {group['profile']} | {group['mode']} | {group['phase']} | {group['workload']} | "
            f"{group['samples']} | {first} | {metrics['total_seconds']['median'] * 1000:.2f} | "
            f"{metrics['output_arrow_bytes_per_second']['median'] / (1 << 20):.2f} |"
        )
    return "\n".join(lines) + "\n"


def profile_samples(connection, config, profile, modes, workloads, expected, recorder, report):
    cases = {w.name: w for w in workloads}

    def timed(workload, mode, phase, iteration, **kwargs):
        return recorder.sample(connection, workload, mode, profile, phase, iteration, expected[workload.name], **kwargs)

    for mode in modes:
        for workload in workloads:
            result = connection.query(workload.sql, options=options(config, mode), rows_per_batch=config.batch_rows)
            try:
                compare(workload, expected[workload.name], result.collect())
            finally:
                close_result(result)
            idle(connection)
            report["validated"].append(
                {"profile": profile, "mode": mode, "workload": workload.name, "rows": expected[workload.name].num_rows}
            )
        for iteration in range(config.warmups):
            for workload in workloads:
                timed(workload, mode, "warmup", iteration)
            idle(connection)
    if "warm" in config.scenarios:
        for iteration in range(config.repetitions):
            order = modes[iteration % len(modes) :] + modes[: iteration % len(modes)]
            for mode in order:
                for workload in workloads[iteration % len(workloads) :] + workloads[: iteration % len(workloads)]:
                    timed(workload, mode, "warm", iteration)
                    idle(connection)
    if "slow" in config.scenarios:
        for iteration in range(config.repetitions):
            for mode in modes:
                timed(cases["scan"], mode, "slow", iteration, paced=True)
                idle(connection)
    if "mixed" in config.scenarios and profile != "local":
        for iteration in range(config.repetitions):
            evidence = {"profile": profile, "iteration": iteration, "worker_reservations": []}
            report.setdefault("mixed_pairs", []).append(evidence)
            with (
                worker_occupancy(connection, evidence["worker_reservations"]),
                connection.cursor() as pipeline,
                connection.cursor() as recovery,
                ThreadPoolExecutor(2) as executor,
            ):
                ready = threading.Event()
                stop = threading.Event()
                long_query = executor.submit(
                    recorder.sample,
                    pipeline,
                    cases["scan"],
                    "pipelined",
                    profile,
                    "mixed",
                    iteration,
                    expected["scan"],
                    paced=True,
                    ready=ready,
                    stop=stop,
                )
                try:
                    deadline = time.monotonic() + config.deadline
                    while not ready.wait(0.05):
                        if long_query.done():
                            long_query.result()
                        if time.monotonic() >= deadline:
                            raise TimeoutError("mixed scan did not deliver its first batch")
                    short_sample, _ = recorder.sample(
                        recovery, cases["aggregate"], "fte", profile, "mixed", iteration, expected["aggregate"]
                    )
                    long_sample, _ = long_query.result(timeout=config.deadline)
                    evidence.update(pipelined_query_id=long_sample["query_id"], fte_query_id=short_sample["query_id"])
                    evidence["max_shared_worker_overlap_seconds"] = mixed_overlap(
                        evidence["worker_reservations"], long_sample["query_id"], short_sample["query_id"]
                    )
                    if evidence["max_shared_worker_overlap_seconds"] <= 0:
                        raise AssertionError(
                            "mixed queries did not overlap in worker reservations; use more rows or a slower consumer"
                        )
                finally:
                    stop.set()
                    pipeline.interrupt()
                    recovery.interrupt()
            idle(connection)
    if "recovery" in config.scenarios and profile != "local":
        for iteration in range(config.repetitions):
            order = ("recovery_control", "recovery") if iteration % 2 == 0 else ("recovery", "recovery_control")
            for phase in order:
                if phase == "recovery_control":
                    timed(cases["aggregate"], "fte", phase, iteration)
                else:
                    recorder.begin(cases["aggregate"], "fte", profile, phase, iteration)
                    with worker_loss(connection) as fault:
                        value, _ = measure(connection, cases["aggregate"], config, "fte", expected["aggregate"])
                    value.update(recovery_details(fault, value))
                    recorder.append(
                        value, profile=profile, mode="fte", phase=phase, iteration=iteration, workload="aggregate"
                    )
                idle(connection)
    # Independent retained-batch snapshots, not RSS peaks or timed measurements.
    snapshots = []
    for mode in modes:
        result = connection.query(cases["scan"].sql, options=options(config, mode), rows_per_batch=config.batch_rows)
        batch = None
        try:
            try:
                batch = result.read_batch()
            except StopIteration:
                pass
            time.sleep(0.1)
            snapshots.append(
                {
                    "profile": profile,
                    "mode": mode,
                    "retained_batch_rows": batch.num_rows if batch is not None else 0,
                    "diagnostics": result.diagnostics(),
                    **(
                        {
                            "server_diagnostics": connection.queries[result.query_id].context.diagnostics(),
                            "client_resources": connection.client.resource_snapshot(),
                            "gateway_links": connection.server._gateway.active_links,
                        }
                        if isinstance(connection, FlightConnection)
                        else {}
                    ),
                }
            )
        finally:
            batch = None
            close_result(result)
        idle(connection)
    return snapshots


def run(config):
    """Use an existing Ray cluster; the CLI owns its cluster separately."""
    config.output.mkdir(parents=True, exist_ok=False)
    recorder = Recorder(config)
    connections = []
    report = {"complete": False}
    try:
        report.update(metadata(config))
        if any(mode != "local" for mode in config.modes):
            import ray

            report["ray_cluster_resources"] = ray.cluster_resources()
        write_json(config.output / "configuration.json", report)
        workloads, files = dataset(config)
        report["input_files"] = files
        cases = {w.name: w for w in workloads}
        with connect(config, "local") as reference:
            expected = {w.name: reference.execute(w.sql).to_arrow_table() for w in workloads}
        if "mixed" in config.scenarios and not expected["scan"].num_rows:
            raise ValueError("mixed requires a nonempty scan; increase rows")
        jobs = [("local", "local")] if "local" in config.modes else []
        jobs += [(p, m) for p in config.profiles for m in config.modes if m != "local"]
        report["resources"] = {p: asdict(resources(config, p)) for p, _ in jobs}
        report["validated"] = []
        if "cold" in config.scenarios:
            for iteration in range(config.repetitions):
                order = jobs[iteration % len(jobs) :] + jobs[: iteration % len(jobs)]
                for profile, mode in order:
                    recorder.begin(cases["tiny"], mode, profile, "cold", iteration)
                    before = time.perf_counter()
                    with connect(config, profile) as connection:
                        opened = time.perf_counter() - before
                        value, _ = measure(connection, cases["tiny"], config, mode, expected["tiny"])
                        value["session_open_seconds"] = opened
                        value["session_first_query_seconds"] = opened + value["total_seconds"]
                        recorder.append(
                            value, profile=profile, mode=mode, phase="cold", iteration=iteration, workload="tiny"
                        )
                        idle(connection)
        # Profiles run sequentially on the same finite cluster. Retaining idle
        # worker pools would reserve the CPUs needed by the next profile.
        snapshots = []
        for profile in dict(jobs):
            with connect(config, profile) as connection:
                connections.append(connection)
                modes = [mode for selected, mode in jobs if selected == profile]
                snapshots.extend(
                    profile_samples(connection, config, profile, modes, workloads, expected, recorder, report)
                )
        write_json(config.output / "resources.json", snapshots)
        report["complete"] = True
    except BaseException as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        # Preserve the primary error even when a closed runtime cannot be probed.
        for connection in connections:
            try:
                report.setdefault("failure_resources", []).append(connection.query_runtime.resource_snapshot())
            except Exception:
                pass
        raise
    finally:
        report["samples"] = recorder.samples
        report["summary"] = summarize(recorder.samples)
        report["recovery_pairs"] = recovery_pairs(recorder.samples)
        write_json(config.output / "report.json", report)
        (config.output / "report.md").write_text(markdown_report(report), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="new directory; existing results are never overwritten"
    )
    for name in (
        "rows",
        "seed",
        "repetitions",
        "warmups",
        "worker-count",
        "worker-threads",
        "partitions",
        "batch-rows",
    ):
        parser.add_argument("--" + name, type=int, default=getattr(Configuration, name.replace("-", "_")))
    parser.add_argument("--consumer-rows-per-second", type=float, default=Configuration.consumer_rows_per_second)
    parser.add_argument("--deadline", type=float, default=Configuration.deadline)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--interface", choices=INTERFACES, default="runtime")
    args = vars(parser.parse_args(argv))
    args["output"] = args["output"].resolve()
    try:
        config = Configuration(**args)
        if config.output.exists():
            raise ValueError("output directory already exists")
    except ValueError as error:
        parser.error(str(error))
    ray = None
    owns_cluster = False
    started = time.perf_counter()
    previous_directory = Path.cwd()
    previous_pythonpath = os.environ.get("PYTHONPATH")
    workdir = tempfile.TemporaryDirectory(prefix="vane-execution-benchmark-")
    try:
        # Ray workers do not inherit the driver's -I. Like the installed pytest
        # launcher, start outside the checkout and prioritize the installed
        # package in worker imports, including pre-existing PYTHONPATH entries.
        os.chdir(workdir.name)
        installed = str(Path(vane.__file__).resolve().parent.parent)
        os.environ["PYTHONPATH"] = installed + (os.pathsep + previous_pythonpath if previous_pythonpath else "")
        if set(config.modes) != {"local"}:
            import ray

            if ray.is_initialized():
                raise RuntimeError("CLI requires its own cluster; use run(config) with an existing test cluster")
            owns_cluster = True
            ray.init(address="local", num_cpus=config.worker_count * config.worker_threads, include_dashboard=False)
        cluster_startup = time.perf_counter() - started if ray is not None else None
        report = run(config)
        report["cluster_startup_seconds"] = cluster_startup
        write_json(config.output / "report.json", report)
        print(f"Report: {config.output / 'report.json'}", flush=True)
    finally:
        try:
            if owns_cluster:
                ray.shutdown()
        finally:
            os.chdir(previous_directory)
            if previous_pythonpath is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = previous_pythonpath
            workdir.cleanup()


if __name__ == "__main__":
    main()

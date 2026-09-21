#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise the internal CPU serving lifecycle against an installed Vane wheel.

This is a deterministic text/RGB feature fixture, not a learned embedding model
or a network server. See LOCAL_SERVING_ACCEPTANCE.md for the measurement scope.
Run with python -I; the CLI uses a temporary working directory for subprocesses.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa

import vane
from vane.execution.request_admission import (
    RequestAdmissionLimits,
    RequestCancelled,
    RequestExecutionTimeout,
    RequestQueueFull,
    RequestQueueTimeout,
)
from vane.execution.resources import ResourceVector
from vane.execution.result_delivery import ResultDeliveryFull, ResultDeliveryLimits, ResultDeliveryTimeout
from vane.execution.udf_data_admission import DataAdmissionLimits
from vane.execution.udf_local_model import LocalModelRuntime
from vane.execution.udf_runtime_admission import TaskAdmissionLimits


def require(condition, message):
    # Acceptance must also fail under python -O.
    if not condition:
        raise AssertionError(message)


def wait_for(predicate, message, timeout=30):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError(message)
        time.sleep(0.01)


@contextmanager
def expect(error_type, message=None):
    try:
        yield
    except error_type as error:
        if message is not None:
            require(message in str(error), f"unexpected {type(error).__name__}: {error}")
    else:
        raise AssertionError(f"expected {error_type.__name__}")


def distribution(values):
    """Nearest-rank quantiles; empty groups carry no invented latency."""
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "mean": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "p99": ordered[math.ceil(0.99 * len(ordered)) - 1],
        "max": ordered[-1],
    }


def model_class(directory):
    # Capture only this immutable string. Rebuilt plans serialize the same
    # constructor/callable, not mutable driver counters or open file handles.
    directory = str(directory)

    class TextImageFeatures:
        def __init__(self):
            import os
            from pathlib import Path

            import numpy as np

            self.directory = Path(directory)
            self.scale = np.array([1 / 255, 1 / 255, 1 / 255], dtype=np.float64)
            with (self.directory / "initializations").open("a") as output:
                output.write(f"{os.getpid()}\n")

        def __call__(self, table):
            import hashlib
            import os
            import time

            import numpy as np
            import pyarrow as pa

            mode = table["mode"][0].as_py()
            token = table["token"][0].as_py()
            with (self.directory / f"calls-{token}").open("a") as output:
                output.write(f"{os.getpid()}\n")
            (self.directory / f"entered-{token}").touch()
            if mode == "worker_exit":
                os._exit(23)
            if mode == "udf_error":
                raise ValueError("planned serving UDF failure")
            if mode == "gated":
                deadline = time.monotonic() + 30
                while not (self.directory / f"release-{token}").exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("acceptance driver did not release gated UDF")
                    time.sleep(0.01)

            features = []
            for text, image in zip(table["text"].to_pylist(), table["image"].to_pylist()):
                pixels = np.frombuffer(image, dtype=np.uint8).reshape(8, 8, 3)
                # Fixed CPU work makes the larger analysis batch cost more.
                # This has no semantic embedding/quality claim.
                digest = hashlib.sha256(text.encode() + image)
                for _ in range(128):
                    digest = hashlib.sha256(digest.digest())
                rgb = pixels.mean(axis=(0, 1)) * self.scale
                features.append([float(len(text.split())), *rgb.tolist()])
            return pa.table(
                {
                    "id": table["id"],
                    "features": pa.array(features, type=pa.list_(pa.float64())),
                    "worker_pid": pa.array([os.getpid()] * len(table), type=pa.int64()),
                    "padding": pa.array([b"x" * (48 * 1024 if mode == "large" else 0)] * len(table)),
                }
            )

    return TextImageFeatures


class Scenario:
    """One session and resident model; independent cursors/plans per request."""

    def __init__(self, directory):
        self.directory = directory
        self.model_type = model_class(directory)
        self.connection = vane.connect(config={"threads": "2"})
        self.checkpoints = {}
        self.observed_worker_failures = 0
        self.recovery_initializations = {}
        self.image = bytes([64, 128, 192]) * 64
        self.runtime = None
        try:
            with self.plan() as (cursor, bound, _bindings, _token):
                self.runtime = LocalModelRuntime(
                    session_id=bound.session_id(),
                    session_config=bound.session_config(),
                    resident_limit=ResourceVector(cpu=1, heap_bytes=16 * 1024**2),
                    task_limit=TaskAdmissionLimits(1, 8),
                    data_limit=DataAdmissionLimits(2 * 1024**2, 64 * 1024, 128 * 1024),
                    request_limit=RequestAdmissionLimits(2, 2, queue_timeout=30),
                    result_limit=ResultDeliveryLimits(2, 64 * 1024),
                )
                node = bound.collect_udf_nodes(conn=cursor)[0]
                self.model = self.runtime.register("text-image", version="fixture-v1", payload=node["payload"])
        except BaseException:
            if self.runtime is not None:
                self.runtime.close(timeout=30, kill=True)
            self.connection.close()
            raise

    @contextmanager
    def plan(self, mode="short", rows=1, token=None):
        token = token or uuid.uuid4().hex
        with self.connection.cursor() as cursor:
            relation = cursor.sql(
                f"SELECT i::BIGINT AS id, 'red green blue' AS text, "
                f"from_hex('{self.image.hex()}') AS image, '{mode}' AS mode, '{token}' AS token "
                f"FROM range({rows}) AS t(i)"
            ).map_batches(
                self.model_type,
                schema={
                    "id": vane.sqltypes.BIGINT,
                    "features": vane.list_type(vane.sqltypes.DOUBLE),
                    "worker_pid": vane.sqltypes.BIGINT,
                    "padding": vane.sqltypes.BLOB,
                },
                execution_backend="subprocess_actor",
                actor_number=1,
                cpus=1,
                memory_bytes=16 * 1024**2,
            )
            bound = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(cursor)
            bindings = {str(node["node_id"]): "text-image" for node in bound.collect_udf_nodes(conn=cursor)}
            yield cursor, bound, bindings, token

    def initializations(self):
        path = self.directory / "initializations"
        return path.read_text().splitlines() if path.exists() else []

    def calls(self, token):
        path = self.directory / f"calls-{token}"
        return path.read_text().splitlines() if path.exists() else []

    def consume(self, result, rows=1):
        ids, pids = [], set()
        with result:
            for table in result:
                expected = [3.0, 64 / 255, 128 / 255, 192 / 255]
                # Native physical results use positional c0/c1/... names.
                require(np.allclose(table.column(1).to_pylist(), [expected] * len(table)), "wrong text/RGB features")
                ids.extend(table.column(0).to_pylist())
                pids.update(table.column(2).to_pylist())
        require(sorted(ids) == list(range(rows)), f"wrong row identities for {rows} rows")
        return sorted(pids)

    def query(self, mode="short", rows=1):
        with self.plan(mode, rows) as (cursor, bound, bindings, _token):
            started = time.monotonic()
            refusals = 0
            with self.runtime.request() as request:
                while True:
                    try:
                        result = request.execute_result(bound, bindings, conn=cursor, execution_timeout=30)
                        break
                    except ResultDeliveryFull:
                        # A finished execution cannot replay, even for byte
                        # pressure. Only an unclaimed ready ticket can retry.
                        if request.state != "ready" or time.monotonic() - started >= 30:
                            raise
                        refusals += 1
                        time.sleep(0.01)
                pids = self.consume(result, rows)
            return {
                "kind": mode,
                "rows": rows,
                "latency_seconds": time.monotonic() - started,
                **request.timing_snapshot(),
                **result.timing_snapshot(),
                "worker_pids": pids,
                "result_slot_refusals": refusals,
            }

    def produce(self, mode="short", **options):
        options.setdefault("execution_timeout", 30)
        with self.plan(mode) as (cursor, bound, bindings, _token), self.runtime.request() as request:
            return request.execute_result(bound, bindings, conn=cursor, **options)

    def checkpoint(self, name):
        snapshot = self.runtime.resource_snapshot()
        self.checkpoints[name] = snapshot
        return snapshot

    def quiescent(self, name, *, closed=False):
        snapshot = self.checkpoint(name)
        require(snapshot["active_borrows"] == 0, f"{name}: model borrow retained")
        requests, tasks, data, results = (
            snapshot[key] for key in ("request_admission", "task_admission", "data", "result_delivery")
        )
        require(
            all(requests[key] == 0 for key in ("active_requests", "queued_requests", "cleanup_pending_requests")),
            f"{name}: request owner retained",
        )
        require(
            all(
                tasks[key] == 0
                for key in (
                    "queries",
                    "ready_tasks",
                    "running_tasks",
                    "waiting_tasks",
                    "resuming_tasks",
                    "queued_tasks",
                )
            ),
            f"{name}: task owner retained",
        )
        require(
            all(data[key] == 0 for key in ("queries", "tasks", "leases", "reservations", "usage_bytes")),
            f"{name}: data owner retained",
        )
        require(
            all(results[key] == 0 for key in ("active_results", "usage_bytes", "buffers")),
            f"{name}: result owner retained",
        )
        expected = ResourceVector() if closed else ResourceVector(cpu=1, heap_bytes=16 * 1024**2)
        require(snapshot["reserved_resources"] == expected.to_dict(), f"{name}: resident accounting changed")
        return snapshot

    def ingress(self):
        requests = [self.runtime.request() for _ in range(4)]
        try:
            require([r.state for r in requests] == ["ready", "ready", "queued", "queued"], "ingress capacities")
            with expect(RequestQueueFull):
                self.runtime.request()
            requests[0].cancel()
            require(requests[2].state == "ready" and requests[3].state == "queued", "FIFO promotion")
            require(requests[3].cancel(), "queued cancellation")
            timed = self.runtime.request(queue_timeout=0.02)
            with timed, self.plan() as (cursor, bound, bindings, token):
                with expect(RequestQueueTimeout):
                    timed.execute_result(bound, bindings, conn=cursor)
                require(not self.calls(token), "expired queued request ran UDF")
        finally:
            for request in requests:
                request.shutdown()
        self.quiescent("ingress_recovered")

    def result_pressure(self):
        results = [self.produce(), self.produce()]
        try:
            with self.plan() as (cursor, bound, bindings, token), self.runtime.request() as request:
                for _ in range(3):
                    with expect(ResultDeliveryFull, "slots"):
                        request.execute_result(bound, bindings, conn=cursor)
                require(not self.calls(token) and request.state == "ready", "slot refusal executed UDF")
                self.checkpoint("slow_consumer_slots")
                self.consume(results.pop())
                self.consume(request.execute_result(bound, bindings, conn=cursor))
                require(len(self.calls(token)) == 1, "slot retry replayed execution")
        finally:
            for result in results:
                result.close()
        self.quiescent("slot_pressure_recovered")

        result = self.produce("large")
        table = result.take()
        result.close()
        try:
            snapshot = self.checkpoint("slow_consumer_view")
            require(snapshot["result_delivery"]["active_results"] == 0, "final handoff kept slot")
            require(48 * 1024 <= snapshot["result_delivery"]["exported_bytes"] <= 64 * 1024, "exported view uncharged")
            with self.plan("large") as (cursor, bound, bindings, token), self.runtime.request() as request:
                with expect(ResultDeliveryFull, "byte capacity"):
                    request.execute_result(bound, bindings, conn=cursor)
                require(len(self.calls(token)) == 1, "byte refusal replayed UDF")
                with expect(RuntimeError, "only execute once"):
                    request.execute_result(bound, bindings, conn=cursor)
        finally:
            del table
        self.quiescent("byte_pressure_recovered")
        self.query()
        result = self.produce()
        require(result.cancel(), "delivery cancellation")
        self.quiescent("delivery_cancelled")
        timeouts = self.runtime.resource_snapshot()["result_delivery"]["timed_out_results"]
        result = None
        try:
            result = self.produce(delivery_timeout=0.1)
        except ResultDeliveryTimeout:
            # A descheduled publisher may observe expiry inside ready().
            # This is the same deadline outcome, with no handle to consume.
            pass
        if result is not None:
            wait_for(lambda: result.state == "delivery_timed_out", "abandoned result did not expire")
            with expect(ResultDeliveryTimeout):
                result.take()
            result.close()
        require(
            self.runtime.resource_snapshot()["result_delivery"]["timed_out_results"] == timeouts + 1,
            "delivery expiry not counted exactly once",
        )
        self.quiescent("delivery_expired")

    def cancellation(self):
        token = uuid.uuid4().hex
        with self.plan("gated", token=token) as (cursor, bound, bindings, _), self.runtime.request() as request:
            with ThreadPoolExecutor(max_workers=1) as threads:
                future = threads.submit(request.execute_result, bound, bindings, conn=cursor)
                try:
                    wait_for(lambda: (self.directory / f"entered-{token}").exists(), "gated UDF did not start")
                    require(request.cancel(), "running cancellation was refused")
                    with expect(RequestCancelled):
                        future.result(timeout=30)
                finally:
                    (self.directory / f"release-{token}").touch()
                    request.cancel()
        self.quiescent("execution_cancelled")
        self.query()
        with self.plan() as (cursor, bound, bindings, token), self.runtime.request() as request:
            with expect(RequestExecutionTimeout):
                request.execute_result(bound, bindings, conn=cursor, execution_timeout=0)
            require(not self.calls(token), "zero execution deadline ran UDF")
        self.quiescent("execution_expired")

    def failures(self):
        for mode in ("udf_error", "worker_exit"):
            before = len(self.initializations())
            with self.plan(mode) as (cursor, bound, bindings, token), self.runtime.request() as request:
                failed_before = self.runtime.resource_snapshot()["request_admission"]["failed_executions"]
                with expect(Exception, "planned serving UDF failure" if mode == "udf_error" else None):
                    request.execute_result(bound, bindings, conn=cursor)
                require(len(self.calls(token)) == 1, f"{mode}: failed UDF not run exactly once")
                require(
                    self.runtime.resource_snapshot()["request_admission"]["failed_executions"] == failed_before + 1,
                    "execution failure not counted",
                )
            self.quiescent(f"{mode}_cleaned")
            self.query()
            after = len(self.initializations())
            self.recovery_initializations[mode] = after - before
            if mode == "worker_exit":
                self.observed_worker_failures += 1
                require(after == before + 1, "lost worker was not replaced exactly once")
            else:
                # Reported UDF errors retire the local worker gracefully;
                # the registered pool, identity, and reservation remain owned.
                require(after == before + 1, "reported-error worker did not recover exactly once")
            self.quiescent(f"{mode}_recovered")

    def close(self):
        try:
            self.runtime.close(timeout=30, kill=True)
        finally:
            self.connection.close()


def run_acceptance(directory, *, requests=20, concurrency=4):
    if type(requests) is not int or requests < 2 or type(concurrency) is not int or not 1 <= concurrency <= 4:
        raise ValueError("requests must be >= 2; concurrency must be between 1 and 4")
    scenario = Scenario(directory)
    try:
        cold = scenario.query()
        cold_initializations = len(scenario.initializations())
        require(cold_initializations == 1, "cold request did not initialize one worker")
        started = time.monotonic()
        scenario.model.prewarm()
        prewarm_seconds = time.monotonic() - started
        warm = [scenario.query() for _ in range(requests)]
        scenario.quiescent("sequential_recovered")
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=concurrency) as threads:
            mixed = list(
                threads.map(
                    lambda i: scenario.query("analysis", 32) if i % 4 == 0 else scenario.query(), range(requests)
                )
            )
        mixed_seconds = time.monotonic() - started
        healthy_additional_initializations = len(scenario.initializations()) - cold_initializations
        require(healthy_additional_initializations == 0, "healthy requests reinitialized model")
        require(all(s["worker_pids"] == cold["worker_pids"] for s in [*warm, *mixed]), "healthy model identity changed")
        scenario.quiescent("mixed_recovered")
        scenario.ingress()
        scenario.result_pressure()
        scenario.cancellation()
        scenario.failures()
        before_close = scenario.quiescent("before_close")
        scenario.runtime.drain()
        with expect(RuntimeError, "draining"):
            scenario.runtime.request()
        scenario.runtime.close(timeout=30)
        scenario.quiescent("closed", closed=True)
        groups = {
            "cold": [cold],
            "warm": warm,
            "mixed_short": [s for s in mixed if s["kind"] == "short"],
            "mixed_analysis": [s for s in mixed if s["kind"] == "analysis"],
        }
        return {
            "schema_version": 1,
            "status": "passed",
            "environment": {
                "python": platform.python_version(),
                "platform": platform.system(),
                "machine": platform.machine(),
                "cpu_count": os.cpu_count(),
                "vane": importlib.metadata.version("vane-ai"),
                "pyarrow": pa.__version__,
            },
            "configuration": {
                "requests_per_phase": requests,
                "client_concurrency": concurrency,
                "actors": 1,
                "active_requests": 2,
                "queued_requests": 2,
                "tasks": 1,
                "results": 2,
                "result_bytes": 64 * 1024,
            },
            "model": {
                "cold_initializations": cold_initializations,
                "healthy_additional_initializations": healthy_additional_initializations,
                "prewarm_seconds": prewarm_seconds,
                "total_initializations": len(scenario.initializations()),
                "observed_worker_exit_failures": scenario.observed_worker_failures,
                "recovery_initializations": scenario.recovery_initializations,
            },
            "measurements": {
                name: {
                    field: distribution([s[field] for s in samples])
                    for field in (
                        "latency_seconds",
                        "queue_wait_seconds",
                        "execution_seconds",
                        "cleanup_seconds",
                        "delivery_seconds",
                    )
                }
                for name, samples in groups.items()
            },
            "mixed_throughput_requests_per_second": len(mixed) / mixed_seconds,
            "load_result_slot_refusals": sum(s["result_slot_refusals"] for s in [cold, *warm, *mixed]),
            "before_close": before_close,
            "checkpoints": scenario.checkpoints,
            "scope": "Synthetic text/RGB CPU features; internal native plans; materialized results. Latency excludes planning, includes admission through consumption. Mixed throughput includes planning. No network sends, native streaming, process RSS bound, learned embedding quality, or GPU claim.",
        }
    finally:
        scenario.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if args.requests < 2 or not 1 <= args.concurrency <= 4:
        parser.error("requests must be >= 2; concurrency must be between 1 and 4")
    report_path = args.report.resolve()
    original_directory = Path.cwd()
    os.environ["VANE_RUNNER"] = "local-fast"
    with TemporaryDirectory(prefix="vane-serving-acceptance-") as directory:
        try:
            # Worker -m imports must resolve the installed package as well.
            os.chdir(directory)
            report = run_acceptance(Path(directory), requests=args.requests, concurrency=args.concurrency)
        finally:
            os.chdir(original_directory)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"CPU serving acceptance passed; report: {report_path}")


if __name__ == "__main__":
    main()

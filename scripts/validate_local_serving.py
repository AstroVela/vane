#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise public local-fast SQL/Relation serving against an installed wheel.

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
from contextlib import ExitStack, contextmanager
from functools import partial
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
def expect(error_type, message=None, **fields):
    try:
        yield
    except error_type as error:
        if message is not None:
            require(message in str(error), f"unexpected {type(error).__name__}: {error}")
        for name, value in fields.items():
            require(
                getattr(error, name) == value, f"unexpected {type(error).__name__}.{name}: {getattr(error, name)!r}"
            )
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


def model_class(directory, *, gpu_device=None):
    # Capture immutable directory/device strings. Rebuilt plans serialize the
    # same constructor/callable, not driver counters or open file handles.
    directory = str(directory)

    class TextImageFeatures:
        def __init__(self):
            import os
            from pathlib import Path

            import numpy as np

            self.directory = Path(directory)
            if gpu_device is None:
                self.scale = np.array([1 / 255, 1 / 255, 1 / 255], dtype=np.float64)
            else:
                import torch

                require(os.environ.get("CUDA_VISIBLE_DEVICES") == gpu_device, "wrong CUDA device assignment")
                require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "expected one CUDA device")
                self.scale = torch.tensor([1 / 255] * 3, dtype=torch.float64, device="cuda")
                torch.cuda.synchronize()
                self.gpu_sample()
            with (self.directory / "initializations").open("a") as output:
                output.write(f"{os.getpid()}\n")

        def gpu_sample(self):
            import json
            import os

            import torch

            sample = {
                "pid": os.getpid(),
                "device": gpu_device,
                "device_count": torch.cuda.device_count(),
                "name": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            temporary = self.directory / "gpu-worker.tmp"
            temporary.write_text(json.dumps(sample))
            temporary.replace(self.directory / "gpu-worker.json")

        def __call__(self, ids, texts, images, modes, tokens):
            import hashlib
            import os
            import time

            import numpy as np
            import pyarrow as pa

            mode = modes[0].as_py()
            token = tokens[0].as_py()
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
            for text, image in zip(texts.to_pylist(), images.to_pylist()):
                pixels = np.frombuffer(image, dtype=np.uint8).reshape(8, 8, 3)
                # Fixed CPU work makes the larger analysis batch cost more.
                # This has no semantic embedding/quality claim.
                digest = hashlib.sha256(text.encode() + image)
                for _ in range(128):
                    digest = hashlib.sha256(digest.digest())
                if gpu_device is None:
                    rgb = pixels.mean(axis=(0, 1)) * self.scale
                else:
                    import torch

                    rgb = torch.tensor(pixels, dtype=torch.float64, device="cuda").mean(dim=(0, 1)) * self.scale
                    # The host copy waits for CUDA completion before returning
                    # an Arrow result and releasing its execution allowance.
                    rgb = rgb.cpu()
                features.append([float(len(text.split())), *rgb.tolist()])
            if gpu_device is not None:
                self.gpu_sample()
            return pa.StructArray.from_arrays(
                [
                    ids,
                    pa.array(features, type=pa.list_(pa.float64())),
                    pa.array([os.getpid()] * len(ids), type=pa.int64()),
                    pa.array([b"x" * (48 * 1024 if mode == "large" else 0)] * len(ids)),
                ],
                names=["id", "features", "worker_pid", "padding"],
            )

    return vane.cls.batch(
        actor_number=1,
        gpus=0 if gpu_device is None else 1,
        name="text_image_features",
        batch_size=32,
        unnest=True,
        return_dtype=pa.struct(
            [
                ("id", pa.int64()),
                ("features", pa.list_(pa.float64())),
                ("worker_pid", pa.int64()),
                ("padding", pa.binary()),
            ]
        ),
    )(TextImageFeatures)


class Scenario:
    """One public session runtime/model; an independent cursor per client."""

    def __init__(self, directory, *, queue_timeout=30, execution_timeout=30, gpu_device=None):
        self.directory = directory
        self.gpu_device = gpu_device
        self.execution_timeout = execution_timeout
        self.resident_resources = ResourceVector(cpu=1, gpu=int(gpu_device is not None), heap_bytes=16 * 1024**2)
        self.model_type = model_class(directory, gpu_device=gpu_device)
        self.connection = vane.connect(config={"threads": "2"})
        self.checkpoints = {}
        self.observed_worker_failures = 0
        self.recovery_initializations = {}
        self.image = bytes([64, 128, 192]) * 64
        self.runtime = None
        try:
            self.runtime = self.connection.configure_local_runtime(
                resident_limit=self.resident_resources,
                gpu_devices=None if gpu_device is None else [gpu_device],
                task_limit=TaskAdmissionLimits(1, 8),
                data_limit=DataAdmissionLimits(2 * 1024**2, 64 * 1024, 128 * 1024),
                request_limit=RequestAdmissionLimits(2, 2, queue_timeout=queue_timeout),
                result_limit=ResultDeliveryLimits(2, 64 * 1024),
                execution_timeout=execution_timeout,
            )
            self.model = self.runtime.register_model(
                "text-image",
                self.model_type(),
                version="fixture-v1",
                parameters=["BIGINT", "VARCHAR", "BLOB", "VARCHAR", "VARCHAR"],
                cpus=1,
                memory_bytes=16 * 1024**2,
                gpu_devices=None if gpu_device is None else [gpu_device],
            )
            vane.attach_function(self.model, "text_image_features", connection=self.connection)
        except BaseException:
            if self.runtime is not None:
                self.runtime.close(timeout=30, kill=True)
            self.connection.close()
            raise

    @contextmanager
    def client(self, api="sql", mode="short", rows=1, token=None):
        token = token or uuid.uuid4().hex
        with self.connection.cursor() as cursor:
            yield cursor, token, partial(self.execute, cursor, api, mode, rows, token)

    def execute(self, cursor, api, mode, rows, token, **options):
        parameters = {"text": "red green blue", "image": self.image, "mode": mode, "token": token, "rows": rows}
        if api == "sql":
            return cursor.execute_result(
                "SELECT encoded.* FROM (SELECT text_image_features(i, $text, $image, $mode, $token) AS encoded "
                "FROM range($rows) AS t(i))",
                parameters,
                **options,
            )
        require(api == "relation", f"unknown query API: {api}")
        relation = cursor.sql(
            "SELECT i::BIGINT AS id, $text AS text, $image AS image, $mode AS mode, $token AS token "
            "FROM range($rows) AS t(i)",
            params=parameters,
        ).project(self.model(*(vane.col(name) for name in ("id", "text", "image", "mode", "token"))))
        return relation.execute_result(**options)

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
                require(table.column_names == ["id", "features", "worker_pid", "padding"], "lost public result schema")
                require(np.allclose(table["features"].to_pylist(), [expected] * len(table)), "wrong text/RGB features")
                ids.extend(table["id"].to_pylist())
                pids.update(table["worker_pid"].to_pylist())
        require(sorted(ids) == list(range(rows)), f"wrong row identities for {rows} rows")
        return sorted(pids)

    def query(self, mode="short", rows=1, *, api="sql"):
        started = time.monotonic()
        with self.client(api, mode, rows) as (_, token, execute):
            result, refusals = self.execute_with_slot_retry(execute, token)
            pids = self.consume(result, rows)
        return {
            "api": api,
            "kind": mode,
            "rows": rows,
            "latency_seconds": time.monotonic() - started,
            **result.timing_snapshot(),
            "worker_pids": pids,
            "result_slot_refusals": refusals,
        }

    def execute_with_slot_retry(self, execute, token):
        deadline = time.monotonic() + 30
        refusals = 0
        while True:
            try:
                return execute(), refusals
            except ResultDeliveryFull as error:
                # Each retry owns a new ticket. Capacity fields, rather than
                # messages or fixture markers, identify a pre-execution refusal.
                if error.reason != "slots" or error.execution_started is not False or time.monotonic() >= deadline:
                    raise
                require(not self.calls(token), "pre-execution slot refusal ran UDF")
                refusals += 1
                time.sleep(0.01)

    def produce(self, mode="short", *, api="sql", **options):
        with self.client(api, mode) as (_, _, execute):
            return execute(**options)

    def checkpoint(self, name):
        snapshot = self.runtime.resource_snapshot()
        self.checkpoints[name] = snapshot
        return snapshot

    def quiescent(self, name, *, closed=False, timeout=30):
        snapshot = self.checkpoint(name)
        require(snapshot["active_borrows"] == 0, f"{name}: model borrow retained")
        requests, tasks = (snapshot[key] for key in ("request_admission", "task_admission"))
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

        def data_released():
            nonlocal snapshot
            # A completed Future's callback can briefly retain the original
            # UDF output after request teardown and managed-result copying.
            # Observe its release without forcing GC or dropping the charge.
            snapshot = self.checkpoint(name)
            return all(
                snapshot["data"][key] == 0 for key in ("queries", "tasks", "leases", "reservations", "usage_bytes")
            )

        wait_for(data_released, f"{name}: data owner retained", timeout=timeout)
        results = snapshot["result_delivery"]
        require(
            all(results[key] == 0 for key in ("active_results", "usage_bytes", "buffers")),
            f"{name}: result owner retained",
        )
        expected = ResourceVector() if closed or not self.initializations() else self.resident_resources
        require(snapshot["reserved_resources"] == expected.to_dict(), f"{name}: resident accounting changed")
        return snapshot

    def gpu_checkpoint(self, name, *, closed=False):
        if self.gpu_device is None:
            return None
        state = self.checkpoint(name)["gpu"]
        require(state["devices"] == [self.gpu_device], "GPU inventory changed")
        require(len(state["models"]) == 1, "unexpected GPU model ownership")
        model = state["models"][0]
        require(model["devices"] == [self.gpu_device], "GPU model assignment changed")
        if closed:
            require(model["pools"] == [], "closed runtime retained a GPU pool")
            return {"device": self.gpu_device, "workers": []}
        require(len(model["pools"]) == 1, "expected one resident GPU pool")
        workers = model["pools"][0]["workers"]
        require(len(workers) == 1, "GPU pool retained extra worker generations")
        worker = workers[0]
        require(worker["device"] == self.gpu_device and worker["replica"] == 0, "GPU worker assignment changed")
        require(worker["pid"] == int(self.initializations()[-1]), "GPU worker PID differs from constructor")
        require(not worker["cleanup_finished"], "GPU pool retained a finished worker")
        os.kill(worker["pid"], 0)
        sample = json.loads((self.directory / "gpu-worker.json").read_text())
        require(sample["pid"] == worker["pid"] and sample["device"] == self.gpu_device, "stale CUDA evidence")
        require(sample["device_count"] == 1 and sample["allocated_bytes"] > 0, "missing resident CUDA tensor")
        return {"worker": worker, "cuda": sample}

    def recover(self):
        if self.gpu_device is not None:
            # Initialization has its own startup bound. Keep it out of the
            # short execution deadline used to exercise an already-warm model.
            self.model.prewarm()
        return self.query()

    def release(self, token):
        (self.directory / f"release-{token}").touch()

    def entered(self, token):
        return (self.directory / f"entered-{token}").exists()

    def request_count(self, name):
        return self.runtime.resource_snapshot()["request_admission"][name]

    @contextmanager
    def gated_clients(self, count):
        with ExitStack() as stack:
            clients = [stack.enter_context(self.client(("sql", "relation")[i % 2], "gated")) for i in range(count)]
            with ThreadPoolExecutor(max_workers=count) as threads:
                try:
                    yield clients, threads
                finally:
                    # Always unblock fixture workers before waiting for client
                    # threads, including when an acceptance assertion fails.
                    for _, token, _ in clients:
                        self.release(token)
                    for cursor, _, _ in clients:
                        cursor.interrupt()

    def run_client(self, token, execute):
        result, _ = self.execute_with_slot_retry(execute, token)
        return self.consume(result)

    def ingress(self):
        with self.gated_clients(4) as (clients, threads):
            futures = [threads.submit(self.run_client, *clients[0][1:])]
            wait_for(lambda: self.entered(clients[0][1]), "first ingress UDF did not start")
            for i in range(1, 4):
                futures.append(threads.submit(self.run_client, *clients[i][1:]))
                field, count = ("running_requests", 2) if i == 1 else ("queued_requests", i - 1)
                wait_for(lambda: self.request_count(field) == count, "ingress did not fill in order")
            with self.client() as (_, token, execute), expect(RequestQueueFull):
                execute()
            require(not self.calls(token), "overload ran UDF")
            state = self.checkpoint("ingress_full")
            require(state["result_delivery"]["active_results"] == 2, "queued requests occupied result capacity")
            clients[3][0].interrupt()
            with expect(RequestCancelled):
                futures[3].result(timeout=30)
            require(not self.calls(clients[3][1]), "queued cancellation ran UDF")
            require(self.request_count("queued_requests") == 1, "queued cancellation did not free ingress")
            # A promoted request may refuse a result slot before its earlier
            # producer's consumer runs. Its retry joins ingress as a new call,
            # so output completion order is not a FIFO-admission assertion.
            for _, token, _ in clients:
                self.release(token)
            for i, future in enumerate(futures[:3]):
                future.result(timeout=30)
                require(len(self.calls(clients[i][1])) == 1, "ingress retry replayed UDF")
        self.quiescent("ingress_recovered")

    def queue_expiry(self):
        with self.gated_clients(3) as (clients, threads):
            active = [threads.submit(self.run_client, *clients[0][1:])]
            wait_for(lambda: self.entered(clients[0][1]), "queue-timeout fixture did not start")
            active.append(threads.submit(self.run_client, *clients[1][1:]))
            wait_for(lambda: self.request_count("running_requests") == 2, "active ingress did not fill")
            with expect(RequestQueueTimeout):
                clients[2][2]()
            require(not self.calls(clients[2][1]), "expired queued request ran UDF")
            state = self.checkpoint("queue_expired")
            require(state["result_delivery"]["active_results"] == 2, "queued cleanup changed active result owners")
            for _, token, _ in clients[:2]:
                self.release(token)
            for future in active:
                future.result(timeout=30)
        self.quiescent("queue_recovered")

    def result_pressure(self):
        results = [self.produce(api="sql"), self.produce(api="relation")]
        try:
            with self.client("relation") as (_, token, execute):
                executed = self.request_count("executed_requests")
                for _ in range(3):
                    with expect(ResultDeliveryFull, reason="slots", execution_started=False):
                        execute()
                    require(self.request_count("active_requests") == 0, "slot refusal kept request capacity")
                require(
                    not self.calls(token) and self.request_count("executed_requests") == executed,
                    "slot refusal executed UDF",
                )
                self.checkpoint("slow_consumer_slots")
                self.consume(results.pop())
                self.consume(execute())
                require(len(self.calls(token)) == 1, "slot retry replayed execution")
        finally:
            for result in results:
                result.close()
        self.quiescent("slot_pressure_recovered")

        result = self.produce("large", api="relation")
        table = result.take()
        view = None
        result.close()
        try:
            snapshot = self.checkpoint("slow_consumer_view")
            require(snapshot["result_delivery"]["active_results"] == 0, "final handoff kept slot")
            require(48 * 1024 <= snapshot["result_delivery"]["exported_bytes"] <= 64 * 1024, "exported view uncharged")
            view = table["worker_pid"].chunk(0).to_numpy(zero_copy_only=True)
            table = None
            require(view.tolist() == [int(self.initializations()[-1])], "NumPy view lost worker identity")
            self.checkpoint("slow_consumer_numpy_view")
            require(
                self.runtime.resource_snapshot()["result_delivery"]["exported_bytes"]
                == snapshot["result_delivery"]["exported_bytes"],
                "NumPy view lost result accounting",
            )
            with self.client("sql", "large") as (_, token, execute):
                with expect(ResultDeliveryFull, reason="bytes", execution_started=True):
                    self.execute_with_slot_retry(execute, token)
                require(len(self.calls(token)) == 1, "byte refusal replayed UDF")
            require(len(self.calls(token)) == 1, "cursor teardown replayed refused execution")
        finally:
            # Neither the table nor an exported zero-copy view may be kept by
            # the driver at the recovery checkpoint.
            del table, view
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
        with self.client("relation", "gated") as (cursor, token, execute):
            with ThreadPoolExecutor(max_workers=1) as threads:
                future = threads.submit(execute)
                try:
                    wait_for(lambda: self.entered(token), "gated UDF did not start")
                    cursor.interrupt()
                    with expect(RequestCancelled):
                        future.result(timeout=30)
                finally:
                    self.release(token)
                    cursor.interrupt()
        self.quiescent("execution_cancelled")
        self.recover()

    def execution_expiry(self):
        with self.client("sql", "gated") as (cursor, token, execute), ThreadPoolExecutor(1) as threads:
            future = threads.submit(execute)
            try:

                def entered():
                    if self.entered(token):
                        return True
                    if future.done():
                        future.result()
                    return False

                wait_for(entered, "deadline fixture did not reach the warm model")
                with expect(RequestExecutionTimeout):
                    future.result(timeout=self.execution_timeout + 30)
                require(len(self.calls(token)) == 1, "execution deadline replayed UDF")
            finally:
                self.release(token)
                cursor.interrupt()
        self.quiescent("execution_expired")
        self.recover()

    def zero_deadline(self):
        for api in ("sql", "relation"):
            with self.client(api) as (_, token, execute):
                with expect(RequestExecutionTimeout):
                    execute()
                require(not self.calls(token), "zero execution deadline ran UDF")
        require(not self.initializations(), "zero execution deadline initialized a worker")
        self.quiescent("execution_expired")

    def failures(self):
        for mode in ("udf_error", "worker_exit"):
            before = len(self.initializations())
            worker_metrics = self.runtime.resource_snapshot()["worker_failures"]
            with self.client("relation" if mode == "udf_error" else "sql", mode) as (_, token, execute):
                failed_before = self.runtime.resource_snapshot()["request_admission"]["failed_executions"]
                with expect(Exception, "planned serving UDF failure" if mode == "udf_error" else None):
                    execute()
                require(len(self.calls(token)) == 1, f"{mode}: failed UDF not run exactly once")
                require(
                    self.runtime.resource_snapshot()["request_admission"]["failed_executions"] == failed_before + 1,
                    "execution failure not counted",
                )
            self.quiescent(f"{mode}_cleaned")
            self.recover()
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
            field = "execution_errors" if mode == "udf_error" else "worker_losses"
            worker_metrics[field] += 1
            require(
                self.runtime.resource_snapshot()["worker_failures"] == worker_metrics,
                f"{mode}: worker outcome not counted exactly once across recovery",
            )

    def close(self):
        try:
            self.runtime.close(timeout=30, kill=True)
        finally:
            self.connection.close()


def request_metrics(before, after):
    """Phase totals from public snapshots; never infer per-request percentiles."""
    return {
        field: after["request_admission"][field] - before["request_admission"][field]
        for field in (
            "admitted_requests",
            "executed_requests",
            "completed_requests",
            "rejected_requests",
            "queue_wait_seconds",
            "execution_seconds",
            "cleanup_seconds",
        )
    }


def deadline_checks(directory):
    # Public session configuration is immutable. Exercise short queue expiry
    # and zero execution deadlines in separate sessions with separate counters.
    reports = {}
    for name, options, action in (
        ("queue", {"queue_timeout": 2}, "queue_expiry"),
        ("execution", {"execution_timeout": 0}, "zero_deadline"),
    ):
        path = directory / name
        path.mkdir()
        scenario = Scenario(path, **options)
        try:
            getattr(scenario, action)()
            scenario.runtime.close(timeout=30)
            reports[name] = {
                "configuration": options,
                "initializations": len(scenario.initializations()),
                "checkpoints": scenario.checkpoints,
                "closed": scenario.quiescent("closed", closed=True),
            }
        finally:
            scenario.close()
    return reports


def run_acceptance(directory, *, requests=20, concurrency=4):
    if type(requests) is not int or requests < 2 or type(concurrency) is not int or not 1 <= concurrency <= 4:
        raise ValueError("requests must be >= 2; concurrency must be between 1 and 4")
    scenario = Scenario(directory)
    try:
        require(not scenario.initializations(), "registration eagerly initialized the model")
        before = scenario.runtime.resource_snapshot()
        cold = scenario.query()
        phase_metrics = {"cold": request_metrics(before, scenario.runtime.resource_snapshot())}
        cold_initializations = len(scenario.initializations())
        require(cold_initializations == 1, "cold request did not initialize one worker")
        started = time.monotonic()
        scenario.model.prewarm()
        prewarm_seconds = time.monotonic() - started
        before = scenario.runtime.resource_snapshot()
        warm = [scenario.query(api=("relation", "sql")[i % 2]) for i in range(requests)]
        phase_metrics["warm"] = request_metrics(before, scenario.quiescent("sequential_recovered"))
        before = scenario.runtime.resource_snapshot()
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=concurrency) as threads:
            mixed = list(
                threads.map(
                    lambda i: scenario.query(
                        "analysis" if i % 4 == 0 else "short",
                        32 if i % 4 == 0 else 1,
                        api=("sql", "relation")[(i + i // 4) % 2],
                    ),
                    range(requests),
                )
            )
        mixed_seconds = time.monotonic() - started
        healthy_additional_initializations = len(scenario.initializations()) - cold_initializations
        require(healthy_additional_initializations == 0, "healthy requests reinitialized model")
        require(all(s["worker_pids"] == cold["worker_pids"] for s in [*warm, *mixed]), "healthy model identity changed")
        phase_metrics["mixed"] = request_metrics(before, scenario.quiescent("mixed_recovered"))
        scenario.ingress()
        scenario.result_pressure()
        scenario.cancellation()
        scenario.failures()
        before_close = scenario.quiescent("before_close")
        scenario.runtime.drain()
        with scenario.client() as (_, token, execute), expect(RuntimeError, "draining"):
            execute()
        require(not scenario.calls(token), "drained session ran UDF")
        scenario.runtime.close(timeout=30)
        scenario.quiescent("closed", closed=True)
        groups = {
            "cold": [cold],
            "warm": warm,
            "mixed_short": [s for s in mixed if s["kind"] == "short"],
            "mixed_analysis": [s for s in mixed if s["kind"] == "analysis"],
        }
        deadlines = deadline_checks(directory)
        return {
            "schema_version": 2,
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
                "query_apis": ["sql", "relation"],
                "queue_timeout_seconds": 30,
                "execution_timeout_seconds": 30,
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
                    for field in ("latency_seconds", "delivery_seconds")
                }
                for name, samples in groups.items()
            },
            "phase_request_metrics": phase_metrics,
            "load_api_counts": {
                api: sum(s["api"] == api for s in [cold, *warm, *mixed]) for api in ("sql", "relation")
            },
            "mixed_throughput_requests_per_second": len(mixed) / mixed_seconds,
            "load_result_slot_refusals": sum(s["result_slot_refusals"] for s in [cold, *warm, *mixed]),
            "before_close": before_close,
            "checkpoints": scenario.checkpoints,
            "deadline_sessions": deadlines,
            "scope": "Synthetic text/RGB CPU features; public SQL/Relation APIs; materialized results. Latency and mixed throughput include cursor creation, binding, admission, consumption and cursor close. Queue/execution/cleanup metrics are phase totals, not per-request distributions. No network sends, native streaming, process RSS bound, learned embedding quality, or GPU claim.",
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

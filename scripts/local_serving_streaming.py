# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Streaming phases for the installed CPU/CUDA serving soak.

Loaded only by its worker; the watchdog stays independent of native locks.
Every client uses the existing public Scenario session and registered model.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from functools import partial

from vane.execution.request_admission import RequestCancelled, RequestExecutionTimeout
from vane.execution.result_delivery import ResultDeliveryClosed, ResultDeliveryTimeout


class StreamingChecks:
    rows = 48
    rows_per_batch = 16

    def __init__(self, scenario, acceptance):
        self.scenario = scenario
        self.require = acceptance["require"]
        self.expect = acceptance["expect"]
        self.wait_for = acceptance["wait_for"]
        self._lock = threading.Lock()
        self.sampled_peak_bytes = 0

    def delivery(self):
        state = self.scenario.runtime.resource_snapshot()["result_delivery"]
        self.require(0 <= state["usage_bytes"] <= state["limit_bytes"], "stream exceeded delivery byte budget")
        with self._lock:
            self.sampled_peak_bytes = max(self.sampled_peak_bytes, state["usage_bytes"])
        return state

    def query(self, *, api="sql"):
        started = time.monotonic()
        first_batch = None
        batches = logical_bytes = 0

        def observe(table):
            nonlocal first_batch, batches, logical_bytes
            if first_batch is None:
                first_batch = time.monotonic() - started
            self.require(0 < len(table) <= self.rows_per_batch, "unexpected stream batch size")
            self.require(all(len(value) == 2048 for value in table["padding"].to_pylist()), "missing stream payload")
            batches += 1
            logical_bytes += table.nbytes
            self.delivery()

        with self.scenario.client(api, "stream", self.rows) as (_, token, execute):
            result, refusals = self.scenario.execute_with_slot_retry(
                partial(execute, stream=True, rows_per_batch=self.rows_per_batch), token
            )
            consumption_started = time.monotonic()
            pids = self.scenario.consume(result, self.rows, observe=observe)
        self.require(logical_bytes > self.delivery()["limit_bytes"], "stream did not exceed full-result budget")
        self.require(batches > 1 and first_batch is not None, "stream was not consumed incrementally")
        return {
            "api": api,
            "delivery": "streaming",
            "kind": "analysis",
            "rows": self.rows,
            "batches": batches,
            "logical_bytes": logical_bytes,
            "first_batch_seconds": first_batch,
            "consumption_seconds": time.monotonic() - consumption_started,
            "latency_seconds": time.monotonic() - started,
            **result.timing_snapshot(),
            "worker_pids": pids,
            "result_slot_refusals": refusals,
        }

    @contextmanager
    def blocked_stream(self, api, *, delivery_timeout=None):
        """Yield only after a retained Arrow buffer has blocked another pull."""
        with self.scenario.client(api, "stream", self.rows) as (cursor, token, execute):
            result = execute(stream=True, rows_per_batch=self.rows_per_batch, delivery_timeout=delivery_timeout)
            table = view = pending = None
            held = []
            with ThreadPoolExecutor(max_workers=1) as reader:
                try:
                    table = result.take()
                    self.require(table["id"].to_pylist() == list(range(self.rows_per_batch)), "wrong first batch")
                    charged = self.delivery()["exported_bytes"]
                    self.require(charged > self.delivery()["limit_bytes"] // 2, "batch cannot force byte pressure")

                    def read_next():
                        try:
                            next_table = result.take()
                        except (
                            RequestCancelled,
                            RequestExecutionTimeout,
                            ResultDeliveryClosed,
                            ResultDeliveryTimeout,
                        ) as error:
                            # Expected outcomes are scalars, not cached Future
                            # exceptions whose tracebacks retain request frames.
                            return type(error)
                        self.delivery()
                        # Return scalars: a Future retaining an Arrow result
                        # would itself keep the next byte reservation alive.
                        return next_table["id"].to_pylist()

                    pending = reader.submit(read_next)

                    def waiting():
                        if pending.done():
                            pending.result()
                            raise AssertionError("stream advanced while the first batch was retained")
                        return self.delivery()["waiting_byte_results"] == 1

                    self.wait_for(waiting, "stream did not wait for retained bytes")
                    with self.expect(FutureTimeoutError):
                        pending.result(timeout=0.05)
                    # Transfer the same charge to a zero-copy view. No table,
                    # Future result or driver sample may hide an extra owner.
                    view = table["worker_pid"].chunk(0).to_numpy(zero_copy_only=True)
                    table = None
                    self.require(self.delivery()["exported_bytes"] == charged, "NumPy view lost its byte charge")
                    self.require(
                        view.tolist() == [int(self.scenario.initializations()[-1])] * self.rows_per_batch,
                        "NumPy view lost worker identity",
                    )
                    state = self.scenario.runtime.resource_snapshot()
                    self.require(state["request_admission"]["active_requests"] == 1, "stream lost request admission")
                    self.require(state["result_delivery"]["active_results"] == 1, "stream lost result admission")
                    # Mutable owner holder lets callers release the view before
                    # awaiting the pending pull; this generator retains none.
                    held.append(view)
                    view = None
                    yield cursor, token, result, pending, held
                finally:
                    table = view = None
                    held.clear()
                    # Interrupt before joining the reader. result.cancel()
                    # can raise while take() still owns cleanup, masking the
                    # original failure and skipping the wait and close below.
                    cursor.interrupt()
                    if pending is not None:
                        try:
                            pending.result(timeout=30)
                        except Exception:
                            pass
                    pending = None
                    result.close()

    def release_pressure(self, *, api):
        with self.blocked_stream(api) as (_, _, result, pending, held):
            started = time.monotonic()
            held.clear()
            ids = pending.result(timeout=30)
            self.require(
                ids == list(range(self.rows_per_batch, 2 * self.rows_per_batch)), "released stream skipped a batch"
            )
            resumed_seconds = time.monotonic() - started
            # Closing here also exercises discarding a partially read stream.
            result.close()
        self.scenario.quiescent("stream_pressure_recovered")
        return {"api": api, "resume_seconds": resumed_seconds, "observed_byte_wait": True}

    def interrupt_pressure(self, action, *, api):
        errors = {
            "cancel": RequestCancelled,
            "delivery_expiry": ResultDeliveryTimeout,
            "execution_expiry": RequestExecutionTimeout,
        }
        before = len(self.scenario.initializations())
        with self.blocked_stream(api, delivery_timeout=2 if action == "delivery_expiry" else None) as (
            cursor,
            _,
            result,
            pending,
            held,
        ):
            started = time.monotonic()
            if action == "cancel":
                cursor.interrupt()
            outcome = pending.result(timeout=30)
            self.require(outcome is errors[action], f"{action}: unexpected stream outcome {outcome}")
            result.close()
            elapsed = time.monotonic() - started
            self.require(
                held[0].tolist() == [int(self.scenario.initializations()[-1])] * self.rows_per_batch,
                "stream cancellation invalidated an exported view",
            )
            self.require(self.delivery()["exported_bytes"] > 0, "cancelled stream dropped exported byte accounting")
        self.scenario.quiescent("stream_" + action + "_cleaned")
        self.scenario.recover()
        replacements = len(self.scenario.initializations()) - before
        self.require(replacements in (0, 1), "stream control churned model workers")
        self.scenario.quiescent("stream_" + action + "_recovered")
        return {
            "api": api,
            "observed_byte_wait": True,
            "wait_and_cleanup_seconds": elapsed,
            "worker_replacements": replacements,
        }

    def shutdown(self):
        with self.blocked_stream("relation") as (_, _, result, pending, held):
            started = time.monotonic()
            self.scenario.runtime.drain()
            with self.scenario.client() as (_, token, execute), self.expect(RuntimeError, "draining"):
                execute()
            self.require(not self.scenario.calls(token), "drained runtime executed a UDF")
            self.scenario.runtime.close(timeout=30, kill=True)
            self.require(pending.result(timeout=30) is ResultDeliveryClosed, "runtime close did not close the stream")
            result.close()
            elapsed = time.monotonic() - started
            state = self.scenario.runtime.resource_snapshot()
            self.require(state["request_admission"]["active_requests"] == 0, "shutdown kept request admission")
            self.require(state["result_delivery"]["active_results"] == 0, "shutdown kept result admission")
            self.require(self.delivery()["exported_bytes"] > 0, "shutdown forgot the retained consumer view")
            self.require(
                held[0].tolist() == [int(self.scenario.initializations()[-1])] * self.rows_per_batch,
                "shutdown invalidated the consumer view",
            )
        self.scenario.close()
        return {"observed_byte_wait": True, "cleanup_seconds": elapsed}

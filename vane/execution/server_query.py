# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""An asynchronous remote query owner; control calls only read metadata."""

from __future__ import annotations

import threading
import time
from typing import Any

from vane.execution.cleanup_deadline import cleanup_deadline
from vane.execution.request_admission import RequestCancelled, RequestExecutionTimeout, RequestQueueTimeout
from vane.execution.result_consumer import NativeResultConsumer
from vane.execution.result_delivery import ResultDeliveryTimeout


def query_error(error: BaseException) -> dict[str, str]:
    codes = {
        RequestCancelled: "CANCELLED",
        RequestExecutionTimeout: "EXECUTION_TIMEOUT",
        RequestQueueTimeout: "ADMISSION_TIMEOUT",
        ResultDeliveryTimeout: "DELIVERY_TIMEOUT",
    }
    code = next((v for t, v in codes.items() if isinstance(error, t)), "QUERY_FAILED")
    return {"code": code, "message": str(error)[:4096]}


class ServerQuery:
    def __init__(self, sequence: int, fingerprint: str, owner: Any, gateway: Any, cleanup_timeout: float) -> None:
        self.sequence = sequence
        self.fingerprint = fingerprint
        self.owner = owner
        self.consumer = NativeResultConsumer(gateway)
        self.cleanup_timeout = cleanup_timeout
        self.lock = threading.RLock()
        self.wakeup = threading.Event()
        self.done = threading.Event()
        self.state = "PREPARING"
        self.error: dict[str, str] | None = None
        self.cleanup_error = ""
        self.context: Any = None
        self.cursor: Any = None
        self.result: Any = None
        self.cancel_requested = False
        self.finish_requested = False
        self.release_requested = False

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            value = {
                "query_id": self.sequence,
                "state": self.state,
                "error": self.error,
                "cleaned": self.done.is_set(),
                "cleanup_pending": bool(self.cleanup_error),
            }
            if self.state == "READY":
                value["result"] = self.consumer.descriptor
            return value

    def request(self, operation: str) -> None:
        with self.lock:
            if operation == "finish":
                self.finish_requested = True
            else:
                self.cancel_requested = True
                self.release_requested |= operation == "close"
            self.wakeup.set()

    def publish(self, context: Any) -> None:
        with self.lock:
            if context is not None:
                self.context = context
                canceled = self.cancel_requested
            else:
                canceled = False
        if canceled:
            context.cancel()

    def run(self, sql: str, options: Any, rows_per_batch: int) -> None:
        # An independent cursor permits concurrent SQL without sharing a
        # native active-result slot. Session cleanup waits for this owner.
        interrupter = threading.Thread(target=self._interrupt, name="vane-remote-cancel", daemon=True)
        try:
            interrupter.start()
            self.cursor = self.owner._connection.cursor()
            self.result = self.owner.submit(
                self.cursor,
                sql,
                None,
                options,
                rows_per_batch,
                {},
                self.publish,
                lambda: None,
                consumer=self.consumer,
            )
            with self.lock:
                self.state = "READY"
            while True:
                with self.result._runtime._condition:
                    self.result._check_locked()
                self.context.check()
                if self.finish_requested and self.context.complete_external(self.result):
                    with self.lock:
                        self.state = "SUCCEEDED"
                    break
                self.wakeup.wait(0.02)
                self.wakeup.clear()
        except BaseException as error:
            with self.lock:
                self.error = query_error(error)
                self.state = "CANCELED" if isinstance(error, RequestCancelled) else "FAILED"
        finally:
            # Cleanup retries retain the query, result slot, native cursor and
            # gateway capability. They never depend on re-reading a failed RPC.
            while True:
                try:
                    with cleanup_deadline(time.monotonic() + self.cleanup_timeout):
                        if self.result is not None:
                            self.result._close_stream(self.cleanup_timeout)
                        elif self.context is not None:
                            if self.context._result is not None:
                                self.context._result.abort_preparation()
                            else:
                                self.context.close()
                                self.context.retire(lambda: None)
                        self.consumer.close()
                        if self.cursor is not None:
                            self.cursor.close()
                    with self.lock:
                        self.cleanup_error = ""
                    self.done.set()
                    self.wakeup.set()
                    break
                except BaseException as error:
                    with self.lock:
                        self.cleanup_error = str(error)[:4096]
                    time.sleep(0.05)
            if interrupter.ident is not None:
                interrupter.join()

    def _interrupt(self) -> None:
        while not self.done.wait(0.02):
            with self.lock:
                cancel = self.cancel_requested
                context, cursor = self.context, self.cursor
            if cancel:
                # interrupt() is independent of the native planning lock.
                if cursor is not None:
                    cursor.interrupt()
                if context is not None:
                    context.cancel()
                return

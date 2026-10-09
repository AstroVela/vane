# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Flight client: SQL control in Python, bounded result transfer in native code."""

from __future__ import annotations

import base64
import json
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import pyarrow as pa
import pyarrow.flight as flight

from vane.execution.batch_lease import BatchLease
from vane.execution.cleanup_deadline import cleanup_deadline, cleanup_timeout
from vane.execution.direct_exchange import DirectExchangeLimits
from vane.execution.query_options import QueryExecutionOptions
from vane.execution.query_runtime import QueryResources
from vane.execution.request_admission import RequestCancelled, RequestExecutionTimeout, RequestQueueTimeout, _timeout
from vane.execution.result_delivery import (
    QueryResult,
    ResultDeliveryLimits,
    ResultDeliveryTimeout,
    RuntimeResultDelivery,
)
from vane.execution.server_session import SessionError

_ERRORS: dict[str, type[Exception]] = {
    "CANCELLED": RequestCancelled,
    "EXECUTION_TIMEOUT": RequestExecutionTimeout,
    "ADMISSION_TIMEOUT": RequestQueueTimeout,
    "DELIVERY_TIMEOUT": ResultDeliveryTimeout,
    "QUERY_FAILED": RuntimeError,
}


def _check_error(status: dict[str, Any]) -> None:
    error = status.get("error")
    if error:
        raise _ERRORS.get(error["code"], RuntimeError)(error["message"])


class Client:
    """One leased server session. This process does not initialize Ray.

    query() returns the same bounded QueryResult used by embedded execution.
    submit() returns a control handle for asynchronous status/cancel/result.
    An uncertain Execute remains owned here; the next submit first confirms
    that same sequence and SQL. Failed close calls retain owners for retry.
    """

    def __init__(
        self,
        location: Any,
        *,
        token: str,
        tls_root_certs: bytes | None = None,
        execution: str = "pipelined",
        resources: QueryResources | None = None,
        result_limits: ResultDeliveryLimits = ResultDeliveryLimits(4, 64 << 20),
        rpc_timeout: float = 5,
    ) -> None:
        self.rpc_timeout = _timeout(rpc_timeout, "RPC timeout")
        if self.rpc_timeout <= 0:
            raise ValueError("RPC timeout must be positive")
        # Match native result/exchange connections: use the explicit endpoint
        # directly, independent of HTTP proxy settings in the application.
        self._flight = flight.FlightClient(
            location, tls_root_certs=tls_root_certs, generic_options=[("grpc.enable_http_proxy", 0)]
        )
        self._roots = tls_root_certs or b""
        self._headers = [(b"authorization", ("Bearer " + token).encode("ascii"))]
        self._delivery = RuntimeResultDelivery(result_limits)
        self._lock = threading.RLock()
        self._submit_lock = threading.Lock()
        self._queries: dict[int, RemoteQuery] = {}
        self._sequence = 0
        self._unconfirmed: RemoteQuery | None = None
        self._closing = False
        self._closed = False
        self._error: BaseException | None = None
        self._stop = threading.Event()
        started = time.monotonic()
        try:
            handle = self._call(
                "session.open", execution=execution, **({"resources": asdict(resources)} if resources else {})
            )
        except BaseException:
            self._flight.close()
            raise
        self.identity = {key: handle[key] for key in ("server_id", "session_id")}
        self._lease = float(handle["lease_seconds"])
        self._expires = started + self._lease
        self._heartbeat = threading.Thread(target=self._renew, name="vane-client-lease", daemon=True)
        self._heartbeat.start()

    def _call(self, operation: str, **fields: Any) -> dict[str, Any]:
        body = json.dumps({"protocol": 1, **fields}, allow_nan=False, separators=(",", ":")).encode()
        if len(body) > 65536:
            raise ValueError("control request exceeds 65536 bytes")
        options = flight.FlightCallOptions(timeout=cleanup_timeout(self.rpc_timeout), headers=self._headers)
        replies = self._flight.do_action(flight.Action("vane." + operation, body), options)
        reply = next(replies)
        if len(reply.body) > 65536:
            raise ValueError("control response exceeds 65536 bytes")
        value = json.loads(reply.body.to_pybytes())
        if next(replies, None) is not None or value.get("protocol") != 1:
            raise ValueError("invalid control response")
        if not value["ok"]:
            raise SessionError(value["error"]["code"], value["error"]["message"])
        return value["result"]

    def _renew(self) -> None:
        while not self._stop.wait(min(self._lease / 3, 1)):
            started = time.monotonic()
            try:
                with cleanup_deadline(self._expires):
                    self._call("session.renew", **self.identity)
                self._expires = started + self._lease
            except BaseException as error:
                if time.monotonic() < self._expires and not isinstance(error, SessionError):
                    continue
                with self._lock:
                    self._error = SessionError("SESSION_EXPIRED", "client could not renew the session lease")
                    queries = tuple(self._queries.values())
                for query in queries:
                    if query.reader is not None:
                        query.reader.fail(self._error)
                return

    def check(self) -> None:
        if self._error is not None:
            raise self._error

    def submit(
        self, sql: str, *, options: QueryExecutionOptions | None = None, rows_per_batch: int = 1024
    ) -> RemoteQuery:
        with self._submit_lock:
            self.check()
            if self._closing:
                raise RuntimeError("client is closing")
            if self._unconfirmed is not None:
                self._execute_pending()
                self._unconfirmed = None
            sequence = self._sequence + 1
            query = RemoteQuery(self, sequence, sql, options, rows_per_batch)
            with self._lock:
                self._queries[sequence] = query
                self._unconfirmed = query
                self._sequence = sequence
            self._execute_pending()
            self._unconfirmed = None
            return query

    def _execute_pending(self) -> None:
        query = self._unconfirmed
        assert query is not None
        try:
            query.execute()
        except (ValueError, pa.ArrowInvalid, SessionError) as error:
            # These refusals occur before reservation. Transport errors are
            # deliberately excluded: their Execute may already own resources.
            if isinstance(error, SessionError) and error.code not in {"QUERY_CAPACITY", "QUERY_UNAVAILABLE"}:
                raise
            with self._lock:
                self._queries.pop(query.query_id, None)
                self._sequence -= 1
                self._unconfirmed = None
            raise

    def query(
        self, sql: str, *, options: QueryExecutionOptions | None = None, rows_per_batch: int = 1024
    ) -> QueryResult:
        query = self.submit(sql, options=options, rows_per_batch=rows_per_batch)
        try:
            return query.result()
        except BaseException as error:
            try:
                query.close()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise

    def resource_snapshot(self) -> dict[str, Any]:
        return self._delivery.snapshot()

    def close(self, *, timeout: float = 10) -> None:
        if self._closed:
            return
        self._closing = True
        with cleanup_deadline(time.monotonic() + _timeout(timeout, "client close timeout")):
            self._delivery.close(timeout=cleanup_timeout(timeout))
            # The session is the owner even if an Execute reply was lost.
            while True:
                try:
                    if self._call("session.close", **self.identity)["state"] == "CLOSED":
                        break
                except SessionError as error:
                    if error.code not in {"SESSION_EXPIRED", "SERVER_CHANGED"}:
                        raise
                    break
                self._stop.wait(min(0.02, cleanup_timeout(timeout)))
            self._stop.set()
            self._heartbeat.join(cleanup_timeout(timeout))
            if self._heartbeat.is_alive():
                raise TimeoutError("client heartbeat is stopping; retry close")
            self._flight.close()
            with self._lock:
                self._queries.clear()
                self._closed = True

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class RemoteQuery:
    def __init__(
        self, client: Client, sequence: int, sql: str, options: QueryExecutionOptions | None, rows: int
    ) -> None:
        self.client = client
        self.query_id = sequence
        self._payload = {
            "sequence": sequence,
            "sql": sql,
            "options": None if options is None else options.to_dict(),
            "rows_per_batch": rows,
        }
        self.reader: _RemoteStream | None = None
        self._result: QueryResult | None = None
        self._closed = False
        self._lock = threading.RLock()
        self._cancelled = False
        self._failure: tuple[type[BaseException], tuple[Any, ...]] | None = None

    def execute(self) -> dict[str, Any]:
        return self.client._call("query.execute", **self.client.identity, **self._payload)

    def _action(self, operation: str) -> dict[str, Any]:
        return self.client._call("query." + operation, **self.client.identity, query_id=self.query_id)

    def status(self) -> dict[str, Any]:
        return self._action("status")

    def cancel(self) -> None:
        self._action("cancel")
        self._cancelled = True
        if self.reader is not None:
            self.reader.fail(RequestCancelled("query canceled"))

    def check(self) -> None:
        self.client.check()
        if self._cancelled:
            raise RequestCancelled("query canceled")
        if self._failure is not None:
            kind, arguments = self._failure
            raise kind(*arguments)

    def result(self) -> QueryResult:
        with self._lock:
            if self._closed:
                raise RuntimeError("query handle is closed")
            if self._result is not None:
                return self._result
            while True:
                self.check()
                status = self.status()
                _check_error(status)
                if status["state"] == "READY":
                    break
                if status["state"] in {"CLOSED", "SUCCEEDED"}:
                    raise RuntimeError("result stream has already completed")
                time.sleep(0.01)
            result = self.client._delivery.begin()
            self._result = result
            self.reader = _RemoteStream(self)
            result.start_stream(self.reader)
            try:
                self.reader.open(status["result"])
                result.query_id = str(self.query_id)
                result.schema = self.reader.schema
                result.completion_status = "streaming"
                result.ready(delivery_timeout=None)  # The server owns this deadline.
                return result
            except BaseException:
                result.abort_preparation()
                raise

    def _close_remote(self) -> None:
        while not self._closed:
            try:
                status = self._action("close")
                if status["state"] != "CLOSED":
                    time.sleep(min(0.02, cleanup_timeout(self.client.rpc_timeout)))
                    continue
            except SessionError as error:
                if error.code not in {"SESSION_EXPIRED", "SERVER_CHANGED", "QUERY_RETIRED"}:
                    raise
            self._closed = True
            with self.client._lock:
                self.client._queries.pop(self.query_id, None)

    def close(self) -> None:
        with cleanup_deadline(time.monotonic() + self.client.rpc_timeout):
            if self._result is not None:
                self._result.close()
            else:
                self._close_remote()


class _RemoteStream:
    def __init__(self, query: RemoteQuery) -> None:
        self.query = query
        self.flight: Any = None
        self.channel: Any = None
        self.schema: Any = None
        self.names: list[str] = []
        self.closed = False
        self._stop = threading.Event()
        self._watcher: threading.Thread | None = None

    def open(self, descriptor: dict[str, Any]) -> None:
        from vane._native import execution_plan
        from vane._native import execution_runtime as native

        if descriptor["engine"] != execution_plan.engine_identity():
            raise RuntimeError("client and server native engine identities differ")
        limits = DirectExchangeLimits(**descriptor["limits"])
        # Bounds the separate native receive window and codec reservation.
        if limits.window_bytes > self.query.client._delivery.limits.max_bytes:
            raise ValueError("client result budget cannot hold the native receive window")
        encoded = base64.b64decode(descriptor["schema"], validate=True)
        self.names = descriptor["names"]
        self.schema = native.arrow_schema(encoded, self.names)
        self.channel = native.DirectChannel(
            encoded,
            native.DirectLimits(limits.window_bytes, limits.frame_bytes, limits.frame_rows, limits.frame_slots),
            1,
            ["client"],
        )
        self.channel.add_producer("server")
        self.channel.seal_producers()
        self.flight = native.DirectFlight(
            "127.0.0.1", "127.0.0.1", 1, native.DirectFlight.staging_per_link(limits.frame_bytes), limits.frame_bytes
        )
        self.flight.subscribe(
            descriptor["location"],
            descriptor["ticket"],
            self.channel,
            "server",
            descriptor["timeout"],
            self.query.client._roots.decode("ascii"),
        )
        self._watcher = threading.Thread(target=self._watch, name="vane-client-query", daemon=True)
        self._watcher.start()

    def fail(self, error: BaseException) -> None:
        kind, arguments = type(error), error.args
        self.query._failure = (kind, arguments)
        if self.flight is not None:
            self.flight.cancel(str(error))
        result = self.query._result
        if result is not None:
            result.request_cancelled(lambda: kind(*arguments))

    def _watch(self) -> None:
        while not self._stop.wait(0.05):
            try:
                self.query.check()
                _check_error(self.query.status())
            except BaseException as error:
                if not self._stop.is_set():
                    self.fail(error)
                return

    def guard(self) -> None:
        from vane._native.execution_runtime import check_entry

        check_entry()

    def check(self) -> None:
        self.query.check()

    def commit(self, operation: Callable[[], None]) -> None:
        self.check()
        operation()

    def read(self, result: QueryResult) -> bool:
        while True:
            self.check()
            try:
                state, batch = self.channel.poll("client")
            except BaseException:
                # Native transport errors can arrive before the coordinator
                # caches its typed outcome. Give the control owner a bounded
                # chance to report the persistent failure.
                deadline = time.monotonic() + self.query.client.rpc_timeout
                while time.monotonic() < deadline:
                    self.check()
                    _check_error(self.query.status())
                    time.sleep(0.01)
                raise
            if state == "data":
                try:
                    arrow = batch.to_arrow(self.names)
                    size = BatchLease.size(arrow)
                    payload = BatchLease(result.own_buffer(size))
                    result.hold(payload)
                    payload.build(arrow, size)
                finally:
                    batch.close()
                    batch = None
                return True
            if state == "end":
                while True:
                    self.check()
                    status = self.query._action("finish")
                    _check_error(status)
                    if status["state"] == "SUCCEEDED" and status["cleaned"]:
                        result.completion_status = "ok"
                        return False
                    time.sleep(0.01)
            if state == "closed":
                self.check()
                _check_error(self.query.status())
                raise RuntimeError("remote result closed before FINISH")
            time.sleep(0.002)

    def close(self) -> None:
        if self.closed:
            return
        self._stop.set()
        if self.flight is not None:
            self.flight.close()
        if self._watcher is not None and self._watcher is not threading.current_thread():
            self._watcher.join(cleanup_timeout(self.query.client.rpc_timeout))
            if self._watcher.is_alive():
                raise TimeoutError("remote status monitor is stopping; retry close")
        with cleanup_deadline(time.monotonic() + self.query.client.rpc_timeout):
            self.query._close_remote()
        self.closed = True

    def cleanup_pending(self) -> bool:
        return not self.closed

    def retire(self, release_result: Callable[[], None]) -> None:
        if not self.closed:
            raise RuntimeError("remote result cleanup is pending")
        release_result()

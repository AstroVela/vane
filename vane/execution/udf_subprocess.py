# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Subprocess + shared-memory UDF executor."""

from __future__ import annotations

import atexit
import hashlib
import math
import os
import queue
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
import weakref
from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, cast

import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future
    from multiprocessing import shared_memory

from vane import pickle as vane_pickle
from vane.execution._common import ensure_table as _ensure_table
from vane.execution.ref_bundle import (
    REF_BUNDLE_RESULT_MARKER,
    SUBMIT_RESULT_MARKER,
    _create_shm,
    _open_existing_shm,
    _unlink_shm,
    can_admit_local_shm_ref_output_submit,
    cancel_local_shm_input_lease,
    consume_local_shm_input_lease,
    create_local_shm_input_lease,
    estimate_local_shm_ref_bundle_ipc_size,
    local_shm_ref_budget_snapshot,
    make_local_ref_bundle_worker_payload,
    make_local_shm_ref_bundle_result,
    make_local_shm_ref_bundle_result_from_descriptor,
    payload_requests_local_ref_bundle_output,
    register_local_shm_ref_budget_wakeup,
    release_local_shm_output_grant,
    release_local_shm_ref_bundle_descriptor,
    request_local_shm_output_grant,
    wake_local_shm_ref_budget_waiters,
)
from vane.execution.udf_admission import (
    AdmissionExecutorMixin,
    AdmissionLease,
    LocalExecutionSlotPool,
    LocalSlotAdmissionAuthority,
)
from vane.execution.udf_lifecycle import (
    ExecutionCancellationScope,
    ExecutionCancelledError,
)
from vane.execution.udf_threading import (
    worker_thread_env as _worker_thread_env,
)
from vane.execution.unified_executor import UDFExecutor as BaseUDFExecutor
from vane.runners.ray.ray_env import build_explicit_session_process_env

_MSG_READY = 0x01
_MSG_SUBMIT = 0x02
_MSG_FINISHED = 0x03
_MSG_CLOSE = 0x04
_MSG_OK = 0x05
_MSG_ERROR = 0x06
_MSG_ACK = 0x07
_MSG_SUBMIT_REF_BUNDLE = 0x08
_MSG_REF_BUNDLE_RESULT = 0x09
_MSG_INPUT_CONSUMED = 0x0A
_MSG_INPUT_CONSUME_FAILED = 0x0B
_MSG_OUTPUT_GRANT_REQUEST = 0x0C
_MSG_OUTPUT_GRANT_GRANTED = 0x0D
_MSG_OUTPUT_GRANT_CANCELLED = 0x0E
_MSG_OUTPUT_GRANT_RELEASE = 0x0F
_MSG_TASK_CANCELLED = 0x10

_HEADER = struct.Struct("=BI")
_IPC_HEADER = struct.Struct("<Q")
_DEFAULT_SHM_SIZE = 1 << 20
_LOCAL_SHM_OUTPUT_BUDGET_OVERHEAD_BYTES = 1 << 20
_LOCAL_SHM_BLOB_OUTPUT_ROW_BUDGET_BYTES = 1 << 20
_LOCAL_SHM_TEXT_OUTPUT_ROW_BUDGET_BYTES = 4 << 10
_LOCAL_SHM_NESTED_OUTPUT_ROW_BUDGET_BYTES = 16 << 10
_DEFAULT_SUBPROCESS_CONTROL_TIMEOUT_S = 30.0
_DEFAULT_SUBPROCESS_SHUTDOWN_GRACE_S = 5.0
_TENSOR_DTYPE_BYTES = {
    "BOOL": 1,
    "BOOLEAN": 1,
    "TINYINT": 1,
    "UTINYINT": 1,
    "INT8": 1,
    "UINT8": 1,
    "SMALLINT": 2,
    "USMALLINT": 2,
    "INT16": 2,
    "UINT16": 2,
    "INTEGER": 4,
    "UINTEGER": 4,
    "INT": 4,
    "INT32": 4,
    "UINT32": 4,
    "FLOAT": 4,
    "FLOAT4": 4,
    "FLOAT32": 4,
    "BIGINT": 8,
    "UBIGINT": 8,
    "INT64": 8,
    "UINT64": 8,
    "DOUBLE": 8,
    "FLOAT8": 8,
    "FLOAT64": 8,
}


def _subprocess_debug_enabled() -> bool:
    for name in ("VANE_UDF_WORKER_SLOT_DEBUG", "DUCKDB_DISTRIBUTED_DEBUG"):
        value = os.environ.get(name, "")
        if value.strip().lower() not in ("", "0", "false", "no", "off"):
            return True
    return False


def _subprocess_debug_log(message: str) -> None:
    if not _subprocess_debug_enabled():
        return
    try:
        print(f"[vane-udf-worker-slots pid={os.getpid()}] {message}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _debug_submit_log_every() -> int:
    value = os.environ.get("VANE_UDF_TASK_LOG_EVERY_N", "").strip()
    if not value:
        return 0
    parsed = int(value)
    if parsed < 0:
        raise ValueError("VANE_UDF_TASK_LOG_EVERY_N must be non-negative")
    return parsed


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _subprocess_control_timeout_s() -> float:
    return _positive_float_env("VANE_UDF_SUBPROCESS_CONTROL_TIMEOUT_S", _DEFAULT_SUBPROCESS_CONTROL_TIMEOUT_S)


def _subprocess_shutdown_grace_s() -> float:
    return _positive_float_env("VANE_UDF_SUBPROCESS_SHUTDOWN_GRACE_S", _DEFAULT_SUBPROCESS_SHUTDOWN_GRACE_S)


def _should_debug_submit(seq: int) -> bool:
    if not _subprocess_debug_enabled():
        return False
    if seq <= 5:
        return True
    every = _debug_submit_log_every()
    return every > 0 and seq % every == 0


def _product_ints(values: Any) -> int:
    result = 1
    for value in values or []:
        parsed = int(value)
        if parsed <= 0:
            return 0
        result *= parsed
    return result


def _payload_output_row_budget_bytes(payload: dict[str, Any]) -> int:
    total = 0
    for entry in payload.get("output_schema") or []:
        if not isinstance(entry, dict):
            continue
        kind = str(entry.get("kind") or "").strip().lower()
        if kind != "tensor":
            type_name = str(entry.get("type") or "").strip().upper()
            if type_name in {"BLOB", "BYTEA", "BINARY", "VARBINARY"}:
                total += _LOCAL_SHM_BLOB_OUTPUT_ROW_BUDGET_BYTES
            elif type_name in {"VARCHAR", "TEXT", "STRING", "JSON"}:
                total += _LOCAL_SHM_TEXT_OUTPUT_ROW_BUDGET_BYTES
            elif "[]" in type_name or type_name.startswith(("LIST", "ARRAY", "STRUCT", "MAP")):
                total += _LOCAL_SHM_NESTED_OUTPUT_ROW_BUDGET_BYTES
            continue
        dtype = str(entry.get("dtype") or "").strip().upper()
        dtype_bytes = _TENSOR_DTYPE_BYTES.get(dtype)
        if dtype_bytes is None:
            continue
        element_count = _product_ints(entry.get("shape") or [])
        if element_count <= 0:
            continue
        total += dtype_bytes * element_count
    return total


def _estimate_output_budget_from_rows(row_bytes: int, num_rows: int | None) -> int:
    if row_bytes <= 0 or num_rows is None or num_rows <= 0:
        return 0
    payload_bytes = int(row_bytes) * int(num_rows)
    return payload_bytes + max(_LOCAL_SHM_OUTPUT_BUDGET_OVERHEAD_BYTES, payload_bytes // 32)


def _make_local_ref_bundle_worker_payload_with_lease(
    block_refs: Any,
    slices: Any,
    metadata: Any,
    names: Any,
    *,
    submit_id: int | None,
    name: str,
    reserve_output_credit: bool,
) -> tuple[dict[str, Any], int] | tuple[None, None]:
    worker_payload = make_local_ref_bundle_worker_payload(block_refs, slices, metadata, names)
    if worker_payload is None:
        return None, None
    lease_id = create_local_shm_input_lease(
        tuple(block_refs),
        name=name,
        submit_id=submit_id,
        reserve_output_credit=reserve_output_credit,
    )
    worker_payload = make_local_ref_bundle_worker_payload(
        block_refs,
        slices,
        metadata,
        names,
        input_lease_id=lease_id,
    )
    if worker_payload is None:
        cancel_local_shm_input_lease(lease_id, name=name)
        raise RuntimeError("local_shm input lease payload creation failed")
    return worker_payload, lease_id


def _read_exact(sock: socket.socket, size: int) -> bytes:
    parts: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("UDF subprocess closed the control socket")
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def _send_message(sock: socket.socket, msg_type: int, payload: bytes = b"") -> None:
    sock.sendall(_HEADER.pack(msg_type, len(payload)) + payload)


def _recv_message(sock: socket.socket) -> tuple[int, bytes]:
    header = _read_exact(sock, _HEADER.size)
    msg_type, payload_len = _HEADER.unpack(header)
    payload = _read_exact(sock, payload_len) if payload_len else b""
    return msg_type, payload


def _arrow_table_from_ipc_bytes(data: bytes) -> pa.Table:
    reader = pa.ipc.open_stream(data)
    return reader.read_all()


def _write_ipc_to_shm(shm: shared_memory.SharedMemory, ipc_bytes: bytes) -> int:
    required = _IPC_HEADER.size + len(ipc_bytes)
    buf = cast(Any, shm.buf)
    if required > len(buf):
        raise BufferError("shared memory segment is too small")
    _IPC_HEADER.pack_into(buf, 0, len(ipc_bytes))
    buf[_IPC_HEADER.size : required] = ipc_bytes
    return required


def _read_ipc_from_shm(shm: shared_memory.SharedMemory, size: int | None = None) -> bytes:
    buf = cast(Any, shm.buf)
    if len(buf) < _IPC_HEADER.size:
        raise BufferError("shared memory segment is too small for IPC header")
    ipc_size = _IPC_HEADER.unpack_from(buf, 0)[0]
    required = _IPC_HEADER.size + ipc_size
    if required > len(buf):
        raise BufferError(f"shared memory IPC payload exceeds local mapping: required={required} capacity={len(buf)}")
    if size is not None and required > size:
        raise BufferError(f"shared memory IPC payload exceeds response size: required={required} size={size}")
    return bytes(buf[_IPC_HEADER.size : required])


class _SubprocessStartupCancelledError(RuntimeError):
    pass


class _SubprocessStartupCleanupError(RuntimeError):
    pass


class _SingleSubprocessExecutor(BaseUDFExecutor):
    """Run Python UDFs in one long-lived worker subprocess."""

    def __init__(
        self,
        payload: dict[str, Any],
        *,
        worker_env: dict[str, str] | None = None,
        session_config: Mapping[str, Any] | None = None,
        startup_observer: Callable[[_SingleSubprocessExecutor], None] | None = None,
    ) -> None:
        if payload is None:
            raise ValueError("UDF payload is required")

        self._queue: deque[Any] = deque()
        self._finished_submitting = False
        self._closed = False
        self._broken_error: str | None = None
        self._actor_lost = False
        self._pending_batches = 0
        self._wakeup: Callable[[], None] | None = None
        self._wakeup_error: BaseException | None = None
        self._ref_bundle_output = payload_requests_local_ref_bundle_output(payload)
        self._worker_env = dict(worker_env or {})
        self._session_config = (
            None if session_config is None else {str(key): str(value) for key, value in session_config.items()}
        )
        self._execution_scope_lock = threading.Lock()
        self._active_execution_scope: ExecutionCancellationScope | None = None
        self._worker_lifetime_scope = ExecutionCancellationScope(f"subprocess-worker:{id(self)}", 1)
        self._close_lock = threading.Lock()
        self._startup_cancel_requested = threading.Event()
        self._active_input_leases: dict[int, ExecutionCancellationScope] = {}
        self._active_input_leases_lock = threading.Lock()
        self._active_output_grants: dict[int, ExecutionCancellationScope] = {}
        self._active_output_grants_lock = threading.Lock()

        self._payload_shm: shared_memory.SharedMemory | None = None
        self._data_shm: shared_memory.SharedMemory | None = None
        self._sock: socket.socket | None = None
        self._proc: subprocess.Popen[bytes] | None = None

        self._start_worker(payload, startup_observer=startup_observer)
        self._finalizer = weakref.finalize(
            self,
            _cleanup_subprocess_executor,
            self._proc,
            self._sock,
            self._payload_shm,
            self._data_shm,
        )

    def _start_worker(
        self,
        payload: dict[str, Any],
        *,
        startup_observer: Callable[[_SingleSubprocessExecutor], None] | None = None,
    ) -> None:
        payload_bytes = vane_pickle.dumps(payload)
        payload_size = _IPC_HEADER.size + len(payload_bytes)
        payload_shm = _create_shm(max(payload_size, 4096), track=False)
        data_shm = _create_shm(_DEFAULT_SHM_SIZE, track=False)
        parent_sock, child_sock = socket.socketpair()
        child_fd = child_sock.fileno()

        try:
            _write_ipc_to_shm(payload_shm, payload_bytes)
            cmd = [
                sys.executable,
                "-m",
                "vane.execution.udf_subprocess_worker",
                str(child_fd),
                payload_shm.name,
                str(payload_size),
                data_shm.name,
            ]
            env = (
                dict(os.environ)
                if self._session_config is None
                else build_explicit_session_process_env(self._session_config)
            )
            env.update(self._worker_env)
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(
                cmd,
                pass_fds=(child_fd,),
                close_fds=True,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=None if _subprocess_debug_enabled() else subprocess.DEVNULL,
            )
        except Exception:
            child_sock.close()
            parent_sock.close()
            payload_shm.close()
            _unlink_shm(payload_shm, track=False)
            data_shm.close()
            _unlink_shm(data_shm, track=False)
            raise

        child_sock.close()
        self._payload_shm = payload_shm
        self._data_shm = data_shm
        self._sock = parent_sock
        self._proc = proc
        try:
            if startup_observer is not None:
                startup_observer(self)
            self._raise_if_startup_cancelled()
            msg_type, payload_data = self._recv_expected(
                (_MSG_READY, _MSG_ERROR),
                timeout_s=_subprocess_control_timeout_s(),
            )
            if msg_type == _MSG_ERROR:
                self._mark_broken(payload_data.decode("utf-8", errors="replace"))
                raise RuntimeError(self._broken_error)
            self._raise_if_startup_cancelled()

            # The worker has loaded the payload. The parent no longer needs this shm.
            self._close_payload_shm()
        except BaseException as startup_error:
            cancel_requested = self._startup_cancel_requested.is_set()
            if isinstance(startup_error, _SubprocessStartupCleanupError):
                raise
            if self._closed:
                if cancel_requested:
                    raise _SubprocessStartupCancelledError(
                        "UDF subprocess worker startup was cancelled"
                    ) from startup_error
                raise
            try:
                self.close(kill=True)
            except BaseException as cleanup_error:
                raise _SubprocessStartupCleanupError(
                    f"UDF subprocess worker startup failed: {type(startup_error).__name__}: {startup_error}; "
                    f"cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
                ) from startup_error
            if cancel_requested:
                raise _SubprocessStartupCancelledError("UDF subprocess worker startup was cancelled") from startup_error
            raise

    def _raise_if_startup_cancelled(self) -> None:
        cancel_requested = getattr(self, "_startup_cancel_requested", None)
        if not self._closed and (cancel_requested is None or not cancel_requested.is_set()):
            return
        try:
            self.close(kill=True)
        except BaseException as exc:
            raise _SubprocessStartupCleanupError(
                f"UDF subprocess worker startup cancellation cleanup failed: {exc}"
            ) from exc
        raise _SubprocessStartupCancelledError("UDF subprocess worker startup was cancelled")

    def _recv_expected(self, expected: tuple[int, ...], *, timeout_s: float | None = None) -> tuple[int, bytes]:
        sock = self._require_socket()
        timeout_overridden = False
        restore_timeout = None
        try:
            if timeout_s is not None and hasattr(sock, "settimeout") and hasattr(sock, "gettimeout"):
                restore_timeout = sock.gettimeout()
                sock.settimeout(max(0.0, float(timeout_s)))
                timeout_overridden = True
            try:
                msg_type, payload = _recv_message(sock)
            finally:
                if timeout_overridden:
                    try:
                        sock.settimeout(restore_timeout)
                    except Exception:
                        pass
        except Exception as exc:
            cancel_requested = getattr(self, "_startup_cancel_requested", None)
            if cancel_requested is not None and cancel_requested.is_set():
                raise _SubprocessStartupCancelledError(f"UDF subprocess worker startup was cancelled: {exc}") from exc
            self._mark_broken(f"UDF subprocess communication failed: {exc}", actor_lost=True)
            raise RuntimeError(self._broken_error) from exc
        if msg_type not in expected:
            self._mark_broken(f"UDF subprocess sent unexpected message type {msg_type:#x}", actor_lost=True)
            raise RuntimeError(self._broken_error)
        return msg_type, payload

    def _require_socket(self) -> socket.socket:
        if self._closed:
            raise RuntimeError("UDF subprocess executor is closed")
        if self._broken_error is not None:
            raise RuntimeError(self._broken_error)
        if self._sock is None:
            raise RuntimeError("UDF subprocess control socket is not available")
        return self._sock

    def _require_data_shm(self) -> shared_memory.SharedMemory:
        if self._data_shm is None:
            raise RuntimeError("UDF subprocess data shared memory is not available")
        return self._data_shm

    def _current_execution_scope(self) -> ExecutionCancellationScope:
        lock = getattr(self, "_execution_scope_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._execution_scope_lock = lock
        with lock:
            scope = getattr(self, "_active_execution_scope", None)
            if scope is not None:
                return cast("ExecutionCancellationScope", scope)
            lifetime_scope = getattr(self, "_worker_lifetime_scope", None)
            if lifetime_scope is None:
                lifetime_scope = ExecutionCancellationScope(f"subprocess-worker:{id(self)}", 1)
                self._worker_lifetime_scope = lifetime_scope
            return cast("ExecutionCancellationScope", lifetime_scope)

    def _run_in_execution_scope(
        self,
        scope: ExecutionCancellationScope,
        fn: Callable[[_SingleSubprocessExecutor], Any | None],
    ) -> Any | None:
        with self._execution_scope_lock:
            if self._active_execution_scope is not None:
                raise RuntimeError("UDF subprocess worker already has an active execution scope")
            self._active_execution_scope = scope
        unregister: Callable[[], None] | None = None
        cancel_cleanup_lock = threading.Lock()
        cancel_cleanup_errors: list[BaseException] = []
        result: Any | None = None
        result_ready = False

        def cleanup_cancelled_scope() -> None:
            # ExecutionCancellationScope deliberately keeps cancellation
            # observable when a wakeup raises. Preserve that wakeup failure
            # here so the worker cannot be returned to a pool as reusable.
            with cancel_cleanup_lock:
                try:
                    self._cancel_scope_resources(scope, cancel_scope=False)
                except BaseException as exc:
                    cancel_cleanup_errors.append(exc)

        try:
            unregister = scope.register_cancel_wakeup(cleanup_cancelled_scope)
            scope.raise_if_cancelled("UDF subprocess task")
            result = fn(self)
            result_ready = True
        finally:
            cleanup_errors: list[BaseException] = []
            if unregister is not None:
                try:
                    unregister()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            with cancel_cleanup_lock:
                cleanup_errors.extend(cancel_cleanup_errors)
                try:
                    self._cancel_scope_resources(scope, cancel_scope=False)
                except BaseException as exc:
                    cleanup_errors.append(exc)
            with self._execution_scope_lock:
                if self._active_execution_scope is scope:
                    self._active_execution_scope = None
            if cleanup_errors:
                if result_ready:
                    _release_local_ref_bundle_result(result)
                details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
                try:
                    self._mark_broken(f"UDF subprocess execution-scope cleanup failed: {details}")
                except BaseException as exc:
                    cleanup_errors.append(exc)
                    details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
                raise RuntimeError(f"UDF subprocess execution-scope cleanup failed: {details}") from cleanup_errors[0]
        try:
            scope.raise_if_cancelled("UDF subprocess task result")
        except BaseException:
            _release_local_ref_bundle_result(result)
            raise
        return result

    def _track_input_lease(
        self,
        lease_id: int,
        scope: ExecutionCancellationScope | None = None,
    ) -> None:
        owner_scope = scope or self._current_execution_scope()
        with self._active_input_leases_lock:
            self._active_input_leases[int(lease_id)] = owner_scope

    def _untrack_input_lease(self, lease_id: int) -> None:
        with self._active_input_leases_lock:
            self._active_input_leases.pop(int(lease_id), None)

    def _cancel_active_input_leases(self, scope: ExecutionCancellationScope | None = None) -> None:
        with self._active_input_leases_lock:
            lease_ids = [
                lease_id
                for lease_id, owner_scope in self._active_input_leases.items()
                if scope is None or owner_scope is scope
            ]
            for lease_id in lease_ids:
                self._active_input_leases.pop(lease_id, None)
        cleanup_errors: list[BaseException] = []
        for lease_id in lease_ids:
            try:
                cancel_local_shm_input_lease(lease_id, name="udf-input-close")
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
            raise RuntimeError(f"UDF subprocess input-lease cancellation failed: {details}") from cleanup_errors[0]

    def _track_output_grant(
        self,
        grant_id: int,
        scope: ExecutionCancellationScope | None = None,
    ) -> None:
        if int(grant_id) <= 0:
            return
        owner_scope = scope or self._current_execution_scope()
        with self._active_output_grants_lock:
            self._active_output_grants[int(grant_id)] = owner_scope

    def _untrack_output_grant(self, grant_id: int) -> None:
        if int(grant_id) <= 0:
            return
        with self._active_output_grants_lock:
            self._active_output_grants.pop(int(grant_id), None)

    def _release_output_grant(self, grant_id: int, *, name: str) -> None:
        if int(grant_id) <= 0:
            return
        try:
            release_local_shm_output_grant(int(grant_id), name=name)
        finally:
            self._untrack_output_grant(int(grant_id))

    def _release_active_output_grants(
        self,
        *,
        name: str = "udf-output-close",
        scope: ExecutionCancellationScope | None = None,
    ) -> None:
        with self._active_output_grants_lock:
            grant_ids = [
                grant_id
                for grant_id, owner_scope in self._active_output_grants.items()
                if scope is None or owner_scope is scope
            ]
            for grant_id in grant_ids:
                self._active_output_grants.pop(grant_id, None)
        cleanup_errors: list[BaseException] = []
        for grant_id in grant_ids:
            try:
                release_local_shm_output_grant(grant_id, name=name)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
            raise RuntimeError(f"UDF subprocess output-grant release failed: {details}") from cleanup_errors[0]

    def _cancel_scope_resources(
        self,
        scope: ExecutionCancellationScope,
        *,
        cancel_scope: bool = True,
    ) -> None:
        cleanup_errors: list[BaseException] = []
        if cancel_scope:
            try:
                scope.cancel("subprocess execution cancelled")
            except BaseException as exc:
                cleanup_errors.append(exc)
        try:
            self._cancel_active_input_leases(scope)
        except BaseException as exc:
            cleanup_errors.append(exc)
        try:
            self._release_active_output_grants(name="udf-output-cancel", scope=scope)
        except BaseException as exc:
            cleanup_errors.append(exc)
        try:
            wake_local_shm_ref_budget_waiters()
        except BaseException as exc:
            cleanup_errors.append(exc)
        if cleanup_errors:
            details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
            raise RuntimeError(f"UDF subprocess scope-resource cleanup failed: {details}") from cleanup_errors[0]

    def _mark_broken(
        self,
        error: str,
        *,
        actor_lost: bool = False,
        graceful_close: bool = False,
    ) -> None:
        self._actor_lost = self._actor_lost or actor_lost
        if self._broken_error is None:
            self._broken_error = error
        self.close(kill=not graceful_close)

    def _mark_reported_error(self, error: str) -> None:
        try:
            self._mark_broken(error, graceful_close=True)
        except BaseException as cleanup_error:
            raise RuntimeError(
                f"{error}; graceful broken-worker cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
            ) from cleanup_error

    def _mark_broken_after_cleanup_failure(self, error: BaseException) -> str:
        try:
            self._mark_broken(f"UDF subprocess resource cleanup failed: {error}")
        except BaseException as cleanup_error:
            return f"; broken-worker cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
        return ""

    def _close_payload_shm(self) -> None:
        shm = self._payload_shm
        self._payload_shm = None
        if shm is None:
            return
        try:
            shm.close()
        finally:
            try:
                _unlink_shm(shm, track=False)
            except FileNotFoundError:
                pass

    def _wrap_output(self, output: pa.Table) -> Any:
        if self._ref_bundle_output:
            return make_local_shm_ref_bundle_result(output)
        return output

    def _notify_wakeup(self) -> None:
        callback = self._wakeup
        if callback is None:
            return
        try:
            callback()
        except Exception as exc:
            self._record_wakeup_error(exc)

    def _record_wakeup_error(self, exc: BaseException) -> None:
        if self._wakeup_error is None:
            self._wakeup_error = exc
        if self._broken_error is None:
            self._broken_error = f"UDF subprocess wakeup callback failed: {exc}"

    def _submit_table(self, args: pa.Table) -> Any | None:
        args = _ensure_table(args)
        if args.num_rows == 0:
            return None

        scope = self._current_execution_scope()
        scope.raise_if_cancelled("UDF subprocess input allocation")
        _marker, refs, metadata, names = make_local_shm_ref_bundle_result(
            args,
            cancel_event=scope,
        )
        lease_id = None
        try:
            scope.raise_if_cancelled("UDF subprocess input allocation")
            worker_payload, lease_id = _make_local_ref_bundle_worker_payload_with_lease(
                refs,
                None,
                metadata,
                names,
                submit_id=None,
                name="udf-materialized-input",
                reserve_output_credit=self._ref_bundle_output,
            )
            if worker_payload is None:
                raise RuntimeError("local_shm descriptor creation failed for subprocess submit")
            return self._submit_ref_bundle_direct(worker_payload)
        except BaseException as submit_error:
            cleanup_error: BaseException | None = None
            if lease_id is not None:
                try:
                    cancel_local_shm_input_lease(lease_id, name="udf-materialized-input")
                except BaseException as exc:
                    cleanup_error = exc
            if cleanup_error is not None:
                broken_cleanup_details = self._mark_broken_after_cleanup_failure(cleanup_error)
                raise RuntimeError(
                    f"UDF subprocess materialized submit failed: {type(submit_error).__name__}: "
                    f"{submit_error}; input-lease cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}{broken_cleanup_details}"
                ) from submit_error
            raise
        finally:
            for ref in refs:
                try:
                    ref.release()
                except Exception:
                    pass

    def _handle_submit_control_message(self, msg_type: int, payload: bytes) -> bool:
        if msg_type == _MSG_INPUT_CONSUMED:
            event = vane_pickle.loads(payload)
            lease_id = int(event["input_lease_id"])
            consume_local_shm_input_lease(lease_id, name="udf-input")
            self._untrack_input_lease(lease_id)
            self._notify_wakeup()
            return True
        if msg_type == _MSG_INPUT_CONSUME_FAILED:
            event = vane_pickle.loads(payload)
            lease_id = int(event["input_lease_id"])
            cancel_local_shm_input_lease(lease_id, name="udf-input")
            self._untrack_input_lease(lease_id)
            self._notify_wakeup()
            return True
        if msg_type == _MSG_OUTPUT_GRANT_REQUEST:
            event = vane_pickle.loads(payload)
            request_id = int(event.get("request_id", 0))
            size = int(event["size_bytes"])
            priority = str(event.get("priority") or "consumer")
            input_lease_id_raw = event.get("input_lease_id")
            input_lease_id = int(input_lease_id_raw) if input_lease_id_raw is not None else None
            scope = self._current_execution_scope()
            grant_id = 0
            try:
                grant_id = request_local_shm_output_grant(
                    size,
                    name=f"udf-output-{request_id}",
                    priority=priority,
                    input_lease_id=input_lease_id,
                    cancel_event=scope,
                )
                self._track_output_grant(grant_id, scope)
                scope.raise_if_cancelled("UDF subprocess output grant")
            except BaseException as exc:
                if grant_id > 0:
                    self._release_output_grant(grant_id, name=f"udf-output-{request_id}-cancelled")
                _send_message(
                    self._require_socket(),
                    _MSG_OUTPUT_GRANT_CANCELLED,
                    str(exc).encode("utf-8", errors="replace"),
                )
                return True
            response = {"request_id": request_id, "grant_id": int(grant_id)}
            try:
                _send_message(self._require_socket(), _MSG_OUTPUT_GRANT_GRANTED, vane_pickle.dumps(response))
            except BaseException as exc:
                self._release_output_grant(grant_id, name=f"udf-output-{request_id}-send-failed")
                self._mark_broken(f"UDF subprocess output grant response failed: {exc}", actor_lost=True)
                raise RuntimeError(self._broken_error) from exc
            self._notify_wakeup()
            return True
        if msg_type == _MSG_OUTPUT_GRANT_RELEASE:
            event = vane_pickle.loads(payload)
            grant_id = int(event["grant_id"])
            self._release_output_grant(grant_id, name="udf-output-worker-release")
            self._notify_wakeup()
            return True
        return False

    def _recv_submit_result(self) -> Any | None:
        msg_type = None
        payload = b""
        while msg_type is None:
            msg_type, payload = self._recv_expected(
                (
                    _MSG_OK,
                    _MSG_REF_BUNDLE_RESULT,
                    _MSG_ERROR,
                    _MSG_INPUT_CONSUMED,
                    _MSG_INPUT_CONSUME_FAILED,
                    _MSG_OUTPUT_GRANT_REQUEST,
                    _MSG_OUTPUT_GRANT_RELEASE,
                    _MSG_TASK_CANCELLED,
                )
            )
            try:
                if self._handle_submit_control_message(msg_type, payload):
                    msg_type = None
                    continue
            except BaseException as exc:
                self._mark_broken(
                    f"UDF subprocess control-message handling failed: {exc}",
                    actor_lost=True,
                )
                raise RuntimeError(self._broken_error) from exc
            if msg_type == _MSG_ERROR:
                error = payload.decode("utf-8", errors="replace")
                self._mark_reported_error(error)
                raise RuntimeError(error)
            if msg_type == _MSG_TASK_CANCELLED:
                error = payload.decode("utf-8", errors="replace") or "UDF subprocess task cancelled"
                scope = self._current_execution_scope()
                if scope.is_set():
                    raise ExecutionCancelledError(f"UDF subprocess task cancelled: {scope.cancel_reason or error}")
                self._mark_broken(f"UDF subprocess unexpectedly cancelled a task: {error}")
                raise RuntimeError(self._broken_error)
            if msg_type == _MSG_REF_BUNDLE_RESULT:
                descriptor = vane_pickle.loads(payload)
                grant_id_raw = descriptor.get("grant_id") if isinstance(descriptor, dict) else None
                grant_id = int(grant_id_raw) if grant_id_raw is not None else None
                scope = self._current_execution_scope()
                if scope.is_set():
                    try:
                        release_local_shm_ref_bundle_descriptor(descriptor)
                    finally:
                        if grant_id is not None:
                            self._release_output_grant(grant_id, name="udf-output-cancelled-result")
                    raise ExecutionCancelledError(
                        f"UDF subprocess task cancelled: {scope.cancel_reason or 'cancelled'}"
                    )
                try:
                    result = make_local_shm_ref_bundle_result_from_descriptor(
                        descriptor,
                        block_on_budget=False,
                        cancel_event=scope,
                    )
                except BaseException:
                    try:
                        release_local_shm_ref_bundle_descriptor(descriptor)
                    finally:
                        if grant_id is not None:
                            self._release_output_grant(grant_id, name="udf-output-descriptor-wrap-failed")
                    raise
                if grant_id is not None:
                    self._untrack_output_grant(grant_id)
                try:
                    scope.raise_if_cancelled("UDF subprocess result")
                except BaseException:
                    _release_local_ref_bundle_result(result)
                    raise
                return result
            if len(payload) != 8:
                self._mark_broken("UDF subprocess returned malformed OK response", actor_lost=True)
                raise RuntimeError(self._broken_error)

            result_size = struct.unpack("<Q", payload)[0]
            scope = self._current_execution_scope()
            scope.raise_if_cancelled("UDF subprocess task")
            if result_size == 0:
                return None
            if self._ref_bundle_output:
                self._mark_broken(
                    "distributed UDF subprocess output must be a local_shm ref-bundle result; "
                    "worker returned direct Arrow IPC output",
                    actor_lost=True,
                )
                raise RuntimeError(self._broken_error)
            data_shm = self._require_data_shm()
            if result_size > len(cast(Any, data_shm.buf)):
                name = data_shm.name
                data_shm.close()
                self._data_shm = data_shm = _open_existing_shm(name, track=False)
            ipc_result = _read_ipc_from_shm(data_shm, result_size)
            return self._wrap_output(_arrow_table_from_ipc_bytes(ipc_result))
        raise RuntimeError("UDF subprocess submit result loop exited unexpectedly")

    def _submit_ref_bundle_direct(self, payload: dict[str, Any]) -> Any | None:
        lease_id_raw = payload.get("input_lease_id")
        if payload.get("estimated_num_rows") == 0:
            # The worker will never receive this bundle and therefore cannot
            # acknowledge its lease. Cancel rather than consume it so an empty
            # result does not leave behind unused output credit.
            if lease_id_raw is not None:
                cancel_local_shm_input_lease(int(lease_id_raw), name="udf-input-zero-row")
            return None
        sock = self._require_socket()
        lease_id = int(lease_id_raw) if lease_id_raw is not None else None
        scope = self._current_execution_scope()
        scope.raise_if_cancelled("UDF subprocess submit")
        if lease_id is not None:
            self._track_input_lease(lease_id, scope)
            if scope.is_set():
                cancel_local_shm_input_lease(lease_id, name="udf-input")
                self._untrack_input_lease(lease_id)
                scope.raise_if_cancelled("UDF subprocess submit")
        try:
            payload_bytes = vane_pickle.dumps(payload)
            _send_message(sock, _MSG_SUBMIT_REF_BUNDLE, payload_bytes)
        except Exception as exc:
            broken_error = f"UDF subprocess ref-bundle submit failed: {exc}"
            try:
                # Closing the broken worker owns every tracked lease. Mark it
                # broken before any fallible lease cleanup so a transport
                # failure can never return this process to an idle pool.
                self._mark_broken(broken_error, actor_lost=True)
            except BaseException as broken_worker_cleanup_error:
                raise RuntimeError(
                    f"{broken_error}; broken-worker cleanup failed: "
                    f"{type(broken_worker_cleanup_error).__name__}: {broken_worker_cleanup_error}"
                ) from exc
            raise RuntimeError(self._broken_error) from exc
        try:
            return self._recv_submit_result()
        except BaseException as submit_error:
            cleanup_error: BaseException | None = None
            if lease_id is not None:
                try:
                    cancel_local_shm_input_lease(lease_id, name="udf-input")
                except BaseException as exc:
                    cleanup_error = exc
                finally:
                    self._untrack_input_lease(lease_id)
            if cleanup_error is not None:
                broken_cleanup_details = self._mark_broken_after_cleanup_failure(cleanup_error)
                raise RuntimeError(
                    f"UDF subprocess result receive failed: {type(submit_error).__name__}: "
                    f"{submit_error}; input-lease cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}{broken_cleanup_details}"
                ) from submit_error
            raise

    def _submit_ref_bundle(self, block_refs: Any, slices: Any, metadata: Any, names: Any) -> Any | None:
        worker_payload, lease_id = _make_local_ref_bundle_worker_payload_with_lease(
            block_refs,
            slices,
            metadata,
            names,
            submit_id=None,
            name="udf-input",
            reserve_output_credit=self._ref_bundle_output,
        )
        if worker_payload is not None:
            try:
                return self._submit_ref_bundle_direct(worker_payload)
            except BaseException as submit_error:
                assert lease_id is not None
                try:
                    cancel_local_shm_input_lease(lease_id, name="udf-input")
                except BaseException as cleanup_error:
                    broken_cleanup_details = self._mark_broken_after_cleanup_failure(cleanup_error)
                    raise RuntimeError(
                        f"UDF subprocess ref-bundle submit failed: {type(submit_error).__name__}: "
                        f"{submit_error}; input-lease cleanup failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}{broken_cleanup_details}"
                    ) from submit_error
                raise

        raise RuntimeError("subprocess UDF ref-bundle input requires local shared-memory descriptors")

    def submit(self, args: pa.Table) -> None:
        self._pending_batches += 1
        try:
            result = self._submit_table(args)
        finally:
            self._pending_batches = max(0, self._pending_batches - 1)
        self._queue.append(result if result is not None else (None, True))
        self._notify_wakeup()

    def submit_with_id(self, submit_id: int, args: pa.Table) -> None:
        self._pending_batches += 1
        try:
            result = self._submit_table(args)
        finally:
            self._pending_batches = max(0, self._pending_batches - 1)
        self._queue.append((SUBMIT_RESULT_MARKER, int(submit_id), result))
        self._notify_wakeup()

    def submit_ref_bundle_with_id(
        self,
        submit_id: int,
        block_refs: Any,
        slices: Any,
        metadata: Any,
        names: Any,
    ) -> None:
        self._pending_batches += 1
        try:
            result = self._submit_ref_bundle(block_refs, slices, metadata, names)
        finally:
            self._pending_batches = max(0, self._pending_batches - 1)
        self._queue.append((SUBMIT_RESULT_MARKER, int(submit_id), result))
        self._notify_wakeup()

    def submit_ref_bundle(self, block_refs: Any, slices: Any, metadata: Any, names: Any) -> None:
        self._pending_batches += 1
        try:
            result = self._submit_ref_bundle(block_refs, slices, metadata, names)
        finally:
            self._pending_batches = max(0, self._pending_batches - 1)
        self._queue.append(result if result is not None else (None, True))
        self._notify_wakeup()

    def take_ready_result(self) -> Any | None:
        if self._wakeup_error is not None:
            raise RuntimeError(f"UDF subprocess wakeup callback failed: {self._wakeup_error}") from self._wakeup_error
        try:
            return self._queue.popleft()
        except IndexError:
            return None

    def finished_submitting(self) -> None:
        if self._finished_submitting:
            return
        if self._closed or self._broken_error is not None:
            self._finished_submitting = True
            return
        sock = self._require_socket()
        try:
            _send_message(sock, _MSG_FINISHED)
            msg_type, payload = self._recv_expected(
                (_MSG_ACK, _MSG_ERROR),
                timeout_s=_subprocess_control_timeout_s(),
            )
        except RuntimeError:
            raise
        except Exception as exc:
            self._mark_broken(f"UDF subprocess finished_submitting failed: {exc}", actor_lost=True)
            raise RuntimeError(self._broken_error) from exc
        if msg_type == _MSG_ERROR:
            error = payload.decode("utf-8", errors="replace")
            self._mark_reported_error(error)
            raise RuntimeError(error)
        self._finished_submitting = True

    def all_tasks_finished(self) -> bool:
        return self._finished_submitting and not self._queue and self._pending_batches == 0

    def stats(self) -> dict[str, int]:
        if self._wakeup_error is not None:
            raise RuntimeError(f"UDF subprocess wakeup callback failed: {self._wakeup_error}") from self._wakeup_error
        running = max(0, int(self._pending_batches))
        return {
            "udf_running_task_count": running,
            "udf_queued_task_count": 0,
            "udf_max_running_tasks": 1,
        }

    def register_wakeup(self, callback: Callable[[], None]) -> None:
        self._wakeup = callback

    def is_reusable(self) -> bool:
        if self._closed or self._broken_error is not None:
            return False
        proc = self._proc
        return proc is not None and proc.poll() is None

    def cancel_output_grants(self) -> None:
        scope = self._current_execution_scope()
        self._cancel_scope_resources(scope)

    def _cancel_startup(self) -> None:
        """Interrupt startup without waiting for its cleanup thread."""
        cleanup_errors: list[BaseException] = []
        cancel_requested = getattr(self, "_startup_cancel_requested", None)
        if cancel_requested is not None:
            cancel_requested.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except BaseException as exc:
                cleanup_errors.append(exc)
        sock = self._sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            except BaseException as exc:
                cleanup_errors.append(exc)
            try:
                sock.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
            raise RuntimeError(f"UDF subprocess startup cancellation failed: {details}") from cleanup_errors[0]

    def close(self, kill: bool = False) -> None:
        close_lock = getattr(self, "_close_lock", None)
        if close_lock is None:
            close_lock = threading.Lock()
            self._close_lock = close_lock
        with close_lock:
            self._close_locked(kill=kill)

    def _close_locked(self, *, kill: bool) -> None:
        if self._closed:
            return
        self._closed = True
        cleanup_errors: list[BaseException] = []
        try:
            self.cancel_output_grants()
        except BaseException as exc:
            cleanup_errors.append(exc)
        try:
            self._cancel_active_input_leases()
        except BaseException as exc:
            cleanup_errors.append(exc)
        try:
            self._release_active_output_grants(name="udf-output-close")
        except BaseException as exc:
            cleanup_errors.append(exc)

        proc = self._proc
        sock = self._sock
        shutdown_error: BaseException | None = None
        graceful_error: BaseException | None = None
        finalizer_cleanup_failed = False

        if proc is not None and proc.poll() is None and sock is not None and not kill:
            deadline = time.monotonic() + _subprocess_shutdown_grace_s()
            try:
                _send_message(sock, _MSG_CLOSE)
                remaining = max(0.0, deadline - time.monotonic())
                if hasattr(sock, "settimeout"):
                    sock.settimeout(remaining)
                msg_type, payload = _recv_message(sock)
                if msg_type == _MSG_ERROR:
                    graceful_error = RuntimeError(
                        "UDF subprocess graceful shutdown failed: " + payload.decode("utf-8", errors="replace")
                    )
                elif msg_type != _MSG_ACK:
                    graceful_error = RuntimeError(
                        f"UDF subprocess graceful shutdown returned unexpected message type {msg_type:#x}"
                    )
            except (socket.timeout, EOFError, OSError) as exc:
                graceful_error = RuntimeError(
                    f"UDF subprocess graceful shutdown timed out or disconnected: {type(exc).__name__}: {exc}"
                )
            except BaseException as exc:
                graceful_error = exc
            if proc.poll() is None:
                remaining = max(0.0, deadline - time.monotonic())
                try:
                    proc.wait(timeout=remaining)
                except (subprocess.TimeoutExpired, TimeoutError) as exc:
                    if graceful_error is None:
                        graceful_error = RuntimeError(
                            "UDF subprocess graceful shutdown did not exit before deadline: "
                            f"{type(exc).__name__}: {exc}"
                        )
                except BaseException as exc:
                    if graceful_error is None:
                        graceful_error = exc

        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except BaseException as exc:
                shutdown_error = exc
            try:
                proc.wait(timeout=_subprocess_control_timeout_s())
            except BaseException as exc:
                if shutdown_error is None:
                    shutdown_error = exc

        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass

        self._proc = None
        self._sock = None
        try:
            self._close_payload_shm()
        except BaseException as exc:
            cleanup_errors.append(exc)
            finalizer_cleanup_failed = True
        data_shm = self._data_shm
        self._data_shm = None
        if data_shm is not None:
            try:
                data_shm.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
                finalizer_cleanup_failed = True
            try:
                _unlink_shm(data_shm, track=False)
            except FileNotFoundError:
                pass
            except BaseException as exc:
                cleanup_errors.append(exc)
                finalizer_cleanup_failed = True

        finalizer = getattr(self, "_finalizer", None)
        if finalizer is not None and finalizer.alive and shutdown_error is None and not finalizer_cleanup_failed:
            try:
                finalizer.detach()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if shutdown_error is not None:
            cleanup_errors.append(RuntimeError(f"UDF subprocess did not terminate cleanly: {shutdown_error}"))
        if graceful_error is not None:
            cleanup_errors.append(graceful_error)
        if cleanup_errors:
            details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
            raise RuntimeError(f"UDF subprocess close failed: {details}") from cleanup_errors[0]

    def __del__(self) -> None:
        try:
            self.close(kill=True)
        except Exception:
            pass


def _payload_positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"payload.{key} must be a positive integer")
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError(f"payload.{key} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"payload.{key} must be a positive integer")
    return parsed


def _payload_subprocess_mode(payload: dict[str, Any]) -> str:
    backend = str(payload.get("execution_backend") or "").strip().lower()
    if backend == "subprocess_task":
        return "task"
    if backend == "subprocess_actor":
        return "actor"
    raise ValueError("payload.execution_backend must be one of: subprocess_task, subprocess_actor")


def _payload_subprocess_pool_size(payload: dict[str, Any], mode: str) -> int:
    if mode == "actor":
        return _payload_positive_int(payload, "actor_number")
    return _payload_positive_int(payload, "udf_worker_slots")


def _worker_env_for_pool_index(payload: dict[str, Any], worker_idx: int, pool_size: int) -> dict[str, str]:
    _payload_subprocess_mode(payload)
    env = {
        "VANE_SUBPROCESS_WORKER_INDEX": str(int(worker_idx)),
        "VANE_SUBPROCESS_POOL_SIZE": str(int(pool_size)),
    }
    env.update(_worker_thread_env(payload))
    return env


def _normalize_session_config_option(options: Mapping[str, Any]) -> dict[str, str] | None:
    if "session_config" not in options:
        return None
    raw_config = options["session_config"]
    if not isinstance(raw_config, Mapping):
        raise TypeError("UDF executor session_config must be a mapping")
    return {str(key): str(value) for key, value in raw_config.items()}


def _payload_task_key(
    payload: dict[str, Any],
    session_config: Mapping[str, Any] | None = None,
) -> str:
    identity = (
        payload,
        None
        if session_config is None
        else tuple(sorted((str(key), str(value)) for key, value in session_config.items())),
    )
    return hashlib.sha256(vane_pickle.dumps(identity)).hexdigest()


def _worker_is_reusable(worker: Any) -> bool:
    check = getattr(worker, "is_reusable", None)
    if callable(check):
        return bool(check())
    if getattr(worker, "_closed", False) or getattr(worker, "_broken_error", None) is not None:
        return False
    proc = getattr(worker, "_proc", None)
    poll = getattr(proc, "poll", None)
    return proc is None or not callable(poll) or poll() is None


def _release_local_ref_bundle_result(value: Any) -> None:
    if isinstance(value, tuple) and len(value) >= 3 and value[0] == SUBMIT_RESULT_MARKER:
        value = value[2]
    if not (isinstance(value, tuple) and len(value) >= 2 and value[0] == REF_BUNDLE_RESULT_MARKER):
        return
    for ref in list(value[1] or []):
        try:
            ref.release()
        except Exception:
            pass


class _PooledTaskWorker:
    def __init__(self, worker: _SingleSubprocessExecutor) -> None:
        self.worker = worker
        self.last_used = time.monotonic()
        self.active_scope: ExecutionCancellationScope | None = None
        self.abort_requested = False


class _TaskWorkerPool:
    def __init__(
        self,
        runtime: _GlobalSubprocessTaskRuntime,
        key: str,
        payload: dict[str, Any],
        pool_size: int,
        session_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self.key = key
        self.payload = dict(payload)
        self.session_config = (
            None if session_config is None else {str(key): str(value) for key, value in session_config.items()}
        )
        self.pool_size = max(1, int(pool_size))
        self.ref_count = 0
        self.closing = False
        self.idle: list[_PooledTaskWorker] = []
        self._active_wrappers: set[_PooledTaskWorker] = set()
        self._spawning_workers: set[int] = set()
        self._spawning_executors: dict[int, _SingleSubprocessExecutor] = {}
        self._spawn_cleanup_errors: list[BaseException] = []
        self.active = 0
        self.total = 0
        self.next_worker_idx = 0
        self.kill_on_release = False
        self.admission_slots = LocalExecutionSlotPool(
            max_slots=self.pool_size,
            execution_slot_prefix=f"subprocess_task:{self.key}",
        )

    def create_admission_authority(self) -> LocalSlotAdmissionAuthority:
        return self.admission_slots.create_authority()

    def acquire_ref(self) -> None:
        with self.runtime.cond:
            if self.closing:
                raise RuntimeError("subprocess task worker pool is closing")
            self.ref_count += 1

    def release_ref(self, *, kill: bool = False) -> None:
        to_close: list[_SingleSubprocessExecutor] = []
        active_to_kill: list[_SingleSubprocessExecutor] = []
        close_pool = False
        close_kill = bool(kill)
        with self.runtime.cond:
            self.ref_count = max(0, self.ref_count - 1)
            if self.ref_count == 0:
                close_pool = True
                self.closing = True
                self.kill_on_release = self.kill_on_release or close_kill
                close_kill = self.kill_on_release
                while self.idle:
                    wrapper = self.idle.pop()
                    self.total = max(0, self.total - 1)
                    self.runtime.total_workers = max(0, self.runtime.total_workers - 1)
                    to_close.append(wrapper.worker)
                if close_kill:
                    active_to_kill.extend(wrapper.worker for wrapper in self._active_wrappers)
                self.runtime.pools.pop(self.key, None)
            self.runtime.cond.notify_all()
        if not close_pool:
            return
        cleanup_errors: list[BaseException] = []
        try:
            self.admission_slots.close()
        except BaseException as exc:
            cleanup_errors.append(exc)
        for worker in to_close:
            try:
                worker.close(kill=close_kill)
            except BaseException as exc:
                cleanup_errors.append(exc)
        for worker in active_to_kill:
            try:
                worker.close(kill=True)
            except BaseException as exc:
                cleanup_errors.append(exc)
        cleanup_errors.extend(self._close_spawning_workers())
        if cleanup_errors:
            details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
            raise RuntimeError(f"subprocess task worker pool close failed: {details}") from cleanup_errors[0]

    def cancel_output_grants(self) -> None:
        workers: list[_SingleSubprocessExecutor] = []
        with self.runtime.cond:
            workers.extend(wrapper.worker for wrapper in self.idle)
            workers.extend(wrapper.worker for wrapper in self._active_wrappers)
        cleanup_errors: list[BaseException] = []
        for worker in workers:
            try:
                worker.cancel_output_grants()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
            raise RuntimeError(f"subprocess task output-grant cancellation failed: {details}") from cleanup_errors[0]

    def _wake_waiters(self) -> None:
        with self.runtime.cond:
            self.runtime.cond.notify_all()

    def _track_spawning_executor(
        self,
        worker_idx: int,
        worker: _SingleSubprocessExecutor,
    ) -> None:
        close_worker = False
        with self.runtime.cond:
            spawning_executors = getattr(self, "_spawning_executors", None)
            if spawning_executors is None:
                spawning_executors = {}
                self._spawning_executors = spawning_executors
            if worker_idx in self._spawning_workers:
                spawning_executors[worker_idx] = worker
                close_worker = self.closing or self.runtime.closed
            self.runtime.cond.notify_all()
        if close_worker:
            try:
                worker._cancel_startup()
            except BaseException as exc:
                self._record_spawn_cleanup_error(worker_idx, exc)
                raise

    def _close_spawning_workers(self, *, deadline: float | None = None) -> list[BaseException]:
        cleanup_errors: list[BaseException] = []
        closed_workers: set[int] = set()
        if deadline is None:
            deadline = time.monotonic() + _subprocess_shutdown_grace_s()
        while True:
            workers_to_close: list[_SingleSubprocessExecutor] = []
            with self.runtime.cond:
                spawning_workers = set(getattr(self, "_spawning_workers", ()))
                if not spawning_workers:
                    cleanup_errors.extend(getattr(self, "_spawn_cleanup_errors", ()))
                    self._spawn_cleanup_errors = []
                    return cleanup_errors
                spawning_executors = getattr(self, "_spawning_executors", {})
                for worker_idx in spawning_workers:
                    worker = spawning_executors.get(worker_idx)
                    if worker is not None and id(worker) not in closed_workers:
                        closed_workers.add(id(worker))
                        workers_to_close.append(worker)
                if not workers_to_close:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        cleanup_errors.append(
                            TimeoutError(
                                "subprocess task worker startup cleanup did not finish before the shutdown deadline"
                            )
                        )
                        cleanup_errors.extend(getattr(self, "_spawn_cleanup_errors", ()))
                        self._spawn_cleanup_errors = []
                        return cleanup_errors
                    self.runtime.cond.wait(timeout=remaining)
                    continue
            for worker in workers_to_close:
                try:
                    # A provisional worker has never accepted user work, so
                    # interrupting it is safe even during graceful shutdown.
                    # The startup thread retains ownership of full cleanup.
                    cancel_startup = getattr(worker, "_cancel_startup", None)
                    if callable(cancel_startup):
                        cancel_startup()
                    else:
                        worker.close(kill=True)
                except BaseException as exc:
                    cleanup_errors.append(exc)

    def _spawn_worker(self, worker_idx: int) -> _PooledTaskWorker:
        worker = _SingleSubprocessExecutor(
            self.payload,
            worker_env=_worker_env_for_pool_index(self.payload, worker_idx, self.pool_size),
            session_config=self.session_config,
            startup_observer=lambda executor: self._track_spawning_executor(worker_idx, executor),
        )
        return _PooledTaskWorker(worker)

    def _record_spawn_cleanup_error(self, worker_idx: int, error: BaseException) -> None:
        cleanup_error = RuntimeError(
            f"subprocess task worker {worker_idx} startup cleanup failed: {type(error).__name__}: {error}"
        )
        with self.runtime.cond:
            errors = getattr(self, "_spawn_cleanup_errors", None)
            if errors is None:
                errors = []
                self._spawn_cleanup_errors = errors
            errors.append(cleanup_error)
            self.runtime.cond.notify_all()

    def acquire_worker(self, scope: ExecutionCancellationScope) -> _PooledTaskWorker:
        wrapper: _PooledTaskWorker | None = None
        spawn_idx: int | None = None
        unregister = scope.register_cancel_wakeup(self._wake_waiters)
        try:
            while wrapper is None and spawn_idx is None:
                evicted: _SingleSubprocessExecutor | None = None
                with self.runtime.cond:
                    scope.raise_if_cancelled("subprocess task worker acquisition")
                    if self.closing or self.runtime.closed:
                        raise RuntimeError("subprocess task worker pool is closed")
                    while self.idle:
                        candidate = self.idle.pop()
                        if _worker_is_reusable(candidate.worker):
                            candidate.active_scope = scope
                            self.active += 1
                            self._active_wrappers.add(candidate)
                            return candidate
                        self.total = max(0, self.total - 1)
                        self.runtime.total_workers = max(0, self.runtime.total_workers - 1)
                        evicted = candidate.worker
                        self.runtime.cond.notify_all()
                        break
                    if evicted is not None:
                        pass
                    elif self.total < self.pool_size and self.runtime.total_workers < self.runtime.max_workers:
                        spawn_idx = self.next_worker_idx
                        self.next_worker_idx += 1
                        self._spawning_workers.add(spawn_idx)
                        self.total += 1
                        self.active += 1
                        self.runtime.total_workers += 1
                        break
                    else:
                        evicted = self.runtime._take_idle_worker_locked()
                        if evicted is None:
                            self.runtime.cond.wait()
                            continue
                if evicted is not None:
                    evicted.close(kill=False)

            try:
                assert spawn_idx is not None
                wrapper = self._spawn_worker(spawn_idx)
            except BaseException:
                with self.runtime.cond:
                    self.total = max(0, self.total - 1)
                    self.active = max(0, self.active - 1)
                    self.runtime.total_workers = max(0, self.runtime.total_workers - 1)
                    getattr(self, "_spawning_executors", {}).pop(spawn_idx, None)
                    self._spawning_workers.discard(spawn_idx)
                    self.runtime.cond.notify_all()
                raise
            close_kill = False
            with self.runtime.cond:
                if self.closing or self.runtime.closed:
                    self.total = max(0, self.total - 1)
                    self.active = max(0, self.active - 1)
                    self.runtime.total_workers = max(0, self.runtime.total_workers - 1)
                    close_kill = self.kill_on_release
                    self.runtime.cond.notify_all()
                else:
                    wrapper.active_scope = scope
                    self._active_wrappers.add(wrapper)
                    getattr(self, "_spawning_executors", {}).pop(spawn_idx, None)
                    self._spawning_workers.discard(spawn_idx)
                    self.runtime.cond.notify_all()
                    return wrapper
            try:
                wrapper.worker.close(kill=close_kill)
            except BaseException as exc:
                self._record_spawn_cleanup_error(spawn_idx, exc)
                raise
            finally:
                with self.runtime.cond:
                    getattr(self, "_spawning_executors", {}).pop(spawn_idx, None)
                    self._spawning_workers.discard(spawn_idx)
                    self.runtime.cond.notify_all()
            raise RuntimeError("subprocess task worker pool is closed")
        finally:
            unregister()

    def release_worker(self, wrapper: _PooledTaskWorker, *, reusable: bool = True) -> None:
        to_close: _SingleSubprocessExecutor | None = None
        kill_close = False
        with self.runtime.cond:
            self._active_wrappers.discard(wrapper)
            self.active = max(0, self.active - 1)
            wrapper.active_scope = None
            if self.closing or wrapper.abort_requested or not reusable or not _worker_is_reusable(wrapper.worker):
                self.total = max(0, self.total - 1)
                self.runtime.total_workers = max(0, self.runtime.total_workers - 1)
                to_close = wrapper.worker
                kill_close = self.kill_on_release or wrapper.abort_requested or not reusable
            else:
                wrapper.last_used = time.monotonic()
                self.idle.append(wrapper)
            self.runtime.cond.notify_all()
        if to_close is not None:
            to_close.close(kill=kill_close)

    def abort_scopes(self, scopes: set[ExecutionCancellationScope]) -> None:
        workers: list[_SingleSubprocessExecutor] = []
        with self.runtime.cond:
            for wrapper in self._active_wrappers:
                if wrapper.active_scope in scopes:
                    wrapper.abort_requested = True
                    workers.append(wrapper.worker)
            self.runtime.cond.notify_all()
        cleanup_errors: list[BaseException] = []
        for worker in workers:
            try:
                worker.close(kill=True)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
            raise RuntimeError(f"subprocess task worker abort failed: {details}") from cleanup_errors[0]


class _GlobalSubprocessTaskRuntime:
    def __init__(self) -> None:
        self.max_workers = max(1, os.cpu_count() or 1)
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="vane-udf-subprocess-task",
        )
        self.cond = threading.Condition()
        self.pools: dict[str, _TaskWorkerPool] = {}
        self.total_workers = 0
        self.closed = False
        self._close_finished = True

    def acquire_pool(
        self,
        payload: dict[str, Any],
        pool_size: int,
        *,
        session_config: Mapping[str, Any] | None = None,
    ) -> _TaskWorkerPool:
        key = _payload_task_key(payload, session_config)
        with self.cond:
            if self.closed:
                raise RuntimeError("global subprocess task runtime is closed")
            pool = self.pools.get(key)
            if pool is None:
                pool = _TaskWorkerPool(self, key, payload, pool_size, session_config)
                self.pools[key] = pool
            pool.acquire_ref()
            return pool

    def submit(
        self,
        pool: _TaskWorkerPool,
        fn: Callable[[_SingleSubprocessExecutor], Any | None],
        scope: ExecutionCancellationScope,
        debug_seq: int = 0,
    ) -> Future[Any]:
        if self.closed:
            raise RuntimeError("global subprocess task runtime is closed")
        return self.executor.submit(self._run_task, pool, fn, scope, debug_seq)

    def _run_task(
        self,
        pool: _TaskWorkerPool,
        fn: Callable[[_SingleSubprocessExecutor], Any | None],
        scope: ExecutionCancellationScope,
        debug_seq: int = 0,
    ) -> Any | None:
        acquire_start = time.perf_counter()
        wrapper: _PooledTaskWorker | None = None
        wrapper = pool.acquire_worker(scope)
        acquire_s = time.perf_counter() - acquire_start
        assert wrapper is not None
        reusable = True
        run_start = time.perf_counter()
        result: Any | None = None
        result_ready = False
        try:
            if _should_debug_submit(debug_seq):
                proc = getattr(wrapper.worker, "_proc", None)
                _subprocess_debug_log(
                    "task_worker_acquired "
                    f"seq={debug_seq} acquire_s={acquire_s:.6f} "
                    f"worker_pid={getattr(proc, 'pid', None)} pool_size={pool.pool_size} "
                    f"pool_total={pool.total} pool_active={pool.active} pool_idle={len(pool.idle)} "
                    f"runtime_total_workers={self.total_workers} runtime_max_workers={self.max_workers}"
                )
            scope.raise_if_cancelled("subprocess task")
            result = wrapper.worker._run_in_execution_scope(scope, fn)
            result_ready = True
        except BaseException:
            reusable = _worker_is_reusable(wrapper.worker)
            raise
        finally:
            scope.finish()
            try:
                try:
                    if _should_debug_submit(debug_seq):
                        _subprocess_debug_log(
                            "task_worker_finished "
                            f"seq={debug_seq} run_s={time.perf_counter() - run_start:.6f} reusable={reusable}"
                        )
                finally:
                    pool.release_worker(wrapper, reusable=reusable)
            except BaseException:
                if result_ready:
                    _release_local_ref_bundle_result(result)
                raise
        try:
            scope.raise_if_cancelled("subprocess task result")
        except BaseException:
            _release_local_ref_bundle_result(result)
            raise
        return result

    def _take_idle_worker_locked(self) -> _SingleSubprocessExecutor | None:
        oldest_pool: _TaskWorkerPool | None = None
        oldest_idx = -1
        oldest_time: float | None = None
        for pool in self.pools.values():
            for idx, wrapper in enumerate(pool.idle):
                if oldest_time is None or wrapper.last_used < oldest_time:
                    oldest_pool = pool
                    oldest_idx = idx
                    oldest_time = wrapper.last_used
        if oldest_pool is None or oldest_idx < 0:
            return None
        wrapper = oldest_pool.idle.pop(oldest_idx)
        oldest_pool.total = max(0, oldest_pool.total - 1)
        self.total_workers = max(0, self.total_workers - 1)
        return wrapper.worker

    def stats(self) -> dict[str, int]:
        with self.cond:
            return {
                "max_workers": self.max_workers,
                "total_workers": self.total_workers,
                "pool_count": len(self.pools),
                "idle_workers": sum(len(pool.idle) for pool in self.pools.values()),
                "active_workers": sum(pool.active for pool in self.pools.values()),
            }

    def close(self, *, kill: bool = False) -> None:
        to_close: list[_SingleSubprocessExecutor] = []
        active_to_kill: list[_SingleSubprocessExecutor] = []
        pools_to_cancel: list[_TaskWorkerPool] = []
        active_workers = 0
        with self.cond:
            if self.closed:
                while not getattr(self, "_close_finished", True):
                    self.cond.wait()
                return
            self.closed = True
            self._close_finished = False
            for pool in list(self.pools.values()):
                pools_to_cancel.append(pool)
                pool.closing = True
                pool.kill_on_release = kill
                while pool.idle:
                    wrapper = pool.idle.pop()
                    to_close.append(wrapper.worker)
                active_to_kill.extend(wrapper.worker for wrapper in pool._active_wrappers)
                pool.total = pool.active
                active_workers += pool.active
            self.pools.clear()
            self.total_workers = active_workers
            self.cond.notify_all()
        try:
            cleanup_errors: list[BaseException] = []
            for pool in pools_to_cancel:
                try:
                    pool.admission_slots.close()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            for pool in pools_to_cancel:
                try:
                    pool.cancel_output_grants()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            try:
                self.executor.shutdown(wait=False, cancel_futures=True)
            except BaseException as exc:
                cleanup_errors.append(exc)
            for worker in to_close:
                try:
                    worker.close(kill=kill)
                except BaseException as exc:
                    cleanup_errors.append(exc)
            for worker in active_to_kill:
                try:
                    worker.close(kill=True)
                except BaseException as exc:
                    cleanup_errors.append(exc)
            spawn_cleanup_deadline = time.monotonic() + _subprocess_shutdown_grace_s()
            for pool in pools_to_cancel:
                cleanup_errors.extend(pool._close_spawning_workers(deadline=spawn_cleanup_deadline))
            if cleanup_errors:
                details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
                raise RuntimeError(f"global subprocess task runtime close failed: {details}") from cleanup_errors[0]
        finally:
            with self.cond:
                self._close_finished = True
                self.cond.notify_all()


_GLOBAL_TASK_RUNTIME_LOCK = threading.Lock()
_GLOBAL_TASK_RUNTIME: _GlobalSubprocessTaskRuntime | None = None


def _global_task_runtime() -> _GlobalSubprocessTaskRuntime:
    global _GLOBAL_TASK_RUNTIME
    with _GLOBAL_TASK_RUNTIME_LOCK:
        if _GLOBAL_TASK_RUNTIME is None or _GLOBAL_TASK_RUNTIME.closed:
            _GLOBAL_TASK_RUNTIME = _GlobalSubprocessTaskRuntime()
        return _GLOBAL_TASK_RUNTIME


def _shutdown_global_task_runtime() -> None:
    global _GLOBAL_TASK_RUNTIME
    runtime = _GLOBAL_TASK_RUNTIME
    if runtime is None:
        return
    runtime.close(kill=True)
    _GLOBAL_TASK_RUNTIME = None


atexit.register(_shutdown_global_task_runtime)


class LocalSubprocessActorPool:
    """Shared subprocess actor pool for one local UDF node."""

    def __init__(
        self,
        payload: dict[str, Any],
        pool_size: int,
        *,
        name: str | None = None,
        session_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.payload = dict(payload)
        self.session_config = (
            None if session_config is None else {str(key): str(value) for key, value in session_config.items()}
        )
        self.pool_size = max(1, int(pool_size))
        self.name = str(name or "")
        self._closed = False
        self._shutdown_finished = True
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._active = 0
        self._terminal_error: BaseException | None = None
        self._active_scopes: dict[int, ExecutionCancellationScope] = {}
        self._worker_generations = [0 for _ in range(self.pool_size)]
        self._replacing_workers: set[int] = set()
        self._replacing_executors: dict[int, _SingleSubprocessExecutor] = {}
        self._replacement_startup_cancel_requested = False
        self._replacement_cleanup_errors: list[BaseException] = []
        self._aborting_workers: set[tuple[int, int]] = set()
        pool_identity = self.name or str(id(self))
        self.admission_slots = LocalExecutionSlotPool(
            max_slots=self.pool_size,
            execution_slot_prefix=f"subprocess_actor:{pool_identity}",
        )
        self._idle_workers: deque[tuple[int, int]] = deque()
        self._workers: list[_SingleSubprocessExecutor] = []
        self._executor: ThreadPoolExecutor | None = None
        try:
            for worker_idx in range(self.pool_size):
                self._workers.append(
                    _SingleSubprocessExecutor(
                        self.payload,
                        worker_env=_worker_env_for_pool_index(self.payload, worker_idx, self.pool_size),
                        session_config=self.session_config,
                    )
                )
            self._executor = ThreadPoolExecutor(
                max_workers=self.pool_size,
                thread_name_prefix="vane-udf-subprocess-actor",
            )
        except BaseException as init_error:
            self._closed = True
            cleanup_errors: list[BaseException] = []
            executor = self._executor
            self._executor = None
            if executor is not None:
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            for worker in reversed(list(self._workers)):
                try:
                    worker.close(kill=True)
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            self._workers = []
            if cleanup_errors:
                details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
                raise RuntimeError(
                    f"local subprocess actor pool initialization cleanup failed: {details}"
                ) from init_error
            raise
        for worker_idx in range(self.pool_size):
            self._idle_workers.append((worker_idx, 0))
        _subprocess_debug_log(
            f"local_actor_pool_created name={self.name!r} pool_size={self.pool_size} worker_pids={self.worker_pids()}"
        )

    def worker_pids(self) -> list[int | None]:
        with self._lock:
            return [getattr(worker._proc, "pid", None) for worker in self._workers]

    def create_admission_authority(self) -> LocalSlotAdmissionAuthority:
        return self.admission_slots.create_authority()

    def first_proc(self) -> Any | None:
        with self._lock:
            if not self._workers:
                return None
            return self._workers[0]._proc

    def _wake_waiters(self) -> None:
        with self._cond:
            self._cond.notify_all()

    def _raise_if_unavailable_locked(self) -> None:
        if self._closed:
            raise RuntimeError("local subprocess actor pool is closed")
        if self._terminal_error is not None:
            raise RuntimeError(f"local subprocess actor pool failed: {self._terminal_error}") from self._terminal_error

    def _spawn_worker(self, worker_idx: int) -> _SingleSubprocessExecutor:
        return _SingleSubprocessExecutor(
            self.payload,
            worker_env=_worker_env_for_pool_index(self.payload, worker_idx, self.pool_size),
            session_config=self.session_config,
            startup_observer=lambda executor: self._track_replacing_executor(worker_idx, executor),
        )

    def _set_terminal_error(self, error: BaseException) -> None:
        should_close_admission = False
        with self._cond:
            if self._terminal_error is None:
                self._terminal_error = error
                should_close_admission = True
            self._cond.notify_all()
        if should_close_admission:
            self.admission_slots.close()

    def _track_replacing_executor(
        self,
        worker_idx: int,
        worker: _SingleSubprocessExecutor,
    ) -> None:
        cancel_startup = False
        with self._cond:
            replacing_executors = getattr(self, "_replacing_executors", None)
            if replacing_executors is None:
                replacing_executors = {}
                self._replacing_executors = replacing_executors
            if worker_idx in self._replacing_workers:
                replacing_executors[worker_idx] = worker
                cancel_startup = bool(getattr(self, "_replacement_startup_cancel_requested", False))
            self._cond.notify_all()
        if cancel_startup:
            worker._cancel_startup()

    def _interrupt_replacing_workers(self, interrupted_workers: set[int]) -> list[BaseException]:
        cleanup_errors: list[BaseException] = []
        workers_to_interrupt: list[_SingleSubprocessExecutor] = []
        with self._cond:
            self._replacement_startup_cancel_requested = True
            replacing_executors = getattr(self, "_replacing_executors", {})
            for worker_idx in self._replacing_workers:
                worker = replacing_executors.get(worker_idx)
                if worker is not None and id(worker) not in interrupted_workers:
                    interrupted_workers.add(id(worker))
                    workers_to_interrupt.append(worker)
            self._cond.notify_all()
        for worker in workers_to_interrupt:
            try:
                # A provisional actor has never accepted user work. Its
                # startup thread retains ownership of full cleanup.
                cancel_startup = getattr(worker, "_cancel_startup", None)
                if callable(cancel_startup):
                    cancel_startup()
                else:
                    worker.close(kill=True)
            except BaseException as exc:
                cleanup_errors.append(exc)
        return cleanup_errors

    def _close_replacing_workers(
        self,
        *,
        deadline: float | None = None,
        interrupted_workers: set[int] | None = None,
    ) -> list[BaseException]:
        cleanup_errors: list[BaseException] = []
        if interrupted_workers is None:
            interrupted_workers = set()
        if deadline is None:
            deadline = time.monotonic() + _subprocess_shutdown_grace_s()
        while True:
            cleanup_errors.extend(self._interrupt_replacing_workers(interrupted_workers))
            with self._cond:
                if not self._replacing_workers:
                    cleanup_errors.extend(getattr(self, "_replacement_cleanup_errors", ()))
                    self._replacement_cleanup_errors = []
                    return cleanup_errors
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    cleanup_errors.append(
                        TimeoutError(
                            "local subprocess actor replacement cleanup did not finish before the shutdown deadline"
                        )
                    )
                    cleanup_errors.extend(getattr(self, "_replacement_cleanup_errors", ()))
                    self._replacement_cleanup_errors = []
                    return cleanup_errors
                self._cond.wait(timeout=remaining)

    def _record_replacement_cleanup_error(
        self,
        worker_idx: int,
        role: str,
        error: BaseException,
    ) -> None:
        cleanup_error = RuntimeError(
            f"local subprocess actor {worker_idx} {role} cleanup failed: {type(error).__name__}: {error}"
        )
        with self._cond:
            errors = getattr(self, "_replacement_cleanup_errors", None)
            if errors is None:
                errors = []
                self._replacement_cleanup_errors = errors
            errors.append(cleanup_error)
            pool_closed = self._closed
            self._cond.notify_all()
        if not pool_closed:
            self._set_terminal_error(cleanup_error)

    def _replace_worker(
        self,
        worker_idx: int,
        worker_generation: int,
        failed_worker: _SingleSubprocessExecutor,
    ) -> None:
        try:
            try:
                failed_worker.close(kill=True)
            except BaseException as exc:
                self._record_replacement_cleanup_error(worker_idx, "failed worker", exc)
                return
            try:
                replacement = self._spawn_worker(worker_idx)
            except BaseException as exc:
                with self._cond:
                    pool_closed = self._closed
                if pool_closed and isinstance(exc, _SubprocessStartupCleanupError):
                    self._record_replacement_cleanup_error(worker_idx, "replacement startup", exc)
                elif not pool_closed:
                    self._set_terminal_error(
                        RuntimeError(f"failed to replace local subprocess actor {worker_idx}: {exc}")
                    )
                return

            close_replacement = False
            with self._cond:
                if (
                    self._closed
                    or worker_idx >= len(self._workers)
                    or self._worker_generations[worker_idx] != worker_generation
                    or self._workers[worker_idx] is not failed_worker
                ):
                    close_replacement = True
                else:
                    next_generation = worker_generation + 1
                    self._workers[worker_idx] = replacement
                    self._worker_generations[worker_idx] = next_generation
                    self._idle_workers.append((worker_idx, next_generation))
            if close_replacement:
                try:
                    replacement.close(kill=True)
                except BaseException as exc:
                    self._record_replacement_cleanup_error(worker_idx, "replacement", exc)
        finally:
            # Keep replacement ownership visible until a rejected replacement
            # has also finished closing. Shutdown uses this set as its join
            # condition, so clearing it before close() would let a newly
            # spawned process and its shared memory outlive the pool.
            with self._cond:
                getattr(self, "_replacing_executors", {}).pop(worker_idx, None)
                self._replacing_workers.discard(worker_idx)
                self._cond.notify_all()

    def _acquire_worker(
        self,
        scope: ExecutionCancellationScope,
    ) -> tuple[int, int, _SingleSubprocessExecutor]:
        unregister = scope.register_cancel_wakeup(self._wake_waiters)
        try:
            while True:
                replacement: tuple[int, int, _SingleSubprocessExecutor] | None = None
                with self._cond:
                    scope.raise_if_cancelled("local subprocess actor acquisition")
                    self._raise_if_unavailable_locked()
                    while self._idle_workers:
                        worker_idx, worker_generation = self._idle_workers.popleft()
                        if worker_idx >= len(self._workers):
                            continue
                        if self._worker_generations[worker_idx] != worker_generation:
                            continue
                        worker = self._workers[worker_idx]
                        if _worker_is_reusable(worker):
                            self._active += 1
                            self._active_scopes[worker_idx] = scope
                            return worker_idx, worker_generation, worker
                        if worker_idx not in self._replacing_workers:
                            self._replacing_workers.add(worker_idx)
                            replacement = (worker_idx, worker_generation, worker)
                        break
                    if replacement is None:
                        self._cond.wait()
                        continue
                assert replacement is not None
                self._replace_worker(*replacement)
        finally:
            unregister()

    def submit(
        self,
        fn: Callable[[_SingleSubprocessExecutor], Any | None],
        scope: ExecutionCancellationScope,
        debug_seq: int = 0,
    ) -> Future[Any]:
        with self._lock:
            self._raise_if_unavailable_locked()
            executor = self._executor
        if executor is None:
            raise RuntimeError("local subprocess actor pool is closed")
        return executor.submit(self._run, fn, scope, debug_seq)

    def _run(
        self,
        fn: Callable[[_SingleSubprocessExecutor], Any | None],
        scope: ExecutionCancellationScope,
        debug_seq: int = 0,
    ) -> Any | None:
        worker_idx: int | None = None
        worker_generation = 0
        worker: _SingleSubprocessExecutor | None = None
        worker_pid = None
        reusable = False
        replace_worker = False
        result: Any | None = None
        result_ready = False
        try:
            worker_idx, worker_generation, worker = self._acquire_worker(scope)
            with self._lock:
                active = self._active
            worker_pid = getattr(worker._proc, "pid", None)
            if _should_debug_submit(debug_seq):
                _subprocess_debug_log(
                    "local_actor_pool_worker_acquired "
                    f"name={self.name!r} seq={debug_seq} worker_idx={worker_idx} active={active} "
                    f"pool_size={self.pool_size} worker_pid={worker_pid}"
                )
            scope.raise_if_cancelled("local subprocess actor task")
            result = worker._run_in_execution_scope(scope, fn)
            result_ready = True
        finally:
            scope.finish()
            try:
                active_after = 0
                if worker_idx is not None and worker is not None:
                    reusable = _worker_is_reusable(worker)
                    with self._cond:
                        worker_identity = (worker_idx, worker_generation)
                        abort_requested = worker_identity in self._aborting_workers
                        self._aborting_workers.discard(worker_identity)
                        self._active = max(0, self._active - 1)
                        self._active_scopes.pop(worker_idx, None)
                        active_after = self._active
                        if not self._closed and reusable and not abort_requested:
                            self._idle_workers.append((worker_idx, worker_generation))
                        elif not self._closed and worker_idx not in self._replacing_workers:
                            self._replacing_workers.add(worker_idx)
                            replace_worker = True
                        self._cond.notify_all()
                    if replace_worker:
                        self._replace_worker(worker_idx, worker_generation, worker)
                if _should_debug_submit(debug_seq):
                    _subprocess_debug_log(
                        "local_actor_pool_worker_finished "
                        f"name={self.name!r} seq={debug_seq} worker_idx={worker_idx} "
                        f"active={active_after} reusable={reusable}"
                    )
            except BaseException:
                if result_ready:
                    _release_local_ref_bundle_result(result)
                raise
        try:
            scope.raise_if_cancelled("local subprocess actor result")
        except BaseException:
            _release_local_ref_bundle_result(result)
            raise
        return result

    def stats(self) -> dict[str, int]:
        with self._lock:
            active = self._active
            idle = len(self._idle_workers)
        return {
            "pool_size": self.pool_size,
            "active_workers": active,
            "idle_workers": idle,
        }

    def cancel_output_grants(self) -> None:
        with self._lock:
            workers = list(self._workers)
        cleanup_errors: list[BaseException] = []
        for worker in workers:
            try:
                worker.cancel_output_grants()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
            raise RuntimeError(
                f"local subprocess actor output-grant cancellation failed: {details}"
            ) from cleanup_errors[0]

    def abort_scopes(self, scopes: set[ExecutionCancellationScope]) -> None:
        workers: list[_SingleSubprocessExecutor] = []
        with self._cond:
            for worker_idx, scope in self._active_scopes.items():
                if scope not in scopes or worker_idx >= len(self._workers):
                    continue
                worker_generation = self._worker_generations[worker_idx]
                self._aborting_workers.add((worker_idx, worker_generation))
                workers.append(self._workers[worker_idx])
            self._cond.notify_all()
        cleanup_errors: list[BaseException] = []
        for worker in workers:
            try:
                worker.close(kill=True)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
            raise RuntimeError(f"local subprocess actor abort failed: {details}") from cleanup_errors[0]

    def shutdown(self, *, kill: bool = False) -> None:
        with self._cond:
            if self._closed:
                while not getattr(self, "_shutdown_finished", True):
                    self._cond.wait()
                return
            self._closed = True
            self._shutdown_finished = False
            self._replacement_startup_cancel_requested = bool(kill)
            active_scopes = list(self._active_scopes.values())
            self._cond.notify_all()
        try:
            cleanup_errors: list[BaseException] = []
            try:
                self.admission_slots.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
            for scope in active_scopes:
                scope.cancel("local subprocess actor pool closed")

            close_kill = bool(kill)
            if not close_kill:
                deadline = time.monotonic() + _subprocess_shutdown_grace_s()
                with self._cond:
                    while self._active > 0 or self._replacing_workers:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            close_kill = True
                            self._replacement_startup_cancel_requested = True
                            self._cond.notify_all()
                            break
                        self._cond.wait(timeout=remaining)
            interrupted_replacements: set[int] = set()
            if close_kill:
                cleanup_errors.extend(self._interrupt_replacing_workers(interrupted_replacements))
                try:
                    self.abort_scopes(set(active_scopes))
                except BaseException as exc:
                    cleanup_errors.append(exc)

            # Once graceful shutdown escalates, interrupt provisional actor
            # startup instead of waiting for the longer control timeout.
            replacement_cleanup_deadline = time.monotonic() + _subprocess_shutdown_grace_s()
            cleanup_errors.extend(
                self._close_replacing_workers(
                    deadline=replacement_cleanup_deadline,
                    interrupted_workers=interrupted_replacements,
                )
            )

            executor = self._executor
            self._executor = None
            if executor is not None:
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except BaseException as exc:
                    cleanup_errors.append(exc)
            with self._lock:
                workers = list(self._workers)
            for worker in workers:
                try:
                    worker.close(kill=close_kill)
                except BaseException as exc:
                    cleanup_errors.append(exc)
            with self._cond:
                self._workers = []
                self._idle_workers.clear()
                self._cond.notify_all()
            _subprocess_debug_log(f"local_actor_pool_shutdown name={self.name!r} kill={close_kill}")
            if cleanup_errors:
                details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
                raise RuntimeError(f"local subprocess actor pool shutdown failed: {details}") from cleanup_errors[0]
        finally:
            with self._cond:
                self._shutdown_finished = True
                self._cond.notify_all()

    def __del__(self) -> None:
        try:
            self.shutdown(kill=True)
        except Exception:
            pass


def _local_actor_pool_size_from_node(node: dict[str, Any], payload: dict[str, Any]) -> int:
    for container, key in (
        (payload, "actor_number"),
        (payload, "udf_worker_slots"),
        (node, "actor_pool_size"),
    ):
        value = container.get(key)
        if value is None:
            continue
        parsed = int(value)
        if parsed > 0:
            return parsed
    raise ValueError("subprocess_actor payload is missing actor_number/udf_worker_slots")


def _local_actor_pool_size_from_pool(actor_pool: Any) -> int:
    try:
        pool_size = int(actor_pool.pool_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("local_actor_pool.pool_size must be a positive integer") from exc
    if pool_size <= 0:
        raise ValueError("local_actor_pool.pool_size must be a positive integer")
    return pool_size


_LOCAL_ACTOR_POOL_CONTRACT_ERROR = (
    "local_actor_pool must expose submit(), create_admission_authority(), pool_size, stats(), "
    "cancel_output_grants(), abort_scopes(), first_proc(), and worker_pids()"
)
_LOCAL_ACTOR_POOL_REQUIRED_METHODS = (
    "submit",
    "create_admission_authority",
    "stats",
    "cancel_output_grants",
    "abort_scopes",
    "first_proc",
    "worker_pids",
)


def _validate_local_actor_pool_contract(actor_pool: Any) -> int:
    if not hasattr(actor_pool, "pool_size"):
        raise ValueError(_LOCAL_ACTOR_POOL_CONTRACT_ERROR)
    actor_pool_size = _local_actor_pool_size_from_pool(actor_pool)
    missing_methods = [
        method_name
        for method_name in _LOCAL_ACTOR_POOL_REQUIRED_METHODS
        if not callable(getattr(actor_pool, method_name, None))
    ]
    if missing_methods:
        raise ValueError(_LOCAL_ACTOR_POOL_CONTRACT_ERROR)
    return actor_pool_size


def ensure_local_subprocess_actor_pools_for_plan(
    plan: Any,
    conn: Any = None,
) -> tuple[list[LocalSubprocessActorPool], dict[str, Any]]:
    """Pre-create local subprocess actors and inject them into UDF nodes."""
    udf_nodes = plan.collect_udf_nodes(conn=conn)
    return ensure_local_subprocess_actor_pools_for_nodes(
        udf_nodes,
        plan_identity=id(plan),
        set_handles=lambda actor_options_map: plan.set_udf_actor_handles(actor_options_map, conn=conn),
    )


def ensure_local_subprocess_actor_pools_for_nodes(
    udf_nodes: Any,
    *,
    plan_identity: Any = None,
    set_handles: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[LocalSubprocessActorPool], dict[str, Any]]:
    """Pre-create local subprocess actors for already-collected UDF nodes."""
    created: list[LocalSubprocessActorPool] = []
    actor_options_map: dict[str, Any] = {}

    try:
        identity = id(udf_nodes) if plan_identity is None else plan_identity
        for node in udf_nodes:
            raw_payload = node.get("payload") or {}
            if not isinstance(raw_payload, dict):
                continue
            if str(raw_payload.get("execution_backend") or "").strip().lower() != "subprocess_actor":
                continue

            node_id = str(node.get("node_id"))
            pool_size = _local_actor_pool_size_from_node(node, raw_payload)
            if float(raw_payload.get("gpus") or 0.0) > 0.0:
                raise ValueError("GPU resources require a Ray UDF backend")
            executor_options = dict(node.get("executor_options") or {})
            session_config = _normalize_session_config_option(executor_options)
            existing_pool = executor_options.get("local_actor_pool")
            if existing_pool is not None:
                existing_pool_size = _validate_local_actor_pool_contract(existing_pool)
                if existing_pool_size != pool_size:
                    raise ValueError(
                        "pre-created local_actor_pool size does not match the UDF actor_number: "
                        f"pool_size={existing_pool_size} actor_number={pool_size}"
                    )
                existing_session_config = getattr(existing_pool, "session_config", None)
                if existing_session_config != session_config:
                    raise ValueError("pre-created local_actor_pool belongs to a different Vane session")
                actor_options_map[node_id] = executor_options
                continue
            pool_name = f"local-subprocess-actor-{identity}-{node_id}"
            pool_kwargs: dict[str, Any] = {"name": pool_name}
            if session_config is not None:
                pool_kwargs["session_config"] = session_config
            pool = LocalSubprocessActorPool(raw_payload, pool_size, **pool_kwargs)
            created.append(pool)
            executor_options["local_actor_pool"] = pool
            actor_options_map[node_id] = executor_options

        if actor_options_map and set_handles is not None:
            set_handles(actor_options_map)
    except BaseException as creation_error:
        cleanup_errors: list[BaseException] = []
        for pool in reversed(created):
            try:
                pool.shutdown(kill=True)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
            raise RuntimeError(f"local subprocess actor pool rollback failed: {details}") from creation_error
        raise

    return created, actor_options_map


class UDFExecutor(AdmissionExecutorMixin, BaseUDFExecutor):
    """Subprocess UDF executor with an optional worker pool."""

    def __init__(self, payload: dict[str, Any], options: dict[str, Any] | None = None) -> None:
        options = dict(options or {})
        session_config = _normalize_session_config_option(options)
        self._subprocess_mode = _payload_subprocess_mode(payload)
        self._pool_size = _payload_subprocess_pool_size(payload, self._subprocess_mode)
        _subprocess_debug_log(
            "executor_init "
            f"mode={self._subprocess_mode} backend={payload.get('execution_backend')!r} "
            f"pool_size={self._pool_size} payload_udf_worker_slots={payload.get('udf_worker_slots')!r} "
            f"actor_number={payload.get('actor_number')!r}"
        )
        self._closed = False
        self._finished_submitting = False
        self._wakeup: Callable[[], None] | None = None
        self._workers: list[_SingleSubprocessExecutor] = []
        self._executor: ThreadPoolExecutor | None = None
        self._idle_workers: queue.Queue[int] | None = None
        self._actor_pool: LocalSubprocessActorPool | None = None
        self._task_runtime: _GlobalSubprocessTaskRuntime | None = None
        self._task_pool: _TaskWorkerPool | None = None
        self._task_futures: set[Future[Any]] = set()
        self._task_futures_cv = threading.Condition()
        self._task_futures_lock = self._task_futures_cv
        self._task_future_meta: dict[
            Future[Any],
            tuple[int | None, int, float, AdmissionLease | None, ExecutionCancellationScope],
        ] = {}
        self._execution_owner_id = uuid.uuid4().hex
        self._execution_scope_generation = 0
        self._execution_scopes: set[ExecutionCancellationScope] = set()
        self._execution_scopes_lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._debug_submit_count = 0
        self._queue: deque[Any] = deque()
        self._result_admissions: deque[AdmissionLease | None] = deque()
        self._queue_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending_batches = 0
        self._ref_bundle_output = payload_requests_local_ref_bundle_output(payload)
        self._output_row_budget_bytes = _payload_output_row_budget_bytes(payload)
        self._learned_output_budget_bytes = 0
        self._last_output_budget_estimate_bytes = 0
        self._output_budget_lock = threading.Lock()
        self._active_input_leases: set[int] = set()
        self._active_input_leases_lock = threading.Lock()
        self._budget_wakeup_unregister: Callable[[], None] | None = None
        self._wakeup_error: BaseException | None = None

        try:
            if self._ref_bundle_output:
                self._budget_wakeup_unregister = register_local_shm_ref_budget_wakeup(self._notify_wakeup)

            if self._subprocess_mode == "task":
                self._task_runtime = _global_task_runtime()
                self._task_pool = self._task_runtime.acquire_pool(
                    payload,
                    self._pool_size,
                    session_config=session_config,
                )
                self._initialize_admission(self._task_pool.create_admission_authority())
                with self._task_runtime.cond:
                    task_pool_ref_count = self._task_pool.ref_count
                    task_pool_capacity = self._task_pool.pool_size
                _subprocess_debug_log(
                    "task_pool_acquired "
                    f"pool_size={self._task_pool.pool_size} ref_count={task_pool_ref_count} "
                    f"capacity={task_pool_capacity} runtime_max_workers={self._task_runtime.max_workers}"
                )
                return

            if options.get("local_actor_pool_name") is not None or payload.get("local_actor_pool_name") is not None:
                raise ValueError("local_actor_pool_name is unsupported; pass local_actor_pool in executor options")

            actor_pool = options.get("local_actor_pool")
            if actor_pool is None:
                raise RuntimeError(
                    "subprocess_actor requires a pre-created local_actor_pool; "
                    "call ensure_local_subprocess_actor_pools_for_plan before execution"
                )
            actor_pool_size = _validate_local_actor_pool_contract(actor_pool)
            if getattr(actor_pool, "session_config", None) != session_config:
                raise ValueError("local_actor_pool belongs to a different Vane session")
            worker_pids = actor_pool.worker_pids()
            self._actor_pool = actor_pool
            self._pool_size = actor_pool_size
            self._initialize_admission(actor_pool.create_admission_authority())
            _subprocess_debug_log(
                "local_actor_pool_attached "
                f"name={getattr(actor_pool, 'name', '')!r} pool_size={self._pool_size} "
                f"worker_pids={worker_pids}"
            )
            return
        except BaseException as init_error:
            try:
                self.close(kill=True)
            except BaseException as cleanup_error:
                raise RuntimeError(
                    f"UDF subprocess executor initialization cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                ) from init_error
            raise

    @property
    def _proc(self) -> Any | None:
        if self._actor_pool is not None:
            return self._actor_pool.first_proc()
        if not self._workers:
            return None
        return self._workers[0]._proc

    def _enqueue_result(self, item: Any | None) -> None:
        if item is not None:
            with self._queue_lock:
                self._queue.append(item)
        self._notify_wakeup()

    @staticmethod
    def _submit_result_item(submit_id: int | None, result: Any | None) -> Any:
        if submit_id is not None:
            return (SUBMIT_RESULT_MARKER, int(submit_id), result)
        return result if result is not None else (None, True)

    def _output_budget_estimate(self, num_rows: int | None) -> int:
        if not self._ref_bundle_output:
            return 0
        schema_estimate = _estimate_output_budget_from_rows(self._output_row_budget_bytes, num_rows)
        with self._output_budget_lock:
            learned_estimate = self._learned_output_budget_bytes
            estimate = max(schema_estimate, learned_estimate)
            if estimate > 0:
                self._last_output_budget_estimate_bytes = int(estimate)
        return estimate

    def _record_output_budget_result(self, result: Any | None) -> None:
        if not self._ref_bundle_output or result is None:
            return
        size = estimate_local_shm_ref_bundle_ipc_size(result)
        if size <= 0:
            return
        with self._output_budget_lock:
            self._learned_output_budget_bytes = max(self._learned_output_budget_bytes, int(size))
            self._last_output_budget_estimate_bytes = max(self._last_output_budget_estimate_bytes, int(size))

    def _output_budget_stats(self) -> dict[str, int]:
        if not self._ref_bundle_output:
            return {}
        with self._output_budget_lock:
            estimated_bytes = max(0, int(self._last_output_budget_estimate_bytes))
        with self._pending_lock:
            pending_batches = max(0, int(self._pending_batches))
        projected_output_bytes = pending_batches * estimated_bytes
        budget_snapshot = local_shm_ref_budget_snapshot()
        return {
            "udf_output_budget_available": int(
                can_admit_local_shm_ref_output_submit(
                    estimated_bytes,
                    projected_output_bytes=projected_output_bytes,
                )
            ),
            "udf_output_budget_estimated_bytes": estimated_bytes,
            "udf_output_budget_limit_bytes": int(budget_snapshot.get("limit_bytes", 0)),
            "udf_output_budget_usage_bytes": int(budget_snapshot.get("usage_bytes", 0)),
            "udf_output_budget_reserved_bytes": int(budget_snapshot.get("reserved_bytes", 0)),
            "udf_output_budget_pending_output_bytes": int(budget_snapshot.get("pending_output_bytes", 0)),
            "udf_local_shm_budget_limit_bytes": int(budget_snapshot.get("limit_bytes", 0)),
            "udf_local_shm_allocated_bytes": int(budget_snapshot.get("allocated_bytes", 0)),
            "udf_local_shm_output_grant_bytes": int(budget_snapshot.get("output_grant_bytes", 0)),
            "udf_local_shm_output_credit_bytes": int(budget_snapshot.get("output_credit_bytes", 0)),
            "udf_local_shm_input_lease_bytes": int(budget_snapshot.get("input_lease_bytes", 0)),
            "udf_local_shm_available_bytes": int(budget_snapshot.get("available_bytes", 0)),
            "udf_local_shm_active_input_leases": int(budget_snapshot.get("active_input_leases", 0)),
            "udf_local_shm_active_output_credits": int(budget_snapshot.get("active_output_credits", 0)),
            "udf_local_shm_waiting_output_grants": int(budget_snapshot.get("waiting_output_grants", 0)),
            "udf_local_shm_input_consumed_count": int(budget_snapshot.get("input_consumed_count", 0)),
            "udf_local_shm_refs_released_by_input_ack": int(budget_snapshot.get("refs_released_by_input_ack", 0)),
            "udf_local_shm_oversized_output_grants": int(budget_snapshot.get("oversized_output_grants", 0)),
        }

    def _track_task_future(
        self,
        future: Future[Any],
        submit_id: int | None,
        debug_seq: int,
        submit_start: float,
        admission: AdmissionLease | None,
        scope: ExecutionCancellationScope,
    ) -> None:
        with self._task_futures_lock:
            self._task_futures.add(future)
            self._task_future_meta[future] = (
                submit_id,
                debug_seq,
                submit_start,
                admission,
                scope,
            )

        def complete_task(done: Future[Any], submit_id: int | None = submit_id) -> None:
            self._complete_task_submit(submit_id, done)

        future.add_done_callback(complete_task)

    def _new_execution_scope(self) -> ExecutionCancellationScope:
        with self._execution_scopes_lock:
            if self._closed:
                raise RuntimeError("UDF subprocess executor is closed")
            self._execution_scope_generation += 1
            scope = ExecutionCancellationScope(
                self._execution_owner_id,
                self._execution_scope_generation,
            )
            self._execution_scopes.add(scope)
            return scope

    def _retire_execution_scope(self, scope: ExecutionCancellationScope) -> None:
        scope.finish()
        with self._execution_scopes_lock:
            self._execution_scopes.discard(scope)

    def _cleanup_unscheduled_submit(
        self,
        scope: ExecutionCancellationScope,
        admission: AdmissionLease | None,
    ) -> None:
        cleanup_errors: list[BaseException] = []
        if admission is not None:
            try:
                admission.release()
            except BaseException as exc:
                cleanup_errors.append(exc)
        try:
            self._retire_execution_scope(scope)
        except BaseException as exc:
            cleanup_errors.append(exc)
        if cleanup_errors:
            details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
            raise RuntimeError(f"unscheduled UDF submit cleanup failed: {details}") from cleanup_errors[0]

    def _cancel_execution_scopes(self, reason: str) -> set[ExecutionCancellationScope]:
        lock = getattr(self, "_execution_scopes_lock", None)
        if lock is None:
            return set()
        with lock:
            scopes = set(self._execution_scopes)
        for scope in scopes:
            scope.cancel(reason)
        return scopes

    def _notify_wakeup(self) -> None:
        callback = self._wakeup
        if callback is None:
            return
        try:
            callback()
        except Exception as exc:
            self._record_wakeup_error(exc)

    def _record_wakeup_error(self, exc: BaseException) -> None:
        if self._wakeup_error is None:
            self._wakeup_error = exc

    def _track_input_lease(self, lease_id: int) -> None:
        with self._active_input_leases_lock:
            self._active_input_leases.add(int(lease_id))

    def _untrack_input_lease(self, lease_id: int) -> None:
        with self._active_input_leases_lock:
            self._active_input_leases.discard(int(lease_id))

    def _cancel_active_input_leases(self) -> None:
        with self._active_input_leases_lock:
            lease_ids = list(self._active_input_leases)
            self._active_input_leases.clear()
        cleanup_errors: list[BaseException] = []
        for lease_id in lease_ids:
            try:
                cancel_local_shm_input_lease(lease_id, name="udf-input-close")
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
            raise RuntimeError(f"UDF input lease cancellation failed: {details}") from cleanup_errors[0]

    def _complete_task_submit(self, submit_id: int | None, future: Future[Any]) -> None:
        item: Any | None = None
        result: Any | None = None
        debug_meta: (
            tuple[
                int | None,
                int,
                float,
                AdmissionLease | None,
                ExecutionCancellationScope,
            ]
            | None
        ) = None
        admission: AdmissionLease | None = None
        scope: ExecutionCancellationScope | None = None
        try:
            result = future.result()
            self._record_output_budget_result(result)
            item = self._submit_result_item(submit_id, result)
        except BaseException as exc:
            _release_local_ref_bundle_result(result)
            result = None
            item = (SUBMIT_RESULT_MARKER, int(submit_id), exc) if submit_id is not None else exc
        finally:
            with self._task_futures_lock:
                debug_meta = self._task_future_meta.get(future)
            if debug_meta is not None:
                debug_submit_id, debug_seq, submit_start, admission, scope = debug_meta
                try:
                    if _should_debug_submit(debug_seq):
                        _subprocess_debug_log(
                            "task_submit_completed "
                            f"seq={debug_seq} submit_id={debug_submit_id} "
                            f"total_s={time.perf_counter() - submit_start:.6f}"
                        )
                except BaseException as exc:
                    self._record_wakeup_error(exc)
            if item is not None:
                with self._queue_lock:
                    if self._closed:
                        _release_local_ref_bundle_result(result)
                        if admission is not None:
                            try:
                                admission.release()
                            except BaseException as exc:
                                self._record_wakeup_error(exc)
                    else:
                        self._queue.append(item)
                        self._result_admissions.append(admission)
            elif admission is not None:
                try:
                    admission.release()
                except BaseException as exc:
                    self._record_wakeup_error(exc)
            with self._pending_lock:
                self._pending_batches = max(0, self._pending_batches - 1)
            with self._task_futures_cv:
                self._task_futures.discard(future)
                self._task_future_meta.pop(future, None)
                self._task_futures_cv.notify_all()
            if scope is not None:
                self._retire_execution_scope(scope)
            self._notify_wakeup()

    def _submit_async(
        self,
        submit_id: int | None,
        fn: Callable[[_SingleSubprocessExecutor], Any | None],
        admission: AdmissionLease | None = None,
    ) -> None:
        lifecycle_lock = getattr(self, "_lifecycle_lock", None)
        if lifecycle_lock is None:
            lifecycle_lock = threading.RLock()
            self._lifecycle_lock = lifecycle_lock
        with lifecycle_lock:
            if self._closed:
                if admission is not None:
                    admission.release()
                raise RuntimeError("UDF subprocess executor is closed")
            self._debug_submit_count += 1
            debug_seq = self._debug_submit_count
            submit_start = time.perf_counter()
            try:
                scope = self._new_execution_scope()
            except BaseException:
                if admission is not None:
                    admission.release()
                raise

            if self._actor_pool is not None:
                actor_pool = self._actor_pool
                with self._pending_lock:
                    self._pending_batches += 1
                    pending = self._pending_batches
                try:
                    if _should_debug_submit(debug_seq):
                        pool_stats = actor_pool.stats()
                        _subprocess_debug_log(
                            "local_actor_pool_submit_scheduled "
                            f"name={getattr(actor_pool, 'name', '')!r} seq={debug_seq} "
                            f"submit_id={submit_id} pending={pending} pool_size={actor_pool.pool_size} "
                            f"pool_active={pool_stats.get('active_workers', 0)} "
                            f"pool_idle={pool_stats.get('idle_workers', 0)}"
                        )
                    future = actor_pool.submit(fn, scope, debug_seq)
                    self._track_task_future(
                        future,
                        submit_id,
                        debug_seq,
                        submit_start,
                        admission,
                        scope,
                    )
                except BaseException as submit_error:
                    with self._pending_lock:
                        self._pending_batches = max(0, self._pending_batches - 1)
                    try:
                        self._cleanup_unscheduled_submit(scope, admission)
                    except BaseException as cleanup_error:
                        raise RuntimeError(
                            f"UDF subprocess actor submit failed: {type(submit_error).__name__}: "
                            f"{submit_error}; {cleanup_error}"
                        ) from submit_error
                    raise
                return

            if self._task_pool is not None:
                runtime = self._task_runtime
                task_pool = self._task_pool
                if runtime is None:
                    missing_runtime_error = RuntimeError("global subprocess task runtime is not available")
                    try:
                        self._cleanup_unscheduled_submit(scope, admission)
                    except BaseException as cleanup_error:
                        raise RuntimeError(f"{missing_runtime_error}; {cleanup_error}") from missing_runtime_error
                    raise missing_runtime_error
                with self._pending_lock:
                    self._pending_batches += 1
                    pending = self._pending_batches
                try:
                    if _should_debug_submit(debug_seq):
                        with runtime.cond:
                            _subprocess_debug_log(
                                "task_submit_scheduled "
                                f"seq={debug_seq} submit_id={submit_id} pending={pending} "
                                f"pool_size={task_pool.pool_size} "
                                f"pool_total={task_pool.total} pool_active={task_pool.active} "
                                f"pool_idle={len(task_pool.idle)} "
                                f"runtime_total_workers={runtime.total_workers} "
                                f"runtime_max_workers={runtime.max_workers}"
                            )
                    future = runtime.submit(task_pool, fn, scope, debug_seq)
                    self._track_task_future(
                        future,
                        submit_id,
                        debug_seq,
                        submit_start,
                        admission,
                        scope,
                    )
                except BaseException as submit_error:
                    with self._pending_lock:
                        self._pending_batches = max(0, self._pending_batches - 1)
                    try:
                        self._cleanup_unscheduled_submit(scope, admission)
                    except BaseException as cleanup_error:
                        raise RuntimeError(
                            f"UDF subprocess task submit failed: {type(submit_error).__name__}: "
                            f"{submit_error}; {cleanup_error}"
                        ) from submit_error
                    raise
                return

            missing_owner_error = RuntimeError(
                "subprocess executor is not initialized with an actor or task worker owner"
            )
            try:
                self._cleanup_unscheduled_submit(scope, admission)
            except BaseException as cleanup_error:
                raise RuntimeError(f"{missing_owner_error}; {cleanup_error}") from missing_owner_error
            raise missing_owner_error

    def submit(self, args: pa.Table) -> None:
        table = _ensure_table(args)
        admission = self._take_task_admission()
        self._submit_async(
            None,
            lambda worker: worker._submit_table(table),
            admission,
        )

    def submit_with_id(self, submit_id: int, args: pa.Table) -> None:
        table = _ensure_table(args)
        admission = self._take_task_admission()
        self._submit_async(
            int(submit_id),
            lambda worker: worker._submit_table(table),
            admission,
        )

    def submit_ref_bundle_with_id(
        self,
        submit_id: int,
        block_refs: Any,
        slices: Any,
        metadata: Any,
        names: Any,
    ) -> None:
        worker_payload, lease_id = _make_local_ref_bundle_worker_payload_with_lease(
            block_refs,
            slices,
            metadata,
            names,
            submit_id=int(submit_id),
            name=f"udf-input-{int(submit_id)}",
            reserve_output_credit=self._ref_bundle_output,
        )
        if worker_payload is not None:
            assert lease_id is not None
            self._track_input_lease(lease_id)

            def submit_worker(
                worker: _SingleSubprocessExecutor,
                payload: dict[str, Any] = worker_payload,
                lease_id: int = lease_id,
            ) -> Any | None:
                try:
                    return worker._submit_ref_bundle_direct(payload)
                except BaseException as submit_error:
                    try:
                        cancel_local_shm_input_lease(lease_id, name=f"udf-input-{int(submit_id)}")
                    except BaseException as cleanup_error:
                        broken_cleanup_details = worker._mark_broken_after_cleanup_failure(cleanup_error)
                        raise RuntimeError(
                            f"UDF ref-bundle worker submit failed: {type(submit_error).__name__}: "
                            f"{submit_error}; input-lease cleanup failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}{broken_cleanup_details}"
                        ) from submit_error
                    raise
                finally:
                    self._untrack_input_lease(lease_id)

            try:
                admission = self._take_task_admission()
                self._submit_async(
                    int(submit_id),
                    submit_worker,
                    admission,
                )
            except BaseException as submit_error:
                cleanup_error: BaseException | None = None
                try:
                    cancel_local_shm_input_lease(lease_id, name=f"udf-input-{int(submit_id)}")
                except BaseException as exc:
                    cleanup_error = exc
                finally:
                    self._untrack_input_lease(lease_id)
                if cleanup_error is not None:
                    raise RuntimeError(
                        f"UDF ref-bundle scheduling failed: {type(submit_error).__name__}: "
                        f"{submit_error}; input-lease cleanup failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    ) from submit_error
                raise
            return
        raise RuntimeError("subprocess UDF ref-bundle input requires local shared-memory descriptors")

    def submit_ref_bundle(self, _block_refs: Any, _slices: Any, _metadata: Any, _names: Any) -> None:
        raise RuntimeError(
            "subprocess UDF ref-bundle submission requires submit_ref_bundle_with_id() and a pregranted admission lease"
        )

    def take_ready_result(self) -> Any | None:
        if self._wakeup_error is not None:
            raise RuntimeError(f"UDF subprocess wakeup callback failed: {self._wakeup_error}") from self._wakeup_error
        with self._queue_lock:
            try:
                result = self._queue.popleft()
            except IndexError:
                return None
            result_admissions = getattr(self, "_result_admissions", None)
            admission = result_admissions.popleft() if result_admissions else None
        if admission is not None:
            admission.release()
        return result

    def finished_submitting(self) -> None:
        self._finished_submitting = True

    def all_tasks_finished(self) -> bool:
        with self._queue_lock:
            queue_empty = not self._queue
        with self._pending_lock:
            pending_empty = self._pending_batches == 0
        return self._finished_submitting and queue_empty and pending_empty

    def stats(self) -> dict[str, int]:
        if self._wakeup_error is not None:
            raise RuntimeError(f"UDF subprocess wakeup callback failed: {self._wakeup_error}") from self._wakeup_error
        with self._pending_lock:
            pending = max(0, int(self._pending_batches))
        max_running = max(1, int(self._pool_size))
        running = min(pending, max_running)
        if self._task_pool is not None and self._task_runtime is not None:
            with self._task_runtime.cond:
                running = min(pending, max(0, int(self._task_pool.active)))
        elif self._actor_pool is not None:
            pool_stats = self._actor_pool.stats()
            running = min(pending, max(0, int(pool_stats.get("active_workers", 0))))
        queued = max(0, pending - running)
        stats = {
            "udf_running_task_count": running,
            "udf_queued_task_count": queued,
            "udf_max_running_tasks": max_running,
        }
        stats.update(self._output_budget_stats())
        return stats

    def register_wakeup(self, callback: Callable[[], None]) -> None:
        self._wakeup = callback
        self._admission_authority.register_wakeup(callback)

    def _cancel_pending_futures(self) -> None:
        cleanup_errors: list[BaseException] = []
        with self._task_futures_cv:
            for future in list(self._task_futures):
                try:
                    future.cancel()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            self._task_futures_cv.notify_all()
        if cleanup_errors:
            details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
            raise RuntimeError(f"UDF pending future cancellation failed: {details}") from cleanup_errors[0]

    def _cancel_local_shm_waits(
        self,
    ) -> tuple[set[ExecutionCancellationScope], list[BaseException]]:
        cleanup_errors: list[BaseException] = []
        scopes: set[ExecutionCancellationScope] = set()
        try:
            scopes = self._cancel_execution_scopes("UDF executor closed")
        except BaseException as exc:
            cleanup_errors.append(exc)
        try:
            self._cancel_active_input_leases()
        except BaseException as exc:
            cleanup_errors.append(exc)
        try:
            wake_local_shm_ref_budget_waiters()
        except BaseException as exc:
            cleanup_errors.append(exc)
        return scopes, cleanup_errors

    def _wait_for_pending_futures(self, timeout_s: float | None = None) -> bool:
        deadline = None if timeout_s is None else time.monotonic() + max(0.0, float(timeout_s))
        with self._task_futures_cv:
            while self._task_futures:
                if deadline is None:
                    self._task_futures_cv.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._task_futures_cv.wait(timeout=remaining)
            return True

    def close(self, kill: bool = False) -> None:
        lifecycle_lock = getattr(self, "_lifecycle_lock", None)
        if lifecycle_lock is None:
            lifecycle_lock = threading.RLock()
            self._lifecycle_lock = lifecycle_lock
        with lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            self._close_after_marked_closed(kill=kill)

    def _close_after_marked_closed(self, *, kill: bool) -> None:
        cleanup_errors: list[BaseException] = []
        authority = getattr(self, "_admission_authority", None)
        if authority is not None:
            try:
                authority.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        queue_lock = getattr(self, "_queue_lock", None)
        result_queue = getattr(self, "_queue", None)
        result_admissions = getattr(self, "_result_admissions", None)
        if queue_lock is not None:
            with queue_lock:
                queued_results = list(result_queue or ())
                if result_queue is not None:
                    result_queue.clear()
                admissions = list(result_admissions or ())
                if result_admissions is not None:
                    result_admissions.clear()
        else:
            queued_results = []
            admissions = []
        for result in queued_results:
            _release_local_ref_bundle_result(result)
        for admission in admissions:
            if admission is not None:
                try:
                    admission.release()
                except BaseException as exc:
                    cleanup_errors.append(exc)
        budget_wakeup_unregister = self._budget_wakeup_unregister
        self._budget_wakeup_unregister = None
        if budget_wakeup_unregister is not None:
            try:
                budget_wakeup_unregister()
            except BaseException as exc:
                cleanup_errors.append(exc)
        cancelled_scopes, wait_cleanup_errors = self._cancel_local_shm_waits()
        cleanup_errors.extend(wait_cleanup_errors)
        close_kill = bool(kill)
        if close_kill:
            actor_pool = self._actor_pool
            if actor_pool is not None:
                try:
                    actor_pool.abort_scopes(cancelled_scopes)
                except BaseException as exc:
                    cleanup_errors.append(exc)
            task_pool = self._task_pool
            if task_pool is not None:
                try:
                    task_pool.abort_scopes(cancelled_scopes)
                except BaseException as exc:
                    cleanup_errors.append(exc)
            try:
                self._cancel_pending_futures()
            except BaseException as exc:
                cleanup_errors.append(exc)
        else:
            if not self._wait_for_pending_futures(_subprocess_shutdown_grace_s()):
                close_kill = True
                actor_pool = self._actor_pool
                if actor_pool is not None:
                    try:
                        actor_pool.abort_scopes(cancelled_scopes)
                    except BaseException as exc:
                        cleanup_errors.append(exc)
                task_pool = self._task_pool
                if task_pool is not None:
                    try:
                        task_pool.abort_scopes(cancelled_scopes)
                    except BaseException as exc:
                        cleanup_errors.append(exc)
                try:
                    self._cancel_pending_futures()
                except BaseException as exc:
                    cleanup_errors.append(exc)
                _, escalation_cleanup_errors = self._cancel_local_shm_waits()
                cleanup_errors.extend(escalation_cleanup_errors)
        actor_pool = self._actor_pool
        if actor_pool is not None:
            self._actor_pool = None
            if cleanup_errors:
                details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
                raise RuntimeError(f"UDF subprocess executor close failed: {details}") from cleanup_errors[0]
            return
        task_pool = self._task_pool
        if task_pool is not None:
            self._task_pool = None
            try:
                task_pool.release_ref(kill=close_kill)
            except BaseException as exc:
                cleanup_errors.append(exc)
            if cleanup_errors:
                details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
                raise RuntimeError(f"UDF subprocess executor close failed: {details}") from cleanup_errors[0]
            return
        executor = self._executor
        self._executor = None
        if executor is not None:
            try:
                executor.shutdown(wait=not close_kill, cancel_futures=True)
            except BaseException as exc:
                cleanup_errors.append(exc)
        for worker in list(self._workers):
            try:
                worker.close(kill=close_kill)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
            raise RuntimeError(f"UDF subprocess executor close failed: {details}") from cleanup_errors[0]

    def __del__(self) -> None:
        try:
            self.close(kill=True)
        except Exception:
            pass


def _cleanup_subprocess_executor(
    proc: subprocess.Popen[bytes] | None,
    sock: socket.socket | None,
    payload_shm: shared_memory.SharedMemory | None,
    data_shm: shared_memory.SharedMemory | None,
) -> None:
    if sock is not None:
        try:
            if proc is not None and proc.poll() is None:
                _send_message(sock, _MSG_CLOSE)
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass
    if proc is not None and proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=_subprocess_control_timeout_s())
        except Exception:
            pass
    for shm in (payload_shm, data_shm):
        if shm is None:
            continue
        try:
            shm.close()
        except Exception:
            pass
        try:
            _unlink_shm(shm, track=False)
        except Exception:
            pass


__all__ = ["UDFExecutor"]

# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Subprocess + shared-memory UDF executor."""

from __future__ import annotations

import atexit
import hashlib
import os
import queue
import socket
import struct
import subprocess
import sys
import threading
import time
import weakref
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future
    from multiprocessing import shared_memory

from vane import pickle as duckdb_pickle
from vane.execution._common import ensure_table as _ensure_table
from vane.execution.ref_bundle import (
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
    request_local_shm_output_grant,
    wake_local_shm_ref_budget_waiters,
)
from vane.execution.udf_admission import (
    AdmissionExecutorMixin,
    AdmissionLease,
    LocalExecutionSlotPool,
    LocalSlotAdmissionAuthority,
)
from vane.execution.udf_threading import (
    worker_thread_env as _worker_thread_env,
)
from vane.execution.unified_executor import UDFExecutor as BaseUDFExecutor

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
    print(f"[vane-udf-worker-slots pid={os.getpid()}] {message}", file=sys.stderr, flush=True)


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
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
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
    block_refs,
    slices,
    metadata,
    names,
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
    if required > len(shm.buf):
        raise BufferError("shared memory segment is too small")
    _IPC_HEADER.pack_into(shm.buf, 0, len(ipc_bytes))
    shm.buf[_IPC_HEADER.size : required] = ipc_bytes
    return required


def _read_ipc_from_shm(shm: shared_memory.SharedMemory, size: int | None = None) -> bytes:
    if len(shm.buf) < _IPC_HEADER.size:
        raise BufferError("shared memory segment is too small for IPC header")
    ipc_size = _IPC_HEADER.unpack_from(shm.buf, 0)[0]
    required = _IPC_HEADER.size + ipc_size
    if required > len(shm.buf):
        raise BufferError(
            f"shared memory IPC payload exceeds local mapping: required={required} capacity={len(shm.buf)}"
        )
    if size is not None and required > size:
        raise BufferError(f"shared memory IPC payload exceeds response size: required={required} size={size}")
    return bytes(shm.buf[_IPC_HEADER.size : required])


class _SingleSubprocessExecutor(BaseUDFExecutor):
    """Run Python UDFs in one long-lived worker subprocess."""

    def __init__(self, payload: dict[str, Any], *, worker_env: dict[str, str] | None = None) -> None:
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
        self._active_input_leases: set[int] = set()
        self._active_input_leases_lock = threading.Lock()
        self._active_output_grants: set[int] = set()
        self._active_output_grants_lock = threading.Lock()
        self._output_grant_cancel_event = threading.Event()

        self._payload_shm: shared_memory.SharedMemory | None = None
        self._data_shm: shared_memory.SharedMemory | None = None
        self._sock: socket.socket | None = None
        self._proc: subprocess.Popen[bytes] | None = None

        self._start_worker(payload)
        self._finalizer = weakref.finalize(
            self,
            _cleanup_subprocess_executor,
            self._proc,
            self._sock,
            self._payload_shm,
            self._data_shm,
        )

    def _start_worker(self, payload: dict[str, Any]) -> None:
        payload_bytes = duckdb_pickle.dumps(payload)
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
            env = dict(os.environ)
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

        msg_type, payload_data = self._recv_expected(
            (_MSG_READY, _MSG_ERROR),
            timeout_s=_subprocess_control_timeout_s(),
        )
        if msg_type == _MSG_ERROR:
            self._mark_broken(payload_data.decode("utf-8", errors="replace"))
            raise RuntimeError(self._broken_error)

        # The worker has loaded the payload. The parent no longer needs this shm.
        self._close_payload_shm()

    def _recv_expected(self, expected: tuple[int, ...], *, timeout_s: float | None = None) -> tuple[int, bytes]:
        sock = self._require_socket()
        restore_timeout = None
        if timeout_s is not None and hasattr(sock, "settimeout") and hasattr(sock, "gettimeout"):
            restore_timeout = sock.gettimeout()
            sock.settimeout(max(0.0, float(timeout_s)))
        try:
            msg_type, payload = _recv_message(sock)
        except Exception as exc:
            self._mark_broken(f"UDF subprocess communication failed: {exc}", actor_lost=True)
            raise RuntimeError(self._broken_error) from exc
        finally:
            if restore_timeout is not None:
                try:
                    sock.settimeout(restore_timeout)
                except Exception:
                    pass
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
        for lease_id in lease_ids:
            cancel_local_shm_input_lease(lease_id, name="udf-input-close")

    def _track_output_grant(self, grant_id: int) -> None:
        if int(grant_id) <= 0:
            return
        with self._active_output_grants_lock:
            self._active_output_grants.add(int(grant_id))

    def _untrack_output_grant(self, grant_id: int) -> None:
        if int(grant_id) <= 0:
            return
        with self._active_output_grants_lock:
            self._active_output_grants.discard(int(grant_id))

    def _release_output_grant(self, grant_id: int, *, name: str) -> None:
        if int(grant_id) <= 0:
            return
        try:
            release_local_shm_output_grant(int(grant_id), name=name)
        finally:
            self._untrack_output_grant(int(grant_id))

    def _release_active_output_grants(self, *, name: str = "udf-output-close") -> None:
        with self._active_output_grants_lock:
            grant_ids = list(self._active_output_grants)
            self._active_output_grants.clear()
        for grant_id in grant_ids:
            release_local_shm_output_grant(grant_id, name=name)

    def _mark_broken(self, error: str, *, actor_lost: bool = False) -> None:
        self._actor_lost = self._actor_lost or actor_lost
        if self._broken_error is None:
            self._broken_error = error
        self._cancel_active_input_leases()
        self._release_active_output_grants(name="udf-output-broken")
        self.close(kill=True)

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

        _marker, refs, metadata, names = make_local_shm_ref_bundle_result(args)
        lease_id = None
        try:
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
        except BaseException:
            if lease_id is not None:
                cancel_local_shm_input_lease(lease_id, name="udf-materialized-input")
            raise
        finally:
            for ref in refs:
                try:
                    ref.release()
                except Exception:
                    pass

    def _handle_submit_control_message(self, msg_type: int, payload: bytes) -> bool:
        if msg_type == _MSG_INPUT_CONSUMED:
            event = duckdb_pickle.loads(payload)
            lease_id = int(event["input_lease_id"])
            consume_local_shm_input_lease(lease_id, name="udf-input")
            self._untrack_input_lease(lease_id)
            self._notify_wakeup()
            return True
        if msg_type == _MSG_INPUT_CONSUME_FAILED:
            event = duckdb_pickle.loads(payload)
            lease_id = int(event["input_lease_id"])
            cancel_local_shm_input_lease(lease_id, name="udf-input")
            self._untrack_input_lease(lease_id)
            self._notify_wakeup()
            return True
        if msg_type == _MSG_OUTPUT_GRANT_REQUEST:
            event = duckdb_pickle.loads(payload)
            request_id = int(event.get("request_id", 0))
            size = int(event["size_bytes"])
            priority = str(event.get("priority") or "consumer")
            input_lease_id_raw = event.get("input_lease_id")
            input_lease_id = int(input_lease_id_raw) if input_lease_id_raw is not None else None
            try:
                grant_id = request_local_shm_output_grant(
                    size,
                    name=f"udf-output-{request_id}",
                    priority=priority,
                    input_lease_id=input_lease_id,
                    cancel_event=self._output_grant_cancel_event,
                )
            except BaseException as exc:
                _send_message(
                    self._require_socket(),
                    _MSG_OUTPUT_GRANT_CANCELLED,
                    str(exc).encode("utf-8", errors="replace"),
                )
                return True
            self._track_output_grant(grant_id)
            response = {"request_id": request_id, "grant_id": int(grant_id)}
            try:
                _send_message(self._require_socket(), _MSG_OUTPUT_GRANT_GRANTED, duckdb_pickle.dumps(response))
            except BaseException as exc:
                self._release_output_grant(grant_id, name=f"udf-output-{request_id}-send-failed")
                self._mark_broken(f"UDF subprocess output grant response failed: {exc}", actor_lost=True)
                raise RuntimeError(self._broken_error) from exc
            self._notify_wakeup()
            return True
        if msg_type == _MSG_OUTPUT_GRANT_RELEASE:
            event = duckdb_pickle.loads(payload)
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
                )
            )
            if self._handle_submit_control_message(msg_type, payload):
                msg_type = None
                continue
            if msg_type == _MSG_ERROR:
                error = payload.decode("utf-8", errors="replace")
                self._mark_broken(error)
                raise RuntimeError(error)
            if msg_type == _MSG_REF_BUNDLE_RESULT:
                descriptor = duckdb_pickle.loads(payload)
                grant_id_raw = descriptor.get("grant_id") if isinstance(descriptor, dict) else None
                grant_id = int(grant_id_raw) if grant_id_raw is not None else None
                try:
                    result = make_local_shm_ref_bundle_result_from_descriptor(descriptor, block_on_budget=False)
                except BaseException:
                    if grant_id is not None:
                        self._release_output_grant(grant_id, name="udf-output-descriptor-wrap-failed")
                    raise
                if grant_id is not None:
                    self._untrack_output_grant(grant_id)
                return result
            if len(payload) != 8:
                self._mark_broken("UDF subprocess returned malformed OK response", actor_lost=True)
                raise RuntimeError(self._broken_error)

            result_size = struct.unpack("<Q", payload)[0]
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
            if result_size > len(data_shm.buf):
                name = data_shm.name
                data_shm.close()
                self._data_shm = data_shm = _open_existing_shm(name, track=False)
            ipc_result = _read_ipc_from_shm(data_shm, result_size)
            return self._wrap_output(_arrow_table_from_ipc_bytes(ipc_result))

    def _submit_ref_bundle_direct(self, payload: dict[str, Any]) -> Any | None:
        if payload.get("estimated_num_rows") == 0:
            return None
        sock = self._require_socket()
        lease_id_raw = payload.get("input_lease_id")
        lease_id = int(lease_id_raw) if lease_id_raw is not None else None
        if lease_id is not None:
            self._track_input_lease(lease_id)
        try:
            payload_bytes = duckdb_pickle.dumps(payload)
            _send_message(sock, _MSG_SUBMIT_REF_BUNDLE, payload_bytes)
        except Exception as exc:
            if lease_id is not None:
                cancel_local_shm_input_lease(lease_id, name="udf-input")
                self._untrack_input_lease(lease_id)
            self._mark_broken(f"UDF subprocess ref-bundle submit failed: {exc}", actor_lost=True)
            raise RuntimeError(self._broken_error) from exc
        try:
            return self._recv_submit_result()
        except BaseException:
            if lease_id is not None:
                cancel_local_shm_input_lease(lease_id, name="udf-input")
                self._untrack_input_lease(lease_id)
            raise

    def _submit_ref_bundle(self, block_refs, slices, metadata, names) -> Any | None:
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
            except BaseException:
                assert lease_id is not None
                cancel_local_shm_input_lease(lease_id, name="udf-input")
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

    def submit_ref_bundle_with_id(self, submit_id: int, block_refs, slices, metadata, names) -> None:
        self._pending_batches += 1
        try:
            result = self._submit_ref_bundle(block_refs, slices, metadata, names)
        finally:
            self._pending_batches = max(0, self._pending_batches - 1)
        self._queue.append((SUBMIT_RESULT_MARKER, int(submit_id), result))
        self._notify_wakeup()

    def submit_ref_bundle(self, block_refs, slices, metadata, names) -> None:
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
            self._mark_broken(error)
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

    def cancel_output_grants(self) -> None:
        self._output_grant_cancel_event.set()
        self._release_active_output_grants(name="udf-output-cancel")
        wake_local_shm_ref_budget_waiters()

    def close(self, kill: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        self.cancel_output_grants()
        self._cancel_active_input_leases()

        proc = self._proc
        sock = self._sock
        self._proc = None
        self._sock = None
        shutdown_error: BaseException | None = None

        if sock is not None:
            try:
                if proc is not None and proc.poll() is None and not kill:
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
            except BaseException as exc:
                shutdown_error = exc
            try:
                proc.wait(timeout=_subprocess_control_timeout_s())
            except BaseException as exc:
                if shutdown_error is None:
                    shutdown_error = exc

        self._close_payload_shm()
        data_shm = self._data_shm
        self._data_shm = None
        if data_shm is not None:
            try:
                data_shm.close()
            finally:
                try:
                    _unlink_shm(data_shm, track=False)
                except FileNotFoundError:
                    pass

        finalizer = getattr(self, "_finalizer", None)
        if finalizer is not None and finalizer.alive:
            finalizer.detach()
        if shutdown_error is not None:
            raise RuntimeError(f"UDF subprocess did not terminate cleanly: {shutdown_error}") from shutdown_error

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


def _payload_task_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(duckdb_pickle.dumps(payload)).hexdigest()


class _PooledTaskWorker:
    def __init__(self, worker: _SingleSubprocessExecutor) -> None:
        self.worker = worker
        self.last_used = time.monotonic()


class _TaskWorkerPool:
    def __init__(
        self,
        runtime: _GlobalSubprocessTaskRuntime,
        key: str,
        payload: dict[str, Any],
        pool_size: int,
    ) -> None:
        self.runtime = runtime
        self.key = key
        self.payload = dict(payload)
        self.pool_size = max(1, int(pool_size))
        self.ref_count = 0
        self.closing = False
        self.idle: list[_PooledTaskWorker] = []
        self._active_wrappers: set[_PooledTaskWorker] = set()
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
        with self.runtime.cond:
            self.ref_count = max(0, self.ref_count - 1)
            if self.ref_count == 0:
                self.closing = True
                self.kill_on_release = kill
                self.admission_slots.close()
                while self.idle:
                    wrapper = self.idle.pop()
                    self.total = max(0, self.total - 1)
                    self.runtime.total_workers = max(0, self.runtime.total_workers - 1)
                    to_close.append(wrapper.worker)
                if kill:
                    active_to_kill.extend(wrapper.worker for wrapper in self._active_wrappers)
                self.runtime.pools.pop(self.key, None)
            self.runtime.cond.notify_all()
        for worker in to_close:
            worker.close(kill=kill)
        for worker in active_to_kill:
            worker.close(kill=True)

    def cancel_output_grants(self) -> None:
        workers: list[_SingleSubprocessExecutor] = []
        with self.runtime.cond:
            workers.extend(wrapper.worker for wrapper in self.idle)
            workers.extend(wrapper.worker for wrapper in self._active_wrappers)
        for worker in workers:
            worker.cancel_output_grants()

    def _spawn_worker(self, worker_idx: int) -> _PooledTaskWorker:
        worker = _SingleSubprocessExecutor(
            self.payload,
            worker_env=_worker_env_for_pool_index(self.payload, worker_idx, self.pool_size),
        )
        return _PooledTaskWorker(worker)

    def acquire_worker(self) -> _PooledTaskWorker:
        wrapper: _PooledTaskWorker | None = None
        spawn_idx: int | None = None
        while wrapper is None and spawn_idx is None:
            evicted: _SingleSubprocessExecutor | None = None
            with self.runtime.cond:
                if self.closing:
                    raise RuntimeError("subprocess task worker pool is closed")
                if self.idle:
                    wrapper = self.idle.pop()
                    self.active += 1
                    self._active_wrappers.add(wrapper)
                    return wrapper
                if self.total < self.pool_size and self.runtime.total_workers < self.runtime.max_workers:
                    spawn_idx = self.next_worker_idx
                    self.next_worker_idx += 1
                    self.total += 1
                    self.active += 1
                    self.runtime.total_workers += 1
                    break
                evicted = self.runtime._take_idle_worker_locked()
                if evicted is None:
                    self.runtime.cond.wait()
                    continue
            if evicted is not None:
                evicted.close(kill=False)

        try:
            assert spawn_idx is not None
            wrapper = self._spawn_worker(spawn_idx)
        except Exception:
            with self.runtime.cond:
                self.total = max(0, self.total - 1)
                self.active = max(0, self.active - 1)
                self.runtime.total_workers = max(0, self.runtime.total_workers - 1)
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
                self._active_wrappers.add(wrapper)
                return wrapper
        wrapper.worker.close(kill=close_kill)
        raise RuntimeError("subprocess task worker pool is closed")

    def release_worker(self, wrapper: _PooledTaskWorker, *, reusable: bool = True) -> None:
        to_close: _SingleSubprocessExecutor | None = None
        kill_close = False
        with self.runtime.cond:
            self._active_wrappers.discard(wrapper)
            self.active = max(0, self.active - 1)
            if (
                self.closing
                or not reusable
                or getattr(wrapper.worker, "_closed", False)
                or getattr(wrapper.worker, "_broken_error", None) is not None
            ):
                self.total = max(0, self.total - 1)
                self.runtime.total_workers = max(0, self.runtime.total_workers - 1)
                to_close = wrapper.worker
                kill_close = self.kill_on_release or not reusable
            else:
                wrapper.last_used = time.monotonic()
                self.idle.append(wrapper)
            self.runtime.cond.notify_all()
        if to_close is not None:
            to_close.close(kill=kill_close)


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

    def acquire_pool(self, payload: dict[str, Any], pool_size: int) -> _TaskWorkerPool:
        key = _payload_task_key(payload)
        with self.cond:
            if self.closed:
                raise RuntimeError("global subprocess task runtime is closed")
            pool = self.pools.get(key)
            if pool is None:
                pool = _TaskWorkerPool(self, key, payload, pool_size)
                self.pools[key] = pool
            pool.acquire_ref()
            return pool

    def submit(
        self,
        pool: _TaskWorkerPool,
        fn: Callable[[_SingleSubprocessExecutor], Any | None],
        debug_seq: int = 0,
    ) -> Future:
        if self.closed:
            raise RuntimeError("global subprocess task runtime is closed")
        return self.executor.submit(self._run_task, pool, fn, debug_seq)

    def _run_task(
        self,
        pool: _TaskWorkerPool,
        fn: Callable[[_SingleSubprocessExecutor], Any | None],
        debug_seq: int = 0,
    ) -> Any | None:
        acquire_start = time.perf_counter()
        wrapper: _PooledTaskWorker | None = None
        wrapper = pool.acquire_worker()
        acquire_s = time.perf_counter() - acquire_start
        assert wrapper is not None
        if _should_debug_submit(debug_seq):
            proc = getattr(wrapper.worker, "_proc", None)
            _subprocess_debug_log(
                "task_worker_acquired "
                f"seq={debug_seq} acquire_s={acquire_s:.6f} "
                f"worker_pid={getattr(proc, 'pid', None)} pool_size={pool.pool_size} "
                f"pool_total={pool.total} pool_active={pool.active} pool_idle={len(pool.idle)} "
                f"runtime_total_workers={self.total_workers} runtime_max_workers={self.max_workers}"
            )
        reusable = True
        run_start = time.perf_counter()
        try:
            return fn(wrapper.worker)
        except BaseException:
            reusable = getattr(wrapper.worker, "_broken_error", None) is None and not getattr(
                wrapper.worker, "_closed", False
            )
            raise
        finally:
            if _should_debug_submit(debug_seq):
                _subprocess_debug_log(
                    "task_worker_finished "
                    f"seq={debug_seq} run_s={time.perf_counter() - run_start:.6f} reusable={reusable}"
                )
            pool.release_worker(wrapper, reusable=reusable)

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
                return
            self.closed = True
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
        for pool in pools_to_cancel:
            pool.cancel_output_grants()
        self.executor.shutdown(wait=False, cancel_futures=True)
        for worker in to_close:
            worker.close(kill=kill)
        for worker in active_to_kill:
            worker.close(kill=True)


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

    def __init__(self, payload: dict[str, Any], pool_size: int, *, name: str | None = None) -> None:
        self.payload = dict(payload)
        self.pool_size = max(1, int(pool_size))
        self.name = str(name or "")
        self._closed = False
        self._lock = threading.Lock()
        self._active = 0
        pool_identity = self.name or str(id(self))
        self.admission_slots = LocalExecutionSlotPool(
            max_slots=self.pool_size,
            execution_slot_prefix=f"subprocess_actor:{pool_identity}",
        )
        self._idle_workers: queue.Queue[int] = queue.Queue()
        self._workers: list[_SingleSubprocessExecutor] = []
        self._executor: ThreadPoolExecutor | None = None
        try:
            for worker_idx in range(self.pool_size):
                self._workers.append(
                    _SingleSubprocessExecutor(
                        self.payload,
                        worker_env=_worker_env_for_pool_index(self.payload, worker_idx, self.pool_size),
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
            self._idle_workers.put(worker_idx)
        _subprocess_debug_log(
            f"local_actor_pool_created name={self.name!r} pool_size={self.pool_size} worker_pids={self.worker_pids()}"
        )

    def worker_pids(self) -> list[int | None]:
        return [getattr(worker._proc, "pid", None) for worker in self._workers]

    def create_admission_authority(self) -> LocalSlotAdmissionAuthority:
        return self.admission_slots.create_authority()

    def first_proc(self):
        if not self._workers:
            return None
        return self._workers[0]._proc

    def submit(
        self,
        fn: Callable[[_SingleSubprocessExecutor], Any | None],
        debug_seq: int = 0,
    ) -> Future:
        with self._lock:
            if self._closed:
                raise RuntimeError("local subprocess actor pool is closed")
            executor = self._executor
        if executor is None:
            raise RuntimeError("local subprocess actor pool is closed")
        return executor.submit(self._run, fn, debug_seq)

    def _run(
        self,
        fn: Callable[[_SingleSubprocessExecutor], Any | None],
        debug_seq: int = 0,
    ) -> Any | None:
        worker_idx = self._idle_workers.get()
        with self._lock:
            self._active += 1
            active = self._active
        worker = self._workers[worker_idx]
        worker_pid = getattr(worker._proc, "pid", None)
        try:
            if _should_debug_submit(debug_seq):
                _subprocess_debug_log(
                    "local_actor_pool_worker_acquired "
                    f"name={self.name!r} seq={debug_seq} worker_idx={worker_idx} active={active} "
                    f"pool_size={self.pool_size} worker_pid={getattr(self._workers[worker_idx]._proc, 'pid', None)}"
                )
            return fn(worker)
        except BaseException as exc:
            if self.payload.get("stateful") and worker._actor_lost:
                udf_name = str(self.payload.get("udf_name") or "udf")
                actor_id = "unknown" if worker_pid is None else str(worker_pid)
                raise RuntimeError(
                    f"stateful UDF {udf_name!r} lost local actor pid {actor_id}; "
                    "state was not recoverable; side effects may already have occurred"
                ) from exc
            raise
        finally:
            with self._lock:
                self._active = max(0, self._active - 1)
                active_after = self._active
            self._idle_workers.put(worker_idx)
            if _should_debug_submit(debug_seq):
                _subprocess_debug_log(
                    "local_actor_pool_worker_finished "
                    f"name={self.name!r} seq={debug_seq} worker_idx={worker_idx} active={active_after}"
                )

    def stats(self) -> dict[str, int]:
        with self._lock:
            active = self._active
        return {
            "pool_size": self.pool_size,
            "active_workers": active,
            "idle_workers": max(0, self.pool_size - active),
        }

    def cancel_output_grants(self) -> None:
        for worker in list(self._workers):
            worker.cancel_output_grants()

    def shutdown(self, *, kill: bool = False) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.admission_slots.close()
        self.cancel_output_grants()
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        for worker in list(self._workers):
            worker.close(kill=kill)
        self._workers = []
        _subprocess_debug_log(f"local_actor_pool_shutdown name={self.name!r} kill={kill}")

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


def _validate_stateful_local_actor_contract(payload: dict[str, Any], pool_size: int) -> None:
    if not payload.get("stateful"):
        return
    actor_number = payload.get("actor_number")
    if type(actor_number) is not int or actor_number != 1 or pool_size != 1:
        raise ValueError(
            "actor_number must be exactly 1 for stateful vane.cls UDFs; multi-actor state semantics are not defined"
        )


_LOCAL_ACTOR_POOL_CONTRACT_ERROR = (
    "local_actor_pool must expose submit(), create_admission_authority(), pool_size, stats(), "
    "cancel_output_grants(), first_proc(), and worker_pids()"
)
_LOCAL_ACTOR_POOL_REQUIRED_METHODS = (
    "submit",
    "create_admission_authority",
    "stats",
    "cancel_output_grants",
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
            _validate_stateful_local_actor_contract(raw_payload, pool_size)
            if float(raw_payload.get("gpus") or 0.0) > 0.0:
                raise ValueError("GPU resources require a Ray UDF backend")
            pool_name = f"local-subprocess-actor-{identity}-{node_id}"
            pool = LocalSubprocessActorPool(raw_payload, pool_size, name=pool_name)
            created.append(pool)
            actor_options_map[node_id] = {
                "local_actor_pool": pool,
            }

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
        self._subprocess_mode = _payload_subprocess_mode(payload)
        self._pool_size = _payload_subprocess_pool_size(payload, self._subprocess_mode)
        if payload.get("stateful"):
            if self._subprocess_mode != "actor":
                raise ValueError("stateful expression UDFs require an actor execution backend")
            _validate_stateful_local_actor_contract(payload, self._pool_size)
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
        self._task_futures: set[Future] = set()
        self._task_futures_cv = threading.Condition()
        self._task_futures_lock = self._task_futures_cv
        self._task_future_meta: dict[
            Future,
            tuple[int | None, int, float, AdmissionLease | None],
        ] = {}
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
                self._task_pool = self._task_runtime.acquire_pool(payload, self._pool_size)
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
            _validate_stateful_local_actor_contract(payload, actor_pool_size)
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
    def _proc(self):
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
        future: Future,
        submit_id: int | None,
        debug_seq: int,
        submit_start: float,
        admission: AdmissionLease | None,
    ) -> None:
        with self._task_futures_lock:
            self._task_futures.add(future)
            self._task_future_meta[future] = (
                submit_id,
                debug_seq,
                submit_start,
                admission,
            )
        future.add_done_callback(lambda done, _submit_id=submit_id: self._complete_task_submit(_submit_id, done))

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
        for lease_id in lease_ids:
            cancel_local_shm_input_lease(lease_id, name="udf-input-close")

    def _complete_task_submit(self, submit_id: int | None, future: Future) -> None:
        item: Any | None = None
        debug_meta: tuple[int | None, int, float, AdmissionLease | None] | None = None
        admission: AdmissionLease | None = None
        try:
            result = future.result()
            self._record_output_budget_result(result)
            item = self._submit_result_item(submit_id, result)
        except BaseException as exc:
            item = (SUBMIT_RESULT_MARKER, int(submit_id), exc) if submit_id is not None else exc
        finally:
            with self._task_futures_lock:
                debug_meta = self._task_future_meta.get(future)
            if debug_meta is not None:
                debug_submit_id, debug_seq, submit_start, admission = debug_meta
                if _should_debug_submit(debug_seq):
                    _subprocess_debug_log(
                        "task_submit_completed "
                        f"seq={debug_seq} submit_id={debug_submit_id} "
                        f"total_s={time.perf_counter() - submit_start:.6f}"
                    )
            if item is not None:
                with self._queue_lock:
                    if self._closed:
                        if admission is not None:
                            admission.release()
                    else:
                        self._queue.append(item)
                        self._result_admissions.append(admission)
            elif admission is not None:
                admission.release()
            with self._pending_lock:
                self._pending_batches = max(0, self._pending_batches - 1)
            self._notify_wakeup()
            with self._task_futures_cv:
                self._task_futures.discard(future)
                self._task_future_meta.pop(future, None)
                self._task_futures_cv.notify_all()

    def _submit_async(
        self,
        submit_id: int | None,
        fn: Callable[[_SingleSubprocessExecutor], Any | None],
        admission: AdmissionLease | None = None,
    ) -> None:
        if self._closed:
            if admission is not None:
                admission.release()
            raise RuntimeError("UDF subprocess executor is closed")
        self._debug_submit_count += 1
        debug_seq = self._debug_submit_count
        submit_start = time.perf_counter()

        if self._actor_pool is not None:
            actor_pool = self._actor_pool
            with self._pending_lock:
                self._pending_batches += 1
                pending = self._pending_batches
            if _should_debug_submit(debug_seq):
                pool_stats = actor_pool.stats()
                _subprocess_debug_log(
                    "local_actor_pool_submit_scheduled "
                    f"name={getattr(actor_pool, 'name', '')!r} seq={debug_seq} submit_id={submit_id} "
                    f"pending={pending} pool_size={actor_pool.pool_size} "
                    f"pool_active={pool_stats.get('active_workers', 0)} "
                    f"pool_idle={pool_stats.get('idle_workers', 0)}"
                )
            try:
                future = actor_pool.submit(fn, debug_seq)
                self._track_task_future(
                    future,
                    submit_id,
                    debug_seq,
                    submit_start,
                    admission,
                )
            except Exception:
                with self._pending_lock:
                    self._pending_batches = max(0, self._pending_batches - 1)
                if admission is not None:
                    admission.release()
                raise
            return

        if self._task_pool is not None:
            runtime = self._task_runtime
            task_pool = self._task_pool
            if runtime is None:
                raise RuntimeError("global subprocess task runtime is not available")
            with self._pending_lock:
                self._pending_batches += 1
                pending = self._pending_batches
            if _should_debug_submit(debug_seq):
                with runtime.cond:
                    _subprocess_debug_log(
                        "task_submit_scheduled "
                        f"seq={debug_seq} submit_id={submit_id} pending={pending} "
                        f"pool_size={task_pool.pool_size} "
                        f"pool_total={task_pool.total} pool_active={task_pool.active} "
                        f"pool_idle={len(task_pool.idle)} runtime_total_workers={runtime.total_workers} "
                        f"runtime_max_workers={runtime.max_workers}"
                    )
            try:
                future = runtime.submit(task_pool, fn, debug_seq)
                self._track_task_future(
                    future,
                    submit_id,
                    debug_seq,
                    submit_start,
                    admission,
                )
            except Exception:
                with self._pending_lock:
                    self._pending_batches = max(0, self._pending_batches - 1)
                if admission is not None:
                    admission.release()
                raise
            return

        if admission is not None:
            admission.release()
        raise RuntimeError("subprocess executor is not initialized with an actor or task worker owner")

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

    def submit_ref_bundle_with_id(self, submit_id: int, block_refs, slices, metadata, names) -> None:
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

            def submit_worker(worker, _payload=worker_payload, _lease_id=lease_id):
                try:
                    return worker._submit_ref_bundle_direct(_payload)
                except BaseException:
                    cancel_local_shm_input_lease(_lease_id, name=f"udf-input-{int(submit_id)}")
                    raise
                finally:
                    self._untrack_input_lease(_lease_id)

            try:
                admission = self._take_task_admission()
                self._submit_async(
                    int(submit_id),
                    submit_worker,
                    admission,
                )
            except BaseException:
                cancel_local_shm_input_lease(lease_id, name=f"udf-input-{int(submit_id)}")
                self._untrack_input_lease(lease_id)
                raise
            return
        raise RuntimeError("subprocess UDF ref-bundle input requires local shared-memory descriptors")

    def submit_ref_bundle(self, _block_refs, _slices, _metadata, _names) -> None:
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
        with self._task_futures_cv:
            for future in list(self._task_futures):
                future.cancel()
            self._task_futures_cv.notify_all()

    def _cancel_worker_output_grants(self) -> None:
        actor_pool = self._actor_pool
        if actor_pool is not None:
            actor_pool.cancel_output_grants()
        task_pool = self._task_pool
        if task_pool is not None:
            task_pool.cancel_output_grants()
        for worker in list(self._workers):
            worker.cancel_output_grants()

    def _cancel_local_shm_waits(self) -> None:
        self._cancel_active_input_leases()
        self._cancel_worker_output_grants()
        wake_local_shm_ref_budget_waiters()

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
        if self._closed:
            return
        self._closed = True
        authority = getattr(self, "_admission_authority", None)
        if authority is not None:
            authority.close()
        queue_lock = getattr(self, "_queue_lock", None)
        result_admissions = getattr(self, "_result_admissions", None)
        if queue_lock is not None and result_admissions is not None:
            with queue_lock:
                admissions = list(result_admissions)
                result_admissions.clear()
        else:
            admissions = []
        for admission in admissions:
            if admission is not None:
                admission.release()
        budget_wakeup_unregister = self._budget_wakeup_unregister
        self._budget_wakeup_unregister = None
        if budget_wakeup_unregister is not None:
            budget_wakeup_unregister()
        self._cancel_local_shm_waits()
        close_kill = bool(kill)
        if close_kill:
            self._cancel_pending_futures()
        else:
            if not self._wait_for_pending_futures(_subprocess_shutdown_grace_s()):
                close_kill = True
                self._cancel_pending_futures()
                self._cancel_local_shm_waits()
        actor_pool = self._actor_pool
        if actor_pool is not None:
            self._actor_pool = None
            return
        task_pool = self._task_pool
        if task_pool is not None:
            self._task_pool = None
            task_pool.release_ref(kill=close_kill)
            return
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=not close_kill, cancel_futures=True)
        for worker in list(self._workers):
            worker.close(kill=close_kill)

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

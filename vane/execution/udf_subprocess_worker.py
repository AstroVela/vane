# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Worker process for the subprocess UDF executor."""

from __future__ import annotations

import os
import socket
import struct
import sys
from traceback import TracebackException
from typing import TYPE_CHECKING

import pyarrow as pa

if TYPE_CHECKING:
    from multiprocessing import shared_memory

from vane import pickle as duckdb_pickle
from vane.execution._common import callable_cache_enabled as _callable_cache_enabled
from vane.execution._udf_runtime import UDFExecutor as RuntimeUDFExecutor
from vane.execution.ref_bundle import (
    _open_existing_shm,
    make_local_shm_ref_bundle_descriptor,
    materialize_ref_bundle,
    payload_requests_local_ref_bundle_output,
    release_local_shm_ref_bundle_descriptor,
)
from vane.execution.udf_row_preserving import (
    fuse_row_preserving_output,
    split_row_preserving_input,
)
from vane.execution.udf_threading import configure_loaded_torch_threads

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


def _debug_enabled() -> bool:
    for name in ("VANE_UDF_WORKER_SLOT_DEBUG", "DUCKDB_DISTRIBUTED_DEBUG"):
        value = os.environ.get(name, "")
        if value.strip().lower() not in ("", "0", "false", "no", "off"):
            return True
    return False


def _process_thread_count() -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("Threads:"):
                    return int(line.split(":", 1)[1].strip())
    except Exception:
        pass
    return -1


def _env_value(name: str) -> str:
    value = os.environ.get(name)
    return "<unset>" if value is None else value


def _torch_thread_fields() -> str:
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        return "torch_loaded=false"
    fields = ["torch_loaded=true"]
    try:
        fields.append(f"torch_num_threads={int(torch_module.get_num_threads())}")
    except Exception:
        fields.append("torch_num_threads=<error>")
    try:
        fields.append(f"torch_interop_threads={int(torch_module.get_num_interop_threads())}")
    except Exception:
        fields.append("torch_interop_threads=<error>")
    return " ".join(fields)


def _thread_log_submit_every() -> int:
    value = os.environ.get("VANE_UDF_WORKER_THREAD_LOG_EVERY_N", "").strip()
    if not value:
        return 0
    parsed = int(value)
    if parsed < 0:
        raise ValueError("VANE_UDF_WORKER_THREAD_LOG_EVERY_N must be non-negative")
    return parsed


def _should_log_submit(submit_count: int) -> bool:
    if not _debug_enabled():
        return False
    if submit_count == 1:
        return True
    every = _thread_log_submit_every()
    return every > 0 and submit_count % every == 0


def _worker_thread_log(event: str, payload: dict | None = None, **fields: object) -> None:
    if not _debug_enabled():
        return
    backend = str((payload or {}).get("execution_backend") or "-")
    parts = [
        f"event={event}",
        f"worker_index={_env_value('VANE_SUBPROCESS_WORKER_INDEX')}",
        f"pool_size={_env_value('VANE_SUBPROCESS_POOL_SIZE')}",
        f"backend={backend}",
        f"process_threads={_process_thread_count()}",
        f"OMP_NUM_THREADS={_env_value('OMP_NUM_THREADS')}",
        f"MKL_NUM_THREADS={_env_value('MKL_NUM_THREADS')}",
        f"OPENBLAS_NUM_THREADS={_env_value('OPENBLAS_NUM_THREADS')}",
        f"NUMEXPR_NUM_THREADS={_env_value('NUMEXPR_NUM_THREADS')}",
        _torch_thread_fields(),
    ]
    parts.extend(f"{key}={value}" for key, value in fields.items())
    print(
        f"[vane-udf-worker-threads pid={os.getpid()}] " + " ".join(parts),
        file=sys.stderr,
        flush=True,
    )


def _read_exact(sock: socket.socket, size: int) -> bytes:
    parts: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("parent closed the control socket")
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


def _read_ipc_from_shm(shm: shared_memory.SharedMemory, size: int | None = None) -> bytes:
    ipc_size = _IPC_HEADER.unpack_from(shm.buf, 0)[0]
    required = _IPC_HEADER.size + ipc_size
    if required > len(shm.buf):
        raise BufferError(
            f"shared memory IPC payload exceeds local mapping: required={required} capacity={len(shm.buf)}"
        )
    if size is not None and required > size:
        raise BufferError(f"shared memory IPC payload exceeds message size: required={required} size={size}")
    return bytes(shm.buf[_IPC_HEADER.size : required])


def _write_ipc_to_shm(shm: shared_memory.SharedMemory, ipc_bytes: bytes) -> int:
    required = _IPC_HEADER.size + len(ipc_bytes)
    if required > len(shm.buf):
        raise BufferError("shared memory segment is too small")
    _IPC_HEADER.pack_into(shm.buf, 0, len(ipc_bytes))
    shm.buf[_IPC_HEADER.size : required] = ipc_bytes
    return required


def _resize_shm(shm: shared_memory.SharedMemory, required: int) -> shared_memory.SharedMemory:
    if required <= len(shm.buf):
        return shm
    new_size = max(required, len(shm.buf) * 2, _DEFAULT_SHM_SIZE)
    name = shm.name
    path = f"/dev/shm/{name}"
    fd = os.open(path, os.O_RDWR)
    try:
        os.ftruncate(fd, new_size)
    finally:
        os.close(fd)
    shm.close()
    return _open_existing_shm(name, track=False)


def _arrow_table_from_ipc_bytes(data: bytes) -> pa.Table:
    reader = pa.ipc.open_stream(data)
    return reader.read_all()


def _arrow_table_to_ipc_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _format_exception(exc: BaseException) -> str:
    try:
        return "".join(TracebackException.from_exception(exc).format())
    except Exception:
        return repr(exc)


def _send_input_consumed(sock: socket.socket, ref_bundle: dict, input_table: pa.Table) -> None:
    lease_id = ref_bundle.get("input_lease_id")
    if lease_id is None:
        return
    payload = {
        "input_lease_id": int(lease_id),
        "worker_pid": os.getpid(),
        "rows": int(input_table.num_rows),
        "bytes": int(getattr(input_table, "nbytes", 0) or 0),
    }
    _send_message(sock, _MSG_INPUT_CONSUMED, duckdb_pickle.dumps(payload))


def _send_input_consume_failed(sock: socket.socket, ref_bundle: dict, exc: BaseException) -> None:
    lease_id = ref_bundle.get("input_lease_id")
    if lease_id is None:
        return
    payload = {
        "input_lease_id": int(lease_id),
        "worker_pid": os.getpid(),
        "error": _format_exception(exc),
    }
    _send_message(sock, _MSG_INPUT_CONSUME_FAILED, duckdb_pickle.dumps(payload))


def _request_output_grant(
    sock: socket.socket,
    *,
    submit_count: int,
    size: int,
    input_lease_id: int | None = None,
) -> int:
    payload = {
        "request_id": int(submit_count),
        "worker_pid": os.getpid(),
        "size_bytes": int(size),
        "priority": "consumer",
    }
    if input_lease_id is not None:
        payload["input_lease_id"] = int(input_lease_id)
    _send_message(sock, _MSG_OUTPUT_GRANT_REQUEST, duckdb_pickle.dumps(payload))
    msg_type, payload_data = _recv_message(sock)
    if msg_type == _MSG_OUTPUT_GRANT_CANCELLED:
        error = payload_data.decode("utf-8", errors="replace") or "local_shm output grant cancelled"
        raise RuntimeError(error)
    if msg_type != _MSG_OUTPUT_GRANT_GRANTED:
        raise RuntimeError(f"unexpected output grant response: {msg_type:#x}")
    response = duckdb_pickle.loads(payload_data)
    return int(response["grant_id"])


def _release_output_grant(sock: socket.socket, grant_id: int) -> None:
    if int(grant_id) <= 0:
        return
    _send_message(sock, _MSG_OUTPUT_GRANT_RELEASE, duckdb_pickle.dumps({"grant_id": int(grant_id)}))


def _payload_subprocess_mode(payload: dict) -> str:
    backend = str(payload.get("execution_backend") or "").strip().lower()
    if backend == "subprocess_task":
        return "task"
    if backend == "subprocess_actor":
        return "actor"
    raise ValueError("payload.execution_backend must be one of: subprocess_task, subprocess_actor")


def _drain_executor_outputs(executor: RuntimeUDFExecutor) -> list[pa.Table]:
    result_tables = executor.drain_outputs()
    if not result_tables:
        raise RuntimeError("UDF returned no result")
    return list(result_tables)


def _concat_executor_outputs(result_tables: list[pa.Table]) -> pa.Table:
    if len(result_tables) == 1:
        return result_tables[0]
    return pa.concat_tables(result_tables, promote_options="default")


def _ipc_response_size(table: pa.Table) -> int:
    return _IPC_HEADER.size + len(_arrow_table_to_ipc_bytes(table))


def _make_local_shm_ref_bundle_descriptor_for_tables(
    tables: list[pa.Table],
    *,
    grant_id: int | None = None,
) -> dict:
    if not tables:
        raise RuntimeError("UDF returned no result")

    descriptor = {
        "block_refs": [],
        "metadata": [],
        "names": list(tables[0].schema.names),
    }
    try:
        for table in tables:
            block_descriptor = make_local_shm_ref_bundle_descriptor(table)
            try:
                block_names = list(block_descriptor.get("names") or [])
                if block_names != descriptor["names"]:
                    raise ValueError(
                        "UDF output block schemas have different column names: "
                        f"expected={descriptor['names']!r} got={block_names!r}"
                    )
                descriptor["block_refs"].extend(block_descriptor.get("block_refs") or [])
                descriptor["metadata"].extend(block_descriptor.get("metadata") or [])
            except Exception:
                release_local_shm_ref_bundle_descriptor(block_descriptor)
                raise
        if grant_id is not None:
            descriptor["grant_id"] = int(grant_id)
        return descriptor
    except Exception:
        release_local_shm_ref_bundle_descriptor(descriptor)
        raise


def _execute_submit(
    executor: RuntimeUDFExecutor,
    input_table: pa.Table,
    data_shm: shared_memory.SharedMemory,
    produce_ref_bundle_output: bool,
    *,
    sock: socket.socket,
    submit_count: int,
    input_lease_id: int | None = None,
    finish_before_drain: bool = False,
) -> tuple[shared_memory.SharedMemory, int, bytes]:
    if input_table.num_rows == 0:
        return data_shm, _MSG_OK, struct.pack("<Q", 0)
    payload = getattr(executor, "_payload", {}) or {}
    row_preserving = str(payload.get("call_mode") or "") == "map_batches_rows"
    passthrough_table = None
    if row_preserving:
        input_table, passthrough_table = split_row_preserving_input(payload, input_table)
    executor.submit(input_table)
    if finish_before_drain:
        executor.finished_submitting()
    output_tables = _drain_executor_outputs(executor)
    if row_preserving:
        if len(output_tables) != 1:
            raise RuntimeError(
                "map_batches_rows subprocess produced %d outputs, expected exactly 1" % len(output_tables)
            )
        output_tables = [fuse_row_preserving_output(payload, passthrough_table, output_tables[0])]
    if produce_ref_bundle_output:
        required = sum(_ipc_response_size(output_table) for output_table in output_tables)
        grant_id = _request_output_grant(
            sock,
            submit_count=submit_count,
            size=required,
            input_lease_id=input_lease_id,
        )
        descriptor = None
        try:
            descriptor = _make_local_shm_ref_bundle_descriptor_for_tables(output_tables, grant_id=grant_id)
            result_payload = duckdb_pickle.dumps(descriptor)
        except Exception:
            if descriptor is not None:
                try:
                    release_local_shm_ref_bundle_descriptor(descriptor)
                except Exception:
                    pass
            try:
                _release_output_grant(sock, grant_id)
            except Exception:
                pass
            raise
        return data_shm, _MSG_REF_BUNDLE_RESULT, result_payload
    output_table = _concat_executor_outputs(output_tables)
    output_ipc = _arrow_table_to_ipc_bytes(output_table)
    required = _IPC_HEADER.size + len(output_ipc)
    data_shm = _resize_shm(data_shm, required)
    result_size = _write_ipc_to_shm(data_shm, output_ipc)
    return data_shm, _MSG_OK, struct.pack("<Q", result_size)


def _execute_task_submit(
    payload: dict,
    input_table: pa.Table,
    data_shm: shared_memory.SharedMemory,
    produce_ref_bundle_output: bool,
    submit_count: int,
    log_submit: bool,
    sock: socket.socket,
    input_lease_id: int | None = None,
) -> tuple[shared_memory.SharedMemory, int, bytes]:
    if input_table.num_rows == 0:
        return data_shm, _MSG_OK, struct.pack("<Q", 0)
    executor = RuntimeUDFExecutor(payload, cache_callable=_callable_cache_enabled(payload))
    configure_loaded_torch_threads()
    try:
        if log_submit:
            _worker_thread_log(
                "task_executor_ready",
                payload,
                submit_count=submit_count,
                rows=input_table.num_rows,
            )
        return _execute_submit(
            executor,
            input_table,
            data_shm,
            produce_ref_bundle_output,
            sock=sock,
            submit_count=submit_count,
            input_lease_id=input_lease_id,
            finish_before_drain=True,
        )
    finally:
        executor.finished_submitting()


def worker_main(sock_fd: int, payload_shm_name: str, payload_size: int, data_shm_name: str) -> None:
    sock = socket.socket(fileno=sock_fd)
    payload_shm = _open_existing_shm(payload_shm_name, track=False)
    data_shm = _open_existing_shm(data_shm_name, track=False)

    executor: RuntimeUDFExecutor | None = None
    try:
        payload_bytes = _read_ipc_from_shm(payload_shm, payload_size)
        payload = duckdb_pickle.loads(payload_bytes)
        configure_loaded_torch_threads()
        produce_ref_bundle_output = payload_requests_local_ref_bundle_output(payload)
        subprocess_mode = _payload_subprocess_mode(payload)
        _worker_thread_log("worker_started", payload, mode=subprocess_mode)
        if subprocess_mode == "actor":
            executor = RuntimeUDFExecutor(payload)
            configure_loaded_torch_threads()
            executor.warm_up()
            _worker_thread_log("actor_executor_ready", payload, mode=subprocess_mode)
        _send_message(sock, _MSG_READY)

        submit_count = 0
        close_requested = False
        while not close_requested:
            msg_type, payload_data = _recv_message(sock)
            if msg_type == _MSG_CLOSE:
                _send_message(sock, _MSG_ACK)
                close_requested = True
                continue
            if msg_type == _MSG_FINISHED:
                try:
                    if executor is not None:
                        executor.finished_submitting()
                    _send_message(sock, _MSG_ACK)
                except Exception as exc:
                    _send_message(sock, _MSG_ERROR, _format_exception(exc).encode("utf-8", errors="replace"))
                continue
            if msg_type not in (_MSG_SUBMIT, _MSG_SUBMIT_REF_BUNDLE):
                raise ValueError(f"unknown UDF subprocess message type: {msg_type:#x}")
            if msg_type == _MSG_SUBMIT and len(payload_data) != 8:
                raise ValueError("submit message payload must contain the shared-memory byte size")

            try:
                input_lease_id = None
                if msg_type == _MSG_SUBMIT:
                    input_size = struct.unpack("<Q", payload_data)[0]
                    if input_size > len(data_shm.buf):
                        name = data_shm.name
                        data_shm.close()
                        data_shm = _open_existing_shm(name, track=False)
                    input_ipc = _read_ipc_from_shm(data_shm, input_size)
                    input_table = _arrow_table_from_ipc_bytes(input_ipc)
                else:
                    ref_bundle = duckdb_pickle.loads(payload_data)
                    lease_id_raw = ref_bundle.get("input_lease_id")
                    input_lease_id = int(lease_id_raw) if lease_id_raw is not None else None
                    try:
                        input_table = materialize_ref_bundle(
                            ref_bundle["block_refs"],
                            ref_bundle.get("slices"),
                            ref_bundle.get("metadata"),
                            ref_bundle.get("names"),
                        )
                    except Exception as exc:
                        _send_input_consume_failed(sock, ref_bundle, exc)
                        raise
                    _send_input_consumed(sock, ref_bundle, input_table)
                submit_count += 1
                log_submit = _should_log_submit(submit_count)
                if log_submit:
                    _worker_thread_log(
                        "submit_received",
                        payload,
                        mode=subprocess_mode,
                        submit_count=submit_count,
                        rows=input_table.num_rows,
                        msg_type=msg_type,
                    )
                if subprocess_mode == "task":
                    data_shm, result_msg_type, result_payload = _execute_task_submit(
                        payload,
                        input_table,
                        data_shm,
                        produce_ref_bundle_output,
                        submit_count,
                        log_submit,
                        sock,
                        input_lease_id=input_lease_id,
                    )
                else:
                    if executor is None:
                        raise RuntimeError("subprocess actor worker executor is not initialized")
                    data_shm, result_msg_type, result_payload = _execute_submit(
                        executor,
                        input_table,
                        data_shm,
                        produce_ref_bundle_output,
                        sock=sock,
                        submit_count=submit_count,
                        input_lease_id=input_lease_id,
                    )
                if log_submit:
                    _worker_thread_log(
                        "submit_finished",
                        payload,
                        mode=subprocess_mode,
                        submit_count=submit_count,
                        rows=input_table.num_rows,
                        result_msg_type=result_msg_type,
                    )
                _send_message(sock, result_msg_type, result_payload)
            except Exception as exc:
                _send_message(sock, _MSG_ERROR, _format_exception(exc).encode("utf-8", errors="replace"))
    except Exception as exc:
        try:
            _send_message(sock, _MSG_ERROR, _format_exception(exc).encode("utf-8", errors="replace"))
        except Exception:
            pass
    finally:
        try:
            payload_shm.close()
        except Exception:
            pass
        try:
            data_shm.close()
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass


def _main(argv: list[str]) -> int:
    if len(argv) != 5:
        return 2
    worker_main(int(argv[1]), argv[2], int(argv[3]), argv[4])
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))

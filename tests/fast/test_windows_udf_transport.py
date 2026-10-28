# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Real Windows mapping ownership and descendant cleanup contracts."""

import gc
import subprocess
import sys
import textwrap
import time

import psutil
import pyarrow as pa
import pytest

from vane.execution import ref_bundle
from vane.execution.udf_process_group import WorkerProcessGroup

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows kernel ownership contracts")


@pytest.mark.parametrize("track", [False, True])
def test_shared_mapping_survives_creator_close_until_last_handle(monkeypatch, track):
    def no_posix_tracker(*args):
        pytest.fail("Windows mappings must not start the POSIX resource tracker")

    monkeypatch.setattr(ref_bundle._resource_tracker, "register", no_posix_tracker)
    monkeypatch.setattr(ref_bundle._resource_tracker, "unregister", no_posix_tracker)
    owner = ref_bundle._create_shm(4096, track=track)
    reader = None
    name = owner.name
    try:
        owner.buf[:4] = b"vane"
        reader = ref_bundle._open_existing_shm(name, track=track)
        ref_bundle._unlink_shm(owner, track=track)
        owner.close()
        assert bytes(reader.buf[:4]) == b"vane"
        reopened = ref_bundle._open_existing_shm(name, track=track)
        reopened.close()
    finally:
        if reader is not None:
            reader.close()
        owner.close()
    with pytest.raises(FileNotFoundError):
        ref_bundle._open_existing_shm(name, track=track)


@pytest.mark.parametrize("cleanup", ["release", "collect", "adopt"])
def test_descriptor_retains_mapping_until_adoption_or_release(cleanup):
    descriptor = ref_bundle.make_local_shm_ref_bundle_descriptor(pa.table({"value": [1, 2, 3]}))
    name = descriptor["block_refs"][0]["shm_name"]
    result = None
    try:
        opened = ref_bundle._open_existing_shm(name, track=False)
        opened.close()
        if cleanup == "adopt":
            result = ref_bundle.make_local_shm_ref_bundle_result_from_descriptor(descriptor, block_on_budget=False)
            assert ref_bundle.materialize_ref_bundle(result[1], metadata=result[2]).to_pydict() == {"value": [1, 2, 3]}
            for ref in result[1]:
                ref.release()
        elif cleanup == "release":
            ref_bundle.release_local_shm_ref_bundle_descriptor(descriptor)
        else:
            del descriptor
            gc.collect()
        with pytest.raises(FileNotFoundError):
            ref_bundle._open_existing_shm(name, track=False)
    finally:
        if result is not None:
            for ref in result[1]:
                ref.release()
        ref_bundle._unlink_shared_memory_name(name)


def test_temporary_descriptor_keeps_publisher_mapping_alive_until_receiver_opens(monkeypatch):
    names = []
    original_create = ref_bundle._create_shm
    before = ref_bundle.local_shm_ref_budget_snapshot()["allocated_bytes"]

    def record_mapping(size, *, track):
        shm = original_create(size, track=track)
        names.append(shm.name)
        return shm

    monkeypatch.setattr(ref_bundle, "_create_shm", record_mapping)
    result = None
    try:
        result = ref_bundle.make_local_shm_ref_bundle_result_from_descriptor(
            ref_bundle.make_local_shm_ref_bundle_descriptor(pa.table({"value": [1, 2, 3]})),
            block_on_budget=False,
        )
        assert ref_bundle.materialize_ref_bundle(result[1], metadata=result[2]).to_pydict() == {"value": [1, 2, 3]}
        reopened = ref_bundle._open_existing_shm(names[0], track=False)
        reopened.close()
    finally:
        if result is not None:
            for ref in result[1]:
                ref.release()
        for name in names:
            ref_bundle._unlink_shared_memory_name(name)
    with pytest.raises(FileNotFoundError):
        ref_bundle._open_existing_shm(names[0], track=False)
    assert ref_bundle.local_shm_ref_budget_snapshot()["allocated_bytes"] == before


def test_job_reaps_descendant_after_worker_has_exited():
    program = textwrap.dedent("""
        import subprocess
        import sys
        sys.stdin.buffer.readline()
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        print(child.pid, flush=True)
    """)
    process = subprocess.Popen(
        [sys.executable, "-c", program],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    group = None
    try:
        group = WorkerProcessGroup(process.pid)
        stdout, stderr = process.communicate(b"start\n", timeout=10)
        assert process.returncode == 0, stderr
        descendant = psutil.Process(int(stdout.strip()))
        assert descendant.is_running()
        group.close(5)
        assert descendant.wait(timeout=5) == 1
        group.close(5)
    finally:
        if group is not None:
            group.close(5)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
def test_cancelled_large_ipc_results_release_old_mappings_and_reuse_worker(monkeypatch, backend):
    from vane import pickle as vane_pickle
    from vane.execution.udf_lifecycle import ExecutionCancellationScope, ExecutionCancelledError
    from vane.execution.udf_subprocess import _MSG_IPC_RESULT, _SingleSubprocessExecutor

    large_value = "x" * (2 * 1024 * 1024)

    def expand(table):
        return pa.table({"out": [large_value] * table.num_rows})

    class Expand:
        def __call__(self, table):
            return expand(table)

    worker = _SingleSubprocessExecutor(
        {
            "function_pickle": vane_pickle.dumps(Expand if backend == "subprocess_actor" else expand),
            "call_mode": "map_batches",
            "execution_backend": backend,
            **({"actor_number": 1} if backend == "subprocess_actor" else {}),
        }
    )
    replacement_name = None
    try:
        process = worker._proc
        # Repeated cancellations must not retain either the initial mapping or
        # a larger mapping from a previous cancelled result.
        for rows in (2, 5):
            old_mapping = worker._data_shm
            old_name = old_mapping.name
            scope = ExecutionCancellationScope(f"cancel-ipc-{rows}", 1)
            recv_expected = worker._recv_expected

            def cancel_after_result(*args, **kwargs):
                nonlocal replacement_name
                msg_type, payload = recv_expected(*args, **kwargs)
                if msg_type == _MSG_IPC_RESULT:
                    replacement_name = payload[8:].decode("ascii")
                    scope.cancel("cancel after result was produced")
                return msg_type, payload

            def no_decode(*_args, **_kwargs):
                pytest.fail("cancelled IPC results must not be decoded")

            with monkeypatch.context() as patch:
                patch.setattr(worker, "_recv_expected", cancel_after_result)
                patch.setattr(worker, "_decode_ipc_result", no_decode)
                with pytest.raises(ExecutionCancelledError, match="cancel after result was produced"):
                    worker._run_in_execution_scope(scope, lambda current: current.submit(pa.table({"x": range(rows)})))
            assert replacement_name is not None and replacement_name != old_name
            assert worker._data_shm.name == replacement_name
            assert old_mapping.buf is None
            with pytest.raises(FileNotFoundError):
                ref_bundle._open_existing_shm(old_name, track=False)
            assert worker._broken_error is None
            assert not worker._awaiting_submit_terminal
            assert process.poll() is None

        worker.submit(pa.table({"x": [1]}))
        assert worker.take_ready_result().column("out").to_pylist() == [large_value]
        assert worker._proc is process
        assert worker._data_shm.name == replacement_name
    finally:
        worker.close(kill=True)
    assert worker._data_shm is None
    # Windows may finish releasing the terminated process's mappings just
    # after its job reports no active processes. Do not retain a probe handle.
    deadline = time.monotonic() + 5
    while True:
        try:
            remaining = ref_bundle._open_existing_shm(replacement_name, track=False)
        except FileNotFoundError:
            break
        remaining.close()
        assert time.monotonic() < deadline, "replacement mapping survived worker shutdown"
        time.sleep(0.01)

# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Inherited views pin pooled regions until every inheriting process exits."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")


_VIEWS = """
import gc, json, os, select, signal, sys, time
from pathlib import Path
import pyarrow as pa
from vane import pickle as vane_pickle
from vane.execution.udf_subprocess import _SingleSubprocessExecutor

kind, action = sys.argv[1:]
def identity(table):
    return table

worker = _SingleSubprocessExecutor({
    'function_pickle': vane_pickle.dumps(identity),
    'call_mode': 'map_batches', 'execution_backend': 'subprocess_task',
    'produce_ref_bundle_output': True, 'streaming_output_mode': 'local_shm_ref_bundle',
})
def submit(value):
    worker.submit(pa.table({'x': [value] * 2048}))
    result, finished = worker.take_ready_result()
    assert not finished
    assert worker.take_ready_result() == (None, True)
    return result[1][0]

first = submit(11)
slot = first._allocation_lease.allocation
store = first._allocation_lease.store
view = first.to_table().column(0).chunk(0)
if kind == 'numpy':
    view = view.to_numpy(zero_copy_only=True)
first.release()
read_gate, write_gate = os.pipe()
read_result, write_result = os.pipe()
read_exit, write_exit = os.pipe()
pid = os.fork()
if pid == 0:
    os.close(write_gate)
    os.close(read_result)
    os.close(write_exit)
    if action == 'descendant' and os.fork() != 0:
        os._exit(0)
    signal.alarm(15)
    assert os.read(read_gate, 1) == b'1'
    values = view.tolist() if kind == 'numpy' else view.to_pylist()
    os.write(write_result, json.dumps({'pid': os.getpid(), 'intact': values == [11] * 2048}).encode())
    assert os.read(read_exit, 1) == b'1'
    os._exit(0)

os.close(read_gate)
os.close(write_result)
os.close(read_exit)
reaped = False
second = None
try:
    if action == 'descendant':
        _, status = os.waitpid(pid, 0)
        reaped = True
        assert os.waitstatus_to_exitcode(status) == 0
    view = None
    gc.collect()
    second = submit(22)
    assert second.to_table().column(0).to_pylist() == [22] * 2048
    assert second._allocation_lease.allocation.offset != slot.offset
    second.release()
    if action == 'close':
        worker.close(kill=True)
        assert store.snapshot()['live_allocations'] == 1
        assert store.snapshot()['mapped_capacity_bytes'] > 0
    os.write(write_gate, b'1')
    assert select.select([read_result], [], [], 15)[0]
    record = json.loads(os.read(read_result, 4096))
    assert record['intact'], record
    assert store.snapshot()['live_allocations'] == 1
    os.write(write_exit, b'1')
    if not reaped:
        _, status = os.waitpid(pid, 0)
        reaped = True
        assert os.waitstatus_to_exitcode(status) == 0
    # A grandchild may still be completing _exit after its last pipe write.
    deadline = time.monotonic() + 5
    while store.snapshot()['live_allocations'] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert store.snapshot()['live_allocations'] == 0
    if action != 'close':
        reused = store.allocate(slot.size)
        assert reused.allocation.offset == slot.offset
        reused.release()
    worker.close(kill=True)
    assert store.snapshot()['mapped_capacity_bytes'] == 0
    assert not (Path('/dev/shm') / slot.shm_name).exists()
    print(json.dumps({'intact': True, 'reclaimed': True}), flush=True)
finally:
    # Closing the gates also releases a waiting descendant on a failed assert.
    os.close(write_gate)
    os.close(write_exit)
    os.close(read_result)
    if not reaped:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
    if second is not None:
        second.release()
    worker.close(kill=True)
"""


@pytest.mark.parametrize("kind", ["arrow", "numpy"])
@pytest.mark.parametrize("action", ["reuse", "close", "descendant"])
def test_inherited_view_survives_reuse_close_and_intermediate_process_exit(kind, action):
    result = subprocess.run(
        [sys.executable, "-I", "-c", _VIEWS, kind, action],
        capture_output=True,
        text=True,
        timeout=40,
        env={**os.environ, "VANE_LOCAL_SHM_STORE_BYTES": "128k"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    assert json.loads(result.stdout) == {"intact": True, "reclaimed": True}


_CAPACITY = """
import atexit, json, os, signal, sys, time
from pathlib import Path
fork_child = False
def finish_child_exit():
    if fork_child:
        os._exit(0)
atexit.register(finish_child_exit)
from vane.execution import udf_shm_store as storage

mode = sys.argv[1]
store = storage.LocalShmStore(4096)
store.add_client('parent')
lease = store.allocate(4096)
slot = lease.allocation
store.buffer(slot)[:] = b'p' * 4096
errors = []
sys.unraisablehook = lambda error: errors.append(str(error.exc_value))
original_thread = storage.threading.Thread
original_pipe = storage.os.pipe
gates = [os.pipe() for _ in range(2 if mode == 'siblings' else 1)]
if mode == 'watcher_failure':
    def reject_watcher(*args, **kwargs):
        raise RuntimeError('planned watcher startup failure')
    storage.threading.Thread = reject_watcher
if mode == 'pipe_failure':
    def reject_pipe():
        raise OSError('planned pipe creation failure')
    storage.os.pipe = reject_pipe

pids = []
for read_fd, write_fd in gates:
    pid = os.fork()
    if pid == 0:
        fork_child = True
        signal.alarm(15)
        os.close(write_fd)
        assert os.read(read_fd, 1) == b'1'
        if mode == 'normal':
            sys.exit(0)
        if mode == 'exec':
            os.execv(sys.executable, [sys.executable, '-I', '-c', 'import time; time.sleep(2)'])
        os._exit(0)
    os.close(read_fd)
    pids.append(pid)
storage.threading.Thread = original_thread
storage.os.pipe = original_pipe
lease.release()

def full():
    try:
        unexpected = store.allocate(4096)
    except storage.LocalShmStoreCapacityError:
        return
    unexpected.release()
    raise AssertionError('inherited allocation was reused')

try:
    full()
    if mode == 'background':
        store.remove_client('parent')
        assert (Path('/dev/shm') / slot.shm_name).exists()
    for index, ((_, write_fd), pid) in enumerate(zip(gates, pids)):
        if mode == 'kill':
            os.kill(pid, signal.SIGKILL)
        else:
            os.write(write_fd, b'1')
        if mode == 'exec':
            deadline = time.monotonic() + 5
            while store.snapshot()['live_allocations'] and time.monotonic() < deadline:
                time.sleep(0.01)
            assert store.snapshot()['live_allocations'] == 0
            assert os.waitpid(pid, os.WNOHANG) == (0, 0), 'pins waited for exit instead of exec'
        _, status = os.waitpid(pid, 0)
        pids[index] = 0
        assert os.waitstatus_to_exitcode(status) == (-signal.SIGKILL if mode == 'kill' else 0)
        if mode == 'siblings' and index == 0:
            full()
    if mode == 'background':
        # No store method or snapshot after child exit: the watcher must
        # reclaim and unlink an idle arena without a subsequent query.
        deadline = time.monotonic() + 5
        while (Path('/dev/shm') / slot.shm_name).exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not (Path('/dev/shm') / slot.shm_name).exists()
        assert not errors, errors
    elif mode == 'pipe_failure':
        # Without a lifetime channel there is no evidence that every inherited
        # mapping has gone; keep the region pinned until owner-process exit.
        full()
        assert errors == ['planned pipe creation failure'], errors
    else:
        reused = store.allocate(4096)
        assert reused.allocation.offset == slot.offset
        reused.release()
        assert store.snapshot()['live_allocations'] == 0
        if mode == 'watcher_failure':
            assert errors == ['planned watcher startup failure'], errors
        else:
            assert not errors, errors
finally:
    for (_, write_fd), pid in zip(gates, pids):
        os.close(write_fd)
        if pid:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
    store.remove_client('parent')
    store.close()
if mode != 'pipe_failure':
    assert not (Path('/dev/shm') / slot.shm_name).exists()
print(json.dumps({'name': slot.shm_name, 'safe': True}), flush=True)
"""


@pytest.mark.parametrize(
    "mode", ["exit", "normal", "kill", "exec", "siblings", "background", "watcher_failure", "pipe_failure"]
)
def test_fork_pins_release_on_last_exit_or_exec_and_retain_ownership_on_setup_failure(mode):
    result = subprocess.run(
        [sys.executable, "-I", "-c", _CAPACITY, mode],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    record = json.loads(result.stdout)
    assert record["safe"]
    assert not (Path("/dev/shm") / record["name"]).exists()

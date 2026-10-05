# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Forked drivers must preserve the parent's arenas and worker channels."""

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")


_NEW_WORKER = """
import json, os, select, signal, sys, threading
from pathlib import Path
import pyarrow as pa
from vane import pickle as vane_pickle
from vane.execution import udf_shm_store as storage
from vane.execution.udf_subprocess import _SingleSubprocessExecutor

def identity(table):
    return table

payload = {
    'function_pickle': vane_pickle.dumps(identity),
    'call_mode': 'map_batches',
    'execution_backend': 'subprocess_task',
    'produce_ref_bundle_output': True,
    'streaming_output_mode': 'local_shm_ref_bundle',
}

def submit(worker, value):
    worker.submit(pa.table({'x': [value]}))
    result, finished = worker.take_ready_result()
    assert finished is False
    assert worker.take_ready_result() == (None, True)
    return result[1][0]

def values(ref):
    return ref.to_table().column(0).to_pylist()

parent = _SingleSubprocessExecutor(payload)
first = submit(parent, 11)
parent_slot = first._allocation_lease.allocation
locked, unlock = threading.Event(), threading.Event()
held_lock = sys.argv[1]
locks = {'registry': storage._registry_lock, 'store': first._allocation_lease.store._lock}

def hold_lock():
    with locks[held_lock]:
        locked.set()
        assert unlock.wait(20)

thread = None
if held_lock != 'none':
    thread = threading.Thread(target=hold_lock)
    thread.start()
    assert locked.wait(5)

to_child_read, to_child_write = os.pipe()
to_parent_read, to_parent_write = os.pipe()
pid = os.fork()
if pid == 0:
    signal.alarm(15)
    os.close(to_child_write)
    os.close(to_parent_read)
    assert os.read(to_child_read, 1) == b'1'
    child = _SingleSubprocessExecutor(payload)
    result = None
    try:
        result = submit(child, 33)
        slot = result._allocation_lease.allocation
        record = {'child_store': slot.store_id, 'child_name': slot.shm_name,
                  'child_values': values(result)}
    finally:
        if result is not None:
            result.release()
        child.close(kill=True)
    snapshot = storage.local_shm_store_snapshot()
    assert snapshot['live_allocations'] == snapshot['mapped_capacity_bytes'] == 0
    os.write(to_parent_write, json.dumps(record).encode())
    # This probe tests allocation while both processes are alive. Clean up the
    # child-owned worker explicitly; inherited finalizers have separate tests.
    os._exit(0)

os.close(to_child_read)
os.close(to_parent_write)
unlock.set()
if thread is not None:
    thread.join(5)
refs = [first]
reaped = False
try:
    second = submit(parent, 22)
    refs.append(second)
    assert values(second) == [22]
    os.write(to_child_write, b'1')
    assert select.select([to_parent_read], [], [], 20)[0], 'fork child did not finish'
    data = os.read(to_parent_read, 4096)
    _, status = os.waitpid(pid, 0)
    reaped = True
    assert os.waitstatus_to_exitcode(status) == 0, data
    record = json.loads(data)
    assert record['child_values'] == [33]
    assert record['child_store'] != parent_slot.store_id, record
    assert record['child_name'] != parent_slot.shm_name, record
    assert not (Path('/dev/shm') / record['child_name']).exists()
    assert values(first) == [11]
    assert values(second) == [22], record
    third = submit(parent, 44)
    refs.append(third)
    assert values(third) == [44]
    assert third._allocation_lease.allocation.store_id == parent_slot.store_id
    print(json.dumps({'parent_values': [values(ref) for ref in refs],
                      'child_values': record['child_values'], 'isolated': True}), flush=True)
finally:
    if not reaped:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
    for ref in refs:
        ref.release()
    parent.close(kill=True)
    os.close(to_child_write)
    os.close(to_parent_read)
assert not (Path('/dev/shm') / parent_slot.shm_name).exists()
"""


@pytest.mark.parametrize("held_lock", ["none", "registry", "store"])
def test_forked_new_worker_uses_independent_arena_and_preserves_parent_outputs(held_lock):
    child = subprocess.run(
        [sys.executable, "-I", "-c", _NEW_WORKER, held_lock],
        text=True,
        capture_output=True,
        timeout=40,
        env={**os.environ, "VANE_LOCAL_SHM_STORE_BYTES": "1m"},
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(child.stdout) == {
        "parent_values": [[11], [22], [44]],
        "child_values": [33],
        "isolated": True,
    }


_INHERITED_HANDLES = """
import json, os, signal, sys, threading
from vane.execution import udf_shm_store as storage

store = storage.current_store('parent')
lease = store.allocate(64)
slot = lease.allocation
region = store.buffer(slot)
region[:] = b'p' * 64
region.release()
locks = {'registry': storage._registry_lock, 'store': store._lock, 'lease': lease._lock}
locked, unlock = threading.Event(), threading.Event()

def hold_lock():
    with locks[sys.argv[1]]:
        locked.set()
        assert unlock.wait(15)

thread = threading.Thread(target=hold_lock)
thread.start()
assert locked.wait(5)
pid = os.fork()
if pid == 0:
    signal.alarm(5)
    # Explicit inherited handles cannot bypass the fresh registry or wait on
    # a vanished thread's allocator/lease lock.
    operations = [lease.fork, lambda: store.add_client('child'), lambda: store.allocate(64),
                  lambda: store.acquire(slot), lambda: store.buffer(slot), store.snapshot]
    for operation in operations:
        try:
            operation()
        except RuntimeError as error:
            assert 'different process' in str(error), str(error)
        else:
            raise AssertionError('inherited handle accepted in fork child')
    assert storage.local_shm_store_snapshot()['capacity_bytes'] == 0
    try:
        storage.acquire_allocation(slot.descriptor())
    except ValueError as error:
        assert 'unknown shared-memory store' in str(error), str(error)
    else:
        raise AssertionError('parent descriptor resolved in child registry')
    # Direct construction must also avoid an inherited registry lock.
    own = storage.LocalShmStore(4096)
    own_lease = own.allocate(64)
    assert own_lease.allocation.shm_name != slot.shm_name
    own_lease.release()
    own.close()
    assert storage.local_shm_store_snapshot()['live_allocations'] == 0
    os._exit(0)

unlock.set()
thread.join(5)
try:
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0, status
    assert storage.current_store('second-parent-client') is store
    assert storage.local_shm_store_snapshot()['live_allocations'] == 1
    region = store.buffer(slot)
    assert bytes(region) == b'p' * 64
    region.release()
    print(json.dumps({'rejected': 6, 'parent_intact': True}), flush=True)
finally:
    lease.release()
    store.remove_client('second-parent-client')
    store.remove_client('parent')
    store.close()
"""


@pytest.mark.parametrize("held_lock", ["registry", "store", "lease"])
def test_forked_child_rejects_inherited_handles_before_taking_locks(held_lock):
    child = subprocess.run(
        [sys.executable, "-I", "-c", _INHERITED_HANDLES, held_lock],
        text=True,
        capture_output=True,
        timeout=20,
        env={**os.environ, "VANE_LOCAL_SHM_STORE_BYTES": "1m"},
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(child.stdout) == {"rejected": 6, "parent_intact": True}


_WORKER_EXIT_FINALIZER = """
import atexit, json, os, select, signal, sys, threading
from pathlib import Path
fork_child = False

def finish_child_exit():
    if fork_child:
        # Register before Vane imports so the inherited executor finalizer
        # must run before this callback. Skip only native-library teardown,
        # which is unrelated to Python cleanup in a multithreaded fork.
        os._exit(7 if worker._finalizer.alive else 0)

atexit.register(finish_child_exit)
import pyarrow as pa
from vane import pickle as vane_pickle
from vane.execution import udf_shm_store as storage
from vane.execution.udf_subprocess import _SingleSubprocessExecutor

def identity(table):
    return table

payload = {
    'function_pickle': vane_pickle.dumps(identity),
    'call_mode': 'map_batches',
    'execution_backend': 'subprocess_task',
    'produce_ref_bundle_output': True,
    'streaming_output_mode': 'local_shm_ref_bundle',
}
worker = _SingleSubprocessExecutor(payload)
peer = worker._shm_peer
proc = worker._proc

def submit(value):
    worker.submit(pa.table({'x': [value]}))
    result, finished = worker.take_ready_result()
    assert finished is False
    assert worker.take_ready_result() == (None, True)
    try:
        slot = result[1][0]._allocation_lease.allocation
        return result[1][0].to_table().column(0).to_pylist(), slot.shm_name
    finally:
        for ref in result[1]:
            ref.release()

locked, unlock = threading.Event(), threading.Event()
def hold_lock():
    with peer._lock:
        locked.set()
        assert unlock.wait(15)

thread = None
try:
    before, name = submit(11)
    assert before == [11]
    if sys.argv[1] == 'peer':
        thread = threading.Thread(target=hold_lock)
        thread.start()
        assert locked.wait(5)
    pid = os.fork()
    if pid == 0:
        fork_child = True
        signal.alarm(8)
        # sys.exit runs weakref/atexit finalizers; os._exit here would miss
        # the inherited peer's shutdown and the inherited-lock deadlock.
        sys.exit(0)
    unlock.set()
    if thread is not None:
        thread.join(5)
        assert not thread.is_alive()
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0, status
    assert proc.poll() is None
    # There are no borrowed inputs in this probe. EOF from shutdown would
    # make this socket readable even if its reader has already consumed EOF.
    assert not select.select([peer.sock], [], [], 0.1)[0], 'parent release channel closed'
    peer.check()
    after, next_name = submit(22)
    assert after == [22]
    assert next_name == name
    assert worker._proc is proc
    print(json.dumps({'after_fork': after, 'same_worker': True}), flush=True)
finally:
    if not fork_child:
        unlock.set()
        if thread is not None:
            thread.join(5)
        worker.close(kill=True)
assert proc.poll() is not None
assert peer._closed
assert peer.sock.fileno() == -1
assert not (Path('/dev/shm') / name).exists()
snapshot = storage.local_shm_store_snapshot()
assert snapshot['live_allocations'] == snapshot['mapped_capacity_bytes'] == 0
"""


@pytest.mark.parametrize("held_lock", ["none", "peer"])
def test_fork_exit_finalizer_preserves_parent_worker_release_channel(held_lock):
    child = subprocess.run(
        [sys.executable, "-I", "-c", _WORKER_EXIT_FINALIZER, held_lock],
        text=True,
        capture_output=True,
        timeout=30,
        env={**os.environ, "VANE_LOCAL_SHM_STORE_BYTES": "1m"},
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert "Exception ignored" not in child.stderr
    assert "Traceback" not in child.stderr
    assert json.loads(child.stdout) == {"after_fork": [22], "same_worker": True}

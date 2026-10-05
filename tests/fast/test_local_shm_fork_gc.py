# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Automatic collection must not invert shared-memory mutation lock order."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")


_RACE = """
import faulthandler, gc, os, sys, threading, time
import pyarrow as pa
from vane.execution import ref_bundle as refs
from vane.execution import udf_shm_store as storage

kind, phase = sys.argv[1:]
faulthandler.dump_traceback_later(6, exit=True)
gc.disable()
gc.collect()
store = storage.LocalShmStore(4096)
store.add_client('probe')
block = refs.prepare_local_shm_block(pa.table({'x': [11, 22, 33]}))
lease = store.allocate(block.ipc_size_bytes)
allocation = lease.allocation
region = store.buffer(allocation)
region[:8] = len(block.ipc).to_bytes(8, 'little')
region[8:] = memoryview(block.ipc).cast('B')
region.release()
ref = refs.LocalShmBlockRef(f'{allocation.identity}:0', allocation.size,
                          budget_bytes=0, allocation_lease=lease)
view = ref.to_table().column(0).chunk(0)
if kind == 'numpy':
    view = view.to_numpy(zero_copy_only=True)
ref.release()
del ref, lease, block, region
print('arena=' + allocation.shm_name, flush=True)
ready = threading.Event()
proceed_read, proceed_write = os.pipe()
def allocator():
    with store._lock:
        ready.set()
        assert os.read(proceed_read, 1) == b'1'
        owned = store.allocate(64)
    owned.release()
thread = threading.Thread(target=allocator, daemon=True)
thread.start()
assert ready.wait(2)
cycle = [view]
cycle.append(cycle)
del cycle, view

def schedule_allocator(phase, info):
    if phase == 'start' and storage._fork_lock._is_owned():
        # Force the allocator to wait for the collector's mutation lock before
        # the collector finalizes the unreachable Arrow/NumPy view.
        os.write(proceed_write, b'1')
        time.sleep(0.05)

gc.callbacks.append(schedule_allocator)
if phase == 'fork':
    gc.set_threshold(gc.get_count()[0] + 1, 100000, 100000)
else:
    # Arm collection inside the lock, after allocating its context-manager
    # entry. Collection before entry cannot exercise the lock-order inversion.
    gc.set_threshold(gc.get_count()[0] + 100000, 100000, 100000)
gc.enable()
if phase == 'fork':
    pid = os.fork()
    if pid == 0:
        os._exit(0)
else:
    with storage._fork_lock:
        gc.set_threshold(gc.get_count()[0] + 1, 100000, 100000)
        garbage = [[] for _ in range(100)]
        os.write(proceed_write, b'1')
# Event.set() can itself trigger GC before releasing this test's gate, even
# after the mutation lock is released. Signal with a nonallocating primitive.
os.write(proceed_write, b'1')
if phase == 'fork':
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
thread.join(2)
assert not thread.is_alive()
os.close(proceed_read)
os.close(proceed_write)
gc.callbacks.remove(schedule_allocator)
gc.collect()
store.remove_client('probe')
end = time.monotonic() + 2
while (store.snapshot()['mapped_capacity_bytes'] or not gc.isenabled()) and time.monotonic() < end:
    time.sleep(0.01)
assert store.snapshot()['mapped_capacity_bytes'] == 0
assert gc.isenabled()
faulthandler.cancel_dump_traceback_later()
print('ok', flush=True)
"""


def _run(script, *args):
    result = subprocess.run([sys.executable, "-I", "-c", script, *args], capture_output=True, text=True, timeout=15)
    try:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Traceback" not in result.stderr
        return result.stdout
    finally:
        # A faulthandler deadline bypasses Python cleanup on a regression.
        for line in result.stdout.splitlines():
            if line.startswith("arena="):
                (Path("/dev/shm") / line.removeprefix("arena=")).unlink(missing_ok=True)


@pytest.mark.parametrize("kind", ["arrow", "numpy"])
@pytest.mark.parametrize("phase", ["fork", "lock"])
def test_automatic_gc_of_cyclic_views_does_not_deadlock_concurrent_allocation(kind, phase):
    assert _run(_RACE, kind, phase).rstrip().endswith("ok")


_STATE = """
import faulthandler, gc, json, os, select, sys, threading, time
from vane.execution import udf_shm_store as storage

mode, enabled = sys.argv[1], sys.argv[2] == '1'
faulthandler.dump_traceback_later(8, exit=True)
store = storage.LocalShmStore(4096)
store.add_client('probe')
lease = store.allocate(64)
print('arena=' + lease.allocation.shm_name, flush=True)
errors = []
sys.unraisablehook = lambda error: errors.append(str(error.exc_value))
original_pipe, original_thread, original_hold = os.pipe, threading.Thread, storage._ForkHold
ready, finish = threading.Event(), threading.Event()
worker = None
gate_read, gate_write = os.pipe()

def wait_for(predicate):
    end = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < end, 'GC state synchronization timed out'
        time.sleep(0.001)

if mode == 'pipe':
    def fail_pipe():
        raise OSError('planned fork pipe failure')
    storage.os.pipe = fail_pipe
elif mode == 'watcher':
    def fail_watcher(*args, **kwargs):
        raise RuntimeError('planned fork watcher failure')
    storage.threading.Thread = fail_watcher
elif mode in ('waiter', 'inherited_gc_lock'):
    def contention():
        if mode == 'waiter':
            with storage._fork_lock:
                assert not gc.isenabled()
                ready.set()
                assert finish.wait(3)
                assert not gc.isenabled()
        else:
            # This lock must be replaced in the child before restoring GC.
            with storage._gc_lock:
                ready.set()
                assert select.select([gate_read], [], [], 3)[0]
                assert os.read(gate_read, 1) == b'1'
    def start_contention():
        global worker
        hold = original_hold()
        worker = original_thread(target=contention, daemon=True)
        worker.start()
        if mode == 'waiter':
            # Schedule the fork with a second pause already registered by the
            # blocked thread, so its count must be discarded in the child.
            wait_for(lambda: storage._gc_holds >= 2)
        else:
            assert ready.wait(3)
        return hold
    storage._ForkHold = start_contention

(gc.enable if enabled else gc.disable)()
if mode == 'nested':
    with storage._fork_lock:
        assert not gc.isenabled()
        try:
            with storage._fork_lock:
                assert not gc.isenabled()
                raise ValueError('planned body failure')
        except ValueError:
            pass
        assert not gc.isenabled()
elif mode == 'acquire':
    class FailedLock:
        def acquire(self):
            raise RuntimeError('planned acquire failure')
    lock = storage._ForkLock()
    lock._lock = FailedLock()
    try:
        lock.acquire()
    except RuntimeError as error:
        assert str(error) == 'planned acquire failure'
    else:
        raise AssertionError('acquire failure was not raised')
else:
    pid = os.fork()
    if pid == 0:
        correct = gc.isenabled() == enabled
        with storage._fork_lock:
            correct = correct and not gc.isenabled()
        correct = correct and gc.isenabled() == enabled
        os.write(gate_write, b'1')
        os._exit(0 if correct else 1)
    storage.os.pipe, storage.threading.Thread, storage._ForkHold = original_pipe, original_thread, original_hold
    if mode == 'waiter':
        assert ready.wait(3)
        assert not gc.isenabled(), 'owner restored GC while another thread held the lock'
        finish.set()
    if worker is not None:
        worker.join(3)
        assert not worker.is_alive()
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0, 'child did not restore its GC state'
    if mode in ('pipe', 'watcher'):
        assert len(errors) == 1 and 'planned fork' in errors[0], errors
    else:
        assert not errors, errors

storage.os.pipe, storage.threading.Thread, storage._ForkHold = original_pipe, original_thread, original_hold
lease.release()
store.remove_client('probe')
wait_for(lambda: storage._gc_holds == 0)
assert gc.isenabled() == enabled, 'parent did not restore its GC state'
os.close(gate_read)
os.close(gate_write)
faulthandler.cancel_dump_traceback_later()
print(json.dumps({'parent_gc': enabled, 'safe': True}), flush=True)
"""


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("mode", ["nested", "acquire", "pipe", "watcher", "waiter", "inherited_gc_lock"])
def test_gc_state_survives_nested_locks_failures_and_fork_with_other_threads(mode, enabled):
    output = _run(_STATE, mode, "1" if enabled else "0")
    assert json.loads(output.splitlines()[-1]) == {"parent_gc": enabled, "safe": True}

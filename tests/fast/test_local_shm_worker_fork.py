# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Worker input views survive fork, worker loss and inherited finalizers."""

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not hasattr(os, "fork"), reason="requires Linux fork/subreaper"
)


_INPUT_VIEWS = """
import ctypes, gc, json, os, signal, sys, threading, time
from pathlib import Path
import pyarrow as pa
from vane import pickle as vane_pickle
from vane.execution.udf_subprocess import _SingleSubprocessExecutor

kind, action, directory = sys.argv[1:]
root = Path(directory)
# Reap descendants after their worker exits instead of leaving host orphans.
assert ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0
def wait_for(predicate):
    end = time.monotonic() + 10
    while not predicate():
        assert time.monotonic() < end, 'worker fork probe timed out'
        time.sleep(0.01)

def publish(name, text):
    pending = root / (name + '.tmp')
    pending.write_text(text)
    pending.replace(root / name)

def identity(table):
    return table

def fork_reader(view):
    from vane.execution import udf_shm_store as storage
    errors = []
    previous_hook = sys.unraisablehook
    sys.unraisablehook = lambda error: errors.append(str(error.exc_value))
    original_pipe = os.pipe
    original_thread = storage.threading.Thread
    if action == 'pipe_failure':
        def fail_pipe():
            raise OSError('planned worker fork pipe failure')
        storage.os.pipe = fail_pipe
    if action == 'watcher_failure':
        def fail_thread(*args, **kwargs):
            raise RuntimeError('planned worker fork watcher failure')
        storage.threading.Thread = fail_thread
    pid = os.fork()
    storage.os.pipe = original_pipe
    storage.threading.Thread = original_thread
    sys.unraisablehook = previous_hook
    if pid == 0:
        signal.alarm(20)
        # Python 3.12.14's tracker destructor waits for its own helper. This
        # read-only descendant needs no tracker; close that unrelated inherited
        # pipe so it does not prevent the worker's normal interpreter exit.
        # All Vane lifetime/release descriptors remain untouched.
        from multiprocessing import resource_tracker
        tracker = resource_tracker._resource_tracker
        if tracker._fd is not None:
            os.close(tracker._fd)
            tracker._fd = None
            tracker._pid = None
        if action == 'descendant' and os.fork() != 0:
            os._exit(0)
        publish('ready', str(os.getpid()))
        wait_for(lambda: (root / 'read').exists())
        values = view.tolist() if kind == 'numpy' else view.to_pylist()
        publish('observed', json.dumps(values))
        wait_for(lambda: (root / 'exit').exists())
        os._exit(0)
    expected = {
        'pipe_failure': ['planned worker fork pipe failure'],
        'watcher_failure': ['planned worker fork watcher failure'],
    }.get(action, [])
    assert errors == expected, errors
    def reap(child):
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        (root / 'reaped').touch()
    if action != 'late_close':
        threading.Thread(target=reap, args=(pid,), daemon=True).start()

def fork_task(table):
    view = table.column(0).chunk(0)
    if kind == 'numpy':
        view = view.to_numpy(zero_copy_only=True)
    if action == 'late_close':
        import atexit
        # This callback runs after worker_main's finally closes the client.
        atexit.register(fork_reader, view)
    else:
        fork_reader(view)
    return pa.table({'x': [0]})

class RetainingActor:
    def __init__(self):
        self.kept = None
    def __call__(self, table):
        if self.kept is None:
            self.kept = table.column(0).chunk(0)
            if kind == 'numpy':
                self.kept = self.kept.to_numpy(zero_copy_only=True)
        else:
            fork_reader(self.kept)
            self.kept = None
        return pa.table({'x': [0]})

def payload(fn, backend='subprocess_task'):
    return {
        'function_pickle': vane_pickle.dumps(fn), 'call_mode': 'map_batches',
        'execution_backend': backend, 'produce_ref_bundle_output': True,
        'streaming_output_mode': 'local_shm_ref_bundle',
    }
producer = _SingleSubprocessExecutor(payload(identity))
consumer = _SingleSubprocessExecutor(payload(RetainingActor, 'subprocess_actor')
                                    if action == 'actor' else payload(fork_task))
def result(worker):
    value, finished = worker.take_ready_result()
    assert not finished
    assert worker.take_ready_result() == (None, True)
    return value
def release(value):
    if value is not None:
        for ref in value[1]:
            ref.release()

first = second = consumed = None
store = None
try:
    producer.submit(pa.table({'x': [11] * 2048}))
    first = result(producer)
    slot = first[1][0]._allocation_lease.allocation
    store = first[1][0]._allocation_lease.store
    consumer.submit_ref_bundle(first[1], None, first[2], first[3])
    consumed = result(consumer)
    release(first)
    if action == 'actor':
        release(consumed)
        consumer.submit(pa.table({'x': [99]}))
        consumed = result(consumer)
    release(consumed)
    gc.collect()
    if action == 'late_close':
        consumer.close(kill=False)
    wait_for(lambda: (root / 'ready').exists())
    if action == 'descendant':
        wait_for(lambda: (root / 'reaped').exists())
    if action in ('close', 'kill', 'descendant', 'cleanup_retry'):
        peer = consumer._shm_peer
        if action == 'cleanup_retry':
            original_remove = store.remove_client
            failed = False
            def fail_once(client_id):
                global failed
                if client_id == peer.client_id and not failed:
                    failed = True
                    raise RuntimeError('planned client retirement failure')
                original_remove(client_id)
            store.remove_client = fail_once
            try:
                consumer.close(kill=True)
            except RuntimeError as error:
                assert 'planned client retirement failure' in str(error)
            else:
                raise AssertionError('cleanup failure was not reported')
            consumer.close(kill=True)
            store.remove_client = original_remove
        elif action == 'close':
            consumer.close(kill=False)
        else:
            consumer.close(kill=True)
    # Give last-buffer notifications time to reach the driver before reuse.
    time.sleep(0.15)
    producer.submit(pa.table({'x': [22] * 2048}))
    second = result(producer)
    assert second[1][0]._allocation_lease.allocation.offset != slot.offset
    release(second)
    if action in ('close', 'kill', 'descendant', 'cleanup_retry', 'late_close'):
        producer.close(kill=True)
        assert store.snapshot()['live_allocations'] == 1
        assert (Path('/dev/shm') / slot.shm_name).exists()
    (root / 'read').touch()
    wait_for(lambda: (root / 'observed').exists())
    assert json.loads((root / 'observed').read_text()) == [11] * 2048
    (root / 'exit').touch()
    if action in ('close', 'kill', 'descendant', 'cleanup_retry', 'late_close'):
        pid = int((root / 'ready').read_text())
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        # No allocator access: the watcher must reclaim an idle arena.
        wait_for(lambda: not (Path('/dev/shm') / slot.shm_name).exists())
    else:
        wait_for(lambda: (root / 'reaped').exists())
        if action in ('pipe_failure', 'watcher_failure'):
            consumer.close(kill=True)
        wait_for(lambda: store.snapshot()['live_allocations'] == 0)
        reused = store.allocate(slot.size)
        assert reused.allocation.offset == slot.offset
        reused.release()
    print(json.dumps({'intact': True, 'reclaimed': True}), flush=True)
finally:
    (root / 'read').touch()
    (root / 'exit').touch()
    release(first)
    release(second)
    release(consumed)
    consumer.close(kill=True)
    producer.close(kill=True)
    # Reap only this probe's child: the Python resource tracker intentionally
    # stays alive until interpreter shutdown and must not be waited for here.
    if (root / 'ready').exists():
        try:
            os.waitpid(int((root / 'ready').read_text()), 0)
        except ChildProcessError:
            pass
    while True:
        try:
            child, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if child == 0:
            break
    if store is not None:
        wait_for(lambda: store.snapshot()['live_allocations'] == 0)
        assert store.snapshot()['mapped_capacity_bytes'] == 0
"""


@pytest.mark.parametrize("kind", ["arrow", "numpy"])
@pytest.mark.parametrize(
    "action",
    ["reuse", "actor", "close", "kill", "descendant", "cleanup_retry", "late_close", "pipe_failure", "watcher_failure"],
)
def test_worker_fork_retains_remote_views_through_reuse_and_worker_exit(tmp_path, kind, action):
    result = subprocess.run(
        [sys.executable, "-I", "-c", _INPUT_VIEWS, kind, action, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=45,
        env={**os.environ, "VANE_LOCAL_SHM_STORE_BYTES": "128k"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    assert json.loads(result.stdout) == {"intact": True, "reclaimed": True}


_NORMAL_EXIT = """
import os, sys
import pyarrow as pa
from vane import pickle as vane_pickle
from vane.execution.udf_subprocess import _SingleSubprocessExecutor

def fork_and_exit(table):
    import atexit, signal
    from vane.execution.udf_shm_store import worker_shm_client
    client = worker_shm_client()
    pid = os.fork()
    if pid == 0:
        signal.alarm(10)
        # Guard release paths before touching a queue inherited from another
        # thread. Python finally blocks still run before the exit callback.
        class ForbiddenQueue:
            def put(self, value):
                raise AssertionError('inherited release queue was touched')
        client._queue = ForbiddenQueue()
        client.release(1)
        client.end_input()
        try:
            client.begin_input({'block_refs': []})
        except RuntimeError as error:
            assert 'different process' in str(error)
        else:
            raise AssertionError('inherited client accepted new input')
        atexit.register(lambda: os._exit(0))
        sys.exit(0)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    return table

worker = _SingleSubprocessExecutor({
    'function_pickle': vane_pickle.dumps(fork_and_exit), 'call_mode': 'map_batches',
    'execution_backend': 'subprocess_task', 'produce_ref_bundle_output': True,
    'streaming_output_mode': 'local_shm_ref_bundle',
})
try:
    for value in [11, 22]:
        worker.submit(pa.table({'x': [value]}))
        result, finished = worker.take_ready_result()
        assert not finished
        try:
            assert result[1][0].to_table().column(0).to_pylist() == [value]
            assert worker.take_ready_result() == (None, True)
            worker._shm_peer.check()
        finally:
            for ref in result[1]:
                ref.release()
finally:
    worker.close(kill=True)
print('ok', flush=True)
"""


def test_normal_worker_fork_exit_preserves_parent_channel_and_rejects_inherited_operations():
    result = subprocess.run([sys.executable, "-I", "-c", _NORMAL_EXIT], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout.strip() == "ok"

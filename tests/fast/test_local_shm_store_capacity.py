# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Physical capacity waits, cancellation, and ownership across cleanup/fork."""

import os
import subprocess
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import nullcontext

import pytest

from vane.execution.udf_lifecycle import ExecutionCancellationScope, ExecutionCancelledError
from vane.execution.udf_shm_store import LocalShmStore, LocalShmStoreCapacityError


def test_darwin_auto_capacity_uses_available_vm_without_linux_paths(monkeypatch):
    from types import SimpleNamespace

    import psutil

    from vane.execution import ref_bundle

    def no_linux_path(*args, **kwargs):
        pytest.fail("Darwin shared-memory capacity must not probe Linux paths")

    monkeypatch.setattr(ref_bundle.sys, "platform", "darwin")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=10 * 1024**3))
    monkeypatch.setattr("builtins.open", no_linux_path)
    monkeypatch.setattr(os, "statvfs", no_linux_path)
    assert ref_bundle._auto_local_shm_store_capacity_bytes() == 3 * 1024**3


def test_darwin_auto_capacity_rejects_exhausted_vm(monkeypatch):
    from types import SimpleNamespace

    import psutil

    from vane.execution import ref_bundle

    monkeypatch.setattr(ref_bundle.sys, "platform", "darwin")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=0))
    with pytest.raises(RuntimeError, match="no available memory"):
        ref_bundle._auto_local_shm_store_capacity_bytes()


def test_arena_without_madv_remove_preserves_live_views_and_closes(monkeypatch):
    import mmap

    monkeypatch.delattr(mmap, "MADV_REMOVE", raising=False)
    arena = LocalShmStore(8192)
    arena.add_client("retained-view")
    first = arena.allocate(4096)
    retained = arena.allocate(4096)
    view = arena.buffer(retained.allocation)
    view[:] = b"x" * len(view)
    try:
        first.release()
        arena.remove_client("retained-view")
        assert bytes(view) == b"x" * 4096
        assert arena._shm is not None
    finally:
        view.release()
        retained.release()
    assert arena._shm is None
    assert arena.snapshot()["live_allocations"] == 0


@pytest.fixture
def store():
    result = LocalShmStore(4096)
    result.add_client("capacity-test")
    yield result
    result.remove_client("capacity-test")
    assert result.snapshot()["live_allocations"] == 0
    assert result._waiting_writes == {}


@pytest.mark.parametrize("cancel", [False, True])
def test_physical_wait_wakes_on_release_or_cancellation(store, cancel):
    held = store.allocate(store.capacity)
    scope = ExecutionCancellationScope("wait", 1)
    waiting = threading.Event()

    def wait_context():
        waiting.set()
        return nullcontext()

    with ThreadPoolExecutor(max_workers=1) as threads:
        future = threads.submit(store.wait_for_capacity, 1024, (), scope, wait_context, frozenset())
        try:
            assert waiting.wait(timeout=5)
            with pytest.raises(FutureTimeoutError):
                future.result(timeout=0.05)
            if cancel:
                scope.cancel("query stopped")
                with pytest.raises(ExecutionCancelledError, match="query stopped"):
                    future.result(timeout=5)
            else:
                held.release()
                future.result(timeout=5)
                output = store.allocate(1024)
                output.release()
        finally:
            scope.cancel()
            held.release()


def test_output_larger_than_space_left_by_its_input_fails_promptly(store):
    held = store.allocate(3072)
    try:
        with pytest.raises(LocalShmStoreCapacityError, match="pinned_input_bytes=3072"):
            store.wait_for_capacity(
                2048, (held.allocation,), ExecutionCancellationScope("too-large", 1), nullcontext, frozenset()
            )
    finally:
        held.release()


@pytest.mark.parametrize("outcome", ["release", "noop", "error", "cancel", "pinned"])
def test_capacity_wait_tracks_inflight_prefetch_reclaim(store, outcome):
    held = store.allocate(3072, producer_id="queued-stream")
    started, resume, waiting = (threading.Event() for _ in range(3))
    scope = ExecutionCancellationScope("inflight-reclaim", 1)

    def reclaim():
        started.set()
        assert resume.wait(5), "reclaim was not resumed"
        if outcome == "release":
            held.release()
        elif outcome == "error":
            raise RuntimeError("planned reclaim failure")

    def wait_context():
        waiting.set()
        return nullcontext()

    unregister = store.register_prefetch(reclaim)
    try:
        with ThreadPoolExecutor(max_workers=2) as threads:
            reclaiming = threads.submit(store.reclaim_prefetched_inputs)
            try:
                assert started.wait(5)
                # Unregistering cannot retract a callback already being run.
                unregister()
                pending = threads.submit(
                    store.wait_for_capacity,
                    2048,
                    (held.allocation,) if outcome == "pinned" else (),
                    scope,
                    wait_context,
                    frozenset({"queued-stream"}),
                )
                assert waiting.wait(5)
                if outcome == "pinned":
                    # Claimed input cannot be reclaimed, even if another callback
                    # is active. Keep the unconditional physical capacity bound.
                    with pytest.raises(LocalShmStoreCapacityError, match="pinned_input_bytes=3072"):
                        pending.result(timeout=5)
                else:
                    with pytest.raises(FutureTimeoutError):
                        pending.result(timeout=0.05)
                if outcome == "cancel":
                    scope.cancel("cancel while reclaiming")
                    with pytest.raises(ExecutionCancelledError, match="cancel while reclaiming"):
                        pending.result(timeout=5)
                resume.set()
                if outcome == "error":
                    with pytest.raises(RuntimeError, match="planned reclaim failure"):
                        reclaiming.result(timeout=5)
                else:
                    reclaiming.result(timeout=5)
                if outcome == "release":
                    pending.result(timeout=5)
                    store.allocate(2048).release()
                elif outcome in {"noop", "error"}:
                    # A no-op or failed callback must wake capacity waiters to
                    # recheck the dependency, even without a physical release.
                    with pytest.raises(LocalShmStoreCapacityError, match="queued_input_bytes=3072"):
                        pending.result(timeout=5)
            finally:
                scope.cancel()
                resume.set()
    finally:
        unregister()
        held.release()


def test_capacity_wait_tracks_all_concurrent_prefetch_reclaim_batches(store):
    held = store.allocate(3072, producer_id="queued-stream")
    started = [threading.Event(), threading.Event()]
    resume = [threading.Event(), threading.Event()]
    calls = []
    scope = ExecutionCancellationScope("concurrent-reclaims", 1)

    def reclaim(index):
        calls.append(index)
        started[index].set()
        assert resume[index].wait(5), "reclaim was not resumed"
        if index == 1:
            held.release()

    unregister = []
    try:
        with ThreadPoolExecutor(max_workers=3) as threads:
            try:
                reclaiming = []
                for index in range(2):
                    unregister.append(store.register_prefetch(lambda index=index: reclaim(index)))
                    reclaiming.append(threads.submit(store.reclaim_prefetched_inputs))
                    assert started[index].wait(5)
                pending = threads.submit(
                    store.wait_for_capacity, 2048, (), scope, nullcontext, frozenset({"queued-stream"})
                )
                with pytest.raises(FutureTimeoutError):
                    pending.result(timeout=0.05)
                resume[0].set()
                reclaiming[0].result(timeout=5)
                with pytest.raises(FutureTimeoutError):
                    pending.result(timeout=0.05)
                resume[1].set()
                reclaiming[1].result(timeout=5)
                pending.result(timeout=5)
                assert calls == [0, 1]
            finally:
                scope.cancel()
                for event in resume:
                    event.set()
    finally:
        for callback in unregister:
            callback()
        held.release()


def test_prefetch_reclaim_failure_does_not_discard_other_callbacks(store):
    called = []

    def reclaim(index):
        called.append(index)
        raise RuntimeError("planned reclaim failure")

    for index in range(3):
        store.register_prefetch(lambda index=index: reclaim(index))
    with pytest.raises(RuntimeError, match="planned reclaim failure"):
        store.reclaim_prefetched_inputs()
    assert sorted(called) == [0, 1, 2]
    # Failed callbacks are still owned by their invocation's cleanup path,
    # rather than run a second time by an unrelated output allocation.
    store.reclaim_prefetched_inputs()
    assert len(called) == 3


@pytest.mark.parametrize("release_before_report", [False, True])
def test_idle_worker_input_report_distinguishes_retained_views_from_pending_releases(
    store, monkeypatch, release_before_report
):
    from vane.execution import udf_shm_store as storage

    def current_store(client_id):
        store.add_client(client_id)
        return store

    monkeypatch.setattr(storage, "current_store", current_store)
    peer = storage.ParentShmPeer()
    held = store.allocate(3072)
    payload = {"block_refs": [{"allocation": held.allocation.descriptor()}]}
    token = peer.borrow_inputs(payload)[0]
    held.release()
    scope = ExecutionCancellationScope("retained-input", 1)
    try:
        if release_before_report:
            peer.release_read(token)
        peer.retain_inputs({token})
        if release_before_report:
            store.wait_for_capacity(2048, (), scope, nullcontext, frozenset())
        else:
            with pytest.raises(LocalShmStoreCapacityError, match="retained_input_bytes=3072"):
                store.wait_for_capacity(2048, (), scope, nullcontext, frozenset())
            # Once another call starts, this reader can release independently
            # of the waiting writer. Clearing the idle observation permits a wait.
            peer.retain_inputs(set())
            with ThreadPoolExecutor(max_workers=1) as threads:
                pending = threads.submit(store.wait_for_capacity, 2048, (), scope, nullcontext, frozenset())
                try:
                    with pytest.raises(FutureTimeoutError):
                        pending.result(timeout=0.05)
                    peer.release_read(token)
                    pending.result(timeout=5)
                finally:
                    scope.cancel()
    finally:
        scope.cancel()
        peer.close_after_exit()
    assert store._retained_inputs == {}


def test_idle_worker_report_wakes_an_already_waiting_writer(store, monkeypatch):
    from vane.execution import udf_shm_store as storage

    def current_store(client_id):
        store.add_client(client_id)
        return store

    monkeypatch.setattr(storage, "current_store", current_store)
    peer = storage.ParentShmPeer()
    held = store.allocate(3072)
    token = peer.borrow_inputs({"block_refs": [{"allocation": held.allocation.descriptor()}]})[0]
    held.release()
    scope = ExecutionCancellationScope("later-retained-report", 1)
    try:
        with ThreadPoolExecutor(max_workers=1) as threads:
            pending = threads.submit(store.wait_for_capacity, 2048, (), scope, nullcontext, frozenset())
            try:
                with store._capacity_condition:
                    assert store._capacity_condition.wait_for(lambda: bool(store._waiting_writes), timeout=5)
                peer.retain_inputs({token})
                with pytest.raises(LocalShmStoreCapacityError, match="retained_input_bytes=3072"):
                    pending.result(timeout=5)
            finally:
                scope.cancel()
    finally:
        peer.close_after_exit()
    assert store._retained_inputs == {}


@pytest.mark.parametrize("queued", [False, True])
def test_mutually_blocked_writers_report_capacity_and_unblock_peer_on_cleanup(store, queued):
    inputs = [store.allocate(2048, producer_id=str(index)) for index in range(2)]
    scopes = [ExecutionCancellationScope("cycle", index + 1) for index in range(2)]

    def write(index):
        try:
            store.wait_for_capacity(
                2048,
                () if queued else (inputs[index].allocation,),
                scopes[index],
                nullcontext,
                frozenset({str(index)}),
            )
            return "ready"
        except LocalShmStoreCapacityError:
            return "capacity"
        finally:
            # Mirrors cleanup of the failed task, whose input keeps the other
            # task from completing until its capacity error is delivered.
            inputs[index].release()

    with ThreadPoolExecutor(max_workers=2) as threads:
        futures = [threads.submit(write, index) for index in range(2)]
        try:
            assert sorted(future.result(timeout=5) for future in futures) == ["capacity", "ready"]
        finally:
            for scope in scopes:
                scope.cancel()
            for lease in inputs:
                lease.release()


@pytest.mark.parametrize(
    "producer,reader_active,reader_finished",
    [("upstream", False, False), ("upstream", True, False), ("unrelated", False, False), ("upstream", True, True)],
)
@pytest.mark.parametrize("current_view", [False, True])
def test_capacity_wait_tracks_queued_input_stream_dependencies(producer, reader_active, reader_finished, current_view):
    store = LocalShmStore(8192)
    store.add_client("queued-inputs")
    current = store.allocate(3072, producer_id="upstream")
    queued = store.allocate(3072, producer_id=producer)
    store.start_input(current.allocation, ("consumer", 1))
    scope = ExecutionCancellationScope("consumer", 1)
    try:
        if not current_view:
            store.finish_input(current.allocation, ("consumer", 1))
            current.release()
        if reader_active:
            store.start_input(queued.allocation, ("independent", 1))
        if reader_finished:
            store.finish_input(queued.allocation, ("independent", 1))
        with ThreadPoolExecutor(max_workers=1) as threads:
            pending = threads.submit(
                store.wait_for_capacity,
                4096,
                (current.allocation,) if current_view else (),
                scope,
                nullcontext,
                frozenset({"upstream"}),
            )
            try:
                if producer == "unrelated" or (reader_active and not reader_finished):
                    with pytest.raises(FutureTimeoutError):
                        pending.result(timeout=0.05)
                    queued.release()
                    pending.result(timeout=5)
                else:
                    with pytest.raises(LocalShmStoreCapacityError, match="queued_input_bytes=3072"):
                        pending.result(timeout=5)
            finally:
                scope.cancel()
    finally:
        current.release()
        queued.release()
        store.remove_client("queued-inputs")
    assert store.snapshot()["live_allocations"] == 0


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_capacity_wait_distinguishes_queued_refs_from_fork_only_holds():
    script = r"""
import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import pytest
from vane.execution.udf_lifecycle import ExecutionCancellationScope
from vane.execution.udf_shm_store import LocalShmStore, LocalShmStoreCapacityError

store = LocalShmStore(4096)
store.add_client('fork-capacity')
previous = store.allocate(3072, producer_id='upstream')
slot = previous.allocation
view = store.buffer(slot)
view[:] = b'x' * len(view)
budget_released = threading.Event()
store.retain_budget(slot, budget_released.set)
read_gate, write_gate = os.pipe()
pid = os.fork()
if pid == 0:
    signal.alarm(15)
    os.close(write_gate)
    assert os.read(read_gate, 1) == b'1'
    assert bytes(view) == b'x' * 3072
    os._exit(0)
os.close(read_gate)
view.release()
current = None
reaped = False
scope = ExecutionCancellationScope('fork-capacity', 1)
try:
    # Allocate after fork so the child holds only the previous input.
    current = store.allocate(1024, producer_id='upstream')
    with store._lock:
        assert store._live[slot.generation].forks
    # A live parent reference can still belong to a queued batch. Having
    # a fork owner as well does not make that batch independently releasable.
    with pytest.raises(LocalShmStoreCapacityError, match='queued_input_bytes=3072'):
        store.wait_for_capacity(2048, (current.allocation,), scope, nullcontext, frozenset({'upstream'}))
    previous.release()
    with store._lock:
        entry = store._live[slot.generation]
        assert entry.refs == 0 and entry.forks and not entry.input_readers
    with ThreadPoolExecutor(max_workers=1) as threads:
        pending = threads.submit(
            store.wait_for_capacity, 2048, (current.allocation,), scope, nullcontext, frozenset({'upstream'})
        )
        try:
            with store._capacity_condition:
                waiting = store._capacity_condition.wait_for(lambda: bool(store._waiting_writes), timeout=2)
            if not waiting:
                pending.result(timeout=1)
                raise AssertionError('fork-only input did not enter a capacity wait')
            # Excluding a fork-only hold from the dependency set must neither
            # return its byte charge nor permit reuse while the child reads it.
            assert not budget_released.is_set()
            with pytest.raises(LocalShmStoreCapacityError):
                store.allocate(2048)
            os.write(write_gate, b'1')
            _, status = os.waitpid(pid, 0)
            reaped = True
            assert os.waitstatus_to_exitcode(status) == 0
            pending.result(timeout=5)
            assert budget_released.wait(5)
            output = store.allocate(2048)
            assert output.allocation.offset == slot.offset
            output.release()
        finally:
            scope.cancel()
finally:
    os.close(write_gate)
    if not reaped:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
    previous.release()
    if current is not None:
        current.release()
    store.remove_client('fork-capacity')
assert store.snapshot()['live_allocations'] == 0
assert store.snapshot()['mapped_capacity_bytes'] == 0
"""
    result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("previous_view", ["retained", "pending_release", "released"])
def test_waiting_worker_pins_only_reported_views_from_earlier_calls(store, monkeypatch, previous_view):
    from vane.execution import udf_shm_store as storage

    def current_store(client_id):
        store.add_client(client_id)
        return store

    monkeypatch.setattr(storage, "current_store", current_store)
    peer = storage.ParentShmPeer()
    previous, current = store.allocate(2048), store.allocate(1024)
    try:
        old_token = peer.borrow_inputs({"block_refs": [{"allocation": previous.allocation.descriptor()}]})[0]
        peer.retain_inputs({old_token})
        peer.retain_inputs(set())  # The next call starts but still holds the old view.
        current_token = peer.borrow_inputs({"block_refs": [{"allocation": current.allocation.descriptor()}]})[0]
        previous.release()
        current.release()
        tokens = {current_token}
        if previous_view != "pending_release":
            tokens.add(old_token)
        if previous_view == "released":
            # The release channel may overtake the control message's snapshot.
            peer.release_read(old_token)
        scope = ExecutionCancellationScope("history", 2)
        with ThreadPoolExecutor(max_workers=1) as threads:
            pending = threads.submit(
                store.wait_for_capacity, 2048, peer.input_allocations(tokens), scope, nullcontext, frozenset()
            )
            try:
                if previous_view == "retained":
                    with pytest.raises(LocalShmStoreCapacityError, match="pinned_input_bytes=3072"):
                        pending.result(timeout=5)
                else:
                    if previous_view == "pending_release":
                        with pytest.raises(FutureTimeoutError):
                            pending.result(timeout=0.05)
                        peer.release_read(old_token)
                    pending.result(timeout=5)
            finally:
                scope.cancel()
    finally:
        previous.release()
        current.release()
        peer.close_after_exit()
    assert peer.input_allocations({old_token, current_token}) == ()


def test_fallible_budget_notification_does_not_repeat_physical_release(store):
    lease = store.allocate(1024)
    calls = []

    def notification():
        calls.append("released")
        raise RuntimeError("notification failed")

    store.retain_budget(lease.allocation, notification)
    with pytest.raises(RuntimeError, match="notification failed"):
        lease.release()
    lease.release()
    assert calls == ["released"]
    assert store.snapshot()["live_bytes"] == 0


@pytest.mark.parametrize("unrelated_bytes", [0, 128])
def test_mutually_blocked_writers_ignore_unrelated_retained_views(unrelated_bytes):
    store = LocalShmStore(8192)
    store.add_client("retained-cycle")
    inputs = [store.allocate(3072), store.allocate(3072)]
    unrelated = store.allocate(unrelated_bytes) if unrelated_bytes else None
    scopes = [ExecutionCancellationScope("retained-cycle", index + 1) for index in range(2)]

    def write(index):
        try:
            store.wait_for_capacity(4096, (inputs[index].allocation,), scopes[index], nullcontext, frozenset())
            return "ready"
        except LocalShmStoreCapacityError:
            return "capacity"
        finally:
            inputs[index].release()

    try:
        with ThreadPoolExecutor(max_workers=2) as threads:
            futures = [threads.submit(write, index) for index in range(2)]
            try:
                done, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
                assert done, "no writer reported the capacity cycle"
                assert any(future.result() == "capacity" for future in done)
                # The unrelated view also fragments the newly freed region.
                # After the cycle is reported, releasing it lets the peer fit.
                if unrelated is not None:
                    unrelated.release()
                assert sorted(future.result(timeout=5) for future in futures) == ["capacity", "ready"]
            finally:
                for scope in scopes:
                    scope.cancel()
    finally:
        for lease in inputs:
            lease.release()
        if unrelated is not None:
            unrelated.release()
        store.remove_client("retained-cycle")
    assert store.snapshot()["live_allocations"] == 0


def test_unrelated_allocation_can_still_release_space_for_waiting_writers():
    store = LocalShmStore(8192)
    store.add_client("reclaimable-space")
    inputs = [store.allocate(1024), store.allocate(1024)]
    unrelated = store.allocate(4096)
    scopes = [ExecutionCancellationScope("reclaimable-space", index + 1) for index in range(2)]

    def write(index):
        try:
            store.wait_for_capacity(3072, (inputs[index].allocation,), scopes[index], nullcontext, frozenset())
        finally:
            inputs[index].release()

    try:
        with ThreadPoolExecutor(max_workers=2) as threads:
            futures = [threads.submit(write, index) for index in range(2)]
            try:
                with store._capacity_condition:
                    assert store._capacity_condition.wait_for(lambda: len(store._waiting_writes) == 2, timeout=5)
                for future in futures:
                    with pytest.raises(FutureTimeoutError):
                        future.result(timeout=0.05)
                unrelated.release()
                for future in futures:
                    future.result(timeout=5)
            finally:
                for scope in scopes:
                    scope.cancel()
    finally:
        unrelated.release()
        for lease in inputs:
            lease.release()
        store.remove_client("reclaimable-space")
    assert store.snapshot()["live_allocations"] == 0


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_fork_resets_budget_and_inherited_ref_cleanup_skips_parent_lock():
    script = r"""
import os
import signal
import threading
from vane.execution import ref_bundle as refs
from vane.execution.udf_shm_store import LocalShmStore

store = LocalShmStore(4096)
store.add_client('parent')
lease = store.allocate(1024)
ref = refs.LocalShmBlockRef(lease.allocation.identity, 1024, allocation_lease=lease)
manager = refs.local_shm_budget_manager()
locked = threading.Event()
release = threading.Event()
def hold():
    with manager._cond:
        locked.set()
        release.wait(10)
thread = threading.Thread(target=hold)
thread.start()
assert locked.wait(5)
pid = os.fork()
if pid == 0:
    signal.alarm(5)
    child = refs.local_shm_budget_manager()
    assert child is not manager
    assert child.snapshot()['usage_bytes'] == 0
    child.acquire_allocation(64)
    ref._finalizer()
    lease.release()
    assert child.snapshot()['usage_bytes'] == 64
    child.release_allocation(64)
    os._exit(0)
try:
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0, status
finally:
    release.set()
    thread.join(5)
ref.release()
store.remove_client('parent')
assert manager.snapshot()['usage_bytes'] == 0
"""
    result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr

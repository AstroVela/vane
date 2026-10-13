# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Physical capacity waits, cancellation, and ownership across cleanup/fork."""

import os
import subprocess
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext

import pytest

from vane.execution.udf_lifecycle import ExecutionCancellationScope, ExecutionCancelledError
from vane.execution.udf_shm_store import LocalShmStore, LocalShmStoreCapacityError


@pytest.fixture
def store():
    result = LocalShmStore(4096)
    result.add_client("capacity-test")
    yield result
    result.remove_client("capacity-test")
    assert result.snapshot()["live_allocations"] == 0


@pytest.mark.parametrize("cancel", [False, True])
def test_physical_wait_wakes_on_release_or_cancellation(store, cancel):
    held = store.allocate(store.capacity)
    scope = ExecutionCancellationScope("wait", 1)
    waiting = threading.Event()

    def wait_context():
        waiting.set()
        return nullcontext()

    with ThreadPoolExecutor(max_workers=1) as threads:
        future = threads.submit(store.wait_for_capacity, 1024, (), scope, wait_context)
        try:
            assert waiting.wait(timeout=5)
            with pytest.raises(TimeoutError):
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
            store.wait_for_capacity(2048, (held.allocation,), ExecutionCancellationScope("too-large", 1), nullcontext)
    finally:
        held.release()


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
            store.wait_for_capacity(2048, (), scope, nullcontext)
        else:
            with pytest.raises(LocalShmStoreCapacityError, match="retained_input_bytes=3072"):
                store.wait_for_capacity(2048, (), scope, nullcontext)
            # Once another call starts, this reader can release independently
            # of the waiting writer. Clearing the idle observation permits a wait.
            peer.retain_inputs(set())
            with ThreadPoolExecutor(max_workers=1) as threads:
                pending = threads.submit(store.wait_for_capacity, 2048, (), scope, nullcontext)
                try:
                    with pytest.raises(TimeoutError):
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
            pending = threads.submit(store.wait_for_capacity, 2048, (), scope, nullcontext)
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


def test_mutually_blocked_writers_report_capacity_and_unblock_peer_on_cleanup(store):
    inputs = [store.allocate(2048), store.allocate(2048)]
    scopes = [ExecutionCancellationScope("cycle", index + 1) for index in range(2)]

    def write(index):
        try:
            store.wait_for_capacity(2048, (inputs[index].allocation,), scopes[index], nullcontext)
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
            store.wait_for_capacity(4096, (inputs[index].allocation,), scopes[index], nullcontext)
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
            store.wait_for_capacity(3072, (inputs[index].allocation,), scopes[index], nullcontext)
        finally:
            inputs[index].release()

    try:
        with ThreadPoolExecutor(max_workers=2) as threads:
            futures = [threads.submit(write, index) for index in range(2)]
            try:
                with store._capacity_condition:
                    assert store._capacity_condition.wait_for(lambda: len(store._waiting_writes) == 2, timeout=5)
                for future in futures:
                    with pytest.raises(TimeoutError):
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

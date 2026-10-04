# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Allocation lifetime, safe reuse, and real child-process borrowed views."""

from __future__ import annotations

import gc
import json
import select
import subprocess
import sys
import time

import pyarrow as pa
import pytest

from vane.execution import ref_bundle as refs
from vane.execution import udf_shm_store as storage


@pytest.fixture
def store():
    result = storage.LocalShmStore(64 * 1024)
    result.add_client("test")
    yield result
    result.remove_client("test")
    assert result.snapshot()["live_allocations"] == 0
    result.close()


def _input(store, values, *, track_budget=False):
    table = pa.table({"value": values})
    block = refs.prepare_local_shm_block(table)
    lease = store.allocate(block.ipc_size_bytes)
    allocation = lease.allocation
    region = store.buffer(allocation)
    try:
        region[:8] = len(block.ipc).to_bytes(8, "little")
        region[8:] = memoryview(block.ipc).cast("B")
    except BaseException:
        lease.release()
        raise
    finally:
        region.release()
    return refs.LocalShmBlockRef(
        f"{allocation.identity}:0", allocation.size, budget_bytes=None if track_budget else 0, allocation_lease=lease
    )


def test_reuse_coalesces_free_blocks_and_rejects_stale_generation(store):
    first = store.allocate(4096)
    middle = store.allocate(4096)
    last = store.allocate(4096)
    old = first.allocation
    first.release()
    middle.release()
    reused = store.allocate(8192)
    try:
        assert reused.allocation.offset == old.offset
        assert reused.allocation.generation != old.generation
        with pytest.raises(ValueError, match="stale"):
            store.acquire(old)
        assert store.snapshot()["reused_allocations"] == 1
    finally:
        reused.release()
        last.release()


def test_retained_arrow_slice_pins_data_after_ref_and_transport_budget_release(store, monkeypatch):
    manager = refs.LocalShmBudgetManager(limit_factory=lambda: store.capacity)
    monkeypatch.setattr(refs, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    ref = _input(store, [11, 22, 33], track_budget=True)
    original = ref._allocation_lease.allocation
    table = ref.to_table()
    kept = table.column(0).chunk(0).slice(1)
    del table
    input_lease = refs.create_local_shm_input_lease([ref], reserve_output_credit=False)
    assert manager.snapshot()["allocated_bytes"] == ref.size
    refs.consume_local_shm_input_lease(input_lease)
    assert manager.snapshot()["allocated_bytes"] == 0
    refs.cancel_local_shm_input_lease(input_lease)
    ref.release()
    gc.collect()
    assert kept.to_pylist() == [22, 33]
    assert store.snapshot()["live_allocations"] == 1
    next_ref = _input(store, [44, 55, 66])
    try:
        assert next_ref._allocation_lease.allocation.offset != original.offset
        assert kept.to_pylist() == [22, 33]
    finally:
        next_ref.release()
    del kept
    gc.collect()
    assert store.snapshot()["live_allocations"] == 0


def test_batch_size_jitter_reuses_a_free_slot_with_other_batches_live():
    store = storage.LocalShmStore(3 * 64 * 1024)
    store.add_client("test")
    slots = [store.allocate(60000) for _ in range(3)]
    offset = slots[0].allocation.offset
    slots[0].release()
    try:
        for size in (60128, 59900, 60200, 60064):
            next_batch = store.allocate(size)
            try:
                assert next_batch.allocation.offset == offset
                assert store.snapshot()["live_allocations"] == 3
            finally:
                next_batch.release()
    finally:
        for slot in slots:
            slot.release()
        store.remove_client("test")
    assert store.snapshot()["mapped_capacity_bytes"] == 0


def test_full_store_refuses_overwrite_and_recovers_after_release(store):
    lease = store.allocate(store.capacity)
    pin = lease.fork()
    lease.release()
    with pytest.raises(storage.LocalShmStoreCapacityError, match="live=.*capacity="):
        store.allocate(1)
    pin.release()
    again = store.allocate(store.capacity)
    again.release()


@pytest.mark.parametrize("failure", ["mapping", "header", "ipc"])
def test_decode_failure_releases_temporary_storage_and_data_leases(store, monkeypatch, failure):
    from vane.execution.udf_data_lease import DataAllocation, RuntimeDataLedger

    ref = _input(store, [1, 2, 3])
    ledger = RuntimeDataLedger()
    query = ledger.open_query()
    task = query.open_task()
    ref.attach_data_lease(task.own_output(DataAllocation("local_shm", ref.name, ref.size)))
    if failure == "mapping":

        def fail_buffer(allocation):
            raise BufferError("planned mapping failure")

        monkeypatch.setattr(store, "buffer", fail_buffer)
    else:
        region = store.buffer(ref._allocation_lease.allocation)
        if failure == "header":
            region[:8] = (ref.size * 2).to_bytes(8, "little")
        else:
            region[8:] = bytes(ref.size - 8)
        region.release()
    try:
        # Keep the exception traceback alive while checking cleanup.
        with pytest.raises((BufferError, pa.ArrowException)) as caught:
            ref.to_table()
        assert caught.value is not None
        ref.release()
        if failure == "ipc":
            # PyArrow's exception frame retains the BufferReader and therefore
            # a real foreign buffer. Do not recycle its region until that last
            # owner dies, even though no table was returned.
            assert store.snapshot()["live_allocations"] == 1
            assert ledger.snapshot()["retained_bytes"] == ref.size
            del caught
            gc.collect()
        assert store.snapshot()["live_allocations"] == 0
        assert ledger.snapshot()["retained_bytes"] == 0
    finally:
        ref.release()
        task.finish()
        query.shutdown()
        ledger.close()


@pytest.mark.parametrize("cleanup", ["unlink", "close"])
def test_failed_last_lease_cleanup_retains_store_and_is_retryable(monkeypatch, cleanup):
    store = storage.LocalShmStore(4096)
    lease = store.allocate(64)
    shm = store._shm
    if cleanup == "unlink":
        target, name = refs, "_unlink_shm"
    else:
        target, name = shm, "close"
    original = getattr(target, name)

    def fail(*args, **kwargs):
        raise OSError("planned arena cleanup failure")

    monkeypatch.setattr(target, name, fail)
    try:
        with pytest.raises(OSError, match="planned arena cleanup failure"):
            lease.release()
        assert store._shm is shm
        assert not store._closed
        assert store.snapshot()["live_allocations"] == 0
    finally:
        monkeypatch.setattr(target, name, original)
        lease.release()
    assert store._closed
    assert store._shm is None


def test_close_with_retained_view_preserves_live_pages_and_finishes_on_last_view():
    store = storage.LocalShmStore(128 * 1024)
    store.add_client("test")
    unused = store.allocate(64 * 1024)
    unused.release()
    ref = _input(store, [3, 5, 8])
    kept = ref.to_table().column(0)
    ref.release()
    store.remove_client("test")
    assert kept.to_pylist() == [3, 5, 8]
    assert not store._closed
    del kept
    gc.collect()
    assert store._closed
    assert store._shm is None


def test_published_blocks_share_one_allocation_and_pin_it_independently(pooled_shm_worker):
    peer = pooled_shm_worker
    blocks = [refs.prepare_local_shm_block(pa.table({"x": [value]})) for value in (11, 22)]
    size = sum(block.ipc_size_bytes for block in blocks)
    grant = refs.request_local_shm_output_grant(size)
    allocation = peer.reserve_write(grant, size)
    descriptor = refs.make_pooled_shm_descriptor(blocks, allocation=allocation, grant_id=grant)
    result = refs.make_local_shm_ref_bundle_result_from_descriptor(descriptor)
    peer.finish_write(grant)
    first, second = result[1]
    kept = second.to_table().column(0)
    first.release()
    second.release()
    assert kept.to_pylist() == [22]
    assert peer.store.snapshot()["live_allocations"] == 1
    del kept
    gc.collect()
    assert peer.store.snapshot()["live_allocations"] == 0


@pytest.mark.parametrize("field,value", [("allocation_offset", 1), ("ipc_size_bytes", 1), ("shm_name", "foreign")])
def test_descriptor_cannot_alias_other_regions(pooled_shm_worker, field, value):
    peer = pooled_shm_worker
    block = refs.prepare_local_shm_block(pa.table({"x": [1]}))
    descriptor = refs.make_pooled_shm_descriptor(
        [block], allocation=peer.reserve_write(17, block.ipc_size_bytes), grant_id=17
    )
    descriptor["block_refs"][0][field] = value
    with pytest.raises(ValueError):
        refs.normalize_local_shm_ref_bundle_descriptor(descriptor)
    peer.finish_write(17)


_CHILD = """
import gc, json, sys
from vane.execution import ref_bundle as refs
from vane.execution.udf_shm_store import initialize_worker_shm_client
client = initialize_worker_shm_client(int(sys.argv[1]))
payload = json.loads(sys.stdin.readline())
client.begin_input(payload)
table = refs.materialize_ref_bundle(payload['block_refs'], payload['slices'], payload['metadata'], payload['names'])
kept = table.column(0)
del table
client.end_input()
print(json.dumps(kept.to_pylist()), flush=True)
for command in sys.stdin:
    if command.strip() == 'read':
        print(json.dumps(kept.to_pylist()), flush=True)
    elif command.strip() == 'release':
        kept = None
        gc.collect()
        print('released', flush=True)
    elif command.strip() == 'exit':
        break
client.close()
"""


def _line(proc):
    assert select.select([proc.stdout], [], [], 15)[0], "child lifetime probe timed out"
    line = proc.stdout.readline()
    assert line, f"child lifetime probe closed stdout (exit={proc.poll()})"
    return line.strip()


@pytest.mark.parametrize("crash", [False, True])
def test_remote_view_survives_parent_ref_release_and_pins_until_last_buffer_or_exit(monkeypatch, crash):
    store = storage.LocalShmStore(4096)

    def acquire(client_id):
        store.add_client(client_id)
        return store

    monkeypatch.setattr(storage, "current_store", acquire)
    peer = storage.ParentShmPeer()
    ref = _input(store, [11, 22, 33])
    allocation = ref._allocation_lease.allocation
    payload = refs.make_local_ref_bundle_worker_payload([ref], None, [{"num_rows": 3}], ["value"])
    peer.borrow_inputs(payload)
    fd = peer.child_sock.fileno()
    proc = subprocess.Popen(
        [sys.executable, "-I", "-c", _CHILD, str(fd)],
        pass_fds=(fd,),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    peer.child_sock.close()
    try:
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        assert json.loads(_line(proc)) == [11, 22, 33]
        ref.release()
        assert store.snapshot()["live_allocations"] == 1
        with pytest.raises(storage.LocalShmStoreCapacityError):
            store.allocate(store.capacity)
        if crash:
            proc.kill()
            proc.wait(timeout=10)
            # EOF alone does not recycle a live process's buffers. Its owner
            # explicitly confirms process death before releasing these pins.
            assert store.snapshot()["live_allocations"] == 1
        else:
            proc.stdin.write("read\n")
            proc.stdin.flush()
            assert json.loads(_line(proc)) == [11, 22, 33]
            proc.stdin.write("release\n")
            proc.stdin.flush()
            assert _line(proc) == "released"
            deadline = time.monotonic() + 5
            while store.snapshot()["live_allocations"] and time.monotonic() < deadline:
                time.sleep(0.01)
            assert store.snapshot()["live_allocations"] == 0
            reused = store.allocate(allocation.size)
            assert reused.allocation.offset == allocation.offset
            assert reused.allocation.generation != allocation.generation
            reused.release()
            proc.stdin.write("exit\n")
            proc.stdin.flush()
            assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        ref.release()
        peer.close_after_exit()
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            stream.close()
    assert store.snapshot()["live_allocations"] == 0
    assert store.snapshot()["mapped_capacity_bytes"] == 0

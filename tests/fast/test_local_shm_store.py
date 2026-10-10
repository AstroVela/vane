# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Allocation lifetime, safe reuse, and real child-process borrowed views."""

from __future__ import annotations

import gc
import json
import select
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

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
        block.write_to(region)
    except BaseException:
        lease.release()
        raise
    finally:
        region.release()
    return refs.LocalShmBlockRef(
        f"{allocation.identity}:0", allocation.size, budget_bytes=None if track_budget else 0, allocation_lease=lease
    )


@pytest.mark.parametrize("kind", ["sliced", "chunked", "dictionary", "nested", "tensor", "empty", "no_columns"])
def test_direct_ipc_matches_stream_size_and_preserves_all_buffers(monkeypatch, kind):
    if kind == "sliced":
        table = pa.table({"x": [1, None, 3], "s": ["one", "two", None]}).slice(1)
    elif kind == "chunked":
        table = pa.table({"x": pa.chunked_array([[1, None], [3, 4]]), "y": ["a", "b", "c", "d"]})
    elif kind == "dictionary":
        table = pa.table(
            {"x": pa.chunked_array([pa.array(["a", "b"]).dictionary_encode(), pa.array(["c"]).dictionary_encode()])}
        )
    elif kind == "nested":
        table = pa.table({"x": [[{"value": "a"}], None, [{"value": None}, {"value": "b"}]]})
    elif kind == "tensor":
        import numpy as np

        table = pa.table(
            {"x": pa.FixedShapeTensorArray.from_numpy_ndarray(np.arange(256, dtype=np.float32).reshape(4, 8, 8))}
        )
    elif kind == "empty":
        table = pa.table({"x": pa.array([], type=pa.list_(pa.int64()))})
    else:
        table = pa.table({})
    table = table.replace_schema_metadata({b"description": b"direct IPC regression"})
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    expected = sink.getvalue().to_pybytes()

    def unexpected_buffer(*args, **kwargs):
        pytest.fail("output preparation allocated an intermediate IPC buffer")

    monkeypatch.setattr(pa, "BufferOutputStream", unexpected_buffer)
    block = refs.prepare_local_shm_block(table)
    assert block.ipc_size_bytes == len(expected) + 8
    backing = bytearray(b"prefix!!" + b"\x00" * block.ipc_size_bytes + b"suffix!!")
    with memoryview(backing)[8:-8] as region:
        block.write_to(region)
        assert int.from_bytes(region[:8], "little") == len(expected)
        assert bytes(region[8:]) == expected
        actual = pa.ipc.open_stream(region[8:]).read_all()
        assert actual.equals(table, check_metadata=True)
        del actual
    assert backing[:8] == b"prefix!!" and backing[-8:] == b"suffix!!"


@pytest.mark.parametrize("size_delta", [-8, 8])
def test_direct_ipc_size_failure_does_not_leave_an_exported_mapping(monkeypatch, size_delta):
    block = refs.prepare_local_shm_block(pa.table({"x": [11, 22, 33]}))
    block = replace(block, ipc_size_bytes=block.ipc_size_bytes + size_delta)
    created = []
    create = refs._create_shm

    def capture(*args, **kwargs):
        shm = create(*args, **kwargs)
        created.append(shm)
        return shm

    monkeypatch.setattr(refs, "_create_shm", capture)
    with pytest.raises((pa.ArrowException, OSError, ValueError)) as error:
        refs.make_local_shm_descriptor_from_block(block)
    assert not isinstance(error.value, BufferError)
    assert len(created) == 1
    # Keep the exception alive: its traceback must not pin the writer's buffer.
    assert created[0]._buf is None
    assert not (Path("/dev/shm") / created[0].name).exists()


def test_query_arena_pin_survives_last_worker_and_retries_failed_cleanup(monkeypatch):
    monkeypatch.setenv("VANE_LOCAL_SHM_STORE_BYTES", "64KiB")
    query = storage.LocalQueryShmStore()
    worker = storage.current_store("query-owner-test")
    assert worker is query.store
    worker.remove_client("query-owner-test")
    assert not worker._closed
    remove = worker.remove_client

    def fail_once(client_id):
        monkeypatch.setattr(worker, "remove_client", remove)
        raise RuntimeError("injected arena cleanup failure")

    monkeypatch.setattr(worker, "remove_client", fail_once)
    try:
        with pytest.raises(RuntimeError, match="arena cleanup failure"):
            query.shutdown()
        assert query.cleanup_pending()
    finally:
        query.shutdown()
    assert not query.cleanup_pending()
    assert worker._closed


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


def test_retained_arrow_slice_keeps_storage_and_transport_bytes_charged(store, monkeypatch):
    manager = refs.LocalShmBudgetManager(limit_factory=lambda: store.capacity)
    monkeypatch.setattr(refs, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    ref = _input(store, [11, 22, 33], track_budget=True)
    original = ref._allocation_lease.allocation
    table = ref.to_table()
    kept = table.column(0).chunk(0).slice(1)
    del table
    input_lease = refs.create_local_shm_input_lease([ref])
    assert manager.snapshot()["allocated_bytes"] == ref.size
    refs.consume_local_shm_input_lease(input_lease)
    assert manager.snapshot()["allocated_bytes"] == ref.size
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
    assert manager.snapshot()["allocated_bytes"] == 0


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


@pytest.mark.parametrize("retire_before_registration", [False, True])
def test_worker_registration_racing_with_last_worker_retirement(monkeypatch, retire_before_registration):
    monkeypatch.setenv("VANE_LOCAL_SHM_STORE_BYTES", "4096")
    monkeypatch.setattr(storage, "_stores", {})
    monkeypatch.setattr(storage, "_current_store", None)
    previous = storage.current_store("old-worker")
    registering = threading.Event()
    proceed = threading.Event()
    original = previous.add_client

    def pause_registration(client_id):
        registering.set()
        assert proceed.wait(5)
        return original(client_id)

    monkeypatch.setattr(previous, "add_client", pause_registration)
    selected = None
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(storage.current_store, "new-worker")
            try:
                assert registering.wait(5)
                if retire_before_registration:
                    previous.remove_client("old-worker")
            finally:
                proceed.set()
            selected = future.result(timeout=5)
        if not retire_before_registration:
            previous.remove_client("old-worker")
        assert (selected is previous) == (not retire_before_registration)
        assert selected.snapshot()["clients"] == 1
        ref = _input(selected, [1, 2, 3])
        try:
            assert ref.to_table().column(0).to_pylist() == [1, 2, 3]
        finally:
            ref.release()
    finally:
        previous.remove_client("old-worker")
        if selected is not None:
            selected.remove_client("new-worker")


_NORMAL_EXIT = """
import atexit, gc, json, sys

def check_late_view():
    from pathlib import Path
    values = None if kept is None else (kept.to_pylist() if hasattr(kept, 'to_pylist') else kept.tolist())
    print(json.dumps({'late_values': values, 'unlinked': not (Path('/dev/shm') / name).exists()}), flush=True)

# Run after the store's exit callback to verify that unlink leaves views valid.
atexit.register(check_late_view)
import pyarrow as pa
from vane.execution import ref_bundle as refs
from vane.execution.udf_shm_store import LocalShmStore
pooled, drop_view = sys.argv[1] == 'True', sys.argv[2] == 'True'
table = pa.table({'x': [1, 2, 3]})
if pooled:
    store = LocalShmStore(4096)
    store.add_client('runtime')
    block = refs.prepare_local_shm_block(table)
    lease = store.allocate(block.ipc_size_bytes)
    allocation = lease.allocation
    region = store.buffer(allocation)
    block.write_to(region)
    region.release()
    ref = refs.LocalShmBlockRef(f'{allocation.identity}:0', allocation.size, budget_bytes=0, allocation_lease=lease)
    name = allocation.shm_name
else:
    result = refs.make_local_shm_ref_bundle_result(table)
    ref = result[1][0]
    name = ref.name
print(json.dumps({'name': name}), flush=True)
kept = ref.to_table().column(0)
if sys.argv[3] == 'numpy':
    kept = kept.chunk(0).to_numpy(zero_copy_only=True)
ref.release()
if pooled:
    store.remove_client('runtime')
if drop_view:
    kept = None
    gc.collect()
"""


@pytest.mark.parametrize("pooled,drop_view", [(False, False), (True, True), (True, False)])
@pytest.mark.parametrize("view_kind", ["arrow", "numpy"])
def test_normal_exit_unlinks_arena_without_invalidating_retained_views(pooled, drop_view, view_kind):
    child = subprocess.run(
        [sys.executable, "-I", "-c", _NORMAL_EXIT, str(pooled), str(drop_view), view_kind],
        text=True,
        capture_output=True,
        timeout=15,
    )
    records = [json.loads(line) for line in child.stdout.splitlines()]
    name = records[0]["name"] if records else None
    try:
        assert child.returncode == 0, child.stderr
        assert name is not None
        assert not (Path("/dev/shm") / name).exists(), f"normal exit leaked arena {name}"
        assert records[-1] == {"late_values": None if drop_view else [1, 2, 3], "unlinked": True}
        assert "Exception ignored" not in child.stderr
    finally:
        if name is not None:
            refs._unlink_shared_memory_name(name)


def test_exit_cleanup_preserves_views_and_attempts_every_arena_after_failure(monkeypatch):
    monkeypatch.setattr(storage, "_stores", {})
    stores = [storage.LocalShmStore(4096) for _ in range(2)]
    refs_to_release = [_input(store, [index]) for index, store in enumerate(stores)]
    kept = [ref.to_table().column(0) for ref in refs_to_release]
    for ref in refs_to_release:
        ref.release()
    original = refs._unlink_shm

    def fail_first(shm, *, track):
        if shm is stores[0]._shm:
            raise OSError("planned exit unlink failure")
        original(shm, track=track)

    monkeypatch.setattr(refs, "_unlink_shm", fail_first)
    try:
        with pytest.raises(OSError, match="planned exit unlink failure"):
            storage._unlink_owned_stores_at_exit()
        assert not stores[0]._unlinked
        assert stores[1]._unlinked
        assert [view.to_pylist() for view in kept] == [[0], [1]]
        monkeypatch.setattr(refs, "_unlink_shm", original)
        storage._unlink_owned_stores_at_exit()
        storage._unlink_owned_stores_at_exit()
        assert [view.to_pylist() for view in kept] == [[0], [1]]
    finally:
        monkeypatch.setattr(refs, "_unlink_shm", original)
        kept.clear()
        gc.collect()
        for store in stores:
            store.close()


_FORK_FINALIZERS = """
import atexit, gc, json, os, signal, sys, threading
from pathlib import Path
fork_child = False
original_refs = []

def finish_child_exit():
    if fork_child:
        # Registered before Vane imports, so all Python exit callbacks,
        # including block-ref weakref finalizers, must have run first. Avoid
        # unrelated native-library teardown in a fork of a multithreaded host.
        pending = any(ref._finalizer.alive for ref in original_refs)
        os._exit(7 if pending else 0)

atexit.register(finish_child_exit)
import pyarrow as pa
from vane.execution import udf_shm_store as storage
from vane.execution import ref_bundle as refs

scenario, held_lock = sys.argv[1:]
store = storage.LocalShmStore(32768)
store.add_client('old-runtime')

def make_ref(values):
    block = refs.prepare_local_shm_block(pa.table({'x': values}))
    lease = store.allocate(block.ipc_size_bytes)
    allocation = lease.allocation
    region = store.buffer(allocation)
    block.write_to(region)
    region.release()
    return refs.LocalShmBlockRef(f'{allocation.identity}:0', allocation.size,
                                budget_bytes=0, allocation_lease=lease)

original_refs = [make_ref([11, 22, 33])]
reuse = scenario in ('reuse', 'arrow', 'numpy')
if reuse:
    original_refs.append(make_ref([44]))
lease = original_refs[0]._allocation_lease
name = lease.allocation.shm_name
print(json.dumps({'name': name}), flush=True)
kept = None
if scenario in ('arrow', 'numpy'):
    kept = original_refs[0].to_table().column(0)
    if scenario == 'numpy':
        kept = kept.chunk(0).to_numpy(zero_copy_only=True)
    original_refs[0].release()
store.remove_client('old-runtime')

def drop_child_view():
    global kept
    kept = None
    gc.collect()

def explicit_child_cleanup():
    store.release(lease.allocation)
    store.drain_if_idle()
    store.remove_client('old-runtime')
    store.close()

locked, proceed = threading.Event(), threading.Event()
locks = {'lease': lease._lock, 'store': store._lock, 'registry': storage._registry_lock}
def hold_lock():
    with locks[held_lock]:
        locked.set()
        assert proceed.wait(10)

thread = None
if held_lock != 'none':
    thread = threading.Thread(target=hold_lock)
    thread.start()
    assert locked.wait(5)
read_fd, write_fd = os.pipe()
pid = os.fork()
if pid == 0:
    fork_child = True
    signal.alarm(5)
    os.close(write_fd)
    assert os.read(read_fd, 1) == b'1'
    os.close(read_fd)
    if scenario in ('arrow', 'numpy'):
        atexit.register(drop_child_view)
    if scenario == 'explicit':
        atexit.register(explicit_child_cleanup)
    # Exercise normal Python exit, not a direct os._exit that skips finalizers.
    sys.exit(0)

os.close(read_fd)
proceed.set()
if thread is not None:
    thread.join(5)
later = None
if reuse:
    store.add_client('new-runtime')
    later = make_ref([17] * 2048)
    assert later.to_table().column(0).to_pylist() == [17] * 2048
os.write(write_fd, b'1')
os.close(write_fd)
try:
    _, status = os.waitpid(pid, 0)
    target = later if later is not None else original_refs[0]
    expected = [17] * 2048 if reuse else [11, 22, 33]
    actual = target.to_table().column(0).to_pylist()
    print(json.dumps({'child_exit': os.waitstatus_to_exitcode(status),
                      'parent_arena_exists': (Path('/dev/shm') / name).exists(),
                      'values_intact': actual == expected,
                      'zeroed_live_rows': actual.count(0)}), flush=True)
finally:
    kept = None
    gc.collect()
    if later is not None:
        later.release()
    for ref in original_refs:
        ref.release()
    if reuse:
        store.remove_client('new-runtime')
    store.close()
"""


@pytest.mark.parametrize(
    "scenario,held_lock",
    [
        ("arena", "none"),
        ("arena", "lease"),
        ("arena", "store"),
        ("arena", "registry"),
        ("reuse", "none"),
        ("arrow", "none"),
        ("numpy", "none"),
        ("explicit", "store"),
    ],
)
def test_fork_exit_finalizers_preserve_parent_data_and_skip_inherited_locks(scenario, held_lock):
    child = subprocess.run(
        [sys.executable, "-I", "-c", _FORK_FINALIZERS, scenario, held_lock],
        text=True,
        capture_output=True,
        timeout=20,
    )
    records = [json.loads(line) for line in child.stdout.splitlines()]
    name = records[0]["name"] if records else None
    try:
        assert child.returncode == 0, child.stderr
        assert name is not None
        assert records[-1] == {
            "child_exit": 0,
            "parent_arena_exists": True,
            "values_intact": True,
            "zeroed_live_rows": 0,
        }, child.stderr
        assert "Exception ignored" not in child.stderr
        assert not (Path("/dev/shm") / name).exists()
    finally:
        if name is not None:
            refs._unlink_shared_memory_name(name)


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
if sys.argv[2] == 'numpy':
    kept = kept.chunk(0).to_numpy(zero_copy_only=True)
def values():
    return kept.to_pylist() if hasattr(kept, 'to_pylist') else kept.tolist()
del table
client.end_input()
print(json.dumps(values()), flush=True)
for command in sys.stdin:
    if command.strip() == 'read':
        print(json.dumps(values()), flush=True)
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
@pytest.mark.parametrize("view_kind", ["arrow", "numpy"])
def test_remote_view_survives_parent_ref_release_and_pins_until_last_buffer_or_exit(monkeypatch, crash, view_kind):
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
        [sys.executable, "-I", "-c", _CHILD, str(fd), view_kind],
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

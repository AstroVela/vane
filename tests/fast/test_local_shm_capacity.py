# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Budget admission, early mapping failure and platform headroom contracts."""

from __future__ import annotations

import ctypes
import gc
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vane.execution import local_shm_capacity as capacity
from vane.execution import ref_bundle
from vane.execution import udf_shm_store as storage

GIB = 1024**3
MIB = 1024**2


@pytest.fixture
def arena(monkeypatch):
    monkeypatch.setattr(storage, "_current_store", None)
    monkeypatch.setattr(storage, "_stores", {})
    monkeypatch.setattr(storage, "_capacity_bytes", None)
    monkeypatch.setattr(storage, "_capacity_pending", None)
    monkeypatch.setattr(capacity, "memory_headroom", lambda: {"physical_available": 10 * GIB})
    monkeypatch.setattr(capacity.os, "statvfs", lambda _: SimpleNamespace(f_frsize=1, f_bavail=8 * GIB), raising=False)
    monkeypatch.delenv("VANE_LOCAL_SHM_STORE_BYTES", raising=False)
    monkeypatch.delenv("VANE_LOCAL_SHM_REF_BUDGET_BYTES", raising=False)
    yield
    for store in storage._stores.values():
        assert not store._live
        assert not store._clients
        store.close()


@pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
def test_auto_uses_tightest_headroom_and_leaves_heap_room(monkeypatch, platform):
    monkeypatch.setattr(capacity.sys, "platform", platform)
    monkeypatch.setattr(
        capacity,
        "memory_headroom",
        lambda: {
            "physical_available": 20 * GIB,
            "commit_available": 2 * GIB,
            "application_available": GIB,
        },
    )
    monkeypatch.setattr(capacity.os, "statvfs", lambda _: SimpleNamespace(f_frsize=1, f_bavail=8 * GIB), raising=False)
    decision = capacity.resolve_capacity()
    assert decision.capacity == GIB * 3 // 10
    assert decision.automatic


def test_linux_shm_space_is_independent_from_memory_budget(monkeypatch):
    monkeypatch.setattr(capacity.sys, "platform", "linux")
    monkeypatch.setattr(capacity, "memory_headroom", lambda: {"physical_available": 10 * GIB})
    monkeypatch.setattr(
        capacity.os, "statvfs", lambda _: SimpleNamespace(f_frsize=1, f_bavail=256 * MIB), raising=False
    )
    assert capacity.resolve_capacity().capacity == 256 * MIB * 95 // 100


@pytest.mark.parametrize("field", ["physical_available", "commit_available", "cgroup2", "application_available"])
def test_exhausted_constraint_fails_instead_of_ignoring_it(monkeypatch, field):
    monkeypatch.setattr(capacity, "memory_headroom", lambda: {"physical_available": 10 * GIB, field: 0})
    with pytest.raises(MemoryError, match="no available memory"):
        capacity.resolve_capacity()


def test_explicit_capacity_is_exact_or_rejected(monkeypatch):
    monkeypatch.setattr(capacity.os, "statvfs", lambda _: SimpleNamespace(f_frsize=1, f_bavail=8 * GIB), raising=False)
    monkeypatch.setattr(capacity, "memory_headroom", lambda: {"physical_available": 2 * GIB, "commit_available": GIB})
    decision = capacity.resolve_capacity(800 * MIB)
    assert decision.capacity == 800 * MIB and not decision.automatic
    with pytest.raises(MemoryError, match="requested_bytes"):
        capacity.resolve_capacity(2 * GIB)


def test_windows_probe_reads_commit_and_virtual_headroom(monkeypatch):
    def query(pointer):
        assert pointer._obj.length == 64
        pointer._obj.commit_available = 123456
        pointer._obj.virtual_available = 987654
        return 1

    monkeypatch.setattr(
        ctypes, "WinDLL", lambda *args, **kwargs: SimpleNamespace(GlobalMemoryStatusEx=query), raising=False
    )
    assert capacity._windows_headroom() == {"commit_available": 123456, "virtual_available": 987654}


def test_windows_budget_includes_process_usage_and_physical_reserve(monkeypatch):
    monkeypatch.setattr(capacity.sys, "platform", "win32")
    monkeypatch.setattr(capacity.psutil, "virtual_memory", lambda: SimpleNamespace(available=10 * GIB))
    monkeypatch.setattr(
        capacity.psutil, "Process", lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(private=GIB))
    )
    monkeypatch.setattr(capacity, "_windows_headroom", lambda: {"commit_available": 3 * GIB})
    monkeypatch.setenv(capacity.MEMORY_LIMIT_ENV, str(4 * GIB))
    monkeypatch.setenv(capacity.MEMORY_RESERVE_ENV, str(4 * GIB))
    assert capacity.memory_headroom() == {
        "physical_available": 6 * GIB,
        "application_available": 3 * GIB,
        "commit_available": 3 * GIB,
    }


@pytest.mark.parametrize("raw", ["-1", "nan", "", "1GiB", "1.5", "１"])
def test_invalid_application_budget_is_not_silently_unbounded(monkeypatch, raw):
    monkeypatch.setenv(capacity.MEMORY_LIMIT_ENV, raw)
    with pytest.raises(ValueError, match=capacity.MEMORY_LIMIT_ENV):
        capacity.memory_headroom()


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("namespaced", [False, True])
def test_cgroup_reads_current_and_ancestor_headroom(tmp_path, version, namespaced):
    proc = tmp_path / "proc"
    proc.mkdir()
    mount = tmp_path / "memory groups"
    child = mount / "nested"
    child.mkdir(parents=True)
    limit_name, used_name = (
        ("memory.max", "memory.current") if version == 2 else ("memory.limit_in_bytes", "memory.usage_in_bytes")
    )
    for directory, limit, used in [(mount, 1000, 900), (child, 600, 200)]:
        (directory / limit_name).write_text(str(limit))
        (directory / used_name).write_text(str(used))
    group = "/" if namespaced else "/nested"
    (proc / "cgroup").write_text(f"0::{group}\n" if version == 2 else f"4:memory:{group}\n")
    mount_root = "/"
    filesystem = "cgroup2 cgroup rw" if version == 2 else "cgroup memory rw,memory"
    mount_text = mount.as_posix().replace(" ", r"\040")
    (proc / "mountinfo").write_text(f"1 0 0:1 {mount_root} {mount_text} rw - {filesystem}\n")
    assert capacity._cgroup_headroom(proc) == {("cgroup2" if version == 2 else "cgroup"): 100}


def test_pool_and_transport_share_one_capacity_decision(arena, monkeypatch):
    monkeypatch.setenv("VANE_LOCAL_SHM_STORE_BYTES", "64MiB")
    factory = ref_bundle._make_local_shm_ref_budget_limit_factory()
    assert factory() == 32 * MIB
    query = storage.LocalQueryShmStore()
    try:
        assert query.store.capacity == 64 * MIB
        monkeypatch.setattr(capacity, "memory_headroom", lambda: {"physical_available": MIB})
        assert factory() == 32 * MIB
        assert storage.current_store_capacity() == 64 * MIB
    finally:
        query.shutdown()


def test_next_arena_recomputes_transport_budget(arena, monkeypatch):
    monkeypatch.setenv("VANE_LOCAL_SHM_STORE_BYTES", "64MiB")
    factory = ref_bundle._make_local_shm_ref_budget_limit_factory()
    query = storage.LocalQueryShmStore()
    assert factory() == 32 * MIB
    query.shutdown()
    monkeypatch.setenv("VANE_LOCAL_SHM_STORE_BYTES", "32MiB")
    query = storage.LocalQueryShmStore()
    try:
        assert factory() == 16 * MIB
    finally:
        query.shutdown()


@pytest.mark.parametrize("backend", ["subprocess_actor", "subprocess_task"])
def test_mapping_failure_precedes_worker_start_and_releases_owner(arena, monkeypatch, caplog, backend):
    from vane.execution import udf_subprocess

    monkeypatch.setenv("VANE_LOCAL_SHM_STORE_BYTES", "1MiB")
    error = OSError(22, "private payload must not be logged")
    error.winerror = 1450
    calls = []

    def fail(size, **kwargs):
        calls.append(size)
        raise error

    def actor(*args, **kwargs):
        pytest.fail("An actor started before the arena was allocated")

    monkeypatch.setattr(ref_bundle, "_create_shm", fail)
    monkeypatch.setattr(udf_subprocess, "LocalSubprocessActorPool", actor)
    monkeypatch.setattr(subprocess, "Popen", actor)
    with pytest.raises(MemoryError, match="requested_bytes=1048576") as observed:
        udf_subprocess.ensure_local_subprocess_actor_pools_for_nodes(
            [{"node_id": "0", "payload": {"execution_backend": backend, "actor_number": 1}}],
            resource_graph=None,
        )
    assert calls == [MIB]
    assert observed.value.__cause__ is error
    assert "winerror=1450" in caplog.text
    assert "private payload" not in caplog.text
    assert storage._current_store._closed
    assert storage.local_shm_store_snapshot()["clients"] == 0
    assert storage.local_shm_store_snapshot()["mapped_capacity_bytes"] == 0


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_posix_address_space_and_resident_budget(monkeypatch, platform):
    monkeypatch.setattr(capacity.sys, "platform", platform)
    monkeypatch.setitem(
        sys.modules, "resource", SimpleNamespace(RLIMIT_AS=9, RLIM_INFINITY=-1, getrlimit=lambda _: (5 * GIB, 5 * GIB))
    )
    monkeypatch.setattr(capacity.psutil, "virtual_memory", lambda: SimpleNamespace(available=10 * GIB))
    monkeypatch.setattr(
        capacity.psutil, "Process", lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(vms=4 * GIB, rss=GIB))
    )
    monkeypatch.setattr(capacity, "_cgroup_headroom", lambda: {"cgroup2": 2 * GIB})
    monkeypatch.setenv(capacity.MEMORY_LIMIT_ENV, str(4 * GIB))
    monkeypatch.setenv(capacity.MEMORY_RESERVE_ENV, str(GIB))
    result = capacity.memory_headroom()
    assert result["address_space_available"] == GIB
    assert result["application_available"] == 3 * GIB
    assert result["physical_available"] == 9 * GIB
    assert ("cgroup2" in result) == (platform == "linux")


@pytest.mark.parametrize("mount_root", ["/unrelated", "/.."])
def test_cgroup_does_not_guess_namespace_or_include_unrelated_bind(tmp_path, mount_root):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "cgroup").write_text("0::/\n")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "memory.max").write_text("1")
    (unrelated / "memory.current").write_text("1")
    ambiguous = f"1 0 0:1 {mount_root} {unrelated.as_posix()} rw - cgroup2 cgroup rw\n"
    (proc / "mountinfo").write_text(ambiguous)
    with pytest.raises(RuntimeError, match="namespace-relative"):
        capacity._cgroup_headroom(proc)
    own = tmp_path / "own"
    own.mkdir()
    (own / "memory.max").write_text("1000")
    (own / "memory.current").write_text("100")
    (proc / "mountinfo").write_text(ambiguous + f"2 0 0:1 / {own.as_posix()} rw - cgroup2 cgroup rw\n")
    assert capacity._cgroup_headroom(proc) == {"cgroup2": 900}


def test_live_queries_and_retained_arrow_view_keep_auto_capacity(arena, monkeypatch):
    import pyarrow as pa

    monkeypatch.setattr(capacity, "memory_headroom", lambda: {"physical_available": MIB})
    first, second = storage.LocalQueryShmStore(), storage.LocalQueryShmStore()
    store = first.store
    assert second.store is store
    _, refs, metadata, names = ref_bundle.make_local_shm_ref_bundle_result(pa.table({"value": [1, 2, 3]}))
    table = ref_bundle.materialize_ref_bundle(refs, None, metadata, names)
    column = table.column("value")
    del table
    refs[0].release()
    first.shutdown()
    second.shutdown()
    monkeypatch.setattr(capacity, "memory_headroom", lambda: {"physical_available": MIB // 2})
    third = storage.LocalQueryShmStore()
    assert third.store is store
    assert column.to_pylist() == [1, 2, 3]
    assert third.store.capacity == MIB * 3 // 10
    del column
    gc.collect()
    third.shutdown()
    assert store._closed
    fourth = storage.LocalQueryShmStore()
    try:
        assert fourth.store.capacity == (MIB // 2) * 3 // 10
    finally:
        fourth.shutdown()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="real POSIX fork-pin reclamation")
def test_budget_snapshot_and_reclaimed_fork_charge_cannot_deadlock():
    script = r"""
import os, threading
import pyarrow as pa
from vane.execution import ref_bundle as refs, udf_shm_store as storage
os.environ['VANE_LOCAL_SHM_STORE_BYTES']='1MiB'
os.environ.pop('VANE_LOCAL_SHM_REF_BUDGET_BYTES',None)
refs._LOCAL_SHM_BUDGET_MANAGER=refs.LocalShmBudgetManager()
manager=refs._LOCAL_SHM_BUDGET_MANAGER
storage._start_fork_watcher=lambda hold: None
_, blocks, _, _=refs.make_local_shm_ref_bundle_result(pa.table({'x':[1,2,3]}))
lease=blocks[0]._allocation_lease
released=threading.Event()
lease.store.retain_budget(lease.allocation, released.set)
read, write=os.pipe()
child=os.fork()
if child==0:
    os.close(write)
    assert os.read(read,1)==b'x'
    os._exit(0)
os.close(read)
blocks[0].release()
assert not released.is_set()
os.write(write,b'x')
os.close(write)
assert os.waitpid(child,0)[1]==0
assert not released.is_set(), 'snapshot must be the sole fork-pin reclaimer'
budget_locked, registry_locked=threading.Event(),threading.Event()
errors=[]
def budgeting():
    try:
        with manager._cond:
            budget_locked.set()
            assert registry_locked.wait(3)
            manager.snapshot()
    except BaseException as error: errors.append(error)
def reclaiming():
    try:
        assert budget_locked.wait(3)
        with storage._registry_lock:
            registry_locked.set()
            storage.local_shm_store_snapshot()
    except BaseException as error: errors.append(error)
threads=[threading.Thread(target=f,daemon=True) for f in (budgeting,reclaiming)]
for thread in threads: thread.start()
for thread in threads: thread.join(4)
assert not any(thread.is_alive() for thread in threads), 'budget/registry deadlock'
assert not errors, errors
assert released.is_set()
assert manager.snapshot()['usage_bytes']==0
"""
    result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_capacity_probe_defers_raw_ref_cycle_gc_and_restores_gc_state():
    script = r"""
import gc, os, weakref
from vane.execution import ref_bundle as refs, udf_shm_store as storage
os.environ['VANE_LOCAL_SHM_STORE_BYTES']='1MiB'
os.environ.pop('VANE_LOCAL_SHM_REF_BUDGET_BYTES',None)
refs._LOCAL_SHM_BUDGET_MANAGER=refs.LocalShmBudgetManager()
manager=refs._LOCAL_SHM_BUDGET_MANAGER
shm=refs._create_shm(128,track=False)
raw=refs.LocalShmBlockRef(shm.name,128,shm=shm)
reference=weakref.ref(raw)
cycle=[raw]
cycle.append(cycle)
holder=[cycle]
del raw, cycle
query=storage.LocalQueryShmStore()
query.shutdown()
assert manager.snapshot()['usage_bytes']>0
os.environ['VANE_LOCAL_SHM_STORE_BYTES']='2MiB'
resolve=storage._resolve_capacity_decision
def probe():
    assert not gc.isenabled(), 'capacity decision must defer automatic finalizers'
    holder.clear()
    temporary=[]
    for _ in range(1000): temporary.append([])
    assert reference() is not None
    return resolve()
storage._resolve_capacity_decision=probe
old_threshold=gc.get_threshold()
gc.enable()
gc.set_threshold(1,1,1)
try:
    query=storage.LocalQueryShmStore()
    assert gc.isenabled()
    gc.collect()
    assert reference() is None
    assert manager.snapshot()['usage_bytes']==0
    query.shutdown()
finally:
    gc.set_threshold(*old_threshold)
"""
    result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_command_guard_exports_stricter_nested_budgets():
    script = Path(__file__).resolve().parents[2] / "scripts/run_with_memory_guard.py"
    spec = importlib.util.spec_from_file_location("shm_memory_guard", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = {capacity.MEMORY_LIMIT_ENV: str(2 * GIB), capacity.MEMORY_RESERVE_ENV: str(5 * GIB), "KEEP": "yes"}
    result = module.local_memory_environment(original, 4 * GIB, 4 * GIB)
    assert result == original
    result = module.local_memory_environment({}, 4 * GIB, GIB)
    assert result == {capacity.MEMORY_LIMIT_ENV: str(4 * GIB), capacity.MEMORY_RESERVE_ENV: str(GIB)}

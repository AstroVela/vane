# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Storage quota and locks remain owned until actual I/O cleanup succeeds."""

import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

from vane import IOException
from vane._native import execution_runtime as native
from vane.execution.fte_store import ActiveStoreLease, ExchangeStore, StorePool, replace_metadata


def pool(tmp_path, **changes):
    config = ExchangeStore(
        "shared", str(tmp_path / "shared"), capacity_bytes=4096, query_bytes=2048, source_bytes=1024, object_bytes=1024
    )
    return StorePool(replace(config, **changes))


def test_store_guards_are_owned_by_each_open_description(tmp_path):
    path = str(tmp_path / "guard")
    first = native.StoreGuard.acquire(path, False, True)
    second = native.StoreGuard.acquire(path, False, False)
    assert first is not None and second is not None
    try:
        assert native.StoreGuard.acquire(path, True, False) is None
        first.close()
        assert native.StoreGuard.acquire(path, True, False) is None
        second.close()
        exclusive = native.StoreGuard.acquire(path, True, False)
        assert exclusive is not None
        assert native.StoreGuard.acquire(path, False, False) is None
        exclusive.close()
    finally:
        first.close()
        second.close()


def test_sessions_share_the_persisted_storage_capacity(tmp_path):
    first, second = pool(tmp_path), pool(tmp_path)
    a, b = first.reserve("a"), second.reserve("b")
    try:
        assert first.snapshot() == {"queries": 2, "reserved_bytes": 4096}
        with pytest.raises(RuntimeError, match="capacity"):
            second.reserve("c")
        a.close(lambda: None)
        c = first.reserve("c")
        c.close(lambda: None)
        assert second.snapshot()["reserved_bytes"] == 2048
    finally:
        a.close(lambda: None)
        b.close(lambda: None)
    assert first.snapshot()["reserved_bytes"] == 0


def test_registration_cannot_change_existing_store_capacity(tmp_path):
    original = pool(tmp_path)
    with pytest.raises(ValueError, match="conflicting"):
        StorePool(replace(original.config, capacity_bytes=8192))


def test_live_native_pin_blocks_query_cleanup(tmp_path):
    store = pool(tmp_path)
    lease = store.reserve("query")
    worker = ActiveStoreLease(lease.to_dict())
    try:
        with pytest.raises(RuntimeError, match="participants"):
            lease.close(lambda: None)
        assert store.snapshot()["reserved_bytes"] == 2048
        worker.close()
        lease.close(lambda: None)
        assert store.snapshot()["reserved_bytes"] == 0
    finally:
        worker.close()
        lease.close(lambda: None)


def test_expired_orphan_is_collected_only_after_all_process_pins_end(tmp_path):
    store = pool(tmp_path, lease_seconds=0.1)
    lease = store.reserve("orphan")
    worker = ActiveStoreLease(lease.to_dict())
    (lease.directory / "private.mat").write_bytes(b"in flight")
    lease.guard.close()  # Coordinator process exits; no release/cleanup RPC.
    time.sleep(0.15)
    try:
        store.collect_expired()
        assert (lease.directory / "private.mat").exists()
        assert store.snapshot()["reserved_bytes"] == 2048
        with pytest.raises(RuntimeError, match="expired"):
            worker.check()
        with pytest.raises(RuntimeError, match="expired"):
            lease.renew()
        worker.close()
        store.collect_expired()
        assert not lease.directory.exists()
        assert store.snapshot()["reserved_bytes"] == 0
    finally:
        worker.close()
        lease.close(lambda: None)


def test_cleanup_failure_retains_external_allocation_record(tmp_path):
    store = pool(tmp_path)
    lease = store.reserve("query")

    # Query data can be removed before cleanup fails. Both the lease lock and
    # allocation live outside that directory and must survive the deletion.
    (lease.directory / "object.mat").write_bytes(b"partial")

    def fail_after_partial_delete():
        (lease.directory / "object.mat").unlink()
        assert lease.lock_path.exists()
        raise OSError("storage unavailable")

    with pytest.raises(OSError, match="storage unavailable"):
        lease.close(fail_after_partial_delete)
    assert store.snapshot()["reserved_bytes"] == 2048
    lease.renew()  # Closing cannot revive a lease for new native participants.
    with pytest.raises(RuntimeError, match="being cleaned"):
        ActiveStoreLease(lease.to_dict())
    lease.close(lambda: None)
    assert store.snapshot()["reserved_bytes"] == 0


def test_lease_renewal_keeps_a_live_query_from_expiring(tmp_path):
    store = pool(tmp_path, lease_seconds=0.2)
    lease = store.reserve("query")
    worker = ActiveStoreLease(lease.to_dict())
    try:
        original = worker.check()["expires"]
        lease.renew()
        assert worker.check()["expires"] > original
        store.collect_expired()
        assert lease.directory.exists()
    finally:
        worker.close()
        lease.close(lambda: None)


def test_replaced_query_directory_is_rejected_before_creating_an_allocation(tmp_path):
    store = pool(tmp_path)
    queries = store.root / "queries"
    queries.rmdir()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    try:
        queries.symlink_to(foreign, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    with pytest.raises(ValueError, match="symlink"):
        store.reserve("query")
    assert list(foreign.iterdir()) == []
    assert store.snapshot()["queries"] == 0


@pytest.mark.parametrize("failure", ["lock", "partial_delete", "allocation_delete"])
def test_orphan_cleanup_failure_is_isolated_charged_and_retried(tmp_path, monkeypatch, failure):
    store = pool(tmp_path, capacity_bytes=3 * 2048)
    blocked, removable = store.reserve("blocked"), store.reserve("removable")
    blocked_record = store.allocations / f"{blocked.value['namespace']}.json"
    (blocked.directory / "object.mat").write_bytes(b"partial")
    for lease in (blocked, removable):
        record = store.allocations / f"{lease.value['namespace']}.json"
        replace_metadata(record, {**lease.value, "expires": time.time() - 1})
        lease.guard.close()
    live = extra = None
    try:
        acquire, rmtree, unlink = native.StoreGuard.acquire, shutil.rmtree, Path.unlink

        def fail_lock(path, exclusive, create):
            if Path(path) == blocked.lock_path:
                raise IOException("orphan lock unavailable")
            return acquire(path, exclusive, create)

        def fail_partial_delete(path, *args, **kwargs):
            if Path(path) == blocked.directory:
                (blocked.directory / "object.mat").unlink(missing_ok=True)
                raise PermissionError("orphan deletion denied")
            return rmtree(path, *args, **kwargs)

        def fail_allocation_delete(path, *args, **kwargs):
            if path == blocked_record:
                raise PermissionError("orphan allocation deletion denied")
            return unlink(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            if failure == "lock":
                patch.setattr(native.StoreGuard, "acquire", fail_lock)
            elif failure == "partial_delete":
                patch.setattr(shutil, "rmtree", fail_partial_delete)
            else:
                patch.setattr(Path, "unlink", fail_allocation_delete)
            store.collect_expired()
            assert not removable.directory.exists()
            assert blocked_record.exists()
            assert store.snapshot() == {"queries": 1, "reserved_bytes": 2048}
            # Admission retries cleanup, but retained quota cannot be reused.
            live, extra = store.reserve("live"), store.reserve("extra")
            live.renew()
            with pytest.raises(RuntimeError, match="capacity"):
                store.reserve("full")
        deadline = time.monotonic() + 5
        while blocked_record.exists():
            store.collect_expired()
            assert time.monotonic() < deadline
            time.sleep(0.005)
        assert not blocked.directory.exists()
        assert not blocked_record.exists()
        assert store.snapshot() == {"queries": 2, "reserved_bytes": 4096}
    finally:
        for lease in (blocked, removable, live, extra):
            if lease is not None:
                lease.close(lambda: None)
    assert store.snapshot() == {"queries": 0, "reserved_bytes": 0}


@pytest.mark.parametrize("cleanup", ["orphan", "close_delete", "close_callback"])
def test_slow_cleanup_keeps_renewal_admission_and_quota_independent(tmp_path, monkeypatch, cleanup):
    store, other = pool(tmp_path, capacity_bytes=6144), pool(tmp_path, capacity_bytes=6144)
    victim, live = store.reserve("victim"), other.reserve("live")
    if cleanup == "orphan":
        record = store.allocations / f"{victim.value['namespace']}.json"
        replace_metadata(record, {**victim.value, "expires": time.time() - 1})
        victim.guard.close()
    entered, release = Event(), Event()
    rmtree = shutil.rmtree

    def pause():
        entered.set()
        assert release.wait(15), "test did not release slow cleanup"

    def slow_delete(path, *args, **kwargs):
        result = rmtree(path, *args, **kwargs)
        if Path(path) == victim.directory:
            # Even removing the entire tree cannot remove the lease lock and
            # allow a competing collector to release this owner's quota.
            pause()
        return result

    extra = None
    try:
        with monkeypatch.context() as patch, ThreadPoolExecutor(1) as executor:
            if cleanup != "close_callback":
                patch.setattr(shutil, "rmtree", slow_delete)

            def clean():
                if cleanup == "orphan":
                    store.collect_expired()
                else:
                    victim.close(pause if cleanup == "close_callback" else lambda: None)

            future = executor.submit(clean)
            try:
                assert entered.wait(5)
                assert native.StoreGuard.acquire(str(victim.lock_path), True, False) is None
                started = time.monotonic()
                live.renew()
                other.collect_expired()  # A second pool cannot steal cleanup.
                extra = other.reserve("new")
                assert other.snapshot() == {"queries": 3, "reserved_bytes": 6144}
                with pytest.raises(RuntimeError, match="capacity"):
                    other.reserve("full")
                with pytest.raises(RuntimeError, match="being cleaned"):
                    ActiveStoreLease(victim.to_dict())
                if cleanup == "orphan":
                    with pytest.raises(RuntimeError, match="participants"):
                        victim.close(lambda: pytest.fail("cleanup must wait for its exclusive lock"))
                assert time.monotonic() - started < 2
                assert not future.done()
            finally:
                release.set()
                future.result(timeout=5)
            if cleanup == "orphan":
                callbacks = []
                victim.close(lambda: callbacks.append(True))
                assert callbacks == [True]  # Still clear coordinator memory after peer collection.
    finally:
        release.set()
        for lease in (victim, live, extra):
            if lease is not None:
                lease.close(lambda: None)
    assert store.snapshot() == {"queries": 0, "reserved_bytes": 0}
    assert list(store.allocations.glob("*.lock")) == []

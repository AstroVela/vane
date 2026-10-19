# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Storage quota and locks remain owned until actual I/O cleanup succeeds."""

import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Condition, Event
from types import SimpleNamespace

import pytest

from vane import IOException
from vane._native import execution_runtime as native
from vane.execution.fte_store import ActiveStoreLease, ExchangeStore, StorePool, replace_metadata
from vane.execution.materialized_store import StorageCleanupPending


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


@pytest.mark.parametrize("capacity", [-1, True, 4096, (1 << 40) + 1])
def test_source_cache_requires_a_finite_separate_store_reservation(tmp_path, capacity):
    with pytest.raises(ValueError, match="cache|capacity"):
        pool(tmp_path, source_cache_bytes=capacity)


def test_source_cache_is_charged_until_explicit_store_close(tmp_path):
    store = pool(tmp_path, source_cache_bytes=1024)
    query = store.reserve("query")
    try:
        cache = store.source_cache(query)
        assert cache.snapshot()["capacity_bytes"] == 1024
        assert store.snapshot() == {"queries": 2, "reserved_bytes": 3072}
        query.close(lambda: None)
        assert store.snapshot() == {"queries": 1, "reserved_bytes": 1024}
        store.close()
        assert store.snapshot() == {"queries": 0, "reserved_bytes": 0}
    finally:
        query.close(lambda: None)
        store.close()


def test_worker_source_pin_survives_coordinator_pin_release(tmp_path):
    store = pool(tmp_path, source_cache_bytes=1024)
    query = store.reserve("query")
    store.source_cache(query)
    worker = ActiveStoreLease(query.to_dict())
    cache_directory = store._cache_lease.directory
    sentinel = cache_directory / "borrowed.parquet"
    sentinel.write_bytes(b"native reader owns these bytes")
    try:
        # Model loss of the coordinator's source pin: the worker must own an
        # independent lock, rather than depending on the sending process.
        for source in query._source_pins:
            source.close()
        with pytest.raises(StorageCleanupPending, match="participants"):
            store.close()
        assert sentinel.read_bytes() == b"native reader owns these bytes"
        assert store.snapshot()["reserved_bytes"] == 3072
        worker.close()
        query.close(lambda: None)
        store.close()
        assert not cache_directory.exists()
        assert store.snapshot()["reserved_bytes"] == 0
    finally:
        worker.close()
        query.close(lambda: None)
        store.close()


@pytest.mark.parametrize("metadata_lock_busy", [False, True])
def test_idle_source_cache_renews_its_lease(tmp_path, monkeypatch, metadata_lock_busy):
    from vane.execution import fte_store

    now = time.time()
    monkeypatch.setattr(
        fte_store, "time", SimpleNamespace(time=lambda: now, monotonic=time.monotonic, sleep=time.sleep)
    )
    store = pool(tmp_path, source_cache_bytes=1024, lease_seconds=0.5)
    query = store.reserve("query")
    blocker = None
    try:
        store.source_cache(query)
        query.close(lambda: None)
        owner = store._cache_lease
        record = store.allocations / f"{owner.value['namespace']}.json"
        original_expiry = fte_store._read(record)["expires"]
        completed_expiry = original_expiry
        renewed = Condition()
        renew = owner.renew

        def notify_renewal():
            nonlocal completed_expiry
            renew()
            with renewed:
                # A visible replacement can precede a failed directory sync.
                # Publish completion only after renewal returns successfully.
                completed_expiry = fte_store._read(record)["expires"]
                renewed.notify_all()

        monkeypatch.setattr(owner, "renew", notify_renewal)
        if metadata_lock_busy:
            blocker = native.StoreGuard.acquire(str(store.root / "allocation.lock"), True, True)
            assert blocker is not None
            acquire = fte_store.store_lock
            monkeypatch.setattr(fte_store, "store_lock", lambda path, **kw: acquire(path, timeout=0.02))
        # Exercise the real heartbeat beyond the original expiry without
        # requiring the CI scheduler or fsync to finish within a 0.5s lease.
        for _ in range(3):
            with renewed:
                now += 0.25
                assert renewed.wait_for(lambda: completed_expiry >= now + store.config.lease_seconds, timeout=5)
        assert now > original_expiry
        assert store._cache_error is None
        pin = ActiveStoreLease(owner.to_dict())
        try:
            assert pin.check()["expires"] > original_expiry
        finally:
            pin.close()
    finally:
        if blocker is not None:
            blocker.close()
        query.close(lambda: None)
        store.close()
    assert store.snapshot()["reserved_bytes"] == 0


def test_failed_source_cache_creation_returns_its_reservation(tmp_path, monkeypatch):
    from vane._native import execution_plan

    store = pool(tmp_path, source_cache_bytes=1024)
    query = store.reserve("query")

    def fail(*args):
        raise RuntimeError("cache creation failed")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(execution_plan, "SourceSnapshotCache", fail)
            with pytest.raises(RuntimeError, match="cache creation failed"):
                store.source_cache(query)
        assert store.snapshot() == {"queries": 1, "reserved_bytes": 2048}
        store.source_cache(query)
        assert store.snapshot()["reserved_bytes"] == 3072
    finally:
        query.close(lambda: None)
        store.close()


def test_failed_source_cache_heartbeat_fences_later_queries(tmp_path, monkeypatch):
    store = pool(tmp_path, source_cache_bytes=1024, lease_seconds=0.5)
    first = store.reserve("first")
    second = None
    try:
        store.source_cache(first)
        first.close(lambda: None)

        def fail():
            raise OSError("cache heartbeat failed")

        monkeypatch.setattr(store._cache_lease, "renew", fail)
        deadline = time.monotonic() + 5
        while store._cache_error is None:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        second = store.reserve("second")
        with pytest.raises(RuntimeError, match="source snapshot cache lease failed"):
            store.source_cache(second)
        assert store.snapshot()["reserved_bytes"] == 3072
    finally:
        first.close(lambda: None)
        if second is not None:
            second.close(lambda: None)
        store.close()
    assert store.snapshot()["reserved_bytes"] == 0


def test_background_collection_defers_a_busy_metadata_lock(tmp_path):
    store = pool(tmp_path)
    orphan = store.reserve("orphan")
    record = store.allocations / f"{orphan.value['namespace']}.json"
    replace_metadata(record, {**orphan.value, "expires": time.time() - 1})
    orphan.guard.close()
    blocker = native.StoreGuard.acquire(str(store.root / "allocation.lock"), True, True)
    assert blocker is not None
    try:
        with ThreadPoolExecutor(1) as workers:
            try:
                workers.submit(store.request_collection).result(timeout=1)
                assert orphan.directory.exists() and record.exists()
            finally:
                blocker.close()
        store.request_collection()
        store._collection_thread.join(timeout=5)
        assert not store._collection_thread.is_alive()
        assert not orphan.directory.exists() and not record.exists()
    finally:
        blocker.close()
        orphan.close(lambda: None)
    assert store.snapshot()["reserved_bytes"] == 0


def test_slow_renewal_keeps_other_leases_independent_and_serializes_own_close(tmp_path, monkeypatch):
    from vane.execution import fte_store

    store = pool(tmp_path, capacity_bytes=8192)
    slow, live = store.reserve("slow"), store.reserve("live")
    record = store.allocations / f"{slow.value['namespace']}.json"
    publish = fte_store.replace_metadata
    entered, release = Event(), Event()
    extra = None
    progressing = None

    def slow_sync(path, value):
        if path == record and not entered.is_set():
            entered.set()
            assert release.wait(10), "test did not release metadata sync"
        publish(path, value)

    def other_work():
        live.renew()
        store.collect_expired()
        return store.reserve("extra")

    try:
        with monkeypatch.context() as patch, ThreadPoolExecutor(3) as workers:
            patch.setattr(fte_store, "replace_metadata", slow_sync)
            renewing = workers.submit(slow.renew)
            try:
                assert entered.wait(5)
                closing = workers.submit(slow.close, lambda: None)
                progressing = workers.submit(other_work)
                extra = progressing.result(timeout=2)
                assert store.snapshot() == {"queries": 3, "reserved_bytes": 6144}
                assert not closing.done()
            finally:
                release.set()
                renewing.result(timeout=5)
            closing.result(timeout=5)
            assert not record.exists()
    finally:
        release.set()
        if extra is None and progressing is not None:
            extra = progressing.result(timeout=5)
        for lease in (slow, live, extra):
            if lease is not None:
                lease.close(lambda: None)
    assert store.snapshot()["reserved_bytes"] == 0


@pytest.mark.parametrize("invalid", ["recursive", "foreign_store", "too_many"])
def test_source_lease_wire_rejects_unbounded_or_foreign_pins(tmp_path, invalid):
    store, other = pool(tmp_path), pool(tmp_path / "other")
    query, foreign = store.reserve("query"), other.reserve("foreign")
    wire = query.to_dict()
    if invalid == "recursive":
        wire["source_leases"] = [wire]
    elif invalid == "foreign_store":
        wire["source_leases"] = [foreign.to_dict()]
    else:
        wire["source_leases"] = [foreign.to_dict(), foreign.to_dict()]
    try:
        with pytest.raises(ValueError, match="source cache"):
            ActiveStoreLease(wire)
        query.close(lambda: None)
        assert store.snapshot()["reserved_bytes"] == 0
    finally:
        query.close(lambda: None)
        foreign.close(lambda: None)


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


def test_lease_renewal_does_not_hide_storage_sync_timeouts(tmp_path, monkeypatch):
    from vane.execution import fte_store

    store = pool(tmp_path)
    lease = store.reserve("query")

    def fail(*args):
        raise TimeoutError("metadata sync timed out")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(fte_store, "replace_metadata", fail)
            with pytest.raises(TimeoutError, match="metadata sync timed out"):
                lease.renew()
    finally:
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
                if cleanup == "orphan":
                    assert not victim.directory.exists()  # Deletion alone does not retire its allocation.
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

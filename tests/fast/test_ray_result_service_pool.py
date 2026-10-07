# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Result actor reuse retains per-query ownership, fencing and failure isolation."""

import pytest

import vane
from tests.fast.test_ray_recovery_runtime import assert_idle, close_result, resources
from vane.execution.result_delivery import ResultDeliveryFull

pytestmark = [pytest.mark.real_ray, pytest.mark.usefixtures("ray_local")]


def test_reuse_across_modes_schemas_and_batch_limits(tmp_path):
    with vane.connect(backend="ray", resources=resources(tmp_path)) as connection:
        actors, epochs = [], []
        for mode, batch_rows, cast in (("pipelined", 2, "bigint"), ("fte", 5, "varchar"), ("pipelined", 1, "double")):
            with connection.query(
                f"select range::{cast} as value from range(17)", execution=mode, rows_per_batch=batch_rows
            ) as result:
                scheduler = result.context._reader
                actors.append(scheduler.relay._actor_id)
                epochs.append(scheduler.relay_epoch)
                rows = []
                for batch in result:
                    assert 0 < batch.num_rows <= batch_rows
                    rows.extend(batch.column(0).to_pylist())
                del batch
                convert = {"bigint": int, "varchar": str, "double": float}[cast]
                assert sorted(rows) == sorted(map(convert, range(17)))
            assert_idle(connection)
            assert connection.query_runtime.resource_snapshot()["result_services"] == {
                "leased": 0,
                "idle": 1,
                "capacity": 4,
            }
        assert len(set(actors)) == 1
        assert len(set(epochs)) == 3


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_stale_control_calls_cannot_touch_a_reused_actor(tmp_path, mode):
    import ray

    with vane.connect(backend="ray", resources=resources(tmp_path)) as connection:
        first = connection.query("select 42", execution=mode)
        old = first.context._reader
        first.collect()
        with connection.query("select range from range(100)", execution=mode) as result:
            current = result.context._reader
            assert current.relay._actor_id == old.relay._actor_id
            assert current.relay_epoch != old.relay_epoch
            assert current.result_endpoint["ticket"] != old.result_endpoint["ticket"]
            ray.get(old.relay.cancel.remote(old.relay_epoch, "late cancellation"), timeout=5)
            ray.get(old.relay.release.remote(old.relay_epoch), timeout=5)
            stale = [
                old.relay.status.remote(old.relay_epoch),
                old.relay.prepare.remote(old.relay_epoch, b"", ""),
                old.relay.connect.remote(old.relay_epoch, "", "", 1),
                old.relay.connect_materialized.remote(old.relay_epoch, {}, {}),
            ]
            for reference in stale:
                with pytest.raises(Exception, match="stale result service lease"):
                    ray.get(reference, timeout=5)
            assert sorted(result.collect().column(0).to_pylist()) == list(range(100))
        assert_idle(connection)


def test_concurrent_modes_have_independent_results_and_cancellation(tmp_path):
    with vane.connect(backend="ray", resources=resources(tmp_path, max_results=2)) as connection:
        sibling = connection.cursor()
        overflow = connection.cursor()
        first = connection.query("select range from range(1000000)")
        second = sibling.query("select range from range(100)", execution="fte")
        try:
            assert first.context._reader.relay._actor_id != second.context._reader.relay._actor_id
            assert connection.query_runtime.resource_snapshot()["result_services"] == {
                "leased": 2,
                "idle": 0,
                "capacity": 2,
            }
            with overflow:
                with pytest.raises(ResultDeliveryFull, match="slots are full"):
                    overflow.query("select 7")
            connection.interrupt()
            close_result(first)
            assert sorted(second.collect().column(0).to_pylist()) == list(range(100))
        finally:
            close_result(first)
            close_result(second)
            sibling.close()
        assert_idle(connection)
        assert connection.query_runtime.resource_snapshot()["result_services"] == {
            "leased": 0,
            "idle": 2,
            "capacity": 2,
        }
        assert connection.query("select 7").collect().column(0).to_pylist() == [7]
        assert_idle(connection)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
@pytest.mark.parametrize("active", [False, True])
def test_dead_actor_is_replaced_without_replaying_a_live_result(tmp_path, mode, active):
    import ray

    with vane.connect(backend="ray", resources=resources(tmp_path)) as connection:
        result = connection.query("select range from range(100)", execution=mode)
        actor = result.context._reader.relay
        if not active:
            result.collect()
        ray.kill(actor, no_restart=True)
        if active:
            try:
                with pytest.raises(Exception):
                    result.collect()
                assert result.execution_state == "FAILED"
            finally:
                close_result(result)
        with connection.query("select 7", execution=mode) as recovered:
            assert recovered.context._reader.relay._actor_id != actor._actor_id
            assert recovered.collect().column(0).to_pylist() == [7]
        assert_idle(connection)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_cancel_during_checkout_discards_the_unpublished_actor(tmp_path, monkeypatch, mode):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from vane.execution import pipelined_runtime
    from vane.execution.request_admission import RequestCancelled

    with vane.connect(backend="ray", resources=resources(tmp_path)) as connection:
        connection.query("select 1", execution=mode).collect()
        pool = connection.query_runtime.pool.results
        entered, proceed = threading.Event(), threading.Event()
        original = pipelined_runtime._get

        def delay_reservation(reference, context=None, timeout=30):
            if context is not None:
                # Warm workers need no describe RPC; the next synchronous
                # control call reserves the cached result actor.
                entered.set()
                assert proceed.wait(10)
            return original(reference, context, timeout)

        with monkeypatch.context() as patch, ThreadPoolExecutor(1) as threads:
            patch.setattr(pipelined_runtime, "_get", delay_reservation)
            pending = threads.submit(connection.query, "select 2", execution=mode)
            try:
                assert entered.wait(10)
                assert pool.snapshot()["leased"] == 1
                connection.interrupt()
            finally:
                proceed.set()
            with pytest.raises(RequestCancelled):
                pending.result(timeout=10)
        assert pool.snapshot() == {"leased": 0, "idle": 0, "capacity": 4}
        assert connection.query("select 7", execution=mode).collect().column(0).to_pylist() == [7]
        assert_idle(connection)


def test_session_close_kills_idle_actors(tmp_path):
    import ray

    with vane.connect(backend="ray", resources=resources(tmp_path)) as connection:
        result = connection.query("select 42")
        scheduler = result.context._reader
        result.collect()
        pool = connection.query_runtime.pool.results
        assert pool.snapshot()["idle"] == 1
    assert pool.snapshot() == {"leased": 0, "idle": 0, "capacity": 4}
    with pytest.raises(ray.exceptions.RayActorError):
        ray.get(scheduler.relay.reserve.remote("after-close", resources(tmp_path)), timeout=10)

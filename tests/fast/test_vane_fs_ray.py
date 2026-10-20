# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Native workspace replay and snapshot ownership across distributed plans."""

import gc
import hashlib
import pickle
import socket
from types import SimpleNamespace

import pytest

import vane

vane_fs = pytest.importorskip("vane_fs")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local")
    database = tmp_path / "workspace.sqlite"
    with vane_fs.Workspace(database) as workspace:
        workspace.checkout().write_file("/table.csv", b"id,value\n1,10\n2,20\n3,30\n")
        arrow = pytest.importorskip("pyarrow")
        parquet = pytest.importorskip("pyarrow.parquet")
        output = arrow.BufferOutputStream()
        parquet.write_table(arrow.table({"id": [1, 2, 3], "value": [10, 20, 30]}), output)
        workspace.checkout().write_file("/table.parquet", output.getvalue().to_pybytes())
        workspace.checkout().write_file("/payload", b"worker-visible-vane-fs-payload")
        frozen = workspace.snapshot()
        yield workspace, database, frozen


def _logical_plan(connection, query, query_id):
    return vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(connection.sql(query), query_id)


def _rows(result):
    return [tuple(row.values()) for table in result.partition_payloads for row in table.to_pylist()]


@pytest.mark.parametrize("file_format", ["csv", "parquet"])
def test_snapshot_scan_survives_logical_and_physical_plan_transport(workspace, file_format, monkeypatch):
    store, database, frozen = workspace
    url = vane_fs.snapshot_url(store.id, frozen, f"/table.{file_format}")
    monkeypatch.chdir(database.parent)
    with vane.connect() as source:
        vane_fs.register_workspace(source, database.name, timeout_ms=1234)
        logical = _logical_plan(source, f"SELECT * FROM read_{file_format}('{url}')", f"vane-fs-{file_format}")
        registration = logical.__getstate__()[3]["vane_fs_workspaces"][0]
        assert registration == {
            "workspace_id": store.id,
            "database": str(database.resolve()),
            "timeout_ms": 1234,
            "hostname": socket.gethostname(),
            "snapshots": [frozen],
        }
        transported_logical = pickle.loads(pickle.dumps(logical))
        vane_fs.unregister_workspace(source, store.id)
    with pytest.raises(BlockingIOError, match="pinned"):
        store.drop_snapshot(frozen)
    store.checkout().write_file(f"/table.{file_format}", b"changed live branch")
    store.collect_garbage()
    with vane.connect() as planning:
        physical = transported_logical.to_physical_plan(planning)
        transported_physical = pickle.loads(pickle.dumps(physical))
    del logical, transported_logical
    gc.collect()
    with pytest.raises(BlockingIOError, match="pinned"):
        store.drop_snapshot(frozen)
    with vane.connect() as worker, worker.cursor() as cursor:
        result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(cursor, transported_physical)
        assert sorted(_rows(result)) == [(1, 10), (2, 20), (3, 30)]
    del physical, transported_physical, result
    gc.collect()
    store.drop_snapshot(frozen)


@pytest.mark.parametrize("file_format", ["csv", "parquet"])
@pytest.mark.parametrize("fails", [False, True])
def test_preflight_preview_does_not_retain_snapshot_pins(workspace, monkeypatch, file_format, fails):
    import vane._ray_cxx as helpers

    store, database, frozen = workspace
    url = vane_fs.snapshot_url(store.id, frozen, f"/table.{file_format}")
    with vane.connect() as source:
        vane_fs.register_workspace(source, database)
        logical = _logical_plan(source, f"SELECT * FROM read_{file_format}('{url}')", "vane-fs-preflight-pins")
    validate = helpers.validate_plan_serialization_for_submission
    previews = []

    def preflight(plan):
        previews.append(plan)
        validate(plan)
        if fails:
            raise ValueError("injected serialization preflight failure")

    monkeypatch.setattr(helpers, "validate_plan_serialization_for_submission", preflight)
    with vane.connect() as planning:
        if fails:
            with pytest.raises(ValueError, match="injected serialization preflight failure") as failure:
                logical.to_physical_plan(planning)
            assert failure.value.__traceback__ is not None
        else:
            physical = logical.to_physical_plan(planning)
    logical._release_vane_fs_query_pins()
    if not fails:
        with pytest.raises(BlockingIOError, match="pinned"):
            store.drop_snapshot(frozen)
        physical._release_vane_fs_query_pins()
    # The preflight preview and any exception traceback remain alive after
    # their lifecycle-owned logical/physical counterparts release ownership.
    assert len(previews) == 1
    store.drop_snapshot(frozen)


@pytest.mark.parametrize("invalid", ["different_workspace", "different_host", "relative_path", "missing_database"])
def test_workspace_replay_rejects_changed_or_unavailable_storage(workspace, tmp_path, invalid):
    store, database, frozen = workspace
    with vane.connect() as source:
        vane_fs.register_workspace(source, database)
        logical = _logical_plan(source, "SELECT 42", f"vane-fs-invalid-{invalid}")
    state = list(logical.__getstate__())
    registration = state[3]["vane_fs_workspaces"][0]
    if invalid == "different_workspace":
        other_database = tmp_path / "other.sqlite"
        with vane_fs.Workspace(other_database):
            pass
        registration["database"] = str(other_database)
        expected_error, message = vane.InvalidInputException, "workspace identity changed"
    elif invalid == "different_host":
        registration["hostname"] += "-other-node"
        expected_error, message = ValueError, "same host"
    elif invalid == "relative_path":
        registration["database"] = "workspace.sqlite"
        expected_error, message = ValueError, "absolute database path"
    else:
        registration["database"] = str(tmp_path / "missing.sqlite")
        expected_error, message = FileNotFoundError, "existing VaneFS database"
    restored = vane.ray_cxx.PyLogicalPlan.__new__(vane.ray_cxx.PyLogicalPlan)
    restored.__setstate__(tuple(state))
    with vane.connect() as worker, pytest.raises(expected_error, match=message):
        restored.to_physical_plan(worker)
    assert not (tmp_path / "missing.sqlite").exists()
    store.drop_snapshot(frozen)


def test_worker_replay_retains_runtime_snapshots_until_query_cleanup(workspace):
    store, database, frozen = workspace
    url = vane_fs.snapshot_url(store.id, frozen, "/payload")
    with vane.connect() as source:
        vane_fs.register_workspace(source, database)
        logical = _logical_plan(source, f"SELECT file_size(to_file('{url}')) FROM range(3)", "vane-fs-runtime-pin")
        assert logical.__getstate__()[3]["vane_fs_workspaces"][0]["snapshots"] == []
        with vane.connect() as planning:
            physical = logical.to_physical_plan(planning)
            transported = pickle.loads(pickle.dumps(physical))
    query_id = transported.idx()
    assert vane.ray_cxx._register_query_python_replay_state(query_id, transported)
    try:
        for _attempt in range(2):
            base = vane.ray_cxx._prepare_query_snapshot_connection(query_id)
            try:
                vane.ray_cxx._validate_query_snapshot_connection(base, query_id)
                with base.cursor() as cursor:
                    result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(cursor, transported)
                    assert _rows(result) == [(len(b"worker-visible-vane-fs-payload"),)] * 3
            finally:
                base.close()
            with pytest.raises(BlockingIOError, match="pinned"):
                store.drop_snapshot(frozen)
            store.collect_garbage()
    finally:
        vane.ray_cxx._cleanup_query_python_replay_state(query_id)
    store.drop_snapshot(frozen)


@pytest.mark.parametrize("consumer", ["run_iter", "run_iter_tables"])
@pytest.mark.parametrize("outcome", ["complete", "error", "cancel"])
def test_runner_releases_binding_pins_after_stream_cleanup(workspace, consumer, outcome):
    from vane.runners.ray.runner import RayRunner

    store, database, frozen = workspace
    url = vane_fs.snapshot_url(store.id, frozen, "/table.parquet")
    with vane.connect() as source:
        vane_fs.register_workspace(source, database)
        logical = _logical_plan(source, f"SELECT * FROM read_parquet('{url}')", f"pin-stream-{consumer}-{outcome}")
    events = []
    marker = SimpleNamespace(partition=lambda: "partition")

    def stream_plan(_plan):
        try:
            yield marker
            if outcome == "error":
                raise RuntimeError("injected query failure")
        finally:
            with pytest.raises(BlockingIOError, match="pinned"):
                store.drop_snapshot(frozen)
            events.append("remote cleanup")

    runner = object.__new__(RayRunner)
    runner._client_for_session = lambda _session: SimpleNamespace(stream_plan=stream_plan)
    iterator = getattr(runner, consumer)(logical)
    assert next(iterator) == (marker if consumer == "run_iter" else "partition")
    if outcome == "cancel":
        iterator.close()
    elif outcome == "error":
        with pytest.raises(RuntimeError, match="injected query failure"):
            next(iterator)
    else:
        assert list(iterator) == []
    assert events == ["remote cleanup"]
    # The logical plan is still alive, as it can be in a result reference cycle.
    store.drop_snapshot(frozen)


@pytest.mark.parametrize("terminal", ["run_write", "run_datasink"])
@pytest.mark.parametrize("fails", [False, True])
def test_runner_releases_binding_pins_after_terminal_cleanup(workspace, terminal, fails):
    from vane.runners.ray.runner import RayRunner

    store, database, frozen = workspace
    url = vane_fs.snapshot_url(store.id, frozen, "/table.parquet")
    with vane.connect() as source:
        vane_fs.register_workspace(source, database)
        logical = _logical_plan(source, f"SELECT * FROM read_parquet('{url}')", f"pin-{terminal}-{fails}")

    def run(_plan):
        with pytest.raises(BlockingIOError, match="pinned"):
            store.drop_snapshot(frozen)
        if fails:
            raise RuntimeError("injected terminal failure")
        return {"rows": 3}

    runner = object.__new__(RayRunner)
    runner._client_for_session = lambda _session: SimpleNamespace(run_copy_plan=run, run_datasink_plan=run)
    if fails:
        with pytest.raises(RuntimeError, match="injected terminal failure"):
            getattr(runner, terminal)(logical)
    else:
        assert getattr(runner, terminal)(logical) == {"rows": 3}
    store.drop_snapshot(frozen)


@pytest.mark.real_ray
@pytest.mark.usefixtures("ray_local")
@pytest.mark.parametrize("file_format", ["csv", "parquet"])
def test_same_host_ray_failed_preflight_releases_snapshot_pins(workspace, monkeypatch, tmp_path, file_format):
    import ray

    from vane.runners.ray.runner import _RAY_RUNNERS

    store, database, frozen = workspace
    monkeypatch.setenv("VANE_RUNNER", "ray")
    url = vane_fs.snapshot_url(store.id, frozen, f"/table.{file_format}")

    def install_failure(actor):
        import vane._ray_cxx as helpers

        original = helpers.validate_plan_serialization_for_submission
        actor._vane_fs_original_preflight = original

        def reject(plan):
            unsupported = vane.ray_cxx._make_non_serializable_physical_plan_for_test(str(plan.idx()))
            original(unsupported)

        helpers.validate_plan_serialization_for_submission = reject

    def restore_and_inspect(actor):
        import vane._ray_cxx as helpers

        helpers.validate_plan_serialization_for_submission = actor._vane_fs_original_preflight
        del actor._vane_fs_original_preflight
        return (
            len(actor._plan_lifecycles),
            len(actor._plan_connections),
            len(actor.curr_plans),
            sum(record.error is not None for record in actor._copy_operation_terminal.values()),
        )

    with vane.connect() as connection:
        assert connection.execute("SELECT 42").fetchall() == [(42,)]
        client = next(iter(_RAY_RUNNERS)).query_driver_client
        assert client is not None
        driver = client.runner
        vane_fs.register_workspace(connection, database)
        ray.get(driver.__ray_call__.remote(install_failure))
        try:
            with pytest.raises(Exception, match="serialization preflight failed") as failure:
                connection.execute(
                    f"COPY (SELECT * FROM read_{file_format}('{url}')) "
                    f"TO '{tmp_path / 'result.parquet'}' (FORMAT PARQUET)"
                ).fetchall()
        finally:
            state = ray.get(driver.__ray_call__.remote(restore_and_inspect))
        assert state[:3] == (0, 0, 0)
        assert state[3] > 0  # The remote error cache and its tracebacks remain intact.
        assert failure.value is not None
        store.drop_snapshot(frozen)


@pytest.mark.real_ray
@pytest.mark.usefixtures("ray_local")
@pytest.mark.parametrize("file_format", ["csv", "parquet"])
@pytest.mark.parametrize("terminal", ["select", "copy"])
def test_same_host_ray_failed_queries_release_snapshot_pins(workspace, monkeypatch, tmp_path, file_format, terminal):
    store, database, frozen = workspace
    monkeypatch.setenv("VANE_RUNNER", "ray")
    url = vane_fs.snapshot_url(store.id, frozen, f"/table.{file_format}")
    query = (
        "SELECT CASE WHEN id = 2 THEN error('injected VaneFS query failure') "
        f"ELSE id END AS id FROM read_{file_format}('{url}')"
    )
    if terminal == "copy":
        query = f"COPY ({query}) TO '{tmp_path / 'result.parquet'}' (FORMAT PARQUET)"
    with vane.connect() as connection:
        vane_fs.register_workspace(connection, database)
        with pytest.raises(Exception, match="injected VaneFS query failure") as failure:
            connection.execute(query).fetchall()
        # Keep the exception and connection alive. COPY also retains its error
        # in the remote driver's outcome cache; neither traceback may pin data
        # after deterministic teardown, without cache eviction or Python GC.
        assert failure.value is not None
        store.drop_snapshot(frozen)


@pytest.mark.real_ray
@pytest.mark.usefixtures("ray_local")
def test_same_host_ray_queries_native_snapshots(workspace, monkeypatch):
    store, database, frozen = workspace
    monkeypatch.setenv("VANE_RUNNER", "ray")
    csv_url = vane_fs.snapshot_url(store.id, frozen, "/table.csv")
    parquet_url = vane_fs.snapshot_url(store.id, frozen, "/table.parquet")
    payload_url = vane_fs.snapshot_url(store.id, frozen, "/payload")
    expected_digest = hashlib.sha256(b"worker-visible-vane-fs-payload").hexdigest()
    with vane.connect() as connection:
        vane_fs.register_workspace(connection, database)
        relation = connection.sql(
            f"""
            SELECT c.id, c.value + p.value AS combined
            FROM read_csv('{csv_url}') c
            JOIN read_parquet('{parquet_url}') p USING (id)
            WHERE c.id > 1
            ORDER BY c.id
            """
        )
        assert relation._get_runner_type() == "ray"
        store.checkout().write_file("/table.csv", b"changed live branch")
        store.collect_garbage()
        assert relation.fetchall() == [(2, 40), (3, 60)]
        fresh = store.snapshot()
        fresh_url = vane_fs.snapshot_url(store.id, fresh, "/table.parquet")
        assert connection.execute("SELECT sum(value)::BIGINT FROM read_parquet(?)", [fresh_url]).fetchall() == [(60,)]
        store.drop_snapshot(fresh)
        result = connection.execute(
            "SELECT file_content_id(file_enrich(to_file(?), ['checksum'])) FROM range(3)", [payload_url]
        ).fetchall()
        assert result == [(f"file-content-v1:checksum:sha256:{expected_digest}",)] * 3
        vane_fs.unregister_workspace(connection, store.id)
        with pytest.raises(vane.IOException, match="not registered"):
            connection.execute("SELECT * FROM read_parquet(?)", [parquet_url]).fetchall()
    store.drop_snapshot(frozen)

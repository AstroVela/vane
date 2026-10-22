# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest

import vane


def _configured_artifact_path(environment_variable: str) -> Path:
    configured_path = os.environ.get(environment_variable)
    if configured_path is None:
        pytest.skip(f"set {environment_variable} to test the staged artifact")

    path = Path(configured_path).resolve()
    assert path.is_file(), f"loadable extension artifact does not exist: {path}"
    return path


@pytest.fixture(scope="module")
def loadable_extension_path() -> Path:
    return _configured_artifact_path("VANE_TEST_LOADABLE_EXTENSION_PATH")


@pytest.fixture(scope="module")
def loadable_httpfs_extension_path() -> Path:
    return _configured_artifact_path("VANE_TEST_LOADABLE_HTTPFS_EXTENSION_PATH")


def _replanned_batch(batch):
    return batch


class _ReplannedActor:
    def __call__(self, batch):
        return batch


@pytest.mark.parametrize("runner", ["local", "local-fast"])
@pytest.mark.parametrize("backend", ["task", "actor"])
@pytest.mark.parametrize("rows", [0, 4101])
def test_private_copy_replanning_preserves_local_udf_execution(monkeypatch, runner, backend, rows):
    artifact = _configured_artifact_path("VANE_TEST_LOADABLE_BATCH_DEMO_PATH")
    monkeypatch.setenv("VANE_RUNNER", runner)

    import ray

    def forbid_ray(*_args, **_kwargs):
        raise AssertionError("local COPY replanning must not initialize Ray")

    monkeypatch.setattr(ray, "init", forbid_ray)
    with vane.connect(config={"autoinstall_known_extensions": "false", "autoload_known_extensions": "false"}) as con:
        con.load_extension(str(artifact))
        source_sql = (
            "SELECT i::BIGINT AS id, CASE WHEN i % 3 = 0 THEN NULL ELSE 'row-' || i END AS payload, "
            f"[i::FLOAT, NULL::FLOAT, (i + 1)::FLOAT]::FLOAT[3] AS vector FROM range({rows}) t(i)"
        )
        expected = con.sql(source_sql + " ORDER BY id").fetchall()
        assert (
            con.execute(f"COPY ({source_sql} ORDER BY id) TO 'unused' (FORMAT batch_demo_replan)").fetchall()
            == expected
        )
        options = {
            "schema": {"id": vane.sqltypes.BIGINT, "payload": vane.sqltypes.VARCHAR, "vector": "FLOAT[3]"},
            "batch_size": 137,
            "cpus": 1,
            "gpus": 0,
            "execution_backend": f"subprocess_{backend}",
        }
        if backend == "actor":
            options["actor_number"] = 1
        con.sql(source_sql).map_batches(
            _ReplannedActor if backend == "actor" else _replanned_batch, **options
        ).create_view("copy_input")
        assert (
            con.execute("COPY (SELECT * FROM copy_input ORDER BY id) TO 'unused' (FORMAT batch_demo_replan)").fetchall()
            == expected
        )
        assert con.execute("SELECT 1").fetchone() == (1,)


def test_staged_tpch_extension_loads_without_static_linkage(loadable_extension_path: Path):
    connection = vane.connect(config={"allow_unsigned_extensions": "true"})
    try:
        initial_state = connection.execute(
            """
            SELECT loaded, installed, install_mode
            FROM duckdb_extensions()
            WHERE extension_name = 'tpch'
            """
        ).fetchone()
        assert initial_state == (False, False, "NOT_INSTALLED")

        connection.load_extension(str(loadable_extension_path))

        loaded_state = connection.execute(
            """
            SELECT loaded, installed, install_mode
            FROM duckdb_extensions()
            WHERE extension_name = 'tpch'
            """
        ).fetchone()
        assert loaded_state == (True, False, "NOT_INSTALLED")
        assert connection.execute("SELECT count(*) FROM tpch_queries()").fetchone() == (22,)
    finally:
        connection.close()


@pytest.mark.parametrize("entry", ["execute", "sql"])
@pytest.mark.parametrize("transaction", [False, True])
def test_direct_tpch_pragma_keeps_native_data_execution(
    loadable_extension_path, monkeypatch, tmp_path, entry, transaction
):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    database = str(tmp_path / "tpch.duckdb")
    config = {"allow_unsigned_extensions": "true"}
    with vane.connect(database, config=config) as setup:
        setup.load_extension(str(loadable_extension_path))
        setup.execute("CALL dbgen(sf=0)")

    monkeypatch.setenv("VANE_RUNNER", "ray")

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("an unchanged native pragma must not initialize Ray")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    with vane.connect(database, config=config) as connection:
        connection.load_extension(str(loadable_extension_path))
        if transaction:
            connection.begin()
        assert getattr(connection, entry)("PRAGMA tpch(1)").fetchall() == []
        if transaction:
            connection.commit()
        # Completed rows pass source admission, but this unsigned fixture was
        # loaded directly and cannot enter a distributed connection snapshot.
        result = connection.sql("PRAGMA tpch(1)")
        with pytest.raises(ValueError, match="not loaded through vane.DynamicExtensionResolver: tpch"):
            vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(result.project("*"), None)


def test_staged_httpfs_extension_loads_without_static_linkage(loadable_httpfs_extension_path: Path):
    connection = vane.connect(config={"allow_unsigned_extensions": "true"})
    try:
        initial_state = connection.execute(
            """
            SELECT loaded, installed, install_mode
            FROM duckdb_extensions()
            WHERE extension_name = 'httpfs'
            """
        ).fetchone()
        assert initial_state == (False, False, "NOT_INSTALLED")

        connection.load_extension(str(loadable_httpfs_extension_path))

        loaded_state = connection.execute(
            """
            SELECT loaded, installed, install_mode
            FROM duckdb_extensions()
            WHERE extension_name = 'httpfs'
            """
        ).fetchone()
        assert loaded_state == (True, False, "NOT_INSTALLED")
    finally:
        connection.close()

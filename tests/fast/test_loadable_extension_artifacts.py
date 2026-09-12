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
        with pytest.raises(vane.NotImplementedException, match="client connection queries"):
            connection.sql("PRAGMA tpch(1)").project("*").fetchall()
        with pytest.raises((ValueError, vane.NotImplementedException), match="client connection quer"):
            vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(connection.sql("PRAGMA tpch(1)"), None)


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

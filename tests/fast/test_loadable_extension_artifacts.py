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

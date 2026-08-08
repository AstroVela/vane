#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Verify install-order, import-order, and uninstall isolation from DuckDB."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import textwrap
import venv
from pathlib import Path

OFFICIAL_DUCKDB_REQUIREMENT = "duckdb"
ADBC_DRIVER_MANAGER_REQUIREMENT = "adbc-driver-manager"


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _pip(python: Path, *arguments: str) -> None:
    _run([str(python), "-m", "pip", "--disable-pip-version-check", *arguments])


def _python(python: Path, source: str, *, cwd: Path) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONSAFEPATH"] = "1"
    subprocess.run(
        [str(python), "-c", textwrap.dedent(source)],
        cwd=cwd,
        env=environment,
        check=True,
    )


def _assert_vane_only(python: Path, *, cwd: Path) -> None:
    _python(
        python,
        """
        import importlib.util
        import sys
        import vane
        import vane.adbc.dbapi as vane_adbc
        from vane import _native

        assert importlib.util.find_spec("duckdb") is None
        assert importlib.util.find_spec("adbc_driver_duckdb") is None
        assert "duckdb" not in sys.modules
        assert "_duckdb" not in sys.modules
        assert _native.__name__ == "vane._native"
        assert vane.connect().execute("SELECT 41 + 1").fetchone() == (42,)
        with vane_adbc.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 21 * 2")
                assert cursor.fetchone() == (42,)
        """,
        cwd=cwd,
    )


def _assert_duckdb_only(python: Path, *, cwd: Path) -> None:
    _python(
        python,
        """
        import importlib.util
        import sys
        import adbc_driver_duckdb.dbapi as duckdb_adbc
        import duckdb
        import _duckdb

        assert importlib.util.find_spec("vane") is None
        assert not any(name == "vane" or name.startswith("vane.") for name in sys.modules)
        assert _duckdb.__name__ == "_duckdb"
        assert duckdb.sql("SELECT 40 + 2").fetchone() == (42,)
        with duckdb_adbc.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 21 * 2")
                assert cursor.fetchone() == (42,)
        """,
        cwd=cwd,
    )


def _assert_both(python: Path, first: str, second: str, *, cwd: Path) -> None:
    _python(
        python,
        f"""
        import importlib
        import importlib.metadata
        from pathlib import Path

        first_module = importlib.import_module({first!r})
        second_module = importlib.import_module({second!r})
        vane = importlib.import_module("vane")
        duckdb = importlib.import_module("duckdb")
        vane_native = importlib.import_module("vane._native")
        duckdb_native = importlib.import_module("_duckdb")
        vane_adbc = importlib.import_module("vane.adbc.dbapi")
        duckdb_adbc = importlib.import_module("adbc_driver_duckdb.dbapi")
        vane_udf = importlib.import_module("vane.udf")
        duckdb_func = importlib.import_module("duckdb.func")
        vane_connection = vane.connect()
        duckdb_connection = duckdb.connect()

        assert first_module is importlib.import_module({first!r})
        assert second_module is importlib.import_module({second!r})
        assert vane_connection.execute("SELECT 6 * 7").fetchone() == (42,)
        assert duckdb_connection.execute("SELECT 84 / 2").fetchone() == (42.0,)
        for type_name in (
            "DuckDBPyConnection",
            "DuckDBPyRelation",
            "Statement",
            "Expression",
            "StatementType",
            "ExpectedResultType",
            "ExplainType",
            "CSVLineTerminator",
            "PythonExceptionHandling",
            "RenderMode",
            "token_type",
        ):
            assert getattr(vane, type_name) is not getattr(duckdb, type_name), type_name
        assert vane.sqltypes.DuckDBPyType is not duckdb.sqltypes.DuckDBPyType
        assert vane_udf.PythonUDFType is not duckdb_func.PythonUDFType
        assert vane_udf.FunctionNullHandling is not duckdb_func.FunctionNullHandling
        assert vane_native is not duckdb_native
        assert Path(vane_native.__file__).resolve() != Path(duckdb_native.__file__).resolve()
        assert vane_native.__name__ == "vane._native"
        assert duckdb_native.__name__ == "_duckdb"

        vane_files = {{str(path) for path in importlib.metadata.distribution("vane-ai").files or ()}}
        duckdb_files = {{str(path) for path in importlib.metadata.distribution("duckdb").files or ()}}
        assert vane_files.isdisjoint(duckdb_files), sorted(vane_files & duckdb_files)
        assert not any(
            path == "duckdb" or path.startswith(("duckdb/", "_duckdb", "adbc_driver_duckdb/"))
            for path in vane_files
        )

        try:
            vane_connection.execute("SELECT * FROM missing_vane_table")
        except vane.CatalogException as error:
            assert type(error).__module__ == "vane._native"
        else:
            raise AssertionError("Vane query unexpectedly succeeded")

        try:
            duckdb_connection.execute("SELECT * FROM missing_duckdb_table")
        except duckdb.CatalogException as error:
            assert type(error).__module__ == "_duckdb"
        else:
            raise AssertionError("DuckDB query unexpectedly succeeded")

        for driver in (vane_adbc, duckdb_adbc):
            with driver.connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 21 * 2")
                    assert cursor.fetchone() == (42,)
        """,
        cwd=cwd,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vane_wheel", type=Path)
    args = parser.parse_args()
    wheel = args.vane_wheel.resolve(strict=True)

    with tempfile.TemporaryDirectory(prefix="vane-duckdb-coexistence-") as temporary_directory:
        cwd = Path(temporary_directory)
        environment_directory = cwd / "venv"
        venv.EnvBuilder(with_pip=True).create(environment_directory)
        executable_name = "python.exe" if os.name == "nt" else "python"
        python = environment_directory / ("Scripts" if os.name == "nt" else "bin") / executable_name
        _pip(python, "install", ADBC_DRIVER_MANAGER_REQUIREMENT)

        # Vane followed by DuckDB.
        _pip(python, "install", str(wheel))
        _assert_vane_only(python, cwd=cwd)
        _pip(python, "install", "--no-deps", OFFICIAL_DUCKDB_REQUIREMENT)
        _assert_both(python, "vane", "duckdb", cwd=cwd)
        _assert_both(python, "duckdb", "vane", cwd=cwd)
        _pip(python, "uninstall", "-y", "vane-ai")
        _assert_duckdb_only(python, cwd=cwd)

        # DuckDB followed by Vane.
        _pip(python, "install", "--no-deps", str(wheel))
        _assert_both(python, "duckdb", "vane", cwd=cwd)
        _assert_both(python, "vane", "duckdb", cwd=cwd)
        _pip(python, "uninstall", "-y", "duckdb")
        _assert_vane_only(python, cwd=cwd)

        # Leave the validation environment usable with both distributions.
        _pip(python, "install", "--no-deps", OFFICIAL_DUCKDB_REQUIREMENT)
        _pip(python, "check")
        _assert_both(python, "vane", "duckdb", cwd=cwd)

    print("verified Vane/DuckDB install, import, execution, and uninstall isolation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
VALIDATION_TIMEOUT_SECONDS = 120


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _pip(python: Path, *arguments: str) -> None:
    _run([str(python), "-m", "pip", "--disable-pip-version-check", *arguments])


def _python(python: Path, source: str, *, cwd: Path) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONOPTIMIZE", None)
    environment["PYTHONSAFEPATH"] = "1"
    validation_source = textwrap.dedent(source)
    validation_program = f"exec(compile({validation_source!r}, '<coexistence-validation>', 'exec', optimize=0))"
    try:
        subprocess.run(
            [str(python), "-I", "-c", validation_program],
            cwd=cwd,
            env=environment,
            check=True,
            timeout=VALIDATION_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"coexistence validation exceeded {VALIDATION_TIMEOUT_SECONDS} seconds") from error


def _assert_vane_only(python: Path, *, cwd: Path) -> None:
    _python(
        python,
        """
        import importlib.metadata
        import importlib.util
        import sys
        import vane
        import vane.adbc.dbapi as vane_adbc
        from vane import _native

        assert importlib.util.find_spec("duckdb") is None
        assert importlib.util.find_spec("_duckdb") is None
        assert importlib.util.find_spec("adbc_driver_duckdb") is None
        assert "duckdb" not in sys.modules
        assert "_duckdb" not in sys.modules
        assert _native.__name__ == "vane._native"
        assert vane.connect().execute("SELECT 41 + 1").fetchone() == (42,)
        with vane_adbc.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 21 * 2")
                assert cursor.fetchone() == (42,)
        try:
            importlib.metadata.distribution("duckdb")
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            raise AssertionError("official DuckDB distribution metadata unexpectedly remains installed")
        """,
        cwd=cwd,
    )


def _assert_duckdb_only(python: Path, *, cwd: Path) -> None:
    _python(
        python,
        """
        import importlib.metadata
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
        try:
            importlib.metadata.distribution("vane-ai")
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            raise AssertionError("Vane distribution metadata unexpectedly remains installed")
        """,
        cwd=cwd,
    )


def _assert_both(python: Path, first: str, second: str, *, cwd: Path) -> None:
    _python(
        python,
        f"""
        import importlib
        import importlib.metadata
        import queue
        import threading
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

        concurrent_errors = queue.SimpleQueue()
        query_barrier = threading.Barrier(2, timeout=10)

        def run_concurrent_queries(engine_name, connect):
            connection = None
            try:
                connection = connect()
                query_barrier.wait()
                for value in range(32):
                    result = connection.execute("SELECT " + str(value) + " + 1").fetchone()
                    if result != (value + 1,):
                        raise AssertionError((engine_name, value, result))
            except BaseException as error:
                concurrent_errors.put((engine_name, error))
            finally:
                if connection is not None:
                    connection.close()

        query_threads = [
            threading.Thread(target=run_concurrent_queries, args=("vane", vane.connect)),
            threading.Thread(target=run_concurrent_queries, args=("duckdb", duckdb.connect)),
        ]
        for thread in query_threads:
            thread.start()
        for thread in query_threads:
            thread.join(timeout=30)
            if thread.is_alive():
                raise AssertionError("concurrent coexistence query did not finish")
        if not concurrent_errors.empty():
            raise AssertionError(concurrent_errors.get())

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

        def assert_rejected(exception_type, callback, message_fragment=None):
            try:
                callback()
            except exception_type as error:
                if message_fragment is not None:
                    assert message_fragment in str(error), str(error)
            else:
                raise AssertionError("one engine accepted the other engine's native object or value")

        vane_udf_args = ([vane.sqltypes.INTEGER], vane.sqltypes.INTEGER)
        assert_rejected(
            TypeError,
            lambda: vane_connection.create_function(
                "reject_official_udf_type",
                lambda value: value,
                *vane_udf_args,
                type=duckdb_func.PythonUDFType.NATIVE,
            ),
        )
        assert_rejected(
            TypeError,
            lambda: vane_connection.create_function(
                "reject_official_null_handling",
                lambda value: value,
                *vane_udf_args,
                null_handling=duckdb_func.FunctionNullHandling.DEFAULT,
            ),
        )
        assert_rejected(
            TypeError,
            lambda: vane_connection.create_function(
                "reject_official_exception_handling",
                lambda value: value,
                *vane_udf_args,
                exception_handling=duckdb.PythonExceptionHandling.DEFAULT,
            ),
        )

        vane_relation = vane_connection.sql("SELECT 1 AS value")
        duckdb_relation = duckdb_connection.sql("SELECT 1 AS value")
        assert_rejected(
            TypeError,
            lambda: vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
                duckdb_relation,
                "reject-official-relation",
            ),
            "Vane DuckDBPyRelation",
        )
        assert_rejected(
            vane.InvalidInputException,
            lambda: vane_connection.execute(duckdb.extract_statements("SELECT 1")[0]),
            "Vane DuckDBPyStatement",
        )
        assert_rejected(TypeError, lambda: vane_relation.explain(duckdb.ExplainType.ANALYZE))
        assert_rejected(
            vane.InvalidInputException,
            lambda: vane_relation.show(render_mode=duckdb.RenderMode.COLUMNS),
        )
        csv_path = Path("coexistence-enum.csv")
        csv_path.write_text("value\\n1\\n", encoding="utf-8")
        assert_rejected(
            vane.BinderException,
            lambda: vane.read_csv(
                str(csv_path),
                lineterminator=duckdb.CSVLineTerminator.LINE_FEED,
            ),
        )

        assert_rejected(
            TypeError,
            lambda: duckdb_connection.create_function(
                "reject_vane_udf_type",
                lambda value: value,
                [duckdb.sqltypes.INTEGER],
                duckdb.sqltypes.INTEGER,
                type=vane_udf.PythonUDFType.NATIVE,
            ),
        )
        assert_rejected(TypeError, lambda: vane.array_type(duckdb.sqltypes.INTEGER))
        assert_rejected(TypeError, lambda: duckdb.array_type(vane.sqltypes.INTEGER))

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

        with vane_adbc.connect() as connection:
            info = connection.adbc_get_info()
            assert info["vendor_name"] == "Vane", info
            assert info["driver_name"] == "ADBC Vane Driver", info

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

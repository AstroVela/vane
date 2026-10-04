# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import contextlib
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import vane


def test_dynamic_pivot_preprocessing_serializes_connection_calls(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    script = textwrap.dedent(
        """
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        import vane

        start = Barrier(4)
        with vane.connect() as connection:
            def run(worker):
                start.wait()
                for _ in range(50):
                    if worker == 1:
                        connection.execute("SET threads=2")
                    elif worker == 3:
                        assert connection.sql("PRAGMA functions").columns[0] == 'name'
                    else:
                        relation = connection.sql(
                            "PIVOT (SELECT 'a' AS k, 1 AS v UNION ALL SELECT 'b', 2) ON k USING sum(v)"
                        )
                        assert relation.columns == ['a', 'b']

            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(run, range(4)))
        """
    )
    # A GIL/context-lock deadlock cannot be interrupted by a Python test timeout.
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


class TestMultiStatement:
    def test_multi_statement(self, duckdb_cursor):
        con = vane.connect(":memory:")

        # test empty statement
        con.execute("")

        # run multiple statements in one call to execute
        con.execute(
            """
        CREATE TABLE integers(i integer);
        insert into integers select * from range(10);
        select * from integers;
        """
        )
        results = [x[0] for x in con.fetchall()]
        assert results == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # test export/import
        export_location = Path.cwd() / "vane_pytest_dir_export"
        with contextlib.suppress(Exception):
            shutil.rmtree(export_location)
        con.execute("CREATE TABLE integers2(i INTEGER)")
        con.execute("INSERT INTO integers2 VALUES (1), (5), (7), (1928)")
        con.execute(f"EXPORT DATABASE '{export_location}'")
        # reset connection
        con = vane.connect(":memory:")
        con.execute(f"IMPORT DATABASE '{export_location}'")
        integers = [x[0] for x in con.execute("SELECT * FROM integers").fetchall()]
        integers2 = [x[0] for x in con.execute("SELECT * FROM integers2").fetchall()]
        assert integers == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert integers2 == [1, 5, 7, 1928]
        shutil.rmtree(export_location)

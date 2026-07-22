# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import contextlib
from pathlib import Path

import pytest

import vane


def get_tables(con):
    tbls = con.execute("SHOW TABLES").fetchall()
    tbls = [x[0] for x in tbls]
    tbls.sort()
    return tbls


def test_multiple_writes():
    with contextlib.suppress(Exception):
        Path("test.db").unlink()
    con1 = vane.connect("test.db")
    con2 = vane.connect("test.db")
    con1.execute("CREATE TABLE foo1 as SELECT 1 as a, 2 as b")
    con2.execute("CREATE TABLE bar1 as SELECT 2 as a, 3 as b")
    con2.close()
    con1.close()
    con3 = vane.connect("test.db")
    tbls = get_tables(con3)
    assert tbls == ["bar1", "foo1"]
    del con1
    del con2
    del con3

    with contextlib.suppress(Exception):
        Path("test.db").unlink()


def test_multiple_writes_memory():
    con1 = vane.connect()
    con2 = vane.connect()
    con1.execute("CREATE TABLE foo1 as SELECT 1 as a, 2 as b")
    con2.execute("CREATE TABLE bar1 as SELECT 2 as a, 3 as b")
    con3 = vane.connect(":memory:")
    tbls = get_tables(con1)
    assert tbls == ["foo1"]
    tbls = get_tables(con2)
    assert tbls == ["bar1"]
    tbls = get_tables(con3)
    assert tbls == []
    del con1
    del con2
    del con3


def test_multiple_writes_named_memory():
    con1 = vane.connect(":memory:1")
    con2 = vane.connect(":memory:1")
    con1.execute("CREATE TABLE foo1 as SELECT 1 as a, 2 as b")
    con2.execute("CREATE TABLE bar1 as SELECT 2 as a, 3 as b")
    con3 = vane.connect(":memory:1")
    tbls = get_tables(con3)
    assert tbls == ["bar1", "foo1"]
    del con1
    del con2
    del con3


def test_diff_config():
    con1 = vane.connect("test.db", False)
    with pytest.raises(
        vane.ConnectionException,
        match="Can't open a connection to same database file with a different configuration than existing connections",
    ):
        vane.connect("test.db", True)
    con1.close()
    del con1


def test_diff_config_extended():
    con1 = vane.connect("test.db", config={"null_order": "NULLS FIRST"})
    with pytest.raises(
        vane.ConnectionException,
        match="Can't open a connection to same database file with a different configuration than existing connections",
    ):
        vane.connect("test.db")
    con1.close()
    del con1

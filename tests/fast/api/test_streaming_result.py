# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import hashlib

import pytest

import vane

STREAMING_BACKPRESSURE_ROW_COUNT = 20_000
STREAMING_DICTIONARY_VALUES = ("alpha", "beta", "gamma", "delta")


def _streaming_backpressure_expected_row(index):
    payload = chr(65 + index % 26) * 256 + f":{index}"
    nullable = None if index % 17 == 0 else index * 3
    return index, payload, nullable


def _streaming_backpressure_hash(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(repr(row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _fetch_streaming_backpressure_rows(cursor, fetch_method, column_names):
    if fetch_method == "fetchall":
        return cursor.fetchall()
    if fetch_method == "fetchmany":
        rows = []
        while True:
            batch = cursor.fetchmany(777)
            if not batch:
                return rows
            rows.extend(batch)

    pytest.importorskip("pyarrow")
    if fetch_method == "arrow_table":
        table = cursor.to_arrow_table()
        return [tuple(row[name] for name in column_names) for row in table.to_pylist()]
    if fetch_method == "arrow_reader":
        rows = []
        for batch in cursor.to_arrow_reader(STREAMING_BACKPRESSURE_ROW_COUNT):
            rows.extend(tuple(row[name] for name in column_names) for row in batch.to_pylist())
        return rows
    raise AssertionError(f"unsupported fetch method: {fetch_method}")


def _streaming_dictionary_input():
    pa = pytest.importorskip("pyarrow")
    indices = pa.array(
        [
            None if index % 19 == 0 else index % len(STREAMING_DICTIONARY_VALUES)
            for index in range(STREAMING_BACKPRESSURE_ROW_COUNT)
        ],
        type=pa.int16(),
    )
    dictionary = pa.DictionaryArray.from_arrays(indices, pa.array(STREAMING_DICTIONARY_VALUES))
    table = pa.table({"i": pa.array(range(STREAMING_BACKPRESSURE_ROW_COUNT)), "category": dictionary})
    return pa.Table.from_batches(table.to_batches(257))


class TestStreamingResult:
    @pytest.mark.parametrize("threads", [1, 4])
    @pytest.mark.parametrize("fetch_method", ["fetchall", "fetchmany", "arrow_table", "arrow_reader"])
    def test_blocked_execution_batch_preserves_payload(self, duckdb_cursor, threads, fetch_method):
        # Each wide result chunk exceeds the buffer, so consuming all rows forces
        # the collector to block and retry successive batches more than twice.
        duckdb_cursor.execute("SET streaming_buffer_size='32KB'")
        duckdb_cursor.execute(f"SET threads={threads}")
        cursor = duckdb_cursor.execute(
            f"""
            SELECT
                i,
                repeat(chr(65 + (i % 26)::INTEGER), 256) || ':' || i::VARCHAR AS payload,
                CASE WHEN i % 17 = 0 THEN NULL ELSE i * 3 END AS nullable
            FROM range({STREAMING_BACKPRESSURE_ROW_COUNT}) t(i)
            """
        )

        rows = _fetch_streaming_backpressure_rows(cursor, fetch_method, ("i", "payload", "nullable"))
        expected = [_streaming_backpressure_expected_row(index) for index in range(STREAMING_BACKPRESSURE_ROW_COUNT)]

        assert len(rows) == STREAMING_BACKPRESSURE_ROW_COUNT
        assert rows[0] == expected[0]
        assert rows[-1] == expected[-1]
        assert [row[0] for row in rows] == list(range(STREAMING_BACKPRESSURE_ROW_COUNT))
        assert [row[0] for row in rows if row[2] is None] == list(range(0, STREAMING_BACKPRESSURE_ROW_COUNT, 17))
        assert _streaming_backpressure_hash(rows) == _streaming_backpressure_hash(expected)

    @pytest.mark.parametrize("threads", [1, 4])
    @pytest.mark.parametrize("fetch_method", ["fetchall", "fetchmany", "arrow_table", "arrow_reader"])
    def test_blocked_execution_batch_preserves_dictionary_input(self, duckdb_cursor, threads, fetch_method):
        duckdb_cursor.execute("SET streaming_buffer_size='32KB'")
        duckdb_cursor.execute(f"SET threads={threads}")
        duckdb_cursor.register("streaming_dictionary_input", _streaming_dictionary_input())
        cursor = duckdb_cursor.execute(
            """
            SELECT
                i,
                category,
                repeat(chr(65 + (i % 26)::INTEGER), 256) || ':' || i::VARCHAR AS payload,
                CASE WHEN i % 17 = 0 THEN NULL ELSE i * 3 END AS nullable
            FROM streaming_dictionary_input
            ORDER BY i
            """
        )

        rows = _fetch_streaming_backpressure_rows(cursor, fetch_method, ("i", "category", "payload", "nullable"))
        expected = [
            (
                index,
                None if index % 19 == 0 else STREAMING_DICTIONARY_VALUES[index % len(STREAMING_DICTIONARY_VALUES)],
                _streaming_backpressure_expected_row(index)[1],
                _streaming_backpressure_expected_row(index)[2],
            )
            for index in range(STREAMING_BACKPRESSURE_ROW_COUNT)
        ]

        assert len(rows) == STREAMING_BACKPRESSURE_ROW_COUNT
        assert rows[0] == expected[0]
        assert rows[-1] == expected[-1]
        assert [row[0] for row in rows] == list(range(STREAMING_BACKPRESSURE_ROW_COUNT))
        assert [row[0] for row in rows if row[1] is None] == list(range(0, STREAMING_BACKPRESSURE_ROW_COUNT, 19))
        assert [row[0] for row in rows if row[3] is None] == list(range(0, STREAMING_BACKPRESSURE_ROW_COUNT, 17))
        assert _streaming_backpressure_hash(rows) == _streaming_backpressure_hash(expected)

    def test_fetch_one(self, duckdb_cursor):
        # fetch one
        res = duckdb_cursor.sql("SELECT * FROM range(100000)")
        result = []
        while len(result) < 5000:
            tpl = res.fetchone()
            result.append(tpl[0])
        assert result == list(range(5000))

        # fetch one with error
        res = duckdb_cursor.sql(
            "SELECT CASE WHEN i < 10000 THEN i ELSE concat('hello', i::VARCHAR)::INT END FROM range(100000) t(i)"
        )
        with pytest.raises(vane.ConversionException):
            res.fetchone()

    def test_fetch_many(self, duckdb_cursor):
        # fetch many
        res = duckdb_cursor.sql("SELECT * FROM range(100000)")
        result = []
        while len(result) < 5000:
            tpl = res.fetchmany(10)
            result += [x[0] for x in tpl]
        assert result == list(range(5000))

        # fetch many with error
        res = duckdb_cursor.sql(
            "SELECT CASE WHEN i < 10000 THEN i ELSE concat('hello', i::VARCHAR)::INT END FROM range(100000) t(i)"
        )
        with pytest.raises(vane.ConversionException):
            res.fetchmany(10)

    def test_record_batch_reader(self, duckdb_cursor):
        pytest.importorskip("pyarrow")
        pytest.importorskip("pyarrow.dataset")
        # record batch reader
        res = duckdb_cursor.sql("SELECT * FROM range(100000) t(i)")
        reader = res.to_arrow_reader(batch_size=16_384)
        result = []
        for batch in reader:
            result += batch.to_pydict()["i"]
        assert result == list(range(100000))

        # record batch reader with error
        res = duckdb_cursor.sql(
            "SELECT CASE WHEN i < 10000 THEN i ELSE concat('hello', i::VARCHAR)::INT END FROM range(100000) t(i)"
        )
        with pytest.raises(vane.ConversionException, match="Could not convert string 'hello10000' to INT32"):
            reader = res.to_arrow_reader(batch_size=16_384)

    def test_9801(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE test(id INTEGER , name VARCHAR NOT NULL);")

        words = ["aaaaaaaaaaaaaaaaaaaaaaa", "bbbb", "ccccccccc", "ííííííííí"]
        lines = [(i, words[i % 4]) for i in range(1000)]
        duckdb_cursor.executemany("INSERT INTO TEST (id, name) VALUES (?, ?)", lines)

        rel1 = duckdb_cursor.sql(
            """
            SELECT id, name FROM test ORDER BY id ASC
        """
        )
        result = rel1.fetchmany(size=5)
        counter = 0
        while result != []:
            for x in result:
                assert x == (counter, words[counter % 4])
                counter += 1
            result = rel1.fetchmany(size=5)

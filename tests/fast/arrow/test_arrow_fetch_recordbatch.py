# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import subprocess
import sys
import textwrap

import pytest

import vane

pa = pytest.importorskip("pyarrow")


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("export", ["reader", "capsule"])
def test_materialized_arrow_result_can_be_rescanned_on_source_connection(monkeypatch, threads, configured, export):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    script = textwrap.dedent(
        """
        import faulthandler
        import sys
        import pyarrow as pa
        import vane
        from vane.execution.request_admission import RequestAdmissionLimits

        threads, configured, export = sys.argv[1:]
        faulthandler.dump_traceback_later(15, exit=True)
        with vane.connect(config={"threads": int(threads)}) as con:
            if configured == "True":
                con.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1), execution_timeout=5)
            for count in [1, 25000]:
                # execute() materializes the relation before exporting it.
                relation = con.sql(f"SELECT i AS x FROM range({count}) r(i)").execute()
                if export == "reader":
                    reader = relation.to_arrow_reader(batch_size=128)
                else:
                    reader = pa.RecordBatchReader._import_from_c_capsule(relation.__arrow_c_stream__())
                if configured == "True":
                    # A configured runtime still rejects opaque input streams.
                    # Consuming the materialized output into a table is safe.
                    try:
                        con.register("rescan", reader)
                    except vane.InvalidInputException as error:
                        assert "does not support opaque Arrow readers" in str(error), str(error)
                    else:
                        raise AssertionError("configured input policy was bypassed")
                    con.register("rescan", reader.read_all())
                else:
                    con.register("rescan", reader)
                # PyArrow may produce batches on its own thread while this
                # query holds the connection lock. The stored rows need no lock.
                assert con.sql("SELECT count(*), sum(x) FROM rescan").fetchall() == [
                    (count, count * (count - 1) // 2)
                ]
                con.unregister("rescan")
                reader.close()
            assert con.execute("SELECT 42").fetchall() == [(42,)]
        faulthandler.cancel_dump_traceback_later()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(threads), str(configured), export],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("target", ["source", "sibling"])
def test_live_arrow_reader_rescan_rejects_nested_execution(monkeypatch, threads, target):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    script = textwrap.dedent(
        """
        import faulthandler
        import sys
        import vane

        threads, target = sys.argv[1:]
        faulthandler.dump_traceback_later(15, exit=True)
        with vane.connect(config={"threads": int(threads)}) as con, con.cursor() as sibling:
            reader = con.sql("SELECT i AS x FROM range(100000) r(i)").to_arrow_reader(batch_size=128)
            consumer = con if target == "source" else sibling
            consumer.register("rescan", reader)
            try:
                result = consumer.sql("SELECT count(*), sum(x) FROM rescan").fetchall()
            except vane.Error as error:
                assert "Python input callback" in str(error), str(error)
            else:
                raise AssertionError("live result executed inside an input callback")
            consumer.unregister("rescan")
            reader.close()
            assert con.execute("SELECT 42").fetchall() == [(42,)]
        faulthandler.cancel_dump_traceback_later()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(threads), target],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


class TestArrowFetchRecordBatch:
    # Test with basic numeric conversion (integers, floats, and others fall this code-path)
    def test_record_batch_next_batch_numeric(self, duckdb_cursor):
        duckdb_cursor = vane.connect()
        duckdb_cursor_check = vane.connect()
        duckdb_cursor.execute("CREATE table t as select range a from range(3000);")
        query = duckdb_cursor.execute("SELECT a FROM t")
        record_batch_reader = query.to_arrow_reader(1024)
        assert record_batch_reader.schema.names == ["a"]
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 1024
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 1024
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 952
        with pytest.raises(StopIteration):
            chunk = record_batch_reader.read_next_batch()
        # Check if we are producing the correct thing
        query = duckdb_cursor.execute("SELECT a FROM t")
        record_batch_reader = query.to_arrow_reader(1024)

        materialized = record_batch_reader.read_all()
        res = duckdb_cursor_check.from_arrow(materialized).fetchall()
        correct = duckdb_cursor.execute("select * from t").fetchall()
        assert res == correct

    # Test With Bool
    def test_record_batch_next_batch_bool(self, duckdb_cursor):
        duckdb_cursor = vane.connect()
        duckdb_cursor_check = vane.connect()
        duckdb_cursor.execute(
            "CREATE table t as SELECT CASE WHEN i % 2 = 0 THEN true ELSE false END AS a from range(3000) as tbl(i);"
        )
        query = duckdb_cursor.execute("SELECT a FROM t")
        record_batch_reader = query.to_arrow_reader(1024)
        assert record_batch_reader.schema.names == ["a"]
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 1024
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 1024
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 952
        with pytest.raises(StopIteration):
            chunk = record_batch_reader.read_next_batch()

        # Check if we are producing the correct thing
        query = duckdb_cursor.execute("SELECT a FROM t")
        record_batch_reader = query.to_arrow_reader(1024)

        materialized = record_batch_reader.read_all()
        res = duckdb_cursor_check.from_arrow(materialized).fetchall()
        correct = duckdb_cursor.execute("select * from t").fetchall()
        assert res == correct

    # Test with Varchar
    def test_record_batch_next_batch_varchar(self, duckdb_cursor):
        duckdb_cursor = vane.connect()
        duckdb_cursor_check = vane.connect()
        duckdb_cursor.execute("CREATE table t as select range::varchar a from range(3000);")
        query = duckdb_cursor.execute("SELECT a FROM t")
        record_batch_reader = query.to_arrow_reader(1024)
        assert record_batch_reader.schema.names == ["a"]
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 1024
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 1024
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 952
        with pytest.raises(StopIteration):
            chunk = record_batch_reader.read_next_batch()

        # Check if we are producing the correct thing
        query = duckdb_cursor.execute("SELECT a FROM t")
        record_batch_reader = query.to_arrow_reader(1024)

        materialized = record_batch_reader.read_all()
        res = duckdb_cursor_check.from_arrow(materialized).fetchall()
        correct = duckdb_cursor.execute("select * from t").fetchall()
        assert res == correct

    # Test with Struct
    def test_record_batch_next_batch_struct(self, duckdb_cursor):
        duckdb_cursor = vane.connect()
        duckdb_cursor_check = vane.connect()
        duckdb_cursor.execute(
            "CREATE table t as select {'x': i, 'y': i::varchar, 'z': i+1} as a from range(3000)  as tbl(i);"
        )
        query = duckdb_cursor.execute("SELECT a FROM t")
        record_batch_reader = query.to_arrow_reader(1024)
        assert record_batch_reader.schema.names == ["a"]
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 1024
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 1024
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 952
        with pytest.raises(StopIteration):
            chunk = record_batch_reader.read_next_batch()

        # Check if we are producing the correct thing
        query = duckdb_cursor.execute("SELECT a FROM t")
        record_batch_reader = query.to_arrow_reader(1024)

        materialized = record_batch_reader.read_all()
        res = duckdb_cursor_check.from_arrow(materialized).fetchall()
        correct = duckdb_cursor.execute("select * from t").fetchall()
        assert res == correct

    # Test with List
    def test_record_batch_next_batch_list(self, duckdb_cursor):
        duckdb_cursor = vane.connect()
        duckdb_cursor_check = vane.connect()
        duckdb_cursor.execute("CREATE table t as select [i,i+1] as a from range(3000)  as tbl(i);")
        query = duckdb_cursor.execute("SELECT a FROM t")
        record_batch_reader = query.to_arrow_reader(1024)
        assert record_batch_reader.schema.names == ["a"]
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 1024
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 1024
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 952
        with pytest.raises(StopIteration):
            chunk = record_batch_reader.read_next_batch()

        # Check if we are producing the correct thing
        query = duckdb_cursor.execute("SELECT a FROM t")
        record_batch_reader = query.to_arrow_reader(1024)

        materialized = record_batch_reader.read_all()
        res = duckdb_cursor_check.from_arrow(materialized).fetchall()
        correct = duckdb_cursor.execute("select * from t").fetchall()

        assert res == correct

    # Test with Map
    def test_record_batch_next_batch_map(self, duckdb_cursor):
        duckdb_cursor = vane.connect()
        duckdb_cursor_check = vane.connect()
        duckdb_cursor.execute("CREATE table t as select map([i], [i+1]) as a from range(3000)  as tbl(i);")
        query = duckdb_cursor.execute("SELECT a FROM t")
        record_batch_reader = query.to_arrow_reader(1024)
        assert record_batch_reader.schema.names == ["a"]
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 1024
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 1024
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 952
        with pytest.raises(StopIteration):
            chunk = record_batch_reader.read_next_batch()

        # Check if we are producing the correct thing
        query = duckdb_cursor.execute("SELECT a FROM t")
        record_batch_reader = query.to_arrow_reader(1024)

        materialized = record_batch_reader.read_all()
        res = duckdb_cursor_check.from_arrow(materialized).fetchall()
        correct = duckdb_cursor.execute("select * from t").fetchall()

        assert res == correct

    # Test with Null Values
    def test_record_batch_next_batch_with_null(self, duckdb_cursor):
        duckdb_cursor = vane.connect()
        duckdb_cursor_check = vane.connect()
        duckdb_cursor.execute(
            "CREATE table t as SELECT CASE WHEN i % 2 = 0 THEN i ELSE NULL END AS a from range(3000) as tbl(i);"
        )
        query = duckdb_cursor.execute("SELECT a FROM t")
        record_batch_reader = query.to_arrow_reader(1024)
        assert record_batch_reader.schema.names == ["a"]
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 1024
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 1024
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 952
        with pytest.raises(StopIteration):
            chunk = record_batch_reader.read_next_batch()

        # Check if we are producing the correct thing
        query = duckdb_cursor.execute("SELECT a FROM t")
        record_batch_reader = query.to_arrow_reader(1024)

        materialized = record_batch_reader.read_all()
        res = duckdb_cursor_check.from_arrow(materialized).fetchall()
        correct = duckdb_cursor.execute("select * from t").fetchall()

        assert res == correct

    def test_record_batch_read_default(self, duckdb_cursor):
        duckdb_cursor = vane.connect()
        duckdb_cursor.execute("CREATE table t as select range a from range(3000);")
        query = duckdb_cursor.execute("SELECT a FROM t")
        record_batch_reader = query.to_arrow_reader()
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 3000

    def test_record_batch_next_batch_multiple_vectors_per_chunk(self, duckdb_cursor):
        duckdb_cursor = vane.connect()
        duckdb_cursor.execute("CREATE table t as select range a from range(5000);")
        query = duckdb_cursor.execute("SELECT a FROM t")
        record_batch_reader = query.to_arrow_reader(2048)
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 2048
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 2048
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 904
        with pytest.raises(StopIteration):
            chunk = record_batch_reader.read_next_batch()

        query = duckdb_cursor.execute("SELECT a FROM t")
        record_batch_reader = query.to_arrow_reader(1)
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 1

        query = duckdb_cursor.execute("SELECT a FROM t")
        record_batch_reader = query.to_arrow_reader(2000)
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 2000

    def test_record_batch_next_batch_multiple_vectors_per_chunk_error(self, duckdb_cursor):
        duckdb_cursor = vane.connect()
        duckdb_cursor.execute("CREATE table t as select range a from range(5000);")
        query = duckdb_cursor.execute("SELECT a FROM t")
        with pytest.raises(RuntimeError, match="Approximate Batch Size of Record Batch MUST be higher than 0"):
            query.to_arrow_reader(0)
        with pytest.raises(TypeError, match="incompatible function arguments"):
            query.to_arrow_reader(-1)

    def test_record_batch_reader_from_relation(self, duckdb_cursor):
        duckdb_cursor = vane.connect()
        duckdb_cursor.execute("CREATE table t as select range a from range(3000);")
        relation = duckdb_cursor.table("t")
        record_batch_reader = relation.to_arrow_reader()
        chunk = record_batch_reader.read_next_batch()
        assert len(chunk) == 3000

    def test_record_coverage(self, duckdb_cursor):
        duckdb_cursor = vane.connect()
        duckdb_cursor.execute("CREATE table t as select range a from range(2048);")
        query = duckdb_cursor.execute("SELECT a FROM t")
        record_batch_reader = query.to_arrow_reader(1024)

        chunk = record_batch_reader.read_all()
        assert len(chunk) == 2048

    def test_record_batch_query_error(self):
        duckdb_cursor = vane.connect()
        duckdb_cursor.execute("CREATE table t as select 'foo' as a;")
        with pytest.raises(vane.ConversionException, match="Conversion Error"):
            # 'execute' materializes the result, causing the error directly
            duckdb_cursor.execute("SELECT cast(a as double) FROM t")

    def test_many_list_batches(self):
        conn = vane.connect()

        conn.execute(
            """
            create or replace table tbl as select * from (select {'a': [5,4,3,2,1]}), range(10000000)
        """
        )

        query = "SELECT * FROM tbl"
        chunk_size = 1_000_000

        # Because this produces multiple chunks, this caused a segfault before
        # because we changed some data in the first batch fetch
        batch_iter = conn.execute(query).to_arrow_reader(chunk_size)
        for batch in batch_iter:
            del batch

    def test_many_chunk_sizes(self):
        object_size = 1000000
        duckdb_cursor = vane.connect()
        query = duckdb_cursor.execute(f"CREATE table t as select range a from range({object_size});")
        for i in [1, 2, 4, 8, 16, 32, 33, 77, 999, 999999]:
            query = duckdb_cursor.execute("SELECT a FROM t")
            record_batch_reader = query.to_arrow_reader(i)
            num_loops = int(object_size / i)
            for _j in range(num_loops):
                assert record_batch_reader.schema.names == ["a"]
                chunk = record_batch_reader.read_next_batch()
                assert len(chunk) == i
            remainder = object_size % i
            if remainder > 0:
                chunk = record_batch_reader.read_next_batch()
                assert len(chunk) == remainder
            with pytest.raises(StopIteration):
                chunk = record_batch_reader.read_next_batch()

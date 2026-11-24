# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pytest

import vane
from vane.execution.request_admission import RequestAdmissionLimits


@pytest.fixture(autouse=True)
def local_fast(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")


def _run(script, *args):
    completed = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(script), *args],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("source", ["reader", "batches"])
@pytest.mark.parametrize("registration", ["direct", "parent_view", "closed_creator_view"])
@pytest.mark.parametrize("target", ["cursor", "parent"])
def test_async_dataset_readers_are_rejected(source, registration, target):
    _run(
        """
        import faulthandler
        import io
        import sys
        import threading
        import time
        import fsspec
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.fs as fs
        import pyarrow.parquet as pq
        import vane
        from vane.execution.request_admission import RequestAdmissionLimits

        faulthandler.dump_traceback_later(8, exit=True)
        source, registration, target = sys.argv[1:]
        query_thread = threading.get_ident()
        runtime = None
        armed = True
        attempts = []
        out = pa.BufferOutputStream()
        pq.write_table(pa.table({"x": range(1_000_000)}), out, row_group_size=2048)
        payload = out.getvalue().to_pybytes()

        class Reader(io.BytesIO):
            def read(self, size=-1):
                if threading.get_ident() != query_thread:
                    # Ensure prefetch overlaps the active query on an unfixed build.
                    time.sleep(0.2)
                    if armed and runtime is not None and not attempts and runtime.resource_snapshot()["request_admission"]["active_requests"]:
                        attempts.append(threading.get_ident())
                        (cursor if target == "cursor" else parent).close()
                return super().read(size)

        class Filesystem(fsspec.AbstractFileSystem):
            protocol = "opaqueinput"
            def info(self, path, **kwargs):
                return {"name": path, "size": len(payload), "type": "file"}
            def _open(self, path, mode="rb", **kwargs):
                return Reader(payload)

        filesystem = fs.PyFileSystem(fs.FSSpecHandler(Filesystem(skip_instance_cache=True)))
        dataset = ds.dataset("input.parquet", filesystem=filesystem, format="parquet")
        if source == "reader":
            data = dataset.scanner().to_reader()
        else:
            data = pa.RecordBatchReader.from_batches(dataset.schema, dataset.to_batches())
        with vane.connect(config={"threads": 4}) as parent:
            if registration == "parent_view":
                parent.from_arrow(data).create_view("shared_input")
            elif registration == "closed_creator_view":
                with parent.cursor() as creator:
                    creator.from_arrow(data).create_view("shared_input")
            runtime = parent.configure_local_runtime(
                request_limit=RequestAdmissionLimits(1, 1), execution_timeout=1
            )
            with parent.cursor() as cursor:
                try:
                    if registration == "direct":
                        cursor.from_arrow(data).aggregate("sum(x)").fetchall()
                    else:
                        cursor.execute("SELECT sum(x) FROM shared_input").fetchall()
                except vane.InvalidInputException as error:
                    assert "opaque Arrow readers or streams" in str(error), str(error)
                else:
                    raise AssertionError("asynchronous reader was accepted")
                armed = False
                assert not attempts, attempts
                state = runtime.resource_snapshot()["request_admission"]
                assert state["active_requests"] == state["cleanup_pending_requests"] == 0, state
                assert not state["draining"]
                # The caller still owns the rejected reader. Drain its prefetch
                # outside query execution before reuse or interpreter shutdown.
                materialized = data.read_all()
                data.close()
                assert cursor.from_arrow(materialized).aggregate("sum(x)").fetchall() == [(499999500000,)]
                assert cursor.execute("SELECT 7").fetchall() == [(7,)]
                assert parent.execute("SELECT 8").fetchall() == [(8,)]
        assert runtime.resource_snapshot()["closed"]
        faulthandler.cancel_dump_traceback_later()
        """,
        source,
        registration,
        target,
    )


@pytest.mark.parametrize("registration", ["direct", "parent_view", "closed_creator_view"])
@pytest.mark.parametrize("target", ["cursor", "parent"])
def test_polars_worker_callbacks_are_rejected(registration, target):
    pytest.importorskip("polars")
    _run(
        """
        import faulthandler
        import sys
        import threading
        import polars as pl
        import vane
        from vane.execution.request_admission import RequestAdmissionLimits

        faulthandler.dump_traceback_later(8, exit=True)
        registration, target = sys.argv[1:]
        runtime = None
        attempts = []
        collections = []
        query_thread = threading.get_ident()
        original_collect = pl.LazyFrame.collect
        def collect(self, *args, **kwargs):
            collections.append(True)
            return original_collect(self, *args, **kwargs)
        pl.LazyFrame.collect = collect

        def callback(values):
            if runtime is not None and not attempts and threading.get_ident() != query_thread and runtime.resource_snapshot()["request_admission"]["active_requests"]:
                attempts.append(threading.get_ident())
                (cursor if target == "cursor" else parent).close()
            return values[0].sum()

        source = pl.DataFrame({"g": [i % 8 for i in range(100_000)], "x": range(100_000)}).lazy().group_by("g").agg(
            pl.map_groups(["x"], callback, return_dtype=pl.Int64, returns_scalar=True)
        ).select("x")
        with vane.connect(config={"threads": 4}) as parent:
            if registration == "parent_view":
                parent.sql("SELECT * FROM source").create_view("shared_input")
            elif registration == "closed_creator_view":
                with parent.cursor() as creator:
                    creator.sql("SELECT * FROM source").create_view("shared_input")
            collections.clear()
            runtime = parent.configure_local_runtime(
                request_limit=RequestAdmissionLimits(1, 1), execution_timeout=1
            )
            with parent.cursor() as cursor:
                try:
                    if registration == "direct":
                        cursor.sql("SELECT sum(x) FROM source").fetchall()
                    else:
                        cursor.execute("SELECT sum(x) FROM shared_input").fetchall()
                except vane.InvalidInputException as error:
                    assert "Polars LazyFrame inputs" in str(error), str(error)
                else:
                    raise AssertionError("asynchronous LazyFrame was accepted")
                assert not attempts and not collections, (attempts, collections)
                state = runtime.resource_snapshot()["request_admission"]
                assert state["active_requests"] == state["cleanup_pending_requests"] == 0, state
                assert not state["draining"]
                # Materialize outside the runtime query, including Python UDF evaluation.
                materialized = source.collect()
                assert cursor.sql("SELECT sum(x) FROM materialized").fetchall() == [(4999950000,)]
                assert not attempts
                assert parent.execute("SELECT 8").fetchall() == [(8,)]
        assert runtime.resource_snapshot()["closed"]
        faulthandler.cancel_dump_traceback_later()
        """,
        registration,
        target,
    )


@pytest.mark.parametrize("source", ["reader", "capsule", "provider", "schema_provider"])
@pytest.mark.parametrize("entry", ["from_arrow", "register", "sql", "execute"])
def test_opaque_inputs_rejected_before_export_or_schema(source, entry):
    calls = []
    table = pa.table({"x": [1, 2, 3]})

    def batches():
        calls.append("next")
        yield from table.to_batches()

    reader = pa.RecordBatchReader.from_batches(table.schema, batches())

    class Provider:
        def __arrow_c_stream__(self, requested_schema=None):
            calls.append("stream")
            return reader.__arrow_c_stream__()

    class SchemaProvider(Provider):
        def __arrow_c_schema__(self):
            calls.append("schema")
            return table.schema.__arrow_c_schema__()

    data = {
        "reader": lambda: reader,
        "capsule": reader.__arrow_c_stream__,
        "provider": Provider,
        "schema_provider": SchemaProvider,
    }[source]()
    with vane.connect() as parent:
        runtime = parent.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1))
        with parent.cursor() as cursor:
            with pytest.raises(vane.InvalidInputException, match="opaque Arrow readers or streams"):
                if entry == "from_arrow":
                    cursor.from_arrow(data)
                elif entry == "register":
                    cursor.register("input", data)
                elif entry == "sql":
                    cursor.sql("SELECT * FROM data")
                else:
                    cursor.execute("SELECT * FROM data")
            assert not calls
            state = runtime.resource_snapshot()["request_admission"]
            assert state["active_requests"] == state["cleanup_pending_requests"] == 0
            assert cursor.from_arrow(table).aggregate("sum(x)").fetchall() == [(6,)]


def test_concurrent_cached_opaque_view_is_revalidated():
    table = pa.table({"x": [1, 2]})
    calls = []

    class Source:
        def __arrow_c_schema__(self):
            calls.append("schema")
            return table.schema.__arrow_c_schema__()

        def __arrow_c_stream__(self, requested_schema=None):
            calls.append("stream")
            return table.__arrow_c_stream__()

    with vane.connect() as parent:
        with parent.cursor() as creator:
            creator.from_arrow(Source()).create_view("shared_input")
        assert calls == ["schema"]
        calls.clear()
        runtime = parent.configure_local_runtime(request_limit=RequestAdmissionLimits(2, 1))
        with parent.cursor() as left, parent.cursor() as right, ThreadPoolExecutor(max_workers=2) as workers:
            futures = [workers.submit(cursor.execute, "SELECT * FROM shared_input") for cursor in (left, right)]
            for future in futures:
                with pytest.raises(vane.InvalidInputException, match="opaque Arrow readers or streams"):
                    future.result(timeout=5)
            assert not calls
            state = runtime.resource_snapshot()["request_admission"]
            assert state["active_requests"] == state["cleanup_pending_requests"] == 0
            assert left.execute("SELECT 7").fetchall() == [(7,)]
            assert right.execute("SELECT 8").fetchall() == [(8,)]


@pytest.mark.parametrize("source", ["table", "batch", "dataset", "union", "polars"])
@pytest.mark.parametrize("entry", ["relation", "sql"])
def test_materialized_inputs_remain_supported(source, entry):
    import pyarrow.dataset as ds

    table = pa.table({"x": [1, 2, 3]})
    data = table
    if source == "batch":
        data = table.to_batches()[0]
    elif source == "dataset":
        data = ds.dataset(table)
    elif source == "union":
        data = ds.UnionDataset(table.schema, [ds.dataset(table)])
    elif source == "polars":
        pl = pytest.importorskip("polars")
        data = pl.from_arrow(table)
    with vane.connect() as connection:
        connection.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1))
        if entry == "relation":
            relation = connection.from_arrow(data)
        else:
            relation = connection.sql("SELECT * FROM data")
        assert relation.filter("x > 1").aggregate("sum(x)").fetchall() == [(5,)]

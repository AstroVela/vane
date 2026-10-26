# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Fileset batches use native COPY results, never a post-write directory scan."""

from __future__ import annotations

import csv
import io
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Event
from urllib.parse import urlsplit

import pyarrow.parquet as pq
import pytest

import vane
from tests.fast.test_gravitino import (
    attach,
    load_gravitino,
    seed_fileset,
)
from tests.fast.test_gravitino import (
    gravitino_connection as gravitino_connection,
)
from tests.fast.test_gravitino import (
    gravitino_http as gravitino_http,
)
from vane.catalog import FilesetWriteResult, GravitinoCatalog
from vane.runners.copy_outcome import CopyOutcomeUnknownError, CopyResultUnavailableError


@pytest.fixture
def fileset_two_node_ray():
    import ray
    from ray._private.resource_and_label_spec import ResourceAndLabelSpec
    from ray.cluster_utils import Cluster

    from tests.ray_test_profile import ray_test_object_store_options

    cluster = Cluster(shutdown_at_exit=False)
    cleanup = {"pending": False}
    try:
        resources = ResourceAndLabelSpec().resolve(is_head=True)
        store = ray_test_object_store_options()
        store.setdefault("object_store_memory", resources.object_store_memory)
        cluster.add_node(num_cpus=2, include_dashboard=False, **store)
        cluster.add_node(num_cpus=2, include_dashboard=False, **store)
        ray.init(address=cluster.address, runtime_env={"env_vars": {"PYTHONPATH": os.environ.get("PYTHONPATH", "")}})
        yield cleanup
    finally:
        try:
            try:
                vane.teardown_runner()
            except RuntimeError:
                if not cleanup["pending"]:
                    raise
        finally:
            ray.shutdown()
            cluster.shutdown()


def configure_s3(connection, endpoint):
    for setting, value in {
        "http_proxy": "",
        "s3_endpoint": urlsplit(endpoint).netloc,
        "s3_access_key_id": "fileset-write-access",
        "s3_secret_access_key": "fileset-write-secret",
        "s3_region": "us-east-1",
        "s3_use_ssl": False,
        "s3_url_style": "path",
        "http_retries": 0,
    }.items():
        connection.execute(f"SET {setting} = ?", [value])


def read_literal_batch(connection, result):
    rows = []
    for file in result.files:
        with vane.open_file(file, "rb", connection=connection) as stream:
            contents = stream.read()
        if result.format == "parquet":
            table = pq.read_table(io.BytesIO(contents))
            assert table.column_names == ["i", "part"]
            rows.extend((row["i"], row["part"]) for row in table.to_pylist())
        else:
            records = list(csv.reader(io.StringIO(contents.decode("utf-8"))))
            assert records[0] == ["i", "part"]
            rows.extend((int(row[0]), row[1] or None) for row in records[1:])
    return sorted(rows)


@pytest.mark.parametrize("format", ["parquet", "csv"])
@pytest.mark.parametrize("empty", [False, True])
def test_fileset_batch_round_trip(gravitino_connection, gravitino_http, format, empty):
    state, endpoint, directory = gravitino_http
    catalog = seed_fileset(gravitino_connection, endpoint, directory)
    relation = gravitino_connection.sql("SELECT i, '中文,' || i::VARCHAR AS text FROM range(5) t(i)")
    if empty:
        relation = relation.filter("false")
    before = len(state["requests"])
    result = catalog.write_files(relation, "clips", "demo", "outputs", format=format)
    assert isinstance(result, FilesetWriteResult)
    assert result.location_name == "unknown"
    assert result.logical_path == "gvfs://fileset/media/clips/demo/outputs"
    assert result.rows_written == (0 if empty else 5)
    assert result.files
    assert all(Path(file).is_relative_to(Path(result.base_path)) for file in result.files)
    assert result.bytes_written == sum(Path(file).stat().st_size for file in result.files)
    assert result.run_id is None
    assert result.manifest_path is None
    assert result.cleanup_state == "complete"
    assert result.cleanup_warnings == ()
    with pytest.raises(FrozenInstanceError):
        result.rows_written = 99
    # Eager relation binding and execution reuse the frozen physical target.
    requests = list(state["requests"])[before:]
    assert len(requests) == 1
    assert requests[0][0] == "GET"
    if format == "parquet":
        restored = gravitino_connection.read_parquet(list(result.files))
    else:
        restored = gravitino_connection.read_csv(
            list(result.files), header=True, columns={"i": "BIGINT", "text": "VARCHAR"}
        )
    expected = [] if empty else [(i, f"中文,{i}") for i in range(5)]
    assert restored.order("i").fetchall() == expected
    assert (directory / "one.txt").read_text() == "first file"


def test_batch_calls_do_not_mix_or_replace_files(gravitino_connection, gravitino_http):
    _state, endpoint, directory = gravitino_http
    catalog = seed_fileset(gravitino_connection, endpoint, directory)
    first = catalog.write_files(gravitino_connection.sql("SELECT 1 AS n"), "clips", "demo")
    second = catalog.write_files(gravitino_connection.sql("SELECT 2 AS n"), "clips", "demo")
    assert first.operation_id != second.operation_id
    assert not set(first.files) & set(second.files)
    assert gravitino_connection.read_parquet(list(first.files)).fetchall() == [(1,)]
    assert gravitino_connection.read_parquet(list(second.files)).fetchall() == [(2,)]


def test_batch_evaluates_its_input_once(gravitino_connection, gravitino_http):
    _state, endpoint, directory = gravitino_http
    catalog = seed_fileset(gravitino_connection, endpoint, directory)
    gravitino_connection.execute("CREATE SEQUENCE input_counter")
    relation = gravitino_connection.sql("SELECT nextval('input_counter') AS n FROM range(4)")
    result = catalog.write_files(relation, "clips", "demo")
    assert result.rows_written == 4
    assert gravitino_connection.execute("SELECT currval('input_counter')").fetchone() == (4,)
    assert gravitino_connection.read_parquet(list(result.files)).order("n").fetchall() == [(1,), (2,), (3,), (4,)]


def test_failed_input_preserves_previous_batches(gravitino_connection, gravitino_http):
    state, endpoint, directory = gravitino_http
    catalog = seed_fileset(gravitino_connection, endpoint, directory)
    first = catalog.write_files(gravitino_connection.sql("SELECT 1 AS n"), "clips", "demo", "failures")
    before = len(state["requests"])
    with pytest.raises(vane.Error, match="injected source failure"):
        catalog.write_files(
            gravitino_connection.sql(
                "SELECT CASE WHEN i=1 THEN CAST(error('injected source failure ' || i::VARCHAR) AS BIGINT) "
                "ELSE i END AS n FROM range(3) AS t(i)"
            ),
            "clips",
            "demo",
            "failures",
        )
    assert len(state["requests"]) == before + 1
    assert gravitino_connection.read_parquet(list(first.files)).fetchall() == [(1,)]


def test_existing_parquet_writer_preserves_its_return_type(gravitino_connection, gravitino_http):
    _state, endpoint, directory = gravitino_http
    seed_fileset(gravitino_connection, endpoint, directory)
    result = gravitino_connection.sql("SELECT 3 AS n").write_parquet(
        "gvfs://fileset/media/clips/demo/existing-writer", append=True, per_thread_output=True
    )
    assert result is None
    assert gravitino_connection.read_parquet(str(directory / "existing-writer" / "*.parquet")).fetchall() == [(3,)]


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a/./b", "a\\b", "%2e%2e", "*.parquet", "[ab]", "x\0y"])
def test_invalid_batch_paths_do_not_write(gravitino_connection, gravitino_http, path):
    _state, endpoint, directory = gravitino_http
    catalog = seed_fileset(gravitino_connection, endpoint, directory)
    before = set(directory.rglob("*"))
    with pytest.raises((ValueError, vane.InvalidInputException)):
        catalog.write_files(gravitino_connection.sql("SELECT 1 AS n"), "clips", "demo", path)
    assert set(directory.rglob("*")) == before


def test_read_only_and_transaction_reject_batch(gravitino_connection, gravitino_http):
    _state, endpoint, directory = gravitino_http
    catalog = seed_fileset(gravitino_connection, endpoint, directory)
    read_only = attach(gravitino_connection, endpoint, "readonly", read_only=True)
    relation = gravitino_connection.sql("SELECT 1 AS n")
    before = set(directory.rglob("*"))
    with pytest.raises(vane.PermissionException, match="READ_ONLY"):
        read_only.write_files(relation, "clips", "demo")
    gravitino_connection.execute("BEGIN")
    try:
        with pytest.raises(vane.TransactionException, match="auto-commit"):
            catalog.write_files(relation, "clips", "demo")
    finally:
        gravitino_connection.execute("ROLLBACK")
    assert set(directory.rglob("*")) == before


@pytest.mark.parametrize("runner", ["local", "local-fast"])
def test_batch_supports_native_local_runner_aliases(gravitino_http, monkeypatch, runner):
    _state, endpoint, directory = gravitino_http
    monkeypatch.setenv("VANE_RUNNER", runner)
    with vane.connect() as connection:
        load_gravitino(connection)
        catalog = seed_fileset(connection, endpoint, directory)
        relation = connection.sql("SELECT 7 AS n")
        assert relation._get_runner_type() == "local"
        result = catalog.write_files(relation, "clips", "demo", "native-runner")
        assert result.rows_written == 1
        assert connection.read_parquet(list(result.files)).fetchall() == [(7,)]


def test_batch_rejects_a_different_connection(gravitino_connection, gravitino_http):
    _state, endpoint, directory = gravitino_http
    catalog = seed_fileset(gravitino_connection, endpoint, directory)
    with vane.connect() as other:
        with pytest.raises(vane.InvalidInputException, match="catalog's connection"):
            catalog.write_files(other.sql("SELECT 1 AS n"), "clips", "demo")


def test_missing_provider_does_not_write_a_local_gvfs_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with vane.connect() as connection:
        catalog = GravitinoCatalog(connection, "unattached")
        with pytest.raises(vane.Error, match="catalog|Catalog"):
            catalog.write_files(connection.sql("SELECT 1 AS n"), "clips", "demo")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("entrypoint", ["batch", "sql", "parquet"])
def test_ray_batch_rejects_node_local_output_before_dispatch(gravitino_http, monkeypatch, entrypoint):
    _state, endpoint, directory = gravitino_http
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    with vane.connect() as connection:
        load_gravitino(connection)
        catalog = seed_fileset(connection, endpoint, directory)
        with pytest.raises(vane.NotImplementedException, match="shared S3"):
            if entrypoint == "batch":
                catalog.write_files(connection.sql("SELECT 1 AS n"), "clips", "demo", "ray")
            elif entrypoint == "sql":
                connection.execute(
                    "COPY (SELECT 1 AS n) TO 'gvfs://fileset/media/clips/demo/ray' "
                    "(FORMAT parquet, APPEND, PER_THREAD_OUTPUT)"
                )
            else:
                connection.sql("SELECT 1 AS n").write_parquet(
                    "gvfs://fileset/media/clips/demo/ray", append=True, per_thread_output=True
                )
    assert not (directory / "ray").exists()


def test_bad_codec_fails_without_output(gravitino_connection, gravitino_http):
    _state, endpoint, directory = gravitino_http
    catalog = seed_fileset(gravitino_connection, endpoint, directory)
    with pytest.raises(vane.Error, match="[Cc]ompression|[Cc]odec"):
        catalog.write_files(
            gravitino_connection.sql("SELECT 1 AS n"), "clips", "demo", "bad-codec", compression="not-a-codec"
        )
    assert not (directory / "bad-codec").exists()


def test_literal_root_and_selected_location(gravitino_connection, gravitino_http):
    _state, endpoint, directory = gravitino_http
    root = directory / "literal*[a]?#%20"
    root.mkdir()
    catalog = attach(gravitino_connection, endpoint, location_name="hot")
    catalog.create_schema("clips")
    catalog.create_fileset("clips", "demo", storage_locations={"hot": str(root), "cold": str(directory / "cold")})
    result = catalog.write_files(gravitino_connection.sql("SELECT 42 AS n"), "clips", "demo")
    assert result.location_name == "hot"
    assert all(Path(file).is_relative_to(root) for file in result.files)
    assert gravitino_connection.read_parquet(list(result.files)).fetchall() == [(42,)]
    assert not (directory / "cold").exists()


@pytest.mark.skipif(os.name == "nt", reason="Backslashes are path separators on Windows")
@pytest.mark.parametrize("format", ["parquet", "csv"])
@pytest.mark.parametrize("path", ["", "nested/output"])
@pytest.mark.parametrize("root_exists", [False, True])
def test_local_batch_preserves_literal_backslash_root(gravitino_connection, gravitino_http, format, path, root_exists):
    _state, endpoint, directory = gravitino_http
    root = directory / r"literal\directory"
    if root_exists:
        root.mkdir()
    catalog = attach(gravitino_connection, endpoint)
    catalog.create_schema("clips")
    catalog.create_fileset("clips", "demo", storage_location=str(root))
    result = catalog.write_files(gravitino_connection.sql("SELECT 42 AS n"), "clips", "demo", path, format=format)
    assert Path(result.base_path).parent == root / path
    assert result.rows_written == 1
    assert result.bytes_written == sum(Path(file).stat().st_size for file in result.files)
    assert all(Path(file).is_relative_to(root / path) for file in result.files)
    assert not (directory / "literal").exists()
    for file in result.files:
        with vane.open_file(file, "rb", connection=gravitino_connection) as stream:
            contents = stream.read()
        if format == "parquet":
            assert pq.read_table(io.BytesIO(contents)).to_pylist() == [{"n": 42}]
        else:
            assert list(csv.reader(io.StringIO(contents.decode("utf-8")))) == [["n"], ["42"]]


def test_prepared_copy_rebinds_without_planning_writes(gravitino_connection, gravitino_http):
    state, endpoint, directory = gravitino_http
    catalog = seed_fileset(gravitino_connection, endpoint, directory)
    query = (
        "COPY (SELECT 7 AS n) TO 'gvfs://fileset/media/clips/demo/prepared' "
        "(FORMAT parquet, APPEND, PER_THREAD_OUTPUT, RETURN_STATS)"
    )
    gravitino_connection.execute("EXPLAIN " + query).fetchall()
    gravitino_connection.execute("PREPARE output AS " + query)
    assert not (directory / "prepared").exists()
    first = gravitino_connection.execute("EXECUTE output").fetchall()
    moved = directory / "moved"
    moved.mkdir()
    metadata = dict(state["filesets"][("clips", "demo")])
    metadata["storageLocations"] = {"unknown": str(moved)}
    state["filesets"][("clips", "demo")] = metadata
    second = gravitino_connection.execute("EXECUTE output").fetchall()
    assert all(Path(row[0]).is_relative_to(directory / "prepared") for row in first)
    assert all(Path(row[0]).is_relative_to(moved / "prepared") for row in second)
    assert catalog.load_fileset("clips", "demo")["storageLocations"] == {"unknown": str(moved)}


@pytest.mark.parametrize(
    "options",
    ["", ", OVERWRITE", ", APPEND", ", PER_THREAD_OUTPUT", ", APPEND, PER_THREAD_OUTPUT, FILENAME_PATTERN 'custom'"],
)
def test_unsupported_copy_modes_do_not_write(gravitino_connection, gravitino_http, options):
    _state, endpoint, directory = gravitino_http
    seed_fileset(gravitino_connection, endpoint, directory)
    with pytest.raises(vane.NotImplementedException):
        gravitino_connection.execute(
            "COPY (SELECT 1 AS n) TO 'gvfs://fileset/media/clips/demo/rejected' (FORMAT parquet" + options + ")"
        )
    assert not (directory / "rejected").exists()


def test_result_conversion_failure_does_not_resubmit(gravitino_connection, gravitino_http, monkeypatch):
    import vane.catalog as catalog_module

    state, endpoint, directory = gravitino_http
    catalog = seed_fileset(gravitino_connection, endpoint, directory)

    def broken_result(**_kwargs):
        raise RuntimeError("injected conversion failure")

    monkeypatch.setattr(catalog_module, "FilesetWriteResult", broken_result)
    with pytest.raises(CopyResultUnavailableError) as error:
        catalog.write_files(gravitino_connection.sql("SELECT 9 AS n"), "clips", "demo", "conversion")
    assert error.value.write_state == "committed"
    assert error.value.safe_to_retry is False
    batches = list((directory / "conversion").iterdir())
    assert len(batches) == 1
    assert Path(error.value.base_path) == batches[0]
    assert gravitino_connection.read_parquet(str(batches[0] / "*.parquet")).fetchall() == [(9,)]


@pytest.mark.parametrize("format", ["parquet", "csv"])
@pytest.mark.parametrize("root", ["output", "literal[1]%20#output"])
def test_s3_batch_round_trip(gravitino_connection, gravitino_http, format, root):
    state, endpoint, _directory = gravitino_http
    connection = gravitino_connection
    configure_s3(connection, endpoint)
    catalog = attach(connection, endpoint)
    catalog.create_schema("clips")
    catalog.create_fileset("clips", "objects", storage_location="s3a://bucket/" + root)
    result = catalog.write_files(connection.sql("SELECT i FROM range(10) t(i)"), "clips", "objects", format=format)
    assert result.rows_written == 10
    assert result.files
    assert result.bytes_written == sum(
        len(state["objects"][file.removeprefix("s3://bucket/")]) for file in result.files
    )
    if root == "output":
        restored = (
            connection.read_parquet(list(result.files))
            if format == "parquet"
            else connection.read_csv(list(result.files), header=True)
        )
        assert restored.order("i").fetchall() == [(i,) for i in range(10)]
    else:
        # Exact file paths can contain glob metacharacters. Open these as
        # literals rather than passing them back to a glob-expanding reader.
        values = []
        for file in result.files:
            assert file.startswith("s3://bucket/" + root + "/")
            with vane.open_file(file, "rb", connection=connection) as stream:
                contents = stream.read()
            if format == "parquet":
                values.extend(pq.read_table(io.BytesIO(contents)).column("i").to_pylist())
            else:
                rows = list(csv.reader(io.StringIO(contents.decode("utf-8"))))
                assert rows[0] == ["i"]
                values.extend(int(row[0]) for row in rows[1:])
        assert sorted(values) == list(range(10))
    assert any(method == "PUT" and "fileset-write-access/" in auth for method, auth in state["object_requests"])


def test_concurrent_connections_produce_distinct_batches(gravitino_connection, gravitino_http):
    _state, endpoint, directory = gravitino_http
    seed_fileset(gravitino_connection, endpoint, directory)

    def write(value):
        with vane.connect() as connection:
            load_gravitino(connection)
            catalog = attach(connection, endpoint)
            return catalog.write_files(connection.sql(f"SELECT {value} AS n"), "clips", "demo", "concurrent")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = executor.map(write, [1, 2])
    assert first.operation_id != second.operation_id
    assert not set(first.files) & set(second.files)
    assert gravitino_connection.read_parquet(list(first.files + second.files)).order("n").fetchall() == [(1,), (2,)]


@pytest.mark.parametrize("format", ["parquet", "csv"])
@pytest.mark.parametrize("empty", [False, True])
def test_partitioned_local_batch_keeps_columns_and_empty_schema(gravitino_connection, gravitino_http, format, empty):
    state, endpoint, directory = gravitino_http
    catalog = seed_fileset(gravitino_connection, endpoint, directory)
    relation = gravitino_connection.sql(
        "SELECT i, CASE i WHEN 0 THEN '../escape' WHEN 1 THEN 'a/b' WHEN 2 THEN '中文=%' "
        "WHEN 3 THEN '__HIVE_DEFAULT_PARTITION__' ELSE NULL END AS part FROM range(5) t(i)"
    )
    if empty:
        relation = relation.filter("false")
    before = len(state["requests"])
    result = catalog.write_files(relation, "clips", "demo", "partitioned", format=format, partition_by=("part",))
    expected = (
        [] if empty else [(0, "../escape"), (1, "a/b"), (2, "中文=%"), (3, "__HIVE_DEFAULT_PARTITION__"), (4, None)]
    )
    assert len(state["requests"]) == before + 1
    assert result.rows_written == len(expected)
    assert result.bytes_written == sum(Path(file).stat().st_size for file in result.files)
    assert result.files
    assert all(Path(file).resolve().is_relative_to(Path(result.base_path)) for file in result.files)
    assert read_literal_batch(gravitino_connection, result) == expected
    restored = (
        gravitino_connection.read_parquet(list(result.files), hive_partitioning=False)
        if format == "parquet"
        else gravitino_connection.read_csv(
            list(result.files), header=True, hive_partitioning=False, columns={"i": "BIGINT", "part": "VARCHAR"}
        )
    )
    assert restored.order("i").fetchall() == expected
    if empty:
        assert len(result.files) == 1
        assert Path(result.files[0]).parent == Path(result.base_path)


@pytest.mark.parametrize("partition_by", ["i", b"i", 1, [""], [None], ["i", "I"], ["missing"]])
def test_invalid_partition_columns_do_not_create_output(gravitino_connection, gravitino_http, partition_by):
    _state, endpoint, directory = gravitino_http
    catalog = seed_fileset(gravitino_connection, endpoint, directory)
    with pytest.raises((ValueError, TypeError, vane.Error)):
        catalog.write_files(
            gravitino_connection.sql("SELECT 1 AS i"), "clips", "demo", "bad-partition", partition_by=partition_by
        )
    assert not (directory / "bad-partition").exists()


def test_partition_by_all_columns(gravitino_connection, gravitino_http):
    _state, endpoint, directory = gravitino_http
    catalog = seed_fileset(gravitino_connection, endpoint, directory)
    result = catalog.write_files(gravitino_connection.sql("SELECT 1 AS i"), "clips", "demo", partition_by=["i"])
    assert gravitino_connection.read_parquet(list(result.files), hive_partitioning=False).fetchall() == [(1,)]


def test_empty_raw_partition_option_cannot_replace_a_file(gravitino_connection, gravitino_http):
    _state, endpoint, directory = gravitino_http
    seed_fileset(gravitino_connection, endpoint, directory)
    with pytest.raises(vane.InvalidInputException, match="PARTITION_BY requires at least one column"):
        gravitino_connection.sql("SELECT 1 AS n").write_parquet(
            "gvfs://fileset/media/clips/demo/one.txt", append=True, partition_by=[]
        )
    assert (directory / "one.txt").read_text() == "first file"


@pytest.mark.real_ray
@pytest.mark.parametrize("format", ["parquet", "csv"])
@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_ray_s3_batches_use_committed_files_without_metadata_on_workers(
    ray_local, gravitino_http, monkeypatch, format, partitioned, empty
):
    from vane.runners.ray.runner import RayRunner

    monkeypatch.delenv("VANE_RUNNER", raising=False)
    state, endpoint, _directory = gravitino_http
    run_write = RayRunner.run_write
    calls = []

    def capture(runner, plan):
        calls.append(plan.idx())
        payload = pickle.dumps(plan)
        assert b"fixture-token" not in payload
        assert b"gvfs://" not in payload
        assert plan.__getstate__()[3]["attached_databases"] == []
        # Any rebind or worker metadata request will now fail. S3 stays live.
        state["fault"] = "invalid-json"
        return run_write(runner, pickle.loads(payload))

    monkeypatch.setattr(RayRunner, "run_write", capture)
    with vane.connect() as connection:
        load_gravitino(connection)
        configure_s3(connection, endpoint)
        catalog = attach(connection, endpoint)
        catalog.create_schema("clips")
        catalog.create_fileset("clips", "objects", storage_location="s3a://bucket/literal[1]%20#output")
        relation = connection.sql(
            "SELECT i, CASE i%4 WHEN 0 THEN '../escape' WHEN 1 THEN 'a/b' WHEN 2 THEN '中文=%' "
            "ELSE NULL END AS part FROM range(21) t(i)"
        )
        if empty:
            relation = relation.filter("false")
        before = len(state["requests"])
        result = catalog.write_files(
            relation, "clips", "objects", format=format, partition_by=["part"] if partitioned else None
        )
        assert calls == [result.operation_id]
        assert len(state["requests"]) == before + 1
        assert result.run_id
        assert result.manifest_path
        committed = vane.ray_cxx.read_committed_copy_direct_write_result(result.base_path, result.run_id, connection)
        assert result.files == tuple(sorted(file["final_path"] for file in committed["files"]))
        assert len(set(result.files)) == len(result.files) > 0
        expected = [] if empty else [(i, ["../escape", "a/b", "中文=%", None][i % 4]) for i in range(21)]
        assert result.rows_written == len(expected)
        assert result.bytes_written == sum(
            len(state["objects"][file.removeprefix("s3://bucket/")]) for file in result.files
        )
        assert read_literal_batch(connection, result) == expected


@pytest.mark.parametrize("failure", ["unknown", "cancel", "result"])
def test_ray_batch_failure_never_resubmits(gravitino_http, monkeypatch, failure):
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    state, endpoint, _directory = gravitino_http
    calls = []

    class FakeRayRunner:
        def run_write(self, plan):
            calls.append(plan.idx())
            if failure == "unknown":
                raise CopyOutcomeUnknownError(plan.idx(), "s3://bucket/output", "run", "manifest", "marker")
            if failure == "cancel":
                raise KeyboardInterrupt()
            return {
                "copy_operation_id": plan.idx(),
                "rows_copied": 1,
                "copy_output_base_path": "s3://bucket/output",
                "copy_output_run_id": "run",
                "copy_output_manifest_path": "manifest",
                "copy_cleanup_state": "pending",
                "copy_cleanup_warnings": ["injected cleanup diagnostic"],
            }

    def unavailable(*_args):
        raise RuntimeError("injected manifest result failure")

    monkeypatch.setattr(vane._native, "set_runner_ray", lambda *_args, **_kwargs: FakeRayRunner())
    monkeypatch.setattr(vane.ray_cxx, "read_committed_copy_direct_write_result", unavailable)
    with vane.connect() as connection:
        load_gravitino(connection)
        catalog = attach(connection, endpoint)
        catalog.create_schema("clips")
        catalog.create_fileset("clips", "objects", storage_location="s3://bucket/output")
        before = len(state["requests"])
        error_type = {
            "unknown": CopyOutcomeUnknownError,
            "cancel": KeyboardInterrupt,
            "result": CopyResultUnavailableError,
        }[failure]
        with pytest.raises(error_type) as raised:
            catalog.write_files(connection.sql("SELECT 1 AS i"), "clips", "objects")
        assert len(calls) == 1
        assert len(state["requests"]) == before + 1
        if failure != "cancel":
            assert raised.value.safe_to_retry is False
            assert raised.value.operation_id == calls[0]
            assert raised.value.base_path == "s3://bucket/output"
            restored = pickle.loads(pickle.dumps(raised.value))
            assert restored.run_id == "run"
            assert restored.manifest_path == "manifest"
            if failure == "result":
                assert restored.cleanup_warnings == ("injected cleanup diagnostic",)


@pytest.mark.real_ray
@pytest.mark.parametrize("fault", ["manifest-write", "marker-unknown", "result-read"])
def test_ray_publication_failures_preserve_batches_and_recovery_identity(ray_local, gravitino_http, monkeypatch, fault):
    from vane.runners.ray.runner import RayRunner

    monkeypatch.delenv("VANE_RUNNER", raising=False)
    state, endpoint, _directory = gravitino_http
    calls = []
    run_write = RayRunner.run_write

    def capture(runner, plan):
        calls.append(plan.idx())
        return run_write(runner, plan)

    monkeypatch.setattr(RayRunner, "run_write", capture)
    with vane.connect() as connection:
        load_gravitino(connection)
        configure_s3(connection, endpoint)
        catalog = attach(connection, endpoint)
        catalog.create_schema("clips")
        catalog.create_fileset("clips", "objects", storage_location="s3://bucket/recovery")
        first = catalog.write_files(connection.sql("SELECT 1 AS i, 'first' AS part"), "clips", "objects")
        state["object_fault"] = fault
        before = len(state["requests"])
        error_type = {
            "manifest-write": Exception,
            "marker-unknown": CopyOutcomeUnknownError,
            "result-read": CopyResultUnavailableError,
        }[fault]
        with pytest.raises(error_type) as raised:
            catalog.write_files(connection.sql("SELECT i, 'second' AS part FROM range(5) t(i)"), "clips", "objects")
        assert len(calls) == 2
        assert len(state["requests"]) == before + 1
        state["object_fault"] = None
        assert read_literal_batch(connection, first) == [(1, "first")]
        if fault != "manifest-write":
            error = raised.value
            assert error.operation_id == calls[-1]
            assert error.safe_to_retry is False
            recovered = vane.ray_cxx.read_committed_copy_direct_write_result(error.base_path, error.run_id, connection)
            assert recovered["rows_copied"] == 5
            assert recovered["copy_output_manifest_path"] == error.manifest_path
            result = FilesetWriteResult(
                logical_path="gvfs://fileset/media/clips/objects/",
                location_name="unknown",
                format="parquet",
                files=tuple(file["final_path"] for file in recovered["files"]),
                rows_written=5,
                bytes_written=sum(file["file_size_bytes"] for file in recovered["files"]),
                operation_id=error.operation_id,
                base_path=error.base_path,
                run_id=error.run_id,
                manifest_path=error.manifest_path,
            )
            assert read_literal_batch(connection, result) == [(i, "second") for i in range(5)]


@pytest.mark.real_ray
def test_concurrent_ray_batches_do_not_mix_committed_files(ray_local, gravitino_http, monkeypatch):
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    state, endpoint, _directory = gravitino_http
    with vane.connect() as connection:
        load_gravitino(connection)
        configure_s3(connection, endpoint)
        catalog = attach(connection, endpoint)
        catalog.create_schema("clips")
        catalog.create_fileset("clips", "objects", storage_location="s3://bucket/concurrent")
        before = len(state["requests"])

        def write(value):
            with vane.connect() as writer:
                load_gravitino(writer)
                configure_s3(writer, endpoint)
                target = attach(writer, endpoint)
                return target.write_files(
                    writer.sql(f"SELECT i, '{value}' AS part FROM range(9) t(i)"),
                    "clips",
                    "objects",
                    partition_by=["part"],
                )

        with ThreadPoolExecutor(max_workers=2) as executor:
            first, second = executor.map(write, ["first", "second"])
        # Each connection attaches once and resolves its Fileset once.
        requests = list(state["requests"])[before:]
        assert len([item for item in requests if item[1].endswith("/filesets/objects")]) == 2
        assert first.operation_id != second.operation_id
        assert first.run_id != second.run_id
        assert not set(first.files) & set(second.files)
        assert read_literal_batch(connection, first) == [(i, "first") for i in range(9)]
        assert read_literal_batch(connection, second) == [(i, "second") for i in range(9)]


@pytest.mark.real_ray
@pytest.mark.parametrize("action", [pytest.param("worker-loss", marks=pytest.mark.ray_cluster_owner), "interrupt"])
def test_ray_interrupted_attempts_preserve_selected_batch(request, gravitino_http, monkeypatch, action):
    import ray

    from vane.runners.ray.runner import RayRunner

    cleanup = request.getfixturevalue("fileset_two_node_ray" if action == "worker-loss" else "ray_local")
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    state, endpoint, _directory = gravitino_http
    calls = []
    runtime = {}
    run_write = RayRunner.run_write
    interrupt_seen = Event()
    if action == "interrupt":
        from vane.runners.ray import safe_get

        check = safe_get.check_query_interrupted

        def observe_interrupt():
            try:
                check()
            except vane.InterruptException:
                interrupt_seen.set()
                raise

        monkeypatch.setattr(safe_get, "check_query_interrupted", observe_interrupt)

    def capture(runner, plan):
        calls.append(plan.idx())
        client = runner._client_for_session(str(plan.session_id()))
        runtime["driver"] = client.runner
        return run_write(runner, plan)

    monkeypatch.setattr(RayRunner, "run_write", capture)
    connection = vane.connect()
    completed = False
    result = None
    try:
        load_gravitino(connection)
        configure_s3(connection, endpoint)
        catalog = attach(connection, endpoint)
        catalog.create_schema("clips")
        catalog.create_fileset("clips", "objects", storage_location="s3://bucket/attempts")
        first = catalog.write_files(connection.sql("SELECT 1 AS i, 'prior' AS part"), "clips", "objects")
        relation = connection.sql("SELECT i, 'current' AS part FROM range(21) t(i)")
        state["object_fault"] = "pause-data"
        before = len(state["requests"])
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(catalog.write_files, relation, "clips", "objects")
            try:
                deadline = time.monotonic() + 30
                while not state["object_blocked"] and time.monotonic() < deadline and not future.done():
                    time.sleep(0.02)
                assert state["object_blocked"], "the COPY attempt never reached its output file"
                blocked = "s3://bucket/" + state["object_blocked"]
                if action == "worker-loss":

                    def kill_native_worker(_driver, query_id):
                        import ray

                        from vane.runners.ray.fragment_registry import _FTE_PARTITION_OWNERS, _FTE_REGISTRY_LOCK

                        with _FTE_REGISTRY_LOCK:
                            owners = {
                                handle.worker_id: handle
                                for key, handle in _FTE_PARTITION_OWNERS.items()
                                if key[0] == query_id
                            }
                        assert len(owners) == 1, owners.keys()
                        owner = next(iter(owners.values()))
                        ray.kill(owner.actor_handle, no_restart=True)
                        return owner.worker_id

                    assert ray.get(runtime["driver"].__ray_call__.remote(kill_native_worker, calls[-1]), timeout=15)
                else:
                    connection.interrupt()
                    assert interrupt_seen.wait(5), "the connection interruption was never observed"
                state["fault"] = "invalid-json"
            finally:
                state["object_fault"] = None
            if action == "worker-loss":
                result = future.result(timeout=120)
                assert blocked not in result.files
                assert result.rows_written == 21
                assert read_literal_batch(connection, result) == [(i, "current") for i in range(21)]
                if result.cleanup_state == "pending":
                    assert result.cleanup_warnings
                    cleanup["pending"] = True
            else:
                try:
                    result = future.result(timeout=120)
                except (vane.InterruptException, CopyOutcomeUnknownError, CopyResultUnavailableError) as error:
                    if hasattr(error, "safe_to_retry"):
                        assert error.safe_to_retry is False
                else:
                    # A cancellation request can race a completed commit.
                    assert result.manifest_path
                    assert result.rows_written == 21
                    assert read_literal_batch(connection, result) == [(i, "current") for i in range(21)]
        assert len(calls) == 2
        assert len(state["requests"]) == before + 1
        assert read_literal_batch(connection, first) == [(1, "prior")]
        completed = True
    finally:
        try:
            connection.close()
        except RuntimeError:
            # The committed receipt reports this independently of COPY success.
            # The owner fixture then shuts down its remaining faulted runtime.
            if not (completed and action == "worker-loss" and result and result.cleanup_state == "pending"):
                raise

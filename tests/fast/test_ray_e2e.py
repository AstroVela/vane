# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
import uuid
import warnings
from urllib.parse import urlparse

import pytest

try:
    import ray
except Exception:
    ray = None

from ray_test_profile import ray_test_object_store_bytes

import vane
from vane import runners as _runners

RAY_E2E_ROWS = 1000


@pytest.fixture(autouse=True)
def _vane_shuffle_env(monkeypatch):
    monkeypatch.setenv("VANE_SHUFFLE_ALGORITHM", "flight_shuffle")
    monkeypatch.setenv("VANE_SHUFFLE_LOCAL_DIRS", "/tmp/duckdb_shuffle")
    monkeypatch.setenv("RAY_DEDUP_LOGS", "0")
    # The default shared test profile reserves a 2 GiB object store. Keep FTE's
    # production block target, while sizing UDF windows so multi-actor tests
    # fit the query's 50% allocation and wide-row batches remain whole.
    monkeypatch.setenv("VANE_UDF_TARGET_MAX_BATCH_BYTES", str(16 * 1024**2))


def _minio_shuffle_config():
    endpoint = os.getenv("TEST_MINIO_ENDPOINT") or os.getenv("AWS_ENDPOINT_URL") or "http://127.0.0.1:9000"
    access_key = os.getenv("TEST_MINIO_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_ID") or ""
    secret_key = os.getenv("TEST_MINIO_SECRET_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY") or ""
    region = os.getenv("TEST_MINIO_REGION") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
    bucket = os.getenv("TEST_MINIO_BUCKET") or "vane-shuffle-test"
    base_uri = f"s3://{bucket}/flight-exchange-minio-e2e/{uuid.uuid4()}"
    return endpoint, access_key, secret_key, region, bucket, base_uri


def _configure_conn_for_s3(conn, endpoint, access_key, secret_key, region):
    parsed = urlparse(endpoint)
    duckdb_endpoint = parsed.netloc or parsed.path
    conn.execute("LOAD httpfs")
    conn.execute(f"SET s3_endpoint='{duckdb_endpoint}'")
    conn.execute(f"SET s3_use_ssl={'true' if parsed.scheme == 'https' else 'false'}")
    conn.execute("SET s3_url_style='path'")
    conn.execute(f"SET s3_region='{region}'")
    conn.execute(f"SET s3_access_key_id='{access_key}'")
    conn.execute(f"SET s3_secret_access_key='{secret_key}'")
    conn.execute("SET http_proxy=''")


def _skip_unless_minio_writable(endpoint, access_key, secret_key, region, bucket):
    probe_path = f"s3://{bucket}/flight-exchange-minio-preflight/{uuid.uuid4()}/probe.parquet"
    conn = vane.connect()
    try:
        _configure_conn_for_s3(conn, endpoint, access_key, secret_key, region)
        conn.execute(f"COPY (SELECT 1 AS value) TO '{probe_path}' (FORMAT PARQUET)")
        assert conn.execute(f"SELECT value FROM read_parquet('{probe_path}')").fetchone()[0] == 1
    except Exception as exc:
        pytest.skip(f"MinIO/S3-compatible endpoint is not writable for this test: {exc}")
    finally:
        conn.close()


def _explain_text(con, sql):
    explain_rows = con.sql("EXPLAIN " + sql).fetchall()
    return "\n".join(
        row[1] if isinstance(row, tuple) and len(row) > 1 else row[0] if isinstance(row, tuple) else str(row)
        for row in explain_rows
    )


def _assert_explain_contains(explain_text, *, require_any=None, require_all=None, label=""):
    explain_upper = explain_text.upper()
    if require_all:
        for keyword in require_all:
            assert keyword in explain_upper, f"{label}: expected {keyword} in EXPLAIN"
    if require_any:
        assert any(keyword in explain_upper for keyword in require_any), (
            f"{label}: expected one of {require_any} in EXPLAIN"
        )


def _log_builder_info(df):
    try:
        getattr(df, "_builder", None)
    except Exception:
        pass


def _get_distributed_plan_info(df, label):
    ray_cxx = getattr(vane, "ray_cxx", None)
    if ray_cxx is None or not hasattr(ray_cxx, "PyLogicalPlan"):
        return None, None
    try:
        logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(df, f"{label}-plan")
        conn = vane.connect()
        plan = logical_plan.to_physical_plan(conn)
    except Exception:
        return None, None
    plan_text = ""
    try:
        plan_text = plan.repr_ascii(False)
    except Exception:
        pass
    num_parts = None
    try:
        num_parts = plan.num_partitions()
    except Exception:
        pass
    return plan_text, num_parts


def _log_distributed_plan(df, label):
    plan_text, num_parts = _get_distributed_plan_info(df, label)
    if plan_text:
        pass
    else:
        pass
    if num_parts is not None:
        pass


def _get_row_count(part):
    try:
        if hasattr(part, "_table"):
            rows = getattr(part._table, "num_rows", None)
            if rows is not None:
                return rows
        if hasattr(part, "__len__"):
            return len(part)
        if hasattr(part, "num_rows"):
            return part.num_rows()
        if hasattr(part, "get_num_rows"):
            return part.get_num_rows()
    except Exception as exc:
        return f"error: {exc}"
    return "unknown"


def _get_schema_info(part):
    try:
        if hasattr(part, "schema"):
            schema = part.schema()
            num_cols = len(schema) if hasattr(schema, "__len__") else "unknown"
            return num_cols, str(schema)
    except Exception as exc:
        return "error", f"error: {exc}"
    return "unknown", "unknown"


def _log_partitions(parts):
    for _, part in enumerate(parts):
        rows = _get_row_count(part)
        _get_schema_info(part)

        if hasattr(part, "_table"):
            try:
                table = part._table
                if hasattr(table, "schema"):
                    pass
                if hasattr(table, "to_pandas"):
                    table.to_pandas()
                elif hasattr(table, "to_pylist"):
                    table.to_pylist()
            except Exception:
                pass

        if isinstance(rows, int) and rows > 0:
            try:
                if hasattr(part, "head"):
                    part.head(min(5, rows))
                elif hasattr(part, "to_pandas"):
                    part.to_pandas()
                elif hasattr(part, "to_arrow"):
                    arrow_table = part.to_arrow()
                    if hasattr(arrow_table, "to_pandas"):
                        arrow_table.to_pandas()
            except Exception:
                pass


def _stringify_value(value):
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _rows_from_pylist(rows, column_names=None):
    if not rows:
        return [], column_names
    if isinstance(rows[0], dict):
        if column_names is None:
            column_names = list(rows[0].keys())
        return [tuple(row.get(col) for col in column_names) for row in rows], column_names
    return [tuple(row) for row in rows], column_names


def _rows_from_partition(part, column_names=None):
    if hasattr(part, "to_pylist"):
        rows = part.to_pylist()
        if column_names is None and hasattr(part, "column_names"):
            try:
                column_names = list(part.column_names())
            except Exception:
                column_names = None
        return _rows_from_pylist(rows, column_names)
    if hasattr(part, "to_arrow"):
        table = part.to_arrow()
        if column_names is None and hasattr(table, "schema"):
            column_names = list(table.schema.names)
        if hasattr(table, "to_pylist"):
            rows = table.to_pylist()
            return _rows_from_pylist(rows, column_names)
    if hasattr(part, "_table"):
        table = part._table
        if column_names is None and hasattr(table, "schema"):
            column_names = list(table.schema.names)
        if hasattr(table, "to_pylist"):
            rows = table.to_pylist()
            return _rows_from_pylist(rows, column_names)
    if isinstance(part, list):
        return _rows_from_pylist(part, column_names)
    return [], column_names


def _collect_result_rows(parts):
    all_rows = []
    column_names = None
    for part in parts:
        rows, column_names = _rows_from_partition(part, column_names)
        all_rows.extend(rows)
    return [tuple(_stringify_value(val) for val in row) for row in all_rows]


def _expected_result_rows(con, sql):
    expected = con.execute(sql).fetchall()
    return [tuple(_stringify_value(val) for val in row) for row in expected]


def _assert_results_match(con, sql, parts, label, *, ordered=False):
    actual_rows = _collect_result_rows(parts)
    expected_rows = _expected_result_rows(con, sql)
    assert len(actual_rows) == len(expected_rows), (
        f"{label}: row count mismatch, actual={len(actual_rows)} expected={len(expected_rows)}"
    )
    if ordered:
        assert actual_rows == expected_rows, f"{label}: result rows do not match expected order"
    else:
        assert sorted(actual_rows) == sorted(expected_rows), f"{label}: result rows do not match expected values"


def _run_iter_tables(runner, builder, label, timeout_s=25.0, results_buffer_size=1):
    start = time.time()
    try:
        parts = list(runner.run_iter_tables(builder, results_buffer_size=results_buffer_size))
        elapsed = time.time() - start
        _log_partitions(parts)
    except Exception:
        elapsed = time.time() - start
        assert elapsed < timeout_s, f"{label}: took too long ({elapsed:.2f}s), scheduler_loop may be stuck"
        raise

    assert elapsed < timeout_s, f"{label}: took too long ({elapsed:.2f}s), scheduler_loop may be stuck"
    return parts


def _run_query_case(
    con,
    runner,
    sql,
    label,
    *,
    require_any=None,
    require_all=None,
    timeout_s=25.0,
    ordered=False,
):
    df = con.sql(sql)
    explain_text = _explain_text(con, sql)
    _assert_explain_contains(explain_text, require_any=require_any, require_all=require_all, label=label)
    _log_builder_info(df)
    _log_distributed_plan(df, label)
    parts = _run_iter_tables(runner, df, label, timeout_s=timeout_s)
    _assert_results_match(con, sql, parts, label, ordered=ordered)
    return explain_text


@pytest.fixture
def duckdb_conn():
    con = vane.connect()
    try:
        yield con
    finally:
        con.close()


@pytest.fixture
def parquet_path(tmp_path):
    parquet_path = tmp_path / "ray_e2e.parquet"
    con = vane.connect()
    try:
        con.execute(f"""
            COPY (
                SELECT 
                    a::INTEGER as a,
                    (a * 10)::INTEGER as b,
                    (a * 100)::INTEGER as c
                FROM generate_series(0, {RAY_E2E_ROWS - 1}) as t(a)
            ) TO '{parquet_path}' (FORMAT PARQUET)
        """)
    finally:
        con.close()
    return str(parquet_path)


@pytest.fixture
def partitioned_parquet_path(tmp_path):
    partitioned_path = tmp_path / "ray_e2e_partitioned"
    con = vane.connect()
    try:
        con.execute(f"""
            COPY (
                SELECT
                    (a % 10)::INTEGER as grp,
                    a::INTEGER as a,
                    (a * 10)::INTEGER as b,
                    (a * 100)::INTEGER as c
                FROM generate_series(0, {RAY_E2E_ROWS - 1}) as t(a)
            ) TO '{partitioned_path}' (FORMAT PARQUET, PARTITION_BY (grp))
        """)
    finally:
        con.close()
    return str(partitioned_path)


@pytest.fixture
def ray_runner(_vane_shuffle_env, request):
    request.getfixturevalue("ray_local")
    try:
        _runners.set_runner_ray(noop_if_initialized=True)
    except Exception:
        pytest.skip("duckdb runner API not available in this environment")

    try:
        runner = _runners.get_or_create_runner()
    except Exception:
        pytest.skip("Ray runner not available in this environment")

    if getattr(runner, "name", None) != "ray":
        pytest.skip("Ray runner not active")
    try:
        yield runner
    finally:
        # Ensure the compiled runner pointer is cleared between tests so we don't
        # reuse a Ray actor from a previous cluster lifecycle.
        vane_mod = getattr(vane, "vane_runners_cpp", None)
        if vane_mod is not None and hasattr(vane_mod, "teardown_runner"):
            vane_mod.teardown_runner()


@pytest.fixture
def ray_subprocess_env(_ray_local_cluster):
    _, cluster_address, _ = _ray_local_cluster
    env = dict(os.environ)
    env["RAY_ADDRESS"] = cluster_address
    return env


@pytest.fixture
def ray_runner_local_cluster(_vane_shuffle_env):
    try:
        import ray
    except Exception:
        pytest.skip("ray not installed")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Tip: In future versions of Ray")
        ray.init(
            address="local",
            ignore_reinit_error=True,
            logging_level="info",
            log_to_driver=True,
            num_cpus=1,
            object_store_memory=ray_test_object_store_bytes(),
        )

    try:
        _runners.set_runner_ray(noop_if_initialized=True)
    except Exception:
        pytest.skip("duckdb runner API not available in this environment")

    try:
        runner = _runners.get_or_create_runner()
    except Exception:
        pytest.skip("Ray runner not available in this environment")

    if getattr(runner, "name", None) != "ray":
        pytest.skip("Ray runner not active")
    try:
        yield runner
    finally:
        vane_mod = getattr(vane, "vane_runners_cpp", None)
        if vane_mod is not None and hasattr(vane_mod, "teardown_runner"):
            vane_mod.teardown_runner()
        try:
            ray.shutdown()
        except Exception:
            pass


def test_ray_scan_filter_projection(ray_runner, duckdb_conn, parquet_path):
    label = "test_ray_e2e: scan + filter + projection"
    sql = f"""
        SELECT 
            a,
            b,
            c,
            a + b AS sum_ab,
            a * b AS product
        FROM read_parquet('{parquet_path}')
        WHERE a >= 100 AND a < 900
    """
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_all=["PARQUET", "FILTER", "PROJECTION"],
    )


def test_ray_vllm_distributed(ray_runner, duckdb_conn):
    pytest.importorskip("pyarrow")
    try:
        __import__("vllm")
    except Exception as exc:
        pytest.skip(f"vllm unavailable: {exc}")
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"torch unavailable: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("vllm requires CUDA")

    timeout_s = int(os.getenv("VLLM_E2E_TIMEOUT", "300"))
    try:
        import signal

        signal.alarm(timeout_s)
    except Exception:
        pass

    label = "test_ray_e2e: vllm distributed"
    model = os.getenv("VLLM_E2E_MODEL", "Qwen/Qwen3-1.7B")
    max_tokens = int(os.getenv("VLLM_E2E_MAX_TOKENS", "32"))
    options = {
        "use_ray": False,
        "concurrency": 1,
        "gpus_per_actor": 1,
        "do_prefix_routing": False,
        "ray_worker_only": True,
        "engine_args": {
            "trust_remote_code": True,
            "max_model_len": 128,
        },
        "generate_args": {
            "sampling_params": {
                "temperature": 0.8,
                "top_p": 0.95,
                "max_tokens": max_tokens,
            }
        },
    }
    sql = (
        "SELECT id, prompt, "
        "vllm('Answer briefly: ' || prompt, '" + model + "', '" + json.dumps(options) + "') AS out "
        "FROM (VALUES "
        "(1, 'What is the capital of China?'), "
        "(2, 'What is the highest mountain in the world?'), "
        "(3, 'What type of database is DuckDB?')"
        ") AS t(id, prompt)"
    )

    # Avoid EXPLAIN here: it executes locally and can trigger vLLM init on the driver.

    df = duckdb_conn.sql(sql)
    _log_builder_info(df)
    _log_distributed_plan(df, label)
    parts = _run_iter_tables(ray_runner, df, label, timeout_s=timeout_s)
    rows = _collect_result_rows(parts)

    assert len(rows) == 3
    for row in rows:
        print(f"vllm_ray_e2e: id={row[0]} prompt={row[1]!r} output={row[2]!r}")
        assert row[2] not in ("", "NULL")


def test_ray_udf_distributed(ray_runner, duckdb_conn, tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    if ray is None:
        pytest.skip("ray not installed")

    label = "test_ray_e2e: udf distributed"

    class Upper:
        def __call__(self, table):
            vals = table.column("val").to_pylist()
            ids = table.column("id").to_pylist()
            return pa.table({"id": ids, "val": vals, "out": [v.upper() for v in vals]})

    input_path = tmp_path / "udf_input.parquet"
    duckdb_conn.execute(
        f"""
        COPY (
            SELECT *
            FROM (VALUES
                (1, 'a'),
                (2, 'b'),
                (3, 'c')
            ) t(id, val)
        ) TO '{input_path!s}' (FORMAT PARQUET)
        """
    )
    base = duckdb_conn.sql(f"SELECT * FROM read_parquet('{input_path!s}')")
    df = base.map_batches(
        Upper,
        schema={"id": vane.sqltype("INTEGER"), "val": vane.sqltype("VARCHAR"), "out": vane.sqltype("VARCHAR")},
        execution_backend="ray_actor",
        actor_number=2,
        gpus=0.0,
    )

    _log_builder_info(df)
    _log_distributed_plan(df, label)
    parts = _run_iter_tables(ray_runner, df, label, timeout_s=30.0)
    rows = _collect_result_rows(parts)

    assert set(rows) == {
        ("1", "a", "A"),
        ("2", "b", "B"),
        ("3", "c", "C"),
    }


def test_ray_udf_block_producing_single_stage_distributed(ray_runner, duckdb_conn, tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    if ray is None:
        pytest.skip("ray not installed")

    label = "test_ray_e2e: udf block producing single stage distributed"

    class Expand:
        def __call__(self, table):
            ids = table.column("id").to_pylist()
            out = []
            for value in ids:
                for piece in range(value):
                    out.append(value * 10 + piece)
            return pa.table({"combined": out})

    input_path = tmp_path / "udf_block_producing_input.parquet"
    duckdb_conn.execute(
        f"""
        COPY (
            SELECT *
            FROM (VALUES
                (1),
                (2),
                (3)
            ) t(id)
        ) TO '{input_path!s}' (FORMAT PARQUET)
        """
    )
    base = duckdb_conn.sql(f"SELECT * FROM read_parquet('{input_path!s}')")
    df = base.map_batches(
        Expand,
        schema={"combined": vane.sqltype("INTEGER")},
        execution_backend="ray_actor",
        actor_number=2,
        gpus=0.0,
        batch_size=1,
    )

    _log_builder_info(df)
    _log_distributed_plan(df, label)
    parts = _run_iter_tables(ray_runner, df, label, timeout_s=30.0)
    rows = _collect_result_rows(parts)

    assert set(rows) == {("10",), ("20",), ("21",), ("30",), ("31",), ("32",)}


def test_ray_udf_lazy_output_cpu_only_uses_direct_stream_output_distributed(ray_runner, duckdb_conn, tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    if ray is None:
        pytest.skip("ray not installed")
    label = "test_ray_e2e: udf lazy output cpu-only uses direct stream output distributed"

    class Upper:
        def __call__(self, table):
            vals = table.column("val").to_pylist()
            ids = table.column("id").to_pylist()
            return pa.table({"id": ids, "out": [v.upper() for v in vals]})

    input_path = tmp_path / "udf_lazy_output_input.parquet"
    duckdb_conn.execute(
        f"""
        COPY (
            SELECT *
            FROM (VALUES
                (1, 'a'),
                (2, 'b'),
                (3, 'c')
            ) t(id, val)
        ) TO '{input_path!s}' (FORMAT PARQUET)
        """
    )
    base = duckdb_conn.sql(f"SELECT * FROM read_parquet('{input_path!s}')")
    df = base.map_batches(
        Upper,
        schema={"id": vane.sqltype("INTEGER"), "out": vane.sqltype("VARCHAR")},
        execution_backend="ray_actor",
        actor_number=2,
        gpus=0.0,
        batch_size=2,
    )
    plan = df.explain()
    assert "ray_block_stream_output" in plan

    _log_builder_info(df)
    _log_distributed_plan(df, label)
    parts = _run_iter_tables(ray_runner, df, label, timeout_s=30.0)
    rows = _collect_result_rows(parts)

    assert set(rows) == {("1", "A"), ("2", "B"), ("3", "C")}


def test_ray_udf_lazy_output_defaults_to_enabled(duckdb_conn):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    if ray is None:
        pytest.skip("ray not installed")

    class Upper:
        def __call__(self, table):
            vals = table.column("val").to_pylist()
            return pa.table({"out": [v.upper() for v in vals]})

    class Suffix:
        def __call__(self, table):
            vals = table.column("out").to_pylist()
            return pa.table({"final": [v + "!" for v in vals]})

    base = duckdb_conn.sql("SELECT * FROM (VALUES ('a'), ('b')) t(val)")
    df = base.map_batches(
        Upper,
        schema={"out": vane.sqltype("VARCHAR")},
        execution_backend="ray_actor",
        actor_number=1,
        gpus=0.0,
        batch_size=2,
    ).map_batches(
        Suffix,
        schema={"final": vane.sqltype("VARCHAR")},
        execution_backend="ray_actor",
        actor_number=1,
        gpus=1,
        batch_size=2,
    )
    plan = df.explain()
    assert "ray_block_stream_output" in plan


def test_ray_udf_lazy_output_input_passthrough_distributed(ray_runner, duckdb_conn, tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    if ray is None:
        pytest.skip("ray not installed")
    if float(ray.cluster_resources().get("GPU", 0.0)) < 2.0:
        pytest.skip("requires a Ray cluster with at least two GPU resources")
    label = "test_ray_e2e: udf lazy output input passthrough distributed"

    class Stage1:
        def __call__(self, table):
            ids = table.column("id").to_pylist()
            return pa.table({"score": [value * 10 for value in ids]})

    class Stage2:
        def __call__(self, table):
            scores = table.column("score").to_pylist()
            return pa.table({"combined": [score + 1 for score in scores]})

    input_path = tmp_path / "udf_lazy_input_passthrough.parquet"
    duckdb_conn.execute(
        f"""
        COPY (
            SELECT *
            FROM (VALUES
                (1),
                (2),
                (3),
                (4)
            ) t(id)
        ) TO '{input_path!s}' (FORMAT PARQUET)
        """
    )
    base = duckdb_conn.sql(f"SELECT * FROM read_parquet('{input_path!s}')")
    df = base.map_batches(
        Stage1,
        schema={"score": vane.sqltype("INTEGER")},
        execution_backend="ray_actor",
        actor_number=2,
        gpus=0.0,
        batch_size=2,
    ).map_batches(
        Stage2,
        schema={"combined": vane.sqltype("INTEGER")},
        execution_backend="ray_actor",
        actor_number=2,
        gpus=1,
        batch_size=2,
    )
    plan = df.explain()
    assert plan.count("ray_block_stream_output") == 2

    _log_builder_info(df)
    _log_distributed_plan(df, label)
    parts = _run_iter_tables(ray_runner, df, label, timeout_s=30.0)
    rows = _collect_result_rows(parts)

    assert set(rows) == {("11",), ("21",), ("31",), ("41",)}


def test_ray_udf_lazy_output_projection_limit_passthrough_distributed(ray_runner, duckdb_conn, tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    if ray is None:
        pytest.skip("ray not installed")
    if float(ray.cluster_resources().get("GPU", 0.0)) < 2.0:
        pytest.skip("requires a Ray cluster with at least two GPU resources")
    label = "test_ray_e2e: udf lazy output projection limit passthrough distributed"

    class Stage1:
        def __call__(self, table):
            ids = table.column("id").to_pylist()
            return pa.table({"id": ids, "score": [value * 10 for value in ids]})

    class Stage2:
        def __call__(self, table):
            if table.column_names != ["score_alias", "id"]:
                raise RuntimeError(f"unexpected lazy input columns: {table.column_names}")
            scores = table.column("score_alias").to_pylist()
            ids = table.column("id").to_pylist()
            return pa.table({"combined": [score + value for score, value in zip(scores, ids, strict=False)]})

    input_path = tmp_path / "udf_lazy_projection_limit_passthrough.parquet"
    duckdb_conn.execute(
        f"""
        COPY (
            SELECT *
            FROM (VALUES
                (1),
                (2),
                (3),
                (4)
            ) t(id)
        ) TO '{input_path!s}' (FORMAT PARQUET)
        """
    )
    base = duckdb_conn.sql(f"SELECT * FROM read_parquet('{input_path!s}')")
    produced = base.map_batches(
        Stage1,
        schema={"id": vane.sqltype("INTEGER"), "score": vane.sqltype("INTEGER")},
        execution_backend="ray_actor",
        actor_number=2,
        gpus=0.0,
        batch_size=2,
    )
    df = (
        produced.project("score as score_alias, id")
        .limit(3)
        .map_batches(
            Stage2,
            schema={"combined": vane.sqltype("INTEGER")},
            execution_backend="ray_actor",
            actor_number=2,
            gpus=1,
            batch_size=2,
        )
    )
    plan = df.explain()
    assert plan.count("ray_block_stream_output") == 2

    _log_builder_info(df)
    _log_distributed_plan(df, label)
    parts = _run_iter_tables(ray_runner, df, label, timeout_s=30.0)
    rows = _collect_result_rows(parts)

    assert len(rows) == 3
    assert set(rows).issubset({("11",), ("22",), ("33",), ("44",)})


def test_ray_udf_lazy_output_local_exchange_passthrough_distributed(ray_runner, duckdb_conn, tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    if ray is None:
        pytest.skip("ray not installed")
    if float(ray.cluster_resources().get("GPU", 0.0)) < 2.0:
        pytest.skip("requires a Ray cluster with at least two GPU resources")
    label = "test_ray_e2e: udf lazy output local exchange passthrough distributed"

    class Stage1:
        def __call__(self, table):
            ids = table.column("id").to_pylist()
            return pa.table({"id": ids, "score": [value * 10 for value in ids]})

    class Stage2:
        def __call__(self, table):
            if table.column_names != ["id", "score"]:
                raise RuntimeError(f"unexpected local exchange lazy input columns: {table.column_names}")
            ids = table.column("id").to_pylist()
            scores = table.column("score").to_pylist()
            return pa.table({"combined": [score + value for value, score in zip(ids, scores, strict=False)]})

    input_path = tmp_path / "udf_lazy_local_exchange_passthrough.parquet"
    duckdb_conn.execute(
        f"""
        COPY (
            SELECT *
            FROM (VALUES
                (1),
                (2),
                (3),
                (4)
            ) t(id)
        ) TO '{input_path!s}' (FORMAT PARQUET)
        """
    )
    base = duckdb_conn.sql(f"SELECT * FROM read_parquet('{input_path!s}')")
    produced = base.map_batches(
        Stage1,
        schema={"id": vane.sqltype("INTEGER"), "score": vane.sqltype("INTEGER")},
        execution_backend="ray_actor",
        actor_number=2,
        gpus=0.0,
        batch_size=2,
    )
    df = produced.local_exchange(2).map_batches(
        Stage2,
        schema={"combined": vane.sqltype("INTEGER")},
        execution_backend="ray_actor",
        actor_number=2,
        gpus=1,
        batch_size=2,
    )
    plan = df.explain()
    assert plan.count("ray_block_stream_output") == 2
    assert "lazy_ref_local_exchange" in plan
    assert "whole_block" in plan

    _log_builder_info(df)
    _log_distributed_plan(df, label)
    parts = _run_iter_tables(ray_runner, df, label, timeout_s=30.0)
    rows = _collect_result_rows(parts)

    assert set(rows) == {("11",), ("22",), ("33",), ("44",)}


def test_ray_udf_lazy_output_filter_materialize_distributed(ray_runner, duckdb_conn, tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    if ray is None:
        pytest.skip("ray not installed")
    label = "test_ray_e2e: udf lazy output filter materialize distributed"

    class Stage1:
        def __call__(self, table):
            ids = table.column("id").to_pylist()
            return pa.table({"id": ids, "score": [value * 10 for value in ids]})

    class Stage2:
        def __call__(self, table):
            if table.column_names != ["id", "score"]:
                raise RuntimeError(f"unexpected materialized filter input columns: {table.column_names}")
            ids = table.column("id").to_pylist()
            scores = table.column("score").to_pylist()
            return pa.table({"combined": [score + value for value, score in zip(ids, scores, strict=False)]})

    input_path = tmp_path / "udf_lazy_filter_materialize.parquet"
    duckdb_conn.execute(
        f"""
        COPY (
            SELECT *
            FROM (VALUES
                (1),
                (2),
                (3),
                (4)
            ) t(id)
        ) TO '{input_path!s}' (FORMAT PARQUET)
        """
    )
    base = duckdb_conn.sql(f"SELECT * FROM read_parquet('{input_path!s}')")
    produced = base.map_batches(
        Stage1,
        schema={"id": vane.sqltype("INTEGER"), "score": vane.sqltype("INTEGER")},
        execution_backend="ray_actor",
        actor_number=2,
        gpus=0.0,
        batch_size=2,
    )
    df = produced.filter("score >= 20").map_batches(
        Stage2,
        schema={"combined": vane.sqltype("INTEGER")},
        execution_backend="ray_actor",
        actor_number=2,
        gpus=0.0,
        batch_size=2,
    )
    plan = df.explain()
    assert "FILTER" in plan
    assert plan.count("ray_block_stream_output") == 2

    _log_builder_info(df)
    _log_distributed_plan(df, label)
    parts = _run_iter_tables(ray_runner, df, label, timeout_s=30.0)
    rows = _collect_result_rows(parts)

    assert set(rows) == {("22",), ("33",), ("44",)}


def test_ray_udf_lazy_output_budget_plan(duckdb_conn):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    if ray is None:
        pytest.skip("ray not installed")

    class Stage1:
        def __call__(self, table):
            ids = table.column("id").to_pylist()
            return pa.table({"score": [value * 10 for value in ids]})

    class Stage2:
        def __call__(self, table):
            scores = table.column("score").to_pylist()
            return pa.table({"combined": [score + 1 for score in scores]})

    duckdb_conn.execute(
        """
        CREATE OR REPLACE TEMP TABLE udf_lazy_budget_input AS
        SELECT * FROM (VALUES (1), (2), (3)) t(id)
        """
    )
    base = duckdb_conn.sql("SELECT * FROM udf_lazy_budget_input")
    df = base.map_batches(
        Stage1,
        schema={"score": vane.sqltype("INTEGER")},
        execution_backend="ray_actor",
        actor_number=2,
        gpus=0.0,
        batch_size=2,
    ).map_batches(
        Stage2,
        schema={"combined": vane.sqltype("INTEGER")},
        execution_backend="ray_actor",
        actor_number=2,
        gpus=1,
        batch_size=2,
    )
    plan = df.explain()
    assert plan.count("ray_block_stream_output") == 2


def test_ray_udf_direct_output_lease_lifetime(ray_runner, duckdb_conn, tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    if ray is None:
        pytest.skip("ray not installed")

    if float(ray.cluster_resources().get("GPU", 0.0)) < 1.0:
        pytest.skip("requires a Ray cluster with at least one GPU resource")
    label = "test_ray_e2e: udf direct output lease lifetime"

    class Stage1:
        def __call__(self, table):
            ids = table.column("id").to_pylist()
            return pa.table({"score": [value * 10 for value in ids]})

    class Stage2:
        def __call__(self, table):
            scores = table.column("score").to_pylist()
            return pa.table({"combined": [score + 1 for score in scores]})

    input_path = tmp_path / "udf_lazy_lifetime_release.parquet"
    duckdb_conn.execute(
        f"""
        COPY (
            SELECT *
            FROM (VALUES
                (1),
                (2),
                (3),
                (4)
            ) t(id)
        ) TO '{input_path!s}' (FORMAT PARQUET)
        """
    )
    base = duckdb_conn.sql(f"SELECT * FROM read_parquet('{input_path!s}')")
    df = base.map_batches(
        Stage1,
        schema={"score": vane.sqltype("INTEGER")},
        execution_backend="ray_actor",
        actor_number=2,
        gpus=0.0,
        batch_size=2,
    ).map_batches(
        Stage2,
        schema={"combined": vane.sqltype("INTEGER")},
        execution_backend="ray_actor",
        actor_number=1,
        gpus=1,
        batch_size=2,
    )

    _log_builder_info(df)
    _log_distributed_plan(df, label)
    parts = _run_iter_tables(ray_runner, df, label, timeout_s=30.0)
    rows = _collect_result_rows(parts)
    assert set(rows) == {("11",), ("21",), ("31",), ("41",)}


def test_ray_udf_streaming_breaker_distributed(ray_runner, duckdb_conn, tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    if ray is None:
        pytest.skip("ray not installed")

    label = "test_ray_e2e: udf streaming breaker distributed"

    class Upper:
        def __call__(self, table):
            vals = table.column("val").to_pylist()
            ids = table.column("id").to_pylist()
            return pa.table({"id": ids, "val": vals, "out": [v.upper() for v in vals]})

    input_path = tmp_path / "udf_streaming_input.parquet"
    duckdb_conn.execute(
        f"""
        COPY (
            SELECT *
            FROM (VALUES
                (1, 'a'),
                (2, 'b'),
                (3, 'c'),
                (4, 'd')
            ) t(id, val)
        ) TO '{input_path!s}' (FORMAT PARQUET)
        """
    )
    base = duckdb_conn.sql(f"SELECT * FROM read_parquet('{input_path!s}')")
    df = base.map_batches(
        Upper,
        schema={"id": vane.sqltype("INTEGER"), "val": vane.sqltype("VARCHAR"), "out": vane.sqltype("VARCHAR")},
        execution_backend="ray_actor",
        actor_number=2,
        gpus=0.0,
        batch_size=2,
        streaming_breaker=True,
    )
    assert "STREAMING_UDF" in df.explain()

    _log_builder_info(df)
    _log_distributed_plan(df, label)
    parts = _run_iter_tables(ray_runner, df, label, timeout_s=30.0)
    rows = _collect_result_rows(parts)

    assert set(rows) == {
        ("1", "a", "A"),
        ("2", "b", "B"),
        ("3", "c", "C"),
        ("4", "d", "D"),
    }


def test_ray_udf_streaming_breaker_non_row_preserving(ray_runner, duckdb_conn, tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    if ray is None:
        pytest.skip("ray not installed")

    label = "test_ray_e2e: udf streaming breaker non row preserving"

    class Explode:
        def __call__(self, table):
            ids = table.column("id").to_pylist()
            out = []
            for value in ids:
                out.extend([value, value + 10])
            return pa.table({"id": out})

    input_path = tmp_path / "udf_streaming_explode_input.parquet"
    duckdb_conn.execute(
        f"""
        COPY (
            SELECT *
            FROM (VALUES (1), (2), (3)) t(id)
        ) TO '{input_path!s}' (FORMAT PARQUET)
        """
    )
    base = duckdb_conn.sql(f"SELECT * FROM read_parquet('{input_path!s}')")
    df = base.map_batches(
        Explode,
        schema={"id": vane.sqltype("INTEGER")},
        execution_backend="ray_actor",
        actor_number=2,
        gpus=0.0,
        batch_size=2,
        streaming_breaker=True,
    )
    assert "STREAMING_UDF" in df.explain()

    _log_builder_info(df)
    _log_distributed_plan(df, label)
    parts = _run_iter_tables(ray_runner, df, label, timeout_s=30.0)
    rows = _collect_result_rows(parts)

    assert set(rows) == {("1",), ("2",), ("3",), ("11",), ("12",), ("13",)}


def test_ray_udf_streaming_breaker_backpressure_preserves_rows(ray_runner, duckdb_conn, tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    if ray is None:
        pytest.skip("ray not installed")
    label = "test_ray_e2e: udf streaming breaker backpressure preserves rows"
    row_count = 5000

    class PlusOne:
        def __call__(self, table):
            ids = table.column("id").to_pylist()
            return pa.table({"id": ids, "out": [value + 1 for value in ids]})

    input_path = tmp_path / "udf_streaming_backpressure_input.parquet"
    duckdb_conn.execute(
        f"""
        COPY (
            SELECT i::INTEGER AS id
            FROM range({row_count}) t(i)
        ) TO '{input_path!s}' (FORMAT PARQUET)
        """
    )
    base = duckdb_conn.sql(f"SELECT * FROM read_parquet('{input_path!s}')")
    df = base.map_batches(
        PlusOne,
        schema={"id": vane.sqltype("INTEGER"), "out": vane.sqltype("INTEGER")},
        execution_backend="ray_actor",
        actor_number=1,
        gpus=0.0,
        batch_size=64,
        streaming_breaker=True,
    )
    assert "STREAMING_UDF" in df.explain()

    _log_builder_info(df)
    _log_distributed_plan(df, label)
    parts = _run_iter_tables(ray_runner, df, label, timeout_s=60.0)
    rows = _collect_result_rows(parts)

    assert len(rows) == row_count
    assert {int(row[0]) for row in rows} == set(range(row_count))
    assert all(int(out) == int(row_id) + 1 for row_id, out in rows)


def test_ray_udf_lazy_streaming_producer_plan(duckdb_conn):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    if ray is None:
        pytest.skip("ray not installed")

    class Stage1:
        def __call__(self, table):
            ids = table.column("id").to_pylist()
            return pa.table({"score": [value * 10 for value in ids]})

    class Stage2:
        def __call__(self, table):
            scores = table.column("score").to_pylist()
            return pa.table({"combined": [score + 1 for score in scores]})

    duckdb_conn.execute(
        """
        CREATE OR REPLACE TEMP TABLE lazy_streaming_plan_input AS
        SELECT * FROM (VALUES (1), (2), (3), (4)) t(id)
        """
    )
    base = duckdb_conn.sql("SELECT * FROM lazy_streaming_plan_input")
    df = base.map_batches(
        Stage1,
        schema={"score": vane.sqltype("INTEGER")},
        execution_backend="ray_actor",
        actor_number=2,
        gpus=0.0,
        batch_size=2,
    ).map_batches(
        Stage2,
        schema={"combined": vane.sqltype("INTEGER")},
        execution_backend="ray_actor",
        actor_number=1,
        gpus=1,
        batch_size=2,
    )

    plan = df.explain()
    assert "STREAMING_UDF" in plan
    assert "ray_block_stream_output" in plan


def test_ray_udf_lazy_streaming_producer_yields_ref_bundles(ray_runner, duckdb_conn, tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    if ray is None:
        pytest.skip("ray not installed")
    if float(ray.cluster_resources().get("GPU", 0.0)) < 1.0:
        pytest.skip("requires a Ray cluster with at least one GPU resource")
    label = "test_ray_e2e: lazy streaming producer yields ref bundles"

    class Stage1:
        def __call__(self, table):
            ids = table.column("id").to_pylist()
            for value in ids:
                yield pa.table({"score": [value * 10]})

    class Stage2:
        def __call__(self, table):
            scores = table.column("score").to_pylist()
            batch_rows = table.num_rows
            return pa.table(
                {
                    "score": scores,
                    "seen_rows": [batch_rows] * len(scores),
                }
            )

    input_path = tmp_path / "udf_lazy_streaming_generator.parquet"
    duckdb_conn.execute(
        f"""
        COPY (
            SELECT *
            FROM (VALUES
                (1),
                (2),
                (3),
                (4)
            ) t(id)
        ) TO '{input_path!s}' (FORMAT PARQUET)
        """
    )
    base = duckdb_conn.sql(f"SELECT * FROM read_parquet('{input_path!s}')")
    df = base.map_batches(
        Stage1,
        schema={"score": vane.sqltype("INTEGER")},
        execution_backend="ray_actor",
        actor_number=1,
        gpus=0.0,
        batch_size=4,
    ).map_batches(
        Stage2,
        schema={
            "score": vane.sqltype("INTEGER"),
            "seen_rows": vane.sqltype("INTEGER"),
        },
        execution_backend="ray_actor",
        actor_number=1,
        gpus=1,
        batch_size=8,
    )

    plan = df.explain()
    assert "STREAMING_UDF" in plan
    assert "ray_block_stream_output" in plan

    _log_builder_info(df)
    _log_distributed_plan(df, label)
    parts = _run_iter_tables(ray_runner, df, label, timeout_s=30.0)
    rows = _collect_result_rows(parts)

    assert set(rows) == {("10", "4"), ("20", "4"), ("30", "4"), ("40", "4")}


def test_ray_task_map_batches_ref_passthrough_to_actor_distributed(ray_runner, duckdb_conn, tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    if ray is None:
        pytest.skip("ray not installed")
    label = "test_ray_e2e: ray_task map_batches ref passthrough to actor"

    def stage1(table):
        values = table.column("id").to_pylist()
        return pa.table({"score": [value * 10 for value in values]})

    class Stage2:
        def __call__(self, table):
            if table.column_names != ["score"]:
                raise RuntimeError(f"unexpected lazy input columns: {table.column_names}")
            scores = table.column("score").to_pylist()
            return pa.table({"combined": [score + 1 for score in scores]})

    input_path = tmp_path / "ray_task_map_batches_ref_passthrough.parquet"
    duckdb_conn.execute(
        f"""
        COPY (
            SELECT *
            FROM (VALUES
                (1),
                (2),
                (3),
                (4)
            ) t(id)
        ) TO '{input_path!s}' (FORMAT PARQUET)
        """
    )
    base = duckdb_conn.sql(f"SELECT * FROM read_parquet('{input_path!s}')")
    df = base.map_batches(
        stage1,
        schema={"score": vane.sqltype("INTEGER")},
        execution_backend="ray_task",
        batch_size=2,
        streaming_breaker=True,
    ).map_batches(
        Stage2,
        schema={"combined": vane.sqltype("INTEGER")},
        execution_backend="ray_actor",
        actor_number=1,
        cpus=1.0,
        gpus=0.0,
        batch_size=2,
    )

    plan = df.explain()
    assert "STREAMING_UDF" in plan
    assert "ray_block_stream_output" in plan
    assert "ray_task" in plan
    assert "ray_actor" in plan

    _log_builder_info(df)
    _log_distributed_plan(df, label)
    parts = _run_iter_tables(ray_runner, df, label, timeout_s=30.0)
    rows = _collect_result_rows(parts)

    assert set(rows) == {("11",), ("21",), ("31",), ("41",)}


def test_ray_task_large_block_stream_reaches_actor_with_bounded_leases(tmp_path, ray_subprocess_env):
    pytest.importorskip("pyarrow")
    import subprocess
    import sys
    import textwrap

    if ray is None:
        pytest.skip("ray not installed")

    script = textwrap.dedent(
        f"""
        import os
        import time
        from pathlib import Path

        import vane
        import pyarrow as pa
        from vane import runners as _runners

        row_count = 512
        payload_len = 524288
        input_path = Path({str(tmp_path / "large_ref_stream_input.parquet")!r})

        con = vane.connect()
        con.execute("SET threads=8")
        con.execute(
            f'''
            COPY (
                SELECT i::INTEGER AS id, repeat('x', {{payload_len}}) AS payload
                FROM range({{row_count}}) t(i)
            ) TO '{{input_path!s}}' (FORMAT PARQUET)
            '''
        )

        def producer(table):
            return pa.table({{"id": table.column("id"), "payload": table.column("payload")}})

        class Consumer:
            def __call__(self, table):
                return pa.table(
                    {{
                        "id": table.column("id"),
                        "payload_len": [len(v.as_py()) for v in table.column("payload")],
                    }}
                )

        _runners.set_runner_ray(noop_if_initialized=True)
        runner = _runners.get_or_create_runner()

        rel = (
            con.read_parquet(str(input_path))
            .repartition(2)
            .map_batches(
                producer,
                schema={{
                    "id": vane.sqltypes.INTEGER,
                    "payload": vane.sqltypes.VARCHAR,
                }},
                execution_backend="ray_task",
                cpus=1.0,
                gpus=0.0,
                batch_size=2048,
                target_max_batch_bytes=128 * 1024 * 1024,
                streaming_breaker=True,
            )
            .map_batches(
                Consumer,
                schema={{
                    "id": vane.sqltypes.INTEGER,
                    "payload_len": vane.sqltypes.INTEGER,
                }},
                execution_backend="ray_actor",
                actor_number=1,
                cpus=1.0,
                gpus=0.0,
                batch_size=32,
                streaming_breaker=True,
            )
        )

        start = time.time()
        parts = list(runner.run_iter_tables(rel, results_buffer_size=1))
        elapsed = time.time() - start
        ids = []
        lengths = []
        for part in parts:
            table = part.to_arrow() if hasattr(part, "to_arrow") else part
            ids.extend(table.column(0).to_pylist())
            lengths.extend(table.column(1).to_pylist())

        assert elapsed < 45, f"distributed streaming UDF stuck for {{elapsed:.2f}}s"
        assert len(ids) == row_count
        assert set(ids) == set(range(row_count))
        assert all(value == payload_len for value in lengths)
        """
    )
    env = dict(ray_subprocess_env)
    env["VANE_PROGRESS"] = "0"
    env["RAY_DEDUP_LOGS"] = "0"
    env["VANE_RUNNER"] = "ray"
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_ray_lazy_tail_block_submits_without_cross_lease_batching(tmp_path, ray_subprocess_env):
    pytest.importorskip("pyarrow")
    import subprocess
    import sys
    import textwrap

    if ray is None:
        pytest.skip("ray not installed")

    script = textwrap.dedent(
        f"""
        import os
        from pathlib import Path

        import vane
        import pyarrow as pa
        from vane import runners as _runners

        row_count = 1000
        input_path = Path({str(tmp_path / "lazy_tail_lease_input.parquet")!r})

        con = vane.connect()
        con.execute("SET threads=8")
        con.execute(
            f'''
            COPY (
                SELECT i::INTEGER AS id
                FROM range({{row_count}}) t(i)
            ) TO '{{input_path!s}}' (FORMAT PARQUET)
            '''
        )

        def producer(table):
            return pa.table({{"id": table.column("id")}})

        class Consumer:
            def __call__(self, table):
                return pa.table({{"id": table.column("id")}})

        _runners.set_runner_ray(noop_if_initialized=True)
        runner = _runners.get_or_create_runner()
        rel = (
            con.read_parquet(str(input_path))
            .map_batches(
                producer,
                schema={{"id": vane.sqltypes.INTEGER}},
                execution_backend="ray_task",
                cpus=1.0,
                gpus=0.0,
                batch_size=1000,
                task_input_max_bytes=1024,
                output_target_max_bytes=1024,
                streaming_breaker=True,
            )
            .map_batches(
                Consumer,
                schema={{"id": vane.sqltypes.INTEGER}},
                execution_backend="ray_actor",
                actor_number=1,
                cpus=1.0,
                gpus=0.0,
                batch_size=1000,
                task_input_max_bytes=64 * 1024,
                output_target_max_bytes=64 * 1024,
                streaming_breaker=True,
            )
        )

        total = 0
        for part in runner.run_iter_tables(rel, results_buffer_size=1):
            table = part.to_arrow() if hasattr(part, "to_arrow") else part
            total += table.num_rows
        assert total == row_count
        """
    )
    env = dict(ray_subprocess_env)
    env["VANE_PROGRESS"] = "0"
    env["RAY_DEDUP_LOGS"] = "0"
    env["VANE_RUNNER"] = "ray"
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise AssertionError(stderr[-40_000:]) from exc
    assert result.returncode == 0, result.stdout + result.stderr


def test_ray_actor_compute_batches_span_upstream_block_boundaries(tmp_path, ray_subprocess_env):
    pytest.importorskip("pyarrow")
    import subprocess
    import sys
    import textwrap

    if ray is None:
        pytest.skip("ray not installed")

    script = textwrap.dedent(
        f"""
        from pathlib import Path

        import vane
        import pyarrow as pa
        from vane import runners as _runners

        row_count = 300
        compute_batch_rows = 100
        payload_len = 100 * 1024
        input_path = Path({str(tmp_path / "actor_compute_batch_boundaries.parquet")!r})

        con = vane.connect()
        con.execute(
            f'''
            COPY (
                SELECT i::INTEGER AS id
                FROM range({{row_count}}) t(i)
            ) TO '{{input_path!s}}' (FORMAT PARQUET)
            '''
        )

        def producer(table):
            payload = "x" * payload_len
            return pa.table(
                {{"id": table.column("id"), "payload": [payload] * table.num_rows}}
            )

        class Consumer:
            def __call__(self, table):
                return pa.table(
                    {{
                        "id": table.column("id"),
                        "compute_batch_rows": [table.num_rows] * table.num_rows,
                    }}
                )

        rel = (
            con.read_parquet(str(input_path))
            .repartition(1)
            .map_batches(
                producer,
                schema={{"id": vane.sqltypes.INTEGER, "payload": vane.sqltypes.VARCHAR}},
                execution_backend="ray_task",
                cpus=1.0,
                gpus=0.0,
                batch_size=compute_batch_rows,
                task_input_max_bytes=16 * 1024 * 1024,
                output_target_max_bytes=8 * 1024 * 1024,
                streaming_breaker=True,
            )
            .map_batches(
                Consumer,
                schema={{"id": vane.sqltypes.INTEGER, "compute_batch_rows": vane.sqltypes.INTEGER}},
                execution_backend="ray_actor",
                actor_number=1,
                cpus=1.0,
                gpus=0.0,
                batch_size=compute_batch_rows,
                task_input_max_bytes=12 * 1024 * 1024,
                output_target_max_bytes=1024 * 1024,
                streaming_breaker=True,
            )
        )

        _runners.set_runner_ray(noop_if_initialized=True)
        runner = _runners.get_or_create_runner()
        total = 0
        observed_batch_rows = set()
        for part in runner.run_iter_tables(rel, results_buffer_size=1):
            table = part.to_arrow() if hasattr(part, "to_arrow") else part
            total += table.num_rows
            observed_batch_rows.update(table.column(1).to_pylist())

        assert total == row_count
        assert observed_batch_rows == {{compute_batch_rows}}, observed_batch_rows
        """
    )
    env = dict(ray_subprocess_env)
    env["VANE_PROGRESS"] = "0"
    env["RAY_DEDUP_LOGS"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_ray_actor_soft_minimum_task_batch_matches_ray_data_bundling(tmp_path, ray_subprocess_env):
    pytest.importorskip("pyarrow")
    import subprocess
    import sys
    import textwrap

    if ray is None:
        pytest.skip("ray not installed")

    script = textwrap.dedent(
        f"""
        from collections import Counter
        from pathlib import Path

        import vane
        import pyarrow as pa
        from vane import runners as _runners

        input_path = Path({str(tmp_path / "soft_min_task_batch.parquet")!r})
        con = vane.connect()
        con.execute(
            f'''COPY (SELECT i::INTEGER AS id FROM range(230) t(i))
                TO '{{input_path!s}}' (FORMAT PARQUET)'''
        )

        def producer(table):
            return pa.table({{"id": table.column("id")}})

        class Consumer:
            def __call__(self, table):
                return pa.table(
                    {{"batch_rows": [table.num_rows] * table.num_rows}}
                )

        rel = (
            con.read_parquet(str(input_path))
            .repartition(1)
            .map_batches(
                producer,
                schema={{"id": vane.sqltypes.INTEGER}},
                execution_backend="ray_task",
                cpus=1.0,
                gpus=0.0,
                batch_size=115,
                output_batch_size=110,
                preserve_compute_batch_boundaries=True,
                output_target_max_bytes=1024 * 1024,
                streaming_breaker=True,
            )
            .map_batches(
                Consumer,
                schema={{"batch_rows": vane.sqltypes.INTEGER}},
                execution_backend="ray_actor",
                actor_number=1,
                cpus=1.0,
                gpus=0.0,
                batch_size=32,
                min_task_batch_size=32,
                task_input_max_bytes=1024 * 1024,
                output_target_max_bytes=1024 * 1024,
                streaming_breaker=True,
            )
        )

        _runners.set_runner_ray(noop_if_initialized=True)
        runner = _runners.get_or_create_runner()
        counts = Counter()
        for part in runner.run_iter_tables(rel, results_buffer_size=1):
            table = part.to_arrow() if hasattr(part, "to_arrow") else part
            counts.update(table.column(0).to_pylist())

        expected = Counter({{32: 192, 14: 14, 19: 19, 5: 5}})
        assert counts == expected, (counts, expected)
        """
    )
    env = dict(ray_subprocess_env)
    env["VANE_PROGRESS"] = "0"
    env["RAY_DEDUP_LOGS"] = "0"
    env["VANE_RUNNER"] = "ray"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_ray_actor_lazy_row_backpressure_preserves_non_tail_batch_alignment(tmp_path, ray_subprocess_env):
    pytest.importorskip("pyarrow")
    import subprocess
    import sys
    import textwrap

    if ray is None:
        pytest.skip("ray not installed")

    script = textwrap.dedent(
        f"""
        import time
        from collections import Counter
        from pathlib import Path

        import vane
        import pyarrow as pa
        from vane import runners as _runners

        row_count = 13_000
        producer_batch_rows = 8_000
        compute_batch_rows = 4_096
        input_path = Path({str(tmp_path / "actor_lazy_row_alignment.parquet")!r})

        con = vane.connect()
        con.execute("SET threads=1")
        con.execute(
            f'''
            COPY (
                SELECT i::INTEGER AS id
                FROM range({{row_count}}) t(i)
            ) TO '{{input_path!s}}' (FORMAT PARQUET)
            '''
        )

        def producer(table):
            return pa.table({{"id": table.column("id")}})

        class SlowConsumer:
            def __call__(self, table):
                # Keep the first submit in flight while the 5k tail descriptor
                # reaches the C++ lazy sink. Row pressure must not flush the
                # queued 3904-row remainder as a non-tail compute batch.
                time.sleep(0.1)
                return pa.table(
                    {{
                        "id": table.column("id"),
                        "compute_batch_rows": [table.num_rows] * table.num_rows,
                    }}
                )

        rel = (
            con.read_parquet(str(input_path))
            .repartition(1)
            .map_batches(
                producer,
                schema={{"id": vane.sqltypes.INTEGER}},
                execution_backend="ray_task",
                cpus=1.0,
                gpus=0.0,
                batch_size=producer_batch_rows,
                task_input_max_bytes=64 * 1024 * 1024,
                output_target_max_bytes=64 * 1024 * 1024,
                streaming_breaker=True,
            )
            .map_batches(
                SlowConsumer,
                schema={{
                    "id": vane.sqltypes.INTEGER,
                    "compute_batch_rows": vane.sqltypes.INTEGER,
                }},
                execution_backend="ray_actor",
                actor_number=1,
                cpus=1.0,
                gpus=0.0,
                batch_size=compute_batch_rows,
                task_input_max_bytes=64 * 1024 * 1024,
                output_target_max_bytes=64 * 1024 * 1024,
                streaming_breaker=True,
            )
        )

        _runners.set_runner_ray(noop_if_initialized=True)
        runner = _runners.get_or_create_runner()
        total = 0
        observed = Counter()
        for part in runner.run_iter_tables(rel, results_buffer_size=1):
            table = part.to_arrow() if hasattr(part, "to_arrow") else part
            total += table.num_rows
            observed.update(table.column(1).to_pylist())

        assert total == row_count
        assert set(observed) == {{compute_batch_rows, 712}}, observed
        assert observed[compute_batch_rows] == compute_batch_rows * 3, observed
        assert observed[712] == 712, observed
        """
    )
    env = dict(ray_subprocess_env)
    env["VANE_PROGRESS"] = "0"
    env["RAY_DEDUP_LOGS"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_ray_task_block_stream_does_not_deadlock_when_sink_and_source_blocked(tmp_path, ray_subprocess_env):
    pytest.importorskip("pyarrow")
    import subprocess
    import sys
    import textwrap

    if ray is None:
        pytest.skip("ray not installed")

    script = textwrap.dedent(
        f"""
        import os
        import time
        from pathlib import Path

        import vane
        import pyarrow as pa
        from vane import runners as _runners

        row_count = 8192
        payload_len = 65536
        input_path = Path({str(tmp_path / "streaming_deadlock_input.parquet")!r})

        con = vane.connect()
        con.execute("SET threads=36")
        con.execute(
            f'''
            COPY (
                SELECT i::INTEGER AS id
                FROM range({{row_count}}) t(i)
            ) TO '{{input_path!s}}' (FORMAT PARQUET)
            '''
        )

        def slow_producer(table):
            time.sleep(0.5)
            payload = "x" * payload_len
            return pa.table({{"id": table.column("id"), "payload": [payload] * table.num_rows}})

        class Consumer:
            def __call__(self, table):
                return pa.table({{"id": table.column("id")}})

        _runners.set_runner_ray(noop_if_initialized=True)
        runner = _runners.get_or_create_runner()

        rel = (
            con.read_parquet(str(input_path))
            .map_batches(
                slow_producer,
                schema={{"id": vane.sqltypes.INTEGER, "payload": vane.sqltypes.VARCHAR}},
                execution_backend="ray_task",
                cpus=1.0,
                gpus=0.0,
                batch_size=2048,
                streaming_breaker=True,
            )
            .map_batches(
                Consumer,
                schema={{"id": vane.sqltypes.INTEGER}},
                execution_backend="ray_actor",
                actor_number=1,
                cpus=1.0,
                gpus=0.0,
                batch_size=32,
                streaming_breaker=True,
            )
        )

        start = time.time()
        parts = list(runner.run_iter_tables(rel, results_buffer_size=1))
        elapsed = time.time() - start
        total = 0
        for part in parts:
            table = part.to_arrow() if hasattr(part, "to_arrow") else part
            total += table.num_rows

        assert elapsed < 20, f"streaming UDF output-event wakeup stuck for {{elapsed:.2f}}s"
        assert total == row_count
        """
    )
    env = dict(ray_subprocess_env)
    env["VANE_PROGRESS"] = "0"
    env["RAY_DEDUP_LOGS"] = "0"
    env["VANE_RUNNER"] = "ray"
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


def test_ray_task_flat_map_ref_stream_preserves_rows_under_actor_backpressure(tmp_path, ray_subprocess_env):
    pytest.importorskip("pyarrow")
    import subprocess
    import sys
    import textwrap

    if ray is None:
        pytest.skip("ray not installed")

    script = textwrap.dedent(
        f"""
        from pathlib import Path

        import vane
        import pyarrow as pa
        from vane import runners as _runners

        row_count = 1500
        expansion = 12
        payload_len = 2048
        part_count = 12
        input_dir = Path({str(tmp_path / "ray_task_flat_map_backpressure_input")!r})
        input_dir.mkdir(parents=True, exist_ok=True)

        con = vane.connect()
        con.execute("SET threads=8")
        for part in range(part_count):
            lo = part * (row_count // part_count)
            hi = (part + 1) * (row_count // part_count) if part + 1 < part_count else row_count
            con.execute(
                f'''
                COPY (
                    SELECT i::INTEGER AS id, repeat('x', {{payload_len}}) AS txt
                    FROM range({{lo}}, {{hi}}) t(i)
                ) TO '{{input_dir!s}}/part_{{part}}.parquet' (FORMAT PARQUET)
                '''
            )

        def expand(row):
            text = row["txt"]
            value = row["id"]
            for chunk_id in range(expansion):
                yield {{"id": value, "chunk_id": chunk_id, "chunk": text + str(chunk_id)}}

        class Identity:
            def __call__(self, table):
                return pa.table(
                    {{
                        "id": table.column("id"),
                        "chunk_id": table.column("chunk_id"),
                        "chunk": table.column("chunk"),
                    }}
                )

        rel = (
            con.read_parquet(str(input_dir))
            .flat_map(
                expand,
                schema={{
                    "id": vane.sqltypes.INTEGER,
                    "chunk_id": vane.sqltypes.INTEGER,
                    "chunk": vane.sqltypes.VARCHAR,
                }},
                execution_backend="ray_task",
                cpus=1.0,
                gpus=0.0,
                streaming_breaker=True,
            )
            .map_batches(
                Identity,
                schema={{
                    "id": vane.sqltypes.INTEGER,
                    "chunk_id": vane.sqltypes.INTEGER,
                    "chunk": vane.sqltypes.VARCHAR,
                }},
                execution_backend="ray_actor",
                actor_number=1,
                cpus=1.0,
                gpus=0.0,
                batch_size=10,
                streaming_breaker=True,
            )
        )

        _runners.set_runner_ray(noop_if_initialized=True)
        runner = _runners.get_or_create_runner()
        count = 0
        ids = set()
        chunk_id_sum = 0
        max_chunk_len = 0
        for part in runner.run_iter_tables(rel, results_buffer_size=1):
            table = part.to_arrow() if hasattr(part, "to_arrow") else part
            count += table.num_rows
            ids.update(table.column(0).to_pylist())
            chunk_id_sum += sum(table.column(1).to_pylist())
            max_chunk_len = max(
                max_chunk_len,
                max((len(value) for value in table.column(2).to_pylist()), default=0),
            )

        assert count == row_count * expansion, (count, row_count * expansion)
        assert len(ids) == row_count
        assert chunk_id_sum == row_count * sum(range(expansion))
        assert max_chunk_len == payload_len + 2
        """
    )
    env = dict(ray_subprocess_env)
    env["VANE_PROGRESS"] = "0"
    env["RAY_DEDUP_LOGS"] = "0"
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_ray_task_flat_map_projected_ref_stream_preserves_column_projection(tmp_path, ray_subprocess_env):
    pytest.importorskip("pyarrow")
    import subprocess
    import sys
    import textwrap

    if ray is None:
        pytest.skip("ray not installed")

    script = textwrap.dedent(
        f"""
        from pathlib import Path

        import vane
        import pyarrow as pa
        from vane import runners as _runners

        row_count = 180
        expansion = 12
        input_dir = Path({str(tmp_path / "ray_task_flat_map_projected_input")!r})
        input_dir.mkdir(parents=True, exist_ok=True)

        con = vane.connect()
        con.execute("SET threads=4")
        con.execute(
            f'''
            COPY (
                SELECT i::INTEGER AS id, repeat('x', 128) AS txt
                FROM range({{row_count}}) t(i)
            ) TO '{{input_dir!s}}/part.parquet' (FORMAT PARQUET)
            '''
        )

        def expand(row):
            text = row["txt"]
            value = row["id"]
            for chunk_id in range(expansion):
                yield {{"id": value, "chunk_id": chunk_id, "chunk": text + str(chunk_id)}}

        class KeepTwo:
            def __call__(self, table):
                if table.column_names != ["id", "chunk_id"]:
                    raise RuntimeError(f"unexpected columns: {{table.column_names}}")
                return pa.table(
                    {{
                        "id": table.column("id"),
                        "chunk_id": table.column("chunk_id"),
                    }}
                )

        rel = (
            con.read_parquet(str(input_dir))
            .flat_map(
                expand,
                schema={{
                    "id": vane.sqltypes.INTEGER,
                    "chunk_id": vane.sqltypes.INTEGER,
                    "chunk": vane.sqltypes.VARCHAR,
                }},
                execution_backend="ray_task",
                cpus=1.0,
                gpus=0.0,
                streaming_breaker=True,
            )
            .project("id, chunk_id")
            .map_batches(
                KeepTwo,
                schema={{
                    "id": vane.sqltypes.INTEGER,
                    "chunk_id": vane.sqltypes.INTEGER,
                }},
                execution_backend="ray_actor",
                actor_number=1,
                cpus=1.0,
                gpus=0.0,
                batch_size=10,
                streaming_breaker=True,
            )
        )

        aggregate = rel.aggregate(
            "count(*) as c, count(distinct id) as ids, sum(chunk_id) as chunk_id_sum"
        )
        _runners.set_runner_ray(noop_if_initialized=True)
        runner = _runners.get_or_create_runner()
        parts = list(runner.run_iter_tables(aggregate, results_buffer_size=1))
        result = pa.concat_tables(
            [part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts]
        )
        assert result.num_rows == 1, result
        count, distinct_ids, chunk_id_sum = (
            result.column(index)[0].as_py() for index in range(result.num_columns)
        )

        assert count == row_count * expansion, (count, row_count * expansion)
        assert distinct_ids == row_count
        assert chunk_id_sum == row_count * sum(range(expansion))
        """
    )
    env = dict(ray_subprocess_env)
    env["VANE_PROGRESS"] = "0"
    env["RAY_DEDUP_LOGS"] = "0"
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_ray_task_flat_map_ref_stream_preserves_variable_and_empty_outputs(tmp_path, ray_subprocess_env):
    pytest.importorskip("pyarrow")
    import subprocess
    import sys
    import textwrap

    if ray is None:
        pytest.skip("ray not installed")

    script = textwrap.dedent(
        f"""
        from pathlib import Path

        import vane
        import pyarrow as pa
        from vane import runners as _runners

        row_count = 257
        input_dir = Path({str(tmp_path / "ray_task_flat_map_variable_outputs_input")!r})
        input_dir.mkdir(parents=True, exist_ok=True)

        con = vane.connect()
        con.execute("SET threads=4")
        con.execute(
            f'''
            COPY (
                SELECT i::INTEGER AS id, ('payload-' || i::VARCHAR) AS txt
                FROM range({{row_count}}) t(i)
            ) TO '{{input_dir!s}}/part.parquet' (FORMAT PARQUET)
            '''
        )

        expected_count = sum(i % 5 for i in range(row_count))
        expected_distinct_ids = sum(1 for i in range(row_count) if i % 5)
        expected_chunk_id_sum = sum(chunk_id for i in range(row_count) for chunk_id in range(i % 5))
        expected_ordinal_sum = sum(i * 100 + chunk_id for i in range(row_count) for chunk_id in range(i % 5))
        expected_max_chunk_id = max(chunk_id for i in range(row_count) for chunk_id in range(i % 5))

        def expand(row):
            value = row["id"]
            for chunk_id in range(value % 5):
                yield {{
                    "id": value,
                    "chunk_id": chunk_id,
                    "ordinal": value * 100 + chunk_id,
                    "chunk": row["txt"] + ":" + str(chunk_id),
                }}

        class Identity:
            def __call__(self, table):
                if table.column_names != ["id", "chunk_id", "ordinal", "chunk"]:
                    raise RuntimeError(f"unexpected columns: {{table.column_names}}")
                return pa.table(
                    {{
                        "id": table.column("id"),
                        "chunk_id": table.column("chunk_id"),
                        "ordinal": table.column("ordinal"),
                        "chunk": table.column("chunk"),
                    }}
                )

        rel = (
            con.read_parquet(str(input_dir))
            .flat_map(
                expand,
                schema={{
                    "id": vane.sqltypes.INTEGER,
                    "chunk_id": vane.sqltypes.INTEGER,
                    "ordinal": vane.sqltypes.INTEGER,
                    "chunk": vane.sqltypes.VARCHAR,
                }},
                execution_backend="ray_task",
                cpus=1.0,
                gpus=0.0,
                output_batch_size=17,
                streaming_breaker=True,
            )
            .map_batches(
                Identity,
                schema={{
                    "id": vane.sqltypes.INTEGER,
                    "chunk_id": vane.sqltypes.INTEGER,
                    "ordinal": vane.sqltypes.INTEGER,
                    "chunk": vane.sqltypes.VARCHAR,
                }},
                execution_backend="ray_actor",
                actor_number=1,
                cpus=1.0,
                gpus=0.0,
                batch_size=7,
                streaming_breaker=True,
            )
        )

        aggregate = rel.aggregate(
            "count(*) as c, count(distinct id) as ids, sum(chunk_id) as chunk_id_sum, "
            "sum(ordinal) as ordinal_sum, max(chunk_id) as max_chunk_id"
        )
        _runners.set_runner_ray(noop_if_initialized=True)
        runner = _runners.get_or_create_runner()
        parts = list(runner.run_iter_tables(aggregate, results_buffer_size=1))
        result = pa.concat_tables(
            [part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts]
        )
        assert result.num_rows == 1, result
        count, distinct_ids, chunk_id_sum, ordinal_sum, max_chunk_id = (
            result.column(index)[0].as_py() for index in range(result.num_columns)
        )

        assert count == expected_count, (count, expected_count)
        assert distinct_ids == expected_distinct_ids
        assert chunk_id_sum == expected_chunk_id_sum
        assert ordinal_sum == expected_ordinal_sum
        assert max_chunk_id == expected_max_chunk_id
        """
    )
    env = dict(ray_subprocess_env)
    env["VANE_PROGRESS"] = "0"
    env["RAY_DEDUP_LOGS"] = "0"
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_ray_task_flat_map_ref_stream_preserves_reordered_alias_projection(tmp_path, ray_subprocess_env):
    pytest.importorskip("pyarrow")
    import subprocess
    import sys
    import textwrap

    if ray is None:
        pytest.skip("ray not installed")

    script = textwrap.dedent(
        f"""
        from pathlib import Path

        import vane
        import pyarrow as pa
        from vane import runners as _runners

        row_count = 211
        expansion = 13
        input_dir = Path({str(tmp_path / "ray_task_flat_map_reordered_alias_input")!r})
        input_dir.mkdir(parents=True, exist_ok=True)

        con = vane.connect()
        con.execute("SET threads=4")
        con.execute(
            f'''
            COPY (
                SELECT i::INTEGER AS id, repeat('z', 64) AS txt
                FROM range({{row_count}}) t(i)
            ) TO '{{input_dir!s}}/part.parquet' (FORMAT PARQUET)
            '''
        )

        def expand(row):
            value = row["id"]
            for chunk_id in range(expansion):
                yield {{
                    "id": value,
                    "chunk_id": chunk_id,
                    "chunk": row["txt"] + str(chunk_id),
                    "weight": value * 10 + chunk_id,
                }}

        class KeepProjected:
            def __call__(self, table):
                if table.column_names != ["cid", "source_id", "weight"]:
                    raise RuntimeError(f"unexpected columns: {{table.column_names}}")
                return pa.table(
                    {{
                        "cid": table.column("cid"),
                        "source_id": table.column("source_id"),
                        "weight": table.column("weight"),
                    }}
                )

        rel = (
            con.read_parquet(str(input_dir))
            .flat_map(
                expand,
                schema={{
                    "id": vane.sqltypes.INTEGER,
                    "chunk_id": vane.sqltypes.INTEGER,
                    "chunk": vane.sqltypes.VARCHAR,
                    "weight": vane.sqltypes.INTEGER,
                }},
                execution_backend="ray_task",
                cpus=1.0,
                gpus=0.0,
                output_batch_size=31,
                streaming_breaker=True,
            )
            .project("chunk_id AS cid, id AS source_id, weight")
            .map_batches(
                KeepProjected,
                schema={{
                    "cid": vane.sqltypes.INTEGER,
                    "source_id": vane.sqltypes.INTEGER,
                    "weight": vane.sqltypes.INTEGER,
                }},
                execution_backend="ray_actor",
                actor_number=1,
                cpus=1.0,
                gpus=0.0,
                batch_size=9,
                streaming_breaker=True,
            )
        )

        aggregate = rel.aggregate(
            "count(*) as c, count(distinct source_id) as ids, sum(cid) as cid_sum, sum(weight) as weight_sum"
        )
        _runners.set_runner_ray(noop_if_initialized=True)
        runner = _runners.get_or_create_runner()
        parts = list(runner.run_iter_tables(aggregate, results_buffer_size=1))
        result = pa.concat_tables(
            [part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts]
        )
        assert result.num_rows == 1, result
        count, distinct_ids, cid_sum, weight_sum = (
            result.column(index)[0].as_py() for index in range(result.num_columns)
        )

        assert count == row_count * expansion
        assert distinct_ids == row_count
        assert cid_sum == row_count * sum(range(expansion))
        assert weight_sum == sum(i * 10 + chunk_id for i in range(row_count) for chunk_id in range(expansion))
        """
    )
    env = dict(ray_subprocess_env)
    env["VANE_PROGRESS"] = "0"
    env["RAY_DEDUP_LOGS"] = "0"
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_ray_task_flat_map_ref_stream_all_empty_output_finishes(tmp_path, ray_subprocess_env):
    pytest.importorskip("pyarrow")
    import subprocess
    import sys
    import textwrap

    if ray is None:
        pytest.skip("ray not installed")

    script = textwrap.dedent(
        f"""
        from pathlib import Path

        import vane
        import pyarrow as pa
        from vane import runners as _runners

        row_count = 97
        input_dir = Path({str(tmp_path / "ray_task_flat_map_all_empty_input")!r})
        input_dir.mkdir(parents=True, exist_ok=True)

        con = vane.connect()
        con.execute("SET threads=4")
        con.execute(
            f'''
            COPY (
                SELECT i::INTEGER AS id, ('payload-' || i::VARCHAR) AS txt
                FROM range({{row_count}}) t(i)
            ) TO '{{input_dir!s}}/part.parquet' (FORMAT PARQUET)
            '''
        )

        def expand(row):
            if False:
                yield {{"id": row["id"], "chunk": row["txt"]}}

        class Identity:
            def __call__(self, table):
                return pa.table({{"id": table.column("id"), "chunk": table.column("chunk")}})

        rel = (
            con.read_parquet(str(input_dir))
            .flat_map(
                expand,
                schema={{
                    "id": vane.sqltypes.INTEGER,
                    "chunk": vane.sqltypes.VARCHAR,
                }},
                execution_backend="ray_task",
                cpus=1.0,
                gpus=0.0,
                output_batch_size=5,
                streaming_breaker=True,
            )
            .map_batches(
                Identity,
                schema={{
                    "id": vane.sqltypes.INTEGER,
                    "chunk": vane.sqltypes.VARCHAR,
                }},
                execution_backend="ray_actor",
                actor_number=1,
                cpus=1.0,
                gpus=0.0,
                batch_size=3,
                streaming_breaker=True,
            )
        )

        aggregate = rel.aggregate("count(*) AS c")
        _runners.set_runner_ray(noop_if_initialized=True)
        runner = _runners.get_or_create_runner()
        parts = list(runner.run_iter_tables(aggregate, results_buffer_size=1))
        result = pa.concat_tables(
            [part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts]
        )
        assert result.num_rows == 1, result
        assert result.column(0)[0].as_py() == 0
        """
    )
    env = dict(ray_subprocess_env)
    env["VANE_PROGRESS"] = "0"
    env["RAY_DEDUP_LOGS"] = "0"
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_ray_python_udf_task_consumes_and_emits_ref_bundles_distributed(ray_runner, duckdb_conn, tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    if ray is None:
        pytest.skip("ray not installed")
    if float(ray.cluster_resources().get("GPU", 0.0)) < 1.0:
        pytest.skip("requires a Ray cluster with at least one GPU resource")
    monkeypatch.setenv("VANE_FUNCTION_UDF_RAY_TASK_REF_PASSTHROUGH", "1")
    label = "test_ray_e2e: python_udf task consumes and emits ref bundles"

    class Stage1:
        def __call__(self, table):
            ids = table.column("id").to_pylist()
            return pa.table({"id": ids, "score": [value * 10 for value in ids]})

    def plus_one(_id: int, score: int) -> int:
        return score + 1

    class Stage2:
        def __call__(self, table):
            if table.column_names != ["id", "bumped"]:
                message = f"unexpected scalar lazy input columns: {table.column_names}"
                raise RuntimeError(message)
            ids = table.column("id").to_pylist()
            bumped = table.column("bumped").to_pylist()
            return pa.table({"combined": [value + score for value, score in zip(ids, bumped, strict=True)]})

    input_path = tmp_path / "python_udf_ref_bundle_passthrough.parquet"
    duckdb_conn.execute(
        f"""
        COPY (
            SELECT *
            FROM (VALUES
                (1),
                (2),
                (3),
                (4)
            ) t(id)
        ) TO '{input_path!s}' (FORMAT PARQUET)
        """
    )
    base = duckdb_conn.sql(f"SELECT * FROM read_parquet('{input_path!s}')")
    produced = base.map_batches(
        Stage1,
        schema={"id": vane.sqltype("INTEGER"), "score": vane.sqltype("INTEGER")},
        execution_backend="ray_actor",
        actor_number=2,
        gpus=0.0,
        batch_size=2,
    )
    scalar = produced.map(plus_one, return_type=vane.sqltype("INTEGER"), execution_backend="ray_task").project(
        "id, value as bumped"
    )
    df = scalar.map_batches(
        Stage2,
        schema={"combined": vane.sqltype("INTEGER")},
        execution_backend="ray_actor",
        actor_number=1,
        gpus=1,
        batch_size=2,
    )

    plan = df.explain()
    assert "INOUT_FUNCTION" in plan
    assert "ray_task" in plan
    assert "strict_ref_aware" in plan
    assert "scalar_arg_count" in plan

    _log_builder_info(df)
    _log_distributed_plan(df, label)
    parts = _run_iter_tables(ray_runner, df, label, timeout_s=30.0)
    rows = _collect_result_rows(parts)

    assert set(rows) == {("12",), ("23",), ("34",), ("45",)}


def test_ray_python_udf_native(ray_runner, duckdb_conn, parquet_path):
    pytest.importorskip("pyarrow")
    if ray is None:
        pytest.skip("ray not installed")

    label = "test_ray_e2e: python_udf native"

    def plus_one(value: int) -> int:
        return value + 1

    base_sql = f"""
        SELECT a
        FROM read_parquet('{parquet_path}')
        WHERE a < 10
    """
    df = (
        duckdb_conn.sql(base_sql)
        .map(plus_one, return_type=vane.sqltype("INTEGER"), execution_backend="ray_task")
        .project("a, value AS out")
    )
    _log_builder_info(df)
    _log_distributed_plan(df, label)
    parts = _run_iter_tables(ray_runner, df, label, timeout_s=30.0)
    sql = f"""
        SELECT
            a,
            a + 1 AS out
        FROM read_parquet('{parquet_path}')
        WHERE a < 10
    """
    _assert_results_match(duckdb_conn, sql, parts, label)


def test_ray_python_udf_terminal_failure_propagates(ray_runner, duckdb_conn, parquet_path):
    pytest.importorskip("pyarrow")
    if ray is None:
        pytest.skip("ray not installed")

    def planned_failure(value: int) -> int:
        raise RuntimeError(f"planned scalar failure: {value}")

    relation = duckdb_conn.sql(f"SELECT a FROM read_parquet('{parquet_path}') WHERE a < 2").map(
        planned_failure,
        return_type=vane.sqltype("INTEGER"),
        execution_backend="ray_task",
    )

    with pytest.raises(Exception, match="planned scalar failure|FTE query .* failed"):
        list(ray_runner.run_iter_tables(relation, results_buffer_size=1))


def test_ray_python_udf_map_batches_arrow(ray_runner, duckdb_conn, parquet_path):
    pytest.importorskip("pyarrow")
    if ray is None:
        pytest.skip("ray not installed")

    label = "test_ray_e2e: python_udf map_batches arrow"

    def plus_one_arrow(table):
        import pyarrow as pa
        import pyarrow.compute as pc

        return pa.table({"a": table.column("a"), "out": pc.add(table.column("a"), 1)})

    base_sql = f"""
        SELECT a
        FROM read_parquet('{parquet_path}')
        WHERE a < 10
    """
    df = duckdb_conn.sql(base_sql).map_batches(
        plus_one_arrow,
        schema={"a": vane.sqltype("INTEGER"), "out": vane.sqltype("INTEGER")},
        execution_backend="ray_task",
        batch_size=64,
    )
    _log_builder_info(df)
    _log_distributed_plan(df, label)
    parts = _run_iter_tables(ray_runner, df, label, timeout_s=30.0)
    sql = f"""
        SELECT
            a,
            a + 1 AS out
        FROM read_parquet('{parquet_path}')
        WHERE a < 10
    """
    _assert_results_match(duckdb_conn, sql, parts, label)


def test_ray_task_credit_admission_runs_multiple_batches_concurrently(
    ray_runner,
    duckdb_conn,
    tmp_path,
):
    pytest.importorskip("pyarrow")
    if ray is None:
        pytest.skip("ray not installed")

    import pyarrow as pa

    input_path = tmp_path / "credit_admission_input.parquet"
    state_path = tmp_path / "credit_admission_state.json"
    duckdb_conn.execute(f"COPY (SELECT i::BIGINT AS x FROM range(8192) t(i)) TO '{input_path!s}' (FORMAT PARQUET)")

    def overlap_probe(table):
        import fcntl
        import json
        import time

        def update(delta):
            lock_path = f"{state_path!s}.lock"
            with open(lock_path, "w", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                try:
                    if state_path.exists():
                        state = json.loads(state_path.read_text(encoding="utf-8"))
                    else:
                        state = {"current": 0, "max": 0}
                    state["current"] += delta
                    state["max"] = max(state["max"], state["current"])
                    state_path.write_text(json.dumps(state), encoding="utf-8")
                finally:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)

        update(1)
        try:
            time.sleep(0.25)
            return pa.table({"x": table.column("x")})
        finally:
            update(-1)

    relation = duckdb_conn.sql(f"SELECT * FROM read_parquet('{input_path!s}')").map_batches(
        overlap_probe,
        schema={"x": vane.sqltype("BIGINT")},
        execution_backend="ray_task",
        cpus=1.0,
        gpus=0.0,
        batch_size=2048,
        task_input_max_bytes=1024 * 1024,
        output_target_max_bytes=1024 * 1024,
        streaming_breaker=True,
    )

    parts = _run_iter_tables(ray_runner, relation, "credit-first ray task admission", timeout_s=30.0)

    assert sum(part.num_rows for part in parts) == 8192
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["current"] == 0
    assert state["max"] >= 2


def test_ray_map_batches_udf(ray_runner, duckdb_conn, parquet_path):
    pytest.importorskip("pyarrow")
    if ray is None:
        pytest.skip("ray not installed")

    label = "test_ray_e2e: map_batches udf"

    def double_values(table):
        import pyarrow as pa

        values = table.column(0).to_pylist()
        return pa.table({"x": [value * 2 for value in values]})

    base_sql = f"""
        SELECT a
        FROM read_parquet('{parquet_path}')
        WHERE a < 10
    """
    df = duckdb_conn.sql(base_sql).map_batches(
        double_values,
        schema={"x": vane.sqltype("INTEGER")},
        execution_backend="ray_task",
        batch_size=64,
    )
    _log_builder_info(df)
    _log_distributed_plan(df, label)
    parts = _run_iter_tables(ray_runner, df, label, timeout_s=30.0)
    sql = f"""
        SELECT a * 2 AS x
        FROM read_parquet('{parquet_path}')
        WHERE a < 10
    """
    _assert_results_match(duckdb_conn, sql, parts, label)


def test_ray_python_stream_table_udf(ray_runner, duckdb_conn, parquet_path):
    pytest.importorskip("pyarrow")
    if ray is None:
        pytest.skip("ray not installed")

    label = "test_ray_e2e: python stream table udf"

    def double_values(table):
        import pyarrow as pa

        values = table.column(0).to_pylist()
        return pa.table({"x": [v * 2 for v in values]})

    base_sql = f"""
        SELECT a
        FROM read_parquet('{parquet_path}')
        WHERE a < 10
    """
    df = duckdb_conn.sql(base_sql).map_batches(
        double_values,
        schema={"x": vane.sqltype("INTEGER")},
        execution_backend="ray_task",
        batch_size=64,
    )
    _log_builder_info(df)
    _log_distributed_plan(df, label)
    parts = _run_iter_tables(ray_runner, df, label, timeout_s=30.0)

    expected_sql = f"""
        SELECT a * 2 AS x
        FROM read_parquet('{parquet_path}')
        WHERE a < 10
    """
    _assert_results_match(duckdb_conn, expected_sql, parts, label)


def test_ray_python_stream_table_udf_multi_partition_plan(ray_runner, duckdb_conn, parquet_path):
    pytest.importorskip("pyarrow")
    if ray is None:
        pytest.skip("ray not installed")

    label = "test_ray_e2e: python stream table udf multi-partition plan"

    def double_values(table):
        import pyarrow as pa

        values = table.column(0).to_pylist()
        return pa.table({"x": [v * 2 for v in values]})

    base = duckdb_conn.sql(
        f"""
        SELECT a
        FROM read_parquet('{parquet_path}')
        WHERE a < 100
        """
    )
    df = base.map_batches(
        double_values,
        schema={"x": vane.sqltype("INTEGER")},
        execution_backend="ray_task",
        batch_size=64,
    ).aggregate("x, COUNT(*) AS cnt", "x")

    _log_distributed_plan(df, label)

    parts = _run_iter_tables(ray_runner, df, label, timeout_s=30.0)
    expected_sql = f"""
        SELECT
            a * 2 AS x,
            COUNT(*) AS cnt
        FROM read_parquet('{parquet_path}')
        WHERE a < 100
        GROUP BY x
    """
    _assert_results_match(duckdb_conn, expected_sql, parts, label)


def test_ray_python_udf_multi_partition_plan(ray_runner, duckdb_conn, parquet_path, tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    if ray is None:
        pytest.skip("ray not installed")

    label = "test_ray_e2e: python_udf multi-partition plan"
    shuffle_dir = tmp_path / "duckdb_shuffle"
    monkeypatch.setenv("VANE_SHUFFLE_ALGORITHM", "flight_shuffle")
    monkeypatch.setenv("VANE_SHUFFLE_LOCAL_DIRS", str(shuffle_dir))

    def plus_one(value: int) -> int:
        return value + 1

    base = duckdb_conn.sql(
        f"""
        SELECT a
        FROM read_parquet('{parquet_path}')
        WHERE a < 100
        """
    )
    df = (
        base.map(plus_one, return_type=vane.sqltype("INTEGER"), execution_backend="ray_task")
        .project("value AS out")
        .aggregate("out, COUNT(*) AS cnt", "out")
    )
    _log_distributed_plan(df, label)
    parts = _run_iter_tables(ray_runner, df, label, timeout_s=30.0)
    sql = f"""
        SELECT
            a + 1 AS out,
            COUNT(*) AS cnt
        FROM read_parquet('{parquet_path}')
        WHERE a < 100
        GROUP BY out
    """
    _assert_results_match(duckdb_conn, sql, parts, label)


def test_ray_task_map_batches_worker_process(ray_runner, duckdb_conn, tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    if ray is None:
        pytest.skip("ray not installed")

    base = 100000

    def pid_tag_fn(table):
        values = table.column(0).to_pylist()
        pid = os.getpid()
        return pa.table({"pid_tag": [pid * base + v for v in values]})

    input_path = tmp_path / "ray_task_worker_process.parquet"
    duckdb_conn.execute(f"COPY (SELECT * FROM (VALUES (1), (2), (3)) t(x)) TO '{input_path!s}' (FORMAT PARQUET)")
    rel = duckdb_conn.sql(f"SELECT x FROM read_parquet('{input_path!s}')")
    out = rel.map_batches(
        pid_tag_fn,
        schema={"pid_tag": vane.sqltypes.BIGINT},
        execution_backend="ray_task",
    )
    label = "test_ray_e2e: ray task map_batches worker process"
    parts = _run_iter_tables(ray_runner, out, label, timeout_s=30.0)
    rows = _collect_result_rows(parts)

    assert len(rows) == 3
    pids = set()
    for (tag,) in rows:
        pids.add(int(tag) // base)
    assert pids
    assert os.getpid() not in pids


def test_ray_task_map_batches_worker_parquet_scan_filter_projection(
    ray_runner,
    duckdb_conn,
    partitioned_parquet_path,
):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    if ray is None:
        pytest.skip("ray not installed")

    def add_offset(table):
        grps = table.column("grp").to_pylist()
        a_vals = table.column("a").to_pylist()
        b_vals = table.column("b").to_pylist()
        return pa.table(
            {
                "grp": grps,
                "a": a_vals,
                "b": b_vals,
                "b_plus": [v + 7 for v in b_vals],
            }
        )

    sql = f"""
        SELECT
            grp,
            a,
            b
        FROM read_parquet('{partitioned_parquet_path}/*/*.parquet', hive_partitioning=1)
        WHERE grp IN (1, 2, 3) AND a < 50
    """
    rel = duckdb_conn.sql(sql)
    out = rel.map_batches(
        add_offset,
        schema={
            "grp": vane.sqltype("INTEGER"),
            "a": vane.sqltype("INTEGER"),
            "b": vane.sqltype("INTEGER"),
            "b_plus": vane.sqltype("INTEGER"),
        },
        execution_backend="ray_task",
    )
    label = "test_ray_e2e: ray task worker parquet scan/filter/projection"
    parts = _run_iter_tables(ray_runner, out, label, timeout_s=30.0)
    rows = _collect_result_rows(parts)

    expected = duckdb_conn.execute(
        f"""
        SELECT
            grp,
            a,
            b,
            b + 7 AS b_plus
        FROM read_parquet('{partitioned_parquet_path}/*/*.parquet', hive_partitioning=1)
        WHERE grp IN (1, 2, 3) AND a < 50
        """
    ).fetchall()
    expected = [tuple(_stringify_value(value) for value in row) for row in expected]

    assert sorted(rows) == sorted(expected)


def test_ray_unnest(ray_runner, duckdb_conn, parquet_path):
    label = "test_ray_e2e: unnest"
    sql = f"""
        SELECT
            a,
            UNNEST([a, a + 1]) AS u
        FROM read_parquet('{parquet_path}')
        WHERE a >= 100 AND a < 105
    """
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_all=["UNNEST"],
    )


def test_ray_reservoir_sample(ray_runner, duckdb_conn, parquet_path):
    label = "test_ray_e2e: reservoir sample"
    sample_rows = RAY_E2E_ROWS * 2
    sql = f"""
        SELECT a, b, c
        FROM read_parquet('{parquet_path}')
        TABLESAMPLE RESERVOIR({sample_rows} ROWS)
        WHERE a >= 100 AND a < 120
    """
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_all=["RESERVOIR_SAMPLE"],
    )


def test_ray_streaming_sample(ray_runner, duckdb_conn, parquet_path):
    label = "test_ray_e2e: streaming sample"
    sql = f"""
        SELECT a, b, c
        FROM read_parquet('{parquet_path}')
        TABLESAMPLE SYSTEM(100 PERCENT)
        WHERE a >= 100 AND a < 120
    """
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_all=["STREAMING_SAMPLE"],
    )


def test_ray_group_by(ray_runner, duckdb_conn, parquet_path):
    label = "test_ray_e2e: group by"
    duckdb_conn.execute("SET perfect_ht_threshold=0")
    sql = f"""
        SELECT
            a,
            COUNT(*) AS cnt,
            SUM(b) AS sum_b,
            MIN(c) AS min_c,
            MAX(c) AS max_c
        FROM read_parquet('{parquet_path}')
        WHERE a >= 100 AND a < 900
        GROUP BY a
    """
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_any=["GROUP_BY", "AGGREGATE", "HASH_GROUP_BY"],
    )


def test_ray_group_by_multi_partition_plan(ray_runner, duckdb_conn, parquet_path):
    label = "test_ray_e2e: group by multi-partition plan"
    duckdb_conn.execute("SET perfect_ht_threshold=0")
    sql = f"""
        SELECT
            a,
            COUNT(*) AS cnt,
            SUM(b) AS sum_b,
            MIN(c) AS min_c,
            MAX(c) AS max_c
        FROM read_parquet('{parquet_path}')
        WHERE a >= 100 AND a < 900
        GROUP BY a
    """
    df = duckdb_conn.sql(sql)
    _log_distributed_plan(df, label)
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_any=["GROUP_BY", "AGGREGATE", "HASH_GROUP_BY"],
    )


@pytest.mark.skip(reason="requires a multi-node Ray cluster; Vane creates one worker per Ray node")
def test_ray_group_by_fte_worker_loss_chaos(ray_runner_local_cluster, duckdb_conn, parquet_path, tmp_path, monkeypatch):
    label = "test_ray_e2e: group by FTE worker-loss chaos"
    shuffle_dir = tmp_path / "duckdb_shuffle_chaos"
    monkeypatch.setenv("VANE_FTE_STATUS_WAIT_TIMEOUT_S", "1")
    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_INITIAL_BACKOFF_S", "0")
    monkeypatch.setenv("VANE_FTE_SPLIT_QUEUE_SPACE_WAIT_TIMEOUT_S", "0.1")
    monkeypatch.setenv("VANE_FTE_CHAOS_KILL_WORKER_ON_RUNNING", "1")
    monkeypatch.setenv("VANE_FTE_CHAOS_KILL_WORKER_INDEX", "0")
    monkeypatch.setenv("VANE_FTE_CHAOS_KILL_ATTEMPT_ID", "0")
    monkeypatch.setenv("VANE_SHUFFLE_ALGORITHM", "flight_shuffle")
    monkeypatch.setenv("VANE_SHUFFLE_LOCAL_DIRS", str(shuffle_dir))
    duckdb_conn.execute("SET perfect_ht_threshold=0")
    sql = f"""
        SELECT
            a,
            COUNT(*) AS cnt,
            SUM(b) AS sum_b,
            MIN(c) AS min_c,
            MAX(c) AS max_c
        FROM read_parquet('{parquet_path}')
        WHERE a >= 100 AND a < 900
        GROUP BY a
    """
    df = duckdb_conn.sql(sql)
    _log_distributed_plan(df, label)
    _run_query_case(
        duckdb_conn,
        ray_runner_local_cluster,
        sql,
        label,
        require_any=["GROUP_BY", "AGGREGATE", "HASH_GROUP_BY"],
        timeout_s=60.0,
    )


@pytest.mark.skip(reason="requires a multi-node Ray cluster; Vane creates one worker per Ray node")
def test_ray_group_by_fte_host_loss_chaos(ray_runner_local_cluster, duckdb_conn, parquet_path, tmp_path, monkeypatch):
    label = "test_ray_e2e: group by FTE host-loss chaos"
    shuffle_dir = tmp_path / "duckdb_shuffle_host_chaos"
    monkeypatch.setenv("VANE_FTE_STATUS_WAIT_TIMEOUT_S", "1")
    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_INITIAL_BACKOFF_S", "0")
    monkeypatch.setenv("VANE_FTE_SPLIT_QUEUE_SPACE_WAIT_TIMEOUT_S", "0.1")
    monkeypatch.setenv("VANE_FTE_CHAOS_KILL_WORKER_ON_RUNNING", "1")
    monkeypatch.setenv("VANE_FTE_CHAOS_FAIL_HOST_ON_WORKER_LOSS", "1")
    monkeypatch.setenv("VANE_FTE_CHAOS_KILL_WORKER_INDEX", "0")
    monkeypatch.setenv("VANE_FTE_CHAOS_KILL_ATTEMPT_ID", "0")
    monkeypatch.setenv("VANE_SHUFFLE_ALGORITHM", "flight_shuffle")
    monkeypatch.setenv("VANE_SHUFFLE_LOCAL_DIRS", str(shuffle_dir))
    duckdb_conn.execute("SET perfect_ht_threshold=0")
    sql = f"""
        SELECT
            a % 64 AS grp,
            COUNT(*) AS cnt,
            SUM(b) AS sum_b,
            MIN(c) AS min_c,
            MAX(c) AS max_c
        FROM read_parquet('{parquet_path}')
        WHERE a >= 100 AND a < 900
        GROUP BY grp
    """
    df = duckdb_conn.sql(sql)
    _log_distributed_plan(df, label)
    _run_query_case(
        duckdb_conn,
        ray_runner_local_cluster,
        sql,
        label,
        require_any=["GROUP_BY", "AGGREGATE", "HASH_GROUP_BY"],
        timeout_s=90.0,
    )


def test_ray_perfect_hash_group_by(ray_runner, duckdb_conn, parquet_path):
    label = "test_ray_e2e: perfect hash group by"
    duckdb_conn.execute("SET perfect_ht_threshold=32")
    sql = f"""
        SELECT
            a,
            COUNT(*) AS cnt,
            SUM(b) AS sum_b
        FROM read_parquet('{parquet_path}')
        GROUP BY a
    """
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_all=["PERFECT_HASH_GROUP_BY"],
    )


def test_ray_partitioned_aggregate(ray_runner, duckdb_conn, partitioned_parquet_path):
    label = "test_ray_e2e: partitioned aggregate"
    sql = f"""
        SELECT
            grp,
            COUNT(*) AS cnt,
            SUM(b) AS sum_b
        FROM read_parquet('{partitioned_parquet_path}/*/*.parquet', hive_partitioning=1)
        GROUP BY grp
    """
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_all=["PARTITIONED_AGGREGATE"],
    )


def test_ray_cte_values(ray_runner, duckdb_conn):
    label = "test_ray_e2e: cte values"
    sql = """
        WITH vals(a, b) AS (
            VALUES (1, 10), (2, 20), (3, 30)
        )
        SELECT a, b, a + b AS sum_ab
        FROM vals
        WHERE a >= 2
    """
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_any=["COLUMN_DATA_SCAN", "EXPRESSION_SCAN"],
    )


def test_ray_pivot(ray_runner, duckdb_conn, parquet_path):
    label = "test_ray_e2e: pivot"
    duckdb_conn.execute("SET pivot_filter_threshold=0")
    sql = f"""
        SELECT *
        FROM (
            SELECT
                a % 3 AS grp,
                a % 2 AS k,
                b
            FROM read_parquet('{parquet_path}')
            WHERE a < 6
        ) PIVOT (SUM(b) FOR k IN (0, 1))
    """
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_all=["PIVOT"],
    )


def test_ray_join(ray_runner, duckdb_conn, parquet_path):
    label = "test_ray_e2e: join"
    sql = f"""
        SELECT
            l.a,
            l.b,
            r.c
        FROM read_parquet('{parquet_path}') l
        JOIN read_parquet('{parquet_path}') r
          ON l.a = r.a
        WHERE l.a >= 100 AND l.a < 200
    """
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_all=["HASH_JOIN"],
    )


def test_ray_join_multi_partition_plan(ray_runner, duckdb_conn, parquet_path):
    label = "test_ray_e2e: join multi-partition plan"
    sql = f"""
        SELECT
            l.a,
            l.b,
            r.c
        FROM read_parquet('{parquet_path}') l
        JOIN read_parquet('{parquet_path}') r
          ON l.a = r.a
        WHERE l.a >= 100 AND l.a < 200
    """
    df = duckdb_conn.sql(sql)
    _log_distributed_plan(df, label)
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_all=["HASH_JOIN"],
    )


@pytest.mark.usefixtures("ray_runner")
def test_ray_join_broadcast_plan(duckdb_conn, parquet_path, monkeypatch):
    label = "test_ray_e2e: join broadcast plan"
    monkeypatch.setenv("VANE_DISTRIBUTED_JOIN_STRATEGY", "broadcast")
    sql = f"""
        SELECT
            l.a,
            l.b,
            r.c
        FROM read_parquet('{parquet_path}') l
        JOIN read_parquet('{parquet_path}') r
          ON l.a = r.a
        WHERE l.a >= 100 AND l.a < 200
    """
    df = duckdb_conn.sql(sql)
    plan_text, num_parts = _get_distributed_plan_info(df, label)
    assert num_parts is not None, f"{label}: expected num_partitions to be available"
    assert plan_text and "BROADCAST JOIN" in plan_text.upper(), f"{label}: expected broadcast join in distributed plan"


def test_ray_join_broadcast_multi_partition_plan(ray_runner, duckdb_conn, parquet_path, monkeypatch):
    label = "test_ray_e2e: join broadcast multi-partition plan"
    monkeypatch.setenv("VANE_DISTRIBUTED_JOIN_STRATEGY", "broadcast")
    monkeypatch.setenv("VANE_DISTRIBUTED_BROADCAST_JOIN_RECEIVER_REPARTITION", "0")
    sql = f"""
        SELECT
            l.a,
            l.b,
            r.c
        FROM read_parquet('{parquet_path}') l
        JOIN read_parquet('{parquet_path}') r
          ON l.a = r.a
        WHERE l.a >= 100 AND l.a < 200
    """
    df = duckdb_conn.sql(sql)
    _log_distributed_plan(df, label)
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_all=["HASH_JOIN"],
    )


def test_ray_chained_broadcast_joins_keep_materialize_outputs_scoped(
    ray_runner, duckdb_conn, parquet_path, monkeypatch
):
    label = "test_ray_e2e: chained broadcast join materialize output scoping"
    monkeypatch.setenv("VANE_DISTRIBUTED_JOIN_STRATEGY", "broadcast")
    monkeypatch.setenv("VANE_DISTRIBUTED_BROADCAST_JOIN_RECEIVER_REPARTITION", "0")
    sql = f"""
        SELECT count(*) AS c
        FROM read_parquet('{parquet_path}') fact
        JOIN (SELECT a, b FROM read_parquet('{parquet_path}') WHERE a < 10) d1
          ON fact.a = d1.a
        JOIN (SELECT a, c FROM read_parquet('{parquet_path}') WHERE a < 10) d2
          ON fact.a = d2.a
        JOIN (SELECT a FROM read_parquet('{parquet_path}') WHERE a < 10) d3
          ON fact.a = d3.a
        WHERE fact.a < 10
    """
    df = duckdb_conn.sql(sql)
    plan_text, _ = _get_distributed_plan_info(df, label)
    assert plan_text and plan_text.upper().count("BROADCAST JOIN") >= 3, (
        f"{label}: expected a chained broadcast join distributed plan"
    )
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_all=["HASH_JOIN"],
        timeout_s=30.0,
    )


def test_ray_left_delim_join(ray_runner, duckdb_conn, parquet_path):
    label = "test_ray_e2e: left delim join"
    sql = f"""
        WITH t AS (
            SELECT a AS id, b AS val
            FROM read_parquet('{parquet_path}')
            WHERE a IN (1, 2, 3)
        ),
        t2 AS (
            SELECT a AS id, b AS val
            FROM read_parquet('{parquet_path}')
            WHERE a IN (1, 2, 4)
        )
        SELECT id
        FROM t
        WHERE (SELECT COUNT(*) FROM t2 WHERE t2.id = t.id) > 0
    """
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_all=["LEFT_DELIM_JOIN"],
    )


def test_ray_left_delim_join_multi_partition_plan(ray_runner, duckdb_conn, parquet_path):
    label = "test_ray_e2e: left delim join multi-partition plan"
    sql = f"""
        WITH t AS (
            SELECT a AS id, b AS val
            FROM read_parquet('{parquet_path}')
            WHERE a IN (1, 2, 3)
        ),
        t2 AS (
            SELECT a AS id, b AS val
            FROM read_parquet('{parquet_path}')
            WHERE a IN (1, 2, 4)
        )
        SELECT id, COUNT(*) AS cnt
        FROM t
        WHERE (SELECT COUNT(*) FROM t2 WHERE t2.id = t.id) > 0
        GROUP BY id
    """
    df = duckdb_conn.sql(sql)
    _log_distributed_plan(df, label)
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_all=["LEFT_DELIM_JOIN"],
        require_any=["HASH_GROUP_BY", "PERFECT_HASH_GROUP_BY"],
    )


def test_ray_inout_function(ray_runner, duckdb_conn, parquet_path):
    label = "test_ray_e2e: inout function"
    duckdb_conn.execute("SET scalar_subquery_error_on_multiple_rows=false")
    sql = f"""
        SELECT *
        FROM unnest((
            SELECT [a, b] AS lst
            FROM read_parquet('{parquet_path}')
            WHERE a < 5
        ))
    """
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_all=["INOUT_FUNCTION"],
    )


def test_ray_join_flight_shuffle_exchange(ray_runner, duckdb_conn, parquet_path, tmp_path, monkeypatch):
    label = "test_ray_e2e: join flight shuffle exchange"
    shuffle_dir = tmp_path / "duckdb_shuffle"
    monkeypatch.setenv("VANE_SHUFFLE_ALGORITHM", "flight_shuffle")
    monkeypatch.setenv("VANE_SHUFFLE_LOCAL_DIRS", str(shuffle_dir))
    monkeypatch.setenv("VANE_DISTRIBUTED_JOIN_STRATEGY", "hash")
    sql = f"""
        SELECT
            l.a,
            l.b,
            r.c
        FROM read_parquet('{parquet_path}') l
        JOIN read_parquet('{parquet_path}') r
          ON l.a = r.a
        WHERE l.a >= 100 AND l.a < 200
    """
    df = duckdb_conn.sql(sql)
    _log_distributed_plan(df, label)
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_all=["HASH_JOIN"],
    )


@pytest.mark.external_service
def test_ray_group_by_flight_shuffle_exchange_minio_durable(ray_runner, duckdb_conn, tmp_path, monkeypatch):
    label = "test_ray_e2e: group by flight shuffle exchange minio durable"
    endpoint, access_key, secret_key, region, bucket, shuffle_uri = _minio_shuffle_config()
    _skip_unless_minio_writable(endpoint, access_key, secret_key, region, bucket)

    monkeypatch.setenv("AWS_ENDPOINT_URL", endpoint)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", access_key)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", secret_key)
    monkeypatch.setenv("AWS_REGION", region)
    monkeypatch.setenv("AWS_DEFAULT_REGION", region)
    monkeypatch.setenv("DUCKDB_SHUFFLE_DIRS", shuffle_uri)
    monkeypatch.setenv("VANE_SHUFFLE_LOCAL_DIRS", shuffle_uri)
    monkeypatch.setenv("VANE_SHUFFLE_ALGORITHM", "flight_shuffle")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "8")
    monkeypatch.setenv("VANE_RAY_SCAN_TASK_SIZE_GROUPING", "0")
    monkeypatch.setenv("VANE_RAY_SCAN_TASK_MIN_BYTES", "1GB")
    monkeypatch.setenv("VANE_RAY_SCAN_TASK_MAX_BYTES", "2GB")

    _configure_conn_for_s3(duckdb_conn, endpoint, access_key, secret_key, region)
    duckdb_conn.execute("SET perfect_ht_threshold=0")
    input_path = tmp_path / "minio_durable_group_by_input"
    duckdb_conn.execute(f"""
        COPY (
            SELECT
                (i % 128)::INTEGER AS a,
                (i * 10)::INTEGER AS b,
                (i % 8)::INTEGER AS file_id
            FROM range(0, 1024) AS t(i)
        ) TO '{input_path}' (FORMAT PARQUET, PARTITION_BY (file_id))
    """)
    sql = f"""
        SELECT
            a,
            COUNT(*) AS cnt,
            SUM(b) AS sum_b
        FROM read_parquet('{input_path}/**/*.parquet')
        GROUP BY a
    """
    df = duckdb_conn.sql(sql)
    plan_text, num_parts = _get_distributed_plan_info(df, label)
    assert num_parts is not None and num_parts >= 4
    assert plan_text and "REPARTITION" in plan_text.upper(), (
        f"{label}: expected distributed plan to contain repartition exchange, got:\n{plan_text}"
    )
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_any=["GROUP_BY", "AGGREGATE", "HASH_GROUP_BY"],
        timeout_s=60.0,
    )

    shuffle_objects = [row[0] for row in duckdb_conn.execute(f"SELECT file FROM glob('{shuffle_uri}/**')").fetchall()]
    assert any(path.endswith("/manifest.txt") for path in shuffle_objects), (
        f"{label}: expected committed exchange manifest objects under {shuffle_uri}, got {shuffle_objects}"
    )
    assert not any(path.endswith(".tmp") for path in shuffle_objects), (
        f"{label}: expected no temporary object-storage exchange files, got {shuffle_objects}"
    )


def test_ray_limit(ray_runner, duckdb_conn, parquet_path):
    label = "test_ray_e2e: limit"
    duckdb_conn.execute("SET threads=4")
    duckdb_conn.execute("SET preserve_insertion_order=true")
    sql = f"""
        SELECT a, b, c
        FROM read_parquet('{parquet_path}')
        WHERE a >= 100 AND a < 900
        LIMIT 50 OFFSET 10
    """
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_any=["LIMIT", "STREAMING_LIMIT"],
        ordered=True,
    )


def test_ray_streaming_limit(ray_runner, duckdb_conn, parquet_path):
    label = "test_ray_e2e: streaming limit"
    duckdb_conn.execute("SET preserve_insertion_order=false")
    sql = f"""
        SELECT a, b, c
        FROM read_parquet('{parquet_path}')
        LIMIT 50
    """
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_all=["STREAMING_LIMIT"],
        ordered=True,
    )


def test_ray_limit_percent(ray_runner, duckdb_conn, parquet_path):
    label = "test_ray_e2e: limit percent"
    sql = f"""
        SELECT a, b, c
        FROM read_parquet('{parquet_path}')
        LIMIT 10 PERCENT
    """
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_all=["LIMIT_PERCENT"],
        ordered=True,
    )


def test_ray_order_by(ray_runner, duckdb_conn, parquet_path):
    label = "test_ray_e2e: order by"
    sql = f"""
        SELECT a, b, c
        FROM read_parquet('{parquet_path}')
        ORDER BY b
    """
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_all=["ORDER_BY"],
        ordered=True,
    )


def test_ray_order_by_partitioned_global(ray_runner, duckdb_conn, partitioned_parquet_path):
    label = "test_ray_e2e: partitioned global order by"
    sql = f"""
        SELECT grp, a, b, c
        FROM read_parquet('{partitioned_parquet_path}/*/*.parquet', hive_partitioning=1)
        ORDER BY b DESC, grp ASC
    """
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_all=["ORDER_BY"],
        ordered=True,
    )


def test_ray_top_n(ray_runner, duckdb_conn, parquet_path):
    label = "test_ray_e2e: top n"
    sql = f"""
        SELECT a, b, c
        FROM read_parquet('{parquet_path}')
        ORDER BY b DESC
        LIMIT 20
    """
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_all=["TOP_N"],
        ordered=True,
    )


def test_ray_window(ray_runner, duckdb_conn, parquet_path):
    label = "test_ray_e2e: window"
    sql = f"""
        SELECT
            a,
            b,
            ROW_NUMBER() OVER (PARTITION BY a % 10 ORDER BY b) AS rn
        FROM read_parquet('{parquet_path}')
        WHERE a >= 100 AND a < 200
    """
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_all=["WINDOW"],
    )


def test_ray_streaming_window(ray_runner, duckdb_conn, parquet_path):
    label = "test_ray_e2e: streaming window"
    sql = f"""
        SELECT
            a,
            b,
            ROW_NUMBER() OVER () AS rn
        FROM read_parquet('{parquet_path}')
        WHERE a = 101
    """
    _run_query_case(
        duckdb_conn,
        ray_runner,
        sql,
        label,
        require_all=["STREAMING_WINDOW"],
    )

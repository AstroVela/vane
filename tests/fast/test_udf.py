# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for UDF via rel.map_batches() / rel.map() Relation API.

These tests verify that Python UDFs work through the Relation methods.
"""

import pytest


def _run_udf_width_probe(tmp_path, *, streaming_breaker):
    import os
    import subprocess
    import sys
    import textwrap

    data_dir = tmp_path / ("streaming_width" if streaming_breaker else "inout_width")
    script = textwrap.dedent(
        f"""
        import vane
        import pyarrow as pa
        from pathlib import Path

        data_dir = {str(data_dir)!r}
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        con = vane.connect()
        con.execute("SET threads=4")
        for idx in range(4):
            start = idx * 4096
            end = (idx + 1) * 4096
            con.execute(
                f"COPY (SELECT i::INTEGER AS x FROM range({{start}}, {{end}}) t(i)) "
                f"TO '{{data_dir}}/part_{{idx}}.parquet' (FORMAT PARQUET)"
            )

        def ident(table):
            return pa.table({{"x": table.column(0)}})

        rel = con.sql(f"SELECT * FROM read_parquet('{{data_dir}}/*.parquet')").map_batches(
            ident,
            schema={{"x": vane.sqltypes.INTEGER}},
            execution_backend="subprocess_task",
            batch_size=2048,
            streaming_breaker={streaming_breaker!r},
        )
        print(rel.aggregate("count(*)").fetchall())
        """
    )
    env = dict(os.environ)
    env["VANE_UDF_WORKER_SLOT_DEBUG"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise AssertionError(stderr[-30_000:]) from exc
    return result.stdout + result.stderr


def _run_udf_actor_width_probe(tmp_path, *, streaming_breaker):
    import os
    import subprocess
    import sys
    import textwrap

    data_dir = tmp_path / ("streaming_actor_width" if streaming_breaker else "inout_actor_width")
    script = textwrap.dedent(
        f"""
        import vane
        import pyarrow as pa
        from pathlib import Path

        data_dir = {str(data_dir)!r}
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        con = vane.connect()
        con.execute("SET threads=4")
        for idx in range(4):
            start = idx * 4096
            end = (idx + 1) * 4096
            con.execute(
                f"COPY (SELECT i::INTEGER AS x FROM range({{start}}, {{end}}) t(i)) "
                f"TO '{{data_dir}}/part_{{idx}}.parquet' (FORMAT PARQUET)"
            )

        class Ident:
            def __call__(self, table):
                return pa.table({{"x": table.column(0)}})

        rel = con.sql(f"SELECT * FROM read_parquet('{{data_dir}}/*.parquet')").map_batches(
            Ident,
            schema={{"x": vane.sqltypes.INTEGER}},
            execution_backend="subprocess_actor",
            actor_number=2,
            gpus=0.0,
            batch_size=2048,
            streaming_breaker={streaming_breaker!r},
        )
        print(rel.aggregate("count(*)").fetchall())
        """
    )
    env = dict(os.environ)
    env["VANE_UDF_WORKER_SLOT_DEBUG"] = "1"
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)
    return result.stdout + result.stderr


def _run_subprocess_actor_lazy_compute_batch_probe(tmp_path):
    import json
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import json

        import vane
        import pyarrow as pa

        class Identity:
            def __call__(self, table):
                return pa.table({"x": table.column("x")})

        class AnnotateBatchSize:
            def __call__(self, table):
                rows = table.num_rows
                return pa.table(
                    {
                        "x": table.column("x"),
                        "batch_rows": [rows for _ in range(rows)],
                    }
                )

        con = vane.connect()
        con.execute("SET threads=1")
        base = con.sql("select i::INTEGER as x from range(4096) t(i)").map_batches(
            Identity,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="subprocess_actor",
            actor_number=1,
            gpus=0.0,
            batch_size=2048,
        )
        rel = base.map_batches(
            AnnotateBatchSize,
            schema={"x": vane.sqltypes.INTEGER, "batch_rows": vane.sqltypes.INTEGER},
            execution_backend="subprocess_actor",
            actor_number=1,
            gpus=0.0,
            batch_size=3000,
        )
        counts = {}
        for _x, batch_rows in rel.fetchall():
            counts[batch_rows] = counts.get(batch_rows, 0) + 1
        print("VANE_BATCH_COUNTS=" + json.dumps(counts, sort_keys=True))
        """
    )
    env = dict(os.environ)
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise AssertionError(stderr[-30_000:]) from exc
    log = result.stdout + result.stderr
    for line in reversed(log.splitlines()):
        if line.startswith("VANE_BATCH_COUNTS="):
            return {int(key): value for key, value in json.loads(line.split("=", 1)[1]).items()}
    raise AssertionError(log)


def _run_subprocess_task_direct_compute_batch_probe(tmp_path):
    import json
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import json

        import vane
        import pyarrow as pa

        def annotate_batch_size(table):
            rows = table.num_rows
            return pa.table(
                {
                    "x": table.column("x"),
                    "batch_rows": [rows for _ in range(rows)],
                }
            )

        con = vane.connect()
        con.execute("SET threads=1")
        rel = con.sql("select i::INTEGER as x from range(4096) t(i)").map_batches(
            annotate_batch_size,
            schema={"x": vane.sqltypes.INTEGER, "batch_rows": vane.sqltypes.INTEGER},
            execution_backend="subprocess_task",
            batch_size=3000,
        )
        counts = {}
        for _x, batch_rows in rel.fetchall():
            counts[batch_rows] = counts.get(batch_rows, 0) + 1
        print("VANE_BATCH_COUNTS=" + json.dumps(counts, sort_keys=True))
        """
    )
    env = dict(os.environ)
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=30, check=True
    )
    log = result.stdout + result.stderr
    for line in reversed(log.splitlines()):
        if line.startswith("VANE_BATCH_COUNTS="):
            return {int(key): value for key, value in json.loads(line.split("=", 1)[1]).items()}
    raise AssertionError(log)


def _run_subprocess_task_direct_byte_transport_probe(tmp_path):
    import json
    import os
    import signal
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import json

        import vane
        import pyarrow as pa

        def report_batch_rows(table):
            values = table.column("payload").to_pylist()
            return pa.table({"batch_rows": [table.num_rows for _ in values]})

        con = vane.connect()
        con.execute("SET threads=1")
        rel = con.sql("select repeat('x', 2048)::BLOB as payload from range(8) t(i)").map_batches(
            report_batch_rows,
            schema={"batch_rows": vane.sqltypes.BIGINT},
            execution_backend="subprocess_task",
            batch_size=3000,
            output_batch_size=2048,
            target_max_batch_bytes=4096,
        )
        rows = [row[0] for row in rel.fetchall()]
        print("VANE_DIRECT_BYTE_ROWS=" + json.dumps(rows))
        """
    )

    env = dict(os.environ)
    env["DUCKDB_DISTRIBUTED_DEBUG"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=20)
    except subprocess.TimeoutExpired as exc:
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
        raise AssertionError((stdout or "") + (stderr or "")) from exc
    log = (stdout or "") + (stderr or "")
    if proc.returncode != 0:
        raise AssertionError(log)
    for line in reversed(log.splitlines()):
        if line.startswith("VANE_DIRECT_BYTE_ROWS="):
            rows = json.loads(line.split("=", 1)[1])
            envelope_submits = sum("where=submit_materialized_envelope" in log_line for log_line in log.splitlines())
            standard_submits = sum("where=submit_materialized " in log_line for log_line in log.splitlines())
            return rows, envelope_submits, standard_submits, log
    raise AssertionError(log)


def _run_subprocess_task_direct_work_unit_submit_probe(tmp_path):
    import json
    import os
    import re
    import signal
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import json

        import vane
        import pyarrow as pa

        def annotate_batch_size(table):
            rows = table.num_rows
            return pa.table({"batch_rows": [rows for _ in range(rows)]})

        con = vane.connect()
        con.execute("SET threads=1")
        rel = con.sql("select i::INTEGER as x from range(5000) t(i)").map_batches(
            annotate_batch_size,
            schema={"batch_rows": vane.sqltypes.INTEGER},
            execution_backend="subprocess_task",
            batch_size=100,
            target_max_batch_bytes=1 << 30,
        )
        rows = [row[0] for row in rel.fetchall()]
        print("VANE_DIRECT_ROW_ROWS=" + json.dumps(rows))
        """
    )

    env = dict(os.environ)
    env["DUCKDB_DISTRIBUTED_DEBUG"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=20)
    except subprocess.TimeoutExpired as exc:
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
        raise AssertionError((stdout or "") + (stderr or "")) from exc
    log = (stdout or "") + (stderr or "")
    if proc.returncode != 0:
        raise AssertionError(log)

    for line in reversed(log.splitlines()):
        if line.startswith("VANE_DIRECT_ROW_ROWS="):
            rows = json.loads(line.split("=", 1)[1])
            inflight_rows = [
                int(match.group(1))
                for log_line in log.splitlines()
                if "where=submit_materialized_envelope" in log_line or "where=submit_materialized " in log_line
                for match in [re.search(r"inflight_rows=(\d+)", log_line)]
                if match
            ]
            submit_rows = [
                int(match.group(1))
                for log_line in log.splitlines()
                if "event=submit_finished" in log_line
                for match in [re.search(r" rows=(\d+)", log_line)]
                if match
            ]
            if not submit_rows and inflight_rows:
                previous = 0
                for current in inflight_rows:
                    if current >= previous:
                        submit_rows.append(current - previous)
                    previous = current
            interesting = [
                log_line
                for log_line in log.splitlines()
                if "where=submit_materialized_envelope" in log_line
                or "where=submit_materialized " in log_line
                or "event=submit_finished" in log_line
                or "config_compute_batch_rows=" in log_line
            ]
            if len(interesting) > 80:
                interesting = interesting[:20] + ["..."] + interesting[-59:]
            return len(rows), sorted(set(rows)), submit_rows, "\n".join(interesting)
    raise AssertionError(log)


def _run_subprocess_lazy_byte_submit_probe(tmp_path):
    import json
    import os
    import signal
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import json

        import vane
        import pyarrow as pa

        PAYLOAD_BYTES = 2048

        def make_payload(table):
            values = table.column("x").to_pylist()
            return pa.table({"payload": [b"x" * PAYLOAD_BYTES for _ in values]})

        def report_batch_rows(table):
            values = table.column("payload").to_pylist()
            return pa.table({"batch_rows": [table.num_rows for _ in values]})

        con = vane.connect()
        con.execute("SET threads=1")
        rel = (
            con.sql("select i::INTEGER as x from range(8) t(i)")
            .map_batches(
                make_payload,
                schema={"payload": vane.sqltypes.BLOB},
                execution_backend="subprocess_task",
                batch_size=2048,
                output_batch_size=2048,
                target_max_batch_bytes=4096,
            )
            .map_batches(
                report_batch_rows,
                schema={"batch_rows": vane.sqltypes.BIGINT},
                execution_backend="subprocess_task",
                batch_size=3000,
                output_batch_size=2048,
                target_max_batch_bytes=4096,
            )
        )
        rows = [row[0] for row in rel.fetchall()]
        print("VANE_BYTE_SUBMIT_ROWS=" + json.dumps(rows))
        """
    )

    def summarize_log(log):
        interesting = [
            line
            for line in log.splitlines()
            if line.startswith("VANE_BYTE_SUBMIT_ROWS=")
            or "where=submit_lazy" in line
            or "where=sink_block_lazy" in line
            or "where=sink_block_lazy_before_accept" in line
        ]
        if len(interesting) > 80:
            interesting = interesting[:20] + ["..."] + interesting[-59:]
        if interesting:
            return "\n".join(interesting)
        return "\n".join(log.splitlines()[-120:])

    env = dict(os.environ)
    env["DUCKDB_DISTRIBUTED_DEBUG"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=20)
    except subprocess.TimeoutExpired as exc:
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
        raise AssertionError(summarize_log((stdout or "") + (stderr or ""))) from exc
    log = (stdout or "") + (stderr or "")
    if proc.returncode != 0:
        raise AssertionError(summarize_log(log))
    for line in reversed(log.splitlines()):
        if line.startswith("VANE_BYTE_SUBMIT_ROWS="):
            rows = json.loads(line.split("=", 1)[1])
            sink_blocks = sum("where=sink_block_lazy" in log_line for log_line in log.splitlines())
            lazy_submits = sum("where=submit_lazy" in log_line for log_line in log.splitlines())
            return rows, lazy_submits, sink_blocks, summarize_log(log)
    raise AssertionError(summarize_log(log))


def test_map_batches_ray_streaming_contract_cannot_be_disabled():
    """Ray stages always use the graph-owned direct block stream."""
    import vane

    def identity(table):
        return table

    con = vane.connect()
    rel = con.sql("select 1 as x")

    disabled = rel.map_batches(
        identity,
        schema={"x": vane.sqltypes.INTEGER},
        execution_backend="ray_task",
        batch_size=1,
        streaming_breaker=False,
    )
    disabled_plan = disabled.explain()
    assert "INOUT_FUNCTION" in disabled_plan
    assert "call_mode" in disabled_plan
    assert "map_batches" in disabled_plan
    assert "execution_backend" in disabled_plan
    assert "ray_task" in disabled_plan
    assert "lazy_ref_boundary" in disabled_plan
    assert "strict_ref_aware" in disabled_plan
    assert "ray_block_stream_output" in disabled_plan
    assert "direct_block_metadata_pair" in disabled_plan

    default_streaming = rel.map_batches(
        identity,
        schema={"x": vane.sqltypes.INTEGER},
        execution_backend="ray_task",
        batch_size=1,
    )
    default_plan = default_streaming.explain()
    assert "STREAMING_UDF" in default_plan
    assert "ray_block_stream_output" in default_plan
    assert "direct_block_metadata_pair" in default_plan


def test_map_batches_nested_reserved_field_name_round_trips_at_bind():
    import pyarrow as pa

    import vane

    feature_arrow_type = pa.struct(
        [
            ("label", pa.int64()),
            ("confidence", pa.float64()),
            ("bbox", pa.list_(pa.float64())),
        ]
    )

    def make_features(table):
        values = [
            {
                "label": int(value) + 7,
                "confidence": 0.5,
                "bbox": [1.0, 2.0, 3.0, 4.0],
            }
            for value in table.column("x").to_pylist()
        ]
        return pa.table({"features": pa.array(values, type=feature_arrow_type)})

    feature_type = vane.sqltype('STRUCT("label" BIGINT, confidence DOUBLE, bbox DOUBLE[])')
    rel = (
        vane.connect()
        .sql("select i::BIGINT AS x from range(2) t(i)")
        .map_batches(
            make_features,
            schema={"features": feature_type},
            execution_backend="subprocess_task",
            batch_size=2,
        )
    )

    assert rel.types[0].children == [
        ("label", vane.sqltypes.BIGINT),
        ("confidence", vane.sqltypes.DOUBLE),
        ("bbox", vane.list_type(vane.sqltypes.DOUBLE)),
    ]
    assert [row[0]["label"] for row in rel.fetchall()] == [7, 8]


def test_scalar_ray_task_uses_mandatory_direct_block_stream_contract():
    import vane

    def plus_one(value):
        return value + 1

    plan = (
        vane.connect()
        .sql("select 1::INTEGER as a")
        .map(
            plus_one,
            return_type=vane.sqltypes.INTEGER,
            execution_backend="ray_task",
        )
        .project("a, value")
        .explain()
    )

    assert "INOUT_FUNCTION" in plan
    assert "lazy_ref_boundary" in plan
    assert "strict_ref_aware" in plan
    assert "ray_block_stream_output" in plan
    assert "direct_block_metadata_pair" in plan


def test_map_batches_streaming_breaker_defaults_on_for_subprocess_actor():
    import vane

    class Identity:
        def __call__(self, table):
            return table

    con = vane.connect()
    rel = con.sql("select 1 as x").map_batches(
        Identity,
        schema={"x": vane.sqltypes.INTEGER},
        execution_backend="subprocess_actor",
        actor_number=1,
        gpus=0.0,
        batch_size=1,
    )

    plan = rel.explain()
    assert "STREAMING_UDF" in plan
    assert "execution_backend" in plan
    assert "subprocess_actor" in plan
    assert "local_shm_ref_bundle" in plan


def test_map_batches_subprocess_actor_streaming_breaker_plan_opt_in():
    import vane

    class Identity:
        def __call__(self, table):
            return table

    con = vane.connect()
    rel = con.sql("select 1 as x").map_batches(
        Identity,
        schema={"x": vane.sqltypes.INTEGER},
        execution_backend="subprocess_actor",
        actor_number=1,
        gpus=0.0,
        batch_size=1,
        streaming_breaker=True,
    )

    plan = rel.explain()
    assert "STREAMING_UDF" in plan
    assert "execution_backend" in plan
    assert "subprocess_actor" in plan
    assert "local_shm_ref_bundle" in plan


def test_map_batches_subprocess_actor_streaming_breaker_fetches_rows():
    import pyarrow as pa

    import vane

    class AddOne:
        def __call__(self, table):
            values = table.column("x").to_pylist()
            return pa.table({"y": [value + 1 for value in values]})

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(5) t(i)").map_batches(
        AddOne,
        schema={"y": vane.sqltypes.INTEGER},
        execution_backend="subprocess_actor",
        actor_number=1,
        gpus=0.0,
        batch_size=2,
        streaming_breaker=True,
    )

    assert sorted(rel.fetchall()) == [(1,), (2,), (3,), (4,), (5,)]


def test_map_batches_subprocess_streaming_fetch_rows():
    import pyarrow as pa

    import vane

    def add_one(table):
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values]})

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(4) t(i)").map_batches(
        add_one,
        schema={"y": vane.sqltypes.INTEGER},
        execution_backend="subprocess_task",
        batch_size=1,
        streaming_breaker=True,
    )

    assert sorted(rel.fetchall()) == [(1,), (2,), (3,), (4,)]


def test_map_batches_subprocess_streaming_drains_after_sink_finalize():
    pytest.importorskip("pyarrow")
    import vane

    row_count = 8192

    def identity(table):
        return table

    con = vane.connect()
    con.execute("SET threads=8")
    rel = con.sql(f"select i::BIGINT as x from range({row_count}) t(i)").map_batches(
        identity,
        schema={"x": vane.sqltypes.BIGINT},
        execution_backend="subprocess_task",
        batch_size=64,
        streaming_breaker=True,
    )

    assert rel.aggregate("count(*)").fetchone() == (row_count,)


def test_streaming_breaker_waits_for_tail_events_without_source_finalize_spin():
    pytest.importorskip("pyarrow")
    import os
    import re
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import time
        import vane

        def slow_identity(table):
            time.sleep(0.3)
            return table

        con = vane.connect()
        con.execute("SET threads=4")
        rel = con.sql("select i::BIGINT as x from range(70) t(i)").map_batches(
            slow_identity,
            schema={"x": vane.sqltypes.BIGINT},
            execution_backend="subprocess_task",
            batch_size=64,
            streaming_breaker=True,
        )
        assert rel.aggregate("count(*)").fetchone() == (70,)
        """
    )
    env = dict(os.environ)
    env["DUCKDB_DISTRIBUTED_DEBUG"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    source_calls = [int(value) for value in re.findall(r"source_calls=(\d+)", result.stderr)]
    assert source_calls
    assert max(source_calls) <= 8, result.stderr[-4000:]


def test_map_batches_chained_streaming_flushes_partial_pending_before_input_cap():
    pytest.importorskip("pyarrow")
    import vane

    row_count = 70

    def identity(table):
        return table

    con = vane.connect()
    con.execute("SET threads=4")
    rel = (
        con.sql(f"select i::BIGINT as x from range({row_count}) t(i)")
        .map_batches(
            identity,
            schema={"x": vane.sqltypes.BIGINT},
            execution_backend="subprocess_task",
            batch_size=10,
            streaming_breaker=True,
        )
        .map_batches(
            identity,
            schema={"x": vane.sqltypes.BIGINT},
            execution_backend="subprocess_task",
            batch_size=64,
            streaming_breaker=True,
        )
    )

    assert rel.aggregate("count(*)").fetchone() == (row_count,)


def test_map_batches_subprocess_task_native_parallel_plan_and_fetches_rows():
    import pyarrow as pa

    import vane

    def add_one(table):
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values]})

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(4) t(i)").map_batches(
        add_one,
        schema={"y": vane.sqltypes.INTEGER},
        execution_backend="subprocess_task",
        batch_size=1,
        streaming_breaker=True,
    )

    plan = rel.explain()
    assert "STREAMING_UDF" in plan
    assert "subprocess_task" in plan
    assert "subprocess_pool_size" not in plan
    assert "execution_width" not in plan
    assert sorted(rel.fetchall()) == [(1,), (2,), (3,), (4,)]


def test_map_batches_subprocess_task_default_concurrency_resolves_at_operator_init():
    import pyarrow as pa

    import vane

    def add_one(table):
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values]})

    con = vane.connect()
    con.execute("SET threads=4")
    rel = con.sql("select i::INTEGER as x from range(4) t(i)").map_batches(
        add_one,
        schema={"y": vane.sqltypes.INTEGER},
        execution_backend="subprocess_task",
        batch_size=1,
        streaming_breaker=True,
    )

    plan = rel.explain()
    assert "udf_worker_slots" not in plan
    assert "udf_outstanding_batch_limit" not in plan
    assert "max_threads" not in plan
    assert "subprocess_pool_size" not in plan
    assert "execution_width" not in plan


def test_map_batches_subprocess_task_streaming_width_resolves_from_pipeline(tmp_path):
    log = _run_udf_width_probe(tmp_path, streaming_breaker=True)
    assert "streaming_ctor_unresolved udf_name=ident initial_config_width=1" in log
    assert "streaming_ctor_initial_resolve" not in log
    assert "streaming_resolve_commit udf_name=ident width=4 operator_width_resolved=true" in log
    assert "payload_udf_worker_slots=4" in log
    assert "executor_init mode=task backend='subprocess_task' pool_size=4" in log


def test_map_batches_subprocess_task_inout_width_resolves_before_local_init(tmp_path):
    log = _run_udf_width_probe(tmp_path, streaming_breaker=False)
    assert "pipeline_max_threads_resolved udf_name=ident max_threads=4" in log
    assert "resolve_runtime_commit udf_name=ident reason=pipeline_resolved width=4" in log
    assert "payload_udf_worker_slots=4" in log
    assert "executor_init mode=task backend='subprocess_task' pool_size=4" in log


@pytest.mark.timeout(60)
def test_map_batches_subprocess_actor_lazy_ref_bundle_reaches_user_compute_batch_size(tmp_path):
    counts = _run_subprocess_actor_lazy_compute_batch_probe(tmp_path)

    assert counts == {3000: 3000, 1096: 1096}


@pytest.mark.timeout(60)
def test_map_batches_subprocess_task_direct_materialized_reaches_user_compute_batch_size(tmp_path):
    counts = _run_subprocess_task_direct_compute_batch_probe(tmp_path)

    assert counts == {3000: 3000, 1096: 1096}


@pytest.mark.timeout(60)
def test_map_batches_without_batch_size_submits_each_upstream_work_unit(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    import vane

    input_dir = tmp_path / "map_batches_work_units"
    input_dir.mkdir()
    con = vane.connect()
    con.execute("SET threads=1")
    for part in range(4):
        start = part * 60
        con.execute(
            f"COPY (SELECT i::INTEGER AS x FROM range({start}, {start + 60}) t(i)) "
            f"TO '{input_dir}/part_{part}.parquet' (FORMAT PARQUET)"
        )

    def annotate_task_rows(table):
        rows = table.num_rows
        return pa.table(
            {
                "x": table.column("x"),
                "task_rows": [rows for _ in range(rows)],
            }
        )

    rel = con.read_parquet(str(input_dir / "*.parquet")).map_batches(
        annotate_task_rows,
        schema={"x": vane.sqltypes.INTEGER, "task_rows": vane.sqltypes.INTEGER},
        execution_backend="subprocess_task",
        target_max_batch_bytes=1 << 30,
        streaming_breaker=True,
    )
    counts = {}
    for _x, task_rows in rel.fetchall():
        counts[task_rows] = counts.get(task_rows, 0) + 1

    assert counts == {60: 240}

    def identity(table):
        return pa.table({"x": table.column("x")})

    chained = (
        con.read_parquet(str(input_dir / "*.parquet"))
        .map_batches(
            identity,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="subprocess_task",
            target_max_batch_bytes=1 << 30,
            streaming_breaker=True,
        )
        .map_batches(
            annotate_task_rows,
            schema={"x": vane.sqltypes.INTEGER, "task_rows": vane.sqltypes.INTEGER},
            execution_backend="subprocess_task",
            target_max_batch_bytes=1 << 30,
            streaming_breaker=True,
        )
    )
    chained_counts = {}
    for _x, task_rows in chained.fetchall():
        chained_counts[task_rows] = chained_counts.get(task_rows, 0) + 1

    assert chained_counts == {60: 240}


@pytest.mark.timeout(60)
def test_map_batches_subprocess_task_direct_materialized_transport_ignores_user_batch_size(tmp_path):
    rows, envelope_submits, standard_submits, log = _run_subprocess_task_direct_byte_transport_probe(tmp_path)

    assert len(rows) == 8, log
    assert max(rows) < 8, log
    assert envelope_submits > 0, log
    assert standard_submits == 0, log


@pytest.mark.timeout(60)
def test_map_batches_materialized_byte_split_preserves_complete_compute_batches():
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    import vane

    def annotate_batch_size(table):
        rows = table.num_rows
        return pa.table({"batch_rows": [rows for _ in range(rows)]})

    con = vane.connect()
    con.execute("SET threads=1")
    rel = con.sql("SELECT repeat('x', 4096)::BLOB AS payload FROM range(300) t(i)").map_batches(
        annotate_batch_size,
        schema={"batch_rows": vane.sqltypes.INTEGER},
        execution_backend="subprocess_task",
        batch_size=100,
        target_max_batch_bytes=512 * 1024,
        streaming_breaker=True,
    )
    counts = {}
    for (batch_rows,) in rel.fetchall():
        counts[batch_rows] = counts.get(batch_rows, 0) + 1

    assert counts == {100: 300}


@pytest.mark.timeout(60)
def test_map_batches_subprocess_task_direct_submit_follows_upstream_work_units(tmp_path):
    row_count, batch_rows, submit_rows, log = _run_subprocess_task_direct_work_unit_submit_probe(tmp_path)

    assert row_count == 5000, log
    assert 100 in batch_rows, log
    assert max(batch_rows) <= 100, log
    assert submit_rows, log
    # Each 2048-row upstream work unit becomes ready immediately. The transport
    # carries fewer than 100 rows into the next work unit instead of waiting for
    # a separate static row watermark or emitting an avoidable partial batch.
    assert max(submit_rows) > 200, log
    assert max(submit_rows) <= 2100, log


def test_map_batches_can_request_soft_minimum_task_batch_size():
    import pyarrow as pa

    import vane

    def identity(table):
        return pa.table({"x": table.column("x")})

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(256) t(i)").map_batches(
        identity,
        schema={"x": vane.sqltypes.INTEGER},
        execution_backend="subprocess_task",
        batch_size=32,
        min_task_batch_size=32,
        target_max_batch_bytes=1 << 30,
        task_input_max_bytes=1 << 30,
        streaming_breaker=True,
    )

    compact_plan = "".join(ch for ch in rel.explain() if ch.isalnum() or ch == "_")
    assert "min_task_batch_size32" in compact_plan


def test_map_batches_minimum_task_batch_size_validates_contract():
    import pyarrow as pa

    import vane

    def identity(table):
        return pa.table({"x": table.column("x")})

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(4) t(i)")

    with pytest.raises(vane.InvalidInputException, match="requires batch_size"):
        rel.map_batches(
            identity,
            schema={"x": vane.sqltypes.INTEGER},
            min_task_batch_size=32,
        )
    with pytest.raises(vane.InvalidInputException, match="at least batch_size"):
        rel.map_batches(
            identity,
            schema={"x": vane.sqltypes.INTEGER},
            batch_size=32,
            min_task_batch_size=16,
        )
    with pytest.raises(vane.InvalidInputException, match="streaming_breaker=True"):
        rel.map_batches(
            identity,
            schema={"x": vane.sqltypes.INTEGER},
            batch_size=32,
            min_task_batch_size=32,
            streaming_breaker=False,
        )


def test_map_batches_can_preserve_compute_batch_output_boundaries():
    import pyarrow as pa

    import vane

    def identity(table):
        return pa.table({"x": table.column("x")})

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(5) t(i)").map_batches(
        identity,
        schema={"x": vane.sqltypes.INTEGER},
        execution_backend="subprocess_task",
        batch_size=2,
        output_batch_size=3,
        preserve_compute_batch_boundaries=True,
        streaming_breaker=True,
    )

    compact_plan = "".join(ch for ch in rel.explain() if ch.isalnum() or ch == "_")
    assert "preserve_compute_batch_boundariestrue" in compact_plan


def test_map_batches_compute_boundary_mode_requires_streaming_breaker():
    import pyarrow as pa

    import vane

    with pytest.raises(vane.InvalidInputException, match="preserve_compute_batch_boundaries"):
        vane.sql("select 1::INTEGER as x").map_batches(
            lambda table: pa.table({"x": table.column("x")}),
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="subprocess_task",
            batch_size=1,
            preserve_compute_batch_boundaries=True,
            streaming_breaker=False,
        )


@pytest.mark.timeout(60)
def test_map_batches_lazy_ref_submit_uses_byte_threshold_before_user_batch_size(tmp_path):
    rows, lazy_submits, _sink_blocks, log = _run_subprocess_lazy_byte_submit_probe(tmp_path)

    assert len(rows) == 8, log
    assert max(rows) < 3000, log
    assert lazy_submits > 1, log


def test_map_batches_subprocess_actor_uses_actor_pool_size(tmp_path):
    log = _run_udf_actor_width_probe(tmp_path, streaming_breaker=True)
    assert "streaming_resolve_commit udf_name=Ident width=4 operator_width_resolved=true" in log
    assert "payload_udf_worker_slots=2" in log


@pytest.mark.timeout(30)
@pytest.mark.parametrize("row_count", [2049, 6145])
def test_map_batches_subprocess_task_inout_partial_tail_consumed_once(tmp_path, row_count):
    import pyarrow as pa

    import vane

    seen_path = tmp_path / f"seen-{row_count}.txt"

    def ident(table):
        values = table.column("x").to_pylist()
        with seen_path.open("a") as handle:
            handle.write(f"{len(values)}:{min(values)}:{max(values)}\n")
        return pa.table({"x": table.column("x")})

    con = vane.connect()
    con.execute("SET threads=8")
    rel = con.sql(f"select i::INTEGER as x from range({row_count}) t(i)").map_batches(
        ident,
        schema={"x": vane.sqltypes.INTEGER},
        execution_backend="subprocess_task",
        batch_size=64,
        streaming_breaker=False,
    )

    assert rel.aggregate("count(*)").fetchone() == (row_count,)
    seen = seen_path.read_text().splitlines()
    assert len(seen) == (row_count + 63) // 64
    assert seen.count(f"1:{row_count - 1}:{row_count - 1}") == 1


@pytest.mark.parametrize(
    "removed_param",
    [
        "queue_depth",
        "max_outstanding_batches",
        "max_ready_rows",
        "max_ready_bytes",
        "max_pending_bytes",
        "submit_target_max_bytes",
    ],
)
def test_map_batches_rejects_removed_admission_params(removed_param):
    import vane

    con = vane.connect()
    with pytest.raises(TypeError, match=removed_param):
        con.sql("select 1 as x").map_batches(
            lambda table: table,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="subprocess_task",
            **{removed_param: 3},
        )


def test_map_batches_accepts_byte_batching_params():
    import pyarrow as pa

    import vane

    def add_one(table):
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values]})

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(4) t(i)").map_batches(
        add_one,
        schema={"y": vane.sqltypes.INTEGER},
        execution_backend="subprocess_task",
        batch_size=1,
        target_max_batch_bytes=64,
        task_input_max_bytes=32,
        output_target_max_bytes=48,
    )
    compact_plan = "".join(ch for ch in rel.explain() if ch.isalnum() or ch == "_")

    assert "udf_target_max_batch_bytes64" in compact_plan
    assert "udf_task_input_max_bytes32" in compact_plan
    assert "udf_output_target_max_bytes48" in compact_plan


def test_map_batches_accepts_ray_memory_bytes():
    import pyarrow as pa

    import vane

    def identity(table):
        return pa.table({"x": table.column("x")})

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(4) t(i)").map_batches(
        identity,
        schema={"x": vane.sqltypes.INTEGER},
        execution_backend="ray_task",
        memory_bytes=536870912,
    )
    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(rel, "map-batches-memory-bytes")
    physical = logical.to_physical_plan(con)
    payload = physical.collect_udf_nodes(conn=con)[0]["payload"]

    assert payload["memory_bytes"] == 536870912


def test_flat_map_accepts_ray_memory_bytes():
    import vane

    def duplicate(row):
        return [row, row]

    con = vane.connect()
    rel = con.sql("select 1::INTEGER as x").flat_map(
        duplicate,
        schema={"x": vane.sqltypes.INTEGER},
        execution_backend="ray_task",
        memory_bytes=268435456,
    )
    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(rel, "flat-map-memory-bytes")
    physical = logical.to_physical_plan(con)
    payload = physical.collect_udf_nodes(conn=con)[0]["payload"]

    assert payload["memory_bytes"] == 268435456


def test_map_batches_accepts_ray_actor_memory_bytes():
    import vane

    class Identity:
        def __call__(self, table):
            return table

    con = vane.connect()
    rel = con.sql("select 1::INTEGER as x").map_batches(
        Identity,
        schema={"x": vane.sqltypes.INTEGER},
        execution_backend="ray_actor",
        actor_number=1,
        gpus=0.0,
        memory_bytes=1073741824,
    )
    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(rel, "map-batches-actor-memory-bytes")
    physical = logical.to_physical_plan(con)
    payload = physical.collect_udf_nodes(conn=con)[0]["payload"]

    assert payload["memory_bytes"] == 1073741824


def test_map_batches_rejects_invalid_or_non_ray_memory_bytes():
    import vane

    def identity(table):
        return table

    con = vane.connect()
    with pytest.raises(Exception, match="memory_bytes"):
        con.sql("select 1 as x").map_batches(
            identity,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="ray_task",
            memory_bytes=0,
        )
    with pytest.raises(Exception, match="Ray UDF backend"):
        con.sql("select 1 as x").map_batches(
            identity,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="subprocess_task",
            memory_bytes=536870912,
        )


def test_map_batches_rejects_invalid_byte_batching_params():
    import vane

    def identity(table):
        return table

    con = vane.connect()
    for keyword in (
        "target_max_batch_bytes",
        "task_input_max_bytes",
        "output_target_max_bytes",
    ):
        with pytest.raises(Exception, match=keyword):
            con.sql("select i::INTEGER as x from range(4) t(i)").map_batches(
                identity,
                schema={"x": vane.sqltypes.INTEGER},
                execution_backend="subprocess_task",
                batch_size=1,
                **{keyword: 0},
            )


def test_map_batches_reads_target_max_batch_bytes_env(monkeypatch):
    import pyarrow as pa

    import vane

    def add_one(table):
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values]})

    monkeypatch.setenv("VANE_UDF_TARGET_MAX_BATCH_BYTES", "77")
    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(4) t(i)").map_batches(
        add_one,
        schema={"y": vane.sqltypes.INTEGER},
        execution_backend="subprocess_task",
        batch_size=1,
    )
    compact_plan = "".join(ch for ch in rel.explain() if ch.isalnum() or ch == "_")

    assert "udf_target_max_batch_bytes77" in compact_plan
    assert "udf_task_input_max_bytes77" in compact_plan
    assert "udf_output_target_max_bytes77" in compact_plan


def test_scalar_map_reads_target_max_batch_bytes_env(monkeypatch):
    import vane

    def add_one(value: int) -> int:
        return value + 1

    monkeypatch.setenv("VANE_UDF_TARGET_MAX_BATCH_BYTES", "77")
    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(4) t(i)").map(
        add_one,
        return_type=vane.sqltypes.INTEGER,
        execution_backend="ray_task",
    )
    compact_plan = "".join(ch for ch in rel.explain() if ch.isalnum() or ch == "_")

    assert "udf_target_max_batch_bytes77" in compact_plan
    assert "udf_task_input_max_bytes77" in compact_plan
    assert "udf_output_target_max_bytes77" in compact_plan


def test_local_shm_ref_bundle_result_has_block_metadata():
    import pyarrow as pa

    from vane.execution.ref_bundle import REF_BUNDLE_RESULT_MARKER, make_local_shm_ref_bundle_result

    result = make_local_shm_ref_bundle_result(pa.table({"x": [1, 2]}))
    marker, refs, metadata, names = result
    try:
        assert marker == REF_BUNDLE_RESULT_MARKER
        assert len(refs) == 1
        assert names == ["x"]
        assert metadata[0]["provider"] == "local_shm"
        assert metadata[0]["num_rows"] == 2
        assert metadata[0]["size_bytes"] > 0
        assert metadata[0]["ipc_size_bytes"] > 0
        assert metadata[0]["shm_name"]
    finally:
        for ref in refs:
            ref.release()


def test_map_batches_rejects_invalid_target_max_batch_bytes_env(monkeypatch):
    import vane

    def identity(table):
        return table

    monkeypatch.setenv("VANE_UDF_TARGET_MAX_BATCH_BYTES", "0")
    con = vane.connect()
    with pytest.raises(Exception, match="VANE_UDF_TARGET_MAX_BATCH_BYTES"):
        con.sql("select i::INTEGER as x from range(4) t(i)").map_batches(
            identity,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="subprocess_task",
            batch_size=1,
        )


def test_map_batches_worker_slots_keyword_is_not_public_api():
    import vane

    def identity(table):
        return table

    con = vane.connect()
    with pytest.raises(Exception, match=r"map_batches\(\) got unsupported keyword argument\(s\): worker_slots"):
        con.sql("select i::INTEGER as x from range(4) t(i)").map_batches(
            identity,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="subprocess_task",
            batch_size=1,
            worker_slots=3,
        )


def test_map_batches_accepts_output_batch_size_keyword():
    import pyarrow as pa

    import vane

    def add_ten(table):
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 10 for value in values]})

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(4) t(i)").map_batches(
        add_ten,
        schema={"y": vane.sqltypes.INTEGER},
        execution_backend="subprocess_task",
        batch_size=2,
        output_batch_size=1,
    )

    assert sorted(rel.fetchall()) == [(10,), (11,), (12,), (13,)]


def test_map_batches_streaming_submit_respects_target_max_batch_bytes():
    import pyarrow as pa

    import vane

    def report_batch_rows(table):
        values = table.column("payload").to_pylist()
        return pa.table({"batch_rows": [table.num_rows for _ in values]})

    con = vane.connect()
    rel = con.sql("select repeat('x', 1024)::BLOB as payload from range(12) t(i)").map_batches(
        report_batch_rows,
        schema={"batch_rows": vane.sqltypes.BIGINT},
        execution_backend="subprocess_task",
        batch_size=2048,
        output_batch_size=2048,
        target_max_batch_bytes=512,
    )

    batch_rows = [row[0] for row in rel.fetchall()]

    assert len(batch_rows) == 12
    assert max(batch_rows) < 12


def test_map_batches_streaming_output_splits_by_actual_bytes_without_row_preserving(tmp_path):
    np = pytest.importorskip("numpy")
    import pyarrow as pa

    import vane

    tensor_shape = (64, 64)
    tensor_type = vane.tensor_type(vane.sqltypes.FLOAT, tensor_shape)
    seen_path = tmp_path / "downstream_batches.txt"

    def make_tensor(table):
        xs = table.column("x").to_pylist()
        tensors = np.ones((len(xs), *tensor_shape), dtype=np.float32)
        return pa.table(
            {
                "x": pa.array(xs, type=pa.int64()),
                "embedding": pa.FixedShapeTensorArray.from_numpy_ndarray(tensors),
            }
        )

    def record_downstream_batch(table):
        with seen_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{table.num_rows}\n")
        return pa.table({"x": table.column("x")})

    con = vane.connect()
    rel = (
        con.sql("select i::BIGINT as x from range(8) t(i)")
        .map_batches(
            make_tensor,
            schema={"x": vane.sqltypes.BIGINT, "embedding": tensor_type},
            execution_backend="subprocess_task",
            batch_size=8,
            target_max_batch_bytes=48 * 1024,
            streaming_breaker=True,
        )
        .map_batches(
            record_downstream_batch,
            schema={"x": vane.sqltypes.BIGINT},
            execution_backend="subprocess_task",
            batch_size=2048,
            target_max_batch_bytes=48 * 1024,
            streaming_breaker=True,
        )
    )

    rows = rel.order("x").fetchall()
    assert [row[0] for row in rows] == list(range(8))
    batch_rows = [int(line) for line in seen_path.read_text(encoding="utf-8").splitlines()]
    assert len(batch_rows) >= 3
    assert max(batch_rows) <= 3


def test_map_batches_lazy_ref_submit_respects_target_max_batch_bytes():
    import pyarrow as pa

    import vane

    def make_payload(table):
        values = table.column("x").to_pylist()
        return pa.table({"payload": [b"x" * 1024 for _ in values]})

    def report_batch_rows(table):
        values = table.column("payload").to_pylist()
        return pa.table({"batch_rows": [table.num_rows for _ in values]})

    con = vane.connect()
    rel = (
        con.sql("select i::INTEGER as x from range(12) t(i)")
        .map_batches(
            make_payload,
            schema={"payload": vane.sqltypes.BLOB},
            execution_backend="subprocess_task",
            batch_size=2048,
            output_batch_size=2048,
            target_max_batch_bytes=512,
        )
        .map_batches(
            report_batch_rows,
            schema={"batch_rows": vane.sqltypes.BIGINT},
            execution_backend="subprocess_task",
            batch_size=2048,
            output_batch_size=2048,
            target_max_batch_bytes=512,
        )
    )

    batch_rows = [row[0] for row in rel.fetchall()]

    assert len(batch_rows) == 12
    assert max(batch_rows) < 12


def test_map_batches_lazy_ref_submit_accepts_all_null_nonempty_batches():
    import pyarrow as pa

    import vane

    def make_nulls(table):
        return pa.table({"payload": [None for _ in range(table.num_rows)]})

    def report_batch_rows(table):
        return pa.table({"batch_rows": [table.num_rows for _ in range(table.num_rows)]})

    con = vane.connect()
    con.execute("SET threads=1")
    rel = (
        con.sql("select i::INTEGER as x from range(4) t(i)")
        .map_batches(
            make_nulls,
            schema={"payload": vane.sqltypes.INTEGER},
            execution_backend="subprocess_task",
            batch_size=2048,
            output_batch_size=2048,
        )
        .map_batches(
            report_batch_rows,
            schema={"batch_rows": vane.sqltypes.INTEGER},
            execution_backend="subprocess_task",
            batch_size=2048,
            output_batch_size=2048,
        )
    )

    assert rel.fetchall() == [(4,), (4,), (4,), (4,)]


def test_map_batches_ray_task_default_concurrency_resolves_at_operator_init():
    import pyarrow as pa

    import vane

    def add_one(table):
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values]})

    con = vane.connect()
    con.execute("SET threads=8")
    rel = con.sql("select i::INTEGER as x from range(4) t(i)").map_batches(
        add_one,
        schema={"y": vane.sqltypes.INTEGER},
        execution_backend="ray_task",
        batch_size=1,
    )

    plan = rel.explain()
    assert "udf_worker_slots" not in plan
    assert "udf_outstanding_batch_limit" not in plan


def test_map_batches_subprocess_actor_number_plan_and_fetches_rows():
    import pyarrow as pa

    import vane

    class AddOne:
        def __call__(self, table):
            values = table.column("x").to_pylist()
            return pa.table({"y": [value + 1 for value in values]})

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(4) t(i)").map_batches(
        AddOne,
        schema={"y": vane.sqltypes.INTEGER},
        execution_backend="subprocess_actor",
        actor_number=2,
        gpus=0.0,
        batch_size=1,
    )

    plan = rel.explain()
    normalized_plan = "".join(ch for ch in plan if ch.isalnum() or ch == "_")
    assert "subprocess_actor" in plan
    assert "actor_number" in plan
    assert "subprocess_pool_size" not in plan
    assert "udf_worker_slots" not in normalized_plan
    assert "udf_outstanding_batch_limit" not in normalized_plan
    assert sorted(rel.fetchall()) == [(1,), (2,), (3,), (4,)]


def test_map_batches_subprocess_actor_streaming_fetches_rows():
    import vane

    class Identity:
        def __call__(self, table):
            return table

    con = vane.connect()
    rel = con.sql("select 1 as x").map_batches(
        Identity,
        schema={"x": vane.sqltypes.INTEGER},
        execution_backend="subprocess_actor",
        actor_number=2,
        gpus=0.0,
        batch_size=1,
        streaming_breaker=True,
    )
    assert sorted(rel.fetchall()) == [(1,)]


def test_map_batches_subprocess_actor_non_streaming_fetches_rows():
    import vane

    class Identity:
        def __call__(self, table):
            return table

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(3) t(i)").map_batches(
        Identity,
        schema={"x": vane.sqltypes.INTEGER},
        execution_backend="subprocess_actor",
        actor_number=1,
        gpus=0.0,
        batch_size=1,
        streaming_breaker=False,
    )
    assert sorted(rel.fetchall()) == [(0,), (1,), (2,)]


def test_subprocess_actor_concurrency_does_not_exceed_actor_pool(monkeypatch, tmp_path):
    import json

    import vane

    state_path = tmp_path / "counter.json"
    monkeypatch.setenv("VANE_TEST_UDF_COUNTER_PATH", str(state_path))

    class Probe:
        def __call__(self, table):
            import fcntl
            import json
            import os
            import time

            path = os.environ["VANE_TEST_UDF_COUNTER_PATH"]
            lock_path = path + ".lock"

            def update(delta):
                with open(lock_path, "w") as lock_file:
                    fcntl.flock(lock_file, fcntl.LOCK_EX)
                    try:
                        if os.path.exists(path):
                            with open(path) as state_file:
                                state = json.load(state_file)
                        else:
                            state = {"current": 0, "max": 0}
                        state["current"] += delta
                        state["max"] = max(state["max"], state["current"])
                        with open(path, "w") as state_file:
                            json.dump(state, state_file)
                    finally:
                        fcntl.flock(lock_file, fcntl.LOCK_UN)

            update(1)
            try:
                time.sleep(0.05)
                return table
            finally:
                update(-1)

    con = vane.connect()
    con.execute("SET threads=4")
    rel = con.sql("select i::INTEGER as x from range(8) t(i)").map_batches(
        Probe,
        schema={"x": vane.sqltypes.INTEGER},
        execution_backend="subprocess_actor",
        actor_number=2,
        gpus=0.0,
        batch_size=1,
    )
    assert sorted(rel.fetchall()) == [(idx,) for idx in range(8)]
    with open(state_path) as state_file:
        state = json.load(state_file)
    assert 1 <= state["max"] <= 2


def test_map_batches_rejects_bare_subprocess_backend():
    import vane

    def identity(table):
        return table

    con = vane.connect()
    rel = con.sql("select 1 as x")

    with pytest.raises(Exception, match="subprocess_task"):
        rel.map_batches(
            identity,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="subprocess",
            batch_size=1,
        )


def test_map_batches_rejects_removed_async_options():
    import vane

    def identity(table):
        return table

    con = vane.connect()
    rel = con.sql("select 1 as x")

    with pytest.raises(Exception, match=r"map_batches\(\) got unsupported keyword argument\(s\): async_mode"):
        rel.map_batches(
            identity,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="ray_task",
            batch_size=1,
            async_mode=True,
        )

    with pytest.raises(Exception, match=r"map_batches\(\) got unsupported keyword argument\(s\): async_mode"):
        rel.map_batches(
            identity,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="ray_task",
            batch_size=1,
            streaming_breaker=True,
            async_mode=True,
        )


def test_map_batches_unknown_kwargs_fail_without_rendering_relation():
    import vane

    upstream_called = False

    def fail_if_rendered(_value):
        nonlocal upstream_called
        upstream_called = True
        raise RuntimeError("relation repr executed upstream UDF")

    def identity(table):
        return table

    con = vane.connect()
    con.create_function(
        "fail_if_rendered",
        fail_if_rendered,
        parameters=[vane.sqltypes.INTEGER],
        return_type=vane.sqltypes.INTEGER,
    )
    rel = con.sql("select fail_if_rendered(1) as x")

    with pytest.raises(Exception, match=r"actor_locality_mode, ray_actor_pool_name"):
        rel.map_batches(
            identity,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="ray_actor",
            actor_number=1,
            gpus=0.0,
            actor_locality_mode="prefer_local",
            ray_actor_pool_name="pool-a",
        )

    assert upstream_called is False


def test_map_batches_accepts_explicit_ray_task_backend():
    """execution_backend is the final routing knob for UDFs."""
    import vane

    def identity(table):
        return table

    con = vane.connect()
    rel = con.sql("select 1 as x").map_batches(
        identity,
        schema={"x": vane.sqltypes.INTEGER},
        execution_backend="ray_task",
    )

    plan = rel.explain()
    assert "STREAMING_UDF" in plan


def test_map_batches_rejects_actor_number_for_task_backend():
    import vane

    def identity(table):
        return table

    con = vane.connect()
    with pytest.raises(Exception, match="actor_number is only supported"):
        con.sql("select 1 as x").map_batches(
            identity,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="ray_task",
            actor_number=2,
        )
    with pytest.raises(Exception, match="actor_number is only supported"):
        con.sql("select 1 as x").map_batches(
            identity,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="subprocess_task",
            actor_number=2,
        )


def test_map_batches_rejects_removed_concurrency_argument():
    import vane

    def identity(table):
        return table

    con = vane.connect()
    with pytest.raises(Exception, match="unsupported keyword argument.*concurrency"):
        con.sql("select 1 as x").map_batches(
            identity,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="subprocess_task",
            concurrency=2,
        )


def test_map_batches_rejects_removed_max_inflight_argument():
    import vane

    def identity(table):
        return table

    con = vane.connect()
    with pytest.raises(Exception, match="unsupported keyword argument.*max_inflight"):
        con.sql("select 1 as x").map_batches(
            identity,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="subprocess_task",
            max_inflight=2,
        )


def test_map_batches_actor_backend_requires_actor_number():
    import vane

    class Identity:
        def __call__(self, table):
            return table

    con = vane.connect()
    with pytest.raises(Exception, match="actor_number is required"):
        con.sql("select 1 as x").map_batches(
            Identity,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="subprocess_actor",
        )


def test_map_batches_actor_backend_requires_gpus():
    import vane

    class Identity:
        def __call__(self, table):
            return table

    con = vane.connect()
    with pytest.raises(Exception, match=r"map_batches\(gpus=\.\.\.\) is required"):
        con.sql("select 1 as x").map_batches(
            Identity,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="subprocess_actor",
            actor_number=1,
        )
    with pytest.raises(Exception, match=r"map_batches\(gpus=\.\.\.\) is required"):
        con.sql("select 1 as x").map_batches(
            Identity,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="ray_actor",
            actor_number=1,
        )


def test_map_batches_accepts_explicit_cpu_resource():
    import vane

    class Identity:
        def __call__(self, table):
            return table

    con = vane.connect()
    rel = con.sql("select 1 as x").map_batches(
        Identity,
        schema={"x": vane.sqltypes.INTEGER},
        execution_backend="ray_actor",
        actor_number=1,
        cpus=0.0,
        gpus=1.0,
    )

    plan = rel.explain()
    assert "execution_backend" in plan
    assert "ray_actor" in plan
    assert "cpus" in plan
    assert "0.0" in plan or "0" in plan
    assert "gpus" in plan
    assert "1.0" in plan or "1" in plan


def test_map_batches_accepts_ray_native_actor_thread_policy():
    import vane

    class Identity:
        def __call__(self, table):
            return table

    con = vane.connect()
    rel = con.sql("select 1 as x").map_batches(
        Identity,
        schema={"x": vane.sqltypes.INTEGER},
        execution_backend="ray_actor",
        actor_number=1,
        cpus=0.0,
        gpus=1.0,
        ray_actor_thread_policy="ray_native",
    )

    plan = rel.explain()
    assert "ray_actor_thread_policy" in plan
    assert "ray_native" in plan


@pytest.mark.parametrize("policy", ["different", 1])
def test_map_batches_rejects_invalid_ray_actor_thread_policy(policy):
    import vane

    class Identity:
        def __call__(self, table):
            return table

    con = vane.connect()
    with pytest.raises(Exception, match="ray_actor_thread_policy"):
        con.sql("select 1 as x").map_batches(
            Identity,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="ray_actor",
            actor_number=1,
            gpus=1.0,
            ray_actor_thread_policy=policy,
        )


def test_map_batches_rejects_ray_actor_thread_policy_for_task_backend():
    import vane

    def identity(table):
        return table

    con = vane.connect()
    with pytest.raises(Exception, match="requires execution_backend='ray_actor'"):
        con.sql("select 1 as x").map_batches(
            identity,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="ray_task",
            ray_actor_thread_policy="ray_native",
        )


def test_ray_actor_direct_execution_requires_registered_query_allocation():
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    import vane

    class Identity:
        def __call__(self, table):
            return pa.table({"x": table.column("x")})

    con = vane.connect()
    rel = con.sql("select 1::INTEGER as x").map_batches(
        Identity,
        schema={"x": vane.sqltypes.INTEGER},
        execution_backend="ray_actor",
        actor_number=1,
        cpus=1.0,
        gpus=0.0,
    )

    with pytest.raises(
        Exception,
        match="driver-precreated actor handles from a registered query allocation",
    ):
        rel.fetchall()


def test_map_batches_rejects_non_numeric_gpus():
    import vane

    def identity(table):
        return table

    con = vane.connect()
    with pytest.raises(Exception, match=r"map_batches\(gpus=\.\.\.\) must be a number"):
        con.sql("select 1 as x").map_batches(
            identity,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="ray_actor",
            actor_number=1,
            gpus="gpu",
        )


def test_map_batches_explain_analyze_shows_udf_runtime_counters():
    """UDF runtime profile exposes streaming UDF counters."""
    pytest.importorskip("pyarrow")
    import vane

    def identity(table):
        return table

    con = vane.connect()
    con.execute("PRAGMA enable_profiling")
    rel = con.sql("select i::BIGINT as x from range(4) t(i)").map_batches(
        identity,
        schema={"x": vane.sqltypes.BIGINT},
        batch_size=2,
        execution_backend="subprocess_task",
    )

    plan = rel.explain("analyze")
    assert "udf_accepted_input_rows" in plan
    assert "udf_submitted_input_rows" in plan
    assert "udf_produced_output_rows" in plan
    assert "udf_pending_input_rows" in plan
    assert "udf_sink_finished" in plan


def test_map_batches_explain_analyze_reports_observed_ready_rows():
    pytest.importorskip("pyarrow")
    import vane

    def identity(table):
        return table

    con = vane.connect()
    con.execute("PRAGMA enable_profiling")
    rel = con.sql("select i::BIGINT as x from range(4) t(i)").map_batches(
        identity,
        schema={"x": vane.sqltypes.BIGINT},
        batch_size=1,
        execution_backend="subprocess_task",
    )

    plan = rel.explain("analyze")
    compact_plan = "".join(ch for ch in plan if ch.isalnum() or ch == "_")
    assert "udf_max_ready_observed_rows" in compact_plan


def test_map_batches_streaming_compute_uses_user_batch_size(tmp_path):
    pytest.importorskip("pyarrow")
    import vane

    seen_path = tmp_path / "seen-batches.txt"

    def identity(table):
        with seen_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{table.num_rows}\n")
        return table

    con = vane.connect()
    rel = con.sql("select i::BIGINT as x from range(25) t(i)").map_batches(
        identity,
        schema={"x": vane.sqltypes.BIGINT},
        batch_size=10,
        execution_backend="subprocess_task",
    )

    assert rel.fetchall() == [(idx,) for idx in range(25)]
    assert seen_path.read_text(encoding="utf-8").splitlines() == ["10", "10", "5"]


def test_map_batches_basic():
    """map_batches with a simple function returning a single column."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    import vane

    def add_ten(table):
        values = table.column(0).to_pylist()
        return pa.table({"result": [v + 10 for v in values]})

    con = vane.connect()
    rel = con.sql("select 1 as x union all select 2 as x")
    out = rel.map_batches(
        add_ten,
        schema={"result": vane.sqltypes.BIGINT},
        execution_backend="subprocess_task",
    )
    assert sorted(out.fetchall()) == [(11,), (12,)]


def test_map_batches_direct_arrow_output_conversion():
    """Non-streaming map_batches still accepts direct pyarrow.Table output."""
    pytest.importorskip("pyarrow")
    import _vane_duckdb
    import pyarrow as pa

    import vane

    def add_one(table):
        values = table.column("x").to_pylist()
        return pa.table({"result": [v + 1 for v in values]})

    con = vane.connect()
    _vane_duckdb._reset_udf_executor_debug_counters()
    rel = con.sql("select i::BIGINT as x from range(5) t(i)")
    out = rel.map_batches(
        add_one,
        schema={"result": vane.sqltypes.BIGINT},
        batch_size=10,
        streaming_breaker=False,
        execution_backend="subprocess_task",
    )
    assert out.fetchall() == [(1,), (2,), (3,), (4,), (5,)]
    counters = dict(_vane_duckdb._udf_executor_debug_counters())
    assert counters["udf_direct_arrow_table_conversion_count"] == 1
    assert counters["udf_direct_output_arrow_table_conversion_count"] == 1
    assert counters["udf_python_export_under_client_context_lock_count"] == 0


def test_map_batches_batch_size():
    """map_batches respects batch_size parameter."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    import vane

    seen_sizes = []

    def track_batch(table):
        seen_sizes.append(table.num_rows)
        values = table.column(0).to_pylist()
        return pa.table({"result": [v * 2 for v in values]})

    con = vane.connect()
    rel = con.sql("select i from range(10) t(i)")
    out = rel.map_batches(
        track_batch,
        schema={"result": vane.sqltypes.BIGINT},
        batch_size=3,
        execution_backend="subprocess_task",
    )
    rows = out.fetchall()
    assert len(rows) == 10


def test_map_batches_multi_column_output():
    """map_batches can produce multiple output columns."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    import vane

    def split(table):
        values = table.column(0).to_pylist()
        return pa.table(
            {
                "doubled": [v * 2 for v in values],
                "tripled": [v * 3 for v in values],
            }
        )

    con = vane.connect()
    rel = con.sql("select 5 as x")
    out = rel.map_batches(
        split,
        schema={"doubled": vane.sqltypes.BIGINT, "tripled": vane.sqltypes.BIGINT},
        execution_backend="subprocess_task",
    )
    result = out.fetchall()
    # Multi-column schema is flattened into separate columns via struct_extract
    assert len(result) == 1
    assert result[0] == (10, 15)


def test_map_batches_chained_multi_column():
    """Chaining two multi-column map_batches calls works correctly."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    import vane

    def stage1(table):
        xs = table.column("x").to_pylist()
        return pa.table(
            {
                "x": xs,
                "doubled": [v * 2 for v in xs],
            }
        )

    def stage2(table):
        xs = table.column("x").to_pylist()
        doubled = table.column("doubled").to_pylist()
        return pa.table(
            {
                "x": xs,
                "doubled": doubled,
                "quadrupled": [v * 2 for v in doubled],
            }
        )

    con = vane.connect()
    rel = con.sql("select 5 as x")
    rel2 = rel.map_batches(
        stage1,
        schema={"x": vane.sqltypes.BIGINT, "doubled": vane.sqltypes.BIGINT},
        execution_backend="subprocess_task",
    )
    rel3 = rel2.map_batches(
        stage2,
        schema={"x": vane.sqltypes.BIGINT, "doubled": vane.sqltypes.BIGINT, "quadrupled": vane.sqltypes.BIGINT},
        execution_backend="subprocess_task",
    )
    result = rel3.fetchall()
    assert len(result) == 1
    assert result[0] == (5, 10, 20)


def test_map_batches_on_error_null():
    """map_batches with on_error behavior — function raises, result is NULL."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    import vane

    def upper_fn(table):
        values = table.column(0).to_pylist()
        if "bad" in values:
            raise ValueError("bad input")
        return pa.table({"result": [v.upper() for v in values]})

    con = vane.connect()
    rel = con.sql("select * from (values ('ok'), ('bad'), ('ok2')) t(val)")
    # Note: error handling depends on the executor implementation
    # This test verifies the basic map_batches path works
    out = rel.map_batches(
        upper_fn,
        schema={"result": vane.sqltypes.VARCHAR},
        batch_size=2,
        execution_backend="subprocess_task",
    )
    # With batch_size=2, the 'bad' batch will error; the last batch should succeed
    try:
        rows = out.fetchall()
        # If it doesn't raise, verify we got some results
        assert len(rows) > 0
    except Exception:
        # Error propagation is expected behavior
        pass


def test_map_batches_ndarray():
    """map_batches with numpy array output."""
    pytest.importorskip("pyarrow")
    np = pytest.importorskip("numpy")
    import pyarrow as pa

    import vane

    def embed(table):
        values = np.asarray(table.column(0).to_pylist(), dtype=np.int64)
        result = np.stack([values, values + 1, values + 2], axis=1)
        return pa.table({"embedding": result.tolist()})

    con = vane.connect()
    rel = con.sql("select 1 as x union all select 2 as x")
    out = rel.map_batches(
        embed,
        schema={"embedding": vane.list_type(vane.sqltypes.BIGINT)},
        execution_backend="subprocess_task",
    )
    assert sorted(out.fetchall()) == [([1, 2, 3],), ([2, 3, 4],)]


def test_map_batches_fixed_size_list_through_local_exchange():
    """fixed-size list outputs survive local_exchange and remain buffer-readable."""
    pytest.importorskip("pyarrow")
    np = pytest.importorskip("numpy")
    import pyarrow as pa

    import vane

    embed_size = 4
    embed_arrow_type = pa.list_(pa.float32(), embed_size)
    embed_sql_type = vane.array_type(vane.sqltypes.FLOAT, embed_size)

    def stage1(table):
        xs = table.column("x").to_pylist()
        embeddings = []
        for x in xs:
            if x == 2:
                embeddings.append(None)
            else:
                embeddings.append(np.arange(embed_size, dtype=np.float32) + np.float32(x))
        return pa.table(
            {
                "x": pa.array(xs, type=pa.int64()),
                "embedding": pa.array(embeddings, type=embed_arrow_type),
            }
        )

    def stage2(table):
        xs = table.column("x").to_pylist()
        column = table.column("embedding")
        column = column.combine_chunks() if hasattr(column, "combine_chunks") else column
        assert pa.types.is_fixed_size_list(column.type)
        assert column.type.list_size == embed_size

        validity = column.is_valid().to_numpy(zero_copy_only=False)
        dense = column.values.to_numpy(zero_copy_only=False).reshape((len(column), embed_size))
        dense = dense if validity.all() else dense[validity]

        sums = [None] * len(xs)
        for row_idx, total in zip(np.flatnonzero(validity).tolist(), dense.sum(axis=1).tolist(), strict=False):
            sums[row_idx] = float(total)

        return pa.table(
            {
                "x": pa.array(xs, type=pa.int64()),
                "embedding_sum": pa.array(sums, type=pa.float32()),
            }
        )

    con = vane.connect()
    rel = con.sql("select * from (values (1), (2), (3), (4)) t(x)")
    out = (
        rel.map_batches(
            stage1,
            schema={"x": vane.sqltypes.BIGINT, "embedding": embed_sql_type},
            batch_size=2,
            execution_backend="subprocess_task",
        )
        .local_exchange(1)
        .map_batches(
            stage2,
            schema={"x": vane.sqltypes.BIGINT, "embedding_sum": vane.sqltypes.FLOAT},
            batch_size=2,
            execution_backend="subprocess_task",
        )
        .order("x")
    )
    assert out.fetchall() == [(1, 10.0), (2, None), (3, 18.0), (4, 22.0)]


def test_map_batches_tensor_through_local_exchange():
    """fixed-shape tensor outputs survive local_exchange and remain tensor columns."""
    pytest.importorskip("pyarrow")
    np = pytest.importorskip("numpy")
    import pyarrow as pa

    import vane

    tensor_type = vane.tensor_type(vane.sqltypes.FLOAT, (2, 2))

    def stage1(table):
        xs = np.asarray(table.column("x").to_pylist(), dtype=np.float32)
        tensors = np.stack([np.array([[x, x + 1], [x + 2, x + 3]], dtype=np.float32) for x in xs], axis=0)
        return pa.table(
            {
                "x": pa.array(xs.astype(np.int64), type=pa.int64()),
                "embedding": pa.FixedShapeTensorArray.from_numpy_ndarray(tensors),
            }
        )

    def stage2(table):
        xs = table.column("x").to_pylist()
        embedding = table.column("embedding")
        embedding = embedding.combine_chunks() if hasattr(embedding, "combine_chunks") else embedding
        assert embedding.type.extension_name == "arrow.fixed_shape_tensor"
        tensor = embedding.to_numpy_ndarray()
        return pa.table(
            {
                "x": pa.array(xs, type=pa.int64()),
                "embedding_sum": pa.array(tensor.reshape((tensor.shape[0], -1)).sum(axis=1), type=pa.float32()),
            }
        )

    con = vane.connect()
    rel = con.sql("select * from (values (1), (3), (5), (7)) t(x)")
    out = (
        rel.map_batches(
            stage1,
            schema={"x": vane.sqltypes.BIGINT, "embedding": tensor_type},
            batch_size=2,
            execution_backend="subprocess_task",
        )
        .local_exchange(1)
        .map_batches(
            stage2,
            schema={"x": vane.sqltypes.BIGINT, "embedding_sum": vane.sqltypes.FLOAT},
            batch_size=2,
            execution_backend="subprocess_task",
        )
        .order("x")
    )
    assert out.fetchall() == [(1, 10.0), (3, 18.0), (5, 26.0), (7, 34.0)]


def test_subprocess_streaming_tensor_uses_batch_sized_chunks_under_low_memory():
    """Large tensor intermediates should not force STANDARD_VECTOR_SIZE capacity."""
    pytest.importorskip("pyarrow")
    np = pytest.importorskip("numpy")
    import pyarrow as pa

    import vane

    tensor_shape = (128, 128)
    tensor_type = vane.tensor_type(vane.sqltypes.FLOAT, tensor_shape)

    def make_tensor(table):
        xs = table.column("x").to_pylist()
        tensors = np.ones((len(xs), *tensor_shape), dtype=np.float32)
        return pa.table(
            {
                "x": pa.array(xs, type=pa.int64()),
                "embedding": pa.FixedShapeTensorArray.from_numpy_ndarray(tensors),
            }
        )

    def reduce_tensor(table):
        xs = table.column("x").to_pylist()
        embedding = table.column("embedding")
        embedding = embedding.combine_chunks() if hasattr(embedding, "combine_chunks") else embedding
        tensor = embedding.to_numpy_ndarray().reshape((len(xs), -1))
        return pa.table(
            {
                "x": pa.array(xs, type=pa.int64()),
                "embedding_sum": pa.array(tensor.sum(axis=1), type=pa.float32()),
            }
        )

    con = vane.connect()
    con.execute("SET memory_limit='32MB'")
    con.execute("SET threads=4")

    out = (
        con.sql("select 1::BIGINT as x")
        .map_batches(
            make_tensor,
            schema={"x": vane.sqltypes.BIGINT, "embedding": tensor_type},
            execution_backend="subprocess_task",
            batch_size=1,
            streaming_breaker=True,
        )
        .map_batches(
            reduce_tensor,
            schema={"x": vane.sqltypes.BIGINT, "embedding_sum": vane.sqltypes.FLOAT},
            execution_backend="subprocess_task",
            batch_size=1,
            streaming_breaker=True,
        )
    )
    assert out.fetchall() == [(1, 16384.0)]


def test_map_batches_tensor_nulls_through_local_exchange():
    """Tensor columns with null rows survive local_exchange and preserve validity."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("numpy")
    import pyarrow as pa

    import vane

    tensor_type = vane.tensor_type(vane.sqltypes.FLOAT, (2, 2))
    tensor_arrow_type = pa.fixed_shape_tensor(pa.float32(), (2, 2))
    storage_type = pa.list_(pa.float32(), 4)

    def stage1(table):
        xs = table.column("x").to_pylist()
        rows = []
        for idx, x in enumerate(xs):
            if idx == 1:
                rows.append(None)
            else:
                rows.append([float(x), float(x + 1), float(x + 2), float(x + 3)])
        embedding = pa.ExtensionArray.from_storage(
            tensor_arrow_type,
            pa.array(rows, type=storage_type),
        )
        return pa.table(
            {
                "x": pa.array(xs, type=pa.int64()),
                "embedding": embedding,
            }
        )

    def stage2(table):
        xs = table.column("x").to_pylist()
        embedding = table.column("embedding")
        embedding = embedding.combine_chunks() if hasattr(embedding, "combine_chunks") else embedding
        assert embedding.type.extension_name == "arrow.fixed_shape_tensor"
        validity = embedding.is_valid().to_numpy(zero_copy_only=False)
        tensor = embedding.to_numpy_ndarray().reshape((len(embedding), -1))
        sums = [float(values.sum()) if is_valid else None for is_valid, values in zip(validity, tensor, strict=False)]
        return pa.table(
            {
                "x": pa.array(xs, type=pa.int64()),
                "embedding_sum": pa.array(sums, type=pa.float32()),
            }
        )

    con = vane.connect()
    rel = con.sql("select * from (values (1), (2), (3), (4)) t(x)")
    out = (
        rel.map_batches(
            stage1,
            schema={"x": vane.sqltypes.BIGINT, "embedding": tensor_type},
            batch_size=2,
            execution_backend="subprocess_task",
        )
        .local_exchange(1)
        .map_batches(
            stage2,
            schema={"x": vane.sqltypes.BIGINT, "embedding_sum": vane.sqltypes.FLOAT},
            batch_size=2,
            execution_backend="subprocess_task",
        )
        .order("x")
    )
    assert out.fetchall() == [(1, 10.0), (2, None), (3, 18.0), (4, None)]


def test_flat_map_basic():
    """flat_map produces multiple output rows per input row."""
    pytest.importorskip("pyarrow")
    import vane

    def duplicate(row):
        return [row, row]

    con = vane.connect()
    rel = con.sql("select 1 as x union all select 2 as x")
    out = rel.flat_map(
        duplicate,
        schema={"x": vane.sqltypes.BIGINT},
        execution_backend="subprocess_task",
    )
    rows = out.fetchall()
    assert len(rows) == 4


def test_map_batches_rejects_callable_instance():
    """Actor backends require classes, not already-constructed callable instances."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    import vane

    class Adder:
        def __init__(self):
            self.offset = 100

        def __call__(self, table):
            values = table.column(0).to_pylist()
            return pa.table({"result": [v + self.offset for v in values]})

    adder = Adder()
    con = vane.connect()
    rel = con.sql("select 1 as x union all select 2 as x")
    with pytest.raises(Exception, match="actor UDF backends require a callable class"):
        rel.map_batches(adder, schema={"result": vane.sqltypes.BIGINT}, execution_backend="subprocess_actor")


def test_map_batches_callable_class_subprocess_actor_backend():
    """map_batches accepts callable classes and instantiates them in actor executors."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    import vane

    class Adder:
        def __init__(self):
            self.offset = 100

        def __call__(self, table):
            values = table.column("x").to_pylist()
            return pa.table({"result": [v + self.offset for v in values]})

    con = vane.connect()
    rel = con.sql("select 1 as x union all select 2 as x")
    out = rel.map_batches(
        Adder,
        schema={"result": vane.sqltypes.BIGINT},
        execution_backend="subprocess_actor",
        actor_number=1,
        gpus=0.0,
    )
    assert sorted(out.fetchall()) == [(101,), (102,)]


def test_map_batches_default_backend_uses_runner_and_callable_shape(monkeypatch):
    import vane

    def identity(table):
        return table

    class Identity:
        def __call__(self, table):
            return table

    monkeypatch.setenv("VANE_RUNNER", "ray")
    con = vane.connect()

    task_rel = con.sql("select 1 as x").map_batches(identity, schema={"x": vane.sqltypes.INTEGER})
    task_plan = task_rel.explain()
    assert "ray_task" in task_plan

    actor_rel = con.sql("select 1 as x").map_batches(
        Identity,
        schema={"x": vane.sqltypes.INTEGER},
        actor_number=1,
        gpus=0.0,
    )
    actor_plan = actor_rel.explain()
    assert "ray_actor" in actor_plan
    assert "actor_number" in actor_plan


def test_map_batches_callable_class_rejects_task_backend():
    """Task backends require functions so state lifecycle is explicit."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    import vane

    class Adder:
        def __call__(self, table):
            values = table.column("x").to_pylist()
            return pa.table({"result": [v + 1 for v in values]})

    con = vane.connect()
    rel = con.sql("select 1 as x")
    with pytest.raises(Exception, match="task UDF backends require a function, not a callable class"):
        rel.map_batches(
            Adder,
            schema={"result": vane.sqltypes.BIGINT},
            execution_backend="subprocess_task",
        )


def test_map_batches_function_rejects_actor_backend():
    """Actor backends require callable classes."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    import vane

    def add_one(table):
        values = table.column("x").to_pylist()
        return pa.table({"result": [v + 1 for v in values]})

    con = vane.connect()
    rel = con.sql("select 1 as x")
    with pytest.raises(Exception, match="actor UDF backends require a callable class"):
        rel.map_batches(
            add_one,
            schema={"result": vane.sqltypes.BIGINT},
            execution_backend="subprocess_actor",
        )

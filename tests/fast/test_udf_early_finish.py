# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Early downstream completion must retire backpressured UDF producers."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest


@pytest.fixture
def ray_subprocess_env(_ray_local_cluster):
    _, address, _ = _ray_local_cluster
    return {**os.environ, "RAY_ADDRESS": address, "VANE_RUNNER": "ray", "VANE_PROGRESS": "0"}


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("chained", [False, True])
@pytest.mark.parametrize(
    "backend",
    [
        "subprocess_task",
        "subprocess_actor",
        pytest.param("ray_task", marks=pytest.mark.real_ray),
        pytest.param("ray_actor", marks=pytest.mark.real_ray),
    ],
)
def test_limit_retires_udf_input_and_allows_connection_reuse(tmp_path, request, backend, chained, threads):
    env = {**os.environ, "VANE_RUNNER": "local-fast", "VANE_PROGRESS": "0"}
    if backend.startswith("ray_"):
        env = request.getfixturevalue("ray_subprocess_env")
    script = textwrap.dedent(
        f"""
        import json
        from pathlib import Path
        import pyarrow as pa
        import vane

        calls = Path({str(tmp_path / "calls.jsonl")!r})

        def expand(batch):
            ids = batch['id'].to_pylist()
            with calls.open('a') as stream:
                stream.write(json.dumps(ids) + '\\n')
            return pa.table({{'id': pa.array([i for i in ids for _ in range(17)], type=pa.int64())}})

        class Expand:
            def __call__(self, batch):
                return expand(batch)

        def identity(batch):
            return batch

        backend = {backend!r}
        with vane.connect(config={{'threads': {threads}}}) as con:
            rel = con.sql('SELECT i::BIGINT AS id FROM range(320) t(i)')
            if {chained!r}:
                rel = rel.map_batches(
                    identity, schema={{'id': 'BIGINT'}},
                    execution_backend='ray_task' if backend.startswith('ray_') else 'subprocess_task',
                    batch_size=32, task_input_max_bytes=256, output_target_max_bytes=64)
            actor_options = {{'actor_number': 1}} if backend.endswith('_actor') else {{}}
            rel = rel.map_batches(
                Expand if backend.endswith('_actor') else expand,
                schema={{'id': 'BIGINT'}}, execution_backend=backend, cpus=1,
                batch_size=32, task_input_max_bytes=102400, output_target_max_bytes=64,
                **actor_options)
            for _ in range(2):
                values = rel.limit(5, offset=3).to_arrow_table()['id'].to_pylist()
                assert len(values) == 5 and all(0 <= value < 320 for value in values), values
                assert con.sql('SELECT 42').fetchone() == (42,)
        vane.teardown_runner()
        """
    )
    try:
        result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired as error:
        for name, output in (("stdout", error.stdout), ("stderr", error.stderr)):
            if output is not None:
                (tmp_path / f"probe.{name}").write_bytes(output.encode() if isinstance(output, str) else output)
        raise
    (tmp_path / "probe.stdout").write_text(result.stdout)
    (tmp_path / "probe.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr

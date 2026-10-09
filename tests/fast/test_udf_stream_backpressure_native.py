# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""The same slow-consumer contract for Ray and local task/actor producers."""

import os
import subprocess
import sys
import textwrap
import time

import pytest


@pytest.mark.parametrize(
    "backend",
    [
        "subprocess_task",
        "subprocess_actor",
        pytest.param("ray_task", marks=pytest.mark.real_ray),
        pytest.param("ray_actor", marks=pytest.mark.real_ray),
    ],
)
def test_fanout_producer_stops_when_downstream_cannot_accept_more(tmp_path, request, backend):
    env = {
        **os.environ,
        "VANE_RUNNER": "local-fast",
        "VANE_PROGRESS": "0",
        "VANE_LOCAL_SHM_STORE_BYTES": "64m",
        "VANE_LOCAL_SHM_REF_BUDGET_BYTES": "16m",
    }
    if backend.startswith("ray_"):
        _, address, _ = request.getfixturevalue("_ray_local_cluster")
        env.update(VANE_RUNNER="ray", RAY_ADDRESS=address)
    produced = tmp_path / "produced"
    entered = tmp_path / "entered"
    proceed = tmp_path / "proceed"
    script = textwrap.dedent(f"""
        from pathlib import Path
        import time
        import pyarrow as pa
        import vane

        produced = Path({str(produced)!r})
        entered = Path({str(entered)!r})
        proceed = Path({str(proceed)!r})

        def expand(batch):
            for value in range(128):
                with produced.open('a') as output:
                    output.write(str(value) + '\\n')
                yield pa.table({{'id': [value], 'payload': [b'x' * 32768]}})

        class Expand:
            def __call__(self, batch):
                return expand(batch)

        class Consume:
            def __call__(self, batch):
                entered.touch()
                deadline = time.monotonic() + 40
                while not proceed.exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError('test consumer was not released')
                    time.sleep(0.01)
                return batch.select(['id'])

        backend = {backend!r}
        with vane.connect(config={{'threads': 4}}) as con:
            options = {{'actor_number': 1}} if backend.endswith('_actor') else {{}}
            rel = con.sql('select 1::BIGINT as id').map_batches(
                Expand if options else expand,
                schema={{'id': 'BIGINT', 'payload': 'BLOB'}},
                execution_backend=backend, batch_size=1, output_batch_size=1,
                min_task_batch_size=1, task_input_max_bytes=1024, output_target_max_bytes=1024,
                **options)
            rel = rel.map_batches(
                Consume, schema={{'id': 'BIGINT'}}, batch_size=1, actor_number=1,
                min_task_batch_size=1, task_input_max_bytes=1024,
                execution_backend='ray_actor' if backend.startswith('ray_') else 'subprocess_actor')
            assert sorted(rel.to_arrow_table()['id'].to_pylist()) == list(range(128))
        vane.teardown_runner()
    """)
    with (tmp_path / "probe.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-c", script], env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while not entered.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert entered.exists(), (tmp_path / "probe.log").read_text()
            time.sleep(0.5)
            count = len(produced.read_text().splitlines())
            # Includes native handoff/compute buffers as well as the two-block
            # generator window. It must remain independent of total fanout.
            assert 1 <= count <= 16, (backend, count)
            proceed.touch()
            assert process.wait(timeout=60) == 0, (tmp_path / "probe.log").read_text()
        finally:
            proceed.touch()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
def test_expanding_pipeline_releases_real_storage_under_small_budget(backend):
    script = textwrap.dedent(f"""
        import gc
        import pyarrow as pa
        import vane
        from vane.execution.ref_bundle import local_shm_ref_budget_snapshot
        from vane.execution.udf_shm_store import local_shm_store_snapshot

        def expand(batch):
            for value in batch['id'].to_pylist():
                for part in range(8):
                    yield pa.table({{'id': [value * 8 + part], 'payload': [b'x' * (512 * 1024)]}})

        class Expand:
            def __call__(self, batch):
                return expand(batch)

        def consume(batch):
            return batch.select(['id'])

        class Consume:
            def __call__(self, batch):
                return consume(batch)

        def forward(batch):
            return batch

        class Forward:
            def __call__(self, batch):
                return forward(batch)

        backend = {backend!r}
        actors = {{'actor_number': 1}} if backend.endswith('_actor') else {{}}
        with vane.connect(config={{'threads': 4}}) as con:
            rel = con.sql('select range::BIGINT as id from range(8)').map_batches(
                Expand if actors else expand, schema={{'id': 'BIGINT', 'payload': 'BLOB'}},
                execution_backend=backend, batch_size=1, min_task_batch_size=1, task_input_max_bytes=1024,
                output_target_max_bytes=1024, **actors)
            # Several invocations consume slices from pooled output. No ACK
            # may release its physical charge or manufacture another credit.
            rel = rel.map_batches(
                Forward if actors else forward, schema={{'id': 'BIGINT', 'payload': 'BLOB'}},
                execution_backend=backend, batch_size=1, min_task_batch_size=1, task_input_max_bytes=1024, **actors)
            rel = rel.map_batches(
                Consume if actors else consume, schema={{'id': 'BIGINT'}},
                execution_backend=backend, batch_size=1, min_task_batch_size=1, task_input_max_bytes=1024,
                **actors)
            assert sorted(rel.to_arrow_table()['id'].to_pylist()) == list(range(64))
        vane.teardown_runner()
        gc.collect()
        assert local_shm_ref_budget_snapshot()['usage_bytes'] == 0
        assert local_shm_store_snapshot()['live_bytes'] == 0
    """)
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        timeout=90,
        env={
            **os.environ,
            "VANE_RUNNER": "local-fast",
            "VANE_PROGRESS": "0",
            "VANE_LOCAL_SHM_STORE_BYTES": "16m",
            "VANE_LOCAL_SHM_REF_BUDGET_BYTES": "2m",
        },
    )
    assert result.returncode == 0, result.stderr

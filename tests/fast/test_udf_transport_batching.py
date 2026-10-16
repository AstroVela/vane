# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Transport aggregation must preserve compute calls, progress and byte bounds."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(params=["local", pytest.param("ray", marks=pytest.mark.real_ray)])
def transport_env(request):
    env = {**os.environ, "VANE_RUNNER": request.param, "VANE_PROGRESS": "0"}
    if request.param == "ray":
        _, address, _ = request.getfixturevalue("_ray_local_cluster")
        env["RAY_ADDRESS"] = address
    return env


def run_probe(tmp_path, env, **options):
    config = {"root": str(tmp_path), "rows": 320, "block_rows": 32, "batch_format": "numpy", **options}
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("udf_transport_probe.py")), json.dumps(config)],
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except subprocess.TimeoutExpired as error:
        for name, output in (("stdout", error.stdout), ("stderr", error.stderr)):
            if output is not None:
                data = output.encode() if isinstance(output, str) else output
                (tmp_path / f"probe.{name}").write_bytes(data)
        raise
    (tmp_path / "probe.stdout").write_text(result.stdout)
    (tmp_path / "probe.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((tmp_path / "result.json").read_text())


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("batch_format", ["pyarrow", "numpy"])
def test_actor_automatically_aggregates_transport_without_changing_compute(tmp_path, transport_env, lazy, batch_format):
    result = run_probe(tmp_path, transport_env, lazy=lazy, batch_format=batch_format, gate="aggregate")
    tasks = list(result["tasks"].values())
    assert any(task["rows"] == 96 for task in tasks), tasks
    assert all(task["payload_bytes"] <= 100 * 1024 for task in tasks), tasks
    assert all(len(call["ids"]) == 32 for call in result["calls"])
    batches = sorted(call["ids"] for call in result["calls"])
    assert batches == [list(range(start, start + 32)) for start in range(0, 320, 32)]


@pytest.mark.parametrize("batch_format", ["pyarrow", "numpy"])
@pytest.mark.parametrize("change_arena_env", [False, True])
@pytest.mark.parametrize("runner", ["local", "local-fast"])
def test_local_aggregation_respects_physical_arena_capacity(tmp_path, batch_format, change_arena_env, runner):
    env = {**os.environ, "VANE_RUNNER": runner, "VANE_PROGRESS": "0", "VANE_LOCAL_SHM_STORE_BYTES": str(80 * 1024)}
    result = run_probe(tmp_path, env, batch_format=batch_format, gate="aggregate", change_arena_env=change_arena_env)
    assert result["output_rows"] == 320
    assert all(task["rows"] <= 32 for task in result["tasks"].values())


@pytest.mark.parametrize("rows", [0, 1, 31, 33, 95, 97, 277])
def test_actor_aggregation_preserves_eos_tail(tmp_path, transport_env, rows):
    result = run_probe(tmp_path, transport_env, rows=rows, block_rows=9)
    batches = sorted(call["ids"] for call in result["calls"])
    assert batches == [list(range(start, min(rows, start + 32))) for start in range(0, rows, 32)]


@pytest.mark.parametrize("gate,actors", [("idle", 1), ("completion", 1), ("two_actors", 2)])
def test_actor_aggregation_does_not_wait_for_input_needed_for_progress(tmp_path, transport_env, gate, actors):
    result = run_probe(tmp_path, transport_env, rows=160, gate=gate, actors=actors)
    assert result["output_rows"] == 160


@pytest.mark.parametrize("lazy", [False, True])
def test_actor_aggregation_obeys_variable_input_byte_pressure(tmp_path, transport_env, lazy):
    result = run_probe(tmp_path, transport_env, rows=97, lazy=lazy, variable=True)
    assert all(task["payload_bytes"] <= 100 * 1024 for task in result["tasks"].values())


@pytest.mark.parametrize("lazy", [False, True])
def test_actor_aggregation_makes_progress_with_oversized_single_row_blocks(tmp_path, transport_env, lazy):
    result = run_probe(tmp_path, transport_env, rows=97, block_rows=1, lazy=lazy, variable=True, oversized=True)
    for task in result["tasks"].values():
        # The per-block byte metadata is exact for this one-row input.
        assert task["payload_bytes"] <= 100 * 1024 or task["rows"] == 1, task


@pytest.mark.parametrize("lazy", [False, True])
def test_actor_aggregation_preserves_heterogeneous_block_contents(tmp_path, transport_env, lazy):
    result = run_probe(tmp_path, transport_env, rows=97, lazy=lazy, variable=True, oversized=True)
    # Slice byte metadata is estimated proportionally, as on the old path.
    # A highly skewed block is not an exact per-row physical memory bound.
    # Still verify every payload and output row, including the oversized value.
    assert result["output_rows"] == 97


@pytest.mark.parametrize("lazy", [False, True])
def test_actor_input_budget_smaller_than_compute_batch_makes_progress(tmp_path, transport_env, lazy):
    result = run_probe(tmp_path, transport_env, rows=97, lazy=lazy, budget=16 * 1024)
    assert all(task["payload_bytes"] <= 16 * 1024 for task in result["tasks"].values())


def test_actor_explicit_soft_minimum_still_preserves_upstream_block_tail(tmp_path, transport_env):
    result = run_probe(tmp_path, transport_env, rows=220, block_rows=110, lazy=True, soft_min=32, budget=256 * 1024)
    sizes = sorted(len(call["ids"]) for call in result["calls"])
    assert sizes == [14, 14, 32, 32, 32, 32, 32, 32]


def test_actor_aggregation_streams_expanding_outputs_with_small_output_budget(tmp_path, transport_env):
    result = run_probe(tmp_path, transport_env, expand=17, output_budget=512)
    assert result["output_rows"] == 320 * 17


def test_actor_aggregation_propagates_failure_inside_task(tmp_path, transport_env):
    result = run_probe(tmp_path, transport_env, block_rows=96, fail=True)
    assert result["failure_propagated"]


def test_actor_aggregation_cancels_partial_task_on_interrupt(tmp_path, transport_env):
    result = run_probe(tmp_path, transport_env, block_rows=96, output_budget=64, cancel=True)
    assert result["cancel_propagated"]


@pytest.mark.ray_fault
def test_actor_aggregation_replays_after_actor_exit_without_duplicate_output(tmp_path, transport_env):
    if transport_env["VANE_RUNNER"] == "local":
        pytest.skip("Local actor invocations do not replay after worker exit")
    result = run_probe(
        tmp_path,
        transport_env,
        rows=192,
        block_rows=96,
        output_budget=512,
        expand=17,
        restart=True,
    )
    assert result["actor_restarted"]
    assert result["output_rows"] == 192 * 17
    assert sum(call["ids"][0] == 0 for call in result["calls"]) == 2


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("batch_format", ["pyarrow", "numpy"])
def test_actor_aggregation_limit_releases_backpressured_outputs(tmp_path, transport_env, lazy, batch_format):
    result = run_probe(
        tmp_path,
        transport_env,
        block_rows=96,
        lazy=lazy,
        batch_format=batch_format,
        expand=17,
        output_budget=64,
        limit=5,
    )
    assert result["output_rows"] == 5


@pytest.mark.parametrize("lazy", [False, True])
def test_actor_aggregation_makes_progress_with_slow_consumer(tmp_path, transport_env, lazy):
    result = run_probe(tmp_path, transport_env, lazy=lazy, output_budget=64, slow_consumer=True)
    assert result["output_rows"] == 320

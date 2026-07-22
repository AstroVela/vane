# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from vane.runners.ray.ray_env import collect_vane_env_overrides
from vane.runners.ray.runner import (
    _configure_scan_task_backlog_env,
)


def test_ray_runner_does_not_inject_udf_stage_count_env(monkeypatch):
    monkeypatch.delenv("VANE_UDF_RAY_TASK_AUTO_STAGE_COUNT", raising=False)
    monkeypatch.delenv("VANE_UDF_RAY_TASK_OUTSTANDING_SCALE", raising=False)

    _configure_scan_task_backlog_env(None)

    assert "VANE_UDF_RAY_TASK_AUTO_STAGE_COUNT" not in os.environ
    assert "VANE_UDF_RAY_TASK_OUTSTANDING_SCALE" not in os.environ


def test_collect_vane_env_overrides_excludes_app_benchmark_env(monkeypatch):
    app_env_keys = (
        "INPUT_PATH",
        "OUTPUT_PATH",
        "TRANSCRIPTION_MODEL",
        "NUM_GPUS",
        "BATCH_SIZE",
        "NEW_SAMPLING_RATE",
        "WRITE_TASK_BACKLOG",
    )
    for key in app_env_keys:
        monkeypatch.setenv(key, f"value-for-{key}")
    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9000")
    monkeypatch.setenv("RAY_ADDRESS", "auto")

    overrides = collect_vane_env_overrides()

    for key in app_env_keys:
        assert key not in overrides
    assert overrides["VANE_RUNNER"] == "ray"
    assert overrides["AWS_ENDPOINT_URL"] == "http://127.0.0.1:9000"
    assert overrides["RAY_ADDRESS"] == "auto"

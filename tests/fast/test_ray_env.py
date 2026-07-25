# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from duckdb.runners.ray.ray_env import (
    build_explicit_session_process_env,
    build_session_runtime_env_vars,
    collect_vane_env_overrides,
    install_explicit_session_runtime_env,
    scrub_shared_runtime_session_env,
    session_environment_overrides,
)
from duckdb.runners.ray.runner import (
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
    monkeypatch.setenv("VANE_OPENAI_API_KEY", "session-api-key")
    monkeypatch.setenv("VANE_PRIVATE_KEY_PATH", "/tmp/session-private-key")
    monkeypatch.setenv("VANE_AUTH_HEADER", "session-auth-header")
    monkeypatch.setenv("VANE_S3_ENDPOINT", "http://session-endpoint")
    monkeypatch.setenv("VANE_SERVICE_URL", "https://session-service")
    monkeypatch.setenv("DUCKDB_SHUFFLE_DIRS", "file:///tmp/vane-shuffle")
    monkeypatch.setenv("DUCKDB_ISSUE75_SESSION_SECRET", "session-duckdb-secret")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9000")
    monkeypatch.setenv("RAY_ADDRESS", "auto")

    overrides = collect_vane_env_overrides()

    for key in app_env_keys:
        assert key not in overrides
    assert overrides["VANE_RUNNER"] == "ray"
    assert "VANE_OPENAI_API_KEY" not in overrides
    assert "VANE_PRIVATE_KEY_PATH" not in overrides
    assert "VANE_AUTH_HEADER" not in overrides
    assert "VANE_S3_ENDPOINT" not in overrides
    assert "VANE_SERVICE_URL" not in overrides
    assert overrides["DUCKDB_SHUFFLE_DIRS"] == "file:///tmp/vane-shuffle"
    assert "DUCKDB_ISSUE75_SESSION_SECRET" not in overrides
    assert "AWS_ENDPOINT_URL" not in overrides
    assert "RAY_ADDRESS" not in overrides


def test_shared_runtime_setup_scrubs_inherited_session_environment(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "inherited-aws-key")
    monkeypatch.setenv("DUCKDB_ISSUE75_SESSION_SECRET", "inherited-duckdb-secret")
    monkeypatch.setenv("DUCKDB_SHUFFLE_DIRS", "file:///tmp/vane-shuffle")
    monkeypatch.setenv("VANE_OPENAI_API_KEY", "inherited-vane-key")
    monkeypatch.setenv("VANE_SERVICE_URL", "https://inherited-service")
    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.setenv("VANE_SESSION_DIR", "/tmp/vane-job-session")

    scrub_shared_runtime_session_env()

    assert "AWS_ACCESS_KEY_ID" not in os.environ
    assert "DUCKDB_ISSUE75_SESSION_SECRET" not in os.environ
    assert os.environ["DUCKDB_SHUFFLE_DIRS"] == "file:///tmp/vane-shuffle"
    assert "VANE_OPENAI_API_KEY" not in os.environ
    assert "VANE_SERVICE_URL" not in os.environ
    assert os.environ["VANE_RUNNER"] == "ray"
    assert os.environ["VANE_SESSION_DIR"] == "/tmp/vane-job-session"


def test_explicit_session_process_env_replaces_inherited_managed_values():
    environment = build_explicit_session_process_env(
        {
            "AWS_ACCESS_KEY_ID": "session-key",
            "VANE_AUTH_HEADER": "session-auth",
        },
        base_env={
            "AWS_SECRET_ACCESS_KEY": "inherited-secret",
            "DUCKDB_ISSUE75_SESSION_SECRET": "inherited-duckdb-secret",
            "VANE_AUTH_HEADER": "inherited-auth",
            "VANE_RUNNER": "ray",
            "VANE_SESSION_DIR": "/tmp/vane-job-session",
            "PATH": "/usr/bin",
        },
    )

    assert environment == {
        "AWS_ACCESS_KEY_ID": "session-key",
        "VANE_AUTH_HEADER": "session-auth",
        "VANE_RUNNER": "ray",
        "VANE_SESSION_DIR": "/tmp/vane-job-session",
        "PATH": "/usr/bin",
    }


def test_session_runtime_setup_scrubs_inheritance_then_installs_explicit_context(monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "inherited-secret")
    monkeypatch.setenv("DUCKDB_ISSUE75_SESSION_SECRET", "inherited-duckdb-secret")
    monkeypatch.setenv("VANE_AUTH_HEADER", "inherited-auth")
    monkeypatch.setenv("VANE_RUNNER", "job-runner")
    monkeypatch.setenv("VANE_SESSION_DIR", "/tmp/vane-job-session")
    config = {
        "AWS_ACCESS_KEY_ID": "session-key",
        "DUCKDB_ISSUE75_SESSION_SECRET": "session-duckdb-secret",
        "VANE_AUTH_HEADER": "session-auth",
        "VANE_RUNNER": "session-runner",
        "VANE_SESSION_DIR": "/tmp/vane-connection-session",
    }

    assert session_environment_overrides(config) == {
        "AWS_ACCESS_KEY_ID": "session-key",
        "DUCKDB_ISSUE75_SESSION_SECRET": "session-duckdb-secret",
        "VANE_AUTH_HEADER": "session-auth",
    }

    carrier = build_session_runtime_env_vars(config)
    os.environ.update(carrier)
    install_explicit_session_runtime_env()

    assert "AWS_SECRET_ACCESS_KEY" not in os.environ
    assert os.environ["AWS_ACCESS_KEY_ID"] == "session-key"
    assert os.environ["DUCKDB_ISSUE75_SESSION_SECRET"] == "session-duckdb-secret"
    assert os.environ["VANE_AUTH_HEADER"] == "session-auth"
    assert os.environ["VANE_RUNNER"] == "job-runner"
    assert os.environ["VANE_SESSION_DIR"] == "/tmp/vane-job-session"
    assert all(key not in os.environ for key in carrier)

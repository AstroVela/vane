# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from duckdb._ray_progress_env import (
    dynamic_ray_progress_enabled,
    ray_log_to_driver_default,
)

_RAY_LOGGING_ENV_NAMES = ("RAY_LOG_TO_DRIVER", "RAY_BACKEND_LOG_LEVEL")


def _assert_import_preserves_ray_logging_env(
    module_name: str,
    *,
    initial_ray_env: dict[str, str] | None = None,
    import_ray_first: bool = False,
) -> None:
    env = os.environ.copy()
    env["VANE_RUNNER"] = "ray"
    env["VANE_PROGRESS"] = "auto"
    for name in _RAY_LOGGING_ENV_NAMES:
        env.pop(name, None)
    env.update(initial_ray_env or {})

    ray_import = "import ray" if import_ray_first else ""
    script = f"""
import importlib
import os

ray_logging_env_names = {_RAY_LOGGING_ENV_NAMES!r}
{ray_import}
before = {{name: os.environ.get(name) for name in ray_logging_env_names}}
importlib.import_module({module_name!r})
after = {{name: os.environ.get(name) for name in ray_logging_env_names}}
assert after == before, (before, after)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


@pytest.mark.parametrize("configured", [None, "", "   "])
def test_dynamic_ray_progress_uses_ray_default(monkeypatch, configured):
    if configured is None:
        monkeypatch.delenv("VANE_RUNNER", raising=False)
    else:
        monkeypatch.setenv("VANE_RUNNER", configured)
    monkeypatch.delenv("VANE_PROGRESS", raising=False)

    assert dynamic_ray_progress_enabled()


def test_dynamic_ray_progress_disables_ray_driver_log_forwarding(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.delenv("VANE_PROGRESS", raising=False)
    monkeypatch.delenv("RAY_LOG_TO_DRIVER", raising=False)
    monkeypatch.delenv("RAY_BACKEND_LOG_LEVEL", raising=False)

    assert dynamic_ray_progress_enabled()
    assert not ray_log_to_driver_default()
    assert os.environ.get("RAY_LOG_TO_DRIVER") is None
    assert os.environ.get("RAY_BACKEND_LOG_LEVEL") is None


def test_log_progress_keeps_ray_driver_log_default(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.setenv("VANE_PROGRESS", "raylog")
    monkeypatch.delenv("RAY_LOG_TO_DRIVER", raising=False)
    monkeypatch.delenv("RAY_BACKEND_LOG_LEVEL", raising=False)

    assert not dynamic_ray_progress_enabled()
    assert ray_log_to_driver_default()
    assert os.environ.get("RAY_LOG_TO_DRIVER") is None
    assert os.environ.get("RAY_BACKEND_LOG_LEVEL") is None


def test_explicit_ray_log_to_driver_is_preserved(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.delenv("VANE_PROGRESS", raising=False)
    monkeypatch.setenv("RAY_LOG_TO_DRIVER", "1")
    monkeypatch.delenv("RAY_BACKEND_LOG_LEVEL", raising=False)

    assert dynamic_ray_progress_enabled()
    assert ray_log_to_driver_default()
    assert os.environ.get("RAY_LOG_TO_DRIVER") == "1"
    assert os.environ.get("RAY_BACKEND_LOG_LEVEL") is None


@pytest.mark.parametrize(
    "module_name",
    [
        "vane",
        "duckdb.runners.ray.runner",
        "duckdb.runners.ray.driver",
    ],
)
def test_import_does_not_set_process_global_ray_logging_defaults(module_name):
    _assert_import_preserves_ray_logging_env(module_name)


@pytest.mark.parametrize(
    "initial_ray_env",
    [
        {},
        {"RAY_BACKEND_LOG_LEVEL": "debug"},
        {"RAY_LOG_TO_DRIVER": "1", "RAY_BACKEND_LOG_LEVEL": "debug"},
        {"RAY_LOG_TO_DRIVER": "0", "RAY_BACKEND_LOG_LEVEL": "error"},
    ],
)
def test_vane_import_coexists_with_existing_ray_user(initial_ray_env):
    _assert_import_preserves_ray_logging_env(
        "vane",
        initial_ray_env=initial_ray_env,
        import_ray_first=True,
    )


def test_ray_runner_scopes_progress_logging_to_ray_init(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.delenv("VANE_PROGRESS", raising=False)
    monkeypatch.delenv("RAY_LOG_TO_DRIVER", raising=False)
    monkeypatch.delenv("RAY_BACKEND_LOG_LEVEL", raising=False)

    from duckdb.runners.ray import runner as ray_runner_module

    init_kwargs = {}
    monkeypatch.setattr(ray_runner_module, "ensure_vane_session_dir", lambda: None)
    monkeypatch.setattr(ray_runner_module.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray_runner_module.ray, "init", lambda **kwargs: init_kwargs.update(kwargs))

    ray_runner_module.RayRunner(address="ray://cluster", max_task_backlog=None)

    assert init_kwargs == {
        "address": "ray://cluster",
        "log_to_driver": False,
    }
    assert os.environ.get("RAY_LOG_TO_DRIVER") is None
    assert os.environ.get("RAY_BACKEND_LOG_LEVEL") is None

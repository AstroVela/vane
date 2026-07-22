# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest

from vane._ray_progress_env import (
    configure_ray_progress_logging_defaults,
    dynamic_ray_progress_enabled,
    ray_log_to_driver_default,
)


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

    configure_ray_progress_logging_defaults()

    assert dynamic_ray_progress_enabled()
    assert not ray_log_to_driver_default()
    assert os.environ.get("RAY_LOG_TO_DRIVER") == "0"
    assert os.environ.get("RAY_BACKEND_LOG_LEVEL") == "fatal"


def test_log_progress_keeps_ray_driver_log_default(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.setenv("VANE_PROGRESS", "raylog")
    monkeypatch.delenv("RAY_LOG_TO_DRIVER", raising=False)
    monkeypatch.delenv("RAY_BACKEND_LOG_LEVEL", raising=False)

    configure_ray_progress_logging_defaults()

    assert not dynamic_ray_progress_enabled()
    assert ray_log_to_driver_default()
    assert os.environ.get("RAY_LOG_TO_DRIVER") is None
    assert os.environ.get("RAY_BACKEND_LOG_LEVEL") is None


def test_explicit_ray_log_to_driver_is_preserved(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.delenv("VANE_PROGRESS", raising=False)
    monkeypatch.setenv("RAY_LOG_TO_DRIVER", "1")
    monkeypatch.delenv("RAY_BACKEND_LOG_LEVEL", raising=False)

    configure_ray_progress_logging_defaults()

    assert dynamic_ray_progress_enabled()
    assert ray_log_to_driver_default()
    assert os.environ.get("RAY_LOG_TO_DRIVER") == "1"
    assert os.environ.get("RAY_BACKEND_LOG_LEVEL") is None


def test_explicit_ray_backend_log_level_is_preserved(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.delenv("VANE_PROGRESS", raising=False)
    monkeypatch.delenv("RAY_LOG_TO_DRIVER", raising=False)
    monkeypatch.setenv("RAY_BACKEND_LOG_LEVEL", "error")

    configure_ray_progress_logging_defaults()

    assert os.environ.get("RAY_LOG_TO_DRIVER") == "0"
    assert os.environ.get("RAY_BACKEND_LOG_LEVEL") == "error"

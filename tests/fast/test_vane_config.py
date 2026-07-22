# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

from vane import configure, current_config, env


def test_configure_sets_registered_environment_variables(monkeypatch):
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    monkeypatch.delenv("VANE_RAY_SCAN_TASK_SIZE_GROUPING", raising=False)

    cfg = configure(runner="ray", ray_scan_task_size_grouping=False)

    assert cfg.runner == "ray"
    assert cfg.ray_scan_task_size_grouping is False
    assert os.environ["VANE_RUNNER"] == "ray"
    assert os.environ["VANE_RAY_SCAN_TASK_SIZE_GROUPING"] == "False"
    assert current_config().runner == "ray"
    assert env.ray_scan_task_size_grouping is False


def test_configure_accepts_local_runner(monkeypatch):
    monkeypatch.delenv("VANE_RUNNER", raising=False)

    cfg = configure(runner="local")

    assert cfg.runner == "local"
    assert os.environ["VANE_RUNNER"] == "local"


def test_public_configuration_rejects_internal_direct_runner(monkeypatch):
    monkeypatch.delenv("VANE_RUNNER", raising=False)

    with pytest.raises(ValueError, match="runner must be 'local' or 'ray'"):
        configure(runner="local-fast")
    with pytest.raises(ValueError, match="runner must be 'local' or 'ray'"):
        env.runner = "local-fast"

    assert "VANE_RUNNER" not in os.environ


@pytest.mark.parametrize("configured", [None, "", "   "])
def test_empty_runner_configuration_resolves_to_ray(monkeypatch, configured):
    if configured is None:
        monkeypatch.delenv("VANE_RUNNER", raising=False)
    else:
        monkeypatch.setenv("VANE_RUNNER", configured)

    assert current_config().runner == "ray"


def test_configure_normalizes_empty_runner_to_ray(monkeypatch):
    monkeypatch.delenv("VANE_RUNNER", raising=False)

    cfg = configure(runner="")

    assert cfg.runner == "ray"
    assert os.environ["VANE_RUNNER"] == "ray"


def test_get_or_create_runner_does_not_create_runner_for_local_fast():
    script = """
import os
import vane.runners as runners

os.environ["VANE_RUNNER"] = "local-fast"
try:
    runners.get_or_create_runner()
except RuntimeError as exc:
    assert "does not create a runner" in str(exc)
else:
    raise AssertionError("expected no runner for local-fast")
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("configured", [None, "", "   "])
def test_get_or_infer_runner_type_defaults_to_ray(configured):
    script = f"""
import os
import vane.runners as runners

configured = {configured!r}
if configured is None:
    os.environ.pop("VANE_RUNNER", None)
else:
    os.environ["VANE_RUNNER"] = configured
assert runners.get_or_infer_runner_type() == "ray"
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("   ", "ray"),
        ("  LOCAL-FAST  ", "local-fast"),
        ("  LoCaL  ", "local"),
        ("  RaY  ", "ray"),
    ],
)
def test_get_or_infer_runner_type_uses_shared_normalization(configured, expected):
    script = f"""
import os
import vane.runners as runners

os.environ["VANE_RUNNER"] = {configured!r}
assert runners.get_or_infer_runner_type() == {expected!r}
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_runner_entry_points_use_shared_invalid_value_error(monkeypatch):
    expected = "Invalid runner type 'invalid-runner'. Please use 'local' or 'ray'."
    script = f"""
import os
import vane.runners as runners

os.environ["VANE_RUNNER"] = "  invalid-runner  "
try:
    runners.get_or_infer_runner_type()
except Exception as exc:
    assert {expected!r} in str(exc)
else:
    raise AssertionError("expected an invalid runner error")
"""
    subprocess.run([sys.executable, "-c", script], check=True)

    import vane

    monkeypatch.setenv("VANE_RUNNER", "  invalid-runner  ")

    @vane.func(return_dtype="INTEGER")
    def identity(value):
        return value

    relation = vane.connect().sql("SELECT 1::INTEGER AS value")
    with pytest.raises(vane.InvalidInputException, match=re.escape(expected)):
        relation.select(identity(vane.col("value"))).explain()


def test_get_or_create_runner_creates_local_runner():
    script = """
import os
import vane.runners as runners

os.environ["VANE_RUNNER"] = "local"
runner = runners.get_or_create_runner()
assert runner.name == "local"
assert runners.get_or_infer_runner_type() == "local"
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_ray_noop_does_not_reuse_local_runner():
    script = """
import vane.runners as runners

runners.set_runner_local()
try:
    runners.set_runner_ray(noop_if_initialized=True)
except RuntimeError as exc:
    assert "Cannot set runner more than once" in str(exc)
else:
    raise AssertionError("expected Ray setup to reject existing local runner")
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_config_registry_contains_stable_public_fields():
    fields = set(current_config().__dict__)

    assert "runner" in fields
    assert "ray_scan_task_open_cost_bytes" in fields
    assert "udf_parallel" in fields


def test_configure_rejects_unregistered_environment_knobs():
    with pytest.raises(AttributeError, match="Unknown config field"):
        configure(unknown_option=True)

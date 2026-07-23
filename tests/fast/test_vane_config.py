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
import duckdb.runners as runners

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
import duckdb.runners as runners

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
import duckdb.runners as runners

os.environ["VANE_RUNNER"] = {configured!r}
assert runners.get_or_infer_runner_type() == {expected!r}
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_runner_entry_points_use_shared_invalid_value_error(monkeypatch):
    expected = "Invalid runner type 'invalid-runner'. Please use 'local' or 'ray'."
    script = f"""
import os
import duckdb.runners as runners

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
import duckdb.runners as runners

os.environ["VANE_RUNNER"] = "local"
runner = runners.get_or_create_runner()
assert runner.name == "local"
assert runners.get_or_infer_runner_type() == "local"
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_concurrent_runner_initialization_does_not_deadlock_on_gil():
    script = r"""
import faulthandler
import os
import sys
import threading

sys.setswitchinterval(1000.0)
faulthandler.dump_traceback_later(5.0, repeat=False)

import duckdb
import duckdb.runners.local.runner as local_runner_module

vane_runners = duckdb.vane_runners_cpp
vane_runners.teardown_runner()
os.environ["VANE_RUNNER"] = "local"

barrier = threading.Barrier(2)
initializer_entered = threading.Event()
constructor_calls = []
results = []
errors = []


class BlockingLocalRunner:
    name = "local"

    def __init__(self, **_kwargs):
        constructor_calls.append(None)
        initializer_entered.set()
        # The first initializer releases the GIL here. The second thread is
        # already holding the GIL when it returns from the same barrier and
        # immediately enters the compiled runner binding.
        barrier.wait(timeout=5.0)


local_runner_module.LocalRunner = BlockingLocalRunner


def initialize():
    try:
        results.append(vane_runners.get_or_create_runner())
    except BaseException as exc:
        errors.append(exc)


def contend():
    try:
        barrier.wait(timeout=5.0)
        results.append(vane_runners.get_or_create_runner())
    except BaseException as exc:
        errors.append(exc)


initializer = threading.Thread(target=initialize, daemon=True, name="runner-initializer")
initializer.start()
assert initializer_entered.wait(timeout=5.0)

waiter = threading.Thread(target=contend, daemon=True, name="runner-waiter")
waiter.start()

initializer.join(timeout=5.0)
waiter.join(timeout=5.0)
faulthandler.cancel_dump_traceback_later()

assert not initializer.is_alive()
assert not waiter.is_alive()
assert errors == []
assert len(constructor_calls) == 1
assert len(results) == 2
assert results[0] is results[1]
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=10)


def test_failed_runner_initialization_is_not_published_and_can_retry():
    script = r"""
import os

import duckdb
import duckdb.runners.local.runner as local_runner_module

vane_runners = duckdb.vane_runners_cpp
vane_runners.teardown_runner()
os.environ["VANE_RUNNER"] = "local"
attempts = 0


class FlakyLocalRunner:
    name = "local"

    def __init__(self, **_kwargs):
        global attempts
        attempts += 1
        self.ready = False
        if attempts == 1:
            raise RuntimeError("forced runner initialization failure")
        self.ready = True


local_runner_module.LocalRunner = FlakyLocalRunner

try:
    vane_runners.get_or_create_runner()
except RuntimeError as exc:
    assert "forced runner initialization failure" in str(exc)
else:
    raise AssertionError("the first runner initialization should fail")

assert vane_runners.get_runner() is None

runner = vane_runners.get_or_create_runner()
assert attempts == 2
assert runner.ready is True
assert vane_runners.get_runner() is runner
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=10)


def test_failed_runner_initialization_wakes_waiter_to_retry():
    script = r"""
import faulthandler
import os
import sys
import threading

sys.setswitchinterval(1000.0)
faulthandler.dump_traceback_later(5.0, repeat=False)

import duckdb
import duckdb.runners.local.runner as local_runner_module

vane_runners = duckdb.vane_runners_cpp
vane_runners.teardown_runner()
os.environ["VANE_RUNNER"] = "local"

barrier = threading.Barrier(2)
initializer_entered = threading.Event()
attempts = 0
results = []
expected_errors = []
unexpected_errors = []


class FailOnceLocalRunner:
    name = "local"

    def __init__(self, **_kwargs):
        global attempts
        attempts += 1
        if attempts == 1:
            initializer_entered.set()
            barrier.wait(timeout=5.0)
            raise RuntimeError("first attempt failed")
        self.ready = True


local_runner_module.LocalRunner = FailOnceLocalRunner


def initialize():
    try:
        vane_runners.get_or_create_runner()
    except RuntimeError as exc:
        expected_errors.append(str(exc))
    except BaseException as exc:
        unexpected_errors.append(exc)


def retry():
    try:
        barrier.wait(timeout=5.0)
        results.append(vane_runners.get_or_create_runner())
    except BaseException as exc:
        unexpected_errors.append(exc)


initializer = threading.Thread(target=initialize, daemon=True, name="failing-initializer")
initializer.start()
assert initializer_entered.wait(timeout=5.0)

waiter = threading.Thread(target=retry, daemon=True, name="retrying-waiter")
waiter.start()

initializer.join(timeout=5.0)
waiter.join(timeout=5.0)
faulthandler.cancel_dump_traceback_later()

assert not initializer.is_alive()
assert not waiter.is_alive()
assert expected_errors == ["first attempt failed"]
assert unexpected_errors == []
assert attempts == 2
assert len(results) == 1
assert results[0].ready is True
assert vane_runners.get_runner() is results[0]
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=10)


def test_ray_noop_does_not_reuse_local_runner():
    script = """
import duckdb.runners as runners

runners.set_runner_local()
try:
    runners.set_runner_ray(noop_if_initialized=True)
except RuntimeError as exc:
    assert "Cannot set runner more than once" in str(exc)
else:
    raise AssertionError("expected Ray setup to reject existing local runner")
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_ray_noop_reuses_existing_runner_without_validating_address():
    script = """
import duckdb
import duckdb.runners.ray.runner as ray_runner_module


class FakeRayRunner:
    name = "ray"

    def __init__(self, *_args):
        pass

    def close(self):
        pass


ray_runner_module.RayRunner = FakeRayRunner
vane_runners = duckdb.vane_runners_cpp
vane_runners.teardown_runner()

first = vane_runners.set_runner_ray(None, False, None, False)
second = vane_runners.set_runner_ray(object(), True, None, False)

assert second is first
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=10)


def test_config_registry_contains_stable_public_fields():
    fields = set(current_config().__dict__)

    assert "runner" in fields
    assert "ray_scan_task_open_cost_bytes" in fields
    assert "udf_parallel" in fields


def test_configure_rejects_unregistered_environment_knobs():
    with pytest.raises(AttributeError, match="Unknown config field"):
        configure(unknown_option=True)

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
    monkeypatch.delenv("VANE_RAY_SCAN_SPLIT_MIN_COUNT", raising=False)
    monkeypatch.delenv("VANE_NDJSON_MAX_SPLIT_BYTES", raising=False)

    cfg = configure(runner="ray", ray_scan_split_min_count=16, ndjson_max_split_bytes=128 * 1024 * 1024)

    assert cfg.runner == "ray"
    assert cfg.ray_scan_split_min_count == 16
    assert os.environ["VANE_RUNNER"] == "ray"
    assert os.environ["VANE_RAY_SCAN_SPLIT_MIN_COUNT"] == "16"
    assert current_config().runner == "ray"
    assert env.ray_scan_split_min_count == 16
    assert cfg.ndjson_max_split_bytes == 128 * 1024 * 1024
    assert os.environ["VANE_NDJSON_MAX_SPLIT_BYTES"] == "134217728"
    assert current_config().ndjson_max_split_bytes == 128 * 1024 * 1024


@pytest.mark.parametrize("runner", ["local", "local-fast", "  LOCAL-FAST  ", "  LoCaL  "])
def test_configure_accepts_local_runner(monkeypatch, runner):
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    cfg = configure(runner=runner)
    assert cfg.runner == current_config().runner == "local"
    assert os.environ["VANE_RUNNER"] == "local"
    env.runner = runner
    assert env.runner == "local"
    assert os.environ["VANE_RUNNER"] == "local"
    monkeypatch.setenv("VANE_RUNNER", runner)
    assert current_config().runner == "local"


def test_public_configuration_rejects_unknown_runner(monkeypatch):
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    with pytest.raises(ValueError, match="runner must be"):
        configure(runner="unknown")
    with pytest.raises(ValueError, match="runner must be"):
        env.runner = "unknown"
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


@pytest.mark.parametrize("runner", ["local", "local-fast"])
def test_native_local_execution_does_not_create_runner(monkeypatch, runner):
    import vane

    monkeypatch.setenv("VANE_RUNNER", runner)
    assert vane.get_runner() is None
    with pytest.raises(RuntimeError, match="does not create a runner"):
        vane.get_or_create_runner()


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
        ("  LOCAL-FAST  ", "local"),
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
    expected = "Invalid runner type 'invalid-runner'. Please use 'local-fast', 'local', or 'ray'."
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

    with pytest.raises(vane.InvalidInputException, match=re.escape(expected)):
        relation = vane.connect().sql("SELECT 1::INTEGER AS value")
        relation.select(identity(vane.col("value"))).explain()


def test_concurrent_runner_initialization_does_not_deadlock_on_gil():
    script = r"""
import faulthandler
import os
import sys
import threading

sys.setswitchinterval(1000.0)
faulthandler.dump_traceback_later(5.0, repeat=False)

import vane
import vane.runners.ray.runner as ray_runner_module

vane_runners = vane
vane_runners.teardown_runner()
os.environ["VANE_RUNNER"] = "ray"

barrier = threading.Barrier(2)
initializer_entered = threading.Event()
constructor_calls = []
results = []
errors = []


class BlockingRayRunner:
    name = "ray"

    def __init__(self, *_args, **_kwargs):
        constructor_calls.append(None)
        initializer_entered.set()
        # The first initializer releases the GIL here. The second thread is
        # already holding the GIL when it returns from the same barrier and
        # immediately enters the compiled runner binding.
        barrier.wait(timeout=5.0)


ray_runner_module.RayRunner = BlockingRayRunner


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

import vane
import vane.runners.ray.runner as ray_runner_module

vane_runners = vane
vane_runners.teardown_runner()
os.environ["VANE_RUNNER"] = "ray"
attempts = 0


class FlakyRayRunner:
    name = "ray"

    def __init__(self, *_args, **_kwargs):
        global attempts
        attempts += 1
        self.ready = False
        if attempts == 1:
            raise RuntimeError("forced runner initialization failure")
        self.ready = True


ray_runner_module.RayRunner = FlakyRayRunner

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

import vane
import vane.runners.ray.runner as ray_runner_module

vane_runners = vane
vane_runners.teardown_runner()
os.environ["VANE_RUNNER"] = "ray"

barrier = threading.Barrier(2)
initializer_entered = threading.Event()
attempts = 0
results = []
expected_errors = []
unexpected_errors = []


class FailOnceRayRunner:
    name = "ray"

    def __init__(self, *_args, **_kwargs):
        global attempts
        attempts += 1
        if attempts == 1:
            initializer_entered.set()
            barrier.wait(timeout=5.0)
            raise RuntimeError("first attempt failed")
        self.ready = True


ray_runner_module.RayRunner = FailOnceRayRunner


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


@pytest.mark.parametrize("first", ["local", "ray"])
@pytest.mark.parametrize("configured", [False, True])
def test_connection_runners_coexist_and_preserve_configuration(monkeypatch, tmp_path, first, configured):
    import pyarrow.parquet as pq

    import vane
    import vane.runners.ray.runner as ray_module

    created = []

    class CapturingRayRunner:
        def __init__(self, address, max_task_backlog):
            self.options = (address, max_task_backlog)
            self.calls = 0
            self.closed = False
            created.append(self)

        def run_write(self, plan):
            assert plan.session_config()["VANE_RUNNER"] == "ray"
            self.calls += 1
            return {"copy_operation_id": plan.idx(), "rows_copied": 1}

        def close(self):
            self.closed = True

    vane.teardown_runner()
    monkeypatch.setattr(ray_module, "RayRunner", CapturingRayRunner)
    order = [first, "ray" if first == "local" else "local"]
    connections = {}
    try:
        for kind in order:
            monkeypatch.setenv("VANE_RUNNER", kind)
            connections[kind] = vane.connect()
        assert created == []
        if configured:
            for kind in order:
                if kind == "local":
                    assert vane.set_runner_local() is None
                else:
                    vane.set_runner_ray("ray://configured", max_task_backlog=11)
        monkeypatch.setenv("VANE_RUNNER", "invalid")
        for index, kind in enumerate(order * 2):
            target = tmp_path / f"{kind}-{index}.parquet"
            connections[kind].sql("SELECT 1 AS value").write_parquet(str(target))
            assert os.environ["VANE_RUNNER"] == "invalid"
            if kind == "local":
                assert pq.read_table(target).to_pylist() == [{"value": 1}]
            else:
                assert not target.exists()
        assert len(created) == 1 and created[0].calls == 2
        if configured:
            assert created[0].options == ("ray://configured", 11)
        for kind in order:
            monkeypatch.setenv("VANE_RUNNER", kind)
            assert vane.get_or_infer_runner_type() == kind
            assert vane.get_runner() is (created[0] if kind == "ray" else None)
        monkeypatch.setenv("VANE_RUNNER", "invalid")
        vane.teardown_runner()
        assert created[0].closed
    finally:
        for connection in connections.values():
            connection.close()
        vane.teardown_runner()


def test_selecting_local_does_not_wait_for_ray_initialization():
    script = r"""
import threading
import vane
import vane.runners.ray.runner as ray_module

entered = threading.Event()
release = threading.Event()
errors = []
class RayRunner:
    def __init__(self, *args):
        entered.set()
        assert release.wait(5)
ray_module.RayRunner = RayRunner

def initialize():
    try:
        vane._native.set_runner_ray()
    except BaseException as error:
        errors.append(error)
worker = threading.Thread(target=initialize, daemon=True)
worker.start()
try:
    assert entered.wait(5)
    assert vane.set_runner_local() is None
    assert vane.get_runner() is None
    with vane.connect() as connection:
        assert connection.execute("SELECT 42").fetchall() == [(42,)]
finally:
    release.set()
    worker.join(5)
assert not worker.is_alive()
assert errors == [], errors
vane.teardown_runner()
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=15)


def test_set_runner_ray_rejects_removed_force_client_mode():
    from vane import runners

    with pytest.raises(TypeError, match="force_client_mode"):
        runners.set_runner_ray(force_client_mode=True)


@pytest.mark.parametrize("method_name", ["run_iter", "run_iter_tables"])
def test_ray_runner_rejects_removed_results_buffer_size(method_name):
    from vane.runners.ray.runner import RayRunner

    runner = object.__new__(RayRunner)
    method = getattr(runner, method_name)

    with pytest.raises(TypeError, match="results_buffer_size"):
        method(object(), results_buffer_size=1)


def test_native_set_runner_ray_rejects_removed_fourth_argument():
    import vane

    with pytest.raises(TypeError):
        vane.set_runner_ray(None, False, None, False)


def test_ray_noop_reuses_existing_runner_without_validating_address():
    script = """
import vane
import vane.runners.ray.runner as ray_runner_module


class FakeRayRunner:
    name = "ray"

    def __init__(self, *_args):
        pass

    def close(self):
        pass


ray_runner_module.RayRunner = FakeRayRunner
vane_runners = vane
vane_runners.teardown_runner()

first = vane_runners.set_runner_ray(None, False, None)
second = vane_runners.set_runner_ray(object(), True, None)

assert second is first
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=10)


def test_config_registry_contains_stable_public_fields():
    fields = set(current_config().__dict__)

    assert "runner" in fields
    assert "ray_scan_split_min_count" in fields
    assert "udf_parallel" in fields


def test_configure_rejects_unregistered_environment_knobs():
    with pytest.raises(AttributeError, match="Unknown config field"):
        configure(unknown_option=True)

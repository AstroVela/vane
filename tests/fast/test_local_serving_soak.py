# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import runpy
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "validate_local_serving_soak.py"


def soak():
    return runpy.run_path(str(SCRIPT))


def soak_output_directory(tmp_path):
    root = os.environ.get("VANE_TEST_DIAGNOSTICS_DIR")
    output = (Path(root) / f"serving-soak-{uuid.uuid4().hex}" if root else tmp_path / "evidence").resolve()
    # The CLI creates the new directory. Report its location before starting so
    # outer timeouts and failures with an empty worker.log still identify it.
    print(f"Serving soak diagnostics: {output}", file=sys.stderr, flush=True)
    return output


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group watchdog")
@pytest.mark.timeout(150)
def test_native_serving_soak_reuses_one_runtime_across_fault_recovery(tmp_path):
    output = soak_output_directory(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(SCRIPT),
            "--output",
            str(output),
            "--rounds",
            "2",
            "--requests",
            "4",
            "--timeout",
            "120",
        ],
        capture_output=True,
        text=True,
        timeout=135,
    )
    log = output / "worker.log"
    assert completed.returncode == 0, (
        f"Serving soak diagnostics: {output}\n"
        + completed.stdout
        + completed.stderr
        + (log.read_text() if log.exists() else "")
    )
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "passed"
    worker = report["worker_report"]
    assert worker["runtime_sessions"] == 1
    assert worker["completed_rounds"] == 2
    assert worker["load_requests"] == 17
    assert worker["observed_worker_exit_failures"] == 2
    rounds = worker["recent_rounds"]
    assert [item["round"] for item in rounds] == [1, 2]
    assert worker["total_initializations"] == 5 + sum(item["cancellation_replacements"] for item in rounds)
    for item in rounds:
        assert item["healthy_additional_initializations"] == 0
        assert item["fault_replacements"] == {"udf_error": 1, "worker_exit": 1}
        assert item["transport"]["usage_bytes"] == 0
        assert item["transport"]["active_input_leases"] == 0
        state = item["resources"]
        assert state["reserved_resources"]["cpu"] == 1
        assert state["active_borrows"] == 0
        for name, key in (
            ("request_admission", "active_requests"),
            ("request_admission", "queued_requests"),
            ("request_admission", "cleanup_pending_requests"),
            ("task_admission", "running_tasks"),
            ("data", "usage_bytes"),
            ("result_delivery", "usage_bytes"),
        ):
            assert state[name][key] == 0
    assert worker["closed"]["reserved_models"] == 0
    assert worker["closed"]["closed"]
    assert worker["closed_transport"]["usage_bytes"] == 0
    assert report["diagnostics"]["resources.json"]
    resources = json.loads((output / "resources.json").read_text())
    assert {"runtime", "transport", "model_workers"} <= resources.keys()
    # Old UDF call markers and initialization log lines cannot grow per round.
    assert not list((output / "work").glob("calls-*"))
    assert len((output / "work" / "initializations").read_text().splitlines()) == 1


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group watchdog")
@pytest.mark.timeout(15)
@pytest.mark.parametrize("diagnostics_root", ["temporary", "absolute", "relative"])
def test_watchdog_captures_threads_and_kills_blocked_child_and_descendant(
    tmp_path, tmp_path_factory, monkeypatch, capsys, diagnostics_root
):
    monkeypatch.chdir(tmp_path)
    if diagnostics_root == "temporary":
        monkeypatch.delenv("VANE_TEST_DIAGNOSTICS_DIR", raising=False)
        expected_root = tmp_path
    else:
        expected_root = tmp_path_factory.mktemp("ci-artifacts") / "local-runtime"
        configured_root = (
            str(expected_root) if diagnostics_root == "absolute" else os.path.relpath(expected_root, tmp_path)
        )
        monkeypatch.setenv("VANE_TEST_DIAGNOSTICS_DIR", configured_root)
    output = soak_output_directory(tmp_path)
    assert output.parent == expected_root
    assert not output.exists()
    output.mkdir(parents=True)
    child = tmp_path / "blocked.py"
    descendant = (
        "import signal, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "while True:\n"
        "    Path('heartbeat').write_text(str(time.monotonic_ns()))\n"
        "    time.sleep(0.01)\n"
    )
    child.write_text(
        "import faulthandler, signal, subprocess, sys, threading\n"
        "from pathlib import Path\n"
        "log = open('threads.log', 'w')\n"
        "faulthandler.register(signal.SIGUSR1, file=log, all_threads=True)\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "Path('watchdog-ready').touch()\n"
        'Path(\'progress.json\').write_text(\'{"round": 2, "phase": "blocked"}\')\n'
        "Path('resources.json').write_text('{\"runtime\": {\"pending_cleanup\": 1}}')\n"
        f"subprocess.Popen([sys.executable, '-c', {descendant!r}])\n"
        "def blocked_cleanup():\n"
        "    lock = threading.Lock()\n"
        "    lock.acquire()\n"
        "    lock.acquire()\n"
        "blocked_cleanup()\n"
    )
    started = time.monotonic()
    report = soak()["supervise"](
        [sys.executable, "-I", str(child)], output, timeout=2, diagnostic_grace=0.3, termination_grace=0.3
    )
    assert time.monotonic() - started < 8
    assert report["status"] == "timed_out"
    assert report["child_returncode"] < 0
    assert "blocked_cleanup" in (output / "threads.log").read_text()
    assert json.loads((output / "resources.json").read_text())["runtime"]["pending_cleanup"] == 1
    assert json.loads((output / "progress.json").read_text())["round"] == 2
    assert json.loads((output / "report.json").read_text())["status"] == "timed_out"
    assert (output / "worker.log").read_text() == ""
    assert str(output) in capsys.readouterr().err
    heartbeat = (output / "heartbeat").read_text()
    time.sleep(0.1)
    assert (output / "heartbeat").read_text() == heartbeat
    if diagnostics_root != "temporary":
        # A retry in the same CI job must preserve the earlier failure evidence.
        retry_output = soak_output_directory(tmp_path)
        assert retry_output.parent == expected_root
        assert retry_output != output
        assert not retry_output.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX shell launchers")
@pytest.mark.parametrize("launcher", ["run_installed_pytest.sh", "run_release_tests.sh", "run_fast_tests.sh"])
@pytest.mark.parametrize("relative", [False, True])
def test_failed_launcher_preserves_diagnostics_outside_its_temporary_directory(tmp_path, launcher, relative):
    caller = tmp_path / "caller"
    caller.mkdir()
    root = caller / "diagnostics with spaces" / "artifacts"
    record = tmp_path / "probe.json"
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import json, os, runpy\n"
        "from pathlib import Path\n"
        f"test = runpy.run_path({str(Path(__file__).resolve())!r})\n"
        "output = test['soak_output_directory'](Path.cwd() / 'pytest-tmp')\n"
        "output.mkdir(parents=True)\n"
        "(output / 'resources.json').write_text('{\"pending_cleanup\": 1}')\n"
        f"Path({str(record)!r}).write_text(json.dumps({{\n"
        "    'cwd': str(Path.cwd()), 'output': str(output),\n"
        "    'root': os.environ['VANE_TEST_DIAGNOSTICS_DIR'],\n"
        "}))\n"
        "raise SystemExit(1)\n"
    )
    # Exercise each real launcher's environment and EXIT trap, substituting only
    # its pytest invocation so this regression does not recursively run a suite.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        '#!/bin/sh\nif [ "$1" = "-m" ] && [ "$2" = "pytest" ]; then\n'
        f"  exec {shlex.quote(sys.executable)} -I {shlex.quote(str(probe))}\n"
        "fi\n"
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    python.chmod(0o755)
    environment = dict(
        os.environ,
        PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
        VANE_TEST_DIAGNOSTICS_DIR=os.path.relpath(root, caller) if relative else str(root),
        VANE_FAST_TEST_JUNIT_DIR="",
        VANE_FAST_TEST_NON_RAY_SHARD_COUNT="1",
        VANE_FAST_TEST_NON_RAY_SHARD_INDEX="0",
        VANE_FAST_TEST_PROCESS_TIMEOUT_SECONDS="0",
    )
    args = {"run_installed_pytest.sh": ["probe.py"], "run_release_tests.sh": [], "run_fast_tests.sh": ["non-ray"]}
    completed = subprocess.run(
        ["bash", str(SCRIPT.parent / launcher), *args[launcher]],
        cwd=caller,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 1, completed.stdout + completed.stderr
    evidence = json.loads(record.read_text())
    assert not Path(evidence["cwd"]).exists(), "launcher must still remove its temporary test directory"
    output = Path(evidence["output"])
    assert (output / "resources.json").is_file(), f"launcher deleted diagnostic evidence at {output}"
    assert Path(evidence["root"]) == root
    assert output.parent == root
    assert str(output) in completed.stderr


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group watchdog")
@pytest.mark.timeout(15)
@pytest.mark.parametrize("shutdown", ["atexit", "atexit_after_error", "thread_join"])
def test_watchdog_captures_worker_interpreter_shutdown(tmp_path, shutdown):
    child = tmp_path / "shutdown.py"
    child.write_text(
        "import atexit, runpy, sys, threading\n"
        "from pathlib import Path\n"
        f"main = runpy.run_path({str(SCRIPT)!r})['main']\n"
        "def blocked_shutdown():\n"
        "    Path('shutdown-entered').touch()\n"
        "    lock = threading.Lock()\n"
        "    lock.acquire()\n"
        "    lock.acquire()\n"
        "def run_soak(*args, **kwargs):\n"
        f"    if {shutdown!r} == 'thread_join':\n"
        "        threading.Thread(target=blocked_shutdown, daemon=False).start()\n"
        "    else:\n"
        "        atexit.register(blocked_shutdown)\n"
        f"    if {shutdown!r} == 'atexit_after_error':\n"
        "        raise RuntimeError('planned worker failure')\n"
        "    return {'status': 'passed'}\n"
        "main.__globals__['run_soak'] = run_soak\n"
        "sys.argv = ['soak', '--worker', '--output', str(Path.cwd())]\n"
        "sys.exit(main())\n"
    )
    report = soak()["supervise"](
        [sys.executable, "-I", str(child)], tmp_path, timeout=2, diagnostic_grace=0.3, termination_grace=0.3
    )
    assert (tmp_path / "shutdown-entered").exists()
    assert report["status"] == "timed_out"
    assert report["child_returncode"] == -signal.SIGTERM
    assert "blocked_shutdown" in (tmp_path / "threads.log").read_text()
    if shutdown == "atexit_after_error":
        assert "planned worker failure" in (tmp_path / "worker.log").read_text()
    else:
        assert json.loads((tmp_path / "worker-report.json").read_text())["status"] == "passed"


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group watchdog")
@pytest.mark.parametrize("outcome", ["missing", "invalid", "failed", "passed", "nonzero"])
def test_supervisor_requires_a_successful_exit_and_report(tmp_path, outcome):
    child = tmp_path / "outcome.py"
    source = "from pathlib import Path\n"
    if outcome == "invalid":
        source += "Path('worker-report.json').write_text('not json')\n"
    elif outcome in ("passed", "failed", "nonzero"):
        status = "failed" if outcome == "failed" else "passed"
        source += f"Path('worker-report.json').write_text('{{\"status\": \"{status}\"}}')\n"
    if outcome == "nonzero":
        source += "raise SystemExit(7)\n"
    child.write_text(source)
    report = soak()["supervise"]([sys.executable, "-I", str(child)], tmp_path, timeout=5)
    assert report["status"] == ("passed" if outcome == "passed" else "failed")
    assert json.loads((tmp_path / "report.json").read_text())["status"] == report["status"]


@pytest.mark.parametrize("blocked", [False, True])
def test_diagnostic_failure_does_not_block_driver_or_erase_previous_snapshot(tmp_path, blocked):
    entered, release = threading.Event(), threading.Event()
    previous = {"runtime": {"active_requests": 1}}
    (tmp_path / "resources.json").write_text(json.dumps(previous))

    def snapshot():
        entered.set()
        if blocked:
            release.wait(10)
        raise RuntimeError("planned snapshot failure")

    with (tmp_path / "threads.log").open("w") as stacks:
        diagnostics = soak()["Diagnostics"](tmp_path, snapshot, stacks)
        diagnostics.thread.start()
        try:
            assert entered.wait(5)
            started = time.monotonic()
            diagnostics.close()
            assert time.monotonic() - started < 2
            assert json.loads((tmp_path / "resources.json").read_text()) == previous
        finally:
            release.set()
            diagnostics.thread.join(timeout=5)


@pytest.mark.parametrize(
    "options",
    [
        {"rounds": 1},
        {"rounds": True},
        {"requests": 1},
        {"requests": 10001},
        {"concurrency": 0},
        {"concurrency": 5},
        {"timeout": 0},
        {"timeout": float("nan")},
        {"timeout": float("inf")},
        {"gpu_device": "0"},
        {"gpu_device": "GPU-abcd"},
        {"gpu_device": "MIG-GPU-aaaaaaaa-0000-0000-0000-000000000001/1/0"},
        {"gpu_device": "GPU-aaaaaaaa-0000-0000-0000-000000000001\n"},
        {"gpu_device": 1},
    ],
)
def test_invalid_soak_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        soak()["validate_options"](**({"rounds": 2, "requests": 4, "concurrency": 4, "timeout": 60} | options))


def test_existing_evidence_directory_is_never_reused(tmp_path):
    marker = tmp_path / "worker-report.json"
    marker.write_text('{"status": "passed"}')
    completed = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--output", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode != 0
    assert marker.read_text() == '{"status": "passed"}'
    assert not (tmp_path / "worker.log").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group watchdog")
def test_cuda_soak_forwards_explicit_device_to_the_supervised_worker(tmp_path, monkeypatch):
    main = soak()["main"]
    device = "GPU-AAAAAAAA-0000-0000-0000-000000000001"
    commands = []

    def supervise(command, directory, *, timeout):
        commands.append(command)
        assert directory == tmp_path / "evidence"
        assert timeout == 600
        return {"status": "passed"}

    monkeypatch.setitem(main.__globals__, "supervise", supervise)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output", str(tmp_path / "evidence"), "--gpu-device", device])
    assert main() == 0
    assert len(commands) == 1
    command = commands[0]
    assert "--worker" in command
    assert command[command.index("--gpu-device") + 1] == "GPU-" + device[4:].lower()


@pytest.mark.parametrize("retained", ["record", "ready", "buffered", "demand"])
def test_soak_rejects_gpu_execution_ownership_after_cpu_resources_are_idle(retained):
    device = {"ready_slots": 0, "retained_slots": 0, "executions": [], "execution_resources": {"gpu": 0}}
    resources = {
        "transport": dict.fromkeys(
            (
                "usage_bytes",
                "active_input_leases",
                "active_input_ref_holds",
                "active_output_credits",
                "waiting_output_grants",
            ),
            0,
        ),
        "model_workers": [],
        "runtime": {"gpu": {"models": [{"pools": [{"devices": [device]}]}]}},
    }
    require_idle = soak()["require_idle_owners"]
    require_idle(resources)
    if retained == "record":
        device["executions"] = [{"state": "cleanup_pending"}]
    elif retained == "demand":
        device["execution_resources"]["gpu"] = 1
    else:
        device["ready_slots" if retained == "ready" else "retained_slots"] = 1
    with pytest.raises(AssertionError, match="idle GPU device retained"):
        require_idle(resources)

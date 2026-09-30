#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0
"""Repeat CPU or CUDA serving lifecycles under a POSIX process watchdog.

Run with an installed wheel: python -I scripts/validate_local_serving_soak.py
--output <new-directory>. See LOCAL_SERVING_ACCEPTANCE.md for report boundaries.
The supervisor uses only the standard library, independently of Vane's locks.
"""

from __future__ import annotations

import argparse
import faulthandler
import importlib.metadata
import json
import math
import os
import platform
import re
import runpy
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_options(rounds, requests, concurrency, timeout, gpu_device=None):
    if type(rounds) is not int or rounds < 2:
        raise ValueError("rounds must be >= 2")
    if type(requests) is not int or not 2 <= requests <= 10_000:
        raise ValueError("requests must be between 2 and 10000 per load phase")
    if type(concurrency) is not int or not 1 <= concurrency <= 4:
        raise ValueError("concurrency must be between 1 and 4")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if gpu_device is not None and (
        type(gpu_device) is not str
        or not re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", gpu_device, re.IGNORECASE)
    ):
        raise ValueError("gpu_device must be one full GPU UUID")


class Diagnostics:
    """Overwrite passive snapshots; a blocked sampler never owns the watchdog."""

    def __init__(self, directory, snapshot, stacks):
        self.directory = directory
        self.snapshot = snapshot
        self.stacks = stacks
        self.started = time.monotonic()
        self.progress = {"round": 0, "phase": "startup"}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, name="serving-soak-diagnostics", daemon=True)

    def phase(self, round_number, name):
        self.progress = {
            "round": round_number,
            "phase": name,
            "elapsed_seconds": time.monotonic() - self.started,
        }
        write_json(self.directory / "progress.json", self.progress)

    def _sample(self):
        while not self.stop.is_set():
            progress = self.progress
            started = time.monotonic() - self.started
            try:
                resources = self.snapshot()
                write_json(
                    self.directory / "resources.json",
                    {
                        "progress_at_sample_start": progress,
                        "sample_started_seconds": started,
                        "sample_finished_seconds": time.monotonic() - self.started,
                        **resources,
                    },
                )
            except Exception as error:
                # Keep the last successful sample and record diagnostics failure
                # separately. Never retain the exception/traceback in the driver.
                write_json(
                    self.directory / "diagnostics-error.json",
                    {"elapsed_seconds": started, "type": type(error).__name__, "message": str(error)[:2048]},
                )
            self.stop.wait(0.2)

    def failure(self):
        # Capture before query teardown; no resource lock is needed for stacks.
        faulthandler.dump_traceback(file=self.stacks, all_threads=True)
        write_json(
            self.directory / "failure.json",
            {"progress": self.progress, "traceback": traceback.format_exc()[-16_384:]},
        )

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=0.5)


def require_idle_owners(resources):
    """Check physical transport/pool ownership as well as runtime accounting."""
    transport = resources["transport"]
    for field in (
        "usage_bytes",
        "active_input_leases",
        "active_input_ref_holds",
        "active_output_credits",
        "waiting_output_grants",
    ):
        if transport[field] != 0:
            raise AssertionError(f"idle transport retained {field}={transport[field]}")
    for pool in resources["model_workers"]:
        admission = pool["admission"]
        if pool["active_workers"] or admission["active_leases"] or admission["ordinary_waiters"]:
            raise AssertionError("idle model pool retained work")
        if admission["available_slots"] != pool["pool_size"]:
            raise AssertionError("idle model pool did not return every slot")
    tasks = resources.get("task_workers")
    if tasks is not None and tasks["reserved_execution_slots"]:
        raise AssertionError("idle task executor retained thread capacity")
    for model in resources["runtime"].get("gpu", {}).get("models", []):
        for pool in model["pools"]:
            for device in pool["devices"]:
                if device["ready_slots"] or device["retained_slots"] or device["executions"]:
                    raise AssertionError("idle GPU device retained execution ownership")
                if any(device["execution_resources"].values()):
                    raise AssertionError("idle GPU device retained execution demand")


def run_soak(directory, *, rounds, requests, concurrency, stacks, gpu_device=None):
    """Reuse the public-API fixture without recreating its session each round."""
    root = Path(__file__).resolve().parents[1]
    acceptance = runpy.run_path(str(root / "scripts" / "validate_local_serving.py"))
    snapshot = runpy.run_path(str(root / "tests" / "local_runtime_helpers.py"))["local_runtime_snapshot"]
    require, distribution = acceptance["require"], acceptance["distribution"]
    scenario = None
    diagnostics = Diagnostics(directory, lambda: snapshot(scenario.runtime._runtime), stacks)
    history = deque(maxlen=8)
    total_load_requests = total_initializations = slot_refusals = 0
    execution_timeout = 30 if gpu_device is None else 5
    gpu_peaks = {"allocated_bytes": 0, "reserved_bytes": 0}
    primary_failed = False
    work = directory / "work"
    work.mkdir()
    original_directory = Path.cwd()
    os.chdir(work)
    try:
        diagnostics.phase(0, "model_registration")
        scenario = acceptance["Scenario"](work, gpu_device=gpu_device, execution_timeout=execution_timeout)
        diagnostics.thread.start()
        prewarm_seconds = None
        if gpu_device is not None:
            diagnostics.phase(0, "cuda_prewarm")
            started = time.monotonic()
            scenario.model.prewarm()
            prewarm_seconds = time.monotonic() - started
        diagnostics.phase(0, "cold_query")
        cold = scenario.query()
        require(len(scenario.initializations()) == 1, "cold phase did not initialize exactly one model")
        total_initializations = 1
        scenario.model.prewarm()
        require(len(scenario.initializations()) == 1, "prewarm reinitialized the healthy model")
        scenario.quiescent("cold_recovered")
        cold_gpu = scenario.gpu_checkpoint("cold_gpu")
        if cold_gpu is not None:
            for field in gpu_peaks:
                gpu_peaks[field] = cold_gpu["cuda"]["peak_" + field]
        total_load_requests = 1
        slot_refusals = cold["result_slot_refusals"]

        for round_number in range(1, rounds + 1):
            before = scenario.runtime.resource_snapshot()
            initializations = len(scenario.initializations())
            worker_pid = int(scenario.initializations()[-1])
            before_gpu = scenario.gpu_checkpoint("round_gpu_start")
            diagnostics.phase(round_number, "warm_queries")
            warm = [scenario.query(api=("sql", "relation")[i % 2]) for i in range(requests)]
            scenario.quiescent("warm_recovered")
            diagnostics.phase(round_number, "mixed_queries")
            with ThreadPoolExecutor(max_workers=concurrency) as clients:
                mixed = list(
                    clients.map(
                        lambda i: scenario.query(
                            "analysis" if i % 4 == 0 else "short",
                            32 if i % 4 == 0 else 1,
                            api=("sql", "relation")[(i + i // 4) % 2],
                        ),
                        range(requests),
                    )
                )
            scenario.quiescent("mixed_recovered")
            require(all(sample["worker_pids"] == [worker_pid] for sample in warm + mixed), "healthy worker changed")
            for name in ("ingress", "result_pressure"):
                diagnostics.phase(round_number, name)
                getattr(scenario, name)()
            require(len(scenario.initializations()) == initializations, "healthy work reinitialized the model")
            healthy_gpu = scenario.gpu_checkpoint("healthy_gpu")
            if before_gpu is not None:
                require(healthy_gpu["worker"] == before_gpu["worker"], "healthy GPU worker generation changed")
            diagnostics.phase(round_number, "cancellation")
            scenario.cancellation()
            cancellation_replacements = len(scenario.initializations()) - initializations
            # Local cancellation may retire its interrupted worker. Recovery
            # must initialize at most one, not churn across later healthy work.
            require(cancellation_replacements in (0, 1), "cancellation churned model workers")
            deadline_replacements = 0
            if gpu_device is not None:
                diagnostics.phase(round_number, "execution_deadline")
                before_deadline = len(scenario.initializations())
                scenario.execution_expiry()
                deadline_replacements = len(scenario.initializations()) - before_deadline
                require(deadline_replacements == 1, "execution deadline did not replace its GPU worker once")
            diagnostics.phase(round_number, "worker_failures")
            scenario.failures()
            replacements = len(scenario.initializations()) - initializations
            require(
                replacements == cancellation_replacements + deadline_replacements + 2,
                "unexpected fault recovery initialization",
            )
            total_initializations += replacements
            diagnostics.phase(round_number, "round_recovered")
            recovered = scenario.quiescent("round_recovered")
            owners = snapshot(scenario.runtime._runtime)
            write_json(directory / "idle-owners.json", owners)
            require_idle_owners(owners)
            recovered_gpu = scenario.gpu_checkpoint("round_gpu_recovered")
            if recovered_gpu is not None:
                require(
                    recovered_gpu["worker"]["generation"] == before_gpu["worker"]["generation"] + replacements,
                    "GPU worker generation differs from observed replacements",
                )
                for sample in (healthy_gpu, recovered_gpu):
                    for field in gpu_peaks:
                        gpu_peaks[field] = max(gpu_peaks[field], sample["cuda"]["peak_" + field])
            total_load_requests += len(warm) + len(mixed)
            slot_refusals += sum(sample["result_slot_refusals"] for sample in warm + mixed)
            history.append(
                {
                    "round": round_number,
                    "healthy_additional_initializations": 0,
                    "cancellation_replacements": cancellation_replacements,
                    "deadline_replacements": deadline_replacements,
                    "fault_replacements": dict(scenario.recovery_initializations),
                    "request_metrics": acceptance["request_metrics"](before, recovered),
                    "latency_seconds": {
                        "warm": distribution([sample["latency_seconds"] for sample in warm]),
                        "mixed_short": distribution(
                            [sample["latency_seconds"] for sample in mixed if sample["kind"] == "short"]
                        ),
                        "mixed_analysis": distribution(
                            [sample["latency_seconds"] for sample in mixed if sample["kind"] == "analysis"]
                        ),
                    },
                    "resources": recovered,
                    "transport": owners["transport"],
                    "gpu": recovered_gpu,
                }
            )
            write_json(directory / "rounds.json", list(history))
            # Keep fixture files and samples bounded across rounds. The actor
            # remains alive; only completed-request markers and old log lines go.
            for pattern in ("calls-*", "entered-*", "release-*"):
                for path in work.glob(pattern):
                    path.unlink()
            (work / "initializations").write_text(scenario.initializations()[-1] + "\n")

        diagnostics.phase(rounds, "drain_and_close")
        final_gpu = scenario.gpu_checkpoint("gpu_before_close")
        scenario.runtime.drain()
        with scenario.client() as (_, token, execute), acceptance["expect"](RuntimeError, "draining"):
            execute()
        require(not scenario.calls(token), "drained runtime executed a UDF")
        scenario.close()
        closed = scenario.quiescent("closed", closed=True)
        owners = snapshot(scenario.runtime._runtime)
        write_json(directory / "idle-owners.json", owners)
        require_idle_owners(owners)
        closed_gpu = scenario.gpu_checkpoint("gpu_closed", closed=True)
        if final_gpu is not None:
            try:
                os.kill(final_gpu["worker"]["pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise AssertionError("runtime close retained its CUDA worker process")
        diagnostics.phase(rounds, "completed")
        return {
            "schema_version": 1,
            "status": "passed",
            "environment": {
                "python": platform.python_version(),
                "platform": platform.system(),
                "machine": platform.machine(),
                "vane": importlib.metadata.version("vane-ai"),
            },
            "configuration": {
                "rounds": rounds,
                "requests_per_phase": requests,
                "concurrency": concurrency,
                "gpu_device": gpu_device,
                "execution_timeout": execution_timeout,
            },
            "completed_rounds": rounds,
            "runtime_sessions": 1,
            "load_requests": total_load_requests,
            "load_result_slot_refusals": slot_refusals,
            "total_initializations": total_initializations,
            "observed_worker_exit_failures": scenario.observed_worker_failures,
            "recent_rounds": list(history),
            "closed": closed,
            "closed_transport": owners["transport"],
            "gpu": None
            if gpu_device is None
            else {
                "cold": cold_gpu,
                "prewarm_seconds": prewarm_seconds,
                "first_query_seconds": cold["latency_seconds"],
                "sampled_memory_peaks": gpu_peaks,
                "closed": closed_gpu,
            },
            "elapsed_seconds": time.monotonic() - diagnostics.started,
            "scope": f"Synthetic {'CPU' if gpu_device is None else 'CUDA'} text/RGB public SQL/Relation lifecycle soak. "
            "Per-round latency samples, "
            "not global quantiles or an SLO; logical ownership checks, not a process RSS bound. "
            "CUDA allocator samples are observations, not a VRAM limit or an assertion that caching is a leak. "
            "Driver worker-exit counts describe injected faults; resource snapshots contain the separately "
            "verified runtime worker outcome counters.",
        }
    except BaseException:
        primary_failed = True
        try:
            diagnostics.failure()
        except Exception:
            pass
        raise
    finally:
        diagnostics.close()
        try:
            if scenario is not None:
                scenario.close()
        except BaseException:
            if not primary_failed:
                raise
            traceback.print_exc(file=sys.stderr)
        finally:
            os.chdir(original_directory)


def _signal_group(process, signum):
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def supervise(command, directory, *, timeout, diagnostic_grace=2.0, termination_grace=2.0):
    """Bound the complete child lifetime, including failed or blocked cleanup."""
    started = time.monotonic()
    status = "failed"
    with (directory / "worker.log").open("wb") as log:
        process = subprocess.Popen(command, cwd=directory, stdout=log, stderr=log, start_new_session=True)
        try:
            try:
                returncode = process.wait(timeout=timeout)
                status = "completed" if returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                status = "timed_out"
                # faulthandler handles this without the GIL. Send only after
                # the child has installed its handler; never a Python snapshot
                # callback that might block on the very lock being diagnosed.
                if (directory / "watchdog-ready").exists():
                    try:
                        process.send_signal(signal.SIGUSR1)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=diagnostic_grace)
                    except subprocess.TimeoutExpired:
                        pass
        finally:
            # All fixture actors inherit this newly created process group.
            # Also clean descendants if the leader exits before its workers.
            _signal_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=termination_grace)
            except subprocess.TimeoutExpired:
                pass
            finally:
                _signal_group(process, signal.SIGKILL)
                process.wait(timeout=termination_grace)
    worker_report = None
    if status == "completed":
        try:
            worker_report = json.loads((directory / "worker-report.json").read_text())
            status = "passed" if worker_report["status"] == "passed" else "failed"
        except (OSError, ValueError, KeyError, TypeError):
            status = "failed"
    report = {
        "schema_version": 1,
        "status": status,
        "child_returncode": process.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "timeout_seconds": timeout,
        "worker_report": worker_report,
        "diagnostics": {
            name: (directory / name).exists()
            for name in (
                "progress.json",
                "resources.json",
                "idle-owners.json",
                "rounds.json",
                "threads.log",
                "failure.json",
                "worker.log",
            )
        },
    }
    write_json(directory / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory for reports and diagnostics")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600, help="whole-run watchdog limit, including cleanup")
    parser.add_argument("--gpu-device", help="full provisioned GPU UUID; enables real CUDA with application PyTorch")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        validate_options(args.rounds, args.requests, args.concurrency, args.timeout, args.gpu_device)
    except ValueError as error:
        parser.error(str(error))
    if os.name != "posix":
        parser.error("the soak watchdog requires POSIX process groups and SIGUSR1")
    if args.gpu_device is not None:
        args.gpu_device = "GPU-" + args.gpu_device[4:].lower()
    directory = args.output.resolve()
    if args.worker:
        os.environ["VANE_RUNNER"] = "local-fast"
        # Process-owned descriptor: keep both it and the handlers alive after
        # main returns, through thread joins, atexit hooks and finalizers. The OS
        # closes it at exit; a file object's destruction could close it too soon.
        stacks = os.open(directory / "threads.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        faulthandler.enable(file=stacks, all_threads=True)
        faulthandler.register(signal.SIGUSR1, file=stacks, all_threads=True)
        (directory / "watchdog-ready").touch()
        report = run_soak(
            directory,
            rounds=args.rounds,
            requests=args.requests,
            concurrency=args.concurrency,
            stacks=stacks,
            gpu_device=args.gpu_device,
        )
        write_json(directory / "worker-report.json", report)
        return 0
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("output directory already exists; choose a new directory to avoid stale evidence")
    command = [
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        "--worker",
        "--output",
        str(directory),
        "--rounds",
        str(args.rounds),
        "--requests",
        str(args.requests),
        "--concurrency",
        str(args.concurrency),
        "--timeout",
        str(args.timeout),
    ]
    if args.gpu_device is not None:
        command.extend(["--gpu-device", args.gpu_device])
    report = supervise(command, directory, timeout=args.timeout)
    print(f"{'CPU' if args.gpu_device is None else 'CUDA'} serving soak {report['status']}; evidence: {directory}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

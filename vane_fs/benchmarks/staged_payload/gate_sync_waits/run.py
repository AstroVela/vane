# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run six frozen AB/BA diagnostic pairs per state with the reduced observer."""

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from checks import owned_state
from prepare import digest, save, validate


def main():
    root, here = Path(sys.argv[1]).resolve(), Path(__file__).resolve().parent
    out = root / "native"
    build = json.loads((out / "build.json").read_text())
    assert build["status"] == "PASS"
    assert json.loads((root / "gate-trace-tests.json").read_text())["status"] == "PASS"
    assert json.loads((out / "observer-tests.json").read_text())["status"] == "PASS"
    preflight = owned_state(root)
    assert not preflight["active_owned_pids"] and not preflight["owned_mounts"]
    save(root / "preflight.json", dict(status="PASS", recorded_epoch=time.time(), **preflight))
    prior = here.parents[2] / "build/sqlite-step-waits-20261019"
    source = code = (prior / "run_fuse.py").read_text()
    changes = [
        (str(here.parent / "sqlite_step_waits/sample.py"), str(here / "sample.py")),
        (
            'm.ORDERS = m.FIRST = [["baseline", "updated"], ["updated", "baseline"]] * 2',
            'm.ORDERS = m.FIRST = [["baseline", "updated"], ["updated", "baseline"]] * 3',
        ),
        (
            "Range writeback phase/wait attribution; all original 9 ordinary FUSE failures remain",
            "Matched paced parent and BEGIN/RESTART gate: native gate/sync attribution only; all ordinary failures remain",
        ),
    ]
    for before, after in changes:
        assert code.count(before) == 1, before
        code = code.replace(before, after)
    reverse = code
    for before, after in reversed(changes):
        reverse = reverse.replace(after, before)
    assert reverse == source
    compile(code, str(out / "run_fuse.py"), "exec")
    (out / "run_fuse.py").write_text(code)
    (out / "workload.py").write_bytes((prior / "workload.py").read_bytes())
    (out / "workload-proof.json").write_bytes((prior / "workload-proof.json").read_bytes())
    setup = json.loads((root / "setup.json").read_text())
    proof = json.loads((out / "workload-proof.json").read_text())
    assert digest(Path(setup["comparison"]) / "full/workload.py") == proof["original_sha256"]
    save(
        out / "tests.json",
        dict(
            status="PASS",
            new_gate_trace_contracts=str(root / "gate-trace-tests.json"),
            sql_observer_contracts=str(out / "observer-tests.json"),
            original_correctness_reused_by_source_reversal=True,
        ),
    )
    inputs = build["inputs"] | build["sources"] | build["binaries"]
    for p in [
        *here.glob("*.py"),
        *here.glob("*.cpp"),
        prior / "run_fuse.py",
        prior / "workload.py",
        prior / "workload-proof.json",
        out / "build.json",
        root / "gate-trace-tests.json",
        out / "observer-tests.json",
        out / "run_fuse.py",
        out / "workload.py",
        out / "workload-proof.json",
        out / "tests.json",
        root / "preflight.json",
    ]:
        inputs[str(p)] = digest(p)
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    modules = {}
    for variant in ("baseline", "updated"):
        module = next((out / "diagnostic" / variant).glob("venv/lib/python3.12/site-packages/vane_fs/_native*.so"))
        modules[variant] = dict(path=str(module), sha256=digest(module))
    save(
        out / "plan.json",
        dict(
            status="FROZEN_BEFORE_DIAGNOSTIC_WINDOW",
            frozen_epoch=time.time(),
            mounts=24,
            orders=[["baseline", "updated"], ["updated", "baseline"]] * 3,
            states=["fresh", "post_gc"],
            medium="ext4",
            kernel=platform.release(),
            inputs=inputs,
            variant_gc_python_modules=modules,
            role="Native-stage attribution only; no performance adoption",
            native_logic_unchanged=True,
            runner_reverse_exact=True,
            runner_changes=changes,
            workload_bytes_unchanged=True,
            active_minimum_age_ns=20000000,
            kernel_observation_minimum_interval_ns=50000000,
            sampler_interval_seconds=0.02,
            thread_frequency_census=False,
            mapping_read_only=True,
            global_cache_eviction=False,
            cpu_device_tuning=False,
            owned_build_or_test_overlap=False,
        ),
    )
    argv = [sys.executable, "-u", str(out / "run_fuse.py"), str(out), "--medium", "ext4", "--diagnostic"]
    started = time.time()
    with (out / "measurement.log").open("w") as log:
        result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
    save(
        out / "measurement-command.json",
        dict(argv=argv, started_epoch=started, ended_epoch=time.time(), returncode=result.returncode),
    )
    assert result.returncode == 0
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    raw = json.loads((out / "ext4-diagnostic/results.json").read_text())
    assert raw["status"] == "PASS" and raw["monitor_stopped"]
    checks = []
    for state in ("fuse", "aged_fuse"):
        assert len(raw[state]) == 12
        for row in raw[state]:
            stats = validate(json.loads(Path(row["waits_file"]).read_text()))
            samples = [json.loads(line) for line in Path(row["cpu_samples_file"]).read_text().splitlines()]
            final = samples[-1]
            assert final["sampler_finished"] and final["stop_requested"] and final["fatal"] is None
            assert final["marker"] is not None and final["mapping_closed"] and final["mapping_read_only"]
            assert all(not s["threads"] and not s["frequency_khz"] for s in samples)
            checks.append(dict(state=state, variant=row["variant"], rep=row["rep"], **stats))
    save(out / "input-check.json", dict(status="PASS", checked_inputs=len(inputs), mounts=checks))
    print("PASS 24 full diagnostic mounts; native validation and reduced observer contracts", flush=True)


if __name__ == "__main__":
    main()

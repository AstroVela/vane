# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Verify unchanged barriers and run the frozen per-sync attribution analyzer."""

import argparse
import collections
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def body(source, name, prefix="\tvoid "):
    start = source.index(prefix + name + "(")
    begin = source.index("{", start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def restart_costs(out):
    """Count actual restart outcomes separately from overlapping phase spans."""
    raw_path = out / "ext4-diagnostic/results.json"
    raw = json.loads(raw_path.read_text())
    result = dict(
        status="PASS",
        scope="Instrumented costs only; whole worker spans can overlap multiple phases and are not additive",
        inputs={str(raw_path): digest(raw_path), str(Path(__file__).resolve()): digest(Path(__file__))},
        mounts=[],
        groups=[],
        restart_syncs_by_outcome=[],
    )
    phases, calls = [], []
    for state, key in (("fresh", "fuse"), ("post_gc", "aged_fuse")):
        for mount in raw[key]:
            path = Path(mount["request_costs_file"])
            result["inputs"][str(path)] = digest(path)
            trace = json.loads(path.read_text())
            stage = trace["stages"].index("restart")
            restarts = []
            for thread in trace["threads"]:
                rows = [dict(zip(trace["columns"], values, strict=True)) for values in thread["rows"]]
                for row in rows:
                    if not row["counts"][stage]:
                        continue
                    assert mount["variant"] == "updated"
                    assert row["name"] == "checkpoint_worker" and row["counts"][stage] == 1
                    assert row["last_checkpoint"][0] == 2 and row["last_checkpoint"][3] in (0, 5)
                    restarts.append(row)
                for values in thread["syncs"]:
                    call = dict(zip(trace["sync_columns"], values, strict=True))
                    if call["site"] != "wal_restart":
                        continue
                    assert call["request_index"] >= 0
                    parent = rows[call["request_index"]]
                    assert parent["name"] == "checkpoint_worker" and parent["counts"][stage] == 1
                    assert call["result"] == call["errno"] == 0
                    calls.append(dict(**call, outcome=parent["last_checkpoint"][3]))
            identity = dict(state=state, variant=mount["variant"], rep=mount["rep"])
            outcomes = collections.Counter(row["last_checkpoint"][3] for row in restarts)
            result["mounts"].append(dict(**identity, attempts=len(restarts), success=outcomes[0], busy=outcomes[5]))
            for phase in mount["samples"]:
                begin = round(phase["started_epoch"] * 10**9) - trace["origin_epoch_ns"]
                end = round(phase["ended_epoch"] * 10**9) - trace["origin_epoch_ns"]
                selected = [row for row in restarts if row["begin_ns"] < end and row["end_ns"] > begin]
                outcomes = collections.Counter(row["last_checkpoint"][3] for row in selected)
                phases.append(
                    dict(
                        **identity,
                        workload=phase["workload"],
                        attempts=len(selected),
                        success=outcomes[0],
                        busy=outcomes[5],
                        restart_ns=sum(row["stage_ns"][stage] for row in selected),
                    )
                )
    for state, variant, workload in sorted({(p["state"], p["variant"], p["workload"]) for p in phases}):
        selected = [p for p in phases if (p["state"], p["variant"], p["workload"]) == (state, variant, workload)]
        result["groups"].append(
            dict(
                state=state,
                variant=variant,
                workload=workload,
                phases=len(selected),
                **{
                    name: statistics.median(p[name] for p in selected)
                    for name in ("attempts", "success", "busy", "restart_ns")
                },
            )
        )
    result["whole_mount_outcomes"] = {
        name: sum(m[name] for m in result["mounts"]) for name in ("attempts", "success", "busy")
    }
    for outcome, kind in sorted({(c["outcome"], c["kind"]) for c in calls}):
        selected = [c for c in calls if (c["outcome"], c["kind"]) == (outcome, kind)]
        result["restart_syncs_by_outcome"].append(
            dict(
                sqlite_result=outcome,
                kind=kind,
                calls=len(selected),
                inclusive_duration_ns=sum(c["end_ns"] - c["begin_ns"] for c in selected),
                max_ns=max(c["end_ns"] - c["begin_ns"] for c in selected),
            )
        )
    (out / "restart-costs.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print("restart outcome analysis PASS", result["whole_mount_outcomes"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    build = json.loads((out / "build.json").read_text())
    parent = Path(build["parent"])
    native = json.loads((parent / "build.json").read_text())
    reference = Path(native["orphan_parent"]) / "updated"
    updated = parent / "updated"
    before = (reference / "src/staging.hpp").read_text()
    after = (updated / "src/staging.hpp").read_text()
    names = ("BeforeFullCommit", "AfterFullCommit", "Publish", "Recover")
    assert all(body(before, name) == body(after, name) for name in names)
    assert (reference / "src/payload_file.hpp").read_bytes() == (updated / "src/payload_file.hpp").read_bytes()
    assert (reference / "tests").is_dir()
    checkpoint_before = (reference / "src/checkpoint.hpp").read_text()
    checkpoint_after = (updated / "src/checkpoint.hpp").read_text()
    assert body(checkpoint_before, "OnCommit", "\tstatic int ") == body(checkpoint_after, "OnCommit", "\tstatic int ")
    assert body(checkpoint_before, "BeforeWrite") == body(checkpoint_after, "BeforeWrite")
    assert [line for line in checkpoint_before.splitlines() if "static constexpr" in line] == [
        line for line in checkpoint_after.splitlines() if "static constexpr" in line
    ]
    assert json.loads((parent / "restart-tests.json").read_text())["status"] == "PASS"
    assert json.loads((out / "sync-tests.json").read_text())["status"] == "PASS"
    assert {
        p for p, h in native["source_inputs"]["orphan-control"].items() if h != native["source_inputs"]["updated"][p]
    } == {"src/checkpoint.hpp"}
    assert all(
        digest(Path(record["path"]) / p) == h
        for record in build["original_sources"].values()
        for p, h in record["sha256"].items()
    )
    assert all(
        digest(out / "diagnostic" / n / p) == h
        for n, entries in build["source_inputs"].items()
        for p, h in entries.items()
    )
    assert all(digest(out / p) == h for p, h in build["binaries"].items())
    proof = dict(
        status="PASS",
        runtime_changed_files=["src/checkpoint.hpp"],
        barrier_and_recovery_bodies_byte_identical=list(names),
        payload_layout_and_reclaim_byte_identical=True,
        component_tests_unchanged=True,
        optional_restart_lock_timeout_ms=0,
        original_wal_thresholds_and_callback_unchanged=True,
        wal_commit_callback_and_write_admission_byte_identical=True,
        typed_sync_recorder_tests_verified=True,
        original_and_instrumented_source_hashes_verified=True,
        inputs={
            str(reference / "src/staging.hpp"): digest(reference / "src/staging.hpp"),
            str(updated / "src/staging.hpp"): digest(updated / "src/staging.hpp"),
            str(reference / "src/checkpoint.hpp"): digest(reference / "src/checkpoint.hpp"),
            str(updated / "src/checkpoint.hpp"): digest(updated / "src/checkpoint.hpp"),
        },
    )
    (out / "source-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    generator = out / "analysis-generator"
    generator.mkdir(exist_ok=True)
    old = here.parent / "sync_costs/analyze.py"
    source = old.read_text()
    begin, end = source.index("def source_proof("), source.index("def main():")
    source = source[:begin] + source[end:]
    source = source.replace('    parser.add_argument("--prior-diagnostic", type=Path, required=True)\n', "")
    marker = '    write(out / "source-proof.json", source_proof(out, args.prior_diagnostic.resolve()))'
    assert source.count(marker) == 1
    source = source.replace(marker, '    assert read(out / "source-proof.json")["status"] == "PASS"')
    source = source.replace("import re\n", "")
    (generator / "analyze.py").write_text(source)
    identity = dict(
        status="PASS",
        inputs={str(old): digest(old), str(here / "analyze_restart.py"): digest(here / "analyze_restart.py")},
        generated_sha256=digest(generator / "analyze.py"),
        adaptation="Use the native barrier/layout proof and verify restart plus typed recorder tests; sync decoding, reconciliation, grouping and slow-call retention unchanged",
    )
    (generator / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    subprocess.run([sys.executable, str(generator / "analyze.py"), str(out)], check=True)
    restart_costs(out)


if __name__ == "__main__":
    main()

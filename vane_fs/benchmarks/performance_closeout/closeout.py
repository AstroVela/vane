# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Audit the bounded closeout and retain every attempt without adopting a rejected format."""

import argparse
import gzip
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import experiment
from experiment import HERE, REPO, STAGED, digest, save


def read(path):
    return json.loads(path.read_text())


def audit(out, candidate, shipping, initial):
    out.mkdir(exist_ok=True)
    source, build, analysis, decision = [
        read(candidate / p) for p in ("source.json", "build.json", "fuse/analysis.json", "decision.json")
    ]
    default, tests = [read(shipping / p) for p in ("shipping-results.json", "default-tests.json")]
    inventory, scope = [read(HERE / p) for p in ("inventory.json", "checkpoint-scope.json")]
    assert all(r["status"] == "PASS" for r in (source, build, analysis, decision, default, tests, inventory, scope))
    assert decision["decision"] == "REJECT" and len(decision["failed_checks"]) == 4
    assert default["mounts"] == 10 and default["sequential_content_checks"] == 44
    assert (
        read(candidate / "full-pauses.json")["status"] == read(candidate / "negative-control.json")["status"] == "PASS"
    )
    identities = (
        source["inputs"]
        | source["sources"]
        | build["inputs"]
        | build["binaries"]
        | analysis["inputs"]
        | default["inputs"]
        | scope["inputs"]
    )
    assert all(digest(Path(p)) == sha for p, sha in identities.items())
    defaults = inventory["default"]["identities"]
    assert all(digest(Path(p)) == sha for p, sha in defaults.items())
    cleanup = []
    for root, names in (
        (
            candidate,
            (
                "build.json",
                "full-pauses.json",
                "initial-full-pauses.json",
                "negative-control.json",
                "fixture-probe.json",
            ),
        ),
        (initial, ("initial-default-tests.json", "default-tests.json")),
    ):
        for name in names:
            cleanup.extend(read(root / name)["cleanup"])
    raw_paths = [candidate / "fuse" / w / "ext4-plain/results.json" for w in ("full", "sustained")]
    raw_paths += [shipping / "fuse" / w / "ext4-plain/results.json" for w in ("full", "sustained")]
    prior = read(initial / "fuse/full/ext4-plain/results.json")
    assert prior["status"] == "FAIL" and prior["monitor_stopped"] and len(prior["fuse"]) == 1
    for path in raw_paths:
        raw = read(path)
        assert raw["status"] == "PASS" and raw["monitor_stopped"]
        cleanup.extend(raw["cleanup"])
    cleanup.extend(prior["cleanup"])
    unique = {r["path"]: r for r in cleanup if r.get("created_by_run", True)}
    assert all(
        r["removed"] and r.get("processes_stopped", r.get("process_stopped")) and not Path(r["path"]).exists()
        for r in unique.values()
    )
    sys.modules["prepare"] = experiment
    path = STAGED / "gate_sync_waits/checks.py"
    spec = importlib.util.spec_from_file_location("closeout_ownership", path)
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    states = {str(p): checks.owned_state(p) for p in (candidate, shipping, initial, out)}
    assert all(not s["active_owned_pids"] and not s["owned_mounts"] for s in states.values())
    full = read(shipping / "fuse/full/ext4-plain/results.json")
    all_observed = {}
    for state in ("fuse", "aged_fuse"):
        rows = full[state] + prior[state]
        workloads = {}
        for name in sorted({p["workload"] for r in rows for p in r["samples"]}):
            phases = [p for r in rows for p in r["samples"] if p["workload"] == name]
            key = "mib_per_second" if "mib_per_second" in phases[0] else "ops_per_second"
            rates = sorted(p[key] for p in phases)
            n = len(rates)
            workloads[name] = dict(
                median_rate=(rates[(n - 1) // 2] + rates[n // 2]) / 2,
                aggregate_rate=sum(p[key] * p["seconds"] for p in phases) / sum(p["seconds"] for p in phases),
                count=n,
                all_rates=[p[key] for p in phases],
            )
        all_observed[state] = dict(
            total_mounts=len(rows),
            post_shutdown_validated_mounts=len(full[state]),
            prior_incomplete_post_shutdown_validation=len(prior[state]),
            workloads=workloads,
        )
    for p in (
        Path(__file__),
        path,
        HERE / "inventory.json",
        HERE / "checkpoint-scope.json",
        REPO / "vane_fs/PERFORMANCE_ROADMAP.md",
        candidate / "decision.json",
        shipping / "shipping-results.json",
        initial / "fuse/full/ext4-plain/results.json",
    ):
        identities[str(p)] = digest(p)
    save(
        out / "closeout.json",
        dict(
            status="PASS",
            recorded_epoch=time.time(),
            decision="CLOSED_DEFAULT_RETAINED",
            milestones=dict(
                S1="COMPLETE", S2="COMPLETE_C1_REJECTED_C2_UNUSED", S3="COMPLETE_OUTSIDE_DEFAULT_SCOPE", S4="COMPLETE"
            ),
            candidate_budget=dict(limit=3, consumed=1, unused_closed=2),
            candidate_decision=decision,
            default_native_unchanged=defaults,
            qualified_mounts=42,
            sequential_content_checks=156,
            retained_initial_default_mounts=1,
            retained_initial_large_file_hashes=2,
            total_observed_mounts=43,
            total_observed_large_file_hashes=158,
            default_all_observed_full=all_observed,
            cleanup_directories=len(unique),
            cleanup=list(unique.values()),
            owned_state=states,
            free_bytes=shutil.disk_usage(out).free,
            inputs=identities,
            artifacts=dict(candidate=str(candidate), shipping=str(shipping), initial_shipping=str(initial)),
            limits=[
                "Default remains SQLite BLOB; no format-1005 migration or adoption",
                "All first-attempt timings are retained and included in all-observed default statistics",
                "No further experiment is authorized by unused budget after this closeout",
            ],
        ),
    )
    print(
        "PASS bounded closeout; 1/3 candidates used, 2 closed unused;",
        len(unique),
        "owned directories cleaned",
        flush=True,
    )


def publish(out):
    result = read(out / "closeout.json")
    assert result["status"] == "PASS" and all(digest(Path(p)) == sha for p, sha in result["inputs"].items())
    files = {}
    roots = {**{k: Path(v) for k, v in result["artifacts"].items()}, "final": out}
    for label, root in roots.items():
        for p in root.rglob("*"):
            parts = p.relative_to(root).parts
            if (
                p.is_file()
                and p.suffix in (".json", ".jsonl", ".py", ".cpp", ".log", ".txt")
                and not {"candidate", "test-env", "__pycache__"}.intersection(parts)
                and not p.name.startswith(("pr967-", "completion"))
            ):
                files[label + "/" + str(p.relative_to(root))] = p
    source = read(roots["candidate"] / "source.json")
    for label, root in (("control", Path(source["original"])), ("C1", Path(source["candidate"]))):
        for name in read(Path(read(HERE / "inventory.json")["mechanism_control"]["component"]) / "source.json")[
            "sources"
        ]:
            relative = Path(name).relative_to(Path(source["original"]))
            files["native-sources/" + label + "/" + str(relative)] = root / relative
    for path in [
        *(REPO / "vane_fs/src").glob("*"),
        *(REPO / "vane_fs/include").rglob("*.hpp"),
        REPO / "vane_fs/CMakeLists.txt",
    ]:
        if path.is_file():
            files["native-sources/default/" + str(path.relative_to(REPO / "vane_fs"))] = path
    audit = dict(
        status="PASS",
        published_epoch=time.time(),
        decision=result["decision"],
        candidate_budget=result["candidate_budget"],
        original_files={},
        retained_files={},
    )
    for relative, path in sorted(files.items()):
        data = path.read_bytes()
        packed, parts = gzip.compress(data, compresslevel=6, mtime=0), []
        for start in range(0, len(packed), 450000):
            target = HERE / "evidence" / relative / f"{len(parts):03}.gz.part"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(packed[start : start + 450000])
            parts.append(str(target.relative_to(HERE)))
        assert gzip.decompress(b"".join((HERE / p).read_bytes() for p in parts)) == data
        audit["original_files"][str(path)] = dict(
            bytes=len(data), sha256=digest(path), retained_as=parts, bytes_unchanged=True
        )
    for name, path in (
        ("closeout.json", out / "closeout.json"),
        ("C1-decision.json", roots["candidate"] / "decision.json"),
        ("shipping-results.json", roots["shipping"] / "shipping-results.json"),
    ):
        (HERE / name).write_text(json.dumps(read(path), separators=(",", ":")) + "\n")
    for p in HERE.rglob("*"):
        if p.is_file() and p.name != "audit.json" and "__pycache__" not in p.parts:
            assert p.stat().st_size < 500 * 1024, p
            audit["retained_files"][str(p.relative_to(HERE))] = dict(bytes=p.stat().st_size, sha256=digest(p))
    (HERE / "audit.json").write_text(json.dumps(audit, separators=(",", ":")) + "\n")
    assert (HERE / "audit.json").stat().st_size < 500 * 1024
    print("PASS lossless closeout evidence;", len(files), "original streams", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "publish"))
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--shipping", type=Path)
    parser.add_argument("--initial-shipping", type=Path)
    args = parser.parse_args()
    if args.action == "audit":
        audit(
            args.artifacts.resolve(), args.candidate.resolve(), args.shipping.resolve(), args.initial_shipping.resolve()
        )
    else:
        publish(args.artifacts.resolve())

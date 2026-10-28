# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Publish lossless small reports and verify measured identities and cleanup."""

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    audit = dict(
        status="PASS",
        diagnostic_mounts=0,
        windows=[],
        cleanup=[],
        retained_files={},
        original_gate_decision="REJECT",
        production_runtime_changed=False,
        measured_wrapper_ast_unchanged_after_formatting=True,
    )
    retained = set()

    def retain(path, expected=None):
        path = path.resolve()
        assert path.is_file(), path
        actual = digest(path)
        assert expected is None or expected == actual, path
        retained.add(path)

    def write(relative, value):
        path = here / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, separators=(",", ":")) + "\n")
        assert path.stat().st_size < 500 * 1024, path

    def copy(path, relative):
        retain(path)
        write(relative, read(path))

    for label, directory in (("original", out), ("causal", out / "causal")):
        build = read(directory / "build.json")
        plan = read(directory / "plan.json")
        proof = read(directory / "source-proof.json")
        assert build["status"] == proof["status"] == read(directory / "input-check.json")["status"] == "PASS"
        for entry in proof["retained_execution_inputs"]:
            retain(Path(entry["retained_path"]), entry["sha256"])
        for name, record in build["original_sources"].items():
            for path, expected in record["sha256"].items():
                retain(Path(record["path"]) / path, expected)
            for path, expected in build["source_inputs"][name].items():
                retain(directory / "diagnostic" / name / path, expected)
        for path, expected in build["binaries"].items():
            retain(directory / path, expected)
        for path, expected in plan["inputs"].items():
            retain(Path(path), expected)
        retain(directory / "run_fuse.py", plan["prepared_runner_sha256"])
        retain(directory / "workload.py", plan["prepared_workload_sha256"])
        for filename in (
            "build.json",
            "tests.json",
            "sync-tests.json",
            "plan.json",
            "input-check.json",
            "source-proof.json",
            "request-costs-summary.json",
            "complete-phases.json",
            "sync-analysis.json",
            "cpu-summary.json",
            "restart-costs.json",
            "slow-syncs.json",
        ):
            copy(directory / filename, label + "/" + filename)
        for filename in ("tests.json", "sync-tests.json"):
            assert read(directory / filename)["status"] == "PASS"
        for path, expected in read(directory / "sync-tests.json")["inputs"].items():
            retain(Path(path), expected)
        if label == "original":
            for filename in ("release-test-trace.json", "sanitize-test-trace.json"):
                copy(directory / filename, label + "/" + filename)
        for path in (directory / "analysis-generator").iterdir():
            if path.is_file():
                retain(path)
        identity = read(directory / "analysis-generator/identity.json")
        for path, expected in identity["inputs"].items():
            retain(Path(path), expected)
        for path, expected in identity["generated"].items():
            retain(directory / "analysis-generator" / path, expected)
        copy(directory / "analysis-generator/identity.json", label + "/analysis-identity.json")
        raw_path = directory / "ext4-diagnostic/results.json"
        raw = read(raw_path)
        retain(raw_path)
        assert raw["status"] == "PASS" and raw["monitor_stopped"]
        assert raw["config"]["instrumented"] and raw["config"]["durability"] == "fsync"
        assert not raw["config"]["global_drop_caches"] and raw["config"]["format_version"] == 1004
        assert raw["config"]["orders"] == [["baseline", "updated"], ["updated", "baseline"]] * 2
        audit["windows"].append(dict(label=label, started_epoch=raw["started_epoch"], ended_epoch=raw["ended_epoch"]))
        for state, key in (("fresh", "fuse"), ("post_gc", "aged_fuse")):
            rebuilt = []
            for variant in ("baseline", "updated"):
                rows = [m for m in raw[key] if m["variant"] == variant]
                assert len(rows) == 4
                relative = f"{label}/mounts/{state}-{variant}.json"
                write(relative, rows)
                rebuilt.extend(read(here / relative))
            assert sorted(rebuilt, key=lambda m: (m["rep"], m["variant"])) == sorted(
                raw[key], key=lambda m: (m["rep"], m["variant"])
            )
            for mount in raw[key]:
                audit["diagnostic_mounts"] += 1
                assert mount["inspection"]["quick_check"] == [["ok"]]
                assert mount["inspection"]["format"] == [[1004, 4096]]
                assert mount["inspection"]["block_high_index_present"]
                for path in ("request_costs_file", "cpu_samples_file"):
                    retain(Path(mount[path]))
                for sample in mount["samples"]:
                    if sample["workload"] in ("stream_256", "write_64", "random_write_4k", "random_read_4k"):
                        assert sample["validation"]
        for item in raw["cleanup"]:
            assert item["removed"] and not Path(item["path"]).exists()
            assert item["mount_absent"] and item["process_stopped"]
            assert not Path(f"/proc/{item['pid']}").exists()
            audit["cleanup"].append(item)
        write(label + "/window.json", {k: v for k, v in raw.items() if k not in ("fuse", "aged_fuse")})
        phases = read(directory / "sync-phases.json")
        retain(directory / "sync-phases.json")
        rebuilt = []
        for state in ("fresh", "post_gc"):
            for variant in ("baseline", "updated"):
                relative = f"{label}/sync-phases/{state}-{variant}.json"
                write(relative, [p for p in phases if (p["state"], p["variant"]) == (state, variant)])
                rebuilt.extend(read(here / relative))
        assert sorted(rebuilt, key=lambda p: (p["state"], p["variant"], p["rep"], p["workload"])) == sorted(
            phases, key=lambda p: (p["state"], p["variant"], p["rep"], p["workload"])
        )
        for path in list(directory.glob("*.log")) + list((directory / "ext4-diagnostic").glob("*.log")):
            retain(path)
            if path.stat().st_size:
                target = here / label / "logs" / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                assert digest(target) == digest(path)
        retain(directory / "ext4-diagnostic/host.jsonl")
    assert audit["diagnostic_mounts"] == 32 and len(audit["cleanup"]) == 32
    assert audit["windows"][0]["ended_epoch"] <= audit["windows"][1]["started_epoch"]
    for run in read(out / "build.json")["commands"] + read(out / "sync-tests.json")["runs"]:
        assert run["ended_epoch"] <= audit["windows"][0]["started_epoch"]
    before = out / "preflight/run_causal.py.txt"
    assert ast.dump(ast.parse(before.read_text())) == ast.dump(ast.parse((here / "run_causal.py").read_text()))
    for path in (out / "preflight").iterdir():
        retain(path)
        target = here / "preflight" / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
    copy(out / "findings.json", "findings.json")
    for path, expected in read(out / "findings.json")["inputs"].items():
        retain(Path(path), expected)
    defaults = here.parent / "writeback_budget/source-identity.json"
    frozen = read(defaults)
    for path, expected in frozen["production"].items():
        retain(here.parents[2] / path, expected)
    retain(Path(frozen["installed_component"]["path"]), frozen["installed_component"]["sha256"])
    retain(defaults)
    retain(Path(__file__).resolve())
    audit["production_default_identity_verified"] = True
    screen_path = Path(read(out / "build.json")["parent"]) / "summary.json"
    screen = read(screen_path)
    assert not screen["pass_all"]
    assert sum(not c["pass_gate"] for c in screen["checks"]) == 6
    retain(screen_path)
    audit["original_plain_screen_failed_checks"] = 6
    for path in sorted(retained):
        audit["retained_files"][str(path)] = dict(bytes=path.stat().st_size, sha256=digest(path))
    audit["identity_count"] = len(retained)
    write("audit.json", audit)
    print(
        "diagnostic publication PASS",
        audit["diagnostic_mounts"],
        "mounts",
        len(retained),
        "retained identities",
        flush=True,
    )


if __name__ == "__main__":
    main()

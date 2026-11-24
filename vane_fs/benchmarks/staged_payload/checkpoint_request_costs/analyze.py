# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reconcile syncs and compare complete FUSE phases without an adoption claim."""

import argparse
import collections
import hashlib
import importlib.util
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

TARGETS = ("random_write_4k", "small_file_fsync", "rename")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n")


def original_text(text):
    """Undo only the frozen generator's diagnostic insertions, exactly."""
    for label, control in (("Begin", 'write ? "BEGIN IMMEDIATE" : "BEGIN"'), ("Commit", '"COMMIT"')):
        needle = f"Control(database.db, {control});"
        marked = "{\n\t\trequest_trace::Span trace(request_trace::Stage::" + label
        marked += ");\n\t\t\t" + needle + "\n\t\t}"
        text = text.replace(marked, needle)
    text = text.replace(
        "\t\t{\n\t\t\trequest_trace::Span trace(request_trace::Stage::DatabaseWait);\n\t\t\tlock.lock();\n\t\t}",
        "\t\tlock.lock();",
    )
    needle = "payload_guard = std::make_unique<PayloadFile::Guard>(*database.payload_file, exclusive_payload);"
    text = text.replace(
        "{\n\t\t\trequest_trace::Span trace(request_trace::Stage::PayloadGuard);\n\t\t\t" + needle + "\n\t\t}",
        needle,
    )
    text = text.replace(
        "auto guard = request_trace::Lock(request_trace::Stage::PinsWait, registry.mutex);",
        "std::lock_guard<std::mutex> guard(registry.mutex);",
    )
    text = text.replace(
        "thread(request_trace::Lock(request_trace::Stage::PublicationWait, stage.mutex))", "thread(stage.mutex)"
    )
    text = text.replace('#include "request_trace.hpp"\n', "")
    text = re.sub(r"\n[\t ]*request_trace::(?:Span|SyncSite|Scope|Steps|CommitFrames)[^\n]*;", "", text)
    text = text.replace("\n\tif (argc > 1) request_trace::Install(argv[1]);", "")
    assert "request_trace::" not in text
    return text


def source_proof(out):
    build = read(out / "build.json")
    result = dict(status="PASS", changed_runtime_logic=False, variants={})
    for name, record in build["original_sources"].items():
        reference, target = Path(record["path"]), out / "diagnostic" / name
        entries = {}
        for relative, expected in record["sha256"].items():
            before, after = reference / relative, target / relative
            assert digest(before) == expected
            assert digest(after) == build["source_inputs"][name].get(relative, expected)
            restored = original_text(after.read_text()) if relative.startswith("src/") else after.read_text()
            assert restored.encode() == before.read_bytes(), (name, relative)
            entries[relative] = dict(original_sha256=expected, diagnostic_sha256=digest(after))
        result["variants"][name] = entries
    assert all(digest(out / p) == h for p, h in build["binaries"].items())
    result["retained_execution_inputs"] = []
    for path, expected in build["inputs"].items():
        retained = Path(path)
        if digest(retained) != expected:
            archived = out.parent / "preflight" / (retained.name + ".txt")
            assert digest(archived) == expected, path
            retained = archived
        result["retained_execution_inputs"].append(
            dict(original_location=path, retained_path=str(retained), sha256=expected)
        )
    return result


def distribution(values):
    values = sorted(values)
    return dict(
        count=len(values),
        sum=sum(values),
        median=statistics.median(values) if values else 0,
        max=max(values, default=0),
    )


def complete_phases(out):
    path = out / "ext4-diagnostic/results.json"
    raw = read(path)
    result = dict(
        status="PASS",
        scope="Diagnostic costs only; nested stages and overlapping workers are not additive",
        inputs={str(path): digest(path), str(Path(__file__).resolve()): digest(Path(__file__))},
        mounts=[],
        groups=[],
    )
    phases = []
    for state, key in (("fresh", "fuse"), ("post_gc", "aged_fuse")):
        for mount in raw[key]:
            path = Path(mount["request_costs_file"])
            result["inputs"][str(path)] = digest(path)
            trace = read(path)
            events = [
                dict(zip(trace["columns"], row, strict=True), tid=thread["tid"])
                for thread in trace["threads"]
                for row in thread["rows"]
            ]
            entries = []
            for sample in mount["samples"]:
                if sample["workload"] not in TARGETS:
                    continue
                begin = round(sample["started_epoch"] * 1e9) - trace["origin_epoch_ns"]
                end = round(sample["ended_epoch"] * 1e9) - trace["origin_epoch_ns"]
                selected = [e for e in events if begin <= e["begin_ns"] < end and e["name"] != "checkpoint_worker"]
                counts = collections.Counter(e["name"] for e in selected)
                if sample["workload"] == "small_file_fsync":
                    assert all(counts[n] == 128 for n in ("create", "write", "fsync")), counts
                else:
                    assert (
                        counts["write" if sample["workload"] == "random_write_4k" else "rename"] == sample["requests"]
                    )
                    assert counts["fsync"] == 1, counts
                entry = dict(
                    state=state,
                    variant=mount["variant"],
                    rep=mount["rep"],
                    workload=sample["workload"],
                    seconds=sample["seconds"],
                    handlers=dict(counts),
                    wall_ns=sum(e["end_ns"] - e["begin_ns"] for e in selected),
                    cpu_ns=sum(e["cpu_ns"] for e in selected),
                    vm_steps=sum(e["vm_steps"] for e in selected),
                    operations={},
                )
                for name in counts:
                    rows = [e for e in selected if e["name"] == name]
                    entry["operations"][name] = dict(
                        count=len(rows),
                        wall_ns=sum(e["end_ns"] - e["begin_ns"] for e in rows),
                        cpu_ns=sum(e["cpu_ns"] for e in rows),
                        vm_steps=sum(e["vm_steps"] for e in rows),
                        stage_ns={n: sum(e["stage_ns"][i] for e in rows) for i, n in enumerate(trace["stages"])},
                        stage_counts={n: sum(e["counts"][i] for e in rows) for i, n in enumerate(trace["stages"])},
                    )
                entries.append(entry)
                phases.append(entry)
            result["mounts"].append(dict(state=state, variant=mount["variant"], rep=mount["rep"], phases=entries))
    for state, variant, workload in sorted({(p["state"], p["variant"], p["workload"]) for p in phases}):
        rows = [p for p in phases if (p["state"], p["variant"], p["workload"]) == (state, variant, workload)]
        result["groups"].append(
            dict(
                state=state,
                variant=variant,
                workload=workload,
                phases=len(rows),
                **{k: distribution([p[k] for p in rows]) for k in ("seconds", "wall_ns", "cpu_ns", "vm_steps")},
                handler_counts={
                    n: distribution([p["handlers"].get(n, 0) for p in rows])
                    for n in sorted({n for p in rows for n in p["handlers"]})
                },
            )
        )
    write(out / "complete-phases.json", result)
    print("complete FUSE phase analysis PASS", len(result["mounts"]), "mounts", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    generator = out / "analysis-generator"
    generator.mkdir(exist_ok=True)
    write(out / "source-proof.json", source_proof(out))
    old = here.parent / "sync_costs/analyze.py"
    sync = old.read_text()
    begin, end = sync.index("def source_proof("), sync.index("def main():")
    sync = sync[:begin] + sync[end:]
    sync = sync.replace('    parser.add_argument("--prior-diagnostic", type=Path, required=True)\n', "")
    marker = '    write(out / "source-proof.json", source_proof(out, args.prior_diagnostic.resolve()))'
    assert sync.count(marker) == 1
    sync = sync.replace(marker, '    assert read(out / "source-proof.json")["status"] == "PASS"').replace(
        "import re\n", ""
    )
    request = here.parent / "request_costs/analyze.py"
    requests = request.read_text().replace(
        '    "rename": "rename",', '    "rename": "rename",\n    "small_file_fsync": "fsync",'
    )
    requests = requests.replace('"mkdir", "rename", "rmdir"):', '"mkdir", "rename", "rmdir", "small_file_fsync"):')
    cpu = here.parent / "request_costs/analyze_cpu.py"
    cpus = cpu.read_text().replace('"mkdir", "rename", "rmdir"):', '"mkdir", "rename", "rmdir", "small_file_fsync"):')
    restart = here.parent / "checkpoint_restart/analyze_restart.py"
    restarts = restart.read_text()
    marker = '                    assert mount["variant"] == "updated"'
    assert restarts.count(marker) == 1
    if "reused_artifacts" in read(out / "build.json"):
        restarts = restarts.replace(marker, '                    assert mount["variant"] in ("baseline", "updated")')
    paths = (("sync.py", sync), ("request.py", requests), ("cpu.py", cpus), ("restart.py", restarts))
    for name, source in paths:
        (generator / name).write_text(source)
    write(
        generator / "identity.json",
        dict(
            status="PASS",
            adaptation="Original decoders/reconciliation; add small-file barriers/CPU phases and exact reversible instrumentation proof",
            inputs={str(p): digest(p) for p in (Path(__file__).resolve(), old, request, cpu, restart)},
            generated={n: digest(generator / n) for n, _ in paths},
        ),
    )
    for name, _ in paths[:3]:
        subprocess.run([sys.executable, str(generator / name), str(out)], check=True)
    spec = importlib.util.spec_from_file_location("restart_costs", generator / "restart.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.restart_costs(out)
    complete_phases(out)


if __name__ == "__main__":
    main()

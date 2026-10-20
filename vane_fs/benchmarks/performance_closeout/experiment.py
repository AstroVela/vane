# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Execute the two identified closeout mechanisms using existing tests and fixed gates."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from inventory import digest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
STAGED = HERE.parent / "staged_payload"


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def execute(path, out, changes=()):
    original = text = path.read_text()
    for before, after, count in changes:
        assert text.count(before) == count, (path, before)
        text = text.replace(before, after)
    reverse = text
    for before, after, count in reversed(changes):
        assert reverse.count(after) == count
        reverse = reverse.replace(after, before)
    assert reverse == original
    name = path.parent.name + "-" + path.stem
    (out / (name + "-adapter.py")).write_text(text)
    save(
        out / (name + "-adapter.json"), dict(parent=str(path), sha256=digest(path), changes=changes, reverse_exact=True)
    )
    saved, sys.argv = sys.argv, [str(path), str(out)]
    sys.modules["generate"] = sys.modules[__name__]
    sys.modules["prepare"] = sys.modules[__name__]
    try:
        exec(compile(text, str(path), "exec"), dict(__name__="__main__", __file__=str(path)))
    finally:
        sys.argv = saved


def prepare(out, candidate):
    inventory = json.loads((HERE / "inventory.json").read_text())
    control = Path(inventory["mechanism_control"]["component"])
    base, build = [json.loads((control / name).read_text()) for name in ("source.json", "build.json")]
    inputs = base["inputs"] | base["sources"] | build["inputs"] | build["binaries"] | base["default_unchanged"]
    for p in [control / "source.json", control / "build.json", HERE / "inventory.json", *HERE.glob("*.py")]:
        inputs[str(p)] = digest(p)
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    out.mkdir()
    target, parent = out / "candidate", Path(base["candidate"])
    for p in base["sources"]:
        source = Path(p)
        destination = target / source.relative_to(parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    relative = "src/prewriteback.hpp" if candidate == "C1" else "src/checkpoint.hpp"
    path = target / relative
    original = text = path.read_text()
    if candidate == "C1":
        changes = [("START_BYTES = 64 * 1024 * 1024;", "START_BYTES = 16 * 1024 * 1024;", 1)]
    else:
        condition = """\t\tif (external_payload && code == SQLITE_OK && frames >= copied && copied >= 0 &&
\t\t    Bytes(frames - copied) > 256 * 1024)
\t\t\tPrepareExternal();
"""
        begin, end = text.index("\tvoid PrepareExternal() {"), text.index("\tstatic void Check(int code)")
        changes = [
            (
                "void BeforeWrite(sqlite3 *writer, bool external_payload = false)",
                "void BeforeWrite(sqlite3 *writer, bool = false)",
                1,
            ),
            (condition, "", 1),
            (text[begin:end], "", 1),
        ]
    for before, after, count in changes:
        assert text.count(before) == count
        text = text.replace(before, after)
    reverse = text
    for before, after, count in reversed(changes):
        if after:
            assert reverse.count(after) == count
            reverse = reverse.replace(after, before)
    if all(after for _, after, _ in changes):
        assert reverse == original
    path.write_text(text)
    sources = {str(p): digest(p) for p in target.rglob("*") if p.is_file()}
    changed = [
        str(Path(p).relative_to(target)) for p in sources if digest(parent / Path(p).relative_to(target)) != sources[p]
    ]
    assert changed == [relative]
    fixture = STAGED / "paced_range_writeback/test_pacing.cpp"
    fixture_text = fixture.read_text()
    if candidate == "C1":
        begin, end = fixture_text.index("static void Rate("), fixture_text.index("static void Pauses(")
        old = fixture_text[begin:end]
        new = old.replace("63 * MIB", "15 * MIB").replace("64 * MIB", "16 * MIB").replace("f.Drain(16)", "f.Drain(4)")
        assert new != old
        fixture_text = fixture_text[:begin] + new + fixture_text[end:]
    (out / "test_pacing.cpp").write_text(fixture_text)
    inputs[str(fixture)] = digest(fixture)
    inputs[str(out / "test_pacing.cpp")] = digest(out / "test_pacing.cpp")
    full = STAGED / "paced_range_writeback/test_full_pause.cpp"
    full_text = full.read_text()
    full_changes = [
        (
            '#include "../payload_range_writeback/test_prewriteback.cpp"',
            '#include "' + str(STAGED / "payload_range_writeback/test_prewriteback.cpp") + '"',
        ),
        ("(publication ? 63 : 64) * MIB", "(publication ? Prewriteback::START_BYTES / MIB - 1 : 64) * MIB"),
        (
            "external->OnCommit(64 * MIB, false, {4096, 63 * MIB + 4096});",
            "external->OnCommit(Prewriteback::START_BYTES, false, {4096, Prewriteback::START_BYTES - MIB + 4096});",
        ),
        (
            'reopened.Checkout()->Read("/file", 62 * MIB)',
            'reopened.Checkout()->Read("/file", publication ? int64_t(Prewriteback::START_BYTES) - 2 * MIB : 62 * MIB)',
        ),
    ]
    for before, after in full_changes:
        assert full_text.count(before) == 1
        full_text = full_text.replace(before, after)
    (out / "test_full_pause.cpp").write_text(full_text)
    for p in (full, out / "test_full_pause.cpp"):
        inputs[str(p)] = digest(p)
    save(out / "source-before.json", {relative: original})
    save(
        out / "source.json",
        base
        | dict(
            created_epoch=time.time(),
            head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            parent=str(control),
            mechanism_parent=str(parent),
            original=str(parent),
            candidate=str(target),
            candidate_id=candidate,
            inputs=inputs,
            sources=sources,
            payload_start_bytes=(16 if candidate == "C1" else 64) * 1024**2,
            transformations={
                relative: dict(
                    original_sha256=digest(parent / relative),
                    candidate_sha256=digest(path),
                    changes=changes,
                    complete_original=original,
                    forward_exact=True,
                )
            },
            checkpoint_source_unchanged=candidate == "C1",
            description=f"Bounded performance closeout {candidate}; same paced parent, original 22 gates, no default adoption",
        ),
    )
    print("PASS", candidate, "prepared; only", relative, "differs", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "build", "validate", "freeze", "run", "analyze"))
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--candidate", choices=("C1", "C3"), default="C1")
    args = parser.parse_args()
    out = args.artifacts.resolve()
    if args.action == "prepare":
        prepare(out, args.candidate)
    elif args.action == "build":
        execute(
            STAGED / "paced_range_writeback/build.py",
            out,
            [
                ('here / "test_pacing.cpp"', f"Path({str(out / 'test_pacing.cpp')!r})", 2),
            ],
        )
    elif args.action == "validate":
        execute(
            STAGED / "paced_range_writeback/validate.py",
            out,
            [
                ('here / "test_full_pause.cpp"', f"Path({str(out / 'test_full_pause.cpp')!r})", 2),
            ],
        )
    elif args.action == "freeze":
        for name in ("initial-preflight.json", "full-pause-preflight.json"):
            if not (out / name).exists():
                save(out / name, dict(status="NOT_NEEDED", inputs={}))
        candidate = json.loads((out / "source.json").read_text())["candidate_id"]
        execute(
            STAGED / "paced_range_writeback/prepare.py",
            out,
            [
                (
                    'before = f"Finite-range 64 MiB payload writeback:',
                    'before = f"Paced 4 MiB range claims, 16 ms completion delay and FULL pause:',
                    1,
                ),
                (
                    'after = f"Paced 4 MiB range claims, 16 ms completion delay and FULL pause:',
                    f'after = f"Bounded closeout {candidate}:',
                    1,
                ),
                (
                    "frozen_epoch=time.time(),",
                    'frozen_epoch=time.time(), payload_start_bytes=dict(baseline=64 * 1024**2, updated=source["payload_start_bytes"]), candidate_id=source["candidate_id"],',
                    1,
                ),
            ],
        )
    elif args.action == "run":
        execute(STAGED / "extent_fuse/run.py", out / "fuse")
    elif args.action == "analyze":
        execute(STAGED / "paced_range_writeback/analyze.py", out / "fuse")
        analysis = json.loads((out / "fuse/analysis.json").read_text())
        source = json.loads((out / "source.json").read_text())
        save(
            out / "decision.json",
            dict(
                status="PASS",
                candidate=source["candidate_id"],
                recorded_epoch=time.time(),
                correctness_pass=True,
                performance_pass=analysis["full_acceptance_pass"],
                failed_checks=analysis["full_failed_checks"],
                failed_details=analysis["failure_details"],
                default_adoption=False,
                compatibility_qualification="not performed for format 1005; default formats 1/2 unchanged",
                decision="REJECT" if not analysis["full_acceptance_pass"] else "PERFORMANCE_QUALIFIED_EXPERIMENT_ONLY",
            ),
        )
        print("PASS decision", (out / "decision.json").read_text()[:1500], flush=True)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Record default versus isolated implementations without interpreting audit PASS as adoption."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    core = REPO / "vane_fs"
    previous = core / "benchmarks/staged_payload/gate_sync_waits/final-checks.json"
    defaults = json.loads(previous.read_text())["default_unchanged"]
    assert len(defaults) == 11 and all(digest(Path(p)) == sha for p, sha in defaults.items())
    workspace = (core / "src/workspace.cpp").read_text()
    assert "CREATE TABLE block_payloads" in workspace and "data BLOB NOT NULL CHECK(length(data)=4096)" in workspace
    assert "meta.Integer(1) != 1 && meta.Integer(1) != 2" in workspace
    assert "Staging::" not in workspace
    assert not subprocess.check_output(["git", "diff", "HEAD", "--", "vane_fs/src", "vane_fs/include"], cwd=REPO)
    adopted = [
        "87b3892943",
        "98b0fe44c6",
        "9ee791a625",
        "49545265f8",
        "08535e6e53",
        "f40c57742e",
        "0fc47514e4",
        "7cccaf814f",
    ]
    commits = []
    for commit in adopted:
        subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=REPO, check=True)
        commits.append(
            subprocess.check_output(["git", "show", "-s", "--format=%H %s", commit], cwd=REPO, text=True).strip()
        )
    reports = []
    for folder in sorted((core / "benchmarks/staged_payload").iterdir()):
        if not folder.is_dir():
            continue
        records = {}
        for name in ("audit.json", "findings.json", "final-checks.json"):
            path = folder / name
            if not path.is_file():
                continue
            data = json.loads(path.read_text())
            records[name] = dict(
                path=str(path.relative_to(REPO)),
                sha256=digest(path),
                status=data.get("status"),
                decision=data.get("decision"),
                explicit_acceptance_pass=data.get("original_full_acceptance_pass", data.get("full_acceptance_pass")),
            )
        reports.append(
            dict(
                experiment=folder.name,
                default_adoption=False,
                classification="isolated evidence; audit status alone is not an adoption decision",
                readme=str((folder / "README.md").relative_to(REPO)) if (folder / "README.md").exists() else None,
                records=records,
            )
        )
    paced = core / "build/paced-range-writeback-20261019-v2"
    source, build = [json.loads((paced / name).read_text()) for name in ("source.json", "build.json")]
    assert source["status"] == build["status"] == "PASS" and source["format"] == 1005
    identities = source["sources"] | build["binaries"]
    assert all(digest(Path(p)) == sha for p, sha in identities.items())
    report = dict(
        status="PASS",
        recorded_epoch=time.time(),
        source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        default=dict(
            storage="SQLite 4 KiB BLOB payloads",
            supported_formats=[1, 2],
            native_sources_clean=True,
            identities=defaults,
            adopted_commits=commits,
        ),
        mechanism_control=dict(
            component=str(paced),
            candidate=source["candidate"],
            format=1005,
            trigger_mib=64,
            claim_mib=4,
            completion_delay_ms=16,
            begin_restart_gate=False,
            adopted=False,
            identities=identities,
        ),
        historical_experiments=reports,
        last_gate_failed_checks=9,
        decision="Default storage retained; isolated variants need separate performance and compatibility decisions",
        candidate_budget=dict(total=3, prewriteback=2, checkpoint=1, consumed=0),
        inputs={str(p): digest(p) for p in (Path(__file__), previous, paced / "source.json", paced / "build.json")},
    )
    (HERE / "inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS", len(reports), "isolated experiments inventoried; 11 default identities and matched control verified")


if __name__ == "__main__":
    main()

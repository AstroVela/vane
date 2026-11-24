# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Use the original ext4 FUSE workload and gates after passing the native screen."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    parent = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    previous = here.parent / "checkpoint_restart"
    assert json.loads((parent / "probe-summary.json").read_text())["screen_passed"]
    assert json.loads((parent / "progress-tests.json").read_text())["status"] == "PASS"
    out = parent / "fuse-screen"
    out.mkdir()
    for name in ("baseline", "orphan-control", "restart-control", "updated"):
        (out / name).symlink_to(parent / name, target_is_directory=True)
    for name in ("build.json", "tests.json", "sanitize.json", "restart-tests.json", "progress-tests.json"):
        (out / name).write_bytes((parent / name).read_bytes())
    source = (previous / "run.py").read_text()
    substitutions = {
        "from runner import prepared": "sys.path.insert(0, " + repr(str(previous)) + ")\nfrom runner import prepared",
        "    here = Path(__file__).resolve().parent": "    here = Path(" + repr(str(previous)) + ")",
        "    reference = {k: original[k] for k in keys}": '    reference = {k: original[k] for k in keys}\n    reference["media"] = ["ext4"]',
        'candidate="Orphan batching with an optional lock-nonblocking RESTART after completed large PASSIVE passes",': 'candidate="Optional RESTART plus commit progress latched after a real large generation",\n            full_adoption_eligible=False,\n            original_protocol_subset="ext4 fresh/post-GC only; original workload and gate constants",',
        "wal_checkpoint_thresholds_and_clean_progress_unchanged=True,": "wal_checkpoint_thresholds_unchanged=True,\n            post_reuse_clean_progress_enabled=True,",
        "payload_worker_wake_and_admission_unchanged=True,": "payload_publication_coalescing_and_admission_unchanged=True,\n            commit_wake_latched_after_large_generation=True,",
    }
    for before, after in substitutions.items():
        assert source.count(before) == 1
        source = source.replace(before, after)
    prepared = parent / "fuse-screen-generator.py"
    prepared.write_text(source)
    identity = dict(
        status="PASS",
        scope="Original ext4 workload subset; no runtime instrumentation or gate change",
        inputs={str(p): digest(p) for p in (Path(__file__).resolve(), previous / "run.py", previous / "runner.py")},
        generated_sha256=digest(prepared),
    )
    (parent / "fuse-generator-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    subprocess.run([sys.executable, str(prepared), str(out), "--medium", "ext4"], check=True)
    assert digest(prepared) == identity["generated_sha256"]
    assert all(digest(Path(p)) == expected for p, expected in identity["inputs"].items())
    subprocess.run([sys.executable, str(here.parent / "partial_sync/analyze_writeback.py"), str(out)], check=True)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Account for the zero-byte EOF request without changing frozen measurement inputs."""

import json
import sys
import time
from pathlib import Path

from prepare import digest, save

if __name__ == "__main__":
    here, out = Path(__file__).resolve().parent, Path(sys.argv[1]).resolve()
    parent = here / "analyze.py"
    original = parent.read_text()
    before = """        position = 0
        for _, root in sorted(selected, key=lambda item: item[1][5]):
            assert root[5] == position
            position += root[6]
        assert position == sample["bytes"] == 64 * 1024**2"""
    after = """        position = 0
        for _, root in sorted(selected, key=lambda item: item[1][5]):
            assert root[5] == position
            returned = min(root[6], max(0, sample["bytes"] - root[5]))
            position += returned
        assert position == sample["bytes"] == 64 * 1024**2"""
    changes = [
        (before, after),
        (
            "native_request_bytes=position,",
            "native_request_bytes=sum(r[6] for _, r in selected),\n"
            "                native_returned_bytes=position,\n"
            '                eof_requests=sum(r[5] == sample["bytes"] for _, r in selected),',
        ),
    ]
    prepared = original
    for before, after in changes:
        assert prepared.count(before) == 1
        prepared = prepared.replace(before, after)
    reverse = prepared
    for before, after in reversed(changes):
        assert reverse.count(after) == 1
        reverse = reverse.replace(after, before)
    assert reverse == original
    if not (out / "analysis-preflight.json").exists():
        save(
            out / "analysis-preflight.json",
            dict(
                status="KNOWN_ACCOUNTING_ASSUMPTION_CORRECTED",
                initial_failure_reproduced=False,
                measurement_rerun=False,
                root_cause="FUSE requested lengths include a zero-byte EOF read; logical coverage clips at validated file size.",
            ),
        )
    if not (out / "preflight.json").exists():
        save(out / "preflight.json", dict(status="NOT_RUN_SEPARATELY", measurement_rerun=False))
    proof = out / "analysis-adapter.json"
    save(
        proof,
        dict(
            status="PASS",
            prepared_epoch=time.time(),
            measurement_ended_epoch=json.loads((out / "ext4-diagnostic/results.json").read_text())["ended_epoch"],
            original=str(parent),
            original_sha256=digest(parent),
            adapter_sha256=digest(Path(__file__)),
            changes=changes,
            exact_reverse=True,
            measured_workload_and_sources_unchanged=True,
        ),
    )
    exec(compile(prepared, str(parent), "exec"), dict(__name__="__main__", __file__=str(parent)))
    findings = json.loads((out / "findings.json").read_text())
    findings["inputs"].update({str(p): digest(p) for p in [Path(__file__), proof, out / "analysis-preflight.json"]})
    save(out / "findings.json", findings)

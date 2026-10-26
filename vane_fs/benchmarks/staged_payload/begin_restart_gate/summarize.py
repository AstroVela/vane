# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain every ordinary sample and distinguish gate causality from combined gains."""

import json
import sys
from pathlib import Path

from generate import digest, save

if __name__ == "__main__":
    parent = Path(__file__).resolve().parent.parent / "paced_range_writeback/summarize.py"
    exec(compile(parent.read_text(), str(parent), "exec"), dict(__name__="__main__", __file__=__file__))
    out = Path(sys.argv[1]).resolve()
    report = json.loads((out / "findings.json").read_text())
    tests = json.loads((out / "gate-tests.json").read_text())
    assert tests["status"] == "PASS"
    report["inputs"].update(tests["inputs"] | tests["binaries"])
    report["inputs"][str(out / "gate-tests.json")] = digest(out / "gate-tests.json")
    report.update(
        begin_gate_cases_per_mode=tests["cases_per_mode"],
        parent_negative_cases_per_mode=tests["parent_cases_per_mode"],
        original_paced_fuse_failures=9,
        limitations=[
            "The ordinary contemporaneous baseline is the original no-hint extent implementation. This window tests the combined paced-plus-gate candidate; it does not isolate incremental gate throughput from pacing. Historical nine failures remain historical, not matched gate effects.",
            "Controlled negative/positive tests isolate the BEGIN/RESTART mechanism using real WAL writer locks and actual default-busy-handler sleeps. They do not bound device sync latency.",
            "Supplementary sustained measurements have two pairs per state and do not replace original six-pair gates.",
            "Ordinary application timings cannot identify individual native mutex, checkpoint, syscall or device waits. Earlier instrumented windows have different overhead and are not adoption measurements.",
            "A local BEGIN gate coordinates this owner and its worker. Foreign connections still use SQLite busy handling; PASSIVE I/O and payload/WAL/CLEAN barriers remain synchronous where required.",
        ],
    )
    save(out / "findings.json", report)

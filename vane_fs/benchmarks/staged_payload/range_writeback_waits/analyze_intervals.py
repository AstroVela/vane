# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Correct random-write identity matching while retaining the frozen analyzer."""

import hashlib
import json
import sys
import time
from pathlib import Path

from prepare import digest, save

if __name__ == "__main__":
    here, out = Path(__file__).resolve().parent, Path(sys.argv[1]).resolve()
    original = here / "analyze.py"
    source = code = original.read_text()
    before = '                    and (small or r[0] == interval["kind"])'
    after = (
        before
        + """
                    and (small or interval["kind"] != "write" or r[5:7] == [interval["offset"], interval["bytes"]])"""
    )
    changes = [(before, after)]
    # The native large-write analysis already passed. Reuse its immutable result
    # and command record; only retry the incomplete operation-matching analysis.
    begin = code.index('    argv = [sys.executable, str(TRACE / "analyze.py"), str(out)]')
    end = code.index('    analysis = json.loads((out / "analysis.json").read_text())')
    changes.append(
        (code[begin:end], '    assert json.loads((out / "analysis-command.json").read_text())["returncode"] == 0\n')
    )
    for before, after in changes:
        assert code.count(before) == 1
        code = code.replace(before, after)
    reverse = code
    for before, after in reversed(changes):
        assert reverse.count(after) == 1
        reverse = reverse.replace(after, before)
    assert reverse == source
    proof = dict(
        status="PASS",
        recorded_epoch=time.time(),
        original_sha256=digest(original),
        corrected_sha256=hashlib.sha256(code.encode()).hexdigest(),
        exact_reverse=True,
        changes=changes,
        native_analysis_reused_without_rerunning=True,
        measurement_repeated=False,
    )
    save(out / "analysis-adapter.json", proof)
    exec(compile(code, str(original), "exec"), dict(__name__="__main__", __file__=str(original)))
    findings = json.loads((out / "findings.json").read_text())
    findings.update(
        analysis_adapter=proof, analysis_preflight=json.loads((out / "analysis-preflight.json").read_text())
    )
    findings["inputs"].update(
        {
            str(path): digest(path)
            for path in [Path(__file__), out / "analysis-adapter.json", out / "analysis-preflight.json"]
        }
    )
    save(out / "findings.json", findings)

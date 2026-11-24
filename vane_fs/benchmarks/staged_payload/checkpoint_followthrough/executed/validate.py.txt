# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reuse the focused suites and add negative controls for both CLEAN budgets."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from prepare import digest, replace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    candidate = json.loads((out / "candidate.json").read_text())
    assert candidate["status"] == "PASS"
    previous = Path(candidate["parent"]) / "coalesced"
    (out / "previous").symlink_to(previous, target_is_directory=True)
    old = here.parent / "clean_coalescing/validate.py"
    source = old.read_text()
    source = replace(source, "here = Path(__file__).resolve().parent", "here = Path(" + repr(str(old.parent)) + ")")
    source = source.replace('"lookup"', '"previous"').replace('"coalesced"', '"updated"')
    source = source.replace('"coalesced-', '"updated-')
    source = replace(
        source,
        'for name, source in (("production", repo / "vane_fs"), ("updated", component)):',
        'for name, source in (("updated", component),):',
    )
    source = replace(
        source,
        '        for name in ("previous", "updated"):',
        '        helpers["budget"] = (Path('
        + repr(str(here / "test_budget.cpp"))
        + '), ("generation", "tail"), ("sqlite3_wal_checkpoint_v2",))\n        for name in ("previous", "updated"):',
    )
    source = replace(
        source, 'if variant == "previous" and name != "clean":', 'if variant == "previous" and name != "budget":'
    )
    source = replace(source, 'for case in cases if variant == "updated" else ("burst",):', "for case in cases:")
    source = replace(
        source,
        'marker="Small FULL commits were not coalesced" if variant == "previous" else None,',
        'marker="Urgent CLEAN work entered the coalescing timer" if variant == "previous" else None,',
    )
    generator = out / "validate-generator.py"
    generator.write_text(source)
    (out / "validation-inputs.json").write_text(
        json.dumps({str(p): digest(p) for p in (Path(__file__), old, here / "test_budget.cpp", generator)}, indent=2)
        + "\n"
    )
    subprocess.run([sys.executable, str(generator), str(out)], check=True)


if __name__ == "__main__":
    main()

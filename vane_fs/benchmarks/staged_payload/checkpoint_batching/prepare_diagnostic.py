# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reuse the request recorder with the preceding budget as its causal control."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--sqlite-prefix", type=Path, required=True)
    parser.add_argument("--fuse-prefix", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    old = here.parent / "request_costs"
    generator = args.parent.resolve() / "diagnostic-generator"
    generator.mkdir()
    source = (old / "prepare.py").read_text()
    substitutions = {
        'parent / "writeback-tests.json"': 'parent / "batch-tests.json"',
        'Path(identity["parent"]) / "updated"': 'Path(identity["budget_parent"]) / "updated"',
        'expected = identity["source_inputs"][name]': 'expected = identity["source_inputs"]["budget-control" if name == "baseline" else name]',
        'if name == "updated":': 'if name in ("baseline", "updated"):',
    }
    for before, after in substitutions.items():
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    (generator / "prepare.py").write_text(source)
    for name in ("trace.hpp", "trace.cpp"):
        shutil.copy2(old / name, generator / name)
    inputs = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (here / "prepare_diagnostic.py", old / "prepare.py", old / "trace.hpp", old / "trace.cpp")
    }
    (generator / "identity.json").write_text(
        json.dumps(
            dict(
                status="PASS",
                inputs=inputs,
                adaptation="Report name, budget control binding, writeback instrumentation for both variants",
            ),
            indent=2,
        )
        + "\n"
    )
    subprocess.run(
        [
            sys.executable,
            str(generator / "prepare.py"),
            str(args.output),
            "--parent",
            str(args.parent),
            "--sqlite-prefix",
            str(args.sqlite_prefix),
            "--fuse-prefix",
            str(args.fuse_prefix),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()

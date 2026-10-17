# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Record presync separately on both variants, with preceding orphan batching as control."""

import argparse
import hashlib
import json
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
    assert source.count('parent / "writeback-tests.json"') == 1
    source = source.replace('parent / "writeback-tests.json"', 'parent / "presync-tests.json"')
    start = source.index('            if name == "updated":')
    end = source.index('            (target / "src/staging.hpp").write_text(stage)', start)
    source = source[:start] + source[end:]
    for before, after in {
        'Path(identity["parent"]) / "updated"': 'Path(identity["orphan_parent"]) / "updated"',
        'expected = identity["source_inputs"][name]': 'expected = identity["source_inputs"]["orphan-control" if name == "baseline" else name]',
    }.items():
        assert source.count(before) == 1
        source = source.replace(before, after)
    marker = '            (target / "src/staging.hpp").write_text(stage)'
    source = source.replace(
        marker,
        '            if name == "updated":\n                stage = replace_once(stage, "void Presync() {", "void Presync() {" + span("Presync"))\n'
        + marker,
    )
    (generator / "prepare.py").write_text(source)
    header = (old / "trace.hpp").read_text()
    assert header.count("\tCount\n") == 1
    (generator / "trace.hpp").write_text(header.replace("\tCount\n", "\tPresync,\n\tCount\n"))
    recorder = (old / "trace.cpp").read_text()
    assert recorder.count('"hint"};') == 1
    (generator / "trace.cpp").write_text(recorder.replace('"hint"};', '"hint", "presync"};'))
    inputs = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            here / "prepare_diagnostic.py",
            old / "prepare.py",
            old / "trace.hpp",
            old / "trace.cpp",
        )
    }
    (generator / "identity.json").write_text(
        json.dumps(
            dict(
                status="PASS",
                inputs=inputs,
                adaptation="Report name, orphan control binding, no optional hint/pause probes, extra presync stage; runtime logic unchanged",
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

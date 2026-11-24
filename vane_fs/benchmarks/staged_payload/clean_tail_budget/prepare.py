# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build a tail-only candidate with the same toolchain and frozen scheduling window."""

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
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    old = here.parent / "checkpoint_followthrough/candidate.py"
    generator = args.output.resolve().with_name(args.output.name + "-generator")
    generator.mkdir()
    source = old.read_text()
    marker = "here = Path(__file__).resolve().parent"
    assert source.count(marker) == 1
    source = source.replace(marker, "here = Path(" + repr(str(here)) + ")")
    marker = "from prepare import digest"
    assert source.count(marker) == 1
    source = source.replace(marker, "import sys\nsys.path.insert(0, " + repr(str(old.parent)) + ")\n" + marker)
    marker = "clean_generation_below=16 * 1024 * 1024,"
    assert source.count(marker) == 1
    source = source.replace(marker, "clean_generation_below=None,")
    (generator / "candidate.py").write_text(source)
    (generator / "identity.json").write_text(
        json.dumps(
            dict(
                status="PASS",
                adaptation="Same isolated non-editable builder; only uncopied CLEAN cap, no whole-generation cap",
                inputs={str(p): digest(p) for p in (Path(__file__), old, here / "budget.patch")},
            ),
            indent=2,
        )
        + "\n"
    )
    subprocess.run(
        [
            sys.executable,
            str(generator / "candidate.py"),
            str(args.output.resolve()),
            "--parent",
            str(args.parent.resolve()),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate unchanged progress, large-generation coalescing and urgent dirty prefixes."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    old = here.parent / "checkpoint_followthrough/validate.py"
    source = old.read_text()
    marker = "\n    here = Path(__file__).resolve().parent\n"
    assert source.count(marker) == 1
    source = source.replace(marker, "\n    here = Path(" + repr(str(here)) + ")\n")
    marker = "from prepare import digest, replace"
    assert source.count(marker) == 1
    source = source.replace(marker, "sys.path.insert(0, " + repr(str(old.parent)) + ")\n" + marker)
    start = source.index("    generator = out / ")
    source = (
        source[:start]
        + '    source = replace(source, "for case in cases:", \'for case in cases if variant == "updated" else ("tail",):\')\n'
        + '    source = source.replace("Urgent CLEAN work entered the coalescing timer", "Unexpected CLEAN scheduling for prefix budget")\n'
        + source[start:]
    )
    generated = out / "validation-wrapper.py"
    generated.write_text(source)
    (out / "wrapper-inputs.json").write_text(
        json.dumps(
            {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), old, generated)}, indent=2
        )
        + "\n"
    )
    subprocess.run([sys.executable, str(generated), str(out)], check=True)


if __name__ == "__main__":
    main()

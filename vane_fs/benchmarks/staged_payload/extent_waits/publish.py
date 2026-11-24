# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reuse lossless indexed trace publication for the extent fsync diagnosis."""

import json
import sys
from pathlib import Path

from prepare import TRACE, digest, save


def main():
    out, here = Path(sys.argv[1]).resolve(), Path(__file__).resolve().parent
    extras = ("wrapper-proof.json", "analysis-command.json")
    for name in extras:
        path = here / "validation" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        value = json.loads((out / name).read_text())
        path.write_text(json.dumps(value, separators=(",", ":")) + "\n")
        assert json.loads(path.read_text()) == value
    source = TRACE / "publish.py"
    exec(compile(source.read_text(), str(source), "exec"), dict(__name__="__main__", __file__=__file__))
    audit = json.loads((here / "audit.json").read_text())
    audit.update(
        format_versions=dict(baseline=1004, updated=1005),
        original_plain_failed_checks=14,
        baseline_diagnostic_reused_by_exact_source_identity=True,
    )
    audit["inputs"][str(source)] = digest(source)
    for name in extras:
        audit["inputs"][str(out / name)] = digest(out / name)
        audit["original_files"][str(out / name)] = dict(
            sha256=digest(out / name), retained_as="validation/" + name, values_unchanged=True
        )
    save(here / "audit.json", audit)


if __name__ == "__main__":
    main()

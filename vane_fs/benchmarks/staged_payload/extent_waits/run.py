# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reuse the tested wait runner with the per-format extent workload template."""

import sys
from pathlib import Path

from prepare import TRACE, digest, save


def main():
    out = Path(sys.argv[1]).resolve()
    original = (TRACE / "run.py").read_text()
    source, changes = original, []

    def replace(before, after):
        nonlocal source
        assert source.count(before) == 1, before
        source = source.replace(before, after)
        changes.append([before, after])

    replace('template = Path(build["parent"]) / "fuse-screen"', 'template = Path(build["template"])')
    replace(
        'replace("(1 if args.diagnostic else 3)", "(2 if args.diagnostic else 3)")',
        'replace(\'m.ORDERS = m.FIRST = [["baseline", "updated"], ["updated", "baseline"]] * 3\', \'m.ORDERS = m.FIRST = [["baseline", "updated"], ["updated", "baseline"]] * 2\')',
    )
    before = '    replace("import types", "import types\\nimport threading")'
    replace(
        before,
        before
        + "\n    replace('assert not args.diagnostic, \"Ordinary uninstrumented components only\"', 'assert args.diagnostic, \"Diagnostic components only\"')",
    )
    replace(
        'interpretation="Borrowed input buffers in matched format-1004 FUSE components on ext4; native CPU screen is separate",',
        'interpretation="Versioned extents: full ordinary FUSE window; per-variant GC modules",',
    )
    replace(
        'interpretation="Wait attribution for both frozen buffer implementations; original FUSE gate failures remain",',
        'interpretation="Format-1004/1005 fsync attribution; all original 14 ordinary FUSE failures remain",',
    )
    reverse = source
    for before, after in reversed(changes):
        assert reverse.count(after) == 1
        reverse = reverse.replace(after, before)
    assert reverse == original
    save(
        out / "wrapper-proof.json",
        dict(status="PASS", original_sha256=digest(TRACE / "run.py"), exact_reverse=True, changes=changes),
    )
    exec(compile(source, str(TRACE / "run.py"), "exec"), dict(__name__="__main__", __file__=__file__))


if __name__ == "__main__":
    main()

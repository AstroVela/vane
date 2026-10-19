# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Preserve complete workloads and four AB/BA pairs for the paced diagnosis."""

from pathlib import Path

from prepare import digest, save


if __name__ == "__main__":
    import sys

    parent = Path(__file__).resolve().parent.parent / "range_writeback_waits/run.py"
    source = parent.read_text()
    changes = [
        (
            "Finite-range 64 MiB payload writeback: full ordinary FUSE window; both formats 1005",
            "Paced 4 MiB range claims, 16 ms completion delay and FULL pause: full ordinary FUSE window; both formats 1005",
        ),
        ("all original 7 ordinary FUSE failures remain", "all original 9 ordinary FUSE failures remain"),
    ]
    code = source
    for before, after in changes:
        assert code.count(before) == 1
        code = code.replace(before, after)
    reverse = code
    for before, after in reversed(changes):
        assert reverse.count(after) == 1
        reverse = reverse.replace(after, before)
    assert reverse == source
    save(
        Path(sys.argv[1]).resolve() / "paced-runner-adapter.json",
        dict(status="PASS", parent=str(parent), sha256=digest(parent), changes=changes, exact_reverse=True),
    )
    exec(compile(code, str(parent), "exec"), dict(__name__="__main__", __file__=__file__))

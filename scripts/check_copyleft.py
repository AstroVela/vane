#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Check the reviewed GPL-family source inventory and native dependency profile."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vane_packaging.copyleft_policy import (  # noqa: E402
    check_installed_notices,
    check_native_manifest,
    check_source_inventory,
    load_policy,
    source_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--share-dir", type=Path, help="Also review installed vcpkg copyright records")
    args = parser.parse_args()
    policy = load_policy(ROOT)
    if (ROOT / ".git").exists():
        paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    else:
        paths = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()]
    check_source_inventory(
        ((path, (ROOT / path).read_bytes()) for path in paths if source_candidate(path)), policy["source_files"]
    )
    manifest = json.loads((ROOT / "vcpkg.json").read_text())
    if manifest["builtin-baseline"] != policy["vcpkg_baseline"]:
        raise ValueError("vcpkg baseline changed; review the GPL-family dependency inventory")
    check_native_manifest(manifest)
    checked = check_installed_notices(args.share_dir, policy["installed_notices"]) if args.share_dir else []
    print(json.dumps({"source_inventory": "passed", "native_profile": "passed", "installed_notices": checked}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

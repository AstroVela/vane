#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare, verify and retrieve the complete native media release delivery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vane_packaging.media_release import download_release, prepare_release, verify_release
from vane_packaging.python_delivery import inventory_delivery


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="clean-verify and stage all files before publication")
    for role in ("base", "provider", "runtime", "source"):
        prepare.add_argument("--" + role, type=Path, required=True)
    prepare.add_argument("--trust-identity", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify", help="check a downloaded directory against a retained manifest digest")
    verify.add_argument("--directory", type=Path, required=True)
    verify.add_argument("--manifest-sha256", required=True)
    verify.add_argument("--trust-identity", required=True)
    download = commands.add_parser("download", help="download and clean-verify every published file")
    download.add_argument("--base-url", required=True)
    download.add_argument("--expected-manifest", type=Path, required=True)
    download.add_argument("--trust-identity", required=True)
    download.add_argument("--output", type=Path, required=True)
    rebuild = commands.add_parser(
        "rebuild", help="rebuild downloaded sources and execute a modified SoXR in a clean install"
    )
    rebuild.add_argument("--directory", type=Path, required=True)
    rebuild.add_argument("--manifest-sha256", required=True)
    rebuild.add_argument("--trust-identity", required=True)
    rebuild.add_argument("--output", type=Path, required=True)
    rebuild.add_argument("--jobs", type=int, default=2)
    inventory = commands.add_parser("inventory-python", help="record the exact wheels shipped in an offline delivery")
    inventory.add_argument("--wheel", type=Path, action="append", required=True)
    inventory.add_argument("--output", type=Path, required=True)
    arguments = vars(parser.parse_args())
    command = arguments.pop("command")
    if command == "prepare":
        digest = prepare_release(**arguments)
    elif command == "download":
        digest = download_release(**arguments)
    elif command == "verify":
        verify_release(**arguments)
        digest = arguments["manifest_sha256"]
    elif command == "rebuild":
        from vane_packaging.media_rebuild import rebuild_release

        print(json.dumps(rebuild_release(**arguments), sort_keys=True))
        return
    else:
        document = inventory_delivery(arguments["wheel"])
        with arguments["output"].open("x") as output:
            output.write(json.dumps(document, sort_keys=True, indent=2) + "\n")
        return
    print(json.dumps({"command": command, "manifest_sha256": digest, "verified": True}, sort_keys=True))


if __name__ == "__main__":
    main()

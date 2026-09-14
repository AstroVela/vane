#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run media publication gates and clean verification of the exact indexed files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vane_packaging.media_publish import preflight, promote_github, publish_github, stage_index, verify_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--github-output", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("preflight")
    prepare.add_argument("--release", action="store_true")
    prepare.add_argument("--output", type=Path, required=True)
    for name in ("stage-index", "verify-index", "publish-candidate", "promote-github"):
        command = commands.add_parser(name)
        command.add_argument("--directory", type=Path, required=True)
        command.add_argument("--manifest-sha256", dest="digest", required=True)
        if name.endswith("index"):
            command.add_argument("--channel", choices=("pypi", "testpypi"), required=True)
            command.add_argument("--output", type=Path, required=True)
    arguments = vars(parser.parse_args())
    command = arguments.pop("command")
    github_output = arguments.pop("github_output")
    if command == "preflight":
        result = preflight(**arguments)
    elif command == "stage-index":
        result = {"publish": "true" if stage_index(**arguments) else "false"}
    elif command == "verify-index":
        result = verify_index(**arguments)
    elif command == "publish-candidate":
        result = {"release_url": publish_github(**arguments)}
    else:
        promote_github(**arguments)
        result = {"promoted": True}
    if github_output:
        with github_output.open("a", encoding="utf-8") as stream:
            for key, value in result.items():
                if isinstance(value, str):
                    if "\n" in value or "\r" in value:
                        raise ValueError("invalid multiline workflow output")
                    stream.write(f"{key}={value}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

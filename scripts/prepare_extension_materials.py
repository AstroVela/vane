#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bind reviewed source/relinking materials to one native extension artifact."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ORIGINAL_SYS_PATH = sys.path.copy()
try:
    sys.path.insert(0, str(REPOSITORY_ROOT))
    from vane_packaging.artifact_limits import MAX_EXTENSION_ARTIFACT_BYTES
    from vane_packaging.extension_materials import (
        MANIFEST_NAME,
        MAX_MANIFEST_BYTES,
        parse_json,
        prepare_manifest,
        read_material_file,
    )
finally:
    sys.path[:] = _ORIGINAL_SYS_PATH
    del _ORIGINAL_SYS_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--extension-name", required=True)
    parser.add_argument("--license-expression", required=True)
    parser.add_argument("--directory", required=True, type=Path, help="Completed source/relinking materials directory")
    parser.add_argument("--inventory", required=True, type=Path, help="Reviewed library and application file inventory")
    arguments = parser.parse_args()
    artifact = arguments.artifact.expanduser().absolute()
    if artifact.name != f"{arguments.extension_name}.duckdb_extension":
        parser.error("artifact basename must match --extension-name")
    inventory = arguments.inventory.expanduser().absolute()
    directory = arguments.directory.expanduser().resolve(strict=True)
    if not directory.is_dir():
        parser.error("--directory must be a directory")
    artifact_contents = read_material_file(artifact.parent, artifact.name, max_bytes=MAX_EXTENSION_ARTIFACT_BYTES)
    artifact_sha256 = hashlib.sha256(artifact_contents).hexdigest()
    del artifact_contents
    contents = prepare_manifest(
        parse_json(read_material_file(inventory.parent, inventory.name, max_bytes=MAX_MANIFEST_BYTES)),
        lambda name: read_material_file(directory, name),
        name=arguments.extension_name,
        artifact_sha256=artifact_sha256,
        license_expression=arguments.license_expression,
    )
    descriptor, temporary_name = tempfile.mkstemp(prefix=".materials-", dir=directory)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(contents)
        destination = directory / MANIFEST_NAME
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

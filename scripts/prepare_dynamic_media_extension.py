#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Relocate an unsigned media extension, then bind its runtime before signing."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vane_packaging.extension_wheel import _parse_elf_dynamic_linkage, _validate_linux_elf_platform
from vane_packaging.media_runtime import exported_versions, validate_library_graph


def prepare(artifact: Path, runtime: Path, patchelf: str = "patchelf") -> None:
    format_path = ROOT / "vane/_native_runtime_format.py"
    if not format_path.is_file():
        format_path = ROOT / "_native_runtime_format.py"
    spec = importlib.util.spec_from_file_location("_native_runtime_format", format_path)
    fmt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fmt)
    document = fmt.read_file(runtime, fmt.MANIFEST, fmt.MAX_MANIFEST_BYTES)
    manifest = fmt.parse_manifest(document)
    fmt.verify_files(runtime / ".libs", manifest)
    libraries = {name: fmt.read_file(runtime / ".libs", name) for name in manifest["files"]}
    validate_library_graph(libraries, manifest["platform"])
    contents = artifact.read_bytes()
    if len(contents) < 512 or contents[-256:] != bytes(256):
        raise ValueError("prepare only an unsigned DuckDB extension with its footer already attached")
    payload = contents[:-512]
    if fmt.trailer_digest(contents) is not None:
        payload = payload[: -fmt.TRAILER_SIZE]
    prefix = manifest["namespace"] + "_"
    if any(not name.startswith(prefix) for name in libraries):
        raise ValueError("runtime library does not use its declared namespace")
    with tempfile.TemporaryDirectory(prefix="vane-media-relocate-", dir=artifact.parent) as temporary:
        target = Path(temporary) / artifact.name
        target.write_bytes(payload)
        arguments = [patchelf, "--set-rpath", "$ORIGIN/.libs"]
        for name in libraries:
            arguments.extend(("--replace-needed", name[len(prefix) :], name))
        subprocess.run([*arguments, str(target)], check=True, timeout=60)
        payload = target.read_bytes()
        _validate_linux_elf_platform(
            payload,
            manifest["platform"],
            description=artifact.name,
            bundled_versions={name: exported_versions(value) for name, value in libraries.items()},
            allowed_runpath="$ORIGIN/.libs",
        )
        linkage = _parse_elf_dynamic_linkage(payload, description=artifact.name, allowed_runpath="$ORIGIN/.libs")
        if not set(linkage.needed) & libraries.keys():
            raise ValueError("media extension does not dynamically link the selected runtime")
        target.write_bytes(fmt.attach_trailer(payload + contents[-512:], fmt.reference(document)["manifest_sha256"]))
        shutil.copymode(artifact, target)
        target.replace(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--patchelf", default="patchelf")
    arguments = parser.parse_args()
    prepare(arguments.artifact, arguments.runtime, arguments.patchelf)

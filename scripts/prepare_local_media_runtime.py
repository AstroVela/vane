#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a locally rebuilt media library without an official signing key."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vane_packaging.media_runtime import validate_library_graph


def prepare_local(runtime: Path, replacements: dict[str, Path], output: Path, *, patchelf="patchelf") -> None:
    format_path = ROOT / "vane/_native_runtime_format.py"
    if not format_path.is_file():
        format_path = ROOT / "_native_runtime_format.py"
    spec = importlib.util.spec_from_file_location("_native_runtime_format", format_path)
    fmt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fmt)
    document = fmt.read_file(runtime, fmt.MANIFEST, fmt.MAX_MANIFEST_BYTES)
    manifest = fmt.parse_manifest(document)
    fmt.verify_files(runtime / ".libs", manifest)
    prefix = manifest["namespace"] + "_"
    mapping = {name.removeprefix(prefix): name for name in manifest["files"]}
    if not replacements or set(replacements) - mapping.keys():
        raise ValueError("select existing media SONAMEs for local replacement")
    if output.exists():
        raise ValueError("local runtime output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".vane-local-media-", dir=output.parent) as temporary:
        stage = Path(temporary) / "runtime"
        stage.mkdir(mode=0o700)
        libraries = stage / ".libs"
        libraries.mkdir(mode=0o700)
        for name in manifest["files"]:
            shutil.copyfile(runtime / ".libs" / name, libraries / name)
        for original, source in replacements.items():
            target = libraries / mapping[original]
            contents = source.read_bytes()
            if not 0 < len(contents) <= fmt.MAX_FILE_BYTES:
                raise ValueError("local replacement library exceeds its size bound")
            target.write_bytes(contents)
            arguments = [patchelf, "--set-soname", mapping[original], "--set-rpath", "$ORIGIN"]
            for old, new in mapping.items():
                arguments.extend(("--replace-needed", old, new))
            subprocess.run([*arguments, str(target)], check=True, timeout=60)
        contents = {path.name: path.read_bytes() for path in libraries.iterdir()}
        graph = validate_library_graph(contents, manifest["platform"])
        for name, value in contents.items():
            manifest["files"][name].update(
                sha256=hashlib.sha256(value).hexdigest(), size=len(value), needed=list(graph[name])
            )
        (stage / fmt.MANIFEST).write_bytes(fmt.canonical_json(manifest))
        (stage / "LOCAL-REBUILD.txt").write_text(
            "This is a locally modified runtime, not an official signed release.\n"
            "The source reference identifies its official base. Preserve your source changes separately.\n"
            "Select it explicitly with vane.use_native_media_runtime before loading any media extension.\n"
        )
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument(
        "--replacement", action="append", required=True, help="Original SONAME=/path/to/rebuilt/library"
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    replacements = {}
    for value in arguments.replacement:
        name, separator, path = value.partition("=")
        if not separator or name in replacements:
            parser.error("each replacement must name a unique SONAME and source path")
        replacements[name] = Path(path)
    prepare_local(arguments.runtime, replacements, arguments.output)

#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build a private runtime fixture from its exported source SDK, plus a SoXR replacement."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vane_packaging.media_rebuild import rebuild_soxr
from vane_packaging.media_runtime import read_runtime_wheel, verify_runtime_source
from vane_packaging.media_sources import export_sdist


def build(vcpkg: Path, directory: Path, platform: str) -> None:
    project = ROOT / "packages/vane-media-runtime"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((project / "vcpkg.json").read_bytes())
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=vcpkg, text=True).strip()
    if actual != manifest["builtin-baseline"]:
        raise ValueError("fixture vcpkg checkout must use the media project's pinned baseline")
    environment = dict(os.environ)
    environment.setdefault("VCPKG_MAX_CONCURRENCY", "2")
    # Binary caches do not contain the upstream archives required by the source
    # SDK, including archives for host helper ports such as vcpkg-make.
    environment["VCPKG_BINARY_SOURCES"] = "clear"
    subprocess.run(
        [
            str(vcpkg / "vcpkg"),
            "install",
            "--triplet=x64-linux-vane-media",
            f"--x-manifest-root={project}",
            f"--overlay-triplets={project / 'triplets'}",
            f"--x-install-root={directory / 'vcpkg/installed'}",
            f"--x-buildtrees-root={directory / 'vcpkg/buildtrees'}",
            f"--x-packages-root={directory / 'vcpkg/packages'}",
        ],
        check=True,
        env=environment,
    )
    archive = export_sdist(
        project,
        vcpkg,
        directory / "vcpkg/installed",
        Path(environment.get("VCPKG_DOWNLOADS", str(vcpkg / "downloads"))),
        directory / "dist",
    )
    source_root = directory / "source"
    source_root.mkdir(exist_ok=True)
    with tarfile.open(archive) as stream:
        stream.extractall(source_root, filter="data")
    source_project = source_root / archive.name[:-7]
    settings = {
        "platform-tag": platform,
        "source-archive": str(archive),
        "test-only": "true",
        "signing-key": str(ROOT / "external/duckdb/test/mbedtls/private.pem"),
    }
    retained_sdk = directory / "sdk"
    local_source = directory / "local-soxr-source"
    # The backend removes its verified build tree after packaging. This fixture
    # explicitly retains the SDK and SoXR sources needed by subsequent checks.
    program = """
import json
import shutil
import sys

sys.path.insert(0, sys.argv[1])
import backend

build_sdk = backend._build_sdk

def retain_fixture_sdk(project):
    prefix = build_sdk(project)
    shutil.copytree(prefix, sys.argv[4], symlinks=True)
    sources = list((project / 'build/buildtrees/soxr/src').glob('*.clean'))
    if len(sources) != 1:
        raise ValueError('expected one rebuilt SoXR source tree')
    shutil.copytree(sources[0], sys.argv[5], symlinks=True)
    return prefix

backend._build_sdk = retain_fixture_sdk
print(backend.build_wheel(sys.argv[2], json.loads(sys.argv[3])))
"""
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            program,
            str(source_project),
            str(directory / "dist"),
            json.dumps(settings),
            str(retained_sdk),
            str(local_source),
        ],
        cwd=source_project,
        env=environment,
        check=True,
    )
    wheel = directory / "dist" / f"{archive.name[:-7]}-py3-none-{platform}.whl"
    _, runtime_manifest, _, _, _ = read_runtime_wheel(wheel, test_only=True)
    verify_runtime_source(archive, runtime_manifest)
    with zipfile.ZipFile(wheel) as stream:
        for name in stream.namelist():
            if not name.startswith("vane_media_runtime/"):
                continue
            destination = directory / "runtime" / name.removeprefix("vane_media_runtime/")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(stream.read(name))
    rebuild_soxr(local_source, directory / "runtime", directory, jobs=int(environment["VCPKG_MAX_CONCURRENCY"]))
    paths = {
        "runtime_wheel": str(wheel),
        "source_archive": str(archive),
        "sdk": str(retained_sdk),
        "runtime": str(directory / "runtime"),
        "local_runtime": str(directory / "local-runtime"),
    }
    (directory / "paths.json").write_text(json.dumps(paths, indent=2) + "\n")
    print(json.dumps(paths))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vcpkg-root", type=Path, required=True)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--platform-tag", required=True)
    arguments = parser.parse_args()
    build(
        arguments.vcpkg_root.resolve(strict=True),
        arguments.build_directory.resolve(),
        arguments.platform_tag,
    )

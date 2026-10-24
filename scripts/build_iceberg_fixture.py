#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build the source-pinned, test-only Iceberg/Avro qualification inputs."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "packages/vane-iceberg-provider"
PATCHES = (
    "iceberg-vane-local.patch",
    "iceberg-vane-frozen-scan.patch",
    "iceberg-vane-file-scan.patch",
    "iceberg-vane-partitions.patch",
    "iceberg-vane-transforms.patch",
    "iceberg-vane-schema.patch",
    "iceberg-vane-deletes.patch",
    "iceberg-vane-s3.patch",
)


def source_profile():
    sources = {}
    for name in ("avro", "iceberg", "httpfs"):
        config = (ROOT / f"external/duckdb/.github/config/extensions/{name}.cmake").read_text()
        url = re.search(r"GIT_URL\s+(https://github.com/[^\s)]+)", config)
        revision = re.search(r"GIT_TAG\s+([0-9a-f]{40})\b", config)
        if not url or not revision:
            raise ValueError(f"{name} must have an immutable source pin")
        sources[name] = {"url": url[1], "revision": revision[1]}
    return {
        "sources": sources,
        "patches": [
            {"name": name, "sha256": hashlib.sha256((PROJECT / "patches" / name).read_bytes()).hexdigest()}
            for name in PATCHES
        ],
        "sdk_baseline": json.loads((PROJECT / "vcpkg.json").read_text())["builtin-baseline"],
    }


def checkout_source(destination, url, revision, run):
    # Never reset or patch a caller's source tree. These are fresh owned clones.
    if destination.exists():
        raise ValueError(f"source destination already exists: {destination}")
    run(["git", "init", str(destination)])
    run(["git", "-C", str(destination), "fetch", "--depth=1", url, revision])
    run(["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"])
    actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise ValueError("source checkout does not match its pinned revision")


def build_inputs(directory, args, run, environment):
    """Reuse the normal base build, optional SDK and provider packaging tools."""
    profile = source_profile()
    vcpkg = args.vcpkg_root.resolve()
    if vcpkg.exists():
        actual = subprocess.check_output(["git", "-C", str(vcpkg), "rev-parse", "HEAD"], text=True).strip()
        if actual != profile["sdk_baseline"]:
            raise ValueError("existing vcpkg checkout must match the SDK baseline")
    sources = directory / "sources"
    sources.mkdir()
    for name, pin in profile["sources"].items():
        checkout_source(sources / name, pin["url"], pin["revision"], run)
    for patch in PATCHES:
        command = ["git", "-C", str(sources / "iceberg"), "apply"]
        path = str(PROJECT / "patches" / patch)
        run([*command, "--check", path])
        run([*command, path])

    base_sdk = (args.base_sdk or directory / "base-sdk").resolve()
    optional_sdk = (args.iceberg_sdk or directory / "iceberg-sdk").resolve()
    triplet = args.triplet
    environment.update(
        VCPKG_ROOT=str(vcpkg),
        VCPKG_INSTALLED_DIR=str(base_sdk),
        VCPKG_TARGET_TRIPLET=triplet,
        VCPKG_HOST_TRIPLET=triplet,
        VCPKG_MAX_CONCURRENCY=str(args.jobs),
        CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs),
    )
    if args.base_sdk is None:
        run(["bash", str(ROOT / "scripts/bootstrap_vcpkg.sh")])
    elif not (vcpkg / "vcpkg").is_file():
        raise ValueError("an existing base SDK requires a bootstrapped vcpkg checkout")
    if args.iceberg_sdk is None:
        run(
            [
                str(vcpkg / "vcpkg"),
                "install",
                f"--x-manifest-root={PROJECT}",
                f"--x-install-root={optional_sdk}",
                f"--triplet={triplet}",
                f"--host-triplet={triplet}",
                "--clean-buildtrees-after-build",
                "--clean-packages-after-build",
            ]
        )
    for project, prefix, notice in (
        (ROOT, base_sdk, "vcpkg-binary-dependencies.txt"),
        (PROJECT, optional_sdk, "iceberg-native-dependencies.txt"),
    ):
        share = prefix / triplet / "share"
        run(
            [
                sys.executable,
                "-I",
                str(ROOT / "scripts/check_copyleft.py"),
                "--manifest",
                str(project / "vcpkg.json"),
                "--share-dir",
                str(share),
            ]
        )
        run(
            [
                sys.executable,
                "-I",
                str(ROOT / "scripts/sync_vcpkg_licenses.py"),
                "--share-dir",
                str(share),
                "--output",
                str(ROOT / "LICENSES" / notice),
                "--check",
            ]
        )
    native = (args.native_build_directory or directory / "native").resolve()
    environment.update(
        SKBUILD_BUILD_DIR=str(native),
        SKBUILD_CMAKE_BUILD_TYPE="Release",
        CMAKE_PREFIX_PATH=str(optional_sdk / triplet),
        **{f"DUCKDB_{name.upper()}_DIRECTORY": str(sources / name) for name in profile["sources"]},
    )
    # The recipe owns these settings; inherited CMAKE_ARGS must not override them.
    environment.pop("CMAKE_ARGS", None)
    base_dist = directory / "base-dist"
    run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(base_dist),
            "-Ccmake.define.VANE_LOADABLE_EXTENSIONS=avro;iceberg",
            "-Ccmake.define.VANE_ENABLE_TEST_EXTENSION_SIGNING_KEY=ON",
            str(ROOT),
        ]
    )
    wheels = list(base_dist.glob("vane_ai-*.whl"))
    if len(wheels) != 1:
        raise ValueError("build must produce exactly one base wheel")
    inputs = {"base_wheel": str(wheels[0])}
    for name in ("avro", "iceberg"):
        path = native / f"duckdb/extension/{name}/{name}.duckdb_extension"
        if not path.is_file():
            raise ValueError(f"native build did not produce {name}")
        inputs[name + "_artifact"] = str(path)
    provenance = {
        "profile": profile,
        "inputs": {
            key: {"path": value, "sha256": hashlib.sha256(Path(value).read_bytes()).hexdigest()}
            for key, value in inputs.items()
        },
        "triplet": triplet,
    }
    (directory / "build-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return inputs

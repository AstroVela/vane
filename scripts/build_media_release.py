#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare unsigned media inputs, then package independently signed bytes without a key."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vane_packaging.archive_safety import open_zip_snapshot, snapshot_archive
from vane_packaging.artifact_limits import MAX_EXTENSION_ARTIFACT_BYTES
from vane_packaging.media_release import prepare_release
from vane_packaging.media_runtime import read_runtime_wheel, verify_runtime_source
from vane_packaging.media_sources import export_sdist, read_source_archive, read_source_file
from vane_packaging.media_version import runtime_format

PLATFORM = "manylinux_2_28_x86_64"
LICENSE_EXPRESSION = (
    "Apache-2.0 AND MIT AND BSL-1.0 AND LGPL-2.1-or-later AND LGPL-2.1-only "
    "AND LGPL-2.0-or-later AND Zlib AND libtiff AND BSD-3-Clause AND IJG"
)


def one(directory: Path, pattern: str) -> Path:
    paths = list(directory.glob(pattern))
    if len(paths) != 1 or not paths[0].is_file() or paths[0].is_symlink():
        raise ValueError(f"expected one regular {pattern} in {directory}")
    return paths[0]


def prepare_runtime(vcpkg: Path, directory: Path, source_url: str) -> None:
    project = ROOT / "packages/vane-media-runtime"
    directory.mkdir(parents=True, exist_ok=False)
    baseline = json.loads((project / "vcpkg.json").read_bytes())["builtin-baseline"]
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=vcpkg, text=True).strip() != baseline:
        raise ValueError("media release requires the pinned vcpkg checkout")
    environment = dict(os.environ)
    environment["VCPKG_BINARY_SOURCES"] = "clear"
    for name in ("VCPKG_OVERLAY_PORTS", "VCPKG_OVERLAY_TRIPLETS"):
        environment.pop(name, None)
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
        directory / "inputs",
    )
    files = read_source_archive(read_source_file(archive), archive.name)
    extracted = directory / "source"
    for name, contents in files.items():
        path = extracted / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        path.chmod(0o755 if name.endswith(".sh") else 0o644)
    spec = importlib.util.spec_from_file_location("_media_release_backend", project / "backend.py")
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    backend.PROJECT = extracted
    wheel = (
        directory
        / "inputs"
        / backend.prepare_unsigned_wheel(
            str(directory / "inputs"),
            {
                "source-archive": str(archive),
                "platform-tag": PLATFORM,
                "source-url": source_url.rstrip("/") + "/" + archive.name,
            },
            sdk_output=directory / "sdk",
        )
    )
    # The archive was produced in this process. No archive code is imported.
    with zipfile.ZipFile(wheel) as stream:
        for name in stream.namelist():
            if name.startswith("vane_media_runtime/"):
                path = directory / "runtime" / name.removeprefix("vane_media_runtime/")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(stream.read(name))
    signing = directory / "signing"
    signing.mkdir()
    shutil.copyfile(directory / "runtime/runtime-manifest.json", signing / "runtime-manifest.json")
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(ROOT / "scripts/sync_vcpkg_licenses.py"),
            "--share-dir",
            str(directory / "sdk/share"),
            "--output",
            str(directory / "inputs/media-native-dependency-notices.txt"),
        ],
        check=True,
    )


def finish_runtime(unsigned: Path, signature: Path, output: Path) -> Path:
    """Change only the fixed signature member and its RECORD entry; validate both layouts."""
    if not unsigned.name.endswith(".whl.unsigned"):
        raise ValueError("runtime packaging requires a deferred-signing input")
    fmt = runtime_format()
    signature_bytes = fmt.read_file(signature.parent, signature.name, 256)
    if len(signature_bytes) != 256 or signature_bytes == bytes(256):
        raise ValueError("runtime packaging requires an independent RSA-2048 signature")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / unsigned.name.removesuffix(".unsigned")
    if destination.exists():
        raise ValueError("refusing to overwrite a runtime wheel")
    with tempfile.TemporaryDirectory(prefix="vane-runtime-package-") as value:
        temporary = Path(value) / destination.name
        with snapshot_archive(
            unsigned, max_bytes=100 * 1024 * 1024, description="unsigned runtime", size_limit_description="100 MiB"
        ) as snapshot:
            # The regular release reader verifies layout, RECORD, ELF policy and
            # metadata before we change anything; native signature verification
            # is deliberately deferred until the clean installation below.
            with temporary.open("xb") as stream:
                shutil.copyfileobj(snapshot.file, stream)
            read_runtime_wheel(temporary)
            snapshot.file.seek(0)
            with open_zip_snapshot(snapshot, max_members=512, description="unsigned runtime") as stream:
                files = {name: stream.read(name) for name in stream.namelist()}
        member = f"{fmt.PACKAGE}/{fmt.SIGNATURE}"
        if files[member] != bytes(256):
            raise ValueError("runtime input already contains a signature")
        files[member] = signature_bytes
        record_name = next(name for name in files if name.endswith(".dist-info/RECORD"))
        record = io.StringIO(newline="")
        writer = csv.writer(record, lineterminator="\n")
        for name, contents in sorted(files.items()):
            if name != record_name:
                checksum = base64.urlsafe_b64encode(hashlib.sha256(contents).digest()).rstrip(b"=").decode()
                writer.writerow((name, "sha256=" + checksum, len(contents)))
        writer.writerow((record_name, "", ""))
        files[record_name] = record.getvalue().encode()
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as stream:
            for name, contents in sorted(files.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                stream.writestr(info, contents)
        read_runtime_wheel(temporary)
        with destination.open("xb") as stream, temporary.open("rb") as source:
            shutil.copyfileobj(source, stream)
    return destination


def package(inputs: Path, unsigned: Path, signed: Path, base: Path, output: Path) -> str:
    fmt = runtime_format()
    original = fmt.read_file(unsigned, "native_media.duckdb_extension", MAX_EXTENSION_ARTIFACT_BYTES)
    artifact = fmt.read_file(signed, "native_media.duckdb_extension", MAX_EXTENSION_ARTIFACT_BYTES)
    if original[-256:] != bytes(256) or artifact[:-256] != original[:-256] or artifact[-256:] == bytes(256):
        raise ValueError("extension packaging may change only the empty native signature slot")
    with tempfile.TemporaryDirectory(prefix="vane-media-package-") as value:
        temporary = Path(value)
        runtime = finish_runtime(one(inputs, "*.whl.unsigned"), signed / fmt.SIGNATURE, temporary)
        source = one(inputs, "*.tar.gz")
        _, manifest, _, document, _ = read_runtime_wheel(runtime)
        if document != fmt.read_file(unsigned, fmt.MANIFEST, fmt.MAX_MANIFEST_BYTES):
            raise ValueError("packaged runtime differs from its signed manifest")
        verify_runtime_source(source, manifest)
        # Run the wheel builder in isolated mode so it imports the installed base
        # Vane runtime, not the source checkout's vane package.
        command = [
            sys.executable,
            "-I",
            str(ROOT / "scripts/build_extension_wheel.py"),
            "--artifact",
            str(signed / "native_media.duckdb_extension"),
            "--extension-name",
            "native_media",
            "--platform-tag",
            PLATFORM,
            "--trust-identity",
            "astrovela/vane",
            "--license-expression",
            LICENSE_EXPRESSION,
            "--runtime-wheel",
            str(runtime),
            "--runtime-source",
            str(source),
            "--output-directory",
            str(temporary),
        ]
        for path in [
            ROOT / "LICENSE",
            ROOT / "NOTICE",
            ROOT / "LICENSES/DuckDB-MIT.txt",
            ROOT / "LICENSES/Bison-parser-notice.txt",
            ROOT / "LICENSES/vcpkg-binary-dependencies.txt",
            inputs / "media-native-dependency-notices.txt",
        ]:
            command.extend(("--license-file", str(path)))
        subprocess.run(command, check=True)
        return prepare_release(
            base=base,
            provider=one(temporary, "vane_extension_native_media-*.whl"),
            runtime=runtime,
            source=source,
            trust_identity="astrovela/vane",
            output=output,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-runtime")
    prepare.add_argument("--vcpkg", type=Path, required=True)
    prepare.add_argument("--directory", type=Path, required=True)
    prepare.add_argument("--source-url", required=True)
    assemble = commands.add_parser("package")
    for name in ("inputs", "unsigned", "signed", "base", "output"):
        assemble.add_argument("--" + name, type=Path, required=True)
    arguments = vars(parser.parse_args())
    command = arguments.pop("command")
    for key, value in arguments.items():
        if isinstance(value, Path):
            arguments[key] = value.resolve()
    if command == "prepare-runtime":
        prepare_runtime(**arguments)
    else:
        print(json.dumps({"manifest_sha256": package(**arguments)}, sort_keys=True))


if __name__ == "__main__":
    main()

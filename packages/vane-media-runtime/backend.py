# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""PEP 517 backend for the independent media source SDK and runtime wheel."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
REPOSITORY = PROJECT if (PROJECT / "source-inventory.json").exists() else PROJECT.parents[1]
sys.path.insert(0, str(REPOSITORY))


def _setting(settings, name, default=None):
    value = (settings or {}).get(name, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"media runtime build requires -C{name}=<value>")
    return value


def build_sdist(sdist_directory, config_settings=None):
    from vane_packaging.media_sources import export_sdist

    return export_sdist(
        PROJECT,
        Path(_setting(config_settings, "vcpkg-root", os.environ.get("VCPKG_ROOT"))),
        Path(_setting(config_settings, "installed-root")),
        Path(_setting(config_settings, "downloads")),
        Path(sdist_directory),
    ).name


def _format():
    path = PROJECT / "_native_runtime_format.py"
    if not path.exists():
        path = REPOSITORY / "vane/_native_runtime_format.py"
    spec = importlib.util.spec_from_file_location("_native_runtime_format", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_sdk(project):
    if not (project / "source-inventory.json").is_file():
        raise ValueError("build runtime wheels from the exported source distribution")
    sdk = project / "sdk"
    build = PROJECT / "build"
    if build.exists():
        raise ValueError("runtime source rebuild requires a fresh SDK extraction without a build directory")
    build.mkdir()
    vcpkg = sdk / "vcpkg/vcpkg"
    (vcpkg.parent / "ports").mkdir(exist_ok=True)
    environment = dict(os.environ)
    for name in ("VCPKG_OVERLAY_PORTS", "VCPKG_OVERLAY_TRIPLETS"):
        environment.pop(name, None)
    environment["VCPKG_ROOT"] = str(vcpkg.parent)
    environment["VCPKG_DOWNLOADS"] = str(sdk / "downloads")
    # A source rebuild must compile the selected sources, including user edits.
    environment["VCPKG_BINARY_SOURCES"] = "clear"
    subprocess.run(
        ["bash", str(vcpkg.parent / "bootstrap-vcpkg.sh"), "-disableMetrics"],
        check=True,
        env=environment,
        cwd=project,
    )
    subprocess.run(
        [
            str(vcpkg),
            "install",
            "--triplet=x64-linux-vane-media",
            f"--x-manifest-root={sdk / 'manifest'}",
            f"--overlay-ports={sdk / 'ports'}",
            f"--overlay-triplets={project / 'triplets'}",
            f"--x-install-root={build / 'installed'}",
            f"--x-buildtrees-root={build / 'buildtrees'}",
            f"--x-packages-root={build / 'packages'}",
        ],
        check=True,
        env=environment,
        cwd=project,
    )
    return build / "installed/x64-linux-vane-media"


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    from vane_packaging.media_sources import read_source_archive, read_source_file
    from vane_packaging.media_version import source_version

    settings = config_settings or {}
    source = Path(_setting(settings, "source-archive")).resolve(strict=True)
    source_contents = read_source_file(source)
    identity = source_version(PROJECT)
    release = identity["version"]
    if source.name != f"vane_media_runtime-{release}.tar.gz":
        raise ValueError("runtime wheel requires its exact version's source archive")
    # A local development SDK is an explicit opt-in and produces a private wheel.
    test_only = settings.get("test-only") == "true"
    if identity["git_dirty"] and not test_only:
        raise ValueError("release runtime wheels require a source SDK exported from a clean Git commit")
    files = read_source_archive(source_contents, source.name)
    for name, contents in files.items():
        local = PROJECT / name
        if local.is_symlink() or not local.is_file() or local.read_bytes() != contents:
            raise ValueError(f"runtime build source differs from its published archive: {name}")
    with tempfile.TemporaryDirectory(prefix="vane-media-source-") as temporary:
        project = Path(temporary)
        for name, contents in files.items():
            destination = project / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(contents)
            destination.chmod(0o755 if name.endswith(".sh") else 0o644)
        return _build_wheel(wheel_directory, settings, source, source_contents, identity, project)


def _build_wheel(wheel_directory, settings, source, source_contents, identity, project):
    from elftools.elf.elffile import ELFFile

    from vane_packaging.media_runtime import (
        PROJECT_NOTICES,
        runtime_license_expression,
        stage_libraries,
        validate_library_graph,
    )

    fmt = _format()
    platform = _setting(settings, "platform-tag")
    release = identity["version"]
    test_only = settings.get("test-only") == "true"
    if "sdk-prefix" in settings:
        if not test_only:
            raise ValueError("official runtime wheels must rebuild from their source distribution")
        prefix = Path(_setting(settings, "sdk-prefix"))
    else:
        prefix = _build_sdk(project)
    components = json.loads((project / "components.json").read_bytes())
    notices = {}
    for name, checksum in PROJECT_NOTICES.items():
        contents = (project / name).read_bytes()
        if hashlib.sha256(contents).hexdigest() != checksum:
            raise ValueError(f"unreviewed runtime project license/notice: {name}")
        notices[name] = contents
    owners = {}
    for component, record in components.items():
        notice = (prefix / "share" / component / "copyright").read_bytes()
        if hashlib.sha256(notice).hexdigest() != record["notice_sha256"]:
            raise ValueError(f"unreviewed media license notice: {component}")
        notices[f"{component}.txt"] = notice
        document = json.loads((prefix / "share" / component / "vcpkg.spdx.json").read_bytes())
        actual_version = next(p["versionInfo"] for p in document["packages"] if p["SPDXID"] == "SPDXRef-port")
        if actual_version != record["version"]:
            raise ValueError(f"unreviewed media component version: {component}")
        for item in document["files"]:
            name = item["fileName"]
            if not name.startswith("./lib/") or ".so" not in name:
                continue
            library = prefix / name
            if not library.is_file():
                continue
            with library.open("rb") as stream:
                dynamic = ELFFile(stream).get_section_by_name(".dynamic")
                for tag in dynamic.iter_tags():
                    if tag.entry.d_tag == "DT_SONAME":
                        owners[tag.soname] = component
    namespace = fmt.runtime_namespace(identity)
    license_expression = runtime_license_expression(components)
    output = Path(wheel_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vane-media-wheel-") as temporary:
        stage = Path(temporary)
        mapping = stage_libraries(prefix, stage / ".libs", namespace=namespace, platform=platform)
        libraries = {p.name: p.read_bytes() for p in (stage / ".libs").iterdir()}
        needed = validate_library_graph(libraries, platform)
        reverse = {new: old for old, new in mapping.items()}
        records = {
            name: {
                "sha256": hashlib.sha256(value).hexdigest(),
                "size": len(value),
                "needed": list(needed[name]),
                "component": owners[reverse[name]],
            }
            for name, value in libraries.items()
        }
        manifest = fmt.canonical_json(
            {
                "schema_version": 1,
                "distribution": fmt.DISTRIBUTION,
                "version": release,
                **fmt.git_provenance(identity),
                "platform": platform,
                "namespace": namespace,
                "license_expression": license_expression,
                "source": {
                    "filename": source.name,
                    "sha256": hashlib.sha256(source_contents).hexdigest(),
                    "url": _setting(
                        settings,
                        "source-url",
                        f"https://pypi.org/project/{fmt.DISTRIBUTION}/{release}/#files",
                    ),
                },
                "components": components,
                "files": records,
            }
        )
        fmt.parse_manifest(manifest)
        signing_key = Path(_setting(settings, "signing-key")).resolve(strict=True)
        digest_path = stage / "digest"
        signature_path = stage / "signature"
        digest_path.write_bytes(hashlib.sha256(fmt.SIGNING_DOMAIN + manifest).digest())
        subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-sign",
                "-inkey",
                str(signing_key),
                "-in",
                str(digest_path),
                "-out",
                str(signature_path),
                "-pkeyopt",
                "digest:sha256",
            ],
            check=True,
        )
        signature = signature_path.read_bytes()
        if len(signature) != 256:
            raise ValueError("runtime manifest signing requires an RSA-2048 key")
        dist_info = f"vane_media_runtime-{release}.dist-info"
        tag = f"py3-none-{platform}"
        metadata = f"Metadata-Version: 2.4\nName: {fmt.DISTRIBUTION}\nVersion: {release}\nSummary: Shared native media libraries for Vane extensions\nRequires-Python: >=3.10,<3.15\nLicense-Expression: {license_expression}\n"
        if test_only:
            metadata += "Classifier: Private :: Do Not Upload\n"
        files = {f"vane_media_runtime/.libs/{name}": value for name, value in libraries.items()}
        for name, notice in notices.items():
            metadata += f"License-File: {name}\n"
            files[f"{dist_info}/licenses/{name}"] = notice
        files.update(
            {
                "vane_media_runtime/__init__.py": (project / "vane_media_runtime/__init__.py").read_bytes(),
                f"vane_media_runtime/{fmt.MANIFEST}": manifest,
                f"vane_media_runtime/{fmt.SIGNATURE}": signature,
                f"{dist_info}/METADATA": (metadata + "\n").encode(),
                f"{dist_info}/WHEEL": f"Wheel-Version: 1.0\nGenerator: vane-media-runtime\nRoot-Is-Purelib: false\nTag: {tag}\n\n".encode(),
            }
        )
        record = io.StringIO(newline="")
        writer = csv.writer(record, lineterminator="\n")
        for name, value in sorted(files.items()):
            digest = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b"=").decode()
            writer.writerow((name, f"sha256={digest}", len(value)))
        writer.writerow((f"{dist_info}/RECORD", "", ""))
        files[f"{dist_info}/RECORD"] = record.getvalue().encode()
        wheel_name = f"vane_media_runtime-{release}-{tag}.whl"
        with zipfile.ZipFile(output / wheel_name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, value in sorted(files.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, value)
    return wheel_name

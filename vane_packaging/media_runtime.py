# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Inspect and relocate media ELF libraries without executing their constructors."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import re
import shutil
import struct
import subprocess
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

from elftools.elf.elffile import ELFFile

from vane_packaging.archive_safety import open_zip_snapshot, snapshot_archive
from vane_packaging.extension_wheel import _ELF_HEADER_64, _parse_elf_dynamic_linkage, _validate_linux_elf_platform
from vane_packaging.manylinux_policy import manylinux_policy

# The project license shipped by the standalone runtime, separate from codec
# notices. Update only when the repository's reviewed LICENSE bytes change.
PROJECT_LICENSE_FILE = "LICENSE"
PROJECT_LICENSE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
PROJECT_NOTICES = {
    PROJECT_LICENSE_FILE: PROJECT_LICENSE_SHA256,
    "NOTICE": "aaf87a040547c1e9df715bc84d4d19474821d460f204a577eb83b944ab2c74d6",
}


def runtime_license_expression(components) -> str:
    from packaging.licenses import canonicalize_license_expression

    return canonicalize_license_expression(
        "Apache-2.0 AND " + " AND ".join(f"({record['license']})" for _, record in sorted(components.items()))
    )


def read_runtime_wheel(path: Path, *, test_only: bool = False):
    """Validate an exact owned runtime wheel without importing its provider code."""
    with snapshot_archive(
        path,
        max_bytes=100 * 1024 * 1024,
        description="runtime wheel",
        size_limit_description="the 100 MiB runtime publication limit",
    ) as snapshot:
        return _read_runtime_wheel_snapshot(snapshot, test_only=test_only)


def _read_runtime_wheel_snapshot(snapshot, *, test_only):
    from packaging.utils import parse_wheel_filename

    from vane_packaging.extension_wheel import _validate_wheel_record
    from vane_packaging.media_version import identity_version

    format_path = Path(__file__).resolve().parents[1] / "vane/_native_runtime_format.py"
    if not format_path.is_file():
        format_path = Path(__file__).resolve().parents[1] / "_native_runtime_format.py"
    spec = importlib.util.spec_from_file_location("_native_runtime_format", format_path)
    fmt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fmt)
    path = snapshot.source_path
    distribution, version, build, tags = parse_wheel_filename(path.name)
    if distribution != fmt.DISTRIBUTION or build or len(tags) != 1:
        raise ValueError("invalid native media runtime wheel filename")
    tag = next(iter(tags))
    if tag.interpreter != "py3" or tag.abi != "none":
        raise ValueError("media runtime must use a Python-independent native-library wheel tag")
    _policy(tag.platform)
    with open_zip_snapshot(snapshot, max_members=512, description="runtime wheel") as wheel:
        entries = wheel.infolist()
        if not 1 <= len(entries) <= 512 or sum(item.file_size for item in entries) > fmt.MAX_TOTAL_BYTES:
            raise ValueError("runtime wheel exceeds its archive bounds")
        if any(item.file_size > fmt.MAX_FILE_BYTES for item in entries):
            raise ValueError("runtime wheel member exceeds its size bound")
        for item in entries:
            limit = fmt.MAX_FILE_BYTES if item.filename.startswith(f"{fmt.PACKAGE}/.libs/") else 1024 * 1024
            if item.filename.endswith(("/METADATA", f"/{fmt.MANIFEST}")):
                limit = fmt.MAX_MANIFEST_BYTES
            elif item.filename.endswith("/__init__.py"):
                limit = 4096
            elif item.filename == f"{fmt.PACKAGE}/{fmt.SIGNATURE}":
                limit = 256
            if item.file_size > limit:
                raise ValueError("runtime wheel metadata member exceeds its size bound")
        names = wheel.namelist()
        if len(set(names)) != len(names) or len({name.casefold() for name in names}) != len(names):
            raise ValueError("runtime wheel contains colliding members")
        document = wheel.read(f"{fmt.PACKAGE}/{fmt.MANIFEST}")
        manifest = fmt.parse_manifest(document)
        if manifest["license_expression"] != runtime_license_expression(manifest["components"]):
            raise ValueError("runtime license expression must cover the project and native components")
        if manifest["version"] != identity_version(manifest):
            raise ValueError("runtime wheel version differs from its source identity")
        if manifest["git_dirty"] and not test_only:
            raise ValueError("development runtime Git snapshots cannot be released")
        if manifest["version"] != str(version) or manifest["platform"] != tag.platform:
            raise ValueError("runtime wheel filename differs from its manifest")
        signature = wheel.read(f"{fmt.PACKAGE}/{fmt.SIGNATURE}")
        if len(signature) != 256:
            raise ValueError("invalid media runtime manifest signature length")
        info = f"vane_media_runtime-{version}.dist-info"
        expected = {
            f"{fmt.PACKAGE}/__init__.py",
            f"{fmt.PACKAGE}/{fmt.MANIFEST}",
            f"{fmt.PACKAGE}/{fmt.SIGNATURE}",
            f"{info}/METADATA",
            f"{info}/WHEEL",
            f"{info}/RECORD",
            *(f"{info}/licenses/{name}" for name in PROJECT_NOTICES),
            *(f"{fmt.PACKAGE}/.libs/{name}" for name in manifest["files"]),
            *(f"{info}/licenses/{name}.txt" for name in manifest["components"]),
        }
        if set(names) != expected:
            raise ValueError("runtime wheel contains missing or unowned members")
        initializer = wheel.read(f"{fmt.PACKAGE}/__init__.py")
        if len(initializer) > 4096:
            raise ValueError("runtime package initializer exceeds its bound")
        body = ast.parse(initializer).body
        if (
            len(body) != 1
            or not isinstance(body[0], ast.Expr)
            or not isinstance(body[0].value, ast.Constant)
            or not isinstance(body[0].value.value, str)
        ):
            raise ValueError("runtime package initializer must contain only an inert docstring")
        metadata = BytesParser(policy=default).parsebytes(wheel.read(f"{info}/METADATA"))
        for key, expected_value in {
            "Metadata-Version": "2.4",
            "Name": fmt.DISTRIBUTION,
            "Version": str(version),
            "Requires-Python": ">=3.10,<3.15",
            "License-Expression": manifest["license_expression"],
        }.items():
            if metadata.get_all(key) != [expected_value]:
                raise ValueError(f"runtime wheel has invalid {key}")
        if metadata.get_all("Requires-Dist"):
            raise ValueError("runtime wheel cannot introduce Python dependencies")
        if not test_only and any(value.startswith("Private ::") for value in metadata.get_all("Classifier", [])):
            raise ValueError("test-only runtime wheels cannot be released")
        license_files = [*PROJECT_NOTICES, *(f"{name}.txt" for name in manifest["components"])]
        if sorted(metadata.get_all("License-File", [])) != sorted(license_files):
            raise ValueError("runtime wheel license metadata differs from its manifest")
        expected_wheel = (
            f"Wheel-Version: 1.0\nGenerator: vane-media-runtime\nRoot-Is-Purelib: false\nTag: {tag}\n\n".encode()
        )
        if wheel.read(f"{info}/WHEEL") != expected_wheel:
            raise ValueError("runtime WHEEL metadata differs from its filename")
        for name, checksum in PROJECT_NOTICES.items():
            if hashlib.sha256(wheel.read(f"{info}/licenses/{name}")).hexdigest() != checksum:
                raise ValueError(f"runtime wheel project license/notice digest differs: {name}")
        for name, record in manifest["components"].items():
            if hashlib.sha256(wheel.read(f"{info}/licenses/{name}.txt")).hexdigest() != record["notice_sha256"]:
                raise ValueError("runtime wheel license notice digest differs")
        libraries = {}
        for name, record in manifest["files"].items():
            contents = wheel.read(f"{fmt.PACKAGE}/.libs/{name}")
            if len(contents) != record["size"] or hashlib.sha256(contents).hexdigest() != record["sha256"]:
                raise ValueError(f"runtime library digest differs: {name}")
            libraries[name] = contents
        graph = validate_library_graph(libraries, tag.platform)
        for name, needed in graph.items():
            if list(needed) != manifest["files"][name]["needed"]:
                raise ValueError(f"runtime manifest dependency graph differs from ELF metadata: {name}")
        _validate_wheel_record(wheel, names=names, record_name=f"{info}/RECORD")
    return fmt.reference(document), manifest, libraries, document, signature


def _policy(platform: str):
    match = re.fullmatch(r"manylinux_(0|[1-9][0-9]*)_(0|[1-9][0-9]*)_x86_64", platform)
    if match is None:
        raise ValueError("media runtime supports canonical manylinux x86-64 tags only")
    return manylinux_policy((int(match[1]), int(match[2])), "x86_64")


def exported_versions(contents: bytes) -> frozenset[str]:
    """Read bounded GNU version definitions without materializing ELF sections."""
    section_header = struct.Struct("<IIQQQQIIQQ")
    definition_header = struct.Struct("<HHHHIII")
    auxiliary_header = struct.Struct("<II")
    if len(contents) < _ELF_HEADER_64.size or contents[:7] != b"\x7fELF\x02\x01\x01":
        raise ValueError("invalid runtime ELF version header")
    fields = _ELF_HEADER_64.unpack_from(contents)
    section_offset, section_size, section_count = fields[6], fields[11], fields[12]
    if section_offset == 0 and section_count == 0:
        return frozenset()
    if (
        not 0 < section_count <= 4096
        or section_size != section_header.size
        or section_offset < _ELF_HEADER_64.size
        or section_offset + section_count * section_size > len(contents)
    ):
        raise ValueError("runtime ELF section table exceeds its bounds")
    sections = [
        section_header.unpack_from(contents, section_offset + index * section_size) for index in range(section_count)
    ]
    definitions = [section for section in sections if section[1] == 0x6FFFFFFD]
    if not definitions:
        return frozenset()
    if len(definitions) != 1:
        raise ValueError("runtime ELF has duplicate version-definition sections")
    section = definitions[0]
    offset, size, link, count = section[4:8]
    if not 0 < count <= 4096 or offset + size > len(contents) or link >= section_count:
        raise ValueError("runtime ELF version definitions exceed their bounds")
    strings = sections[link]
    string_offset, string_size = strings[4:6]
    if strings[1] != 3 or string_offset + string_size > len(contents):
        raise ValueError("runtime ELF version string table exceeds its bounds")
    result = set()
    remaining = 4096
    end = offset + size
    for index in range(count):
        if offset + definition_header.size > end:
            raise ValueError("truncated runtime ELF version definition")
        version, _flags, _index, auxiliary_count, _hash, auxiliary_offset, next_offset = definition_header.unpack_from(
            contents, offset
        )
        if version != 1 or not 0 < auxiliary_count <= remaining:
            raise ValueError("runtime ELF version auxiliary count exceeds its bound")
        remaining -= auxiliary_count
        record_end = offset + next_offset if index < count - 1 else end
        if (
            (index < count - 1 and (next_offset < definition_header.size or record_end > end))
            or (index == count - 1 and next_offset != 0)
            or auxiliary_offset < definition_header.size
        ):
            raise ValueError("invalid runtime ELF version-definition chain")
        auxiliary = offset + auxiliary_offset
        for auxiliary_index in range(auxiliary_count):
            if auxiliary + auxiliary_header.size > record_end:
                raise ValueError("truncated runtime ELF version auxiliary")
            name_offset, next_auxiliary = auxiliary_header.unpack_from(contents, auxiliary)
            if name_offset >= string_size:
                raise ValueError("runtime ELF version name exceeds its string table")
            start = string_offset + name_offset
            stop = contents.find(b"\0", start, min(start + 129, string_offset + string_size))
            if stop <= start:
                raise ValueError("runtime ELF version name exceeds its bound")
            try:
                result.add(contents[start:stop].decode("ascii"))
            except UnicodeError as exception:
                raise ValueError("invalid runtime ELF version name") from exception
            if (auxiliary_index < auxiliary_count - 1 and next_auxiliary < auxiliary_header.size) or (
                auxiliary_index == auxiliary_count - 1 and next_auxiliary != 0
            ):
                raise ValueError("invalid runtime ELF version-auxiliary chain")
            auxiliary += next_auxiliary
        offset = record_end
    return frozenset(result)


def validate_library_graph(libraries: dict[str, bytes], platform: str) -> dict[str, tuple[str, ...]]:
    """Check every object, including recursive dependencies and symbol versions."""
    policy = _policy(platform)
    if not 1 <= len(libraries) <= 128 or sum(map(len, libraries.values())) > 512 * 1024 * 1024:
        raise ValueError("runtime library graph exceeds its bound")
    if set(libraries) & policy.external_libraries:
        raise ValueError("runtime must not replace platform system libraries")
    linkage = {}
    for name, contents in libraries.items():
        if len(contents) > 128 * 1024 * 1024:
            raise ValueError("runtime library exceeds its size bound")
        value = _parse_elf_dynamic_linkage(contents, description=name, allowed_runpath="$ORIGIN")
        if value.soname != name or value.filters or value.auxiliaries:
            raise ValueError(f"runtime library has an unexpected SONAME or loader filter: {name}")
        linkage[name] = value
    versions = {name: exported_versions(contents) for name, contents in libraries.items()}
    for name, contents in libraries.items():
        _validate_linux_elf_platform(
            contents, platform, description=name, bundled_versions=versions, allowed_runpath="$ORIGIN"
        )
    return {name: value.needed for name, value in linkage.items()}


def stage_libraries(
    prefix: Path, destination: Path, *, namespace: str, platform: str, patchelf: str = "patchelf"
) -> dict[str, str]:
    """Copy and namespace a complete vcpkg media installation; never modify its SDK.

    Names depend on the release namespace, not the final file digest, allowing a
    user to rebuild a compatible library under the same name. Hashes are computed
    only after relocation by the manifest writer.
    """
    if re.fullmatch(r"vane_media_[a-z0-9_]{1,64}", namespace) is None:
        raise ValueError("invalid media runtime SONAME namespace")
    policy = _policy(platform)
    sources: dict[str, Path] = {}
    for candidate in sorted((prefix / "lib").glob("*.so*")):
        if candidate.is_symlink():
            continue
        contents = candidate.read_bytes()
        if len(contents) > 128 * 1024 * 1024:
            raise ValueError("media library exceeds its size bound")
        # Build-tree RUNPATHs are removed before the strict release inspection.
        elf = ELFFile(io.BytesIO(contents))
        dynamic = elf.get_section_by_name(".dynamic")
        sonames = [tag.soname for tag in dynamic.iter_tags() if tag.entry.d_tag == "DT_SONAME"]
        if len(sonames) != 1 or sonames[0] in sources:
            raise ValueError(f"missing or duplicate media SONAME: {candidate.name}")
        if sonames[0] in policy.external_libraries and sonames[0] != "libz.so.1":
            raise ValueError(f"media SDK unexpectedly includes a system library: {sonames[0]}")
        if re.fullmatch(r"lib[A-Za-z0-9_+.-]+\.so(?:\.[0-9]+)*", sonames[0]) is None:
            raise ValueError(f"invalid media SONAME: {sonames[0]}")
        sources[sonames[0]] = candidate
    if not 1 <= len(sources) <= 128:
        raise ValueError("media SDK has no libraries or too many libraries")
    mapping = {name: f"{namespace}_{name}" for name in sources}
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for name, source in sources.items():
            output = destination / mapping[name]
            shutil.copyfile(source, output)
            arguments = [patchelf, "--set-soname", mapping[name], "--set-rpath", "$ORIGIN"]
            for original, replacement in mapping.items():
                arguments.extend(("--replace-needed", original, replacement))
            subprocess.run([*arguments, str(output)], check=True, timeout=60)
        validate_library_graph({p.name: p.read_bytes() for p in destination.iterdir()}, platform)
    except BaseException:
        shutil.rmtree(destination)
        raise
    return mapping


def verify_runtime_source(path: Path, manifest) -> None:
    from vane_packaging.media_sources import read_source_archive, read_source_file

    if path.name != manifest["source"]["filename"]:
        raise ValueError("runtime source archive filename differs from the manifest")
    contents = read_source_file(path)
    if hashlib.sha256(contents).hexdigest() != manifest["source"]["sha256"]:
        raise ValueError("runtime corresponding-source archive digest differs from the signed manifest")
    read_source_archive(contents, path.name)

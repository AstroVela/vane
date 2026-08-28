#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Install a base Vane wheel and extension wheel in a clean environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path, PurePosixPath

from packaging.requirements import InvalidRequirement, Requirement
from packaging.tags import Tag, parse_tag
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ORIGINAL_SYS_PATH = sys.path.copy()
try:
    sys.path.insert(0, str(REPOSITORY_ROOT))
    from scripts.check_release_artifacts import check_artifact as _check_release_artifact
    from vane_packaging.extension_wheel import (
        _ELF_MAGIC,
        _MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES,
        _PE_DOS_MAGIC,
        _PLATFORM_BUILD_DETAILS_FILENAME,
        ENTRY_POINT_GROUP,
        _current_musl_version,
        _entry_points,
        _extension_distribution_version_from_digest,
        _extension_interpreter_tag,
        _is_macos_binary,
        _metadata_license_file_members,
        _PlatformBuildDetails,
        _provider_module_source,
        _read_core_metadata,
        _read_extension_descriptor_member,
        _read_platform_build_details_member,
        _validate_artifact_platform_tag,
        _validate_dependency_platform_tag,
        _validate_exact_requirements,
        _validate_extension_wheel_archive_size,
        _validate_extension_wheel_member_count,
        _validate_extension_wheel_size,
        _validate_linux_elf_platform,
        _validate_macos_wheel_binary_platform,
        _validate_metadata_license_expression,
        _validate_metadata_requires_python,
        _validate_metadata_version,
        _validate_native_binary_platform,
        _validate_owned_extension_wheel_members,
        _validate_wheel_path_component_lengths,
        _validate_wheel_record,
        _validate_windows_pe_platform,
        _wheel_metadata,
        _wheel_platform_policy,
    )
finally:
    sys.path[:] = _ORIGINAL_SYS_PATH
    del _ORIGINAL_SYS_PATH

_EXTENSION_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_EXTENSION_ARTIFACT_RE = re.compile(
    r"^vane_extensions/(?P<name>[a-z][a-z0-9]*(?:_[a-z0-9]+)*)_"
    r"(?P<digest>[0-9a-f]{64})/(?P=name)\.duckdb_extension$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GENERIC_LINUX_BASE_TAG_RE = re.compile(r"^linux_(?:x86_64|aarch64)$")
_LEGACY_MANYLINUX_BASE_TAG_RE = re.compile(
    r"^(?P<policy>manylinux1|manylinux2010|manylinux2014)_(?P<architecture>x86_64|aarch64)$"
)
_MACOS_UNIVERSAL2_BASE_TAG_RE = re.compile(r"^macosx_(?P<major>[0-9]+)_(?P<minor>[0-9]+)_universal2$")
_LEGACY_MANYLINUX_MINIMUMS = {
    "manylinux1": (2, 5),
    "manylinux2010": (2, 12),
    "manylinux2014": (2, 17),
}


@dataclass(frozen=True)
class _ExtensionWheelLayout:
    name: str
    vane_version: str
    distribution_version: str
    identity: tuple[str, str, str]
    dependencies: tuple[tuple[str, str, str], ...]
    requirements: tuple[Requirement, ...]
    interpreter_tag: str
    platform_tag: str
    trust_identity: str
    platform_build_details: _PlatformBuildDetails


def _run(command: list[str], *, cwd: Path, environment: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def _python_path(environment_directory: Path) -> Path:
    executable_name = "python.exe" if os.name == "nt" else "python"
    return environment_directory / ("Scripts" if os.name == "nt" else "bin") / executable_name


def _pip_command(python: Path, *arguments: str) -> list[str]:
    """Run pip without inheriting user configuration or PIP_* variables."""
    return [str(python), "-m", "pip", "--isolated", *arguments]


def _assert_bounded_wheel_contents(wheel: zipfile.ZipFile, *, description: str) -> None:
    try:
        _validate_extension_wheel_archive_size(wheel, description=description)
    except ValueError as exception:
        raise RuntimeError(str(exception)) from exception


def _base_platform_tag_covers_extension(base_platform_tag: str, extension_platform_tag: str) -> bool:
    extension_policy = _wheel_platform_policy(extension_platform_tag)
    extension_minimum_version = extension_policy.minimum_version
    if base_platform_tag == "any":
        return False
    legacy_manylinux = _LEGACY_MANYLINUX_BASE_TAG_RE.fullmatch(base_platform_tag)
    if legacy_manylinux is not None:
        policy = legacy_manylinux["policy"]
        architecture = legacy_manylinux["architecture"]
        return (
            (architecture == "x86_64" or policy == "manylinux2014")
            and extension_policy.family == "manylinux"
            and architecture == extension_policy.architecture
            and extension_minimum_version is not None
            and _LEGACY_MANYLINUX_MINIMUMS[policy] <= extension_minimum_version
        )
    macos_universal2 = _MACOS_UNIVERSAL2_BASE_TAG_RE.fullmatch(base_platform_tag)
    if macos_universal2 is not None:
        minimum_version = (int(macos_universal2["major"]), int(macos_universal2["minor"]))
        canonical_version = macos_universal2["major"] == str(minimum_version[0]) and macos_universal2["minor"] == str(
            minimum_version[1]
        )
        valid_version = (minimum_version[0] == 10 and 4 <= minimum_version[1] <= 16) or (
            minimum_version[0] >= 11 and minimum_version[1] == 0
        )
        effective_minimum_version = (11, 0) if minimum_version == (10, 16) else minimum_version
        return (
            canonical_version
            and valid_version
            and extension_policy.family == "macosx"
            and extension_policy.architecture in {"x86_64", "arm64"}
            and extension_minimum_version is not None
            and effective_minimum_version <= extension_minimum_version
        )
    try:
        _validate_dependency_platform_tag(
            extension_platform_tag,
            base_platform_tag,
            dependency_name="vane-ai",
        )
    except ValueError:
        return False
    return True


def _base_linux_policy_tag(platform_tag: str) -> str | None:
    legacy_manylinux = _LEGACY_MANYLINUX_BASE_TAG_RE.fullmatch(platform_tag)
    if legacy_manylinux is not None:
        minimum_version = _LEGACY_MANYLINUX_MINIMUMS[legacy_manylinux["policy"]]
        return f"manylinux_{minimum_version[0]}_{minimum_version[1]}_{legacy_manylinux['architecture']}"
    try:
        policy = _wheel_platform_policy(platform_tag)
    except ValueError:
        return None
    return platform_tag if policy.family in {"manylinux", "musllinux"} else None


def _base_macos_policy_tag(platform_tag: str) -> str | None:
    if _MACOS_UNIVERSAL2_BASE_TAG_RE.fullmatch(platform_tag) is not None:
        return platform_tag
    try:
        policy = _wheel_platform_policy(platform_tag)
    except ValueError:
        return None
    return platform_tag if policy.family == "macosx" else None


def _base_windows_policy_tag(platform_tag: str) -> str | None:
    try:
        policy = _wheel_platform_policy(platform_tag)
    except ValueError:
        return None
    return platform_tag if policy.family == "windows" else None


def _assert_base_wheel(
    base_wheel: Path,
    *,
    expected_vane_version: str,
    required_interpreter_tag: str,
    required_platform_tag: str,
) -> None:
    try:
        _validate_extension_wheel_size(base_wheel, description="base Vane wheel")
        _validate_extension_wheel_member_count(base_wheel, description="base Vane wheel")
    except ValueError as exception:
        raise RuntimeError(str(exception)) from exception
    try:
        filename_name, filename_version, build_tag, filename_tags = parse_wheel_filename(base_wheel.name)
    except InvalidWheelFilename as exception:
        raise RuntimeError(f"base Vane wheel has an invalid filename: {base_wheel.name!r}") from exception
    if filename_name != canonicalize_name("vane-ai"):
        raise RuntimeError(f"base wheel must be the vane-ai distribution, not {filename_name!r}")
    if build_tag:
        raise RuntimeError(f"base Vane wheel must not use a build tag: {base_wheel.name!r}")
    filename_platform_tags = {tag.platform for tag in filename_tags}
    if "any" in filename_platform_tags:
        raise RuntimeError(f"base Vane wheel must not use the platform-neutral 'any' tag: {base_wheel.name!r}")
    generic_linux_tags = tuple(
        sorted(tag for tag in filename_platform_tags if _GENERIC_LINUX_BASE_TAG_RE.fullmatch(tag) is not None)
    )
    if generic_linux_tags:
        raise RuntimeError(
            f"base Vane wheel must not use generic Linux platform tags without a libc policy: {generic_linux_tags}"
        )
    incompatible_interpreter_tags = tuple(
        sorted(
            str(tag)
            for tag in filename_tags
            if tag.interpreter != required_interpreter_tag or tag.abi != required_interpreter_tag
        )
    )
    if incompatible_interpreter_tags:
        raise RuntimeError(
            f"base Vane wheel tags must use interpreter and ABI {required_interpreter_tag!r} exactly: "
            f"{incompatible_interpreter_tags}"
        )
    base_platform_tags = tuple(sorted(filename_platform_tags))
    incompatible_platform_tags = tuple(
        base_platform_tag
        for base_platform_tag in base_platform_tags
        if not _base_platform_tag_covers_extension(base_platform_tag, required_platform_tag)
    )
    if incompatible_platform_tags:
        raise RuntimeError(
            f"base wheel platform tags {incompatible_platform_tags} do not cover extension wheel platform tag "
            f"{required_platform_tag!r}"
        )
    base_linux_policy_tags = tuple(
        policy_tag
        for platform_tag in sorted(filename_platform_tags)
        if (policy_tag := _base_linux_policy_tag(platform_tag)) is not None
    )
    base_macos_policy_tags = tuple(
        policy_tag
        for platform_tag in sorted(filename_platform_tags)
        if (policy_tag := _base_macos_policy_tag(platform_tag)) is not None
    )
    base_windows_policy_tags = tuple(
        policy_tag
        for platform_tag in sorted(filename_platform_tags)
        if (policy_tag := _base_windows_policy_tag(platform_tag)) is not None
    )

    base_elf_members: tuple[tuple[str, bytes], ...] = ()
    base_macho_members: tuple[tuple[str, bytes], ...] = ()
    base_pe_members: tuple[tuple[str, bytes], ...] = ()
    with zipfile.ZipFile(base_wheel) as wheel:
        _assert_bounded_wheel_contents(wheel, description="base Vane wheel")
        names = wheel.namelist()
        artifacts = [name for name in names if name.casefold().endswith(".duckdb_extension")]
        metadata_members = [
            name for name in names if name.count("/") == 1 and name.casefold().endswith(".dist-info/metadata")
        ]
        wheel_metadata_members = [
            name for name in names if name.count("/") == 1 and name.casefold().endswith(".dist-info/wheel")
        ]
        if len(metadata_members) != 1:
            raise RuntimeError(f"base Vane wheel must contain exactly one top-level METADATA: {metadata_members}")
        expected_wheel_metadata = metadata_members[0].removesuffix("METADATA") + "WHEEL"
        if wheel_metadata_members != [expected_wheel_metadata]:
            raise RuntimeError(
                f"base Vane wheel must contain exactly one WHEEL beside its METADATA: {wheel_metadata_members}"
            )
        try:
            metadata = _read_core_metadata(
                wheel,
                metadata_members[0],
                description="base Vane wheel METADATA",
            )
            wheel_metadata = _read_core_metadata(
                wheel,
                expected_wheel_metadata,
                description="base Vane wheel WHEEL metadata",
            )
        except ValueError as exception:
            raise RuntimeError(str(exception)) from exception
        if base_linux_policy_tags:
            native_members = [
                name
                for name in names
                if PurePosixPath(name).parent == PurePosixPath("vane")
                and PurePosixPath(name).name.startswith("_native.")
                and PurePosixPath(name).suffix == ".so"
            ]
            if len(native_members) != 1:
                raise RuntimeError(
                    f"base Vane wheel must contain exactly one Linux native module, found {native_members}"
                )
            collected_elf_members: list[tuple[str, bytes]] = []
            for name in names:
                with wheel.open(name) as member_file:
                    prefix = member_file.read(len(_ELF_MAGIC))
                    if prefix == _ELF_MAGIC:
                        collected_elf_members.append((name, prefix + member_file.read()))
            if native_members[0] not in {name for name, _contents in collected_elf_members}:
                raise RuntimeError(f"base Vane wheel native module {native_members[0]!r} must contain ELF data")
            base_elf_members = tuple(collected_elf_members)
        if base_macos_policy_tags:
            native_members = [
                name
                for name in names
                if PurePosixPath(name).parent == PurePosixPath("vane")
                and PurePosixPath(name).name.startswith("_native.")
                and PurePosixPath(name).suffix == ".so"
            ]
            if len(native_members) != 1:
                raise RuntimeError(
                    f"base Vane wheel must contain exactly one macOS native module, found {native_members}"
                )
            collected_macho_members: list[tuple[str, bytes]] = []
            for name in names:
                with wheel.open(name) as member_file:
                    prefix = member_file.read(4)
                    if _is_macos_binary(prefix):
                        collected_macho_members.append((name, prefix + member_file.read()))
            if native_members[0] not in {name for name, _contents in collected_macho_members}:
                raise RuntimeError(f"base Vane wheel native module {native_members[0]!r} must contain Mach-O data")
            base_macho_members = tuple(collected_macho_members)
        if base_windows_policy_tags:
            native_members = [
                name
                for name in names
                if PurePosixPath(name).parent == PurePosixPath("vane")
                and PurePosixPath(name).name.startswith("_native.")
                and PurePosixPath(name).suffix.casefold() == ".pyd"
            ]
            if len(native_members) != 1:
                raise RuntimeError(
                    f"base Vane wheel must contain exactly one Windows native module, found {native_members}"
                )
            collected_pe_members: list[tuple[str, bytes]] = []
            for name in names:
                with wheel.open(name) as member_file:
                    prefix = member_file.read(len(_PE_DOS_MAGIC))
                    if prefix == _PE_DOS_MAGIC:
                        collected_pe_members.append((name, prefix + member_file.read()))
            if native_members[0] not in {name for name, _contents in collected_pe_members}:
                raise RuntimeError(f"base Vane wheel native module {native_members[0]!r} must contain PE data")
            base_pe_members = tuple(collected_pe_members)
    if artifacts:
        raise RuntimeError(f"base Vane wheel must not contain dynamic extension artifacts: {artifacts}")
    metadata_names = metadata.get_all("Name", [])
    if len(metadata_names) != 1 or canonicalize_name(metadata_names[0]) != canonicalize_name("vane-ai"):
        raise RuntimeError(f"base wheel METADATA must identify vane-ai exactly once: {metadata_names}")
    try:
        _validate_metadata_version(metadata, description="base Vane wheel")
        _validate_metadata_requires_python(metadata, description="base Vane wheel")
    except ValueError as exception:
        raise RuntimeError(str(exception)) from exception
    metadata_versions = metadata.get_all("Version", [])
    if len(metadata_versions) != 1:
        raise RuntimeError(f"base wheel METADATA must contain exactly one Version: {metadata_versions}")
    try:
        metadata_version = Version(metadata_versions[0])
        expected_version = Version(expected_vane_version)
    except InvalidVersion as exception:
        raise RuntimeError("base or extension wheel contains an invalid Vane version") from exception
    if filename_version != expected_version or metadata_version != expected_version:
        raise RuntimeError(
            f"base wheel version must match extension Vane version {expected_vane_version!r}: "
            f"filename={filename_version}, metadata={metadata_version}"
        )
    if wheel_metadata.get_all("Wheel-Version", []) != ["1.0"]:
        raise RuntimeError("base Vane wheel must declare exactly Wheel-Version: 1.0")
    if wheel_metadata.get_all("Root-Is-Purelib", []) != ["false"]:
        raise RuntimeError("base Vane wheel must declare exactly Root-Is-Purelib: false")
    wheel_tag_values = [str(value) for value in wheel_metadata.get_all("Tag", [])]
    try:
        parsed_wheel_tags = tuple(tag for value in wheel_tag_values for tag in parse_tag(value))
    except ValueError as exception:
        raise RuntimeError(f"base wheel WHEEL contains an invalid compatibility tag: {wheel_tag_values}") from exception
    wheel_tags = frozenset(parsed_wheel_tags)
    if len(wheel_tags) != len(parsed_wheel_tags):
        raise RuntimeError(f"base wheel WHEEL must not contain duplicate compatibility tags: {wheel_tag_values}")
    if wheel_tags != filename_tags:
        raise RuntimeError(
            "base wheel WHEEL tags must match its filename tags exactly: "
            f"filename={tuple(sorted(map(str, filename_tags)))}, "
            f"metadata={tuple(sorted(map(str, wheel_tags)))}"
        )
    try:
        for policy_tag in base_linux_policy_tags:
            policy = _wheel_platform_policy(policy_tag)
            platform_build_details = _PlatformBuildDetails(
                platform_tag=policy_tag,
                musl_version=policy.minimum_version if policy.family == "musllinux" else None,
            )
            for member_name, contents in base_elf_members:
                _validate_linux_elf_platform(
                    contents,
                    policy_tag,
                    description=f"base Vane wheel member {member_name!r}",
                    platform_build_details=platform_build_details,
                )
    except ValueError as exception:
        raise RuntimeError(str(exception)) from exception
    try:
        for policy_tag in base_macos_policy_tags:
            for member_name, contents in base_macho_members:
                _validate_macos_wheel_binary_platform(
                    contents,
                    policy_tag,
                    description=f"base Vane wheel member {member_name!r}",
                )
    except ValueError as exception:
        raise RuntimeError(str(exception)) from exception
    try:
        for policy_tag in base_windows_policy_tags:
            for member_name, contents in base_pe_members:
                _validate_windows_pe_platform(
                    contents,
                    policy_tag,
                    description=f"base Vane wheel member {member_name!r}",
                    interpreter_tag=required_interpreter_tag,
                )
    except ValueError as exception:
        raise RuntimeError(str(exception)) from exception
    try:
        _check_release_artifact(base_wheel, expected_version=expected_vane_version)
    except (ValueError, zipfile.BadZipFile) as exception:
        raise RuntimeError(f"base Vane wheel failed release-artifact validation: {exception}") from exception


def _descriptor_identity(value: dict[str, object], *, description: str) -> tuple[str, str, str]:
    name = value.get("name")
    extension_version = value.get("extension_version")
    sha256 = value.get("sha256")
    if not isinstance(name, str) or _EXTENSION_NAME_RE.fullmatch(name) is None:
        raise RuntimeError(f"{description} must contain a valid extension name")
    if not isinstance(extension_version, str) or not extension_version:
        raise RuntimeError(f"{description} must contain a non-empty extension version")
    if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
        raise RuntimeError(f"{description} must contain a lowercase SHA-256")
    return name, extension_version, sha256


def _descriptor_dependencies(descriptor_document: dict[str, object]) -> tuple[tuple[str, str, str], ...]:
    values = descriptor_document.get("dependencies")
    if not isinstance(values, list):
        raise RuntimeError("extension wheel descriptor dependencies must be a list")
    if len(values) > _MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES:
        raise RuntimeError(
            f"extension wheel descriptor contains more than {_MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES} dependencies"
        )
    dependencies: list[tuple[str, str, str]] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict) or set(value) != {"name", "extension_version", "sha256"}:
            raise RuntimeError(
                f"extension wheel descriptor dependency {index} must contain only name, extension_version, and sha256"
            )
        dependencies.append(_descriptor_identity(value, description=f"descriptor dependency {index}"))
    if len(set(dependencies)) != len(dependencies):
        raise RuntimeError("extension wheel descriptor dependencies must be unique")
    return tuple(dependencies)


def _extension_wheel_tag(
    extension_wheel: Path,
    *,
    distribution_name: str,
    distribution_version: str,
) -> Tag:
    try:
        filename_name, filename_version, build_tag, filename_tags = parse_wheel_filename(extension_wheel.name)
    except InvalidWheelFilename as exception:
        raise RuntimeError(f"extension wheel has an invalid filename: {extension_wheel.name!r}") from exception
    if build_tag:
        raise RuntimeError(f"extension wheel must not use a build tag: {extension_wheel.name!r}")
    if filename_name != canonicalize_name(distribution_name) or filename_version != Version(distribution_version):
        raise RuntimeError(
            "extension wheel filename must match its descriptor-bound distribution: "
            f"expected={distribution_name}==={distribution_version}, "
            f"actual={filename_name}==={filename_version}"
        )
    if len(filename_tags) != 1:
        raise RuntimeError(f"extension wheel must have exactly one compatibility tag: {extension_wheel.name!r}")
    filename_tag = next(iter(filename_tags))
    expected_interpreter_tag = _extension_interpreter_tag()
    if filename_tag.interpreter != expected_interpreter_tag or filename_tag.abi != "none":
        raise RuntimeError(
            f"extension wheel must use exactly one {expected_interpreter_tag}-none platform tag: {filename_tag}"
        )
    return filename_tag


def _assert_extension_wheel_layout(extension_wheel: Path, extension_name: str) -> _ExtensionWheelLayout:
    if _EXTENSION_NAME_RE.fullmatch(extension_name) is None:
        raise ValueError("extension_name must use wheel-safe lowercase ASCII snake_case with single underscores")
    try:
        _validate_extension_wheel_size(extension_wheel)
        _validate_extension_wheel_member_count(extension_wheel)
    except ValueError as exception:
        raise RuntimeError(str(exception)) from exception
    with zipfile.ZipFile(extension_wheel) as wheel:
        _assert_bounded_wheel_contents(wheel, description="extension wheel")
        names = wheel.namelist()
        try:
            _validate_wheel_path_component_lengths(
                (extension_wheel.name, *names),
                description="extension wheel",
            )
        except ValueError as exception:
            raise RuntimeError(str(exception)) from exception
        descriptors = [name for name in names if name.casefold().endswith(".dynamic-extension.json")]
        if len(descriptors) != 1:
            raise RuntimeError(f"extension wheel must contain exactly one descriptor: {descriptors}")
        try:
            descriptor_contents = _read_extension_descriptor_member(
                wheel,
                descriptors[0],
                description="extension wheel descriptor",
            )
        except ValueError as exception:
            raise RuntimeError(str(exception)) from exception
        metadata_members = [
            name for name in names if name.count("/") == 1 and name.casefold().endswith(".dist-info/metadata")
        ]
    try:
        descriptor_document = json.loads(descriptor_contents)
        canonical_descriptor = json.dumps(
            descriptor_document,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exception:
        raise RuntimeError("extension wheel descriptor must contain valid JSON") from exception
    if descriptor_contents != canonical_descriptor + b"\n":
        raise RuntimeError("extension wheel descriptor must use canonical JSON followed by one newline")
    if not isinstance(descriptor_document, dict):
        raise RuntimeError("extension wheel descriptor must contain a JSON object")
    if descriptor_document.get("name") != extension_name:
        raise RuntimeError(
            f"extension wheel descriptor name must be {extension_name!r}: {descriptor_document.get('name')!r}"
        )
    descriptor_identity = _descriptor_identity(descriptor_document, description="extension wheel descriptor")
    descriptor_dependencies = _descriptor_dependencies(descriptor_document)
    vane_version = descriptor_document.get("vane_version")
    if not isinstance(vane_version, str) or not vane_version:
        raise RuntimeError("extension wheel descriptor must contain a non-empty Vane version")
    artifact_platform = descriptor_document.get("platform")
    if not isinstance(artifact_platform, str) or not artifact_platform:
        raise RuntimeError("extension wheel descriptor must contain a non-empty artifact platform")
    trust_identity = descriptor_document.get("trust_identity")
    if not isinstance(trust_identity, str) or not trust_identity:
        raise RuntimeError("extension wheel descriptor must contain a non-empty trust identity")
    descriptor_digest = hashlib.sha256(canonical_descriptor).hexdigest()
    distribution_version = _extension_distribution_version_from_digest(vane_version, descriptor_digest)
    distribution_root = f"vane_extension_{extension_name}-{distribution_version}"
    expected_metadata = f"{distribution_root}.dist-info/METADATA"
    expected_platform_build_details = f"{distribution_root}.dist-info/{_PLATFORM_BUILD_DETAILS_FILENAME}"
    if metadata_members != [expected_metadata]:
        raise RuntimeError(f"extension wheel must contain exactly {expected_metadata!r}: metadata={metadata_members}")
    expected_distribution_name = f"vane-extension-{extension_name}"
    filename_tag = _extension_wheel_tag(
        extension_wheel,
        distribution_name=expected_distribution_name,
        distribution_version=distribution_version,
    )
    platform_tag = filename_tag.platform
    try:
        _validate_artifact_platform_tag(artifact_platform, platform_tag)
    except ValueError as exception:
        raise RuntimeError(str(exception)) from exception
    with zipfile.ZipFile(extension_wheel) as wheel:
        _assert_bounded_wheel_contents(wheel, description="extension wheel")
        try:
            metadata = _read_core_metadata(
                wheel,
                expected_metadata,
                description="extension wheel METADATA",
            )
            platform_build_details = _read_platform_build_details_member(
                wheel,
                expected_platform_build_details,
                expected_platform_tag=platform_tag,
                description="extension wheel",
            )
        except ValueError as exception:
            raise RuntimeError(str(exception)) from exception
    if metadata.get_all("Name", []) != [expected_distribution_name]:
        raise RuntimeError(
            f"extension wheel Name must be {expected_distribution_name!r}: {metadata.get_all('Name', [])}"
        )
    if metadata.get_all("Version", []) != [distribution_version]:
        raise RuntimeError(
            f"extension wheel Version must be {distribution_version!r}: {metadata.get_all('Version', [])}"
        )
    try:
        _validate_metadata_version(metadata)
        _validate_metadata_requires_python(metadata)
        _validate_metadata_license_expression(metadata)
    except ValueError as exception:
        raise RuntimeError(str(exception)) from exception
    try:
        requirements = tuple(Requirement(value) for value in metadata.get_all("Requires-Dist", []))
    except InvalidRequirement as exception:
        raise RuntimeError("extension wheel contains an invalid dependency requirement") from exception
    expected_package_root = f"vane_extensions/{extension_name}_{descriptor_digest}/"
    expected_artifact = f"{expected_package_root}{extension_name}.duckdb_extension"
    expected_descriptor = f"{expected_package_root}{extension_name}.dynamic-extension.json"
    expected_provider = f"{expected_package_root}__init__.py"
    artifacts = [name for name in names if name.casefold().endswith(".duckdb_extension")]
    if artifacts != [expected_artifact] or descriptors != [expected_descriptor]:
        raise RuntimeError(
            f"extension wheel must contain exactly {expected_artifact!r} and {expected_descriptor!r}: "
            f"artifacts={artifacts}, descriptors={descriptors}"
        )
    try:
        license_members = _metadata_license_file_members(
            metadata,
            dist_info_root=distribution_root,
            windows_paths=artifact_platform.startswith("windows_"),
        )
        _validate_owned_extension_wheel_members(
            names,
            expected_provider=expected_provider,
            expected_artifact=expected_artifact,
            expected_descriptor=expected_descriptor,
            expected_platform_build_details=expected_platform_build_details,
            dist_info_root=distribution_root,
            license_members=license_members,
        )
    except ValueError as exception:
        raise RuntimeError(str(exception)) from exception
    expected_wheel_metadata = f"{distribution_root}.dist-info/WHEEL"
    expected_entry_points = f"{distribution_root}.dist-info/entry_points.txt"
    expected_record = f"{distribution_root}.dist-info/RECORD"
    with zipfile.ZipFile(extension_wheel) as wheel:
        _assert_bounded_wheel_contents(wheel, description="extension wheel")
        if wheel.read(expected_wheel_metadata) != _wheel_metadata(str(filename_tag)).encode("utf-8"):
            raise RuntimeError("extension wheel must contain its exact generated WHEEL metadata")
        if wheel.read(expected_provider) != _provider_module_source(extension_name).encode("utf-8"):
            raise RuntimeError("extension wheel provider module must match the generated provider exactly")
        if wheel.read(expected_entry_points) != _entry_points(
            extension_name,
            f"{extension_name}_{descriptor_digest}",
        ).encode("utf-8"):
            raise RuntimeError("extension wheel must advertise its exact generated provider entry point")
        artifact_contents = wheel.read(expected_artifact)
        if hashlib.sha256(artifact_contents).hexdigest() != descriptor_identity[2]:
            raise RuntimeError("extension wheel artifact SHA-256 must match its descriptor")
        try:
            _validate_native_binary_platform(
                artifact_contents,
                platform_tag,
                description="extension wheel artifact",
                platform_build_details=platform_build_details,
                interpreter_tag=filename_tag.interpreter,
            )
        except ValueError as exception:
            raise RuntimeError(str(exception)) from exception
        try:
            _validate_wheel_record(wheel, names=names, record_name=expected_record)
        except ValueError as exception:
            raise RuntimeError(str(exception)) from exception
    return _ExtensionWheelLayout(
        name=extension_name,
        vane_version=vane_version,
        distribution_version=distribution_version,
        identity=descriptor_identity,
        dependencies=descriptor_dependencies,
        requirements=requirements,
        interpreter_tag=filename_tag.interpreter,
        platform_tag=platform_tag,
        trust_identity=trust_identity,
        platform_build_details=platform_build_details,
    )


def _assert_extension_requirements(
    layout: _ExtensionWheelLayout,
    layouts_by_identity: dict[tuple[str, str, str], _ExtensionWheelLayout],
) -> None:
    expected_versions = {canonicalize_name("vane-ai"): layout.vane_version}
    for dependency_identity in layout.dependencies:
        dependency_layout = layouts_by_identity.get(dependency_identity)
        if dependency_layout is None:
            name, extension_version, sha256 = dependency_identity
            raise RuntimeError(
                f"extension {layout.name!r} descriptor dependency "
                f"{name}@{extension_version}#sha256:{sha256} has no exact supplied wheel"
            )
        if dependency_layout.interpreter_tag != layout.interpreter_tag:
            raise RuntimeError(
                f"dependency {dependency_layout.name!r} wheel interpreter tag "
                f"{dependency_layout.interpreter_tag!r} does not match parent extension wheel interpreter tag "
                f"{layout.interpreter_tag!r}"
            )
        try:
            _validate_dependency_platform_tag(
                layout.platform_tag,
                dependency_layout.platform_tag,
                dependency_name=dependency_layout.name,
            )
        except ValueError as exception:
            raise RuntimeError(str(exception)) from exception
        dependency_name = canonicalize_name(f"vane-extension-{dependency_layout.name}")
        if dependency_name in expected_versions:
            raise RuntimeError(f"extension {layout.name!r} descriptor repeats dependency {dependency_layout.name!r}")
        expected_versions[dependency_name] = dependency_layout.distribution_version

    try:
        _validate_exact_requirements(
            layout.requirements,
            expected_versions,
            description=f"extension wheel {layout.name!r}",
        )
    except ValueError as exception:
        raise RuntimeError(str(exception)) from exception


def _extension_name_from_artifact_path(extension_wheel: Path) -> str:
    try:
        _validate_extension_wheel_size(extension_wheel, description="dependency extension wheel")
        _validate_extension_wheel_member_count(extension_wheel, description="dependency extension wheel")
    except ValueError as exception:
        raise RuntimeError(str(exception)) from exception
    with zipfile.ZipFile(extension_wheel) as wheel:
        _assert_bounded_wheel_contents(wheel, description="dependency extension wheel")
        artifacts = [name for name in wheel.namelist() if name.casefold().endswith(".duckdb_extension")]
    if len(artifacts) != 1:
        raise RuntimeError(f"dependency extension wheel must contain exactly one artifact: {artifacts}")
    match = _EXTENSION_ARTIFACT_RE.fullmatch(artifacts[0])
    if match is None:
        raise RuntimeError(f"dependency extension wheel has an invalid artifact path: {artifacts[0]!r}")
    return match["name"]


def _extension_name_from_wheel(extension_wheel: Path) -> str:
    extension_name = _extension_name_from_artifact_path(extension_wheel)
    return _assert_extension_wheel_layout(extension_wheel, extension_name).name


def _explicit_dependency_trust_identities(
    values: Iterable[str],
    dependency_layouts: tuple[_ExtensionWheelLayout, ...],
) -> frozenset[str]:
    if isinstance(values, str):
        raise RuntimeError("dependency trust identities must be supplied as an iterable")
    try:
        identities = tuple(islice(iter(values), _MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES + 1))
    except TypeError as exception:
        raise RuntimeError("dependency trust identities must be supplied as an iterable") from exception
    if len(identities) > _MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES:
        raise RuntimeError(
            f"dependency trust identities contain more than {_MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES} identities"
        )
    if any(not isinstance(identity, str) or not identity for identity in identities):
        raise RuntimeError("dependency trust identities must contain only non-empty strings")
    if len(set(identities)) != len(identities):
        raise RuntimeError("dependency trust identities must not contain duplicates")
    supplied = frozenset(identities)
    required = frozenset(layout.trust_identity for layout in dependency_layouts)
    if supplied != required:
        raise RuntimeError(
            "dependency trust identities must be supplied explicitly and match exactly: "
            f"required={tuple(sorted(required))}, supplied={tuple(sorted(supplied))}"
        )
    return supplied


def _assert_musl_verification_runtime(layout: _ExtensionWheelLayout) -> None:
    policy = _wheel_platform_policy(layout.platform_tag)
    if policy.family != "musllinux":
        return
    runtime_version = _current_musl_version()
    if runtime_version != policy.minimum_version:
        rendered = "not musl" if runtime_version is None else f"musl {runtime_version[0]}.{runtime_version[1]}"
        raise RuntimeError(
            f"clean verification for {layout.platform_tag!r} must run on its exact minimum musl runtime, not {rendered}"
        )


def verify_extension_wheel(
    *,
    base_wheel: str | Path,
    extension_wheel: str | Path,
    extension_name: str,
    trust_identity: str,
    dependency_wheels: Iterable[str | Path] = (),
    dependency_trust_identities: Iterable[str] = (),
) -> None:
    """Verify clean installation, metadata discovery, and local artifact loading."""
    if isinstance(dependency_wheels, (str, os.PathLike)):
        raise RuntimeError("dependency wheels must be supplied as an iterable of wheel paths")
    try:
        unresolved_dependency_wheels = tuple(
            islice(iter(dependency_wheels), _MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES + 1)
        )
    except TypeError as exception:
        raise RuntimeError("dependency wheels must be supplied as an iterable of wheel paths") from exception
    if len(unresolved_dependency_wheels) > _MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES:
        raise RuntimeError(f"dependency wheels contain more than {_MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES} wheel paths")
    if any(not isinstance(wheel, (str, os.PathLike)) for wheel in unresolved_dependency_wheels):
        raise RuntimeError("dependency wheels must contain only wheel paths")

    resolved_base_wheel = Path(base_wheel).expanduser().resolve(strict=True)
    resolved_extension_wheel = Path(extension_wheel).expanduser().resolve(strict=True)
    resolved_dependency_wheels = tuple(
        Path(dependency_wheel).expanduser().resolve(strict=True) for dependency_wheel in unresolved_dependency_wheels
    )
    root_layout = _assert_extension_wheel_layout(resolved_extension_wheel, extension_name)
    if root_layout.trust_identity != trust_identity:
        raise RuntimeError(
            f"root extension trust identity must be {trust_identity!r}, not {root_layout.trust_identity!r}"
        )
    dependency_layouts = tuple(
        _assert_extension_wheel_layout(wheel, _extension_name_from_artifact_path(wheel))
        for wheel in resolved_dependency_wheels
    )
    trusted_dependency_identities = _explicit_dependency_trust_identities(
        dependency_trust_identities,
        dependency_layouts,
    )
    _assert_musl_verification_runtime(root_layout)
    _assert_base_wheel(
        resolved_base_wheel,
        expected_vane_version=root_layout.vane_version,
        required_interpreter_tag=root_layout.interpreter_tag,
        required_platform_tag=root_layout.platform_tag,
    )
    dependency_names = tuple(layout.name for layout in dependency_layouts)
    all_extension_names = (extension_name, *dependency_names)
    if len(set(all_extension_names)) != len(all_extension_names):
        raise RuntimeError(f"extension and dependency wheels must have unique names: {all_extension_names}")
    layouts = (root_layout, *dependency_layouts)
    if any(layout.vane_version != root_layout.vane_version for layout in dependency_layouts):
        raise RuntimeError("extension and dependency wheels must require the same exact Vane version")
    layouts_by_identity = {layout.identity: layout for layout in layouts}
    if len(layouts_by_identity) != len(layouts):
        raise RuntimeError("extension and dependency wheels must have unique descriptor identities")
    for layout in layouts:
        _assert_extension_requirements(layout, layouts_by_identity)
    trusted_identities = frozenset({trust_identity, *trusted_dependency_identities})

    with tempfile.TemporaryDirectory(prefix="vane-extension-wheel-") as temporary_directory:
        workspace = Path(temporary_directory)
        environment_directory = workspace / "venv"
        environment = os.environ.copy()
        for variable in tuple(environment):
            if variable.upper().startswith("PYTHON"):
                environment.pop(variable)
        environment.pop("__PYVENV_LAUNCHER__", None)
        environment.pop("VIRTUAL_ENV", None)
        environment["PIP_CONFIG_FILE"] = os.devnull
        environment["PYTHONSAFEPATH"] = "1"
        _run(
            [sys.executable, "-I", "-m", "venv", "--clear", "--copies", str(environment_directory)],
            cwd=workspace,
            environment=environment,
        )
        python = _python_path(environment_directory)
        _run(
            _pip_command(
                python,
                "--disable-pip-version-check",
                "install",
                str(resolved_base_wheel),
                *(str(dependency_wheel) for dependency_wheel in resolved_dependency_wheels),
                str(resolved_extension_wheel),
            ),
            cwd=workspace,
            environment=environment,
        )
        _run(_pip_command(python, "check"), cwd=workspace, environment=environment)

        validation = textwrap.dedent(
            f"""
            from importlib import import_module
            from importlib.metadata import entry_points
            from pathlib import Path

            import vane
            from vane.extensions import DynamicExtensionResolver

            installed_entry_points = tuple(entry_points(group={ENTRY_POINT_GROUP!r}))
            provider_by_name = {{}}
            entry_point_by_name = {{}}

            def installed_provider(name):
                if name not in provider_by_name:
                    matches = [candidate for candidate in installed_entry_points if candidate.name == name]
                    assert len(matches) == 1, matches
                    entry_point_by_name[name] = matches[0]
                    provider_by_name[name] = matches[0].load()()
                return provider_by_name[name]

            installed_provider({extension_name!r})
            root_entry_point = entry_point_by_name[{extension_name!r}]
            root_module = import_module(root_entry_point.module)
            descriptor = root_module.descriptor()
            assert descriptor.name == {extension_name!r}
            assert descriptor.trust_identity == {trust_identity!r}

            descriptor_by_name = {{}}
            pending = [descriptor]
            while pending:
                candidate = pending.pop()
                existing = descriptor_by_name.get(candidate.name)
                if existing is not None:
                    assert existing == candidate, (existing, candidate)
                    continue
                provider = installed_provider(candidate.name)
                artifact = provider.find(candidate.identity)
                assert artifact is not None, candidate.identity
                assert artifact.descriptor == candidate
                provider_module = import_module(entry_point_by_name[candidate.name].module)
                assert artifact.path.name == candidate.name + ".duckdb_extension"
                assert artifact.path.parent == Path(provider_module.__file__).resolve().parent
                assert candidate.vane_version == vane.__version__
                assert candidate.trust_identity in {trusted_identities!r}
                descriptor_by_name[candidate.name] = candidate
                for dependency in reversed(candidate.dependencies):
                    dependency_provider = installed_provider(dependency.name)
                    dependency_artifact = dependency_provider.find(dependency.identity)
                    assert dependency_artifact is not None, dependency.identity
                    pending.append(dependency_artifact.descriptor)

            assert set(descriptor_by_name) == set({all_extension_names!r})

            connection = vane.connect()
            try:
                resolved = DynamicExtensionResolver(
                    trusted_identities={trusted_identities!r},
                    providers=tuple(provider_by_name.values()),
                    cache_directory=Path.cwd() / "extension-cache",
                ).load(connection, descriptor)
                assert resolved.identity == descriptor.identity
                assert connection.execute(
                    "SELECT loaded FROM duckdb_extensions() WHERE extension_name = ?", [descriptor.name]
                ).fetchone() == (True,)
            finally:
                connection.close()
            """
        )
        program = f"exec(compile({validation!r}, '<extension-wheel-validation>', 'exec', optimize=0))"
        _run([str(python), "-I", "-c", program], cwd=workspace, environment=environment)


def main() -> int:
    """Run clean-install verification for one extension wheel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-wheel", required=True, type=Path)
    parser.add_argument("--extension-wheel", required=True, type=Path)
    parser.add_argument("--extension-name", required=True)
    parser.add_argument("--trust-identity", required=True)
    parser.add_argument(
        "--dependency-wheel",
        action="append",
        default=[],
        type=Path,
        help="Complete dependency-wheel closure in load order; repeat once per wheel",
    )
    parser.add_argument(
        "--dependency-trust-identity",
        action="append",
        default=[],
        help="Explicitly trusted dependency signer identity; repeat once per unique identity",
    )
    arguments = parser.parse_args()
    verify_extension_wheel(
        base_wheel=arguments.base_wheel,
        extension_wheel=arguments.extension_wheel,
        extension_name=arguments.extension_name,
        trust_identity=arguments.trust_identity,
        dependency_wheels=arguments.dependency_wheel,
        dependency_trust_identities=arguments.dependency_trust_identity,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build a platform-specific wheel for one verified Vane extension artifact.

The base ``vane-ai`` wheel deliberately does not contain optional DuckDB
extension binaries. This module packages one already-built artifact and its
immutable descriptor into an independent wheel with an installed-provider
entry point.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
import stat
import struct
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING

from elftools.common.exceptions import ELFError
from elftools.common.utils import struct_parse
from elftools.elf.elffile import ELFFile
from packaging.licenses import InvalidLicenseExpression, canonicalize_license_expression
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from vane_packaging.archive_safety import ArchiveSnapshot, open_zip_snapshot, snapshot_archive
from vane_packaging.artifact_limits import (
    ARCHIVE_MEMBER_LIMIT_DESCRIPTION,
    ARCHIVE_TOTAL_LIMIT_DESCRIPTION,
    EXTENSION_ARTIFACT_LIMIT_DESCRIPTION,
    MAX_ARCHIVE_MEMBER_BYTES,
    MAX_ARCHIVE_UNCOMPRESSED_BYTES,
    MAX_EXTENSION_ARTIFACT_BYTES,
    MAX_PUBLICATION_FILE_BYTES,
    PUBLICATION_FILE_LIMIT_DESCRIPTION,
)
from vane_packaging.manylinux_policy import ManylinuxPolicy, manylinux_policy

if TYPE_CHECKING:
    from vane.extensions import DynamicExtensionDependency, DynamicExtensionDescriptor

ENTRY_POINT_GROUP = "vane.dynamic_extension_providers"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_EXTENSION_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_EXTENSION_INTERPRETER_TAG_RE = re.compile(r"^cp3(?:10|11|12|13|14)$")
_WHEEL_PLATFORM_TAG_RE = re.compile(r"^[a-z0-9_]+$")
_MAX_WHEEL_PATH_COMPONENT_BYTES = 255
_MAX_EXTENSION_WHEEL_BYTES = MAX_PUBLICATION_FILE_BYTES
_MAX_EXTENSION_WHEEL_MEMBER_BYTES = MAX_ARCHIVE_MEMBER_BYTES
_MAX_EXTENSION_WHEEL_UNCOMPRESSED_BYTES = MAX_ARCHIVE_UNCOMPRESSED_BYTES
_MAX_EXTENSION_ARTIFACT_BYTES = MAX_EXTENSION_ARTIFACT_BYTES
_MAX_EXTENSION_WHEEL_MEMBERS = 10_000
_MAX_CORE_METADATA_BYTES = 1024 * 1024
_MAX_CORE_METADATA_HEADERS = 1024
_MAX_CORE_METADATA_LINES = 10_000
_MAX_CORE_METADATA_LINE_BYTES = 64 * 1024
_MAX_EXTENSION_DESCRIPTOR_BYTES = 64 * 1024
_MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES = 256
_MAX_EXTENSION_DESCRIPTOR_JSON_DEPTH = 3
_MAX_EXTENSION_DESCRIPTOR_JSON_SEPARATORS = 4096
_MAX_EXTENSION_DESCRIPTOR_JSON_STRINGS = 2048
_MAX_EXTENSION_DESCRIPTOR_JSON_STRING_BYTES = 4096
_MAX_EXTENSION_DESCRIPTOR_JSON_SCALAR_BYTES = 128
_MAX_PLATFORM_BUILD_DETAILS_BYTES = 1024
_WHEEL_RECORD_ROW_FIXED_CHARS = 128
_EXTENSION_REQUIRES_PYTHON = ">=3.10,<3.15"
_EXTENSION_REQUIRES_PYTHON_SPECIFIERS = SpecifierSet(_EXTENSION_REQUIRES_PYTHON)
_EXTENSION_METADATA_VERSION = "2.4"
_DESCRIPTOR_DIGEST_CHUNK_HEX_LENGTH = 8
_VANE_RELEASE_COMPONENT_COUNT = 8
_PRERELEASE_STAGE = {"a": 1, "b": 2, "rc": 3}
_WINDOWS_INVALID_PATH_CHARACTERS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_PATH_NAMES = frozenset(
    {
        "aux",
        "con",
        "conin$",
        "conout$",
        "nul",
        "prn",
        *(f"com{number}" for number in range(1, 10)),
        *(f"lpt{number}" for number in range(1, 10)),
    }
)
_ARTIFACT_PLATFORM_TAGS = {
    "linux_amd64": re.compile(r"^manylinux_[0-9]+_[0-9]+_x86_64$"),
    "linux_amd64_musl": re.compile(r"^musllinux_[0-9]+_[0-9]+_x86_64$"),
    "linux_arm64": re.compile(r"^manylinux_[0-9]+_[0-9]+_aarch64$"),
    "linux_arm64_musl": re.compile(r"^musllinux_[0-9]+_[0-9]+_aarch64$"),
    "osx_amd64": re.compile(r"^macosx_[0-9]+_[0-9]+_x86_64$"),
    "osx_arm64": re.compile(r"^macosx_[0-9]+_[0-9]+_arm64$"),
    "windows_amd64": re.compile(r"^win_amd64$"),
    "windows_arm64": re.compile(r"^win_arm64$"),
}
_VERSIONED_WHEEL_PLATFORM_TAG_RE = re.compile(
    r"^(?P<family>manylinux|musllinux|macosx)_(?P<major>[0-9]+)_(?P<minor>[0-9]+)_"
    r"(?P<architecture>x86_64|aarch64|arm64)$"
)
_WINDOWS_WHEEL_PLATFORM_TAG_RE = re.compile(r"^win_(?P<architecture>amd64|arm64)$")
_MANYLINUX_POLICY_FLOORS = {
    "x86_64": (2, 5),
    "aarch64": (2, 17),
}
_ELF_MAGIC = b"\x7fELF"
_ELF_SHARED_OBJECT_TYPE = 3
_ELF_MACHINE_BY_ARCHITECTURE = {
    "x86_64": 62,
    "aarch64": 183,
}
_GLIBC_VERSION_NAME_RE = re.compile(r"^GLIBC_([0-9]+(?:\.[0-9]+)+)$")
_MAX_LIBC_VERSION_COMPONENT_DIGITS = 4
_PLATFORM_BUILD_DETAILS_FILENAME = "vane-extension-platform.json"
_ELF_HEADER_64 = struct.Struct("<16sHHIQQQIHHHHHH")
_ELF_PROGRAM_HEADER_64 = struct.Struct("<IIQQQQQQ")
_ELF_DYNAMIC_ENTRY_64 = struct.Struct("<qQ")
_ELF_DYNAMIC_SYMBOL_64 = struct.Struct("<IBBHQQ")
_ELF_SYSV_HASH_HEADER = struct.Struct("<II")
_ELF_GNU_HASH_HEADER = struct.Struct("<IIII")
_ELF_VERSION_NEEDED_ENTRY_64 = struct.Struct("<HHIII")
_ELF_VERSION_NEEDED_AUXILIARY_64 = struct.Struct("<IHHII")
_ELF_LOAD_SEGMENT_TYPE = 1
_ELF_DYNAMIC_SEGMENT_TYPE = 2
_ELF_PROGRAM_INTERPRETER_TYPE = 3
_ELF_DYNAMIC_NULL_TAG = 0
_ELF_DYNAMIC_NEEDED_TAG = 1
_ELF_DYNAMIC_HASH_TAG = 4
_ELF_DYNAMIC_STRING_TABLE_TAG = 5
_ELF_DYNAMIC_SYMBOL_TABLE_TAG = 6
_ELF_DYNAMIC_STRING_TABLE_SIZE_TAG = 10
_ELF_DYNAMIC_SYMBOL_ENTRY_SIZE_TAG = 11
_ELF_DYNAMIC_SONAME_TAG = 14
_ELF_DYNAMIC_RPATH_TAG = 15
_ELF_DYNAMIC_RUNPATH_TAG = 29
_ELF_DYNAMIC_CONFIG_TAG = 0x6FFFFEFA
_ELF_DYNAMIC_DEPENDENCY_AUDIT_TAG = 0x6FFFFEFB
_ELF_DYNAMIC_AUDIT_TAG = 0x6FFFFEFC
_ELF_DYNAMIC_FLAGS_1_TAG = 0x6FFFFFFB
_ELF_DYNAMIC_GNU_HASH_TAG = 0x6FFFFEF5
_ELF_DYNAMIC_VERSION_NEEDED_TAG = 0x6FFFFFFE
_ELF_DYNAMIC_VERSION_NEEDED_COUNT_TAG = 0x6FFFFFFF
_ELF_DYNAMIC_AUXILIARY_TAG = 0x7FFFFFFD
_ELF_DYNAMIC_FILTER_TAG = 0x7FFFFFFF
_ELF_DYNAMIC_UNSUPPORTED_LOADER_CONFIGURATION_TAG_NAMES = {
    _ELF_DYNAMIC_RPATH_TAG: "DT_RPATH",
    _ELF_DYNAMIC_RUNPATH_TAG: "DT_RUNPATH",
    _ELF_DYNAMIC_CONFIG_TAG: "DT_CONFIG",
    _ELF_DYNAMIC_DEPENDENCY_AUDIT_TAG: "DT_DEPAUDIT",
    _ELF_DYNAMIC_AUDIT_TAG: "DT_AUDIT",
}
_ELF_DYNAMIC_FLAGS_1_NO_DEFAULT_LIBRARY = 0x800
_MAX_ELF_PROGRAM_HEADERS = 1024
_MAX_ELF_INTERPRETER_BYTES = 4096
_MAX_ELF_DYNAMIC_ENTRIES = 4096
_MAX_ELF_LOADER_DEPENDENCIES = 256
_MAX_ELF_DYNAMIC_STRING_TABLE_BYTES = 1024 * 1024
_MAX_ELF_LIBRARY_NAME_BYTES = 255
_MAX_ELF_DYNAMIC_SYMBOLS = 262_144
_MAX_ELF_SYMBOL_NAME_BYTES = 4096
_MAX_ELF_VERSION_NEEDED_FILES = 256
_MAX_ELF_VERSION_REQUIREMENTS = 4096
_MAX_ELF_VERSION_NAME_BYTES = 255
_ELF_MAX_ADDRESS = (1 << 64) - 1
_ELF_LIBRARY_NAME_RE = re.compile(rb"^[A-Za-z0-9_][A-Za-z0-9._+~-]*$")
_ELF_VERSION_NAME_RE = re.compile(rb"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PE_DOS_MAGIC = b"MZ"
_PE_SIGNATURE = b"PE\0\0"
_PE_COFF_HEADER = struct.Struct("<HHIIIHH")
_PE_SECTION_HEADER = struct.Struct("<8sIIIIIIHHI")
_PE_EXPORT_DIRECTORY = struct.Struct("<IIHHIIIIIII")
_PE_IMPORT_DESCRIPTOR = struct.Struct("<5I")
_PE_DELAY_IMPORT_DESCRIPTOR = struct.Struct("<8I")
_PE_OPTIONAL_HEADER_64_MAGIC = 0x20B
_PE_FILE_EXECUTABLE_IMAGE = 0x2
_PE_FILE_DLL = 0x2000
_PE_MACHINE_BY_ARCHITECTURE = {
    "amd64": 0x8664,
    "arm64": 0xAA64,
}
_PE_OPTIONAL_HEADER_64_FIXED_SIZE = 112
_PE_OPTIONAL_HEADER_SIZE_OF_HEADERS_OFFSET = 60
_PE_OPTIONAL_HEADER_DATA_DIRECTORY_COUNT_OFFSET = 108
_PE_EXPORT_DIRECTORY_INDEX = 0
_PE_IMPORT_DIRECTORY_INDEX = 1
_PE_DELAY_IMPORT_DIRECTORY_INDEX = 13
_PE_MAX_ADDRESS = (1 << 32) - 1
_MAX_PE_HEADER_OFFSET = 1024 * 1024
_MAX_PE_OPTIONAL_HEADER_BYTES = 4096
_MAX_PE_SECTIONS = 96
_MAX_PE_DATA_DIRECTORIES = 16
_MAX_PE_EXPORT_DIRECTORY_BYTES = 1024 * 1024
_MAX_PE_EXPORTS = 65_536
_MAX_PE_IMPORT_DIRECTORY_BYTES = 1024 * 1024
_MAX_PE_IMPORT_LIBRARIES = 256
_MAX_PE_LIBRARY_NAME_BYTES = 255
_PE_LIBRARY_NAME_RE = re.compile(rb"^[A-Za-z0-9_][A-Za-z0-9._+-]*\.[Dd][Ll][Ll]$")
_WINDOWS_API_SET_LIBRARIES = frozenset(
    {
        "api-ms-win-crt-conio-l1-1-0.dll",
        "api-ms-win-crt-convert-l1-1-0.dll",
        "api-ms-win-crt-environment-l1-1-0.dll",
        "api-ms-win-crt-filesystem-l1-1-0.dll",
        "api-ms-win-crt-heap-l1-1-0.dll",
        "api-ms-win-crt-locale-l1-1-0.dll",
        "api-ms-win-crt-math-l1-1-0.dll",
        "api-ms-win-crt-multibyte-l1-1-0.dll",
        "api-ms-win-crt-process-l1-1-0.dll",
        "api-ms-win-crt-runtime-l1-1-0.dll",
        "api-ms-win-crt-stdio-l1-1-0.dll",
        "api-ms-win-crt-string-l1-1-0.dll",
        "api-ms-win-crt-time-l1-1-0.dll",
        "api-ms-win-crt-utility-l1-1-0.dll",
    }
)
_WINDOWS_SYSTEM_LIBRARIES = frozenset(
    {
        "advapi32.dll",
        "bcrypt.dll",
        "bcryptprimitives.dll",
        "cabinet.dll",
        "cfgmgr32.dll",
        "combase.dll",
        "comctl32.dll",
        "comdlg32.dll",
        "crypt32.dll",
        "cryptbase.dll",
        "dbghelp.dll",
        "dnsapi.dll",
        "dwmapi.dll",
        "gdi32.dll",
        "gdi32full.dll",
        "imm32.dll",
        "iphlpapi.dll",
        "kernel32.dll",
        "kernelbase.dll",
        "mpr.dll",
        "mswsock.dll",
        "msvcp_win.dll",
        "ncrypt.dll",
        "netapi32.dll",
        "normaliz.dll",
        "ntdll.dll",
        "ole32.dll",
        "oleaut32.dll",
        "powrprof.dll",
        "psapi.dll",
        "rpcrt4.dll",
        "secur32.dll",
        "setupapi.dll",
        "shell32.dll",
        "shlwapi.dll",
        "ucrtbase.dll",
        "user32.dll",
        "userenv.dll",
        "version.dll",
        "winhttp.dll",
        "wininet.dll",
        "winmm.dll",
        "wintrust.dll",
        "ws2_32.dll",
        "wtsapi32.dll",
    }
)
_WINDOWS_CPYTHON_RUNTIME_LIBRARIES = frozenset({"vcruntime140.dll", "vcruntime140_1.dll"})
_MANYLINUX_DYNAMIC_LOADER_BY_ARCHITECTURE = {
    "x86_64": "ld-linux-x86-64.so.2",
    "aarch64": "ld-linux-aarch64.so.1",
}
_MUSLLINUX_LIBC_BY_ARCHITECTURE = {
    "x86_64": "libc.musl-x86_64.so.1",
    "aarch64": "libc.musl-aarch64.so.1",
}
_MUSL_VERSION_RE = re.compile(rb"(?:^|\n)Version ([0-9]+)\.([0-9]+)(?:\.[0-9]+)?(?:\r?\n|$)")
_MACHO_MAGIC_64 = b"\xcf\xfa\xed\xfe"
_MACHO_HEADER_64 = struct.Struct("<8I")
_MACHO_LOAD_COMMAND = struct.Struct("<2I")
_MACHO_DYLIB_COMMAND = struct.Struct("<6I")
_MACHO_RPATH_COMMAND = struct.Struct("<3I")
_MACHO_VERSION_COMMAND = struct.Struct("<4I")
_MACHO_BUILD_VERSION_COMMAND = struct.Struct("<6I")
_MACHO_BUILD_TOOL_VERSION = struct.Struct("<2I")
_MACHO_CPU_BY_ARCHITECTURE = {
    "x86_64": 0x01000007,
    "arm64": 0x0100000C,
}
_MACHO_CPU_SUBTYPES_BY_ARCHITECTURE = {
    "x86_64": frozenset({3}),
    "arm64": frozenset({0, 1}),
}
_MACHO_CPU_SUBTYPE_BASE_MASK = 0x00FFFFFF
_MACHO_SHARED_OBJECT_TYPES = frozenset({6, 8})
_MACHO_LOAD_DYLIB = 0xC
_MACHO_ID_DYLIB = 0xD
_MACHO_LOAD_WEAK_DYLIB = 0x80000018
_MACHO_RPATH = 0x8000001C
_MACHO_REEXPORT_DYLIB = 0x8000001F
_MACHO_LAZY_LOAD_DYLIB = 0x20
_MACHO_LOAD_UPWARD_DYLIB = 0x80000023
_MACHO_DYNAMIC_LIBRARY_DEPENDENCY_COMMANDS = frozenset(
    {
        _MACHO_LOAD_DYLIB,
        _MACHO_LOAD_WEAK_DYLIB,
        _MACHO_REEXPORT_DYLIB,
        _MACHO_LAZY_LOAD_DYLIB,
        _MACHO_LOAD_UPWARD_DYLIB,
    }
)
_MACHO_DYLIB_COMMANDS = _MACHO_DYNAMIC_LIBRARY_DEPENDENCY_COMMANDS | {_MACHO_ID_DYLIB}
_MACHO_UNSUPPORTED_DYNAMIC_LINKER_COMMANDS = frozenset(
    {
        0x6,  # LC_LOADFVMLIB
        0x7,  # LC_IDFVMLIB
        0x9,  # LC_FVMFILE
        0xE,  # LC_LOAD_DYLINKER
        0xF,  # LC_ID_DYLINKER
        0x10,  # LC_PREBOUND_DYLIB
        0x27,  # LC_DYLD_ENVIRONMENT
    }
)
_MACHO_VERSION_MIN_MACOSX = 0x24
_MACHO_NON_MACOS_VERSION_COMMANDS = frozenset({0x25, 0x2F, 0x30})
_MACHO_BUILD_VERSION = 0x32
_MACHO_PLATFORM_MACOS = 1
_MAX_MACHO_LOAD_COMMANDS = 4096
_MAX_MACHO_LOAD_PATH_BYTES = 4096
_MACOS_SYSTEM_LIBRARY_PREFIXES = ("/usr/lib/", "/System/Library/")
_MACOS_RELOCATABLE_RPATH_PREFIXES = ("@loader_path", "@executable_path")
_MACOS_UNIVERSAL2_PLATFORM_TAG_RE = re.compile(r"^macosx_(?P<major>[0-9]+)_(?P<minor>[0-9]+)_universal2$")
_MACHO_FAT_MAGIC = b"\xca\xfe\xba\xbe"
_MACHO_FAT64_MAGIC = b"\xca\xfe\xba\xbf"
_MACHO_FAT_HEADER = struct.Struct(">2I")
_MACHO_FAT_ARCH = struct.Struct(">5I")
_MACHO_FAT64_ARCH = struct.Struct(">2I2Q2I")
_MAX_MACHO_FAT_ARCHITECTURES = 16


@dataclass(frozen=True)
class BuiltExtensionWheel:
    """The generated wheel and the descriptor embedded in it."""

    path: Path
    descriptor: DynamicExtensionDescriptor
    distribution_name: str
    distribution_version: str
    wheel_tag: str


@dataclass(frozen=True)
class _DependencyWheel:
    descriptor: DynamicExtensionDescriptor
    interpreter_tag: str
    platform_tag: str
    distribution_version: str
    requirements: tuple[Requirement, ...]


@dataclass(frozen=True)
class _WheelPlatformPolicy:
    family: str
    minimum_version: tuple[int, int] | None
    architecture: str


@dataclass(frozen=True)
class _PlatformBuildDetails:
    """Build-time facts needed to verify a wheel's platform promise."""

    platform_tag: str
    musl_version: tuple[int, int] | None


@dataclass(frozen=True)
class _ElfDynamicLinkage:
    """Bounded dynamic-library names declared by one ELF object."""

    needed: tuple[str, ...]
    filters: tuple[str, ...]
    auxiliaries: tuple[str, ...]
    soname: str | None
    versioned_symbols: tuple[tuple[str, str], ...]
    undefined_symbols: frozenset[str]

    @property
    def loader_dependencies(self) -> tuple[str, ...]:
        return self.needed + self.filters + self.auxiliaries


@dataclass(frozen=True)
class _PeSection:
    """One bounded PE section mapping."""

    raw_offset: int
    virtual_address: int
    raw_size: int
    mapped_size: int


def build_extension_wheel(
    *,
    artifact: str | Path,
    extension_name: str,
    output_directory: str | Path,
    platform_tag: str,
    trust_identity: str,
    license_expression: str,
    license_files: Iterable[str | Path],
    dependency_wheels: Iterable[str | Path] = (),
    dependency_trust_identities: Iterable[str] = (),
) -> BuiltExtensionWheel:
    """Build one platform-specific wheel from an already-built local artifact.

    ``artifact`` must be the exact, self-contained
    ``<extension_name>.duckdb_extension`` emitted by the Vane build.
    ``license_files`` must explicitly cover that artifact's redistributed
    source and binary dependencies, and ``license_expression`` must be the
    corresponding SPDX expression. ``dependency_wheels`` contains the complete
    transitive closure of exact, ordered, separately packaged dependency
    wheels. ``dependency_trust_identities`` is the exact explicit allowlist of
    unique signer identities used by that closure. The root descriptor is
    generated from the artifact using the currently installed Vane runtime,
    which pins the wheel to its Vane and DuckDB identities.
    """
    name = _validate_extension_name(extension_name)
    interpreter_tag = _extension_interpreter_tag()
    normalized_platform_tag = _validate_platform_tag(platform_tag)
    normalized_license_expression = _validate_license_expression(license_expression)
    artifact_path = Path(artifact).expanduser().resolve()
    expected_artifact_name = f"{name}.duckdb_extension"
    if artifact_path.name != expected_artifact_name:
        raise ValueError(f"artifact must be named {expected_artifact_name!r}: {artifact_path}")
    _validate_extension_artifact_size(artifact_path)
    import vane

    resolved_dependency_wheels = _read_dependency_wheels(dependency_wheels)
    _validate_dependency_trust_identities(
        dependency_trust_identities,
        resolved_dependency_wheels,
        description="extension wheel builder",
    )
    dependencies = _validate_dependency_descriptors(
        (dependency.descriptor for dependency in resolved_dependency_wheels),
        extension_name=name,
        runtime_source_id=vane.__git_revision__,
        runtime_vane_version=vane.__version__,
    )
    _validate_dependency_wheel_requirements(resolved_dependency_wheels)
    _validate_extension_artifact_size(artifact_path)
    descriptor = _create_descriptor(
        artifact_path,
        name=name,
        trust_identity=trust_identity,
        dependencies=tuple(_dependency_reference(dependency) for dependency in dependencies),
    )
    artifact_contents = _read_extension_artifact(artifact_path)
    _validate_descriptor(
        descriptor,
        artifact_contents=artifact_contents,
        extension_name=name,
        trust_identity=trust_identity,
        runtime_source_id=vane.__git_revision__,
        runtime_vane_version=vane.__version__,
        dependencies=dependencies,
    )
    _validate_extension_name_for_platform(name, descriptor.platform)
    _validate_artifact_platform_tag(descriptor.platform, normalized_platform_tag)
    platform_build_details = _platform_build_details_for_build(normalized_platform_tag)
    _validate_native_binary_platform(
        artifact_contents,
        normalized_platform_tag,
        description="extension artifact",
        platform_build_details=platform_build_details,
        interpreter_tag=interpreter_tag,
    )
    _validate_dependency_wheel_platforms(
        normalized_platform_tag,
        resolved_dependency_wheels,
        root_interpreter_tag=interpreter_tag,
    )

    vane_version = _validate_vane_version(vane.__version__)
    descriptor_digest = _descriptor_digest(descriptor)
    distribution_version = _extension_distribution_version(vane_version, descriptor)
    distribution_name = f"vane-extension-{name}"
    distribution_root = f"vane_extension_{name}-{distribution_version}"
    wheel_tag = f"{interpreter_tag}-none-{normalized_platform_tag}"
    output_path = Path(output_directory).expanduser().resolve() / f"{distribution_root}-{wheel_tag}.whl"
    provider_package = f"{name}_{descriptor_digest}"
    package_root = f"vane_extensions/{provider_package}"
    descriptor_name = f"{package_root}/{name}.dynamic-extension.json"
    artifact_name = f"{package_root}/{name}.duckdb_extension"
    dist_info_root = f"{distribution_root}.dist-info"
    platform_build_details_name = f"{dist_info_root}/{_PLATFORM_BUILD_DETAILS_FILENAME}"
    license_entries = _license_entries(
        license_files,
        dist_info_root,
        windows_paths=descriptor.platform.startswith("windows_"),
    )

    entries = {
        f"{package_root}/__init__.py": _provider_module_source(name).encode("utf-8"),
        descriptor_name: (descriptor.to_json() + "\n").encode("utf-8"),
        artifact_name: artifact_contents,
        f"{dist_info_root}/METADATA": _metadata(
            distribution_name,
            distribution_version,
            vane_version,
            normalized_license_expression,
            tuple(sorted(license_entries)),
            dist_info_root,
            tuple(
                (
                    dependency.name,
                    _extension_distribution_version(vane_version, dependency),
                )
                for dependency in dependencies
            ),
        ).encode("utf-8"),
        f"{dist_info_root}/WHEEL": _wheel_metadata(wheel_tag).encode("utf-8"),
        f"{dist_info_root}/entry_points.txt": _entry_points(name, provider_package).encode("utf-8"),
        platform_build_details_name: _platform_build_details_bytes(platform_build_details),
    }
    entries.update(license_entries)
    record_name = f"{dist_info_root}/RECORD"
    _validate_extension_wheel_entries_size(entries)
    _validate_extension_wheel_entries_count(entries, additional_members=1)
    entries[record_name] = _record(entries, record_name).encode("utf-8")
    _validate_extension_wheel_entries_size(entries)
    _validate_extension_wheel_entries_count(entries)
    _validate_wheel_path_component_lengths((output_path.name, *entries))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor_handle, temporary_name = tempfile.mkstemp(
        prefix=".vane-extension-wheel-",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(descriptor_handle)
    temporary_path = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary_path, mode="w", compression=zipfile.ZIP_DEFLATED) as wheel:
            for member_name, contents in sorted(entries.items()):
                wheel.writestr(_zip_info(member_name), contents)
            _validate_extension_wheel_archive_size(wheel)
        _validate_extension_wheel_size(temporary_path)
        temporary_path.chmod(0o644)
        try:
            os.link(temporary_path, output_path)
        except FileExistsError:
            output_metadata = output_path.lstat()
            if not stat.S_ISREG(output_metadata.st_mode) or output_metadata.st_nlink != 1:
                raise FileExistsError(
                    f"refusing to reuse a non-regular or multiply linked extension wheel at {output_path}"
                ) from None
            _validate_extension_wheel_size(output_path, description="existing extension wheel")
            if not _bounded_files_equal(output_path, temporary_path):
                raise FileExistsError(
                    f"refusing to replace a different extension wheel at {output_path}; "
                    "choose a new output directory or version"
                ) from None
        _normalize_extension_wheel_permissions(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return BuiltExtensionWheel(
        path=output_path,
        descriptor=descriptor,
        distribution_name=distribution_name,
        distribution_version=distribution_version,
        wheel_tag=wheel_tag,
    )


def _create_descriptor(
    artifact_path: Path,
    *,
    name: str,
    trust_identity: str,
    dependencies: tuple[DynamicExtensionDependency, ...],
) -> DynamicExtensionDescriptor:
    from vane.extensions import create_dynamic_extension_descriptor

    return create_dynamic_extension_descriptor(
        artifact_path,
        name=name,
        trust_identity=trust_identity,
        dependencies=dependencies,
    )


def _dependency_reference(descriptor: DynamicExtensionDescriptor) -> DynamicExtensionDependency:
    from vane.extensions import DynamicExtensionDependency

    return DynamicExtensionDependency(
        name=descriptor.name,
        extension_version=descriptor.extension_version,
        sha256=descriptor.sha256,
    )


def _read_dependency_wheels(values: Iterable[str | Path]) -> tuple[_DependencyWheel, ...]:
    if isinstance(values, (str, os.PathLike)):
        raise ValueError("dependency_wheels must be an iterable of wheel paths, not one path")
    try:
        unresolved_paths = tuple(islice(iter(values), _MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES + 1))
    except TypeError as exception:
        raise ValueError("dependency_wheels must be an iterable of wheel paths") from exception
    if len(unresolved_paths) > _MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES:
        raise ValueError(f"dependency_wheels contains more than {_MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES} wheel paths")
    if any(not isinstance(value, (str, os.PathLike)) for value in unresolved_paths):
        raise ValueError("dependency_wheels must contain only wheel paths")
    paths = tuple(Path(value).expanduser().resolve(strict=True) for value in unresolved_paths)
    return tuple(_read_dependency_wheel(path) for path in paths)


def _validate_dependency_trust_identities(
    values: Iterable[str],
    dependency_wheels: tuple[_DependencyWheel, ...],
    *,
    description: str,
) -> frozenset[str]:
    if isinstance(values, str):
        raise ValueError("dependency_trust_identities must be an iterable of identities, not one identity")
    try:
        identities = tuple(islice(iter(values), _MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES + 1))
    except TypeError as exception:
        raise ValueError("dependency_trust_identities must be an iterable of identities") from exception
    if len(identities) > _MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES:
        raise ValueError(
            f"dependency_trust_identities contains more than {_MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES} identities"
        )
    if any(not isinstance(identity, str) or not identity for identity in identities):
        raise ValueError("dependency_trust_identities must contain only non-empty strings")
    if len(set(identities)) != len(identities):
        raise ValueError("dependency_trust_identities must not contain duplicates")

    supplied = frozenset(identities)
    required = frozenset(dependency.descriptor.trust_identity for dependency in dependency_wheels)
    if supplied != required:
        raise ValueError(
            f"{description} dependency trust identities must be supplied explicitly and match exactly: "
            f"required={tuple(sorted(required))}, supplied={tuple(sorted(supplied))}"
        )
    return supplied


def _read_dependency_wheel(path: Path) -> _DependencyWheel:
    with snapshot_archive(
        path,
        max_bytes=_MAX_EXTENSION_WHEEL_BYTES,
        description="dependency extension wheel",
        size_limit_description=PUBLICATION_FILE_LIMIT_DESCRIPTION,
    ) as snapshot:
        return _read_dependency_wheel_snapshot(snapshot)


def _read_dependency_wheel_snapshot(snapshot: ArchiveSnapshot) -> _DependencyWheel:
    from vane.extensions import DynamicExtensionDescriptor, DynamicExtensionError

    path = snapshot.source_path
    try:
        filename_name, filename_version, build_tag, filename_tags = parse_wheel_filename(path.name)
    except InvalidWheelFilename as exception:
        raise ValueError(f"dependency wheel has an invalid filename: {path.name!r}") from exception
    if build_tag:
        raise ValueError(f"dependency extension wheel must not use a build tag: {path.name!r}")
    if len(filename_tags) != 1:
        raise ValueError(f"dependency extension wheel must have exactly one compatibility tag: {path.name!r}")
    filename_tag = next(iter(filename_tags))
    if _EXTENSION_INTERPRETER_TAG_RE.fullmatch(filename_tag.interpreter) is None or filename_tag.abi != "none":
        raise ValueError(
            f"dependency extension wheel must use exactly one supported CPython-none platform tag: {filename_tag}"
        )
    platform_tag = _validate_platform_tag(filename_tag.platform)

    try:
        with open_zip_snapshot(
            snapshot,
            max_members=_MAX_EXTENSION_WHEEL_MEMBERS,
            description="dependency extension wheel",
        ) as wheel:
            _validate_extension_wheel_archive_size(wheel, description="dependency extension wheel")
            names = wheel.namelist()
            _validate_wheel_path_component_lengths(
                (path.name, *names),
                description="dependency extension wheel",
            )
            descriptor_members = [name for name in names if name.casefold().endswith(".dynamic-extension.json")]
            artifact_members = [name for name in names if name.casefold().endswith(".duckdb_extension")]
            if len(descriptor_members) != 1 or len(artifact_members) != 1:
                raise ValueError(
                    "dependency extension wheel must contain exactly one descriptor and artifact: "
                    f"descriptors={descriptor_members}, artifacts={artifact_members}"
                )
            descriptor_contents = _read_extension_descriptor_member(
                wheel,
                descriptor_members[0],
                description="dependency extension wheel descriptor",
            )
            try:
                descriptor = DynamicExtensionDescriptor.from_json(descriptor_contents)
            except (TypeError, ValueError, DynamicExtensionError) as exception:
                raise ValueError("dependency extension wheel contains an invalid descriptor") from exception
            canonical_descriptor = (descriptor.to_json() + "\n").encode("utf-8")
            if descriptor_contents != canonical_descriptor:
                raise ValueError(
                    "dependency extension wheel descriptor must use canonical JSON followed by one newline"
                )

            _validate_extension_name(descriptor.name)
            descriptor_digest = _descriptor_digest(descriptor)
            expected_package_root = f"vane_extensions/{descriptor.name}_{descriptor_digest}"
            expected_descriptor = f"{expected_package_root}/{descriptor.name}.dynamic-extension.json"
            expected_artifact = f"{expected_package_root}/{descriptor.name}.duckdb_extension"
            if descriptor_members != [expected_descriptor] or artifact_members != [expected_artifact]:
                raise ValueError(
                    "dependency extension wheel descriptor and artifact paths do not match its descriptor: "
                    f"descriptors={descriptor_members}, artifacts={artifact_members}"
                )
            artifact_contents = wheel.read(expected_artifact)
            if hashlib.sha256(artifact_contents).hexdigest() != descriptor.sha256:
                raise ValueError("dependency extension wheel artifact SHA-256 does not match its descriptor")

            distribution_name = f"vane-extension-{descriptor.name}"
            distribution_version = _extension_distribution_version(descriptor.vane_version, descriptor)
            if filename_name != canonicalize_name(distribution_name) or filename_version != Version(
                distribution_version
            ):
                raise ValueError(
                    "dependency extension wheel filename does not match its descriptor-bound distribution: "
                    f"expected={distribution_name}==={distribution_version}, "
                    f"actual={filename_name}==={filename_version}"
                )
            distribution_root = f"vane_extension_{descriptor.name}-{distribution_version}"
            expected_metadata = f"{distribution_root}.dist-info/METADATA"
            expected_wheel_metadata = f"{distribution_root}.dist-info/WHEEL"
            expected_entry_points = f"{distribution_root}.dist-info/entry_points.txt"
            expected_record = f"{distribution_root}.dist-info/RECORD"
            expected_provider = f"{expected_package_root}/__init__.py"
            expected_platform_build_details = f"{distribution_root}.dist-info/{_PLATFORM_BUILD_DETAILS_FILENAME}"
            metadata_members = [
                name for name in names if name.count("/") == 1 and name.casefold().endswith(".dist-info/metadata")
            ]
            wheel_metadata_members = [
                name for name in names if name.count("/") == 1 and name.casefold().endswith(".dist-info/wheel")
            ]
            if metadata_members != [expected_metadata] or wheel_metadata_members != [expected_wheel_metadata]:
                raise ValueError(
                    "dependency extension wheel metadata paths do not match its descriptor-bound distribution"
                )
            platform_build_details = _read_platform_build_details_member(
                wheel,
                expected_platform_build_details,
                expected_platform_tag=platform_tag,
                description="dependency extension wheel",
            )
            _validate_native_binary_platform(
                artifact_contents,
                platform_tag,
                description="dependency extension artifact",
                platform_build_details=platform_build_details,
                interpreter_tag=filename_tag.interpreter,
            )
            metadata = _read_core_metadata(
                wheel,
                expected_metadata,
                description="dependency extension wheel METADATA",
            )
            if metadata.get_all("Name", []) != [distribution_name] or metadata.get_all("Version", []) != [
                distribution_version
            ]:
                raise ValueError("dependency extension wheel METADATA does not match its descriptor-bound distribution")
            _validate_metadata_version(metadata, description="dependency extension wheel")
            _validate_metadata_requires_python(metadata)
            _validate_metadata_license_expression(metadata)
            try:
                requirements = tuple(Requirement(value) for value in metadata.get_all("Requires-Dist", []))
            except InvalidRequirement as exception:
                raise ValueError("dependency extension wheel contains an invalid requirement") from exception
            if wheel.read(expected_wheel_metadata) != _wheel_metadata(str(filename_tag)).encode("utf-8"):
                raise ValueError("dependency extension wheel must contain its exact generated WHEEL metadata")
            if names.count(expected_provider) != 1 or wheel.read(expected_provider) != _provider_module_source(
                descriptor.name
            ).encode("utf-8"):
                raise ValueError("dependency extension wheel must contain its exact generated provider module")
            if names.count(expected_entry_points) != 1 or wheel.read(expected_entry_points) != _entry_points(
                descriptor.name,
                f"{descriptor.name}_{descriptor_digest}",
            ).encode("utf-8"):
                raise ValueError("dependency extension wheel must advertise its exact generated provider entry point")
            license_members = _metadata_license_file_members(
                metadata,
                dist_info_root=distribution_root,
                windows_paths=descriptor.platform.startswith("windows_"),
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
            _validate_wheel_record(wheel, names=names, record_name=expected_record)
            _validate_dependency_artifact_descriptor(artifact_contents, descriptor)
    except zipfile.BadZipFile as exception:
        raise ValueError(f"dependency extension wheel is not a valid ZIP archive: {path}") from exception

    _validate_artifact_platform_tag(descriptor.platform, platform_tag)
    return _DependencyWheel(
        descriptor=descriptor,
        interpreter_tag=filename_tag.interpreter,
        platform_tag=platform_tag,
        distribution_version=distribution_version,
        requirements=requirements,
    )


def _validate_dependency_artifact_descriptor(
    artifact_contents: bytes,
    descriptor: DynamicExtensionDescriptor,
) -> None:
    from vane.extensions import DynamicExtensionError

    with tempfile.TemporaryDirectory(prefix="vane-extension-wheel-dependency-") as temporary_directory:
        artifact_path = Path(temporary_directory) / f"{descriptor.name}.duckdb_extension"
        artifact_path.write_bytes(artifact_contents)
        try:
            inspected_descriptor = _create_descriptor(
                artifact_path,
                name=descriptor.name,
                trust_identity=descriptor.trust_identity,
                dependencies=descriptor.dependencies,
            )
        except DynamicExtensionError as exception:
            raise ValueError("dependency extension wheel artifact has invalid native footer metadata") from exception
    if inspected_descriptor != descriptor:
        raise ValueError("dependency extension wheel artifact native metadata does not match its descriptor")


def _validate_dependency_wheel_platforms(
    root_platform_tag: str,
    dependency_wheels: tuple[_DependencyWheel, ...],
    *,
    root_interpreter_tag: str,
) -> None:
    wheels_by_identity = {dependency.descriptor.identity: dependency for dependency in dependency_wheels}
    for dependency in dependency_wheels:
        if dependency.interpreter_tag != root_interpreter_tag:
            raise ValueError(
                f"dependency {dependency.descriptor.name!r} wheel interpreter tag "
                f"{dependency.interpreter_tag!r} does not match root extension wheel interpreter tag "
                f"{root_interpreter_tag!r}"
            )
        _validate_dependency_platform_tag(
            root_platform_tag,
            dependency.platform_tag,
            dependency_name=dependency.descriptor.name,
        )
    for parent in dependency_wheels:
        for dependency_reference in parent.descriptor.dependencies:
            dependency = wheels_by_identity.get(dependency_reference.identity)
            if dependency is None:
                continue
            _validate_dependency_platform_tag(
                parent.platform_tag,
                dependency.platform_tag,
                dependency_name=dependency.descriptor.name,
            )


def _validate_dependency_wheel_requirements(dependency_wheels: tuple[_DependencyWheel, ...]) -> None:
    wheels_by_identity = {dependency.descriptor.identity: dependency for dependency in dependency_wheels}
    for parent in dependency_wheels:
        expected_versions = {canonicalize_name("vane-ai"): parent.descriptor.vane_version}
        for dependency_reference in parent.descriptor.dependencies:
            dependency = wheels_by_identity.get(dependency_reference.identity)
            if dependency is None:
                raise ValueError(
                    f"dependency extension wheel {parent.descriptor.name!r} references "
                    f"{dependency_reference.identity}, but no exact dependency wheel was supplied"
                )
            dependency_name = canonicalize_name(f"vane-extension-{dependency.descriptor.name}")
            if dependency_name in expected_versions:
                raise ValueError(
                    f"dependency extension wheel {parent.descriptor.name!r} repeats dependency "
                    f"{dependency.descriptor.name!r}"
                )
            expected_versions[dependency_name] = dependency.distribution_version
        _validate_exact_requirements(
            parent.requirements,
            expected_versions,
            description=f"dependency extension wheel {parent.descriptor.name!r}",
        )


def _validate_dependency_descriptors(
    values: Iterable[DynamicExtensionDescriptor],
    *,
    extension_name: str,
    runtime_source_id: str,
    runtime_vane_version: str,
) -> tuple[DynamicExtensionDescriptor, ...]:
    from vane.extensions import DynamicExtensionDescriptor

    try:
        descriptors = tuple(islice(iter(values), _MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES + 1))
    except TypeError as exception:
        raise ValueError("dependency_descriptors must be an iterable of descriptors") from exception
    if any(not isinstance(descriptor, DynamicExtensionDescriptor) for descriptor in descriptors):
        raise ValueError("dependency_descriptors must contain DynamicExtensionDescriptor values")
    if len(descriptors) > _MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES:
        raise ValueError(
            f"dependency_descriptors contains more than {_MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES} dependencies"
        )

    seen_names: set[str] = set()
    for descriptor in descriptors:
        if len(descriptor.dependencies) > _MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES:
            raise ValueError(
                f"dependency descriptor {descriptor.name!r} contains more than "
                f"{_MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES} dependencies"
            )
        try:
            _validate_extension_name(descriptor.name)
        except ValueError as exception:
            raise ValueError(
                f"dependency descriptor name {descriptor.name!r} cannot form a normalized wheel name"
            ) from exception
        _validate_extension_name_for_platform(descriptor.name, descriptor.platform)
        if descriptor.name == extension_name:
            raise ValueError(f"extension wheel {extension_name!r} cannot depend on itself")
        if descriptor.name in seen_names:
            raise ValueError(f"dependency_descriptors contains extension name {descriptor.name!r} more than once")
        seen_names.add(descriptor.name)
        if descriptor.duckdb_source_id != runtime_source_id:
            raise RuntimeError(
                f"dependency {descriptor.identity} SourceID {descriptor.duckdb_source_id} does not match "
                f"installed Vane runtime SourceID {runtime_source_id}"
            )
        if descriptor.vane_version != runtime_vane_version:
            raise RuntimeError(
                f"dependency {descriptor.identity} Vane version {descriptor.vane_version} does not match "
                f"installed Vane runtime version {runtime_vane_version}"
            )
    _validate_dependency_graph(descriptors, extension_name=extension_name)
    return descriptors


def _validate_dependency_graph(
    descriptors: tuple[DynamicExtensionDescriptor, ...],
    *,
    extension_name: str,
) -> None:
    """Reject every dependency cycle represented by the supplied descriptors."""
    descriptor_by_name = {descriptor.name: descriptor for descriptor in descriptors}
    descriptor_names = set(descriptor_by_name)
    for descriptor in descriptors:
        for dependency in descriptor.dependencies:
            represented = descriptor_by_name.get(dependency.name)
            if represented is None:
                continue
            expected = _dependency_reference(represented)
            if dependency != expected:
                raise ValueError(
                    f"dependency {descriptor.identity} references {dependency.identity}, but supplied descriptor "
                    f"{represented.identity} has a different identity"
                )
    represented_names = {extension_name, *descriptor_names}
    dependencies_by_name: dict[str, set[str]] = {
        extension_name: set(descriptor_names),
        **{
            descriptor.name: {
                dependency.name for dependency in descriptor.dependencies if dependency.name in represented_names
            }
            for descriptor in descriptors
        },
    }

    remaining = {name: set(dependencies) for name, dependencies in dependencies_by_name.items()}
    ready = sorted(name for name, dependencies in remaining.items() if not dependencies)
    while ready:
        completed = ready.pop()
        remaining.pop(completed)
        for name, dependencies in remaining.items():
            if completed not in dependencies:
                continue
            dependencies.remove(completed)
            if not dependencies:
                ready.append(name)
    if remaining:
        raise ValueError(
            "dependency_descriptors contains a dependency cycle involving "
            + ", ".join(repr(name) for name in sorted(remaining))
        )


def _validate_descriptor(
    descriptor: DynamicExtensionDescriptor,
    *,
    artifact_contents: bytes,
    extension_name: str,
    trust_identity: str,
    runtime_source_id: str,
    runtime_vane_version: str,
    dependencies: tuple[DynamicExtensionDescriptor, ...],
) -> None:
    from vane.extensions import DynamicExtensionDescriptor

    if not isinstance(descriptor, DynamicExtensionDescriptor):
        raise RuntimeError("descriptor creation did not return a DynamicExtensionDescriptor")
    if descriptor.name != extension_name:
        raise RuntimeError(
            f"created descriptor name {descriptor.name!r} does not match requested extension {extension_name!r}"
        )
    if descriptor.trust_identity != trust_identity:
        raise RuntimeError(
            f"created descriptor trust identity {descriptor.trust_identity!r} does not match "
            f"requested identity {trust_identity!r}"
        )
    if descriptor.duckdb_source_id != runtime_source_id:
        raise RuntimeError(
            f"artifact SourceID {descriptor.duckdb_source_id} does not match installed Vane runtime "
            f"SourceID {runtime_source_id}"
        )
    if descriptor.vane_version != runtime_vane_version:
        raise RuntimeError(
            f"artifact Vane version {descriptor.vane_version} does not match installed Vane runtime "
            f"version {runtime_vane_version}"
        )
    if tuple(descriptor.dependencies) != tuple(_dependency_reference(value) for value in dependencies):
        raise RuntimeError("created descriptor does not preserve the requested dependency order")
    if any(dependency.platform != descriptor.platform for dependency in dependencies):
        raise RuntimeError("every dependency descriptor must target the same platform as the root artifact")
    if hashlib.sha256(artifact_contents).hexdigest() != descriptor.sha256:
        raise RuntimeError("artifact changed while creating its extension wheel")
    _validate_extension_descriptor_json_bounds(
        (descriptor.to_json() + "\n").encode("utf-8"),
        description="created extension descriptor",
    )


def _validate_extension_descriptor_json_bounds(contents: bytes, *, description: str) -> None:
    """Bound descriptor JSON bytes and lexical structure before object parsing."""
    if not isinstance(contents, bytes):
        raise ValueError(f"{description} must contain bytes")
    if len(contents) > _MAX_EXTENSION_DESCRIPTOR_BYTES:
        raise ValueError(
            f"{description} exceeds the bounded {_MAX_EXTENSION_DESCRIPTOR_BYTES // 1024} KiB descriptor limit"
        )
    if not contents.isascii():
        raise ValueError(f"{description} must use canonical ASCII JSON encoding")

    depth = 0
    object_count = 0
    separator_count = 0
    string_count = 0
    string_start = -1
    escaped = False
    scalar_bytes = 0
    for offset, value in enumerate(contents):
        if string_start >= 0:
            if offset - string_start > _MAX_EXTENSION_DESCRIPTOR_JSON_STRING_BYTES:
                raise ValueError(
                    f"{description} contains a JSON string longer than "
                    f"{_MAX_EXTENSION_DESCRIPTOR_JSON_STRING_BYTES} encoded bytes"
                )
            if escaped:
                escaped = False
            elif value == ord("\\"):
                escaped = True
            elif value == ord('"'):
                string_count += 1
                if string_count > _MAX_EXTENSION_DESCRIPTOR_JSON_STRINGS:
                    raise ValueError(
                        f"{description} contains more than {_MAX_EXTENSION_DESCRIPTOR_JSON_STRINGS} JSON strings"
                    )
                string_start = -1
            continue

        if value == ord('"'):
            string_start = offset + 1
            scalar_bytes = 0
        elif value in (ord("{"), ord("[")):
            depth += 1
            scalar_bytes = 0
            if depth > _MAX_EXTENSION_DESCRIPTOR_JSON_DEPTH:
                raise ValueError(
                    f"{description} exceeds the bounded JSON nesting depth of {_MAX_EXTENSION_DESCRIPTOR_JSON_DEPTH}"
                )
            if value == ord("{"):
                object_count += 1
                if object_count > _MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES + 1:
                    raise ValueError(
                        f"{description} contains more than {_MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES} "
                        "nested dependency objects"
                    )
        elif value in (ord("}"), ord("]")):
            depth -= 1
            scalar_bytes = 0
            if depth < 0:
                raise ValueError(f"{description} contains invalid JSON container structure")
        elif value in (ord(","), ord(":")):
            separator_count += 1
            scalar_bytes = 0
            if separator_count > _MAX_EXTENSION_DESCRIPTOR_JSON_SEPARATORS:
                raise ValueError(
                    f"{description} contains more than {_MAX_EXTENSION_DESCRIPTOR_JSON_SEPARATORS} JSON separators"
                )
        elif value in b" \t\r\n":
            scalar_bytes = 0
        else:
            scalar_bytes += 1
            if scalar_bytes > _MAX_EXTENSION_DESCRIPTOR_JSON_SCALAR_BYTES:
                raise ValueError(
                    f"{description} contains a JSON scalar longer than "
                    f"{_MAX_EXTENSION_DESCRIPTOR_JSON_SCALAR_BYTES} encoded bytes"
                )

    if string_start >= 0 or depth != 0:
        raise ValueError(f"{description} contains unterminated JSON structure")


def _read_extension_descriptor_member(
    wheel: zipfile.ZipFile,
    member_name: str,
    *,
    description: str,
) -> bytes:
    """Read one descriptor only after enforcing its dedicated size bound."""
    member = wheel.getinfo(member_name)
    if member.file_size > _MAX_EXTENSION_DESCRIPTOR_BYTES:
        raise ValueError(
            f"{description} exceeds the bounded {_MAX_EXTENSION_DESCRIPTOR_BYTES // 1024} KiB descriptor limit"
        )
    with wheel.open(member) as descriptor_file:
        contents = descriptor_file.read(_MAX_EXTENSION_DESCRIPTOR_BYTES + 1)
        if len(contents) > _MAX_EXTENSION_DESCRIPTOR_BYTES or descriptor_file.read(1):
            raise ValueError(
                f"{description} exceeds the bounded {_MAX_EXTENSION_DESCRIPTOR_BYTES // 1024} KiB descriptor limit"
            )
    _validate_extension_descriptor_json_bounds(contents, description=description)
    return contents


def _validate_extension_name(value: str) -> str:
    if _EXTENSION_NAME_RE.fullmatch(value) is None:
        raise ValueError("extension_name must use wheel-safe lowercase ASCII snake_case with single underscores")
    return value


def _validate_extension_name_for_platform(value: str, platform: str) -> None:
    if platform.startswith("windows_") and _windows_path_part_is_unsafe(value):
        raise ValueError(f"extension name {value!r} is a Windows-reserved file basename")


def _validate_platform_tag(value: str) -> str:
    if _WHEEL_PLATFORM_TAG_RE.fullmatch(value) is None:
        raise ValueError("platform_tag must contain lowercase ASCII letters, digits, and underscores")
    return value


def _validate_artifact_platform_tag(artifact_platform: str, wheel_platform_tag: str) -> None:
    allowed_tags = _ARTIFACT_PLATFORM_TAGS.get(artifact_platform)
    if allowed_tags is None:
        raise ValueError(
            f"extension artifact platform {artifact_platform!r} has no supported wheel platform tag policy"
        )
    if allowed_tags.fullmatch(wheel_platform_tag) is None:
        raise ValueError(
            f"wheel platform tag {wheel_platform_tag!r} does not match extension artifact platform "
            f"{artifact_platform!r}"
        )
    _wheel_platform_policy(wheel_platform_tag)


def _wheel_platform_policy(platform_tag: str) -> _WheelPlatformPolicy:
    versioned_match = _VERSIONED_WHEEL_PLATFORM_TAG_RE.fullmatch(platform_tag)
    if versioned_match is not None:
        family = versioned_match["family"]
        minimum_version = (int(versioned_match["major"]), int(versioned_match["minor"]))
        architecture = versioned_match["architecture"]
        canonical_version = versioned_match["major"] == str(minimum_version[0]) and versioned_match["minor"] == str(
            minimum_version[1]
        )
        if not canonical_version:
            raise ValueError(
                f"wheel platform tag {platform_tag!r} version components must use canonical decimal spelling"
            )
        if family == "manylinux":
            policy_floor = _MANYLINUX_POLICY_FLOORS.get(architecture)
            if policy_floor is None or minimum_version < policy_floor:
                raise ValueError(f"manylinux wheel platform tag {platform_tag!r} is below the supported policy floor")
            manylinux_policy(minimum_version, architecture)
        if family == "macosx":
            major, minor = minimum_version
            canonical_macos_tag = (architecture == "arm64" and major >= 11 and minor == 0) or (
                architecture == "x86_64" and ((major == 10 and 4 <= minor <= 16) or (major >= 11 and minor == 0))
            )
            if not canonical_macos_tag:
                raise ValueError(
                    f"macOS wheel platform tag {platform_tag!r} is not a canonical architecture-specific tag"
                )
            if minimum_version == (10, 16):
                minimum_version = (11, 0)
        return _WheelPlatformPolicy(
            family=family,
            minimum_version=minimum_version,
            architecture=architecture,
        )
    windows_match = _WINDOWS_WHEEL_PLATFORM_TAG_RE.fullmatch(platform_tag)
    if windows_match is not None:
        return _WheelPlatformPolicy(
            family="windows",
            minimum_version=None,
            architecture=windows_match["architecture"],
        )
    raise ValueError(f"wheel platform tag has no supported compatibility policy: {platform_tag!r}")


def _platform_build_details_for_build(platform_tag: str) -> _PlatformBuildDetails:
    policy = _wheel_platform_policy(platform_tag)
    musl_version = None
    if policy.family == "musllinux":
        musl_version = _current_musl_version()
        if musl_version is None:
            raise ValueError(
                f"musllinux wheel platform tag {platform_tag!r} requires a dynamically linked musl build runtime"
            )
        if musl_version != policy.minimum_version:
            raise ValueError(
                f"musllinux wheel platform tag {platform_tag!r} must match the build runtime's musl "
                f"{musl_version[0]}.{musl_version[1]} baseline exactly"
            )
    return _PlatformBuildDetails(platform_tag=platform_tag, musl_version=musl_version)


def _platform_build_details_bytes(details: _PlatformBuildDetails) -> bytes:
    document = {
        "musl_version": list(details.musl_version) if details.musl_version is not None else None,
        "platform_tag": details.platform_tag,
    }
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _parse_platform_build_details(
    contents: bytes,
    *,
    expected_platform_tag: str,
    description: str,
) -> _PlatformBuildDetails:
    if len(contents) > _MAX_PLATFORM_BUILD_DETAILS_BYTES:
        raise ValueError(f"{description} platform build details exceed their bounded maximum size")
    try:
        document = json.loads(contents)
    except (TypeError, ValueError) as exception:
        raise ValueError(f"{description} contains invalid platform build details JSON") from exception
    if not isinstance(document, dict) or set(document) != {"musl_version", "platform_tag"}:
        raise ValueError(f"{description} platform build details have an invalid structure")

    platform_tag = document["platform_tag"]
    if platform_tag != expected_platform_tag:
        raise ValueError(
            f"{description} platform build details tag {platform_tag!r} does not match wheel tag "
            f"{expected_platform_tag!r}"
        )
    raw_musl_version = document["musl_version"]
    if raw_musl_version is None:
        musl_version = None
    elif (
        isinstance(raw_musl_version, list)
        and len(raw_musl_version) == 2
        and all(isinstance(component, int) and not isinstance(component, bool) for component in raw_musl_version)
        and all(0 <= component <= 9999 for component in raw_musl_version)
    ):
        musl_version = (raw_musl_version[0], raw_musl_version[1])
    else:
        raise ValueError(f"{description} platform build details contain an invalid musl version")

    details = _PlatformBuildDetails(platform_tag=platform_tag, musl_version=musl_version)
    if contents != _platform_build_details_bytes(details):
        raise ValueError(f"{description} platform build details must use canonical JSON followed by one newline")
    _validate_platform_build_details(details, description=description)
    return details


def _validate_platform_build_details(details: _PlatformBuildDetails, *, description: str) -> None:
    policy = _wheel_platform_policy(details.platform_tag)
    if policy.family == "musllinux":
        if details.musl_version != policy.minimum_version:
            raise ValueError(
                f"{description} musl build baseline {details.musl_version!r} does not match platform tag "
                f"{details.platform_tag!r}"
            )
    elif details.musl_version is not None:
        raise ValueError(f"{description} must not declare a musl build baseline for {details.platform_tag!r}")


def _read_bounded_wheel_member(
    wheel: zipfile.ZipFile,
    member_name: str,
    *,
    max_bytes: int,
    description: str,
) -> bytes:
    members = [member for member in wheel.infolist() if member.filename == member_name]
    if len(members) != 1:
        raise ValueError(f"{description} must contain exactly one {member_name!r}")
    if members[0].file_size > max_bytes:
        raise ValueError(f"{description} exceeds its bounded maximum size")
    return wheel.read(members[0])


def _read_platform_build_details_member(
    wheel: zipfile.ZipFile,
    member_name: str,
    *,
    expected_platform_tag: str,
    description: str,
) -> _PlatformBuildDetails:
    contents = _read_bounded_wheel_member(
        wheel,
        member_name,
        max_bytes=_MAX_PLATFORM_BUILD_DETAILS_BYTES,
        description=f"{description} platform build details",
    )
    return _parse_platform_build_details(
        contents,
        expected_platform_tag=expected_platform_tag,
        description=description,
    )


def _read_core_metadata(wheel: zipfile.ZipFile, member_name: str, *, description: str):
    contents = _read_bounded_wheel_member(
        wheel,
        member_name,
        max_bytes=_MAX_CORE_METADATA_BYTES,
        description=description,
    )
    _validate_core_metadata_shape(contents, description=description)
    return BytesParser(policy=default).parsebytes(contents)


def _validate_core_metadata_shape(contents: bytes, *, description: str) -> None:
    line_count = 0
    header_count = 0
    in_headers = True
    have_header = False
    for line in _core_metadata_lines(contents):
        line_count += 1
        if line_count > _MAX_CORE_METADATA_LINES:
            raise ValueError(f"{description} contains more than {_MAX_CORE_METADATA_LINES} lines")
        if len(line) > _MAX_CORE_METADATA_LINE_BYTES:
            raise ValueError(f"{description} contains a line longer than {_MAX_CORE_METADATA_LINE_BYTES} bytes")
        if not in_headers:
            continue
        if not line:
            in_headers = False
        elif line.startswith((b" ", b"\t")):
            if not have_header:
                raise ValueError(f"{description} starts with an invalid continuation line")
        else:
            have_header = True
            header_count += 1
            if header_count > _MAX_CORE_METADATA_HEADERS:
                raise ValueError(f"{description} contains more than {_MAX_CORE_METADATA_HEADERS} headers")


def _core_metadata_lines(contents: bytes) -> Iterable[bytes]:
    line_start = 0
    offset = 0
    while offset < len(contents):
        if contents[offset] not in (10, 13):
            offset += 1
            continue
        yield contents[line_start:offset]
        if contents[offset] == 13 and offset + 1 < len(contents) and contents[offset + 1] == 10:
            offset += 1
        offset += 1
        line_start = offset
    if line_start < len(contents):
        yield contents[line_start:]


def _validate_native_binary_platform(
    contents: bytes,
    platform_tag: str,
    *,
    description: str,
    platform_build_details: _PlatformBuildDetails,
    interpreter_tag: str,
) -> None:
    if platform_build_details.platform_tag != platform_tag:
        raise ValueError(
            f"{description} platform build details tag {platform_build_details.platform_tag!r} does not match "
            f"platform tag {platform_tag!r}"
        )
    _validate_platform_build_details(platform_build_details, description=description)
    policy = _wheel_platform_policy(platform_tag)
    if policy.family in {"manylinux", "musllinux"}:
        _validate_linux_elf_platform(
            contents,
            platform_tag,
            description=description,
            platform_build_details=platform_build_details,
        )
    elif policy.family == "macosx":
        _validate_macos_binary_platform(contents, platform_tag, description=description)
    elif policy.family == "windows":
        _validate_windows_pe_platform(
            contents,
            platform_tag,
            description=description,
            interpreter_tag=interpreter_tag,
        )


def _validate_windows_pe_platform(
    contents: bytes,
    platform_tag: str,
    *,
    description: str,
    interpreter_tag: str,
) -> tuple[str, ...]:
    policy = _wheel_platform_policy(platform_tag)
    if policy.family != "windows":
        raise ValueError(f"Windows binary validation requires a Windows platform tag, not {platform_tag!r}")
    if len(contents) < 64 or not contents.startswith(_PE_DOS_MAGIC):
        raise ValueError(f"{description} must be a PE DLL for platform tag {platform_tag!r}")
    pe_header_offset = int.from_bytes(contents[60:64], "little")
    if not 64 <= pe_header_offset <= _MAX_PE_HEADER_OFFSET:
        raise ValueError(f"{description} contains an invalid PE header offset")
    coff_header_offset = pe_header_offset + len(_PE_SIGNATURE)
    optional_header_offset = coff_header_offset + _PE_COFF_HEADER.size
    if optional_header_offset > len(contents) or contents[pe_header_offset:coff_header_offset] != _PE_SIGNATURE:
        raise ValueError(f"{description} contains an invalid or truncated PE signature")

    machine, section_count, _timestamp, _symbol_table, _symbol_count, optional_header_size, characteristics = (
        _PE_COFF_HEADER.unpack_from(contents, coff_header_offset)
    )
    expected_machine = _PE_MACHINE_BY_ARCHITECTURE[policy.architecture]
    if machine != expected_machine:
        raise ValueError(
            f"{description} PE machine {machine:#x} does not match platform architecture {policy.architecture!r}"
        )
    required_characteristics = _PE_FILE_EXECUTABLE_IMAGE | _PE_FILE_DLL
    if characteristics & required_characteristics != required_characteristics:
        raise ValueError(f"{description} must use the PE executable DLL file type")
    if not 0 < section_count <= _MAX_PE_SECTIONS:
        raise ValueError(f"{description} contains an invalid PE section count")
    if not _PE_OPTIONAL_HEADER_64_FIXED_SIZE <= optional_header_size <= _MAX_PE_OPTIONAL_HEADER_BYTES:
        raise ValueError(f"{description} contains an invalid PE optional-header size")
    optional_header_end = optional_header_offset + optional_header_size
    section_table_end = optional_header_end + section_count * _PE_SECTION_HEADER.size
    if section_table_end > len(contents):
        raise ValueError(f"{description} has truncated PE optional or section headers")
    if int.from_bytes(contents[optional_header_offset : optional_header_offset + 2], "little") != (
        _PE_OPTIONAL_HEADER_64_MAGIC
    ):
        raise ValueError(f"{description} must use a 64-bit PE32+ optional header")

    size_of_headers = int.from_bytes(
        contents[
            optional_header_offset + _PE_OPTIONAL_HEADER_SIZE_OF_HEADERS_OFFSET : optional_header_offset
            + _PE_OPTIONAL_HEADER_SIZE_OF_HEADERS_OFFSET
            + 4
        ],
        "little",
    )
    if size_of_headers < section_table_end or size_of_headers > len(contents):
        raise ValueError(f"{description} contains an invalid PE SizeOfHeaders")
    data_directory_count = int.from_bytes(
        contents[
            optional_header_offset + _PE_OPTIONAL_HEADER_DATA_DIRECTORY_COUNT_OFFSET : optional_header_offset
            + _PE_OPTIONAL_HEADER_DATA_DIRECTORY_COUNT_OFFSET
            + 4
        ],
        "little",
    )
    data_directories_size = data_directory_count * 8
    if (
        data_directory_count > _MAX_PE_DATA_DIRECTORIES
        or _PE_OPTIONAL_HEADER_64_FIXED_SIZE + data_directories_size > optional_header_size
    ):
        raise ValueError(f"{description} contains an invalid PE data-directory count")

    sections: list[_PeSection] = []
    for index in range(section_count):
        section_offset = optional_header_end + index * _PE_SECTION_HEADER.size
        (
            _name,
            virtual_size,
            virtual_address,
            raw_size,
            raw_offset,
            _relocations,
            _line_numbers,
            _relocation_count,
            _line_number_count,
            _section_characteristics,
        ) = _PE_SECTION_HEADER.unpack_from(contents, section_offset)
        mapped_size = max(virtual_size, raw_size)
        if (
            mapped_size == 0
            or virtual_address < size_of_headers
            or mapped_size > _PE_MAX_ADDRESS + 1 - virtual_address
            or (raw_size != 0 and (raw_offset < size_of_headers or raw_offset + raw_size > len(contents)))
        ):
            raise ValueError(f"{description} contains an out-of-bounds PE section")
        sections.append(
            _PeSection(
                raw_offset=raw_offset,
                virtual_address=virtual_address,
                raw_size=raw_size,
                mapped_size=mapped_size,
            )
        )
    if not any(section.raw_size for section in sections):
        raise ValueError(f"{description} must contain at least one file-backed PE section")
    _validate_pe_section_mappings(sections, description=description)

    data_directory_offset = optional_header_offset + _PE_OPTIONAL_HEADER_64_FIXED_SIZE

    def data_directory(index: int) -> tuple[int, int]:
        if index >= data_directory_count:
            return 0, 0
        return struct.unpack_from("<II", contents, data_directory_offset + index * 8)

    _validate_pe_export_forwarders(
        contents,
        sections,
        size_of_headers=size_of_headers,
        directory=data_directory(_PE_EXPORT_DIRECTORY_INDEX),
        description=description,
    )
    imports = _pe_import_library_names(
        contents,
        sections,
        size_of_headers=size_of_headers,
        directory=data_directory(_PE_IMPORT_DIRECTORY_INDEX),
        delay=False,
        description=description,
    )
    delay_imports = _pe_import_library_names(
        contents,
        sections,
        size_of_headers=size_of_headers,
        directory=data_directory(_PE_DELAY_IMPORT_DIRECTORY_INDEX),
        delay=True,
        description=description,
    )
    dependencies = tuple(dict.fromkeys((*imports, *delay_imports)))
    allowed_libraries = _windows_policy_external_libraries(interpreter_tag)
    unsupported_libraries = tuple(sorted(set(dependencies).difference(allowed_libraries)))
    if unsupported_libraries:
        raise ValueError(f"{description} requires non-policy Windows DLLs: {unsupported_libraries}")
    return dependencies


def _validate_pe_section_mappings(sections: Iterable[_PeSection], *, description: str) -> None:
    resolved_sections = tuple(sections)
    raw_ranges = sorted(
        (section.raw_offset, section.raw_offset + section.raw_size) for section in resolved_sections if section.raw_size
    )
    virtual_ranges = sorted(
        (section.virtual_address, section.virtual_address + section.mapped_size) for section in resolved_sections
    )
    if any(start < previous_end for (_, previous_end), (start, _) in zip(raw_ranges, raw_ranges[1:])):
        raise ValueError(f"{description} contains overlapping file-backed PE sections")
    if any(start < previous_end for (_, previous_end), (start, _) in zip(virtual_ranges, virtual_ranges[1:])):
        raise ValueError(f"{description} contains overlapping virtual PE sections")


def _validate_pe_export_forwarders(
    contents: bytes,
    sections: Iterable[_PeSection],
    *,
    size_of_headers: int,
    directory: tuple[int, int],
    description: str,
) -> None:
    directory_rva, directory_size = directory
    if directory_rva == 0 and directory_size == 0:
        return
    if directory_rva == 0 or directory_size == 0:
        raise ValueError(f"{description} contains incomplete PE export directory metadata")
    if not _PE_EXPORT_DIRECTORY.size <= directory_size <= _MAX_PE_EXPORT_DIRECTORY_BYTES:
        raise ValueError(f"{description} contains an invalid PE export directory size")
    directory_start, _directory_end = _pe_file_range_for_rva_range(
        sections,
        size_of_headers=size_of_headers,
        rva=directory_rva,
        size=directory_size,
        description=description,
        range_description="export directory",
    )
    export_fields = _PE_EXPORT_DIRECTORY.unpack_from(contents, directory_start)
    function_count = export_fields[6]
    name_count = export_fields[7]
    function_table_rva = export_fields[8]
    if function_count > _MAX_PE_EXPORTS or name_count > function_count:
        raise ValueError(f"{description} contains an invalid PE export count")
    if function_count == 0:
        return
    if function_table_rva == 0:
        raise ValueError(f"{description} contains a null PE export address table")
    function_table_start, _function_table_end = _pe_file_range_for_rva_range(
        sections,
        size_of_headers=size_of_headers,
        rva=function_table_rva,
        size=function_count * 4,
        description=description,
        range_description="export address table",
    )
    directory_end_rva = directory_rva + directory_size
    for index in range(function_count):
        function_rva = struct.unpack_from("<I", contents, function_table_start + index * 4)[0]
        if directory_rva <= function_rva < directory_end_rva:
            raise ValueError(f"{description} contains a PE export forwarder")


def _pe_import_library_names(
    contents: bytes,
    sections: Iterable[_PeSection],
    *,
    size_of_headers: int,
    directory: tuple[int, int],
    delay: bool,
    description: str,
) -> tuple[str, ...]:
    directory_rva, directory_size = directory
    directory_description = "delay-import" if delay else "import"
    if directory_rva == 0 and directory_size == 0:
        return ()
    if directory_rva == 0 or directory_size == 0:
        raise ValueError(f"{description} contains incomplete PE {directory_description} directory metadata")
    descriptor = _PE_DELAY_IMPORT_DESCRIPTOR if delay else _PE_IMPORT_DESCRIPTOR
    if not descriptor.size <= directory_size <= _MAX_PE_IMPORT_DIRECTORY_BYTES:
        raise ValueError(f"{description} contains an invalid PE {directory_description} directory size")
    directory_start, directory_end = _pe_file_range_for_rva_range(
        sections,
        size_of_headers=size_of_headers,
        rva=directory_rva,
        size=directory_size,
        description=description,
        range_description=f"{directory_description} directory",
    )
    names: list[str] = []
    terminated = False
    for descriptor_offset in range(directory_start, directory_end - descriptor.size + 1, descriptor.size):
        values = descriptor.unpack_from(contents, descriptor_offset)
        if not any(values):
            terminated = True
            break
        if len(names) >= _MAX_PE_IMPORT_LIBRARIES:
            raise ValueError(f"{description} declares too many PE import libraries")
        if delay:
            attributes = values[0]
            if attributes != 1:
                raise ValueError(f"{description} contains a non-RVA PE delay-import descriptor")
            name_rva = values[1]
        else:
            name_rva = values[3]
        names.append(
            _pe_library_name(
                contents,
                sections,
                size_of_headers=size_of_headers,
                rva=name_rva,
                description=description,
            )
        )
    if not terminated:
        raise ValueError(f"{description} PE {directory_description} directory has no terminating descriptor")
    if len(set(names)) != len(names):
        raise ValueError(f"{description} contains duplicate PE {directory_description} libraries")
    return tuple(names)


def _pe_library_name(
    contents: bytes,
    sections: Iterable[_PeSection],
    *,
    size_of_headers: int,
    rva: int,
    description: str,
) -> str:
    if rva == 0:
        raise ValueError(f"{description} contains a null PE import-library name address")
    string_offset, available_size = _pe_file_offset_and_available_size(
        sections,
        size_of_headers=size_of_headers,
        rva=rva,
        description=description,
        range_description="import-library name",
    )
    candidate = contents[string_offset : string_offset + min(available_size, _MAX_PE_LIBRARY_NAME_BYTES + 1)]
    terminator = candidate.find(b"\0")
    if terminator < 0:
        raise ValueError(f"{description} contains an unterminated or oversized PE import-library name")
    raw_name = candidate[:terminator]
    if _PE_LIBRARY_NAME_RE.fullmatch(raw_name) is None:
        raise ValueError(f"{description} contains an invalid PE import-library name")
    return raw_name.decode("ascii").lower()


def _pe_file_range_for_rva_range(
    sections: Iterable[_PeSection],
    *,
    size_of_headers: int,
    rva: int,
    size: int,
    description: str,
    range_description: str,
) -> tuple[int, int]:
    if size <= 0 or rva > _PE_MAX_ADDRESS or size > _PE_MAX_ADDRESS + 1 - rva:
        raise ValueError(f"{description} contains an out-of-bounds PE {range_description}")
    rva_end = rva + size
    mappings: set[tuple[int, int]] = set()
    if rva_end <= size_of_headers:
        mappings.add((rva, rva_end))
    for section in sections:
        section_end = section.virtual_address + section.raw_size
        if rva >= section.virtual_address and rva_end <= section_end:
            file_start = section.raw_offset + rva - section.virtual_address
            mappings.add((file_start, file_start + size))
    if len(mappings) != 1:
        raise ValueError(f"{description} PE {range_description} has no unique file-backed RVA mapping")
    return next(iter(mappings))


def _pe_file_offset_and_available_size(
    sections: Iterable[_PeSection],
    *,
    size_of_headers: int,
    rva: int,
    description: str,
    range_description: str,
) -> tuple[int, int]:
    mappings: set[tuple[int, int]] = set()
    if rva < size_of_headers:
        mappings.add((rva, size_of_headers - rva))
    for section in sections:
        section_end = section.virtual_address + section.raw_size
        if section.virtual_address <= rva < section_end:
            relative_offset = rva - section.virtual_address
            mappings.add((section.raw_offset + relative_offset, section.raw_size - relative_offset))
    if len(mappings) != 1:
        raise ValueError(f"{description} PE {range_description} has no unique file-backed RVA mapping")
    return next(iter(mappings))


def _windows_policy_external_libraries(interpreter_tag: str) -> frozenset[str]:
    if _EXTENSION_INTERPRETER_TAG_RE.fullmatch(interpreter_tag) is None:
        raise ValueError(f"unsupported interpreter tag for Windows DLL policy: {interpreter_tag!r}")
    python_library = f"python{interpreter_tag.removeprefix('cp')}.dll"
    return (
        _WINDOWS_SYSTEM_LIBRARIES | _WINDOWS_API_SET_LIBRARIES | _WINDOWS_CPYTHON_RUNTIME_LIBRARIES | {python_library}
    )


def _validate_linux_elf_platform(
    contents: bytes,
    platform_tag: str,
    *,
    description: str,
    platform_build_details: _PlatformBuildDetails | None = None,
) -> tuple[int, int] | None:
    policy = _wheel_platform_policy(platform_tag)
    if policy.family not in {"manylinux", "musllinux"}:
        return None
    if len(contents) < 20 or not contents.startswith(_ELF_MAGIC):
        raise ValueError(f"{description} must be an ELF shared object for platform tag {platform_tag!r}")
    if contents[4] != 2 or contents[5] != 1:
        raise ValueError(f"{description} must use a 64-bit little-endian ELF header")
    if int.from_bytes(contents[16:18], "little") != _ELF_SHARED_OBJECT_TYPE:
        raise ValueError(f"{description} must use the ELF shared-object file type")
    expected_machine = _ELF_MACHINE_BY_ARCHITECTURE[policy.architecture]
    actual_machine = int.from_bytes(contents[18:20], "little")
    if actual_machine != expected_machine:
        raise ValueError(
            f"{description} ELF machine {actual_machine} does not match platform architecture {policy.architecture!r}"
        )

    dynamic_linkage = _parse_elf_dynamic_linkage(contents, description=description)
    external_libraries = _linux_policy_external_libraries(policy)
    loader_dependencies = dynamic_linkage.loader_dependencies
    unexpected_libraries = tuple(sorted(set(loader_dependencies).difference(external_libraries)))
    if unexpected_libraries:
        raise ValueError(f"{description} requires non-policy ELF shared libraries: {unexpected_libraries}")

    if policy.family == "manylinux":
        blacklisted_symbols = _manylinux_policy_blacklisted_undefined_symbols(
            policy,
            dynamic_linkage.needed,
            dynamic_linkage.undefined_symbols,
        )
        if blacklisted_symbols:
            raise ValueError(
                f"{description} requires blacklisted undefined ELF symbols outside platform policy "
                f"{platform_tag!r}: {blacklisted_symbols}"
            )

    allowed_versioned_symbols = (
        _manylinux_policy_versioned_symbols(policy) if policy.family == "manylinux" else frozenset()
    )
    unsupported_versioned_symbols: set[str] = set()
    unsupported_glibc_versions: dict[str, tuple[int, ...]] = {}
    actual_floor: tuple[int, int] | None = None
    has_glibc_version_requirement = False
    for _library, versioned_symbol in dynamic_linkage.versioned_symbols:
        match = _GLIBC_VERSION_NAME_RE.fullmatch(versioned_symbol)
        if match is None:
            if versioned_symbol.startswith("GLIBC_"):
                has_glibc_version_requirement = True
            if versioned_symbol not in allowed_versioned_symbols:
                unsupported_versioned_symbols.add(versioned_symbol)
            continue
        has_glibc_version_requirement = True
        components = tuple(match[1].split("."))
        if any(len(component) > _MAX_LIBC_VERSION_COMPONENT_DIGITS for component in components):
            raise ValueError(f"{description} contains an invalid glibc version requirement")
        complete_requirement = tuple(int(component) for component in components)
        requirement = complete_requirement[:2]
        if actual_floor is None or requirement > actual_floor:
            actual_floor = requirement
        if policy.family == "manylinux" and versioned_symbol not in allowed_versioned_symbols:
            unsupported_glibc_versions[versioned_symbol] = complete_requirement
    if unsupported_glibc_versions:
        highest_requirement = max(unsupported_glibc_versions, key=unsupported_glibc_versions.__getitem__)
        raise ValueError(
            f"{description} requires glibc {highest_requirement.removeprefix('GLIBC_')}, which is outside "
            f"exact platform policy {platform_tag!r}: {tuple(sorted(unsupported_glibc_versions))}"
        )
    if unsupported_versioned_symbols:
        raise ValueError(
            f"{description} requires versioned ELF symbols outside platform policy {platform_tag!r}: "
            f"{tuple(sorted(unsupported_versioned_symbols))}"
        )
    if policy.family == "musllinux":
        if platform_build_details is None or platform_build_details.musl_version != policy.minimum_version:
            raise ValueError(
                f"{description} requires build-time musl baseline details matching platform tag {platform_tag!r}"
            )
        if has_glibc_version_requirement or "libc.so.6" in loader_dependencies:
            raise ValueError(f"{description} carries glibc requirements but uses musllinux tag {platform_tag!r}")
        return None
    if any(library.startswith(("libc.musl-", "ld-musl-")) for library in loader_dependencies):
        raise ValueError(f"{description} carries musl requirements but uses manylinux tag {platform_tag!r}")
    if "libc.so.6" in loader_dependencies and actual_floor is None:
        raise ValueError(f"{description} glibc floor cannot be established from its ELF requirements")

    return actual_floor


def _parse_elf_dynamic_linkage(contents: bytes, *, description: str) -> _ElfDynamicLinkage:
    if len(contents) < _ELF_HEADER_64.size:
        raise ValueError(f"{description} has a truncated ELF header")
    fields = _ELF_HEADER_64.unpack_from(contents)
    identifier = fields[0]
    if (
        not identifier.startswith(_ELF_MAGIC)
        or identifier[4] != 2
        or identifier[5] != 1
        or identifier[6] != 1
        or fields[3] != 1
    ):
        raise ValueError(f"{description} must use a current 64-bit little-endian ELF header")

    program_header_offset = fields[5]
    elf_header_size = fields[8]
    program_header_size = fields[9]
    program_header_count = fields[10]
    if elf_header_size != _ELF_HEADER_64.size:
        raise ValueError(f"{description} contains an invalid ELF header size")
    if program_header_size != _ELF_PROGRAM_HEADER_64.size:
        raise ValueError(f"{description} contains an invalid ELF program-header size")
    if not 0 < program_header_count <= _MAX_ELF_PROGRAM_HEADERS:
        raise ValueError(f"{description} contains an invalid ELF program-header count")
    program_headers_end = program_header_offset + program_header_size * program_header_count
    if program_header_offset < elf_header_size or program_headers_end > len(contents):
        raise ValueError(f"{description} has an out-of-bounds ELF program-header table")

    load_segments: list[tuple[int, int, int]] = []
    dynamic_segments: list[tuple[int, int, int]] = []
    for index in range(program_header_count):
        values = _ELF_PROGRAM_HEADER_64.unpack_from(contents, program_header_offset + index * program_header_size)
        segment_type = values[0]
        file_offset = values[2]
        virtual_address = values[3]
        file_size = values[5]
        memory_size = values[6]
        if segment_type not in {_ELF_LOAD_SEGMENT_TYPE, _ELF_DYNAMIC_SEGMENT_TYPE}:
            continue
        if (
            file_size > memory_size
            or file_offset + file_size > len(contents)
            or virtual_address > _ELF_MAX_ADDRESS - file_size
        ):
            raise ValueError(f"{description} contains an out-of-bounds ELF segment")
        if segment_type == _ELF_LOAD_SEGMENT_TYPE:
            load_segments.append((file_offset, virtual_address, file_size))
        else:
            dynamic_segments.append((file_offset, virtual_address, file_size))

    if not load_segments:
        raise ValueError(f"{description} must contain at least one file-backed ELF load segment")
    if len(dynamic_segments) > 1:
        raise ValueError(f"{description} contains more than one ELF dynamic segment")
    if not dynamic_segments:
        return _ElfDynamicLinkage(
            needed=(),
            filters=(),
            auxiliaries=(),
            soname=None,
            versioned_symbols=(),
            undefined_symbols=frozenset(),
        )

    dynamic_offset, dynamic_address, dynamic_size = dynamic_segments[0]
    if dynamic_size == 0 or dynamic_size % _ELF_DYNAMIC_ENTRY_64.size != 0:
        raise ValueError(f"{description} contains an invalid ELF dynamic-segment size")
    dynamic_entry_count = dynamic_size // _ELF_DYNAMIC_ENTRY_64.size
    if dynamic_entry_count > _MAX_ELF_DYNAMIC_ENTRIES:
        raise ValueError(f"{description} contains too many ELF dynamic entries")
    mapped_dynamic_range = _elf_file_range_for_virtual_range(
        load_segments,
        dynamic_address,
        dynamic_size,
        description=description,
        range_description="dynamic segment",
    )
    if mapped_dynamic_range != (dynamic_offset, dynamic_offset + dynamic_size):
        raise ValueError(f"{description} ELF dynamic segment does not match its load mapping")

    needed_offsets: list[int] = []
    filter_offsets: list[int] = []
    auxiliary_offsets: list[int] = []
    hash_table_addresses: list[int] = []
    gnu_hash_table_addresses: list[int] = []
    string_table_addresses: list[int] = []
    string_table_sizes: list[int] = []
    symbol_table_addresses: list[int] = []
    symbol_entry_sizes: list[int] = []
    soname_offsets: list[int] = []
    version_needed_addresses: list[int] = []
    version_needed_counts: list[int] = []
    terminated = False
    for index in range(dynamic_entry_count):
        tag, value = _ELF_DYNAMIC_ENTRY_64.unpack_from(contents, dynamic_offset + index * _ELF_DYNAMIC_ENTRY_64.size)
        if tag == _ELF_DYNAMIC_NULL_TAG:
            terminated = True
            break
        if tag == _ELF_DYNAMIC_NEEDED_TAG:
            needed_offsets.append(value)
        elif tag == _ELF_DYNAMIC_FILTER_TAG:
            filter_offsets.append(value)
        elif tag == _ELF_DYNAMIC_AUXILIARY_TAG:
            auxiliary_offsets.append(value)
        elif tag == _ELF_DYNAMIC_HASH_TAG:
            hash_table_addresses.append(value)
        elif tag == _ELF_DYNAMIC_GNU_HASH_TAG:
            gnu_hash_table_addresses.append(value)
        elif tag == _ELF_DYNAMIC_STRING_TABLE_TAG:
            string_table_addresses.append(value)
        elif tag == _ELF_DYNAMIC_STRING_TABLE_SIZE_TAG:
            string_table_sizes.append(value)
        elif tag == _ELF_DYNAMIC_SYMBOL_TABLE_TAG:
            symbol_table_addresses.append(value)
        elif tag == _ELF_DYNAMIC_SYMBOL_ENTRY_SIZE_TAG:
            symbol_entry_sizes.append(value)
        elif tag == _ELF_DYNAMIC_SONAME_TAG:
            soname_offsets.append(value)
        elif tag == _ELF_DYNAMIC_VERSION_NEEDED_TAG:
            version_needed_addresses.append(value)
        elif tag == _ELF_DYNAMIC_VERSION_NEEDED_COUNT_TAG:
            version_needed_counts.append(value)
        elif tag in _ELF_DYNAMIC_UNSUPPORTED_LOADER_CONFIGURATION_TAG_NAMES:
            tag_name = _ELF_DYNAMIC_UNSUPPORTED_LOADER_CONFIGURATION_TAG_NAMES[tag]
            raise ValueError(f"{description} contains unsupported ELF loader configuration {tag_name}")
        elif tag == _ELF_DYNAMIC_FLAGS_1_TAG and value & _ELF_DYNAMIC_FLAGS_1_NO_DEFAULT_LIBRARY:
            raise ValueError(f"{description} disables the default ELF shared-library search path")
        if len(needed_offsets) + len(filter_offsets) + len(auxiliary_offsets) > _MAX_ELF_LOADER_DEPENDENCIES:
            raise ValueError(f"{description} declares too many ELF loader dependencies")
    if not terminated:
        raise ValueError(f"{description} ELF dynamic segment has no terminating DT_NULL entry")
    if (
        len(hash_table_addresses) > 1
        or len(gnu_hash_table_addresses) > 1
        or len(string_table_addresses) > 1
        or len(string_table_sizes) > 1
        or len(symbol_table_addresses) > 1
        or len(symbol_entry_sizes) > 1
        or len(soname_offsets) > 1
        or len(version_needed_addresses) > 1
        or len(version_needed_counts) > 1
    ):
        raise ValueError(f"{description} contains duplicate ELF dynamic linkage metadata")
    if bool(version_needed_addresses) != bool(version_needed_counts):
        raise ValueError(f"{description} contains incomplete ELF version-needed metadata")
    if bool(symbol_table_addresses) != bool(symbol_entry_sizes):
        raise ValueError(f"{description} contains incomplete ELF dynamic symbol-table metadata")
    if (hash_table_addresses or gnu_hash_table_addresses) and not symbol_table_addresses:
        raise ValueError(f"{description} contains an ELF dynamic hash table without a symbol table")
    if symbol_table_addresses and not (hash_table_addresses or gnu_hash_table_addresses):
        raise ValueError(f"{description} ELF dynamic symbol table has no bounded hash-table count")

    dependency_offsets = (*needed_offsets, *filter_offsets, *auxiliary_offsets)
    string_offsets = (*dependency_offsets, *soname_offsets)
    if not string_offsets and not version_needed_addresses and not symbol_table_addresses:
        return _ElfDynamicLinkage(
            needed=(),
            filters=(),
            auxiliaries=(),
            soname=None,
            versioned_symbols=(),
            undefined_symbols=frozenset(),
        )
    if len(string_table_addresses) != 1 or len(string_table_sizes) != 1:
        raise ValueError(f"{description} ELF dynamic strings require exactly one string table and size")
    string_table_size = string_table_sizes[0]
    if not 0 < string_table_size <= _MAX_ELF_DYNAMIC_STRING_TABLE_BYTES:
        raise ValueError(f"{description} contains an invalid ELF dynamic string-table size")
    string_table_start, string_table_end = _elf_file_range_for_virtual_range(
        load_segments,
        string_table_addresses[0],
        string_table_size,
        description=description,
        range_description="dynamic string table",
    )
    string_table = contents[string_table_start:string_table_end]

    needed = tuple(
        _elf_dynamic_library_name(string_table, offset, description=description) for offset in needed_offsets
    )
    filters = tuple(
        _elf_dynamic_library_name(string_table, offset, description=description) for offset in filter_offsets
    )
    auxiliaries = tuple(
        _elf_dynamic_library_name(string_table, offset, description=description) for offset in auxiliary_offsets
    )
    loader_dependencies = needed + filters + auxiliaries
    if len(set(loader_dependencies)) != len(loader_dependencies):
        raise ValueError(f"{description} contains duplicate ELF loader dependencies")
    soname = (
        _elf_dynamic_library_name(string_table, soname_offsets[0], description=description) if soname_offsets else None
    )
    versioned_symbols = (
        _parse_elf_version_requirements(
            contents,
            load_segments,
            version_needed_addresses[0],
            version_needed_counts[0],
            string_table,
            needed=needed,
            description=description,
        )
        if version_needed_addresses
        else ()
    )
    undefined_symbols = (
        _parse_elf_undefined_symbols(
            contents,
            load_segments,
            symbol_table_addresses[0],
            symbol_entry_sizes[0],
            hash_table_address=hash_table_addresses[0] if hash_table_addresses else None,
            gnu_hash_table_address=gnu_hash_table_addresses[0] if gnu_hash_table_addresses else None,
            string_table=string_table,
            description=description,
        )
        if symbol_table_addresses
        else frozenset()
    )
    return _ElfDynamicLinkage(
        needed=needed,
        filters=filters,
        auxiliaries=auxiliaries,
        soname=soname,
        versioned_symbols=versioned_symbols,
        undefined_symbols=undefined_symbols,
    )


def _elf_file_range_for_virtual_range(
    load_segments: Iterable[tuple[int, int, int]],
    virtual_address: int,
    size: int,
    *,
    description: str,
    range_description: str,
) -> tuple[int, int]:
    virtual_end = virtual_address + size
    mappings = {
        (file_offset + virtual_address - segment_address, file_offset + virtual_end - segment_address)
        for file_offset, segment_address, file_size in load_segments
        if virtual_address >= segment_address and virtual_end <= segment_address + file_size
    }
    if len(mappings) != 1:
        raise ValueError(f"{description} ELF {range_description} has no unique file-backed load mapping")
    return next(iter(mappings))


def _parse_elf_undefined_symbols(
    contents: bytes,
    load_segments: Iterable[tuple[int, int, int]],
    symbol_table_address: int,
    symbol_entry_size: int,
    *,
    hash_table_address: int | None,
    gnu_hash_table_address: int | None,
    string_table: bytes,
    description: str,
) -> frozenset[str]:
    if symbol_entry_size != _ELF_DYNAMIC_SYMBOL_64.size:
        raise ValueError(f"{description} contains an invalid ELF dynamic symbol-entry size")
    symbol_counts: list[int] = []
    if hash_table_address is not None:
        symbol_counts.append(
            _elf_sysv_hash_symbol_count(
                contents,
                load_segments,
                hash_table_address,
                description=description,
            )
        )
    if gnu_hash_table_address is not None:
        symbol_counts.append(
            _elf_gnu_hash_symbol_count(
                contents,
                load_segments,
                gnu_hash_table_address,
                description=description,
            )
        )
    if not symbol_counts or len(set(symbol_counts)) != 1:
        raise ValueError(f"{description} ELF dynamic hash tables do not provide one exact symbol count")
    symbol_count = symbol_counts[0]
    symbol_table_start, _symbol_table_end = _elf_file_range_for_virtual_range(
        load_segments,
        symbol_table_address,
        symbol_count * symbol_entry_size,
        description=description,
        range_description="dynamic symbol table",
    )

    undefined_symbols: set[str] = set()
    try:
        elf_file = ELFFile(io.BytesIO(contents))
        for index in range(symbol_count):
            symbol = struct_parse(
                elf_file.structs.Elf_Sym,
                elf_file.stream,
                stream_pos=symbol_table_start + index * symbol_entry_size,
            )
            symbol_name = _elf_dynamic_symbol_name(
                string_table,
                symbol["st_name"],
                description=description,
            )
            if symbol["st_shndx"] == "SHN_UNDEF" and symbol["st_info"]["bind"] != "STB_WEAK" and symbol_name:
                undefined_symbols.add(symbol_name)
    except ELFError as exception:
        raise ValueError(f"{description} contains an invalid ELF dynamic symbol table") from exception
    return frozenset(undefined_symbols)


def _elf_sysv_hash_symbol_count(
    contents: bytes,
    load_segments: Iterable[tuple[int, int, int]],
    hash_table_address: int,
    *,
    description: str,
) -> int:
    header_start, _header_end = _elf_file_range_for_virtual_range(
        load_segments,
        hash_table_address,
        _ELF_SYSV_HASH_HEADER.size,
        description=description,
        range_description="SysV hash header",
    )
    bucket_count, symbol_count = _ELF_SYSV_HASH_HEADER.unpack_from(contents, header_start)
    if not 0 < bucket_count <= _MAX_ELF_DYNAMIC_SYMBOLS or not 0 < symbol_count <= _MAX_ELF_DYNAMIC_SYMBOLS:
        raise ValueError(f"{description} contains invalid or excessive ELF SysV hash dimensions")
    table_size = _ELF_SYSV_HASH_HEADER.size + 4 * (bucket_count + symbol_count)
    table_start, _table_end = _elf_file_range_for_virtual_range(
        load_segments,
        hash_table_address,
        table_size,
        description=description,
        range_description="SysV hash table",
    )
    indices_start = table_start + _ELF_SYSV_HASH_HEADER.size
    for index in range(bucket_count + symbol_count):
        symbol_index = struct.unpack_from("<I", contents, indices_start + index * 4)[0]
        if symbol_index >= symbol_count:
            raise ValueError(f"{description} contains an out-of-range ELF SysV hash symbol index")
    return symbol_count


def _elf_gnu_hash_symbol_count(
    contents: bytes,
    load_segments: Iterable[tuple[int, int, int]],
    hash_table_address: int,
    *,
    description: str,
) -> int:
    header_start, _header_end = _elf_file_range_for_virtual_range(
        load_segments,
        hash_table_address,
        _ELF_GNU_HASH_HEADER.size,
        description=description,
        range_description="GNU hash header",
    )
    bucket_count, symbol_offset, bloom_size, _bloom_shift = _ELF_GNU_HASH_HEADER.unpack_from(contents, header_start)
    if (
        not 0 < bucket_count <= _MAX_ELF_DYNAMIC_SYMBOLS
        or not 0 < symbol_offset <= _MAX_ELF_DYNAMIC_SYMBOLS
        or not 0 < bloom_size <= _MAX_ELF_DYNAMIC_SYMBOLS
    ):
        raise ValueError(f"{description} contains invalid or excessive ELF GNU hash dimensions")
    fixed_size = _ELF_GNU_HASH_HEADER.size + 8 * bloom_size + 4 * bucket_count
    table_start, _table_end = _elf_file_range_for_virtual_range(
        load_segments,
        hash_table_address,
        fixed_size,
        description=description,
        range_description="GNU hash table",
    )
    buckets_start = table_start + _ELF_GNU_HASH_HEADER.size + 8 * bloom_size
    buckets = tuple(struct.unpack_from("<I", contents, buckets_start + index * 4)[0] for index in range(bucket_count))
    if any(bucket != 0 and not symbol_offset <= bucket < _MAX_ELF_DYNAMIC_SYMBOLS for bucket in buckets):
        raise ValueError(f"{description} contains an out-of-range ELF GNU hash bucket")
    nonempty_buckets = tuple(bucket for bucket in buckets if bucket)
    if not nonempty_buckets:
        return symbol_offset

    last_symbol_start = max(nonempty_buckets)
    chain_index = last_symbol_start - symbol_offset
    chain_relative_offset = fixed_size + chain_index * 4
    if hash_table_address > _ELF_MAX_ADDRESS - chain_relative_offset:
        raise ValueError(f"{description} contains an out-of-bounds ELF GNU hash chain")
    chain_start, chain_bytes = _elf_file_offset_and_available_size(
        load_segments,
        hash_table_address + chain_relative_offset,
        description=description,
        range_description="GNU hash chain",
    )
    maximum_chain_entries = min(chain_bytes // 4, _MAX_ELF_DYNAMIC_SYMBOLS - last_symbol_start)
    for index in range(maximum_chain_entries):
        chain_value = struct.unpack_from("<I", contents, chain_start + index * 4)[0]
        if chain_value & 1:
            return last_symbol_start + index + 1
    raise ValueError(f"{description} contains an unterminated or excessive ELF GNU hash chain")


def _elf_file_offset_and_available_size(
    load_segments: Iterable[tuple[int, int, int]],
    virtual_address: int,
    *,
    description: str,
    range_description: str,
) -> tuple[int, int]:
    mappings = {
        (
            file_offset + virtual_address - segment_address,
            file_size - (virtual_address - segment_address),
        )
        for file_offset, segment_address, file_size in load_segments
        if segment_address <= virtual_address < segment_address + file_size
    }
    if len(mappings) != 1:
        raise ValueError(f"{description} ELF {range_description} has no unique file-backed load mapping")
    return next(iter(mappings))


def _elf_dynamic_symbol_name(string_table: bytes, offset: int, *, description: str) -> str:
    if offset >= len(string_table):
        raise ValueError(f"{description} contains an out-of-bounds ELF dynamic symbol-name offset")
    bounded_end = min(len(string_table), offset + _MAX_ELF_SYMBOL_NAME_BYTES + 1)
    terminator = string_table.find(b"\0", offset, bounded_end)
    if terminator < 0:
        raise ValueError(f"{description} contains an unterminated or oversized ELF dynamic symbol name")
    return string_table[offset:terminator].decode("utf-8", errors="surrogateescape")


def _parse_elf_version_requirements(
    contents: bytes,
    load_segments: Iterable[tuple[int, int, int]],
    version_needed_address: int,
    version_needed_count: int,
    string_table: bytes,
    *,
    needed: tuple[str, ...],
    description: str,
) -> tuple[tuple[str, str], ...]:
    if not 0 < version_needed_count <= _MAX_ELF_VERSION_NEEDED_FILES:
        raise ValueError(f"{description} contains an invalid ELF version-needed file count")
    current_address = version_needed_address
    seen_libraries: set[str] = set()
    seen_requirements: set[tuple[str, str]] = set()
    occupied_words: set[int] = set()
    requirements: list[tuple[str, str]] = []
    for file_index in range(version_needed_count):
        _record_elf_metadata_range(
            occupied_words,
            current_address,
            _ELF_VERSION_NEEDED_ENTRY_64.size,
            description=description,
        )
        entry_start, _entry_end = _elf_file_range_for_virtual_range(
            load_segments,
            current_address,
            _ELF_VERSION_NEEDED_ENTRY_64.size,
            description=description,
            range_description="version-needed entry",
        )
        version, auxiliary_count, library_offset, auxiliary_offset, next_offset = (
            _ELF_VERSION_NEEDED_ENTRY_64.unpack_from(contents, entry_start)
        )
        if version != 1 or auxiliary_count == 0:
            raise ValueError(f"{description} contains an invalid ELF version-needed entry")
        if len(requirements) + auxiliary_count > _MAX_ELF_VERSION_REQUIREMENTS:
            raise ValueError(f"{description} declares too many versioned ELF symbol requirements")
        library = _elf_dynamic_library_name(string_table, library_offset, description=description)
        if library not in needed:
            raise ValueError(f"{description} contains an ELF version requirement for an undeclared library")
        if library in seen_libraries:
            raise ValueError(f"{description} contains duplicate ELF version-needed library entries")
        seen_libraries.add(library)

        if auxiliary_offset < _ELF_VERSION_NEEDED_ENTRY_64.size or auxiliary_offset % 4 != 0:
            raise ValueError(f"{description} contains an invalid ELF version-needed auxiliary offset")
        auxiliary_address = _checked_elf_relative_address(
            current_address,
            auxiliary_offset,
            description=description,
        )
        for auxiliary_index in range(auxiliary_count):
            _record_elf_metadata_range(
                occupied_words,
                auxiliary_address,
                _ELF_VERSION_NEEDED_AUXILIARY_64.size,
                description=description,
            )
            auxiliary_start, _auxiliary_end = _elf_file_range_for_virtual_range(
                load_segments,
                auxiliary_address,
                _ELF_VERSION_NEEDED_AUXILIARY_64.size,
                description=description,
                range_description="version-needed auxiliary entry",
            )
            _name_hash, _flags, _other, name_offset, auxiliary_next_offset = (
                _ELF_VERSION_NEEDED_AUXILIARY_64.unpack_from(contents, auxiliary_start)
            )
            requirement = (library, _elf_dynamic_version_name(string_table, name_offset, description=description))
            if requirement in seen_requirements:
                raise ValueError(f"{description} contains duplicate versioned ELF symbol requirements")
            seen_requirements.add(requirement)
            requirements.append(requirement)
            if auxiliary_index == auxiliary_count - 1:
                if auxiliary_next_offset != 0:
                    raise ValueError(f"{description} ELF version-needed auxiliary chain exceeds its declared count")
            else:
                if auxiliary_next_offset < _ELF_VERSION_NEEDED_AUXILIARY_64.size or auxiliary_next_offset % 4 != 0:
                    raise ValueError(f"{description} contains an invalid ELF version-needed auxiliary link")
                auxiliary_address = _checked_elf_relative_address(
                    auxiliary_address,
                    auxiliary_next_offset,
                    description=description,
                )

        if file_index == version_needed_count - 1:
            if next_offset != 0:
                raise ValueError(f"{description} ELF version-needed chain exceeds its declared count")
        else:
            if next_offset < _ELF_VERSION_NEEDED_ENTRY_64.size or next_offset % 4 != 0:
                raise ValueError(f"{description} contains an invalid ELF version-needed link")
            current_address = _checked_elf_relative_address(
                current_address,
                next_offset,
                description=description,
            )
    return tuple(requirements)


def _record_elf_metadata_range(
    occupied_words: set[int],
    start: int,
    size: int,
    *,
    description: str,
) -> None:
    if start % 4 != 0 or size % 4 != 0 or start > _ELF_MAX_ADDRESS - size:
        raise ValueError(f"{description} contains an invalid ELF metadata address")
    end = start + size
    words = set(range(start, end, 4))
    if not occupied_words.isdisjoint(words):
        raise ValueError(f"{description} contains overlapping ELF version-needed metadata")
    occupied_words.update(words)


def _checked_elf_relative_address(address: int, offset: int, *, description: str) -> int:
    if address > _ELF_MAX_ADDRESS - offset:
        raise ValueError(f"{description} contains an overflowing ELF relative metadata address")
    return address + offset


def _elf_dynamic_library_name(string_table: bytes, offset: int, *, description: str) -> str:
    if offset >= len(string_table):
        raise ValueError(f"{description} contains an out-of-bounds ELF dynamic string offset")
    bounded_end = min(len(string_table), offset + _MAX_ELF_LIBRARY_NAME_BYTES + 1)
    terminator = string_table.find(b"\0", offset, bounded_end)
    if terminator < 0:
        raise ValueError(f"{description} contains an unterminated or oversized ELF shared-library name")
    raw_name = string_table[offset:terminator]
    if _ELF_LIBRARY_NAME_RE.fullmatch(raw_name) is None:
        raise ValueError(f"{description} contains an invalid ELF shared-library name")
    return raw_name.decode("ascii")


def _elf_dynamic_version_name(string_table: bytes, offset: int, *, description: str) -> str:
    if offset >= len(string_table):
        raise ValueError(f"{description} contains an out-of-bounds ELF version-name offset")
    bounded_end = min(len(string_table), offset + _MAX_ELF_VERSION_NAME_BYTES + 1)
    terminator = string_table.find(b"\0", offset, bounded_end)
    if terminator < 0:
        raise ValueError(f"{description} contains an unterminated or oversized ELF version name")
    raw_name = string_table[offset:terminator]
    if _ELF_VERSION_NAME_RE.fullmatch(raw_name) is None:
        raise ValueError(f"{description} contains an invalid ELF version name")
    return raw_name.decode("ascii")


def _linux_policy_external_libraries(policy: _WheelPlatformPolicy) -> frozenset[str]:
    if policy.family == "manylinux":
        pinned_policy = _manylinux_policy_data(policy)
        libraries = set(pinned_policy.external_libraries)
        libraries.add(_MANYLINUX_DYNAMIC_LOADER_BY_ARCHITECTURE[policy.architecture])
        return frozenset(libraries)
    if policy.family == "musllinux":
        return frozenset({_MUSLLINUX_LIBC_BY_ARCHITECTURE[policy.architecture], "libz.so.1"})
    raise AssertionError(f"Linux external-library policy is undefined for {policy.family!r}")


def _manylinux_policy_versioned_symbols(policy: _WheelPlatformPolicy) -> frozenset[str]:
    return _manylinux_policy_data(policy).versioned_symbols


def _manylinux_policy_blacklisted_undefined_symbols(
    policy: _WheelPlatformPolicy,
    needed_libraries: Iterable[str],
    undefined_symbols: frozenset[str],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    dependency_names = frozenset(needed_libraries)
    matches: list[tuple[str, tuple[str, ...]]] = []
    for library, blacklisted_symbols in _manylinux_policy_data(policy).undefined_symbol_blacklist:
        if library not in dependency_names:
            continue
        matched_symbols = set(undefined_symbols.intersection(blacklisted_symbols))
        if "*" in blacklisted_symbols:
            matched_symbols.add("*")
        if matched_symbols:
            matches.append((library, tuple(sorted(matched_symbols))))
    return tuple(matches)


def _manylinux_policy_data(policy: _WheelPlatformPolicy) -> ManylinuxPolicy:
    if policy.family != "manylinux" or policy.minimum_version is None:
        raise AssertionError("manylinux policy data requires a versioned manylinux platform")
    return manylinux_policy(policy.minimum_version, policy.architecture)


def _validate_macos_binary_platform(
    contents: bytes | memoryview,
    platform_tag: str,
    *,
    description: str,
) -> tuple[int, int, int]:
    policy = _wheel_platform_policy(platform_tag)
    if policy.family != "macosx":
        raise ValueError(f"macOS binary validation requires a macOS platform tag, not {platform_tag!r}")
    if len(contents) < _MACHO_HEADER_64.size or bytes(contents[: len(_MACHO_MAGIC_64)]) != _MACHO_MAGIC_64:
        raise ValueError(f"{description} must be a thin 64-bit little-endian Mach-O binary")
    _magic, cpu_type, cpu_subtype, file_type, command_count, commands_size, _flags, reserved = (
        _MACHO_HEADER_64.unpack_from(contents)
    )
    expected_cpu = _MACHO_CPU_BY_ARCHITECTURE[policy.architecture]
    if cpu_type != expected_cpu:
        raise ValueError(
            f"{description} Mach-O CPU type {cpu_type} does not match platform architecture {policy.architecture!r}"
        )
    _validate_macho_cpu_subtype(cpu_subtype, policy.architecture, description=description)
    if reserved != 0:
        raise ValueError(f"{description} contains a nonzero reserved Mach-O header field")
    if file_type not in _MACHO_SHARED_OBJECT_TYPES:
        raise ValueError(f"{description} must use a Mach-O dynamic-library or bundle file type")
    if command_count > _MAX_MACHO_LOAD_COMMANDS:
        raise ValueError(f"{description} contains too many Mach-O load commands")

    command_offset = _MACHO_HEADER_64.size
    commands_end = command_offset + commands_size
    if commands_end > len(contents):
        raise ValueError(f"{description} has truncated Mach-O load commands")
    deployment_targets: set[tuple[int, int, int]] = set()
    dynamic_library_dependencies: list[str] = []
    runtime_search_paths: list[str] = []
    for _ in range(command_count):
        if command_offset + _MACHO_LOAD_COMMAND.size > commands_end:
            raise ValueError(f"{description} has a truncated Mach-O load command")
        command, command_size = _MACHO_LOAD_COMMAND.unpack_from(contents, command_offset)
        if command_size < _MACHO_LOAD_COMMAND.size or command_size % 8 != 0:
            raise ValueError(f"{description} contains an invalid Mach-O load-command size")
        next_command = command_offset + command_size
        if next_command > commands_end:
            raise ValueError(f"{description} has a Mach-O load command outside its declared bounds")
        if command == _MACHO_VERSION_MIN_MACOSX:
            if command_size != _MACHO_VERSION_COMMAND.size:
                raise ValueError(f"{description} contains an invalid LC_VERSION_MIN_MACOSX command size")
            _command, _size, encoded_version, _sdk = _MACHO_VERSION_COMMAND.unpack_from(contents, command_offset)
            deployment_targets.add(_decode_macho_version(encoded_version))
        elif command in _MACHO_NON_MACOS_VERSION_COMMANDS:
            raise ValueError(f"{description} contains a non-macOS deployment-target load command")
        elif command == _MACHO_BUILD_VERSION:
            if command_size < _MACHO_BUILD_VERSION_COMMAND.size:
                raise ValueError(f"{description} contains a truncated LC_BUILD_VERSION command")
            _command, _size, build_platform, encoded_version, _sdk, tool_count = (
                _MACHO_BUILD_VERSION_COMMAND.unpack_from(contents, command_offset)
            )
            expected_command_size = _MACHO_BUILD_VERSION_COMMAND.size + tool_count * _MACHO_BUILD_TOOL_VERSION.size
            if command_size != expected_command_size:
                raise ValueError(f"{description} contains an invalid LC_BUILD_VERSION tool table")
            if build_platform != _MACHO_PLATFORM_MACOS:
                raise ValueError(f"{description} LC_BUILD_VERSION does not target macOS")
            deployment_targets.add(_decode_macho_version(encoded_version))
        elif command in _MACHO_DYLIB_COMMANDS:
            if command_size < _MACHO_DYLIB_COMMAND.size:
                raise ValueError(f"{description} contains a truncated Mach-O dylib command")
            _command, _size, name_offset, _timestamp, _current_version, _compatibility_version = (
                _MACHO_DYLIB_COMMAND.unpack_from(contents, command_offset)
            )
            install_name = _macho_load_command_string(
                contents,
                command_offset=command_offset,
                command_size=command_size,
                string_offset=name_offset,
                minimum_string_offset=_MACHO_DYLIB_COMMAND.size,
                description=description,
                field_description="dynamic-library install name",
            )
            if command in _MACHO_DYNAMIC_LIBRARY_DEPENDENCY_COMMANDS:
                dynamic_library_dependencies.append(install_name)
        elif command == _MACHO_RPATH:
            if command_size < _MACHO_RPATH_COMMAND.size:
                raise ValueError(f"{description} contains a truncated LC_RPATH command")
            _command, _size, path_offset = _MACHO_RPATH_COMMAND.unpack_from(contents, command_offset)
            runtime_search_paths.append(
                _macho_load_command_string(
                    contents,
                    command_offset=command_offset,
                    command_size=command_size,
                    string_offset=path_offset,
                    minimum_string_offset=_MACHO_RPATH_COMMAND.size,
                    description=description,
                    field_description="runtime search path",
                )
            )
        elif command in _MACHO_UNSUPPORTED_DYNAMIC_LINKER_COMMANDS:
            raise ValueError(f"{description} contains unsupported Mach-O dynamic-linker command 0x{command:x}")
        command_offset = next_command
    if command_offset != commands_end:
        raise ValueError(f"{description} Mach-O load-command sizes do not match their declared total")
    if len(set(dynamic_library_dependencies)) != len(dynamic_library_dependencies):
        raise ValueError(f"{description} contains duplicate Mach-O dynamic-library dependencies")
    non_system_dependencies = tuple(
        sorted(path for path in dynamic_library_dependencies if not _is_macos_system_library_path(path))
    )
    if non_system_dependencies:
        raise ValueError(f"{description} requires non-system Mach-O dynamic libraries: {non_system_dependencies}")
    if len(set(runtime_search_paths)) != len(runtime_search_paths):
        raise ValueError(f"{description} contains duplicate Mach-O runtime search paths")
    unsafe_runtime_search_paths = tuple(
        sorted(path for path in runtime_search_paths if not _is_relocatable_macos_runtime_search_path(path))
    )
    if unsafe_runtime_search_paths:
        raise ValueError(
            f"{description} contains non-relocatable Mach-O runtime search paths: {unsafe_runtime_search_paths}"
        )
    if len(deployment_targets) != 1:
        raise ValueError(f"{description} must declare exactly one consistent macOS deployment target")

    deployment_target = next(iter(deployment_targets))
    minimum_version = policy.minimum_version
    if minimum_version is None:
        raise AssertionError("macOS platform policy must contain a minimum version")
    normalized_target = _normalize_macos_version(deployment_target)
    declared_target = (*minimum_version, 0)
    if normalized_target > declared_target:
        version = ".".join(str(component) for component in deployment_target)
        raise ValueError(f"{description} requires macOS {version}, which exceeds platform tag {platform_tag!r}")
    return deployment_target


def _macho_load_command_string(
    contents: bytes | memoryview,
    *,
    command_offset: int,
    command_size: int,
    string_offset: int,
    minimum_string_offset: int,
    description: str,
    field_description: str,
) -> str:
    if string_offset < minimum_string_offset or string_offset >= command_size:
        raise ValueError(f"{description} contains an invalid Mach-O {field_description} offset")
    string_start = command_offset + string_offset
    command_end = command_offset + command_size
    search_end = min(command_end, string_start + _MAX_MACHO_LOAD_PATH_BYTES + 1)
    candidate = bytes(contents[string_start:search_end])
    terminator = candidate.find(b"\0")
    if terminator < 0:
        raise ValueError(f"{description} contains an unterminated or oversized Mach-O {field_description}")
    raw_value = candidate[:terminator]
    if not raw_value:
        raise ValueError(f"{description} contains an empty Mach-O {field_description}")
    try:
        return raw_value.decode("utf-8")
    except UnicodeDecodeError as exception:
        raise ValueError(f"{description} contains a non-UTF-8 Mach-O {field_description}") from exception


def _is_macos_system_library_path(path: str) -> bool:
    if not path.startswith(_MACOS_SYSTEM_LIBRARY_PREFIXES):
        return False
    components = path.split("/")
    return not path.endswith("/") and all(component not in {"", ".", ".."} for component in components[1:])


def _is_relocatable_macos_runtime_search_path(path: str) -> bool:
    for prefix in _MACOS_RELOCATABLE_RPATH_PREFIXES:
        if path == prefix:
            return True
        marker = f"{prefix}/"
        if path.startswith(marker):
            components = path[len(marker) :].split("/")
            return all(component not in {"", ".", ".."} for component in components)
    return False


def _validate_macos_wheel_binary_platform(contents: bytes, platform_tag: str, *, description: str) -> None:
    universal2 = _MACOS_UNIVERSAL2_PLATFORM_TAG_RE.fullmatch(platform_tag)
    magic = contents[:4]
    if universal2 is None and magic not in {_MACHO_FAT_MAGIC, _MACHO_FAT64_MAGIC}:
        _validate_macos_binary_platform(contents, platform_tag, description=description)
        return

    slices = _macho_fat_slices(contents, description=description)
    if universal2 is None:
        policy = _wheel_platform_policy(platform_tag)
        if policy.family != "macosx":
            raise ValueError(f"macOS wheel binary validation requires a macOS platform tag, not {platform_tag!r}")
        selected = slices.get(policy.architecture)
        if selected is None:
            raise ValueError(f"{description} universal Mach-O does not contain {policy.architecture!r}")
        _validate_macos_binary_platform(selected, platform_tag, description=description)
        return

    raw_minimum = (int(universal2["major"]), int(universal2["minor"]))
    x86_tag = f"macosx_{universal2['major']}_{universal2['minor']}_x86_64"
    x86_policy = _wheel_platform_policy(x86_tag)
    if set(slices) != {"x86_64", "arm64"}:
        raise ValueError(
            f"{description} universal2 Mach-O must contain exactly x86_64 and arm64 slices: {tuple(sorted(slices))}"
        )
    arm_minimum = max(x86_policy.minimum_version or raw_minimum, (11, 0))
    arm_tag = f"macosx_{arm_minimum[0]}_{arm_minimum[1]}_arm64"
    _validate_macos_binary_platform(slices["x86_64"], x86_tag, description=description)
    _validate_macos_binary_platform(slices["arm64"], arm_tag, description=description)


def _is_macos_binary(contents: bytes) -> bool:
    return contents[:4] in {_MACHO_MAGIC_64, _MACHO_FAT_MAGIC, _MACHO_FAT64_MAGIC}


def _macho_fat_slices(contents: bytes, *, description: str) -> dict[str, memoryview]:
    magic = contents[:4]
    if magic == _MACHO_FAT_MAGIC:
        architecture_struct = _MACHO_FAT_ARCH
        is_fat64 = False
    elif magic == _MACHO_FAT64_MAGIC:
        architecture_struct = _MACHO_FAT64_ARCH
        is_fat64 = True
    else:
        raise ValueError(f"{description} must contain a big-endian universal Mach-O binary")
    if len(contents) < _MACHO_FAT_HEADER.size:
        raise ValueError(f"{description} has a truncated universal Mach-O header")
    _magic, architecture_count = _MACHO_FAT_HEADER.unpack_from(contents)
    if architecture_count == 0 or architecture_count > _MAX_MACHO_FAT_ARCHITECTURES:
        raise ValueError(f"{description} contains an invalid universal Mach-O architecture count")
    table_end = _MACHO_FAT_HEADER.size + architecture_count * architecture_struct.size
    if table_end > len(contents):
        raise ValueError(f"{description} has a truncated universal Mach-O architecture table")

    architecture_by_cpu = {cpu: architecture for architecture, cpu in _MACHO_CPU_BY_ARCHITECTURE.items()}
    slices: dict[str, memoryview] = {}
    intervals: list[tuple[int, int]] = []
    for index in range(architecture_count):
        values = architecture_struct.unpack_from(contents, _MACHO_FAT_HEADER.size + index * architecture_struct.size)
        cpu_type, cpu_subtype, slice_offset, slice_size, alignment = values[:5]
        if is_fat64 and values[5] != 0:
            raise ValueError(f"{description} contains a nonzero reserved universal Mach-O field")
        architecture = architecture_by_cpu.get(cpu_type)
        if architecture is None or architecture in slices:
            raise ValueError(f"{description} contains unsupported or duplicate universal Mach-O architectures")
        _validate_macho_cpu_subtype(cpu_subtype, architecture, description=description)
        if alignment > 31 or slice_offset % (1 << alignment) != 0:
            raise ValueError(f"{description} contains an invalid universal Mach-O slice alignment")
        slice_end = slice_offset + slice_size
        if slice_size < _MACHO_HEADER_64.size or slice_offset < table_end or slice_end > len(contents):
            raise ValueError(f"{description} contains an out-of-bounds universal Mach-O slice")
        intervals.append((slice_offset, slice_end))
        slices[architecture] = memoryview(contents)[slice_offset:slice_end]
    sorted_intervals = sorted(intervals)
    if any(
        current_start < previous_end
        for (_, previous_end), (current_start, _) in zip(sorted_intervals, sorted_intervals[1:])
    ):
        raise ValueError(f"{description} contains overlapping universal Mach-O slices")
    return slices


def _validate_macho_cpu_subtype(cpu_subtype: int, architecture: str, *, description: str) -> None:
    base_subtype = cpu_subtype & _MACHO_CPU_SUBTYPE_BASE_MASK
    if base_subtype not in _MACHO_CPU_SUBTYPES_BY_ARCHITECTURE[architecture]:
        raise ValueError(
            f"{description} Mach-O CPU subtype {base_subtype} is not compatible with generic {architecture!r}"
        )


def _decode_macho_version(value: int) -> tuple[int, int, int]:
    return value >> 16, (value >> 8) & 0xFF, value & 0xFF


def _normalize_macos_version(value: tuple[int, int, int]) -> tuple[int, int, int]:
    return (11, 0, value[2]) if value[:2] == (10, 16) else value


def _current_musl_version() -> tuple[int, int] | None:
    interpreter = _elf_interpreter(Path(sys.executable))
    if interpreter is None or "musl" not in interpreter:
        return None
    try:
        completed = subprocess.run(
            [interpreter],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env={"LC_ALL": "C", "LANG": "C", "PATH": os.defpath},
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = tuple(line.strip() for line in completed.stderr.splitlines() if line.strip())
    if not lines or not lines[0].startswith(b"musl"):
        return None
    match = _MUSL_VERSION_RE.search(completed.stderr)
    if match is None or any(len(component) > _MAX_LIBC_VERSION_COMPONENT_DIGITS for component in match.groups()):
        return None
    return int(match[1]), int(match[2])


def _elf_interpreter(path: Path) -> str | None:
    try:
        with path.open("rb") as executable:
            metadata = os.fstat(executable.fileno())
            header = executable.read(_ELF_HEADER_64.size)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or len(header) != _ELF_HEADER_64.size
                or not header.startswith(_ELF_MAGIC)
                or header[4] != 2
                or header[5] != 1
            ):
                return None
            fields = _ELF_HEADER_64.unpack(header)
            program_header_offset = fields[5]
            program_header_size = fields[9]
            program_header_count = fields[10]
            if (
                program_header_size < _ELF_PROGRAM_HEADER_64.size
                or program_header_count > _MAX_ELF_PROGRAM_HEADERS
                or program_header_offset + program_header_size * program_header_count > metadata.st_size
            ):
                return None
            for index in range(program_header_count):
                executable.seek(program_header_offset + index * program_header_size)
                program_header = executable.read(_ELF_PROGRAM_HEADER_64.size)
                if len(program_header) != _ELF_PROGRAM_HEADER_64.size:
                    return None
                values = _ELF_PROGRAM_HEADER_64.unpack(program_header)
                if values[0] != _ELF_PROGRAM_INTERPRETER_TYPE:
                    continue
                interpreter_offset = values[2]
                interpreter_size = values[5]
                if (
                    interpreter_size < 2
                    or interpreter_size > _MAX_ELF_INTERPRETER_BYTES
                    or interpreter_offset + interpreter_size > metadata.st_size
                ):
                    return None
                executable.seek(interpreter_offset)
                raw_interpreter = executable.read(interpreter_size)
                if len(raw_interpreter) != interpreter_size or not raw_interpreter.endswith(b"\0"):
                    return None
                return raw_interpreter[:-1].decode("utf-8")
    except (OSError, UnicodeError):
        return None
    return None


def _validate_dependency_platform_tag(
    parent_platform_tag: str,
    dependency_platform_tag: str,
    *,
    dependency_name: str,
) -> None:
    parent_policy = _wheel_platform_policy(parent_platform_tag)
    dependency_policy = _wheel_platform_policy(dependency_platform_tag)
    parent_minimum_version = parent_policy.minimum_version
    dependency_minimum_version = dependency_policy.minimum_version
    compatible = (
        dependency_policy.family == parent_policy.family
        and dependency_policy.architecture == parent_policy.architecture
        and (
            dependency_minimum_version is None
            or (
                parent_minimum_version is not None
                and (
                    dependency_minimum_version == parent_minimum_version
                    if parent_policy.family == "musllinux"
                    else dependency_minimum_version <= parent_minimum_version
                )
            )
        )
    )
    if not compatible:
        raise ValueError(
            f"dependency {dependency_name!r} wheel platform tag {dependency_platform_tag!r} is not compatible "
            f"with parent wheel platform tag {parent_platform_tag!r}"
        )


def _validate_license_expression(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("license_expression must be a non-empty SPDX expression")
    expression = value.strip()
    if not expression:
        raise ValueError("license_expression must be a non-empty SPDX expression")
    if any(ord(character) < 32 or ord(character) > 126 for character in expression):
        raise ValueError("license_expression must contain printable ASCII SPDX syntax only")
    try:
        return canonicalize_license_expression(expression)
    except InvalidLicenseExpression as exception:
        raise ValueError(f"license_expression must be a valid SPDX expression: {expression!r}") from exception


def _validate_metadata_license_expression(metadata) -> str:
    values = metadata.get_all("License-Expression", [])
    if len(values) != 1:
        raise ValueError(f"extension wheel METADATA must contain exactly one License-Expression: {values}")
    expression = str(values[0])
    canonical_expression = _validate_license_expression(expression)
    if expression != canonical_expression:
        raise ValueError(f"extension wheel License-Expression must use canonical SPDX syntax: {expression!r}")
    return canonical_expression


def _validate_metadata_version(metadata, *, description: str = "extension wheel") -> str:
    values = [str(value) for value in metadata.get_all("Metadata-Version", [])]
    if values != [_EXTENSION_METADATA_VERSION]:
        raise ValueError(
            f"{description} Metadata-Version must match the generated core metadata version exactly: "
            f"expected={_EXTENSION_METADATA_VERSION!r}, actual={values}"
        )
    return values[0]


def _validate_metadata_requires_python(metadata, *, description: str = "extension wheel") -> str:
    values = [str(value) for value in metadata.get_all("Requires-Python", [])]
    try:
        matches_supported_range = len(values) == 1 and SpecifierSet(values[0]) == _EXTENSION_REQUIRES_PYTHON_SPECIFIERS
    except InvalidSpecifier:
        matches_supported_range = False
    if not matches_supported_range:
        raise ValueError(
            f"{description} Requires-Python must match the supported Python range exactly: "
            f"expected={_EXTENSION_REQUIRES_PYTHON!r}, actual={values}"
        )
    return values[0]


def _metadata_license_file_members(metadata, *, dist_info_root: str, windows_paths: bool) -> tuple[str, ...]:
    license_files = tuple(str(value) for value in metadata.get_all("License-File", []))
    if not license_files:
        raise ValueError("extension wheel METADATA must declare at least one License-File")
    members: list[str] = []
    for license_file in license_files:
        parts = license_file.split("/")
        if (
            not license_file
            or license_file.startswith("/")
            or "\\" in license_file
            or any(part in {"", ".", ".."} or part != part.strip() for part in parts)
            or any(ord(character) < 32 or ord(character) > 126 for character in license_file)
            or (windows_paths and any(_windows_path_part_is_unsafe(part) for part in parts))
        ):
            raise ValueError(f"extension wheel contains an unsafe License-File path: {license_file!r}")
        members.append(f"{dist_info_root}.dist-info/licenses/{license_file}")
    if len({member.casefold() for member in members}) != len(members):
        raise ValueError("extension wheel License-File paths must be unique without case")
    member_paths = {member.casefold() for member in members}
    for member in members:
        parent = member.casefold().rpartition("/")[0]
        while parent:
            if parent in member_paths:
                raise ValueError("extension wheel License-File paths must not contain file/parent conflicts")
            parent = parent.rpartition("/")[0]
    return tuple(members)


def _validate_owned_extension_wheel_members(
    names: list[str],
    *,
    expected_provider: str,
    expected_artifact: str,
    expected_descriptor: str,
    expected_platform_build_details: str,
    dist_info_root: str,
    license_members: tuple[str, ...],
) -> None:
    if len(names) != len(set(names)):
        raise ValueError("extension wheel archive members must not be duplicated")
    casefolded_names = {name.casefold() for name in names}
    if len(casefolded_names) != len(names):
        raise ValueError("extension wheel archive members must be unique without case")
    expected_members = {
        expected_provider,
        expected_artifact,
        expected_descriptor,
        expected_platform_build_details,
        f"{dist_info_root}.dist-info/METADATA",
        f"{dist_info_root}.dist-info/WHEEL",
        f"{dist_info_root}.dist-info/entry_points.txt",
        f"{dist_info_root}.dist-info/RECORD",
        *license_members,
    }
    actual_members = set(names)
    if actual_members != expected_members:
        unexpected = tuple(sorted(actual_members - expected_members))
        missing = tuple(sorted(expected_members - actual_members))
        raise ValueError(
            f"extension wheel contains unowned or missing archive members: unexpected={unexpected}, missing={missing}"
        )


def _validate_exact_requirements(
    requirements: Iterable[Requirement],
    expected_versions: dict[str, str],
    *,
    description: str,
) -> None:
    resolved_requirements = tuple(requirements)
    actual_requirements: dict[str, Requirement] = {}
    for requirement in resolved_requirements:
        normalized_name = canonicalize_name(requirement.name)
        if normalized_name in actual_requirements:
            raise ValueError(f"{description} contains duplicate Requires-Dist entries for {normalized_name}")
        actual_requirements[normalized_name] = requirement

    valid = set(actual_requirements) == set(expected_versions)
    if valid:
        for normalized_name, expected_version in expected_versions.items():
            requirement = actual_requirements[normalized_name]
            specifiers = tuple(requirement.specifier)
            if (
                requirement.extras
                or requirement.url is not None
                or requirement.marker is not None
                or len(specifiers) != 1
                or specifiers[0].operator != "==="
                or specifiers[0].version != expected_version
            ):
                valid = False
                break
    if not valid:
        expected = tuple(f"{name}==={version}" for name, version in sorted(expected_versions.items()))
        raise ValueError(
            f"{description} Requires-Dist must match its descriptor exactly: "
            f"expected={expected}, actual={resolved_requirements}"
        )


def _validate_vane_version(value: str) -> str:
    try:
        Version(value)
    except (InvalidVersion, TypeError) as exception:
        raise RuntimeError(f"installed Vane version is not a valid package version: {value!r}") from exception
    return value


def _extension_interpreter_tag() -> str:
    interpreter_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if sys.implementation.name != "cpython" or _EXTENSION_INTERPRETER_TAG_RE.fullmatch(interpreter_tag) is None:
        raise RuntimeError("extension wheels require a supported CPython 3.10 through 3.14 interpreter")
    return interpreter_tag


def _descriptor_digest(descriptor: DynamicExtensionDescriptor) -> str:
    return hashlib.sha256(descriptor.to_json().encode("utf-8")).hexdigest()


def _extension_distribution_version(
    vane_version: str,
    descriptor: DynamicExtensionDescriptor,
) -> str:
    """Return a bounded public version that identifies one exact descriptor."""
    return _extension_distribution_version_from_digest(vane_version, _descriptor_digest(descriptor))


def _extension_distribution_version_from_digest(vane_version: str, descriptor_digest: str) -> str:
    """Return the public extension version for one canonical descriptor digest."""
    if re.fullmatch(r"[0-9a-f]{64}", descriptor_digest) is None:
        raise RuntimeError(f"descriptor digest must be a lowercase SHA-256: {descriptor_digest!r}")
    parsed_vane_version = Version(vane_version)
    if parsed_vane_version.pre is None:
        prerelease_stage = 0 if parsed_vane_version.dev is not None and parsed_vane_version.post is None else 4
        prerelease_number = 0
    else:
        prerelease_stage = _PRERELEASE_STAGE[parsed_vane_version.pre[0]]
        prerelease_number = parsed_vane_version.pre[1]

    post_present = int(parsed_vane_version.post is not None)
    post_number = parsed_vane_version.post or 0
    dev_absent = int(parsed_vane_version.dev is None)
    dev_number = parsed_vane_version.dev or 0
    vane_release = list(parsed_vane_version.release)
    while len(vane_release) > 1 and vane_release[-1] == 0:
        vane_release.pop()
    if len(vane_release) > _VANE_RELEASE_COMPONENT_COUNT:
        raise RuntimeError(
            f"installed Vane version has more than {_VANE_RELEASE_COMPONENT_COUNT} effective release components: "
            f"{vane_version!r}"
        )
    vane_release.extend([0] * (_VANE_RELEASE_COMPONENT_COUNT - len(vane_release)))
    digest_chunks = tuple(
        int(descriptor_digest[offset : offset + _DESCRIPTOR_DIGEST_CHUNK_HEX_LENGTH], 16)
        for offset in range(0, len(descriptor_digest), _DESCRIPTOR_DIGEST_CHUNK_HEX_LENGTH)
    )
    release = (
        *vane_release,
        prerelease_stage,
        prerelease_number,
        post_present,
        post_number,
        dev_absent,
        dev_number,
        *digest_chunks,
    )
    epoch = f"{parsed_vane_version.epoch}!" if parsed_vane_version.epoch else ""
    return str(Version(epoch + ".".join(str(component) for component in release)))


def _provider_module_source(name: str) -> str:
    return f'''# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0
"""Installed provider for the {name} Vane dynamic extension."""

from __future__ import annotations

from pathlib import Path

from vane.extensions import (
    DynamicExtensionDescriptor,
    LocalExtensionArtifact,
    LocalExtensionProvider,
)

_PACKAGE_DIRECTORY = Path(__file__).resolve().parent


def descriptor() -> DynamicExtensionDescriptor:
    """Return the immutable descriptor shipped with this extension wheel."""
    return DynamicExtensionDescriptor.from_json(
        (_PACKAGE_DIRECTORY / "{name}.dynamic-extension.json").read_bytes()
    )


def provider() -> LocalExtensionProvider:
    """Return the explicit provider advertised through package metadata."""
    extension_descriptor = descriptor()
    return LocalExtensionProvider(
        extension_descriptor.trust_identity,
        (
            LocalExtensionArtifact(
                extension_descriptor,
                _PACKAGE_DIRECTORY / "{name}.duckdb_extension",
            ),
        ),
    )


__all__ = ["descriptor", "provider"]
'''


def _metadata(
    distribution_name: str,
    distribution_version: str,
    vane_version: str,
    license_expression: str,
    license_members: tuple[str, ...],
    dist_info_root: str,
    dependency_requirements: tuple[tuple[str, str], ...],
) -> str:
    lines = [
        f"Metadata-Version: {_EXTENSION_METADATA_VERSION}",
        f"Name: {distribution_name}",
        f"Version: {distribution_version}",
        "Summary: Platform-specific Vane dynamic extension artifact",
        f"License-Expression: {license_expression}",
        f"Requires-Python: {_EXTENSION_REQUIRES_PYTHON}",
        f"Requires-Dist: vane-ai==={vane_version}",
    ]
    for dependency_name, dependency_version in dependency_requirements:
        lines.append(f"Requires-Dist: vane-extension-{dependency_name}==={dependency_version}")
    for member_name in license_members:
        lines.append(f"License-File: {member_name.removeprefix(dist_info_root + '/licenses/')}")
    lines.append("")
    return "\n".join(lines)


def _wheel_metadata(wheel_tag: str) -> str:
    return "\n".join(
        (
            "Wheel-Version: 1.0",
            "Generator: vane-extension-wheel",
            "Root-Is-Purelib: false",
            f"Tag: {wheel_tag}",
            "",
        )
    )


def _entry_points(name: str, provider_package: str) -> str:
    return "\n".join(
        (
            f"[{ENTRY_POINT_GROUP}]",
            f"{name} = vane_extensions.{provider_package}:provider",
            "",
        )
    )


def _record(entries: dict[str, bytes], record_name: str) -> str:
    record_buffer = io.StringIO()
    writer = csv.writer(record_buffer, lineterminator="\n")
    for member_name, member_contents in sorted(entries.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(member_contents).digest()).rstrip(b"=").decode("ascii")
        writer.writerow((member_name, f"sha256={digest}", len(member_contents)))
    writer.writerow((record_name, "", ""))
    return record_buffer.getvalue()


def _validate_wheel_record(
    wheel: zipfile.ZipFile,
    *,
    names: list[str],
    record_name: str,
) -> None:
    if names.count(record_name) != 1:
        raise ValueError(f"extension wheel must contain exactly one RECORD at {record_name!r}")
    archive_names = set(names)
    if len(archive_names) != len(names):
        raise ValueError("extension wheel archive members must not be duplicated")
    max_row_chars = max(2 * len(name) + _WHEEL_RECORD_ROW_FIXED_CHARS for name in archive_names)
    recorded: set[str] = set()
    try:
        with wheel.open(record_name) as record_file:
            with io.TextIOWrapper(record_file, encoding="utf-8", newline="") as record_text:
                row_number = 0
                while record_line := record_text.readline(max_row_chars + 1):
                    if len(record_line) > max_row_chars:
                        raise ValueError("extension wheel RECORD row exceeds its bounded maximum length")
                    row_number += 1
                    if row_number > len(archive_names):
                        raise ValueError("extension wheel RECORD contains more rows than archive members")
                    row = next(csv.reader((record_line,), strict=True))
                    if len(row) != 3:
                        raise ValueError(f"extension wheel contains an invalid RECORD row: {row!r}")
                    member_name, digest, size = row
                    if member_name in recorded:
                        raise ValueError(f"extension wheel RECORD repeats member {member_name!r}")
                    if member_name not in archive_names:
                        raise ValueError(f"extension wheel RECORD names missing member {member_name!r}")
                    recorded.add(member_name)
                    if member_name == record_name:
                        if digest or size:
                            raise ValueError("extension wheel RECORD must not hash or size itself")
                        continue
                    contents = wheel.read(member_name)
                    expected_digest = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(contents).digest()).rstrip(
                        b"="
                    ).decode("ascii")
                    if digest != expected_digest or size != str(len(contents)):
                        raise ValueError(f"extension wheel RECORD entry is invalid for {member_name!r}")
    except UnicodeError:
        raise ValueError("extension wheel RECORD must be valid UTF-8") from None
    except csv.Error as exception:
        raise ValueError("extension wheel RECORD must be valid CSV") from exception
    missing = archive_names - recorded
    if missing:
        raise ValueError(f"extension wheel RECORD omits archive members: {tuple(sorted(missing))}")


def _validate_wheel_path_component_lengths(
    paths: Iterable[str],
    *,
    description: str = "generated wheel",
) -> None:
    for path in paths:
        for component in path.split("/"):
            if len(component.encode("utf-8")) > _MAX_WHEEL_PATH_COMPONENT_BYTES:
                raise ValueError(f"{description} path component exceeds the portable 255-byte limit: {component!r}")


def _normalize_extension_wheel_permissions(path: Path) -> None:
    path.chmod(0o644)
    permissions = stat.S_IMODE(path.stat().st_mode)
    normalized = permissions & stat.S_IWRITE != 0 if os.name == "nt" else permissions == 0o644
    if not normalized:
        raise PermissionError(f"extension wheel permissions could not be normalized for publication: {path}")


def _validate_extension_wheel_size(path: Path, *, description: str = "extension wheel") -> None:
    if path.stat().st_size > _MAX_EXTENSION_WHEEL_BYTES:
        raise ValueError(f"{description} exceeds {PUBLICATION_FILE_LIMIT_DESCRIPTION}")


def _validate_extension_wheel_archive_size(
    wheel: zipfile.ZipFile,
    *,
    description: str = "extension wheel",
) -> None:
    _validate_extension_wheel_member_sizes(
        ((member.filename, member.file_size) for member in wheel.infolist()),
        description=description,
    )


def _validate_extension_wheel_entries_size(
    entries: dict[str, bytes],
    *,
    description: str = "extension wheel",
) -> None:
    _validate_extension_wheel_member_sizes(
        ((member_name, len(contents)) for member_name, contents in entries.items()),
        description=description,
    )


def _validate_extension_wheel_entries_count(
    entries: dict[str, bytes],
    *,
    additional_members: int = 0,
    description: str = "generated extension wheel",
) -> None:
    if len(entries) + additional_members > _MAX_EXTENSION_WHEEL_MEMBERS:
        raise ValueError(f"{description} contains more than {_MAX_EXTENSION_WHEEL_MEMBERS} archive members")


def _validate_extension_wheel_member_sizes(
    members: Iterable[tuple[str, int]],
    *,
    description: str,
) -> None:
    total_size = 0
    for member_name, member_size in members:
        if member_size > _MAX_EXTENSION_WHEEL_MEMBER_BYTES:
            raise ValueError(f"{description} member {member_name!r} exceeds {ARCHIVE_MEMBER_LIMIT_DESCRIPTION}")
        total_size += member_size
        if total_size > _MAX_EXTENSION_WHEEL_UNCOMPRESSED_BYTES:
            raise ValueError(f"{description} decompressed contents exceed {ARCHIVE_TOTAL_LIMIT_DESCRIPTION}")


def _validate_bounded_file_metadata(
    path: Path,
    metadata: os.stat_result,
    *,
    description: str,
    max_bytes: int,
    limit_description: str,
) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{description} must be a regular file: {path}")
    if metadata.st_size > max_bytes:
        raise ValueError(f"{description} exceeds {limit_description}: {path}")


def _validate_extension_artifact_size(path: Path) -> None:
    try:
        metadata = path.stat()
    except OSError as exception:
        raise ValueError(f"could not inspect extension artifact: {path}") from exception
    _validate_bounded_file_metadata(
        path,
        metadata,
        description="extension artifact",
        max_bytes=_MAX_EXTENSION_ARTIFACT_BYTES,
        limit_description=EXTENSION_ARTIFACT_LIMIT_DESCRIPTION,
    )


def _read_bounded_file(
    path: Path,
    *,
    description: str,
    max_bytes: int,
    limit_description: str,
) -> bytes:
    try:
        with path.open("rb") as artifact_file:
            _validate_bounded_file_metadata(
                path,
                os.fstat(artifact_file.fileno()),
                description=description,
                max_bytes=max_bytes,
                limit_description=limit_description,
            )
            contents = artifact_file.read(max_bytes + 1)
    except OSError as exception:
        raise ValueError(f"could not read {description}: {path}") from exception
    if len(contents) > max_bytes:
        raise ValueError(f"{description} exceeds {limit_description}: {path}")
    return contents


def _read_extension_artifact(path: Path) -> bytes:
    return _read_bounded_file(
        path,
        description="extension artifact",
        max_bytes=_MAX_EXTENSION_ARTIFACT_BYTES,
        limit_description=EXTENSION_ARTIFACT_LIMIT_DESCRIPTION,
    )


def _bounded_files_equal(first: Path, second: Path) -> bool:
    with first.open("rb") as first_file, second.open("rb") as second_file:
        first_metadata = os.fstat(first_file.fileno())
        second_metadata = os.fstat(second_file.fileno())
        if not stat.S_ISREG(first_metadata.st_mode) or not stat.S_ISREG(second_metadata.st_mode):
            return False
        if first_metadata.st_size > _MAX_EXTENSION_WHEEL_BYTES:
            raise ValueError(f"existing extension wheel exceeds {PUBLICATION_FILE_LIMIT_DESCRIPTION}")
        if first_metadata.st_size != second_metadata.st_size:
            return False
        remaining = first_metadata.st_size
        while remaining:
            chunk_size = min(remaining, 1024 * 1024)
            first_chunk = first_file.read(chunk_size)
            second_chunk = second_file.read(chunk_size)
            if not first_chunk or first_chunk != second_chunk:
                return False
            remaining -= len(first_chunk)
        return first_file.read(1) == second_file.read(1) == b""


def _license_entries(
    license_files: Iterable[str | Path],
    dist_info_root: str,
    *,
    windows_paths: bool,
) -> dict[str, bytes]:
    entries: dict[str, bytes] = {}
    casefolded_paths: dict[str, str] = {}
    for license_file in license_files:
        path = Path(license_file).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"license file is not a regular file: {path}")
        relative_path = _license_member_path(path, windows_paths=windows_paths)
        member_name = f"{dist_info_root}/licenses/{relative_path}"
        if member_name in entries:
            raise ValueError(f"license files must have unique wheel paths: {relative_path}")
        colliding_path = casefolded_paths.get(member_name.casefold())
        if colliding_path is not None:
            raise ValueError(
                "license files must not have case-insensitive wheel path collisions: "
                f"{colliding_path!r} and {relative_path!r}"
            )
        casefolded_paths[member_name.casefold()] = relative_path
        entries[member_name] = _read_bounded_file(
            path,
            description="license file",
            max_bytes=_MAX_EXTENSION_WHEEL_MEMBER_BYTES,
            limit_description=ARCHIVE_MEMBER_LIMIT_DESCRIPTION,
        )
        if sum(map(len, entries.values())) > _MAX_EXTENSION_WHEEL_UNCOMPRESSED_BYTES:
            raise ValueError(f"license files exceed {ARCHIVE_TOTAL_LIMIT_DESCRIPTION}")
    if not entries:
        raise ValueError("license_files must contain every license required by the extension artifact")
    for member_name, relative_path in sorted(casefolded_paths.items()):
        parent_name = member_name.rpartition("/")[0]
        while parent_name:
            colliding_path = casefolded_paths.get(parent_name)
            if colliding_path is not None:
                raise ValueError(
                    "license files must not have file/parent wheel path conflicts: "
                    f"{colliding_path!r} and {relative_path!r}"
                )
            parent_name = parent_name.rpartition("/")[0]
    return entries


def _license_member_path(path: Path, *, windows_paths: bool = False) -> str:
    try:
        relative_path = path.relative_to(REPOSITORY_ROOT)
    except ValueError:
        relative_path = Path(path.name)
    member_path = relative_path.as_posix()
    if (
        not member_path
        or "\\" in member_path
        or any(part in {"", ".", ".."} for part in relative_path.parts)
        or any(part != part.strip() for part in relative_path.parts)
        or any(ord(character) < 32 or ord(character) > 126 for character in member_path)
        or (windows_paths and any(_windows_path_part_is_unsafe(part) for part in relative_path.parts))
    ):
        raise ValueError(f"license file must have a safe ASCII relative path: {path}")
    return member_path


def _windows_path_part_is_unsafe(part: str) -> bool:
    return (
        part.endswith(".")
        or any(character in _WINDOWS_INVALID_PATH_CHARACTERS for character in part)
        or part.partition(".")[0].casefold() in _WINDOWS_RESERVED_PATH_NAMES
    )


def _zip_info(member_name: str) -> zipfile.ZipInfo:
    source_date_epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "315532800"))
    timestamp = datetime.fromtimestamp(max(source_date_epoch, 315532800), tz=timezone.utc)
    info = zipfile.ZipInfo(member_name, date_time=timestamp.timetuple()[:6])
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


__all__ = ["ENTRY_POINT_GROUP", "BuiltExtensionWheel", "build_extension_wheel"]

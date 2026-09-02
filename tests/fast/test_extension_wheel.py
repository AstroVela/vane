# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform as platform_module
import re
import stat
import struct
import subprocess
import sys
import zipfile
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
from packaging.version import Version

import scripts.verify_extension_wheel as verify_extension_wheel_module
import vane
import vane_packaging.archive_safety as archive_safety_module
import vane_packaging.artifact_limits as artifact_limits_module
import vane_packaging.extension_wheel as extension_wheel_module
import vane_packaging.manylinux_policy as manylinux_policy_module
from scripts import check_release_artifacts
from scripts.verify_extension_wheel import _extension_name_from_wheel, verify_extension_wheel
from vane.extensions import DynamicExtensionDependency, DynamicExtensionDescriptor
from vane_packaging.extension_wheel import ENTRY_POINT_GROUP, build_extension_wheel

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TEST_TRUST_IDENTITY = "vane-tests"


def test_extension_wheel_uses_the_shared_layered_size_budgets() -> None:
    assert extension_wheel_module._MAX_EXTENSION_WHEEL_BYTES == artifact_limits_module.MAX_PUBLICATION_FILE_BYTES
    assert extension_wheel_module._MAX_EXTENSION_WHEEL_MEMBER_BYTES == artifact_limits_module.MAX_ARCHIVE_MEMBER_BYTES
    assert (
        extension_wheel_module._MAX_EXTENSION_WHEEL_UNCOMPRESSED_BYTES
        == artifact_limits_module.MAX_ARCHIVE_UNCOMPRESSED_BYTES
    )
    assert extension_wheel_module._MAX_EXTENSION_ARTIFACT_BYTES == artifact_limits_module.MAX_EXTENSION_ARTIFACT_BYTES


def _runtime_platform() -> str:
    connection = vane.connect()
    try:
        return connection.execute("SELECT platform FROM pragma_platform()").fetchone()[0]
    finally:
        connection.close()


def _wheel_platform_tag() -> str:
    runtime_platform = _runtime_platform()
    static_tags = {
        "windows_amd64": "win_amd64",
        "windows_arm64": "win_arm64",
    }
    if runtime_platform in static_tags:
        return static_tags[runtime_platform]

    platform_tags = {
        "linux_amd64": ("manylinux_", "_x86_64"),
        "linux_amd64_musl": ("musllinux_", "_x86_64"),
        "linux_arm64": ("manylinux_", "_aarch64"),
        "linux_arm64_musl": ("musllinux_", "_aarch64"),
        "osx_amd64": ("macosx_", "_x86_64"),
        "osx_arm64": ("macosx_", "_arm64"),
    }
    prefix, suffix = platform_tags[runtime_platform]
    try:
        return next(
            tag.platform for tag in sys_tags() if tag.platform.startswith(prefix) and tag.platform.endswith(suffix)
        )
    except StopIteration:
        raise AssertionError(f"no compatible wheel platform tag for {runtime_platform}") from None


def _extension_interpreter_tag() -> str:
    return extension_wheel_module._extension_interpreter_tag()


def _other_extension_interpreter_tag() -> str:
    current = _extension_interpreter_tag()
    return next(tag for tag in ("cp310", "cp311", "cp312", "cp313", "cp314") if tag != current)


def _synthetic_elf(
    payload: bytes = b"",
    *,
    architecture: str | None = None,
    glibc_version: str | None = None,
    needed: tuple[str, ...] = (),
    filters: tuple[str, ...] = (),
    auxiliaries: tuple[str, ...] = (),
    soname: str | None = "synthetic.so",
    versioned_symbols: dict[str, tuple[str, ...]] | None = None,
    undefined_symbols: tuple[str, ...] = (),
    weak_undefined_symbols: tuple[str, ...] = (),
    hash_style: str = "sysv",
) -> bytes:
    resolved_architecture = architecture
    if resolved_architecture is None:
        resolved_architecture = "aarch64" if platform_module.machine().lower() in {"aarch64", "arm64"} else "x86_64"
    machine = {"x86_64": 62, "aarch64": 183}[resolved_architecture]
    elf_ident = b"\x7fELF" + bytes((2, 1, 1, 0)) + (b"\0" * 8)
    requirements = dict(versioned_symbols or {})
    if glibc_version is not None:
        requirements["libc.so.6"] = (*requirements.get("libc.so.6", ()), f"GLIBC_{glibc_version}")
    resolved_needed = list(needed)
    for library in requirements:
        if library not in resolved_needed:
            resolved_needed.append(library)

    string_table = bytearray(b"\0")
    string_offsets: dict[str, int] = {}

    def add_string(value: str) -> int:
        offset = string_offsets.get(value)
        if offset is not None:
            return offset
        offset = len(string_table)
        string_offsets[value] = offset
        string_table.extend(value.encode("ascii") + b"\0")
        return offset

    needed_offsets = [add_string(library) for library in resolved_needed]
    filter_offsets = [add_string(library) for library in filters]
    auxiliary_offsets = [add_string(library) for library in auxiliaries]
    soname_offset = None
    if soname is not None:
        soname_offset = add_string(soname)
    for library, versions in requirements.items():
        add_string(library)
        for version in versions:
            add_string(version)
    for symbol_name in (*undefined_symbols, *weak_undefined_symbols):
        add_string(symbol_name)

    elf_header_size = 64
    program_header_size = 56
    program_header_count = 2
    dynamic_offset = elf_header_size + program_header_size * program_header_count
    dynamic_entry_count = (
        len(needed_offsets)
        + len(filter_offsets)
        + len(auxiliary_offsets)
        + 3
        + (soname_offset is not None)
        + (2 if requirements else 0)
        + (3 if undefined_symbols or weak_undefined_symbols else 0)
    )
    dynamic_size = dynamic_entry_count * 16
    string_table_offset = dynamic_offset + dynamic_size
    version_table_padding = b"\0" * (-(string_table_offset + len(string_table)) % 4)
    version_table_offset = string_table_offset + len(string_table) + len(version_table_padding)
    version_table = bytearray()
    version_index = 2
    for file_index, (library, versions) in enumerate(requirements.items()):
        if not versions:
            raise ValueError("synthetic version-needed entries require at least one version")
        block_size = 16 + 16 * len(versions)
        next_offset = block_size if file_index < len(requirements) - 1 else 0
        version_table.extend(struct.pack("<HHIII", 1, len(versions), string_offsets[library], 16, next_offset))
        for auxiliary_index, version in enumerate(versions):
            auxiliary_next_offset = 16 if auxiliary_index < len(versions) - 1 else 0
            version_table.extend(
                struct.pack("<IHHII", 0, 0, version_index, string_offsets[version], auxiliary_next_offset)
            )
            version_index += 1

    dynamic_symbol_padding = b"\0" * (-(version_table_offset + len(version_table)) % 8)
    dynamic_symbol_table_offset = version_table_offset + len(version_table) + len(dynamic_symbol_padding)
    dynamic_symbol_table = bytearray(struct.pack("<IBBHQQ", 0, 0, 0, 0, 0, 0))
    for symbol_name in undefined_symbols:
        dynamic_symbol_table.extend(struct.pack("<IBBHQQ", string_offsets[symbol_name], 1 << 4, 0, 0, 0, 0))
    for symbol_name in weak_undefined_symbols:
        dynamic_symbol_table.extend(struct.pack("<IBBHQQ", string_offsets[symbol_name], 2 << 4, 0, 0, 0, 0))
    hash_table_offset = dynamic_symbol_table_offset + len(dynamic_symbol_table)
    symbol_count = 1 + len(undefined_symbols) + len(weak_undefined_symbols)
    if hash_style == "sysv":
        hash_table_tag = 4
        hash_table = (
            struct.pack("<II", 1, symbol_count)
            + struct.pack("<I", 1 if symbol_count > 1 else 0)
            + b"\0" * (4 * symbol_count)
        )
    elif hash_style == "gnu":
        hash_table_tag = 0x6FFFFEF5
        hash_table = (
            struct.pack("<IIIIQ", 1, 1, 1, 0, 0)
            + struct.pack("<I", 1)
            + b"\0" * (4 * (symbol_count - 2))
            + struct.pack("<I", 1)
        )
    else:
        raise ValueError(f"unsupported synthetic ELF hash style: {hash_style}")

    dynamic_entries = [*(struct.pack("<qQ", 1, offset) for offset in needed_offsets)]
    dynamic_entries.extend(struct.pack("<qQ", 0x7FFFFFFF, offset) for offset in filter_offsets)
    dynamic_entries.extend(struct.pack("<qQ", 0x7FFFFFFD, offset) for offset in auxiliary_offsets)
    dynamic_entries.extend(
        (
            struct.pack("<qQ", 5, string_table_offset),
            struct.pack("<qQ", 10, len(string_table)),
        )
    )
    if soname_offset is not None:
        dynamic_entries.append(struct.pack("<qQ", 14, soname_offset))
    if requirements:
        dynamic_entries.append(struct.pack("<qQ", 0x6FFFFFFE, version_table_offset))
        dynamic_entries.append(struct.pack("<qQ", 0x6FFFFFFF, len(requirements)))
    if undefined_symbols or weak_undefined_symbols:
        dynamic_entries.append(struct.pack("<qQ", hash_table_tag, hash_table_offset))
        dynamic_entries.append(struct.pack("<qQ", 6, dynamic_symbol_table_offset))
        dynamic_entries.append(struct.pack("<qQ", 11, 24))
    dynamic_entries.append(struct.pack("<qQ", 0, 0))
    body = (
        b"".join(dynamic_entries)
        + bytes(string_table)
        + version_table_padding
        + bytes(version_table)
        + (
            dynamic_symbol_padding + bytes(dynamic_symbol_table) + hash_table
            if undefined_symbols or weak_undefined_symbols
            else b""
        )
        + payload
    )
    file_size = dynamic_offset + len(body)
    header = struct.pack(
        "<16sHHIQQQIHHHHHH",
        elf_ident,
        3,
        machine,
        1,
        0,
        elf_header_size,
        0,
        0,
        elf_header_size,
        program_header_size,
        program_header_count,
        0,
        0,
        0,
    )
    load_segment = struct.pack("<IIQQQQQQ", 1, 5, 0, 0, 0, file_size, file_size, 0x1000)
    dynamic_segment = struct.pack(
        "<IIQQQQQQ",
        2,
        4,
        dynamic_offset,
        dynamic_offset,
        dynamic_offset,
        dynamic_size,
        dynamic_size,
        8,
    )
    return header + load_segment + dynamic_segment + body


def _synthetic_pe(
    *,
    architecture: str,
    forwarded_exports: tuple[str, ...] = (),
    imports: tuple[str, ...] = (),
    delay_imports: tuple[str, ...] = (),
    payload: bytes = b"",
) -> bytes:
    section_virtual_address = 0x1000
    section = bytearray()
    export_directory_offset = len(section)
    export_directory_size = 0
    if forwarded_exports:
        section.extend(b"\0" * 40)
        export_address_table_offset = len(section)
        section.extend(b"\0" * (4 * len(forwarded_exports)))
        forwarder_rvas: list[int] = []
        for forwarder in forwarded_exports:
            forwarder_rvas.append(section_virtual_address + len(section))
            section.extend(forwarder.encode("ascii") + b"\0")
        export_directory_size = len(section) - export_directory_offset
        struct.pack_into(
            "<IIHHIIIIIII",
            section,
            export_directory_offset,
            0,
            0,
            0,
            0,
            0,
            1,
            len(forwarded_exports),
            0,
            section_virtual_address + export_address_table_offset,
            0,
            0,
        )
        for index, forwarder_rva in enumerate(forwarder_rvas):
            struct.pack_into("<I", section, export_address_table_offset + index * 4, forwarder_rva)

    import_table_offset = len(section)
    import_table_size = 20 * (len(imports) + 1) if imports else 0
    section.extend(b"\0" * import_table_size)
    delay_table_offset = len(section)
    delay_table_size = 32 * (len(delay_imports) + 1) if delay_imports else 0
    section.extend(b"\0" * delay_table_size)

    import_name_rvas: list[int] = []
    for library in imports:
        import_name_rvas.append(section_virtual_address + len(section))
        section.extend(library.encode("ascii") + b"\0")
    delay_name_rvas: list[int] = []
    for library in delay_imports:
        delay_name_rvas.append(section_virtual_address + len(section))
        section.extend(library.encode("ascii") + b"\0")
    for index, name_rva in enumerate(import_name_rvas):
        struct.pack_into("<5I", section, import_table_offset + index * 20, 0, 0, 0, name_rva, 0)
    for index, name_rva in enumerate(delay_name_rvas):
        struct.pack_into("<8I", section, delay_table_offset + index * 32, 1, name_rva, 0, 0, 0, 0, 0, 0)
    section.extend(payload)
    if not section:
        section.append(0)

    file_alignment = 0x200
    section_alignment = 0x1000
    raw_section_size = (len(section) + file_alignment - 1) & ~(file_alignment - 1)
    pe_header_offset = 64
    optional_header_size = 240
    section_table_offset = pe_header_offset + 4 + 20 + optional_header_size
    size_of_headers = (section_table_offset + 40 + file_alignment - 1) & ~(file_alignment - 1)

    dos_header = bytearray(pe_header_offset)
    dos_header[:2] = b"MZ"
    struct.pack_into("<I", dos_header, 60, pe_header_offset)
    machine = {"amd64": 0x8664, "arm64": 0xAA64}[architecture]
    coff_header = struct.pack("<HHIIIHH", machine, 1, 0, 0, 0, optional_header_size, 0x2022)
    optional_header = bytearray(optional_header_size)
    struct.pack_into("<H", optional_header, 0, 0x20B)
    struct.pack_into("<I", optional_header, 8, raw_section_size)
    struct.pack_into("<Q", optional_header, 24, 0x180000000)
    struct.pack_into("<I", optional_header, 32, section_alignment)
    struct.pack_into("<I", optional_header, 36, file_alignment)
    struct.pack_into("<I", optional_header, 56, section_virtual_address + section_alignment)
    struct.pack_into("<I", optional_header, 60, size_of_headers)
    struct.pack_into("<H", optional_header, 68, 3)
    struct.pack_into("<I", optional_header, 108, 16)
    if forwarded_exports:
        struct.pack_into(
            "<II",
            optional_header,
            112,
            section_virtual_address + export_directory_offset,
            export_directory_size,
        )
    if imports:
        struct.pack_into(
            "<II",
            optional_header,
            112 + 8,
            section_virtual_address + import_table_offset,
            import_table_size,
        )
    if delay_imports:
        struct.pack_into(
            "<II",
            optional_header,
            112 + 13 * 8,
            section_virtual_address + delay_table_offset,
            delay_table_size,
        )
    section_header = struct.pack(
        "<8sIIIIIIHHI",
        b".rdata\0\0",
        len(section),
        section_virtual_address,
        raw_section_size,
        size_of_headers,
        0,
        0,
        0,
        0,
        0x40000040,
    )
    headers = bytes(dos_header) + b"PE\0\0" + coff_header + bytes(optional_header) + section_header
    return headers.ljust(size_of_headers, b"\0") + bytes(section).ljust(raw_section_size, b"\0")


def _synthetic_macho(
    *,
    architecture: str,
    minimum_macos: tuple[int, int, int],
    cpu_subtype: int | None = None,
    file_type: int = 6,
    build_platform: int = 1,
    dynamic_libraries: tuple[str, ...] = (),
    dynamic_library_command: int = 0xC,
    rpaths: tuple[str, ...] = (),
    payload: bytes = b"",
) -> bytes:
    encoded_version = (minimum_macos[0] << 16) | (minimum_macos[1] << 8) | minimum_macos[2]
    build_version = struct.pack("<6I", 0x32, 24, build_platform, encoded_version, encoded_version, 0)
    load_commands = [build_version]
    for install_name in dynamic_libraries:
        encoded_name = install_name.encode("utf-8") + b"\0"
        command_size = (24 + len(encoded_name) + 7) & ~7
        load_commands.append(
            struct.pack("<6I", dynamic_library_command, command_size, 24, 0, 0, 0)
            + encoded_name
            + b"\0" * (command_size - 24 - len(encoded_name))
        )
    for rpath in rpaths:
        encoded_path = rpath.encode("utf-8") + b"\0"
        command_size = (12 + len(encoded_path) + 7) & ~7
        load_commands.append(
            struct.pack("<3I", 0x8000001C, command_size, 12)
            + encoded_path
            + b"\0" * (command_size - 12 - len(encoded_path))
        )
    commands = b"".join(load_commands)
    cpu_type = {"x86_64": 0x01000007, "arm64": 0x0100000C}[architecture]
    resolved_cpu_subtype = {"x86_64": 3, "arm64": 0}[architecture] if cpu_subtype is None else cpu_subtype
    header = struct.pack(
        "<8I",
        0xFEEDFACF,
        cpu_type,
        resolved_cpu_subtype,
        file_type,
        len(load_commands),
        len(commands),
        0,
        0,
    )
    return header + commands + payload


def _macos_minimum_from_tag(platform_tag: str) -> tuple[int, int, int]:
    _family, major, minor, _architecture = platform_tag.split("_", 3)
    version = (int(major), int(minor), 0)
    return (11, 0, 0) if version[:2] == (10, 16) else version


def _synthetic_fat_macho(
    *,
    x86_minimum_macos: tuple[int, int, int],
    arm_minimum_macos: tuple[int, int, int],
    dynamic_libraries: tuple[str, ...] = (),
    rpaths: tuple[str, ...] = (),
) -> bytes:
    slices = (
        (
            0x01000007,
            3,
            _synthetic_macho(
                architecture="x86_64",
                minimum_macos=x86_minimum_macos,
                dynamic_libraries=dynamic_libraries,
                rpaths=rpaths,
            ),
        ),
        (
            0x0100000C,
            0,
            _synthetic_macho(
                architecture="arm64",
                minimum_macos=arm_minimum_macos,
                dynamic_libraries=dynamic_libraries,
                rpaths=rpaths,
            ),
        ),
    )
    alignment = 3
    table_size = 8 + 20 * len(slices)
    entries: list[tuple[int, int, int, int, int]] = []
    payload = bytearray()
    current_offset = table_size
    for cpu_type, cpu_subtype, contents in slices:
        aligned_offset = (current_offset + (1 << alignment) - 1) & ~((1 << alignment) - 1)
        payload.extend(b"\0" * (aligned_offset - current_offset))
        entries.append((cpu_type, cpu_subtype, aligned_offset, len(contents), alignment))
        payload.extend(contents)
        current_offset = aligned_offset + len(contents)
    header = struct.pack(">2I", 0xCAFEBABE, len(entries))
    architecture_table = b"".join(
        struct.pack(">5I", cpu_type, cpu_subtype, offset, size, slice_alignment)
        for cpu_type, cpu_subtype, offset, size, slice_alignment in entries
    )
    return header + architecture_table + payload


def _write_artifact(
    path: Path,
    payload: bytes = b"Vane extension wheel test payload",
    *,
    architecture: str | None = None,
    glibc_version: str | None = None,
    needed: tuple[str, ...] = (),
    filters: tuple[str, ...] = (),
    auxiliaries: tuple[str, ...] = (),
    versioned_symbols: dict[str, tuple[str, ...]] | None = None,
    undefined_symbols: tuple[str, ...] = (),
    weak_undefined_symbols: tuple[str, ...] = (),
    windows_forwarded_exports: tuple[str, ...] = (),
    windows_imports: tuple[str, ...] = (),
    windows_delay_imports: tuple[str, ...] = (),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32" and architecture is None and glibc_version is None:
        path.write_bytes(
            _synthetic_pe(
                architecture="arm64" if platform_module.machine().lower() in {"aarch64", "arm64"} else "amd64",
                forwarded_exports=windows_forwarded_exports,
                imports=windows_imports,
                delay_imports=windows_delay_imports,
                payload=payload,
            )
        )
    elif sys.platform == "darwin" and architecture is None and glibc_version is None:
        platform_tag = _wheel_platform_tag()
        path.write_bytes(
            _synthetic_macho(
                architecture="arm64" if platform_tag.endswith("_arm64") else "x86_64",
                minimum_macos=_macos_minimum_from_tag(platform_tag),
                payload=payload,
            )
        )
    else:
        path.write_bytes(
            _synthetic_elf(
                payload,
                architecture=architecture,
                glibc_version=glibc_version,
                needed=needed,
                filters=filters,
                auxiliaries=auxiliaries,
                versioned_symbols=versioned_symbols,
                undefined_symbols=undefined_symbols,
                weak_undefined_symbols=weak_undefined_symbols,
            )
        )
    return path


def _write_minimal_base_wheel(
    directory: Path,
    *,
    distribution_name: str = "vane-ai",
    version: str | None = None,
    interpreter_tag: str | None = None,
    abi_tag: str | None = None,
    platform_tag: str | None = None,
    requires_python: str | None = "<3.15,>=3.10",
    metadata_version: str | None = "2.4",
    build_tag: str | None = None,
    native_architecture: str | None = None,
    native_glibc_version: str | None = None,
    native_needed: tuple[str, ...] = (),
    native_filters: tuple[str, ...] = (),
    native_auxiliaries: tuple[str, ...] = (),
    native_versioned_symbols: dict[str, tuple[str, ...]] | None = None,
    native_undefined_symbols: tuple[str, ...] = (),
    native_weak_undefined_symbols: tuple[str, ...] = (),
    native_macos_version: tuple[int, int, int] | None = None,
    native_macos_dynamic_libraries: tuple[str, ...] = (),
    native_macos_rpaths: tuple[str, ...] = (),
    native_windows_forwarded_exports: tuple[str, ...] = (),
    native_windows_imports: tuple[str, ...] = (),
    native_windows_delay_imports: tuple[str, ...] = (),
) -> Path:
    wheel_version = version or vane.__version__
    wheel_interpreter_tag = _extension_interpreter_tag() if interpreter_tag is None else interpreter_tag
    wheel_abi_tag = wheel_interpreter_tag if abi_tag is None else abi_tag
    wheel_platform_tag = _wheel_platform_tag() if platform_tag is None else platform_tag
    wheel_tag = f"{wheel_interpreter_tag}-{wheel_abi_tag}-{wheel_platform_tag}"
    wheel_distribution = distribution_name.replace("-", "_")
    build_component = f"-{build_tag}" if build_tag is not None else ""
    path = directory / f"{wheel_distribution}-{wheel_version}{build_component}-{wheel_tag}.whl"
    dist_info = f"{wheel_distribution}-{wheel_version}.dist-info"
    metadata = (
        f"Name: {distribution_name}\nVersion: {wheel_version}\nLicense-Expression: Apache-2.0\nLicense-File: LICENSE\n"
    )
    if metadata_version is not None:
        metadata = f"Metadata-Version: {metadata_version}\n" + metadata
    if requires_python is not None:
        metadata += f"Requires-Python: {requires_python}\n"
    for requirement in check_release_artifacts.EXPECTED_REQUIRES_DIST:
        metadata += f"Requires-Dist: {requirement}\n"
    for extra in check_release_artifacts.EXPECTED_PROVIDES_EXTRA:
        metadata += f"Provides-Extra: {extra}\n"
    primary_platform_tag = wheel_platform_tag.split(".", 1)[0]
    native_suffix = ".pyd" if primary_platform_tag.startswith("win_") else ".so"
    if primary_platform_tag.startswith("win_"):
        native_contents = _synthetic_pe(
            architecture="arm64" if primary_platform_tag == "win_arm64" else "amd64",
            forwarded_exports=native_windows_forwarded_exports,
            imports=native_windows_imports,
            delay_imports=native_windows_delay_imports,
        )
    elif primary_platform_tag.startswith("macosx_"):
        minimum_macos = native_macos_version or _macos_minimum_from_tag(primary_platform_tag)
        if primary_platform_tag.endswith("_universal2"):
            native_contents = _synthetic_fat_macho(
                x86_minimum_macos=minimum_macos,
                arm_minimum_macos=max(minimum_macos, (11, 0, 0)),
                dynamic_libraries=native_macos_dynamic_libraries,
                rpaths=native_macos_rpaths,
            )
        else:
            native_contents = _synthetic_macho(
                architecture="arm64" if primary_platform_tag.endswith("_arm64") else "x86_64",
                minimum_macos=minimum_macos,
                dynamic_libraries=native_macos_dynamic_libraries,
                rpaths=native_macos_rpaths,
            )
    else:
        if native_architecture is None:
            native_architecture = "aarch64" if "aarch64" in wheel_platform_tag else "x86_64"
        native_contents = _synthetic_elf(
            b"test native extension",
            architecture=native_architecture,
            glibc_version=native_glibc_version,
            needed=native_needed,
            filters=native_filters,
            auxiliaries=native_auxiliaries,
            versioned_symbols=native_versioned_symbols,
            undefined_symbols=native_undefined_symbols,
            weak_undefined_symbols=native_weak_undefined_symbols,
        )
    entries = {
        "vane/py.typed": b"",
        f"vane/_native.test{native_suffix}": native_contents,
        "vane/_native/__init__.pyi": b"",
        "vane/_native/_func.pyi": b"",
        "vane/_native/_sqltypes.pyi": b"",
        "vane/_native/ray_cxx.pyi": b"",
        "vane/sqltypes/__init__.pyi": b"",
        "vane/udf.pyi": b"",
        f"{dist_info}/METADATA": metadata.encode("utf-8"),
        f"{dist_info}/WHEEL": "\n".join(
            (
                "Wheel-Version: 1.0",
                "Generator: vane-test",
                "Root-Is-Purelib: false",
                f"Tag: {wheel_tag}",
                "",
            )
        ).encode("utf-8"),
        f"{dist_info}/licenses/LICENSE": b"test license",
    }
    record_name = f"{dist_info}/RECORD"
    entries[record_name] = extension_wheel_module._record(entries, record_name).encode("utf-8")
    with zipfile.ZipFile(path, mode="w") as wheel:
        for member_name, contents in entries.items():
            wheel.writestr(member_name, contents)
    return path


def _rewrite_wheel(
    source: Path,
    destination: Path,
    *,
    transforms: dict[str, Callable[[bytes], bytes]] | None = None,
    extra_members: dict[str, bytes] | None = None,
    removed_members: set[str] | None = None,
    update_record: bool = True,
) -> Path:
    transforms = transforms or {}
    extra_members = extra_members or {}
    removed_members = removed_members or set()
    with zipfile.ZipFile(source) as input_wheel:
        member_info = {member.filename: member for member in input_wheel.infolist()}
        entries = {
            member.filename: input_wheel.read(member)
            for member in input_wheel.infolist()
            if member.filename not in removed_members
        }
    for member_name, transform in transforms.items():
        entries[member_name] = transform(entries[member_name])
    entries.update(extra_members)
    if update_record:
        record_members = [name for name in entries if name.endswith(".dist-info/RECORD")]
        assert len(record_members) == 1
        record_member = record_members[0]
        recorded_entries = {name: contents for name, contents in entries.items() if name != record_member}
        entries[record_member] = extension_wheel_module._record(recorded_entries, record_member).encode("utf-8")
    with zipfile.ZipFile(destination, mode="w") as output_wheel:
        for member_name, contents in entries.items():
            output_wheel.writestr(member_info.get(member_name, member_name), contents)
    return destination


def _rewrite_wheel_metadata(source: Path, destination_directory: Path, transform) -> Path:
    destination_directory.mkdir()
    with zipfile.ZipFile(source) as input_wheel:
        metadata_members = [name for name in input_wheel.namelist() if name.endswith(".dist-info/METADATA")]
    assert len(metadata_members) == 1
    metadata_member = metadata_members[0]
    return _rewrite_wheel(
        source,
        destination_directory / source.name,
        transforms={metadata_member: lambda contents: transform(contents.decode("utf-8")).encode("utf-8")},
    )


def _relabel_wheel_platform(
    source: Path,
    destination_directory: Path,
    *,
    original: str,
    replacement: str,
) -> Path:
    destination_directory.mkdir()
    with zipfile.ZipFile(source) as input_wheel:
        wheel_members = [name for name in input_wheel.namelist() if name.endswith(".dist-info/WHEEL")]
        platform_members = [
            name for name in input_wheel.namelist() if name.endswith(".dist-info/vane-extension-platform.json")
        ]
    assert len(wheel_members) == 1
    assert len(platform_members) <= 1
    transforms = {
        wheel_members[0]: lambda contents: contents.replace(
            original.encode("ascii"),
            replacement.encode("ascii"),
        )
    }
    if platform_members:
        transforms[platform_members[0]] = lambda contents: contents.replace(
            original.encode("ascii"),
            replacement.encode("ascii"),
        )
    return _rewrite_wheel(
        source,
        destination_directory / source.name.replace(original, replacement),
        transforms=transforms,
    )


def _descriptor(
    artifact_path: Path,
    *,
    name: str = "sample",
    trust_identity: str = TEST_TRUST_IDENTITY,
    source_id: str | None = None,
    vane_version: str | None = None,
    platform: str | None = None,
    dependencies: tuple[DynamicExtensionDependency, ...] = (),
) -> DynamicExtensionDescriptor:
    return DynamicExtensionDescriptor(
        name=name,
        extension_version="test-version",
        abi_type="CPP",
        duckdb_source_id=source_id or vane.__git_revision__,
        vane_version=vane_version or vane.__version__,
        platform=platform or _runtime_platform(),
        sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        trust_identity=trust_identity,
        dependencies=dependencies,
    )


def _descriptor_digest(descriptor: DynamicExtensionDescriptor) -> str:
    return hashlib.sha256(descriptor.to_json().encode("utf-8")).hexdigest()


def _assert_public_descriptor_version(
    distribution_version: str,
    vane_version: str,
    descriptor: DynamicExtensionDescriptor,
) -> None:
    generated = Version(distribution_version)
    base = Version(vane_version)
    if base.pre is None:
        prerelease_stage = 0 if base.dev is not None and base.post is None else 4
        prerelease_number = 0
    else:
        prerelease_stage = {"a": 1, "b": 2, "rc": 3}[base.pre[0]]
        prerelease_number = base.pre[1]
    vane_release = list(base.release)
    while len(vane_release) > 1 and vane_release[-1] == 0:
        vane_release.pop()
    vane_release.extend([0] * (8 - len(vane_release)))
    descriptor_digest = _descriptor_digest(descriptor)
    digest_chunks = tuple(int(descriptor_digest[offset : offset + 8], 16) for offset in range(0, 64, 8))
    expected_release = (
        *vane_release,
        prerelease_stage,
        prerelease_number,
        int(base.post is not None),
        base.post or 0,
        int(base.dev is None),
        base.dev or 0,
        *digest_chunks,
    )

    assert generated.local is None
    assert generated.epoch == base.epoch
    assert generated.release == expected_release
    assert generated.pre is None
    assert generated.post is None
    assert generated.dev is None
    assert all(component <= (1 << 32) - 1 for component in digest_chunks)


@pytest.fixture
def synthetic_descriptor_factory(monkeypatch):
    def create_descriptor(artifact_path, *, name, trust_identity, dependencies):
        return _descriptor(
            artifact_path,
            name=name,
            trust_identity=trust_identity,
            dependencies=dependencies,
        )

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_descriptor)


def _build_sample_wheel(
    tmp_path: Path,
    *,
    artifact_path: Path | None = None,
    output_name: str = "dist",
    platform_tag: str | None = None,
    dependencies: tuple[Path, ...] = (),
):
    resolved_platform_tag = platform_tag or _wheel_platform_tag()
    if artifact_path is None:
        artifact_path = tmp_path / "sample.duckdb_extension"
        if resolved_platform_tag.startswith("win_"):
            architecture = "arm64" if resolved_platform_tag == "win_arm64" else "amd64"
            artifact_path.write_bytes(_synthetic_pe(architecture=architecture))
        elif resolved_platform_tag.startswith("macosx_"):
            architecture = "arm64" if resolved_platform_tag.endswith("_arm64") else "x86_64"
            artifact_path.write_bytes(
                _synthetic_macho(
                    architecture=architecture,
                    minimum_macos=_macos_minimum_from_tag(resolved_platform_tag),
                )
            )
        else:
            architecture = "aarch64" if resolved_platform_tag.endswith("_aarch64") else "x86_64"
            artifact_path = _write_artifact(
                artifact_path,
                architecture=architecture,
            )
    return build_extension_wheel(
        artifact=artifact_path,
        extension_name="sample",
        output_directory=tmp_path / output_name,
        platform_tag=resolved_platform_tag,
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0 AND MIT",
        license_files=[
            REPOSITORY_ROOT / "LICENSE",
            REPOSITORY_ROOT / "NOTICE",
            REPOSITORY_ROOT / "LICENSES" / "DuckDB-MIT.txt",
        ],
        dependency_wheels=dependencies,
        dependency_trust_identities=(TEST_TRUST_IDENTITY,) if dependencies else (),
    )


def test_platform_wheel_contains_one_verified_artifact_descriptor_and_provider(
    tmp_path,
    synthetic_descriptor_factory,
):
    artifact_path = _write_artifact(tmp_path / "sample.duckdb_extension")
    dependency_path = _write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency")
    platform_tag = _wheel_platform_tag()
    dependency_built = build_extension_wheel(
        artifact=dependency_path,
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=platform_tag,
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    dependency = dependency_built.descriptor

    built = _build_sample_wheel(tmp_path, artifact_path=artifact_path, dependencies=(dependency_built.path,))

    descriptor_digest = _descriptor_digest(built.descriptor)
    interpreter_tag = _extension_interpreter_tag()
    assert built.distribution_name == "vane-extension-sample"
    _assert_public_descriptor_version(built.distribution_version, vane.__version__, built.descriptor)
    assert built.wheel_tag == f"{interpreter_tag}-none-{platform_tag}"
    assert built.path.name == (
        f"vane_extension_sample-{built.distribution_version}-{interpreter_tag}-none-{platform_tag}.whl"
    )
    parsed_name, parsed_version, _, parsed_tags = parse_wheel_filename(built.path.name)
    assert parsed_name == "vane-extension-sample"
    assert parsed_version == Version(built.distribution_version)
    assert {str(tag) for tag in parsed_tags} == {built.wheel_tag}
    assert _extension_name_from_wheel(built.path) == "sample"
    assert built.descriptor.dependencies == (
        DynamicExtensionDependency(
            name=dependency.name,
            extension_version=dependency.extension_version,
            sha256=dependency.sha256,
        ),
    )

    package_root = f"vane_extensions/sample_{descriptor_digest}"
    dist_info_root = f"vane_extension_sample-{built.distribution_version}.dist-info"
    with zipfile.ZipFile(built.path) as wheel:
        names = set(wheel.namelist())
        assert names == {
            f"{package_root}/__init__.py",
            f"{package_root}/sample.duckdb_extension",
            f"{package_root}/sample.dynamic-extension.json",
            f"{dist_info_root}/METADATA",
            f"{dist_info_root}/WHEEL",
            f"{dist_info_root}/entry_points.txt",
            f"{dist_info_root}/vane-extension-platform.json",
            f"{dist_info_root}/RECORD",
            f"{dist_info_root}/licenses/LICENSE",
            f"{dist_info_root}/licenses/NOTICE",
            f"{dist_info_root}/licenses/LICENSES/DuckDB-MIT.txt",
        }
        assert not any(name.startswith("vane/") for name in names)
        assert wheel.read(f"{package_root}/sample.duckdb_extension") == artifact_path.read_bytes()
        descriptor = DynamicExtensionDescriptor.from_json(wheel.read(f"{package_root}/sample.dynamic-extension.json"))
        assert descriptor == built.descriptor
        assert descriptor.sha256 == hashlib.sha256(artifact_path.read_bytes()).hexdigest()

        metadata = wheel.read(f"{dist_info_root}/METADATA").decode("utf-8")
        assert metadata.startswith("Metadata-Version: 2.4\n")
        assert f"Name: {built.distribution_name}" in metadata
        assert f"Version: {built.distribution_version}" in metadata
        assert "License-Expression: Apache-2.0 AND MIT" in metadata
        assert "Requires-Python: >=3.10,<3.15" in metadata
        assert f"Requires-Dist: vane-ai==={vane.__version__}" in metadata
        assert f"Requires-Dist: vane-extension-dependency==={dependency_built.distribution_version}" in metadata
        assert "License-File: LICENSE" in metadata
        assert "License-File: NOTICE" in metadata
        assert "License-File: LICENSES/DuckDB-MIT.txt" in metadata
        assert (
            wheel.read(f"{dist_info_root}/WHEEL")
            .decode("utf-8")
            .endswith(f"Tag: {interpreter_tag}-none-{platform_tag}\n")
        )
        assert wheel.read(f"{dist_info_root}/entry_points.txt").decode("utf-8") == (
            f"[{ENTRY_POINT_GROUP}]\nsample = vane_extensions.sample_{descriptor_digest}:provider\n"
        )
        assert wheel.read(f"{dist_info_root}/vane-extension-platform.json") == (
            extension_wheel_module._platform_build_details_bytes(
                extension_wheel_module._platform_build_details_for_build(platform_tag)
            )
        )
        assert "LocalExtensionProvider" in wheel.read(f"{package_root}/__init__.py").decode("utf-8")

        record_rows = list(csv.reader(io.StringIO(wheel.read(f"{dist_info_root}/RECORD").decode("utf-8"))))
        assert {row[0] for row in record_rows} == names
        assert next(row for row in record_rows if row[0].endswith("/RECORD")) == [
            f"{dist_info_root}/RECORD",
            "",
            "",
        ]

    if os.name != "nt":
        assert stat.S_IMODE(built.path.stat().st_mode) == 0o644


def test_clean_verifier_rejects_a_platform_wheel_with_a_broader_python_tag(
    tmp_path,
    synthetic_descriptor_factory,
):
    built = _build_sample_wheel(tmp_path)
    interpreter_tag = _extension_interpreter_tag()
    with zipfile.ZipFile(built.path) as wheel:
        wheel_metadata = next(name for name in wheel.namelist() if name.endswith(".dist-info/WHEEL"))
    tampered_directory = tmp_path / "broader-python-tag"
    tampered_directory.mkdir()
    tampered_name = built.path.name.replace(f"-{interpreter_tag}-none-", "-py3-none-")
    tampered_wheel = _rewrite_wheel(
        built.path,
        tampered_directory / tampered_name,
        transforms={
            wheel_metadata: lambda contents: contents.replace(
                f"{interpreter_tag}-none-".encode(),
                b"py3-none-",
            )
        },
    )

    with pytest.raises(RuntimeError, match=f"must use exactly one {interpreter_tag}-none platform tag"):
        _extension_name_from_wheel(tampered_wheel)


def test_clean_verifier_rejects_an_extension_wheel_with_a_relabelled_glibc_floor(
    tmp_path,
    monkeypatch,
):
    artifact = _write_artifact(
        tmp_path / "sample.duckdb_extension",
        architecture="x86_64",
        glibc_version="2.39",
    )

    def create_descriptor(path, *, name, trust_identity, dependencies):
        return _descriptor(
            path,
            name=name,
            trust_identity=trust_identity,
            platform="linux_amd64",
            dependencies=dependencies,
        )

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_descriptor)
    built = _build_sample_wheel(
        tmp_path,
        artifact_path=artifact,
        platform_tag="manylinux_2_39_x86_64",
    )
    relabelled_wheel = _relabel_wheel_platform(
        built.path,
        tmp_path / "relabelled-extension",
        original="manylinux_2_39_x86_64",
        replacement="manylinux_2_17_x86_64",
    )

    with pytest.raises(RuntimeError, match="requires glibc 2.39.*manylinux_2_17_x86_64"):
        _extension_name_from_wheel(relabelled_wheel)


def test_platform_wheel_exact_requirements_reject_local_version_variants(
    tmp_path,
    synthetic_descriptor_factory,
):
    dependency_built = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency"),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    built = _build_sample_wheel(tmp_path, dependencies=(dependency_built.path,))
    with zipfile.ZipFile(built.path) as wheel:
        metadata_member = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = wheel.read(metadata_member).decode("utf-8")
    requirements = tuple(
        Requirement(line.removeprefix("Requires-Dist: "))
        for line in metadata.splitlines()
        if line.startswith("Requires-Dist: ")
    )

    assert {requirement.name for requirement in requirements} == {
        "vane-ai",
        "vane-extension-dependency",
    }
    for requirement in requirements:
        specifier = next(iter(requirement.specifier))
        assert specifier.operator == "==="
        assert not requirement.specifier.contains(f"{Version(specifier.version).public}+private")


def test_platform_wheel_requires_one_explicit_platform_tag(tmp_path):
    artifact_path = _write_artifact(tmp_path / "sample.duckdb_extension")

    with pytest.raises(ValueError, match="platform_tag"):
        build_extension_wheel(
            artifact=artifact_path,
            extension_name="sample",
            output_directory=tmp_path / "dist",
            platform_tag="linux-x86_64",
            trust_identity=TEST_TRUST_IDENTITY,
            license_expression="Apache-2.0 AND MIT",
            license_files=[REPOSITORY_ROOT / "LICENSE"],
        )


@pytest.mark.parametrize("extension_name", ["sample__nested", "sample_"])
def test_platform_wheel_rejects_names_that_cannot_form_normalized_wheel_names(
    tmp_path,
    extension_name,
):
    artifact_path = _write_artifact(tmp_path / f"{extension_name}.duckdb_extension")

    with pytest.raises(ValueError, match="wheel-safe"):
        build_extension_wheel(
            artifact=artifact_path,
            extension_name=extension_name,
            output_directory=tmp_path / "dist",
            platform_tag="linux_x86_64",
            trust_identity=TEST_TRUST_IDENTITY,
            license_expression="Apache-2.0",
            license_files=[REPOSITORY_ROOT / "LICENSE"],
        )


def test_platform_wheel_rejects_names_that_exceed_generated_component_limits(tmp_path, monkeypatch):
    extension_name = "a" * 100
    artifact_path = _write_artifact(tmp_path / f"{extension_name}.duckdb_extension")

    def create_descriptor(path, *, name, trust_identity, dependencies):
        return _descriptor(
            path,
            name=name,
            trust_identity=trust_identity,
            dependencies=dependencies,
        )

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_descriptor)
    with pytest.raises(ValueError, match="generated wheel path component exceeds"):
        build_extension_wheel(
            artifact=artifact_path,
            extension_name=extension_name,
            output_directory=tmp_path / "dist",
            platform_tag=_wheel_platform_tag(),
            trust_identity=TEST_TRUST_IDENTITY,
            license_expression="Apache-2.0",
            license_files=[REPOSITORY_ROOT / "LICENSE"],
        )


@pytest.mark.parametrize("extension_name", ["con", "aux", "nul", "prn", "com1", "lpt9"])
def test_platform_wheel_rejects_windows_reserved_extension_names(
    tmp_path,
    monkeypatch,
    extension_name,
):
    artifact_path = _write_artifact(tmp_path / f"{extension_name}.duckdb_extension")

    def create_descriptor(path, *, name, trust_identity, dependencies):
        return _descriptor(
            path,
            name=name,
            trust_identity=trust_identity,
            platform="windows_amd64",
            dependencies=dependencies,
        )

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_descriptor)
    with pytest.raises(ValueError, match="Windows-reserved"):
        build_extension_wheel(
            artifact=artifact_path,
            extension_name=extension_name,
            output_directory=tmp_path / "dist",
            platform_tag="win_amd64",
            trust_identity=TEST_TRUST_IDENTITY,
            license_expression="Apache-2.0",
            license_files=[REPOSITORY_ROOT / "LICENSE"],
        )


def test_platform_wheel_rejects_a_generic_linux_tag(tmp_path, monkeypatch):
    artifact_path = _write_artifact(tmp_path / "sample.duckdb_extension")

    def create_descriptor(path, *, name, trust_identity, dependencies):
        return _descriptor(
            path,
            name=name,
            trust_identity=trust_identity,
            platform="linux_amd64",
            dependencies=dependencies,
        )

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_descriptor)
    with pytest.raises(ValueError, match="does not match extension artifact platform"):
        build_extension_wheel(
            artifact=artifact_path,
            extension_name="sample",
            output_directory=tmp_path / "dist",
            platform_tag="linux_x86_64",
            trust_identity=TEST_TRUST_IDENTITY,
            license_expression="Apache-2.0 AND MIT",
            license_files=[REPOSITORY_ROOT / "LICENSE"],
        )


def test_platform_wheel_rejects_an_invalid_license_expression(tmp_path):
    artifact_path = _write_artifact(tmp_path / "sample.duckdb_extension")

    with pytest.raises(ValueError, match="license_expression"):
        build_extension_wheel(
            artifact=artifact_path,
            extension_name="sample",
            output_directory=tmp_path / "dist",
            platform_tag="linux_x86_64",
            trust_identity=TEST_TRUST_IDENTITY,
            license_expression="not-an-spdx-expression",
            license_files=[REPOSITORY_ROOT / "LICENSE"],
        )


def test_platform_wheel_rejects_an_unsafe_license_file_path(tmp_path, synthetic_descriptor_factory):
    artifact_path = _write_artifact(tmp_path / "sample.duckdb_extension")
    license_path = tmp_path / "license\ninjected.txt"
    license_path.write_text("license", encoding="utf-8")

    with pytest.raises(ValueError, match="safe ASCII relative path"):
        build_extension_wheel(
            artifact=artifact_path,
            extension_name="sample",
            output_directory=tmp_path / "dist",
            platform_tag=_wheel_platform_tag(),
            trust_identity=TEST_TRUST_IDENTITY,
            license_expression="Apache-2.0 AND MIT",
            license_files=[license_path],
        )


@pytest.mark.parametrize("member_name", [" leading-space.txt", "trailing-space.txt "])
def test_platform_wheel_rejects_license_paths_ambiguous_in_metadata(member_name):
    with pytest.raises(ValueError, match="safe ASCII relative path"):
        extension_wheel_module._license_member_path(Path(member_name))


@pytest.mark.parametrize(
    "member_name",
    ["NUL.txt", "license.", "license:stream", 'license"quoted.txt'],
)
def test_platform_wheel_rejects_windows_unsafe_license_paths(member_name):
    assert extension_wheel_module._windows_path_part_is_unsafe(member_name)


def test_platform_wheel_rejects_case_insensitive_license_path_collisions(
    tmp_path,
    synthetic_descriptor_factory,
):
    artifact_path = _write_artifact(tmp_path / "sample.duckdb_extension")
    first_license = tmp_path / "licenses" / "Foo.txt"
    second_license = tmp_path / "licenses" / "foo.txt"
    first_license.parent.mkdir()
    first_license.write_text("first", encoding="utf-8")
    second_license.write_text("second", encoding="utf-8")

    with pytest.raises(ValueError, match="case-insensitive wheel path collisions"):
        build_extension_wheel(
            artifact=artifact_path,
            extension_name="sample",
            output_directory=tmp_path / "dist",
            platform_tag=_wheel_platform_tag(),
            trust_identity=TEST_TRUST_IDENTITY,
            license_expression="Apache-2.0",
            license_files=[first_license, second_license],
        )


@pytest.mark.parametrize("reverse", [False, True])
def test_platform_wheel_rejects_license_file_parent_path_conflicts(
    tmp_path,
    synthetic_descriptor_factory,
    reverse,
):
    artifact_path = _write_artifact(tmp_path / "sample.duckdb_extension")
    parent_license = tmp_path / "licenses"
    parent_license.write_text("parent license", encoding="utf-8")
    license_files = [parent_license, REPOSITORY_ROOT / "LICENSES" / "DuckDB-MIT.txt"]
    if reverse:
        license_files.reverse()

    with pytest.raises(ValueError, match="file/parent wheel path conflicts"):
        build_extension_wheel(
            artifact=artifact_path,
            extension_name="sample",
            output_directory=tmp_path / "dist",
            platform_tag=_wheel_platform_tag(),
            trust_identity=TEST_TRUST_IDENTITY,
            license_expression="Apache-2.0 AND MIT",
            license_files=license_files,
        )


def test_extension_wheel_cli_scripts_import_from_outside_the_repository(tmp_path):
    for script_name in ("build_extension_wheel.py", "verify_extension_wheel.py"):
        completed = subprocess.run(
            [sys.executable, "-I", str(REPOSITORY_ROOT / "scripts" / script_name), "--help"],
            check=True,
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

        assert "usage:" in completed.stdout


def test_platform_wheel_is_deterministic_regardless_of_license_argument_order(
    tmp_path,
    synthetic_descriptor_factory,
):
    artifact_path = _write_artifact(tmp_path / "sample.duckdb_extension")
    license_files = [REPOSITORY_ROOT / "LICENSE", REPOSITORY_ROOT / "LICENSES" / "DuckDB-MIT.txt"]
    first = build_extension_wheel(
        artifact=artifact_path,
        extension_name="sample",
        output_directory=tmp_path / "first",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0 AND MIT",
        license_files=license_files,
    )
    second = build_extension_wheel(
        artifact=artifact_path,
        extension_name="sample",
        output_directory=tmp_path / "second",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0 AND MIT",
        license_files=list(reversed(license_files)),
    )

    assert first.path.read_bytes() == second.path.read_bytes()


def test_extension_distribution_version_binds_the_complete_descriptor(tmp_path):
    artifact_path = _write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency")
    descriptor = _descriptor(artifact_path, name="dependency")
    different_extension_version = replace(descriptor, extension_version="different-version")
    different_artifact = replace(descriptor, sha256="0" * 64)

    versions = {
        extension_wheel_module._extension_distribution_version(vane.__version__, candidate)
        for candidate in (descriptor, different_extension_version, different_artifact)
    }

    assert len(versions) == 3
    for candidate in (descriptor, different_extension_version, different_artifact):
        _assert_public_descriptor_version(
            extension_wheel_module._extension_distribution_version(vane.__version__, candidate),
            vane.__version__,
            candidate,
        )


@pytest.mark.parametrize(
    "vane_version",
    [
        "1.2.3",
        "1.2.3.dev4",
        "1.2.3rc1",
        "1.2.3.post5",
        "1!2.3rc1.post5.dev6",
        "1.2.3+private.7",
    ],
)
def test_extension_distribution_version_is_public_for_every_vane_version(tmp_path, vane_version):
    artifact_path = _write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency")
    descriptor = _descriptor(artifact_path, name="dependency", vane_version=vane_version)
    generated = extension_wheel_module._extension_distribution_version(vane_version, descriptor)

    _assert_public_descriptor_version(generated, vane_version, descriptor)


def test_extension_distribution_version_rejects_more_than_eight_release_components(tmp_path):
    artifact_path = _write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency")
    vane_version = "1.2.3.4.5.6.7.8.9"
    descriptor = _descriptor(artifact_path, name="dependency", vane_version=vane_version)

    with pytest.raises(RuntimeError, match="more than 8 effective release components"):
        extension_wheel_module._extension_distribution_version(vane_version, descriptor)


def test_extension_distribution_version_preserves_vane_version_order(tmp_path):
    artifact_path = _write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency")
    vane_versions = (
        "1.2.3.dev3",
        "1.2.3.dev4",
        "1.2.3a0.dev1",
        "1.2.3a0",
        "1.2.3a0.post0.dev1",
        "1.2.3a0.post0",
        "1.2.3a1.dev1",
        "1.2.3a1",
        "1.2.3rc1",
        "1.2.3",
        "1.2.3.post0.dev1",
        "1.2.3.post0",
        "1.2.3.post1.dev1",
        "1.2.3.post1",
        "1.2.3.0.1.dev0",
        "1.2.3.0.1",
        "1.2.3.1.dev0",
        "1.2.4.dev0",
    )
    extension_versions = tuple(
        Version(
            extension_wheel_module._extension_distribution_version(
                vane_version,
                _descriptor(artifact_path, name="dependency", vane_version=vane_version),
            )
        )
        for vane_version in vane_versions
    )

    assert tuple(sorted(Version(version) for version in vane_versions)) == tuple(
        Version(version) for version in vane_versions
    )
    assert tuple(sorted(extension_versions)) == extension_versions


def test_platform_wheel_escapes_license_paths_in_record_csv(tmp_path, synthetic_descriptor_factory):
    artifact_path = _write_artifact(tmp_path / "sample.duckdb_extension")
    license_path = tmp_path / 'license,"quoted".txt'
    license_path.write_text("license", encoding="utf-8")
    built = build_extension_wheel(
        artifact=artifact_path,
        extension_name="sample",
        output_directory=tmp_path / "dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[license_path],
    )

    with zipfile.ZipFile(built.path) as wheel:
        distribution_root = built.distribution_name.replace("-", "_")
        record_name = f"{distribution_root}-{built.distribution_version}.dist-info/RECORD"
        record_rows = list(csv.reader(io.StringIO(wheel.read(record_name).decode("utf-8"))))

    assert any(row[0].endswith('licenses/license,"quoted".txt') for row in record_rows)
    assert all(len(row) == 3 for row in record_rows)


def test_platform_wheel_rejects_a_tag_for_a_different_artifact_platform(
    tmp_path,
    synthetic_descriptor_factory,
):
    artifact_path = _write_artifact(tmp_path / "sample.duckdb_extension")
    wrong_platform_tag = "win_amd64" if _runtime_platform() != "windows_amd64" else "linux_x86_64"

    with pytest.raises(ValueError, match="does not match extension artifact platform"):
        build_extension_wheel(
            artifact=artifact_path,
            extension_name="sample",
            output_directory=tmp_path / "dist",
            platform_tag=wrong_platform_tag,
            trust_identity=TEST_TRUST_IDENTITY,
            license_expression="Apache-2.0 AND MIT",
            license_files=[REPOSITORY_ROOT / "LICENSE"],
        )


def test_windows_pe_platform_accepts_policy_imports_and_delay_imports():
    interpreter_tag = _extension_interpreter_tag()
    python_library = f"python{interpreter_tag.removeprefix('cp')}.dll"
    contents = _synthetic_pe(
        architecture="amd64",
        imports=("KERNEL32.DLL", python_library),
        delay_imports=("api-ms-win-crt-runtime-l1-1-0.dll", "VCRUNTIME140.dll"),
    )

    assert extension_wheel_module._validate_windows_pe_platform(
        contents,
        "win_amd64",
        description="test PE binary",
        interpreter_tag=interpreter_tag,
    ) == (
        "kernel32.dll",
        python_library,
        "api-ms-win-crt-runtime-l1-1-0.dll",
        "vcruntime140.dll",
    )


@pytest.mark.parametrize("import_kind", ["imports", "delay_imports"])
def test_windows_pe_platform_rejects_private_imports(import_kind):
    contents = _synthetic_pe(
        architecture="amd64",
        **{import_kind: ("publisher-private.dll",)},
    )

    with pytest.raises(ValueError, match="non-policy Windows DLLs.*publisher-private.dll"):
        extension_wheel_module._validate_windows_pe_platform(
            contents,
            "win_amd64",
            description="test PE binary",
            interpreter_tag=_extension_interpreter_tag(),
        )


def test_windows_pe_platform_rejects_export_forwarders():
    contents = _synthetic_pe(
        architecture="amd64",
        forwarded_exports=("publisher-private.forwarded_initializer",),
    )

    with pytest.raises(ValueError, match="PE export forwarder"):
        extension_wheel_module._validate_windows_pe_platform(
            contents,
            "win_amd64",
            description="test PE binary",
            interpreter_tag=_extension_interpreter_tag(),
        )


def test_windows_pe_platform_rejects_a_spoofed_api_set_import():
    library = "api-ms-win-publisher-private-l1-1-0.dll"
    contents = _synthetic_pe(architecture="amd64", imports=(library,))

    with pytest.raises(ValueError, match=rf"non-policy Windows DLLs.*{re.escape(library)}"):
        extension_wheel_module._validate_windows_pe_platform(
            contents,
            "win_amd64",
            description="test PE binary",
            interpreter_tag=_extension_interpreter_tag(),
        )


def test_windows_pe_platform_rejects_another_cpython_runtime():
    other_interpreter_tag = _other_extension_interpreter_tag()
    library = f"python{other_interpreter_tag.removeprefix('cp')}.dll"
    contents = _synthetic_pe(architecture="amd64", imports=(library,))

    with pytest.raises(ValueError, match=rf"non-policy Windows DLLs.*{re.escape(library)}"):
        extension_wheel_module._validate_windows_pe_platform(
            contents,
            "win_amd64",
            description="test PE binary",
            interpreter_tag=_extension_interpreter_tag(),
        )


def test_windows_pe_platform_rejects_another_machine():
    contents = _synthetic_pe(architecture="arm64", imports=("kernel32.dll",))

    with pytest.raises(ValueError, match="PE machine 0xaa64.*amd64"):
        extension_wheel_module._validate_windows_pe_platform(
            contents,
            "win_amd64",
            description="test PE binary",
            interpreter_tag=_extension_interpreter_tag(),
        )


def test_windows_pe_platform_requires_a_dll_file_type():
    contents = bytearray(_synthetic_pe(architecture="amd64"))
    characteristics_offset = 64 + 4 + 18
    struct.pack_into("<H", contents, characteristics_offset, 0x22)

    with pytest.raises(ValueError, match="must use the PE executable DLL file type"):
        extension_wheel_module._validate_windows_pe_platform(
            bytes(contents),
            "win_amd64",
            description="test PE binary",
            interpreter_tag=_extension_interpreter_tag(),
        )


def test_windows_pe_platform_requires_a_file_backed_section():
    contents = bytearray(_synthetic_pe(architecture="amd64"))
    section_header_offset = 64 + 4 + 20 + 240
    struct.pack_into("<I", contents, section_header_offset + 16, 0)

    with pytest.raises(ValueError, match="must contain at least one file-backed PE section"):
        extension_wheel_module._validate_windows_pe_platform(
            bytes(contents),
            "win_amd64",
            description="test PE binary",
            interpreter_tag=_extension_interpreter_tag(),
        )


def test_windows_pe_platform_rejects_overlapping_virtual_section_mappings():
    contents = bytearray(_synthetic_pe(architecture="amd64"))
    coff_header_offset = 64 + 4
    section_header_offset = 64 + 4 + 20 + 240
    struct.pack_into("<H", contents, coff_header_offset + 2, 2)
    struct.pack_into("<I", contents, section_header_offset + 8, 0x1800)
    contents.extend(b"\0" * 0x200)
    struct.pack_into(
        "<8sIIIIIIHHI",
        contents,
        section_header_offset + 40,
        b".other\0\0",
        0x200,
        0x2000,
        0x200,
        0x400,
        0,
        0,
        0,
        0,
        0x40000040,
    )

    with pytest.raises(ValueError, match="overlapping virtual PE sections"):
        extension_wheel_module._validate_windows_pe_platform(
            bytes(contents),
            "win_amd64",
            description="test PE binary",
            interpreter_tag=_extension_interpreter_tag(),
        )


def test_windows_pe_platform_rejects_import_paths():
    contents = _synthetic_pe(architecture="amd64", imports=("C:\\private\\library.dll",))

    with pytest.raises(ValueError, match="invalid PE import-library name"):
        extension_wheel_module._validate_windows_pe_platform(
            contents,
            "win_amd64",
            description="test PE binary",
            interpreter_tag=_extension_interpreter_tag(),
        )


def test_windows_pe_platform_rejects_non_rva_delay_imports():
    contents = bytearray(_synthetic_pe(architecture="amd64", delay_imports=("kernel32.dll",)))
    raw_section_offset = 0x200
    struct.pack_into("<I", contents, raw_section_offset, 0)

    with pytest.raises(ValueError, match="non-RVA PE delay-import descriptor"):
        extension_wheel_module._validate_windows_pe_platform(
            bytes(contents),
            "win_amd64",
            description="test PE binary",
            interpreter_tag=_extension_interpreter_tag(),
        )


def test_windows_pe_platform_requires_a_terminated_import_directory():
    contents = bytearray(_synthetic_pe(architecture="amd64", imports=("kernel32.dll",)))
    raw_section_offset = 0x200
    name_rva = struct.unpack_from("<I", contents, raw_section_offset + 12)[0]
    struct.pack_into("<5I", contents, raw_section_offset + 20, 0, 0, 0, name_rva, 0)

    with pytest.raises(ValueError, match="import directory has no terminating descriptor"):
        extension_wheel_module._validate_windows_pe_platform(
            bytes(contents),
            "win_amd64",
            description="test PE binary",
            interpreter_tag=_extension_interpreter_tag(),
        )


def test_windows_pe_platform_rejects_an_unmapped_import_directory():
    contents = bytearray(_synthetic_pe(architecture="amd64", imports=("kernel32.dll",)))
    optional_header_offset = 64 + 4 + 20
    struct.pack_into("<I", contents, optional_header_offset + 112 + 8, 0xFFFF0000)

    with pytest.raises(ValueError, match="import directory has no unique file-backed RVA mapping"):
        extension_wheel_module._validate_windows_pe_platform(
            bytes(contents),
            "win_amd64",
            description="test PE binary",
            interpreter_tag=_extension_interpreter_tag(),
        )


def test_platform_wheel_rejects_a_private_windows_import(tmp_path, monkeypatch):
    artifact = tmp_path / "sample.duckdb_extension"
    artifact.write_bytes(_synthetic_pe(architecture="amd64", imports=("publisher-private.dll",)))

    def create_descriptor(path, *, name, trust_identity, dependencies):
        return _descriptor(
            path,
            name=name,
            trust_identity=trust_identity,
            platform="windows_amd64",
            dependencies=dependencies,
        )

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_descriptor)

    with pytest.raises(ValueError, match="non-policy Windows DLLs.*publisher-private.dll"):
        _build_sample_wheel(
            tmp_path,
            artifact_path=artifact,
            platform_tag="win_amd64",
        )


def test_platform_wheel_rejects_a_windows_export_forwarder(tmp_path, monkeypatch):
    artifact = tmp_path / "sample.duckdb_extension"
    artifact.write_bytes(
        _synthetic_pe(
            architecture="amd64",
            forwarded_exports=("publisher-private.forwarded_initializer",),
        )
    )

    def create_descriptor(path, *, name, trust_identity, dependencies):
        return _descriptor(
            path,
            name=name,
            trust_identity=trust_identity,
            platform="windows_amd64",
            dependencies=dependencies,
        )

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_descriptor)

    with pytest.raises(ValueError, match="PE export forwarder"):
        _build_sample_wheel(
            tmp_path,
            artifact_path=artifact,
            platform_tag="win_amd64",
        )


@pytest.mark.parametrize(
    ("artifact_platform", "platform_tag"),
    [
        ("osx_arm64", "macosx_10_15_arm64"),
        ("osx_arm64", "macosx_14_5_arm64"),
        ("osx_amd64", "macosx_14_5_x86_64"),
        ("osx_amd64", "macosx_10_3_x86_64"),
        ("osx_amd64", "macosx_10_17_x86_64"),
    ],
)
def test_platform_wheel_rejects_noncanonical_macos_platform_tags(artifact_platform, platform_tag):
    with pytest.raises(ValueError, match="not a canonical architecture-specific tag"):
        extension_wheel_module._validate_artifact_platform_tag(artifact_platform, platform_tag)


@pytest.mark.parametrize(
    ("artifact_platform", "platform_tag"),
    [
        ("osx_arm64", "macosx_11_0_arm64"),
        ("osx_arm64", "macosx_14_0_arm64"),
        ("osx_amd64", "macosx_10_4_x86_64"),
        ("osx_amd64", "macosx_10_16_x86_64"),
        ("osx_amd64", "macosx_11_0_x86_64"),
        ("osx_amd64", "macosx_14_0_x86_64"),
    ],
)
def test_platform_wheel_accepts_canonical_macos_platform_tags(artifact_platform, platform_tag):
    extension_wheel_module._validate_artifact_platform_tag(artifact_platform, platform_tag)


def test_platform_wheel_rejects_a_macos_tag_below_the_binary_deployment_target(tmp_path, monkeypatch):
    artifact = tmp_path / "sample.duckdb_extension"
    artifact.write_bytes(_synthetic_macho(architecture="arm64", minimum_macos=(14, 0, 0)))

    def create_descriptor(path, *, name, trust_identity, dependencies):
        return _descriptor(
            path,
            name=name,
            trust_identity=trust_identity,
            platform="osx_arm64",
            dependencies=dependencies,
        )

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_descriptor)

    with pytest.raises(ValueError, match="requires macOS 14.0.0.*macosx_11_0_arm64"):
        _build_sample_wheel(
            tmp_path,
            artifact_path=artifact,
            platform_tag="macosx_11_0_arm64",
        )


def test_platform_wheel_rejects_a_non_system_macos_dependency(tmp_path, monkeypatch):
    artifact = tmp_path / "sample.duckdb_extension"
    artifact.write_bytes(
        _synthetic_macho(
            architecture="arm64",
            minimum_macos=(14, 0, 0),
            dynamic_libraries=("/usr/local/lib/libprivate.dylib",),
        )
    )

    def create_descriptor(path, *, name, trust_identity, dependencies):
        return _descriptor(
            path,
            name=name,
            trust_identity=trust_identity,
            platform="osx_arm64",
            dependencies=dependencies,
        )

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_descriptor)

    with pytest.raises(ValueError, match="non-system Mach-O dynamic libraries.*libprivate.dylib"):
        _build_sample_wheel(
            tmp_path,
            artifact_path=artifact,
            platform_tag="macosx_14_0_arm64",
        )


@pytest.mark.parametrize(
    ("architecture", "platform_tag", "cpu_subtype"),
    [
        ("x86_64", "macosx_11_0_x86_64", 8),
        ("arm64", "macosx_11_0_arm64", 2),
    ],
)
def test_macos_platform_validation_rejects_specialized_cpu_subtypes(
    architecture,
    platform_tag,
    cpu_subtype,
):
    contents = _synthetic_macho(
        architecture=architecture,
        minimum_macos=(11, 0, 0),
        cpu_subtype=cpu_subtype,
    )

    with pytest.raises(ValueError, match="CPU subtype.*not compatible with generic"):
        extension_wheel_module._validate_macos_binary_platform(
            contents,
            platform_tag,
            description="test Mach-O binary",
        )


def test_macos_platform_validation_rejects_non_macos_deployment_commands():
    contents = bytearray(
        _synthetic_macho(
            architecture="arm64",
            minimum_macos=(11, 0, 0),
        )
    )
    contents[32:36] = (0x25).to_bytes(4, "little")

    with pytest.raises(ValueError, match="non-macOS deployment-target load command"):
        extension_wheel_module._validate_macos_binary_platform(
            bytes(contents),
            "macosx_11_0_arm64",
            description="test Mach-O binary",
        )


def test_macos_platform_validation_accepts_system_dependencies_and_relocatable_rpaths():
    contents = _synthetic_macho(
        architecture="arm64",
        minimum_macos=(11, 0, 0),
        dynamic_libraries=(
            "/usr/lib/libSystem.B.dylib",
            "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation",
        ),
        rpaths=("@loader_path", "@loader_path/.dylibs", "@executable_path/Frameworks"),
    )

    assert extension_wheel_module._validate_macos_binary_platform(
        contents,
        "macosx_11_0_arm64",
        description="test Mach-O binary",
    ) == (11, 0, 0)


@pytest.mark.parametrize(
    "command",
    [0xC, 0x80000018, 0x8000001F, 0x20, 0x80000023],
)
def test_macos_platform_validation_rejects_non_system_dynamic_library_commands(command):
    contents = _synthetic_macho(
        architecture="arm64",
        minimum_macos=(11, 0, 0),
        dynamic_libraries=("/usr/local/lib/libprivate.dylib",),
        dynamic_library_command=command,
    )

    with pytest.raises(ValueError, match="non-system Mach-O dynamic libraries.*libprivate.dylib"):
        extension_wheel_module._validate_macos_binary_platform(
            contents,
            "macosx_11_0_arm64",
            description="test Mach-O binary",
        )


@pytest.mark.parametrize(
    "install_name",
    [
        "/Library/Frameworks/Private.framework/Private",
        "/System/Volumes/Data/usr/local/lib/libprivate.dylib",
        "@rpath/libprivate.dylib",
        "@loader_path/libprivate.dylib",
        "relative/libprivate.dylib",
    ],
)
def test_macos_platform_validation_rejects_non_system_dynamic_library_paths(install_name):
    contents = _synthetic_macho(
        architecture="arm64",
        minimum_macos=(11, 0, 0),
        dynamic_libraries=(install_name,),
    )

    with pytest.raises(ValueError, match="non-system Mach-O dynamic libraries"):
        extension_wheel_module._validate_macos_binary_platform(
            contents,
            "macosx_11_0_arm64",
            description="test Mach-O binary",
        )


@pytest.mark.parametrize("command", [0x6, 0x7, 0x9, 0xE, 0xF, 0x10, 0x27])
def test_macos_platform_validation_rejects_legacy_dynamic_linker_commands(command):
    contents = _synthetic_macho(
        architecture="arm64",
        minimum_macos=(11, 0, 0),
        dynamic_libraries=("/usr/lib/libSystem.B.dylib",),
        dynamic_library_command=command,
    )

    with pytest.raises(ValueError, match=f"unsupported Mach-O dynamic-linker command 0x{command:x}"):
        extension_wheel_module._validate_macos_binary_platform(
            contents,
            "macosx_11_0_arm64",
            description="test Mach-O binary",
        )


@pytest.mark.parametrize(
    "rpath",
    ["/usr/local/lib", "@rpath", "relative/lib", "@loader_path/../lib", "@loader_path//lib"],
)
def test_macos_platform_validation_rejects_non_relocatable_runtime_search_paths(rpath):
    contents = _synthetic_macho(
        architecture="arm64",
        minimum_macos=(11, 0, 0),
        rpaths=(rpath,),
    )

    with pytest.raises(ValueError, match="non-relocatable Mach-O runtime search paths"):
        extension_wheel_module._validate_macos_binary_platform(
            contents,
            "macosx_11_0_arm64",
            description="test Mach-O binary",
        )


def test_macos_platform_validation_rejects_an_invalid_dynamic_library_name_offset():
    contents = bytearray(
        _synthetic_macho(
            architecture="arm64",
            minimum_macos=(11, 0, 0),
            dynamic_libraries=("/usr/lib/libSystem.B.dylib",),
        )
    )
    dylib_command_offset = 32 + 24
    struct.pack_into("<I", contents, dylib_command_offset + 8, 8)

    with pytest.raises(ValueError, match="invalid Mach-O dynamic-library install name offset"):
        extension_wheel_module._validate_macos_binary_platform(
            bytes(contents),
            "macosx_11_0_arm64",
            description="test Mach-O binary",
        )


def test_macos_platform_validation_rejects_an_unterminated_dynamic_library_name():
    contents = bytearray(
        _synthetic_macho(
            architecture="arm64",
            minimum_macos=(11, 0, 0),
            dynamic_libraries=("/usr/lib/libSystem.B.dylib",),
        )
    )
    dylib_command_offset = 32 + 24
    command_size = struct.unpack_from("<I", contents, dylib_command_offset + 4)[0]
    name_offset = struct.unpack_from("<I", contents, dylib_command_offset + 8)[0]
    string_start = dylib_command_offset + name_offset
    command_end = dylib_command_offset + command_size
    contents[string_start:command_end] = b"A" * (command_end - string_start)

    with pytest.raises(ValueError, match="unterminated or oversized Mach-O dynamic-library install name"):
        extension_wheel_module._validate_macos_binary_platform(
            bytes(contents),
            "macosx_11_0_arm64",
            description="test Mach-O binary",
        )


def test_clean_verifier_rejects_a_relabelled_macos_deployment_target(tmp_path, monkeypatch):
    artifact = tmp_path / "sample.duckdb_extension"
    artifact.write_bytes(_synthetic_macho(architecture="arm64", minimum_macos=(14, 0, 0)))

    def create_descriptor(path, *, name, trust_identity, dependencies):
        return _descriptor(
            path,
            name=name,
            trust_identity=trust_identity,
            platform="osx_arm64",
            dependencies=dependencies,
        )

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_descriptor)
    built = _build_sample_wheel(
        tmp_path,
        artifact_path=artifact,
        platform_tag="macosx_14_0_arm64",
    )
    relabelled = _relabel_wheel_platform(
        built.path,
        tmp_path / "relabelled-macos",
        original="macosx_14_0_arm64",
        replacement="macosx_11_0_arm64",
    )

    with pytest.raises(RuntimeError, match="requires macOS 14.0.0.*macosx_11_0_arm64"):
        _extension_name_from_wheel(relabelled)


def test_platform_wheel_requires_the_exact_build_runtime_musl_floor(tmp_path, monkeypatch):
    artifact = _write_artifact(tmp_path / "sample.duckdb_extension", architecture="x86_64")

    def create_descriptor(path, *, name, trust_identity, dependencies):
        return _descriptor(
            path,
            name=name,
            trust_identity=trust_identity,
            platform="linux_amd64_musl",
            dependencies=dependencies,
        )

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_descriptor)
    monkeypatch.setattr(extension_wheel_module, "_current_musl_version", lambda: (1, 2))

    with pytest.raises(ValueError, match="must match the build runtime's musl 1.2 baseline exactly"):
        _build_sample_wheel(tmp_path, artifact_path=artifact, platform_tag="musllinux_1_1_x86_64")


def test_clean_verifier_rejects_a_relabelled_musl_floor(tmp_path, monkeypatch):
    artifact = _write_artifact(tmp_path / "sample.duckdb_extension", architecture="x86_64")

    def create_descriptor(path, *, name, trust_identity, dependencies):
        return _descriptor(
            path,
            name=name,
            trust_identity=trust_identity,
            platform="linux_amd64_musl",
            dependencies=dependencies,
        )

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_descriptor)
    monkeypatch.setattr(extension_wheel_module, "_current_musl_version", lambda: (1, 2))
    built = _build_sample_wheel(
        tmp_path,
        artifact_path=artifact,
        platform_tag="musllinux_1_2_x86_64",
    )
    relabelled = _relabel_wheel_platform(
        built.path,
        tmp_path / "relabelled-musl",
        original="musllinux_1_2_x86_64",
        replacement="musllinux_1_1_x86_64",
    )

    with pytest.raises(RuntimeError, match="musl build baseline.*does not match platform tag"):
        _extension_name_from_wheel(relabelled)


def test_clean_verifier_requires_the_exact_tagged_musl_runtime(tmp_path, monkeypatch):
    artifact = _write_artifact(tmp_path / "sample.duckdb_extension", architecture="x86_64")

    def create_descriptor(path, *, name, trust_identity, dependencies):
        return _descriptor(
            path,
            name=name,
            trust_identity=trust_identity,
            platform="linux_amd64_musl",
            dependencies=dependencies,
        )

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_descriptor)
    monkeypatch.setattr(extension_wheel_module, "_current_musl_version", lambda: (1, 2))
    built = _build_sample_wheel(
        tmp_path,
        artifact_path=artifact,
        platform_tag="musllinux_1_2_x86_64",
    )
    layout = verify_extension_wheel_module._assert_extension_wheel_layout(built.path, "sample")
    monkeypatch.setattr(verify_extension_wheel_module, "_current_musl_version", lambda: (1, 3))

    with pytest.raises(RuntimeError, match="must run on its exact minimum musl runtime, not musl 1.3"):
        verify_extension_wheel_module._assert_musl_verification_runtime(layout)


@pytest.mark.parametrize(
    ("artifact_platform", "platform_tag"),
    [
        ("linux_amd64", "manylinux_02_28_x86_64"),
        ("linux_amd64_musl", "musllinux_01_02_x86_64"),
        ("osx_arm64", "macosx_011_0_arm64"),
        ("osx_amd64", "macosx_10_04_x86_64"),
    ],
)
def test_platform_wheel_rejects_noncanonical_decimal_policy_versions(artifact_platform, platform_tag):
    with pytest.raises(ValueError, match="must use canonical decimal spelling"):
        extension_wheel_module._validate_artifact_platform_tag(artifact_platform, platform_tag)


@pytest.mark.parametrize(
    ("artifact_platform", "platform_tag"),
    [
        ("linux_amd64", "manylinux_2_4_x86_64"),
        ("linux_arm64", "manylinux_2_16_aarch64"),
    ],
)
def test_platform_wheel_rejects_manylinux_tags_below_supported_architecture_floors(
    artifact_platform,
    platform_tag,
):
    with pytest.raises(ValueError, match="below the supported policy floor"):
        extension_wheel_module._validate_artifact_platform_tag(artifact_platform, platform_tag)


@pytest.mark.parametrize(
    ("artifact_platform", "platform_tag"),
    [
        ("linux_amd64", "manylinux_2_5_x86_64"),
        ("linux_arm64", "manylinux_2_17_aarch64"),
    ],
)
def test_platform_wheel_accepts_manylinux_tags_at_supported_architecture_floors(
    artifact_platform,
    platform_tag,
):
    extension_wheel_module._validate_artifact_platform_tag(artifact_platform, platform_tag)


@pytest.mark.parametrize(
    ("artifact_platform", "platform_tag"),
    [
        ("linux_amd64", "manylinux_2_16_x86_64"),
        ("linux_amd64", "manylinux_2_42_x86_64"),
        ("linux_arm64", "manylinux_2_18_aarch64"),
    ],
)
def test_platform_wheel_rejects_manylinux_tags_without_a_pinned_auditwheel_policy(
    artifact_platform,
    platform_tag,
):
    with pytest.raises(ValueError, match="not present in the pinned auditwheel 6.8.1 policy"):
        extension_wheel_module._validate_artifact_platform_tag(artifact_platform, platform_tag)


def test_manylinux_policy_snapshot_matches_every_pinned_auditwheel_policy():
    policy_path = manylinux_policy_module.AUDITWHEEL_MANYLINUX_POLICY_PATH
    contents = policy_path.read_bytes()
    assert hashlib.sha256(contents).hexdigest() == manylinux_policy_module.AUDITWHEEL_MANYLINUX_POLICY_SHA256
    upstream_policies = json.loads(contents)

    expected: dict[tuple[str, tuple[int, int]], dict[str, object]] = {}
    for raw_policy in upstream_policies:
        if raw_policy["name"] == "linux":
            continue
        _family, major, minor = raw_policy["name"].split("_")
        minimum_version = (int(major), int(minor))
        for architecture in ("x86_64", "aarch64"):
            if architecture not in raw_policy["symbol_versions"]:
                continue
            exact_symbols = frozenset(
                f"{namespace}_{version}"
                for namespace, versions in raw_policy["symbol_versions"][architecture].items()
                for version in versions
            )
            expected[(architecture, minimum_version)] = {
                "external_libraries": frozenset(raw_policy["lib_whitelist"]),
                "versioned_symbols": exact_symbols,
                "undefined_symbol_blacklist": tuple(
                    sorted((library, frozenset(symbols)) for library, symbols in raw_policy["blacklist"].items())
                ),
            }

    assert len(expected) == 30
    assert manylinux_policy_module.manylinux_policy_combinations() == tuple(
        sorted(expected, key=lambda item: (item[0], item[1]))
    )
    for (architecture, minimum_version), expected_policy in expected.items():
        actual_policy = manylinux_policy_module.manylinux_policy(minimum_version, architecture)
        assert actual_policy.external_libraries == expected_policy["external_libraries"]
        assert actual_policy.versioned_symbols == expected_policy["versioned_symbols"]
        assert actual_policy.undefined_symbol_blacklist == expected_policy["undefined_symbol_blacklist"]


def test_platform_wheel_rejects_an_artifact_built_for_newer_glibc(tmp_path, monkeypatch):
    artifact = _write_artifact(
        tmp_path / "sample.duckdb_extension",
        architecture="x86_64",
        glibc_version="2.39",
    )

    def create_descriptor(path, *, name, trust_identity, dependencies):
        return _descriptor(
            path,
            name=name,
            trust_identity=trust_identity,
            platform="linux_amd64",
            dependencies=dependencies,
        )

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_descriptor)

    with pytest.raises(ValueError, match="requires glibc 2.39.*manylinux_2_28_x86_64"):
        _build_sample_wheel(
            tmp_path,
            artifact_path=artifact,
            platform_tag="manylinux_2_28_x86_64",
        )


def test_platform_wheel_rejects_an_artifact_for_another_elf_machine(tmp_path, monkeypatch):
    artifact = _write_artifact(
        tmp_path / "sample.duckdb_extension",
        architecture="aarch64",
    )

    def create_descriptor(path, *, name, trust_identity, dependencies):
        return _descriptor(
            path,
            name=name,
            trust_identity=trust_identity,
            platform="linux_amd64",
            dependencies=dependencies,
        )

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_descriptor)

    with pytest.raises(ValueError, match="ELF machine 183.*x86_64"):
        _build_sample_wheel(
            tmp_path,
            artifact_path=artifact,
            platform_tag="manylinux_2_28_x86_64",
        )


@pytest.mark.parametrize(
    ("artifact_contents", "expected_message"),
    [
        (
            _synthetic_elf(architecture="x86_64")[:16]
            + (2).to_bytes(2, "little")
            + _synthetic_elf(architecture="x86_64")[18:],
            "ELF shared-object file type",
        ),
        (
            _synthetic_elf(architecture="x86_64", glibc_version="12345.1"),
            "invalid glibc version requirement",
        ),
    ],
)
def test_linux_platform_policy_rejects_invalid_elf_metadata(artifact_contents, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        extension_wheel_module._validate_linux_elf_platform(
            artifact_contents,
            "manylinux_2_39_x86_64",
            description="test artifact",
        )


def test_linux_platform_policy_tracks_only_the_highest_glibc_requirement():
    contents = _synthetic_elf(
        architecture="x86_64",
        versioned_symbols={"libc.so.6": ("GLIBC_2.17", "GLIBC_2.39", "GLIBC_2.28")},
    )

    assert extension_wheel_module._validate_linux_elf_platform(
        contents,
        "manylinux_2_39_x86_64",
        description="test artifact",
    ) == (2, 39)


@pytest.mark.parametrize(
    ("architecture", "platform_tag", "glibc_version"),
    [
        ("x86_64", "manylinux_2_17_x86_64", "2.17.999"),
        ("x86_64", "manylinux_2_5_x86_64", "2.1"),
    ],
)
def test_linux_platform_policy_requires_exact_glibc_symbol_membership(
    architecture,
    platform_tag,
    glibc_version,
):
    contents = _synthetic_elf(
        architecture=architecture,
        glibc_version=glibc_version,
    )

    with pytest.raises(ValueError, match=rf"glibc {re.escape(glibc_version)}.*exact platform policy"):
        extension_wheel_module._validate_linux_elf_platform(
            contents,
            platform_tag,
            description="test artifact",
        )


def test_linux_platform_policy_computes_the_glibc_floor_separately_from_exact_membership():
    contents = _synthetic_elf(
        architecture="aarch64",
        glibc_version="2.18",
    )

    assert extension_wheel_module._validate_linux_elf_platform(
        contents,
        "manylinux_2_17_aarch64",
        description="test artifact",
    ) == (2, 18)


@pytest.mark.parametrize("hash_style", ["sysv", "gnu"])
def test_linux_platform_policy_rejects_auditwheel_blacklisted_undefined_symbols(hash_style):
    contents = _synthetic_elf(
        architecture="x86_64",
        needed=("libz.so.1",),
        undefined_symbols=("_dist_code",),
        hash_style=hash_style,
    )

    with pytest.raises(ValueError, match=r"blacklisted undefined ELF symbols.*libz\.so\.1.*_dist_code"):
        extension_wheel_module._validate_linux_elf_platform(
            contents,
            "manylinux_2_17_x86_64",
            description="test artifact",
        )


def test_linux_platform_policy_ignores_weak_and_unprovided_blacklisted_symbols():
    weak_contents = _synthetic_elf(
        architecture="x86_64",
        needed=("libz.so.1",),
        weak_undefined_symbols=("_dist_code",),
    )
    assert (
        extension_wheel_module._validate_linux_elf_platform(
            weak_contents,
            "manylinux_2_17_x86_64",
            description="test artifact",
        )
        is None
    )

    unrelated_contents = _synthetic_elf(
        architecture="x86_64",
        undefined_symbols=("_dist_code",),
    )
    assert (
        extension_wheel_module._validate_linux_elf_platform(
            unrelated_contents,
            "manylinux_2_17_x86_64",
            description="test artifact",
        )
        is None
    )


def test_linux_platform_policy_allows_only_policy_shared_libraries():
    policy_contents = _synthetic_elf(
        architecture="x86_64",
        glibc_version="2.17",
        needed=("libstdc++.so.6", "libc.so.6", "ld-linux-x86-64.so.2"),
    )
    assert extension_wheel_module._validate_linux_elf_platform(
        policy_contents,
        "manylinux_2_17_x86_64",
        description="test artifact",
    ) == (2, 17)

    private_library_contents = _synthetic_elf(
        architecture="x86_64",
        needed=("libprivate.so.1",),
    )
    with pytest.raises(ValueError, match="non-policy ELF shared libraries.*libprivate.so.1"):
        extension_wheel_module._validate_linux_elf_platform(
            private_library_contents,
            "manylinux_2_17_x86_64",
            description="test artifact",
        )


@pytest.mark.parametrize("dependency_kind", ["filters", "auxiliaries"])
def test_linux_platform_policy_applies_to_filter_and_auxiliary_dependencies(dependency_kind):
    private_contents = _synthetic_elf(
        architecture="x86_64",
        **{dependency_kind: ("libprivate.so.1",)},
    )
    with pytest.raises(ValueError, match="non-policy ELF shared libraries.*libprivate.so.1"):
        extension_wheel_module._validate_linux_elf_platform(
            private_contents,
            "manylinux_2_17_x86_64",
            description="test artifact",
        )

    policy_contents = _synthetic_elf(
        architecture="x86_64",
        **{dependency_kind: ("libstdc++.so.6",)},
    )
    assert (
        extension_wheel_module._validate_linux_elf_platform(
            policy_contents,
            "manylinux_2_17_x86_64",
            description="test artifact",
        )
        is None
    )


def test_linux_platform_policy_rejects_duplicate_loader_dependencies_across_tags():
    contents = _synthetic_elf(
        architecture="x86_64",
        needed=("libstdc++.so.6",),
        filters=("libstdc++.so.6",),
    )

    with pytest.raises(ValueError, match="duplicate ELF loader dependencies"):
        extension_wheel_module._validate_linux_elf_platform(
            contents,
            "manylinux_2_17_x86_64",
            description="test artifact",
        )


@pytest.mark.parametrize(
    ("tag", "value", "expected_message"),
    [
        (15, None, "unsupported ELF loader configuration DT_RPATH"),
        (29, None, "unsupported ELF loader configuration DT_RUNPATH"),
        (0x6FFFFEFA, None, "unsupported ELF loader configuration DT_CONFIG"),
        (0x6FFFFEFB, None, "unsupported ELF loader configuration DT_DEPAUDIT"),
        (0x6FFFFEFC, None, "unsupported ELF loader configuration DT_AUDIT"),
        (0x6FFFFFFB, 0x800, "disables the default ELF shared-library search path"),
    ],
)
def test_linux_platform_policy_rejects_loader_configuration_that_changes_dependency_resolution(
    tag,
    value,
    expected_message,
):
    contents = bytearray(_synthetic_elf(architecture="x86_64"))
    dynamic_offset = 64 + 2 * 56
    for entry_offset in range(dynamic_offset, len(contents), 16):
        existing_tag, existing_value = struct.unpack_from("<qQ", contents, entry_offset)
        if existing_tag == 14:
            struct.pack_into("<qQ", contents, entry_offset, tag, existing_value if value is None else value)
            break
    else:
        raise AssertionError("synthetic ELF has no DT_SONAME entry")

    with pytest.raises(ValueError, match=expected_message):
        extension_wheel_module._validate_linux_elf_platform(
            bytes(contents),
            "manylinux_2_17_x86_64",
            description="test artifact",
        )


@pytest.mark.parametrize(
    ("library", "versioned_symbol"),
    [
        ("libstdc++.so.6", "GLIBCXX_3.4.32"),
        ("libstdc++.so.6", "CXXABI_1.3.15"),
        ("libgcc_s.so.1", "GCC_14.0.0"),
        ("libz.so.1", "ZLIB_1.2.12"),
    ],
)
def test_linux_platform_policy_validates_non_glibc_versioned_symbols(library, versioned_symbol):
    contents = _synthetic_elf(
        architecture="x86_64",
        versioned_symbols={library: (versioned_symbol,)},
    )

    with pytest.raises(ValueError, match=f"versioned ELF symbols.*{re.escape(versioned_symbol)}"):
        extension_wheel_module._validate_linux_elf_platform(
            contents,
            "manylinux_2_28_x86_64",
            description="test artifact",
        )
    assert (
        extension_wheel_module._validate_linux_elf_platform(
            contents,
            "manylinux_2_39_x86_64",
            description="test artifact",
        )
        is None
    )


@pytest.mark.parametrize("versioned_symbol", ["NEWABI_1.0", "UNNAMESPACED"])
def test_linux_platform_policy_rejects_unknown_versioned_symbols(versioned_symbol):
    contents = _synthetic_elf(
        architecture="x86_64",
        versioned_symbols={"libstdc++.so.6": (versioned_symbol,)},
    )

    with pytest.raises(ValueError, match=f"versioned ELF symbols.*{re.escape(versioned_symbol)}"):
        extension_wheel_module._validate_linux_elf_platform(
            contents,
            "manylinux_2_39_x86_64",
            description="test artifact",
        )


def test_linux_platform_policy_uses_the_declared_manylinux_library_allowlist():
    contents = _synthetic_elf(architecture="x86_64", needed=("libexpat.so.1",))

    with pytest.raises(ValueError, match="non-policy ELF shared libraries.*libexpat.so.1"):
        extension_wheel_module._validate_linux_elf_platform(
            contents,
            "manylinux_2_5_x86_64",
            description="test artifact",
        )
    assert (
        extension_wheel_module._validate_linux_elf_platform(
            contents,
            "manylinux_2_12_x86_64",
            description="test artifact",
        )
        is None
    )


def test_linux_platform_policy_rejects_an_out_of_bounds_needed_name():
    contents = bytearray(_synthetic_elf(architecture="x86_64", needed=("libprivate.so.1",)))
    dynamic_offset = 64 + 2 * 56
    struct.pack_into("<Q", contents, dynamic_offset + 8, 2**64 - 1)

    with pytest.raises(ValueError, match="out-of-bounds ELF dynamic string offset"):
        extension_wheel_module._validate_linux_elf_platform(
            bytes(contents),
            "manylinux_2_17_x86_64",
            description="test artifact",
        )


def test_linux_platform_policy_bounds_the_sysv_dynamic_symbol_count():
    contents = bytearray(
        _synthetic_elf(
            architecture="x86_64",
            undefined_symbols=("_dist_code",),
        )
    )
    dynamic_offset = 64 + 2 * 56
    for entry_offset in range(dynamic_offset, len(contents), 16):
        tag, value = struct.unpack_from("<qQ", contents, entry_offset)
        if tag == 4:
            struct.pack_into("<I", contents, value + 4, extension_wheel_module._MAX_ELF_DYNAMIC_SYMBOLS + 1)
            break
    else:
        raise AssertionError("synthetic ELF has no DT_HASH entry")

    with pytest.raises(ValueError, match="invalid or excessive ELF SysV hash dimensions"):
        extension_wheel_module._validate_linux_elf_platform(
            bytes(contents),
            "manylinux_2_17_x86_64",
            description="test artifact",
        )


def test_linux_platform_policy_bounds_the_gnu_dynamic_symbol_chain():
    contents = bytearray(
        _synthetic_elf(
            architecture="x86_64",
            undefined_symbols=("_dist_code",),
            hash_style="gnu",
        )
    )
    dynamic_offset = 64 + 2 * 56
    for entry_offset in range(dynamic_offset, len(contents), 16):
        tag, hash_table_offset = struct.unpack_from("<qQ", contents, entry_offset)
        if tag == 0x6FFFFEF5:
            bucket_count, symbol_offset, bloom_size, _bloom_shift = struct.unpack_from(
                "<IIII",
                contents,
                hash_table_offset,
            )
            buckets_offset = hash_table_offset + 16 + 8 * bloom_size
            last_symbol = max(
                struct.unpack_from("<I", contents, buckets_offset + index * 4)[0] for index in range(bucket_count)
            )
            chain_offset = buckets_offset + 4 * bucket_count + 4 * (last_symbol - symbol_offset)
            struct.pack_into("<I", contents, chain_offset, 0)
            break
    else:
        raise AssertionError("synthetic ELF has no DT_GNU_HASH entry")

    with pytest.raises(ValueError, match="unterminated or excessive ELF GNU hash chain"):
        extension_wheel_module._validate_linux_elf_platform(
            bytes(contents),
            "manylinux_2_17_x86_64",
            description="test artifact",
        )


def test_linux_platform_policy_bounds_the_version_needed_file_count():
    contents = bytearray(
        _synthetic_elf(
            architecture="x86_64",
            versioned_symbols={"libstdc++.so.6": ("GLIBCXX_3.4.19",)},
        )
    )
    dynamic_offset = 64 + 2 * 56
    for entry_offset in range(dynamic_offset, len(contents), 16):
        tag, _value = struct.unpack_from("<qQ", contents, entry_offset)
        if tag == 0x6FFFFFFF:
            struct.pack_into("<Q", contents, entry_offset + 8, 257)
            break
    else:
        raise AssertionError("synthetic ELF has no DT_VERNEEDNUM entry")

    with pytest.raises(ValueError, match="invalid ELF version-needed file count"):
        extension_wheel_module._validate_linux_elf_platform(
            bytes(contents),
            "manylinux_2_17_x86_64",
            description="test artifact",
        )


@pytest.mark.parametrize("dependency_kind", ["needed", "filters", "auxiliaries"])
def test_platform_wheel_rejects_an_unbundled_private_elf_dependency(tmp_path, monkeypatch, dependency_kind):
    artifact = _write_artifact(
        tmp_path / "sample.duckdb_extension",
        architecture="x86_64",
        **{dependency_kind: ("libprivate.so.1",)},
    )

    def create_descriptor(path, *, name, trust_identity, dependencies):
        return _descriptor(
            path,
            name=name,
            trust_identity=trust_identity,
            platform="linux_amd64",
            dependencies=dependencies,
        )

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_descriptor)

    with pytest.raises(ValueError, match="non-policy ELF shared libraries.*libprivate.so.1"):
        _build_sample_wheel(
            tmp_path,
            artifact_path=artifact,
            platform_tag="manylinux_2_17_x86_64",
        )


def test_platform_wheel_rejects_a_blacklisted_undefined_symbol(tmp_path, monkeypatch):
    artifact = _write_artifact(
        tmp_path / "sample.duckdb_extension",
        architecture="x86_64",
        needed=("libz.so.1",),
        undefined_symbols=("_dist_code",),
    )

    def create_descriptor(path, *, name, trust_identity, dependencies):
        return _descriptor(
            path,
            name=name,
            trust_identity=trust_identity,
            platform="linux_amd64",
            dependencies=dependencies,
        )

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_descriptor)

    with pytest.raises(ValueError, match=r"blacklisted undefined ELF symbols.*libz\.so\.1.*_dist_code"):
        _build_sample_wheel(
            tmp_path,
            artifact_path=artifact,
            platform_tag="manylinux_2_17_x86_64",
        )


def test_platform_wheel_revalidates_blacklisted_symbols_in_dependency_artifacts(tmp_path, monkeypatch):
    def create_descriptor(path, *, name, trust_identity, dependencies):
        return _descriptor(
            path,
            name=name,
            trust_identity=trust_identity,
            platform="linux_amd64",
            dependencies=dependencies,
        )

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_descriptor)
    dependency_artifact = _write_artifact(
        tmp_path / "dependency.duckdb_extension",
        b"dependency",
        architecture="x86_64",
        needed=("libz.so.1",),
        undefined_symbols=("_dist_code",),
    )
    with monkeypatch.context() as bypass:
        bypass.setattr(extension_wheel_module, "_validate_linux_elf_platform", lambda *_args, **_kwargs: None)
        dependency = build_extension_wheel(
            artifact=dependency_artifact,
            extension_name="dependency",
            output_directory=tmp_path / "dependency-dist",
            platform_tag="manylinux_2_17_x86_64",
            trust_identity=TEST_TRUST_IDENTITY,
            license_expression="Apache-2.0",
            license_files=[REPOSITORY_ROOT / "LICENSE"],
        )

    with pytest.raises(ValueError, match=r"blacklisted undefined ELF symbols.*libz\.so\.1.*_dist_code"):
        _build_sample_wheel(
            tmp_path,
            platform_tag="manylinux_2_17_x86_64",
            dependencies=(dependency.path,),
        )


def test_platform_wheel_rejects_a_newer_versioned_elf_symbol(tmp_path, monkeypatch):
    artifact = _write_artifact(
        tmp_path / "sample.duckdb_extension",
        architecture="x86_64",
        versioned_symbols={"libstdc++.so.6": ("GLIBCXX_3.4.32",)},
    )

    def create_descriptor(path, *, name, trust_identity, dependencies):
        return _descriptor(
            path,
            name=name,
            trust_identity=trust_identity,
            platform="linux_amd64",
            dependencies=dependencies,
        )

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_descriptor)

    with pytest.raises(ValueError, match="versioned ELF symbols.*GLIBCXX_3.4.32"):
        _build_sample_wheel(
            tmp_path,
            artifact_path=artifact,
            platform_tag="manylinux_2_28_x86_64",
        )


def test_platform_wheel_rejects_a_dependency_with_a_narrower_platform_policy(tmp_path, monkeypatch):
    def create_descriptor(path, *, name, trust_identity, dependencies):
        return _descriptor(
            path,
            name=name,
            trust_identity=trust_identity,
            platform="linux_amd64",
            dependencies=dependencies,
        )

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_descriptor)
    dependency = build_extension_wheel(
        artifact=_write_artifact(
            tmp_path / "dependency.duckdb_extension",
            b"dependency",
            architecture="x86_64",
        ),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag="manylinux_2_39_x86_64",
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )

    with pytest.raises(ValueError, match="is not compatible with parent wheel platform tag"):
        _build_sample_wheel(
            tmp_path,
            platform_tag="manylinux_2_28_x86_64",
            dependencies=(dependency.path,),
        )


def test_platform_wheel_requires_explicit_dependency_signer_trust(tmp_path, synthetic_descriptor_factory):
    dependency = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency"),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity="dependency-signer",
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )

    with pytest.raises(ValueError, match="dependency trust identities must be supplied explicitly and match exactly"):
        _build_sample_wheel(tmp_path, dependencies=(dependency.path,))

    built = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "root.duckdb_extension", b"root"),
        extension_name="root",
        output_directory=tmp_path / "root-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
        dependency_wheels=(dependency.path,),
        dependency_trust_identities=("dependency-signer",),
    )

    assert built.descriptor.dependencies[0].name == "dependency"


def test_platform_wheel_rejects_a_dependency_with_a_relabelled_glibc_floor(tmp_path, monkeypatch):
    def create_descriptor(path, *, name, trust_identity, dependencies):
        return _descriptor(
            path,
            name=name,
            trust_identity=trust_identity,
            platform="linux_amd64",
            dependencies=dependencies,
        )

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_descriptor)
    dependency = build_extension_wheel(
        artifact=_write_artifact(
            tmp_path / "dependency.duckdb_extension",
            b"dependency",
            architecture="x86_64",
            glibc_version="2.39",
        ),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag="manylinux_2_39_x86_64",
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    relabelled_dependency = _relabel_wheel_platform(
        dependency.path,
        tmp_path / "relabelled-dependency",
        original="manylinux_2_39_x86_64",
        replacement="manylinux_2_17_x86_64",
    )

    with pytest.raises(ValueError, match="requires glibc 2.39.*manylinux_2_17_x86_64"):
        _build_sample_wheel(
            tmp_path,
            platform_tag="manylinux_2_28_x86_64",
            dependencies=(relabelled_dependency,),
        )


def test_platform_wheel_rejects_a_dependency_for_another_python(
    tmp_path,
    synthetic_descriptor_factory,
):
    dependency = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency"),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    interpreter_tag = _extension_interpreter_tag()
    other_interpreter_tag = _other_extension_interpreter_tag()
    with zipfile.ZipFile(dependency.path) as dependency_wheel:
        wheel_metadata = next(name for name in dependency_wheel.namelist() if name.endswith(".dist-info/WHEEL"))
    tampered_directory = tmp_path / "other-python"
    tampered_directory.mkdir()
    tampered_name = dependency.path.name.replace(
        f"-{interpreter_tag}-none-",
        f"-{other_interpreter_tag}-none-",
    )
    tampered_dependency = _rewrite_wheel(
        dependency.path,
        tampered_directory / tampered_name,
        transforms={
            wheel_metadata: lambda contents: contents.replace(
                f"{interpreter_tag}-none-".encode(),
                f"{other_interpreter_tag}-none-".encode(),
            )
        },
    )

    with pytest.raises(ValueError, match="does not match root extension wheel interpreter tag"):
        _build_sample_wheel(tmp_path, dependencies=(tampered_dependency,))


@pytest.mark.parametrize(
    ("root_platform_tag", "root_musl_version"),
    [
        ("musllinux_1_1_x86_64", (1, 1)),
        ("musllinux_2_0_x86_64", (2, 0)),
    ],
    ids=["different-minor", "different-major"],
)
def test_platform_wheel_requires_one_exact_musl_policy_across_dependencies(
    tmp_path,
    monkeypatch,
    root_platform_tag,
    root_musl_version,
):
    def create_descriptor(path, *, name, trust_identity, dependencies):
        return _descriptor(
            path,
            name=name,
            trust_identity=trust_identity,
            platform="linux_amd64_musl",
            dependencies=dependencies,
        )

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_descriptor)
    monkeypatch.setattr(extension_wheel_module, "_current_musl_version", lambda: (1, 2))
    dependency = build_extension_wheel(
        artifact=_write_artifact(
            tmp_path / "dependency.duckdb_extension",
            b"dependency",
            architecture="x86_64",
        ),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag="musllinux_1_2_x86_64",
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    monkeypatch.setattr(extension_wheel_module, "_current_musl_version", lambda: root_musl_version)

    with pytest.raises(ValueError, match="is not compatible with parent wheel platform tag"):
        _build_sample_wheel(
            tmp_path,
            platform_tag=root_platform_tag,
            dependencies=(dependency.path,),
        )


@pytest.mark.parametrize(
    ("member_suffix", "expected_message"),
    [
        ("/__init__.py", "exact generated provider module"),
        (".dist-info/entry_points.txt", "exact generated provider entry point"),
    ],
)
def test_platform_wheel_rejects_dependency_wheels_without_the_exact_provider(
    tmp_path,
    synthetic_descriptor_factory,
    member_suffix,
    expected_message,
):
    dependency = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency"),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    with zipfile.ZipFile(dependency.path) as wheel:
        removed_member = next(name for name in wheel.namelist() if name.endswith(member_suffix))
    tampered_directory = tmp_path / f"missing-{removed_member.rsplit('/', 1)[-1]}"
    tampered_directory.mkdir()
    tampered_dependency = _rewrite_wheel(
        dependency.path,
        tampered_directory / dependency.path.name,
        removed_members={removed_member},
    )

    with pytest.raises(ValueError, match=expected_message):
        _build_sample_wheel(tmp_path, output_name="root-dist", dependencies=(tampered_dependency,))


def test_platform_wheel_rejects_unowned_members_in_dependency_wheels(
    tmp_path,
    synthetic_descriptor_factory,
):
    dependency = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency"),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    tampered_directory = tmp_path / "dependency-with-unowned-member"
    tampered_directory.mkdir()
    tampered_dependency = _rewrite_wheel(
        dependency.path,
        tampered_directory / dependency.path.name,
        extra_members={"requests/__init__.py": b"raise RuntimeError('wheel path collision')\n"},
    )

    with pytest.raises(ValueError, match="unowned or missing archive members"):
        _build_sample_wheel(tmp_path, output_name="root-dist", dependencies=(tampered_dependency,))


def test_platform_wheel_requires_exact_generated_wheel_metadata_from_dependencies(
    tmp_path,
    synthetic_descriptor_factory,
):
    dependency = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency"),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    with zipfile.ZipFile(dependency.path) as wheel:
        wheel_metadata = next(name for name in wheel.namelist() if name.endswith(".dist-info/WHEEL"))
    tampered_directory = tmp_path / "dependency-with-newer-wheel-version"
    tampered_directory.mkdir()
    tampered_dependency = _rewrite_wheel(
        dependency.path,
        tampered_directory / dependency.path.name,
        transforms={wheel_metadata: lambda contents: contents.replace(b"Wheel-Version: 1.0", b"Wheel-Version: 2.0")},
    )

    with pytest.raises(ValueError, match="exact generated WHEEL metadata"):
        _build_sample_wheel(tmp_path, output_name="root-dist", dependencies=(tampered_dependency,))


def test_platform_wheel_rejects_nonportable_paths_in_dependency_wheels(
    tmp_path,
    synthetic_descriptor_factory,
):
    dependency = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency"),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    with zipfile.ZipFile(dependency.path) as wheel:
        metadata_member = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
        license_member = next(name for name in wheel.namelist() if name.endswith(".dist-info/licenses/LICENSE"))
        license_contents = wheel.read(license_member)
    long_component = "a" * 256
    long_license_member = f"{license_member.rpartition('/')[0]}/{long_component}"
    tampered_directory = tmp_path / "dependency-with-nonportable-path"
    tampered_directory.mkdir()
    tampered_dependency = _rewrite_wheel(
        dependency.path,
        tampered_directory / dependency.path.name,
        transforms={
            metadata_member: lambda contents: contents.replace(
                b"License-File: LICENSE\n",
                f"License-File: {long_component}\n".encode("utf-8"),
            )
        },
        extra_members={long_license_member: license_contents},
        removed_members={license_member},
    )

    with pytest.raises(ValueError, match="portable 255-byte limit"):
        _build_sample_wheel(tmp_path, output_name="root-dist", dependencies=(tampered_dependency,))


def test_platform_wheel_rejects_dependency_artifacts_that_disagree_with_their_descriptors(
    tmp_path,
    synthetic_descriptor_factory,
    monkeypatch,
):
    dependency = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency"),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    create_descriptor = extension_wheel_module._create_descriptor

    def inspect_different_footer(path, *, name, trust_identity, dependencies):
        inspected = create_descriptor(
            path,
            name=name,
            trust_identity=trust_identity,
            dependencies=dependencies,
        )
        return replace(inspected, extension_version=f"{inspected.extension_version}-different")

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", inspect_different_footer)

    with pytest.raises(ValueError, match="native metadata does not match its descriptor"):
        _build_sample_wheel(tmp_path, output_name="root-dist", dependencies=(dependency.path,))


def test_platform_wheel_rejects_oversized_dependency_wheels(
    tmp_path,
    synthetic_descriptor_factory,
    monkeypatch,
):
    dependency = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency"),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    monkeypatch.setattr(extension_wheel_module, "_MAX_EXTENSION_WHEEL_BYTES", 0)

    with pytest.raises(ValueError, match="128 MiB publication limit"):
        _build_sample_wheel(tmp_path, output_name="root-dist", dependencies=(dependency.path,))


def test_platform_wheel_rejects_dependency_wheels_with_a_narrower_python_range(
    tmp_path,
    synthetic_descriptor_factory,
):
    dependency = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency"),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    tampered_dependency = _rewrite_wheel_metadata(
        dependency.path,
        tmp_path / "dependency-with-narrower-python-range",
        lambda metadata: metadata.replace("Requires-Python: >=3.10,<3.15", "Requires-Python: >=3.11,<3.15"),
    )

    with pytest.raises(ValueError, match="Requires-Python must match the supported Python range exactly"):
        _build_sample_wheel(tmp_path, output_name="root-dist", dependencies=(tampered_dependency,))


@pytest.mark.parametrize("metadata_version", [None, "3.0"])
def test_platform_wheel_requires_the_generated_metadata_version_from_dependencies(
    tmp_path,
    synthetic_descriptor_factory,
    metadata_version,
):
    dependency = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency"),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )

    def transform(metadata: str) -> str:
        replacement = "" if metadata_version is None else f"Metadata-Version: {metadata_version}\n"
        return metadata.replace("Metadata-Version: 2.4\n", replacement)

    tampered_dependency = _rewrite_wheel_metadata(
        dependency.path,
        tmp_path / f"dependency-metadata-{metadata_version or 'missing'}",
        transform,
    )

    with pytest.raises(ValueError, match="Metadata-Version must match the generated core metadata version exactly"):
        _build_sample_wheel(tmp_path, output_name="root-dist", dependencies=(tampered_dependency,))


@pytest.mark.parametrize("line_ending", [b"\n", b"\r"], ids=["lf", "bare-cr"])
def test_platform_wheel_bounds_dependency_metadata_headers_before_email_parsing(
    tmp_path,
    synthetic_descriptor_factory,
    monkeypatch,
    line_ending,
):
    dependency = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency"),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    with zipfile.ZipFile(dependency.path) as wheel:
        metadata_member = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
    tampered_directory = tmp_path / "many-metadata-headers"
    tampered_directory.mkdir()
    tampered = _rewrite_wheel(
        dependency.path,
        tampered_directory / dependency.path.name,
        transforms={metadata_member: lambda contents: ((b"X-Untrusted: value" + line_ending) * 32) + contents},
    )

    class RejectingParser:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("email parser was constructed before metadata headers were bounded")

    monkeypatch.setattr(extension_wheel_module, "_MAX_CORE_METADATA_HEADERS", 16)
    monkeypatch.setattr(extension_wheel_module, "BytesParser", RejectingParser)

    with pytest.raises(ValueError, match="METADATA contains more than 16 headers"):
        _build_sample_wheel(tmp_path, output_name="root-dist", dependencies=(tampered,))


@pytest.mark.parametrize("line_ending", [b"\n", b"\r"], ids=["lf", "bare-cr"])
def test_clean_verifier_bounds_metadata_headers_before_email_parsing(
    tmp_path,
    synthetic_descriptor_factory,
    monkeypatch,
    line_ending,
):
    built = _build_sample_wheel(tmp_path)
    with zipfile.ZipFile(built.path) as wheel:
        metadata_member = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
    tampered_directory = tmp_path / "many-root-metadata-headers"
    tampered_directory.mkdir()
    tampered = _rewrite_wheel(
        built.path,
        tampered_directory / built.path.name,
        transforms={metadata_member: lambda contents: ((b"X-Untrusted: value" + line_ending) * 32) + contents},
    )

    class RejectingParser:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("email parser was constructed before metadata headers were bounded")

    monkeypatch.setattr(extension_wheel_module, "_MAX_CORE_METADATA_HEADERS", 16)
    monkeypatch.setattr(extension_wheel_module, "BytesParser", RejectingParser)

    with pytest.raises(RuntimeError, match="METADATA contains more than 16 headers"):
        _extension_name_from_wheel(tampered)


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        ("missing-requirement", "Requires-Dist must match its descriptor exactly"),
        ("wrong-requirement-version", "Requires-Dist must match its descriptor exactly"),
        ("missing-dependency-wheel", "no exact dependency wheel was supplied"),
    ],
)
def test_platform_wheel_rejects_incomplete_dependency_wheel_closure_or_requirements(
    tmp_path,
    synthetic_descriptor_factory,
    mutation,
    expected_message,
):
    leaf = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "leaf.duckdb_extension", b"leaf"),
        extension_name="leaf",
        output_directory=tmp_path / "leaf-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    parent = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "parent.duckdb_extension", b"parent"),
        extension_name="parent",
        output_directory=tmp_path / "parent-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
        dependency_wheels=(leaf.path,),
        dependency_trust_identities=(TEST_TRUST_IDENTITY,),
    )
    dependencies = (parent.path,)
    if mutation != "missing-dependency-wheel":
        leaf_requirement = f"Requires-Dist: vane-extension-leaf==={leaf.distribution_version}"

        def transform(metadata: str) -> str:
            if mutation == "missing-requirement":
                return metadata.replace(leaf_requirement + "\n", "")
            assert mutation == "wrong-requirement-version"
            return metadata.replace(leaf_requirement, "Requires-Dist: vane-extension-leaf===0")

        tampered_parent = _rewrite_wheel_metadata(
            parent.path,
            tmp_path / f"parent-with-{mutation}",
            transform,
        )
        dependencies = (leaf.path, tampered_parent)

    with pytest.raises(ValueError, match=expected_message):
        _build_sample_wheel(
            tmp_path,
            output_name="root-dist",
            dependencies=dependencies,
        )


def test_platform_wheel_rejects_a_narrower_policy_inside_the_dependency_graph(tmp_path):
    leaf = _descriptor(
        _write_artifact(tmp_path / "leaf.duckdb_extension", b"leaf"),
        name="leaf",
        platform="linux_amd64",
    )
    parent = _descriptor(
        _write_artifact(tmp_path / "parent.duckdb_extension", b"parent"),
        name="parent",
        platform="linux_amd64",
        dependencies=(extension_wheel_module._dependency_reference(leaf),),
    )
    dependency_wheels = (
        extension_wheel_module._DependencyWheel(
            descriptor=parent,
            interpreter_tag=_extension_interpreter_tag(),
            platform_tag="manylinux_2_28_x86_64",
            distribution_version="parent-version",
            requirements=(),
        ),
        extension_wheel_module._DependencyWheel(
            descriptor=leaf,
            interpreter_tag=_extension_interpreter_tag(),
            platform_tag="manylinux_2_39_x86_64",
            distribution_version="leaf-version",
            requirements=(),
        ),
    )

    with pytest.raises(ValueError, match="is not compatible with parent wheel platform tag"):
        extension_wheel_module._validate_dependency_wheel_platforms(
            "manylinux_2_39_x86_64",
            dependency_wheels,
            root_interpreter_tag=_extension_interpreter_tag(),
        )


@pytest.mark.parametrize(
    ("descriptor_change", "expected_message"),
    [
        ({"source_id": "0" * 40}, "SourceID"),
        ({"vane_version": "0.0.0"}, "Vane version"),
        ({"trust_identity": "other-trust"}, "trust identity"),
        ({"name": "other"}, "descriptor name"),
    ],
)
def test_platform_wheel_rejects_a_descriptor_that_differs_from_the_request(
    tmp_path,
    monkeypatch,
    descriptor_change,
    expected_message,
):
    artifact_path = _write_artifact(tmp_path / "sample.duckdb_extension")

    def create_descriptor(path, *, name, trust_identity, dependencies):
        return _descriptor(
            path,
            name=descriptor_change.get("name", name),
            trust_identity=descriptor_change.get("trust_identity", trust_identity),
            source_id=descriptor_change.get("source_id"),
            vane_version=descriptor_change.get("vane_version"),
            dependencies=dependencies,
        )

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_descriptor)
    with pytest.raises(RuntimeError, match=expected_message):
        _build_sample_wheel(tmp_path, artifact_path=artifact_path)


def test_platform_wheel_rejects_an_artifact_named_for_another_extension(tmp_path):
    artifact_path = _write_artifact(tmp_path / "another.duckdb_extension")

    with pytest.raises(ValueError, match="artifact must be named"):
        build_extension_wheel(
            artifact=artifact_path,
            extension_name="sample",
            output_directory=tmp_path / "dist",
            platform_tag="linux_x86_64",
            trust_identity=TEST_TRUST_IDENTITY,
            license_expression="Apache-2.0 AND MIT",
            license_files=[REPOSITORY_ROOT / "LICENSE"],
        )


def test_platform_wheel_rejects_duplicate_self_or_incompatible_dependencies(
    tmp_path,
):
    dependency_path = _write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency")
    dependency = _descriptor(dependency_path, name="dependency")
    conflicting_dependency = _descriptor(
        _write_artifact(tmp_path / "dependency-2.duckdb_extension", b"dependency-2"),
        name="dependency",
    )
    self_dependency = _descriptor(_write_artifact(tmp_path / "self.duckdb_extension"), name="sample")
    wrong_source = _descriptor(dependency_path, name="wrong_source", source_id="0" * 40)
    wrong_version = _descriptor(dependency_path, name="wrong_version", vane_version="0.0.0")
    wrong_platform = _descriptor(dependency_path, name="wrong_platform", platform="test_other_platform")
    invalid_wheel_name = _descriptor(dependency_path, name="wrong__name")

    def validate(*dependencies):
        return extension_wheel_module._validate_dependency_descriptors(
            dependencies,
            extension_name="sample",
            runtime_source_id=vane.__git_revision__,
            runtime_vane_version=vane.__version__,
        )

    with pytest.raises(ValueError, match="more than once"):
        validate(dependency, conflicting_dependency)
    with pytest.raises(ValueError, match="depend on itself"):
        validate(self_dependency)
    with pytest.raises(RuntimeError, match="SourceID"):
        validate(wrong_source)
    with pytest.raises(RuntimeError, match="Vane version"):
        validate(wrong_version)
    with pytest.raises(ValueError, match="cannot form a normalized wheel name"):
        validate(invalid_wheel_name)

    root_artifact = _write_artifact(tmp_path / "sample.duckdb_extension")
    root_descriptor = _descriptor(
        root_artifact,
        dependencies=(extension_wheel_module._dependency_reference(wrong_platform),),
    )
    with pytest.raises(RuntimeError, match="same platform"):
        extension_wheel_module._validate_descriptor(
            root_descriptor,
            artifact_contents=root_artifact.read_bytes(),
            extension_name="sample",
            trust_identity=TEST_TRUST_IDENTITY,
            runtime_source_id=vane.__git_revision__,
            runtime_vane_version=vane.__version__,
            dependencies=(wrong_platform,),
        )


def test_platform_wheel_rejects_a_cycle_through_a_supplied_dependency(
    tmp_path,
):
    dependency_path = _write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency")
    dependency = _descriptor(
        dependency_path,
        name="dependency",
        dependencies=(
            DynamicExtensionDependency(
                name="sample",
                extension_version="test-version",
                sha256="0" * 64,
            ),
        ),
    )

    with pytest.raises(ValueError, match="dependency cycle"):
        extension_wheel_module._validate_dependency_descriptors(
            (dependency,),
            extension_name="sample",
            runtime_source_id=vane.__git_revision__,
            runtime_vane_version=vane.__version__,
        )


def test_platform_wheel_rejects_a_mismatched_identity_for_a_represented_dependency(
    tmp_path,
):
    leaf_path = _write_artifact(tmp_path / "leaf.duckdb_extension", b"leaf")
    leaf = _descriptor(leaf_path, name="leaf")
    parent = _descriptor(
        _write_artifact(tmp_path / "parent.duckdb_extension", b"parent"),
        name="parent",
        dependencies=(
            DynamicExtensionDependency(
                name=leaf.name,
                extension_version=leaf.extension_version,
                sha256="0" * 64,
            ),
        ),
    )

    with pytest.raises(ValueError, match="different identity"):
        extension_wheel_module._validate_dependency_descriptors(
            (parent, leaf),
            extension_name="sample",
            runtime_source_id=vane.__git_revision__,
            runtime_vane_version=vane.__version__,
        )


@pytest.mark.parametrize(
    ("distribution_name", "version", "expected_message"),
    [
        ("another-project", None, "must be the vane-ai distribution"),
        ("vane-ai", "0.0.0", "must match extension Vane version"),
    ],
)
def test_clean_verifier_rejects_the_wrong_base_wheel_identity(
    tmp_path,
    synthetic_descriptor_factory,
    distribution_name,
    version,
    expected_message,
):
    base_wheel = _write_minimal_base_wheel(
        tmp_path,
        distribution_name=distribution_name,
        version=version,
    )
    extension_wheel = _build_sample_wheel(tmp_path).path

    with pytest.raises(RuntimeError, match=expected_message):
        verify_extension_wheel(
            base_wheel=base_wheel,
            extension_wheel=extension_wheel,
            extension_name="sample",
            trust_identity=TEST_TRUST_IDENTITY,
        )


@pytest.mark.parametrize("requires_python", [None, ">=3.11,<3.15", "not-a-specifier"])
def test_clean_verifier_requires_the_exact_python_range_from_the_base_wheel(tmp_path, requires_python):
    base_wheel = _write_minimal_base_wheel(tmp_path, requires_python=requires_python)

    with pytest.raises(RuntimeError, match="base Vane wheel Requires-Python must match"):
        verify_extension_wheel_module._assert_base_wheel(
            base_wheel,
            expected_vane_version=vane.__version__,
            required_interpreter_tag=_extension_interpreter_tag(),
            required_platform_tag=_wheel_platform_tag(),
        )


@pytest.mark.parametrize("metadata_version", [None, "3.0"])
def test_clean_verifier_requires_the_generated_metadata_version_from_the_base_wheel(tmp_path, metadata_version):
    base_wheel = _write_minimal_base_wheel(tmp_path, metadata_version=metadata_version)

    with pytest.raises(RuntimeError, match="base Vane wheel Metadata-Version must match"):
        verify_extension_wheel_module._assert_base_wheel(
            base_wheel,
            expected_vane_version=vane.__version__,
            required_interpreter_tag=_extension_interpreter_tag(),
            required_platform_tag=_wheel_platform_tag(),
        )


def test_clean_verifier_accepts_the_base_wheel_build_backend_python_range_order(tmp_path):
    base_wheel = _write_minimal_base_wheel(tmp_path, requires_python="<3.15,>=3.10")

    verify_extension_wheel_module._assert_base_wheel(
        base_wheel,
        expected_vane_version=vane.__version__,
        required_interpreter_tag=_extension_interpreter_tag(),
        required_platform_tag=_wheel_platform_tag(),
    )


def test_clean_verifier_rejects_a_base_wheel_build_tag(tmp_path):
    base_wheel = _write_minimal_base_wheel(tmp_path, build_tag="1")

    with pytest.raises(RuntimeError, match="base Vane wheel must not use a build tag"):
        verify_extension_wheel_module._assert_base_wheel(
            base_wheel,
            expected_vane_version=vane.__version__,
            required_interpreter_tag=_extension_interpreter_tag(),
            required_platform_tag=_wheel_platform_tag(),
        )


def test_clean_verifier_requires_base_wheel_tags_to_match_its_filename(tmp_path):
    base_wheel = _write_minimal_base_wheel(tmp_path, platform_tag="manylinux_2_39_x86_64")
    renamed_wheel = base_wheel.with_name(base_wheel.name.replace("manylinux_2_39", "manylinux_2_17"))
    base_wheel.rename(renamed_wheel)

    with pytest.raises(RuntimeError, match="WHEEL tags must match its filename tags exactly"):
        verify_extension_wheel_module._assert_base_wheel(
            renamed_wheel,
            expected_vane_version=vane.__version__,
            required_interpreter_tag=_extension_interpreter_tag(),
            required_platform_tag="manylinux_2_17_x86_64",
        )


@pytest.mark.parametrize(
    ("original", "replacement", "expected_message"),
    [
        (b"Root-Is-Purelib: false", b"Root-Is-Purelib: true", "exactly Root-Is-Purelib: false"),
        (b"Wheel-Version: 1.0", b"Wheel-Version: 2.0", "exactly Wheel-Version: 1.0"),
    ],
)
def test_clean_verifier_requires_generated_base_wheel_installation_metadata(
    tmp_path,
    original,
    replacement,
    expected_message,
):
    base_wheel = _write_minimal_base_wheel(tmp_path, platform_tag="manylinux_2_39_x86_64")
    with zipfile.ZipFile(base_wheel) as wheel:
        wheel_member = next(name for name in wheel.namelist() if name.endswith(".dist-info/WHEEL"))
    tampered_directory = tmp_path / "purelib-base"
    tampered_directory.mkdir()
    tampered_wheel = _rewrite_wheel(
        base_wheel,
        tampered_directory / base_wheel.name,
        transforms={wheel_member: lambda contents: contents.replace(original, replacement)},
    )

    with pytest.raises(RuntimeError, match=expected_message):
        verify_extension_wheel_module._assert_base_wheel(
            tampered_wheel,
            expected_vane_version=vane.__version__,
            required_interpreter_tag=_extension_interpreter_tag(),
            required_platform_tag="manylinux_2_39_x86_64",
        )


def test_clean_verifier_rejects_a_base_wheel_with_a_relabelled_glibc_floor(tmp_path):
    base_wheel = _write_minimal_base_wheel(
        tmp_path,
        platform_tag="manylinux_2_39_x86_64",
        native_glibc_version="2.39",
    )
    relabelled_wheel = _relabel_wheel_platform(
        base_wheel,
        tmp_path / "relabelled-base",
        original="manylinux_2_39_x86_64",
        replacement="manylinux_2_17_x86_64",
    )

    with pytest.raises(RuntimeError, match="requires glibc 2.39.*manylinux_2_17_x86_64"):
        verify_extension_wheel_module._assert_base_wheel(
            relabelled_wheel,
            expected_vane_version=vane.__version__,
            required_interpreter_tag=_extension_interpreter_tag(),
            required_platform_tag="manylinux_2_17_x86_64",
        )


@pytest.mark.parametrize("dependency_kind", ["native_needed", "native_filters", "native_auxiliaries"])
def test_clean_verifier_rejects_an_unbundled_private_base_elf_dependency(tmp_path, dependency_kind):
    base_wheel = _write_minimal_base_wheel(
        tmp_path,
        platform_tag="manylinux_2_17_x86_64",
        **{dependency_kind: ("libprivate.so.1",)},
    )

    with pytest.raises(RuntimeError, match="non-policy ELF shared libraries.*libprivate.so.1"):
        verify_extension_wheel_module._assert_base_wheel(
            base_wheel,
            expected_vane_version=vane.__version__,
            required_interpreter_tag=_extension_interpreter_tag(),
            required_platform_tag="manylinux_2_17_x86_64",
        )


def test_clean_verifier_rejects_a_blacklisted_undefined_symbol_in_a_base_member(tmp_path):
    base_wheel = _write_minimal_base_wheel(
        tmp_path,
        platform_tag="manylinux_2_17_x86_64",
        native_needed=("libz.so.1",),
        native_undefined_symbols=("_dist_code",),
    )

    with pytest.raises(RuntimeError, match=r"blacklisted undefined ELF symbols.*libz\.so\.1.*_dist_code"):
        verify_extension_wheel_module._assert_base_wheel(
            base_wheel,
            expected_vane_version=vane.__version__,
            required_interpreter_tag=_extension_interpreter_tag(),
            required_platform_tag="manylinux_2_17_x86_64",
        )


def test_clean_verifier_rejects_a_newer_base_elf_symbol_policy(tmp_path):
    base_wheel = _write_minimal_base_wheel(
        tmp_path,
        platform_tag="manylinux_2_28_x86_64",
        native_versioned_symbols={"libstdc++.so.6": ("GLIBCXX_3.4.32",)},
    )

    with pytest.raises(RuntimeError, match="versioned ELF symbols.*GLIBCXX_3.4.32"):
        verify_extension_wheel_module._assert_base_wheel(
            base_wheel,
            expected_vane_version=vane.__version__,
            required_interpreter_tag=_extension_interpreter_tag(),
            required_platform_tag="manylinux_2_28_x86_64",
        )


def test_clean_verifier_rejects_an_unknown_base_elf_symbol_policy(tmp_path):
    base_wheel = _write_minimal_base_wheel(
        tmp_path,
        platform_tag="manylinux_2_39_x86_64",
        native_versioned_symbols={"libstdc++.so.6": ("NEWABI_1.0",)},
    )

    with pytest.raises(RuntimeError, match="versioned ELF symbols.*NEWABI_1.0"):
        verify_extension_wheel_module._assert_base_wheel(
            base_wheel,
            expected_vane_version=vane.__version__,
            required_interpreter_tag=_extension_interpreter_tag(),
            required_platform_tag="manylinux_2_39_x86_64",
        )


def test_clean_verifier_rejects_a_base_wheel_with_a_relabelled_macos_target(tmp_path):
    base_wheel = _write_minimal_base_wheel(
        tmp_path,
        platform_tag="macosx_14_0_arm64",
        native_macos_version=(14, 0, 0),
    )
    relabelled_wheel = _relabel_wheel_platform(
        base_wheel,
        tmp_path / "relabelled-macos-base",
        original="macosx_14_0_arm64",
        replacement="macosx_11_0_arm64",
    )

    with pytest.raises(RuntimeError, match="requires macOS 14.0.0.*macosx_11_0_arm64"):
        verify_extension_wheel_module._assert_base_wheel(
            relabelled_wheel,
            expected_vane_version=vane.__version__,
            required_interpreter_tag=_extension_interpreter_tag(),
            required_platform_tag="macosx_11_0_arm64",
        )


def test_clean_verifier_rejects_a_non_system_base_macos_dependency(tmp_path):
    base_wheel = _write_minimal_base_wheel(
        tmp_path,
        platform_tag="macosx_14_0_arm64",
        native_macos_dynamic_libraries=("/usr/local/lib/libprivate.dylib",),
    )

    with pytest.raises(RuntimeError, match="non-system Mach-O dynamic libraries.*libprivate.dylib"):
        verify_extension_wheel_module._assert_base_wheel(
            base_wheel,
            expected_vane_version=vane.__version__,
            required_interpreter_tag=_extension_interpreter_tag(),
            required_platform_tag="macosx_14_0_arm64",
        )


@pytest.mark.parametrize("import_kind", ["native_windows_imports", "native_windows_delay_imports"])
def test_clean_verifier_rejects_a_private_base_windows_import(tmp_path, import_kind):
    base_wheel = _write_minimal_base_wheel(
        tmp_path,
        platform_tag="win_amd64",
        **{import_kind: ("publisher-private.dll",)},
    )

    with pytest.raises(RuntimeError, match="non-policy Windows DLLs.*publisher-private.dll"):
        verify_extension_wheel_module._assert_base_wheel(
            base_wheel,
            expected_vane_version=vane.__version__,
            required_interpreter_tag=_extension_interpreter_tag(),
            required_platform_tag="win_amd64",
        )


def test_clean_verifier_rejects_a_base_windows_export_forwarder(tmp_path):
    base_wheel = _write_minimal_base_wheel(
        tmp_path,
        platform_tag="win_amd64",
        native_windows_forwarded_exports=("publisher-private.forwarded_initializer",),
    )

    with pytest.raises(RuntimeError, match="PE export forwarder"):
        verify_extension_wheel_module._assert_base_wheel(
            base_wheel,
            expected_vane_version=vane.__version__,
            required_interpreter_tag=_extension_interpreter_tag(),
            required_platform_tag="win_amd64",
        )


def test_clean_verifier_accepts_base_windows_policy_imports(tmp_path):
    interpreter_tag = _extension_interpreter_tag()
    python_library = f"python{interpreter_tag.removeprefix('cp')}.dll"
    base_wheel = _write_minimal_base_wheel(
        tmp_path,
        platform_tag="win_amd64",
        native_windows_imports=("kernel32.dll", python_library),
        native_windows_delay_imports=("vcruntime140.dll",),
    )

    verify_extension_wheel_module._assert_base_wheel(
        base_wheel,
        expected_vane_version=vane.__version__,
        required_interpreter_tag=interpreter_tag,
        required_platform_tag="win_amd64",
    )


@pytest.mark.parametrize("base_platform_tag", ["any", "any.manylinux_2_17_x86_64"])
def test_clean_verifier_rejects_platform_neutral_base_wheel_tags(tmp_path, base_platform_tag):
    base_wheel = _write_minimal_base_wheel(tmp_path, platform_tag=base_platform_tag)

    with pytest.raises(RuntimeError, match="must not use the platform-neutral 'any' tag"):
        verify_extension_wheel_module._assert_base_wheel(
            base_wheel,
            expected_vane_version=vane.__version__,
            required_interpreter_tag=_extension_interpreter_tag(),
            required_platform_tag="manylinux_2_17_x86_64",
        )


@pytest.mark.parametrize(
    "base_platform_tag",
    [
        "linux_x86_64",
        "linux_x86_64.manylinux_2_17_x86_64",
        "linux_aarch64",
        "linux_aarch64.manylinux_2_17_aarch64",
    ],
)
def test_clean_verifier_rejects_generic_linux_base_wheel_tags(tmp_path, base_platform_tag):
    base_wheel = _write_minimal_base_wheel(tmp_path, platform_tag=base_platform_tag)

    with pytest.raises(RuntimeError, match="must not use generic Linux platform tags"):
        verify_extension_wheel_module._assert_base_wheel(
            base_wheel,
            expected_vane_version=vane.__version__,
            required_interpreter_tag=_extension_interpreter_tag(),
            required_platform_tag="manylinux_2_17_x86_64",
        )


@pytest.mark.parametrize(
    ("interpreter_tag", "abi_tag"),
    [
        (_other_extension_interpreter_tag(), _other_extension_interpreter_tag()),
        (_extension_interpreter_tag(), "none"),
        (_extension_interpreter_tag(), "abi3"),
        (_extension_interpreter_tag(), "invalidabi"),
    ],
)
def test_clean_verifier_requires_a_matching_native_base_interpreter_tag(tmp_path, interpreter_tag, abi_tag):
    base_wheel = _write_minimal_base_wheel(
        tmp_path,
        interpreter_tag=interpreter_tag,
        abi_tag=abi_tag,
    )

    with pytest.raises(RuntimeError, match="tags must use interpreter and ABI .* exactly"):
        verify_extension_wheel_module._assert_base_wheel(
            base_wheel,
            expected_vane_version=vane.__version__,
            required_interpreter_tag=_extension_interpreter_tag(),
            required_platform_tag=_wheel_platform_tag(),
        )


@pytest.mark.parametrize(
    ("base_platform_tag", "extension_platform_tag", "covers"),
    [
        ("manylinux_2_28_x86_64", "manylinux_2_17_x86_64", False),
        ("manylinux_2_17_x86_64", "manylinux_2_28_x86_64", True),
        ("manylinux2014_x86_64", "manylinux_2_17_x86_64", True),
        ("musllinux_1_1_x86_64", "musllinux_1_2_x86_64", False),
        ("musllinux_1_2_x86_64", "musllinux_1_1_x86_64", False),
        ("musllinux_1_2_x86_64", "musllinux_2_0_x86_64", False),
        ("macosx_13_0_universal2", "macosx_14_0_arm64", True),
        ("macosx_14_0_universal2", "macosx_13_0_arm64", False),
        ("macosx_11_0_x86_64", "macosx_10_16_x86_64", True),
        ("macosx_11_0_universal2", "macosx_10_16_x86_64", True),
        ("macosx_10_16_universal2", "macosx_11_0_arm64", True),
    ],
)
def test_clean_verifier_requires_the_base_platform_to_cover_the_extension(
    tmp_path,
    base_platform_tag,
    extension_platform_tag,
    covers,
):
    base_wheel = _write_minimal_base_wheel(tmp_path, platform_tag=base_platform_tag)

    if covers:
        verify_extension_wheel_module._assert_base_wheel(
            base_wheel,
            expected_vane_version=vane.__version__,
            required_interpreter_tag=_extension_interpreter_tag(),
            required_platform_tag=extension_platform_tag,
        )
    else:
        with pytest.raises(RuntimeError, match="do not cover extension wheel platform tag"):
            verify_extension_wheel_module._assert_base_wheel(
                base_wheel,
                expected_vane_version=vane.__version__,
                required_interpreter_tag=_extension_interpreter_tag(),
                required_platform_tag=extension_platform_tag,
            )


def test_clean_verifier_requires_every_base_platform_to_cover_the_extension(tmp_path):
    base_wheel = _write_minimal_base_wheel(
        tmp_path,
        platform_tag="manylinux_2_17_x86_64.win_amd64",
    )

    with pytest.raises(RuntimeError, match=r"platform tags \('win_amd64',\) do not cover"):
        verify_extension_wheel_module._assert_base_wheel(
            base_wheel,
            expected_vane_version=vane.__version__,
            required_interpreter_tag=_extension_interpreter_tag(),
            required_platform_tag="manylinux_2_17_x86_64",
        )


def test_clean_verifier_rejects_unowned_archive_members(tmp_path, synthetic_descriptor_factory):
    built = _build_sample_wheel(tmp_path)
    tampered_directory = tmp_path / "tampered-unowned-member"
    tampered_directory.mkdir()
    tampered_wheel = _rewrite_wheel(
        built.path,
        tampered_directory / built.path.name,
        extra_members={"requests/__init__.py": b"raise RuntimeError('wheel path collision')\n"},
    )

    with pytest.raises(RuntimeError, match="unowned or missing archive members"):
        verify_extension_wheel(
            base_wheel=_write_minimal_base_wheel(tmp_path),
            extension_wheel=tampered_wheel,
            extension_name="sample",
            trust_identity=TEST_TRUST_IDENTITY,
        )


def test_clean_verifier_rejects_unowned_members_in_the_base_wheel(tmp_path):
    base_wheel = _write_minimal_base_wheel(tmp_path)
    tampered_directory = tmp_path / "tampered-base-unowned-member"
    tampered_directory.mkdir()
    tampered_wheel = _rewrite_wheel(
        base_wheel,
        tampered_directory / base_wheel.name,
        extra_members={"requests/__init__.py": b"raise RuntimeError('wheel path collision')\n"},
    )

    with pytest.raises(RuntimeError, match="Vane wheel contains conflicting Python package path"):
        verify_extension_wheel_module._assert_base_wheel(
            tampered_wheel,
            expected_vane_version=vane.__version__,
            required_interpreter_tag=_extension_interpreter_tag(),
            required_platform_tag=_wheel_platform_tag(),
        )


def test_clean_verifier_rejects_undeclared_base_wheel_entry_points(tmp_path):
    base_wheel = _write_minimal_base_wheel(tmp_path)
    tampered_directory = tmp_path / "tampered-base-entry-points"
    tampered_directory.mkdir()
    with zipfile.ZipFile(base_wheel) as wheel:
        metadata_member = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
    entry_points_member = metadata_member.removesuffix("METADATA") + "entry_points.txt"
    tampered_wheel = _rewrite_wheel(
        base_wheel,
        tampered_directory / base_wheel.name,
        extra_members={entry_points_member: b"[console_scripts]\npip = missing:main\n"},
    )

    with pytest.raises(RuntimeError, match="project metadata declares no entry points"):
        verify_extension_wheel_module._assert_base_wheel(
            tampered_wheel,
            expected_vane_version=vane.__version__,
            required_interpreter_tag=_extension_interpreter_tag(),
            required_platform_tag=_wheel_platform_tag(),
        )


def test_clean_verifier_rejects_base_contents_that_do_not_match_record(tmp_path):
    base_wheel = _write_minimal_base_wheel(tmp_path)
    tampered_directory = tmp_path / "tampered-base-record"
    tampered_directory.mkdir()
    tampered_wheel = _rewrite_wheel(
        base_wheel,
        tampered_directory / base_wheel.name,
        transforms={"vane/py.typed": lambda _contents: b"changed without updating RECORD\n"},
        update_record=False,
    )

    with pytest.raises(RuntimeError, match="invalid RECORD entry for vane/py.typed"):
        verify_extension_wheel_module._assert_base_wheel(
            tampered_wheel,
            expected_vane_version=vane.__version__,
            required_interpreter_tag=_extension_interpreter_tag(),
            required_platform_tag=_wheel_platform_tag(),
        )


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        ("added", "Requires-Dist must match project metadata exactly"),
        ("missing", "Requires-Dist must match project metadata exactly"),
        ("missing-extra", "Provides-Extra must match project metadata exactly"),
    ],
)
def test_clean_verifier_requires_the_base_wheel_complete_dependency_metadata(
    tmp_path,
    mutation,
    expected_message,
):
    base_wheel = _write_minimal_base_wheel(tmp_path)

    def transform(metadata: str) -> str:
        if mutation == "added":
            return metadata + "Requires-Dist: requests\n"
        if mutation == "missing":
            return metadata.replace("Requires-Dist: numpy\n", "")
        return metadata.replace("Provides-Extra: openai\n", "")

    tampered_directory = tmp_path / f"tampered-base-requirements-{mutation}"
    tampered_wheel = _rewrite_wheel_metadata(base_wheel, tampered_directory, transform)

    with pytest.raises(RuntimeError, match=expected_message):
        verify_extension_wheel_module._assert_base_wheel(
            tampered_wheel,
            expected_vane_version=vane.__version__,
            required_interpreter_tag=_extension_interpreter_tag(),
            required_platform_tag=_wheel_platform_tag(),
        )


def test_clean_verifier_rejects_contents_that_do_not_match_record(tmp_path, synthetic_descriptor_factory):
    built = _build_sample_wheel(tmp_path)
    with zipfile.ZipFile(built.path) as wheel:
        license_member = next(name for name in wheel.namelist() if name.endswith(".dist-info/licenses/LICENSE"))
    tampered_directory = tmp_path / "tampered-record"
    tampered_directory.mkdir()
    tampered_wheel = _rewrite_wheel(
        built.path,
        tampered_directory / built.path.name,
        transforms={license_member: lambda _contents: b"changed without updating RECORD\n"},
        update_record=False,
    )

    with pytest.raises(RuntimeError, match="RECORD entry is invalid"):
        verify_extension_wheel(
            base_wheel=_write_minimal_base_wheel(tmp_path),
            extension_wheel=tampered_wheel,
            extension_name="sample",
            trust_identity=TEST_TRUST_IDENTITY,
        )


def test_extension_wheel_record_is_streamed_instead_of_read_whole(
    tmp_path,
    synthetic_descriptor_factory,
    monkeypatch,
):
    built = _build_sample_wheel(tmp_path)
    read_member = extension_wheel_module.zipfile.ZipFile.read

    with zipfile.ZipFile(built.path) as wheel:
        names = wheel.namelist()
        record_name = next(name for name in names if name.endswith(".dist-info/RECORD"))

        def reject_whole_record_read(archive, member, *args, **kwargs):
            member_name = member.filename if isinstance(member, zipfile.ZipInfo) else member
            if member_name == record_name:
                raise AssertionError("extension wheel RECORD was read as one allocation")
            return read_member(archive, member, *args, **kwargs)

        monkeypatch.setattr(extension_wheel_module.zipfile.ZipFile, "read", reject_whole_record_read)
        extension_wheel_module._validate_wheel_record(
            wheel,
            names=names,
            record_name=record_name,
        )


def test_extension_wheel_record_rows_cannot_outnumber_archive_members(tmp_path):
    wheel_path = tmp_path / "record-row-limit.whl"
    record_name = "sample-1.dist-info/RECORD"
    payload_name = "sample/payload"
    with zipfile.ZipFile(wheel_path, mode="w") as wheel:
        wheel.writestr(payload_name, b"payload")
        wheel.writestr(
            record_name,
            extension_wheel_module._record({payload_name: b"payload"}, record_name) + "extra-member,,\n",
        )

    with zipfile.ZipFile(wheel_path) as wheel:
        names = wheel.namelist()
        with pytest.raises(ValueError, match="RECORD contains more rows than archive members"):
            extension_wheel_module._validate_wheel_record(
                wheel,
                names=names,
                record_name=record_name,
            )


def test_extension_wheel_record_bounds_each_row_before_csv_parsing(tmp_path, monkeypatch):
    wheel_path = tmp_path / "record-row-size-limit.whl"
    record_name = "sample-1.dist-info/RECORD"
    payload_name = "sample/payload"
    with zipfile.ZipFile(wheel_path, mode="w") as wheel:
        wheel.writestr(payload_name, b"payload")
        wheel.writestr(record_name, "," * 2048 + "\n")

    def reject_csv_parsing(*args, **kwargs):
        raise AssertionError("oversized RECORD row reached the CSV parser")

    monkeypatch.setattr(extension_wheel_module.csv, "reader", reject_csv_parsing)

    with zipfile.ZipFile(wheel_path) as wheel:
        with pytest.raises(ValueError, match="RECORD row exceeds its bounded maximum length"):
            extension_wheel_module._validate_wheel_record(
                wheel,
                names=wheel.namelist(),
                record_name=record_name,
            )


def test_clean_verifier_requires_exact_generated_wheel_metadata(tmp_path, synthetic_descriptor_factory):
    built = _build_sample_wheel(tmp_path)
    with zipfile.ZipFile(built.path) as wheel:
        wheel_metadata = next(name for name in wheel.namelist() if name.endswith(".dist-info/WHEEL"))
    tampered_directory = tmp_path / "tampered-wheel-version"
    tampered_directory.mkdir()
    tampered_wheel = _rewrite_wheel(
        built.path,
        tampered_directory / built.path.name,
        transforms={wheel_metadata: lambda contents: contents.replace(b"Wheel-Version: 1.0", b"Wheel-Version: 2.0")},
    )

    with pytest.raises(RuntimeError, match="exact generated WHEEL metadata"):
        verify_extension_wheel(
            base_wheel=_write_minimal_base_wheel(tmp_path),
            extension_wheel=tampered_wheel,
            extension_name="sample",
            trust_identity=TEST_TRUST_IDENTITY,
        )


@pytest.mark.parametrize(
    ("license_expression", "expected_message"),
    [
        (None, "exactly one License-Expression"),
        ("not-an-spdx-expression", "valid SPDX expression"),
        ("apache-2.0", "canonical SPDX syntax"),
    ],
)
def test_clean_verifier_requires_a_canonical_license_expression(
    tmp_path,
    synthetic_descriptor_factory,
    license_expression,
    expected_message,
):
    built = _build_sample_wheel(tmp_path)

    def transform(metadata: str) -> str:
        current = "License-Expression: Apache-2.0 AND MIT\n"
        replacement = "" if license_expression is None else f"License-Expression: {license_expression}\n"
        return metadata.replace(current, replacement)

    tampered_wheel = _rewrite_wheel_metadata(
        built.path,
        tmp_path / f"tampered-license-{license_expression or 'missing'}",
        transform,
    )

    with pytest.raises(RuntimeError, match=expected_message):
        verify_extension_wheel(
            base_wheel=_write_minimal_base_wheel(tmp_path),
            extension_wheel=tampered_wheel,
            extension_name="sample",
            trust_identity=TEST_TRUST_IDENTITY,
        )


@pytest.mark.parametrize("metadata_version", [None, "3.0"])
def test_clean_verifier_requires_the_generated_core_metadata_version(
    tmp_path,
    synthetic_descriptor_factory,
    metadata_version,
):
    built = _build_sample_wheel(tmp_path)

    def transform(metadata: str) -> str:
        replacement = "" if metadata_version is None else f"Metadata-Version: {metadata_version}\n"
        return metadata.replace("Metadata-Version: 2.4\n", replacement)

    tampered_wheel = _rewrite_wheel_metadata(
        built.path,
        tmp_path / f"tampered-metadata-{metadata_version or 'missing'}",
        transform,
    )

    with pytest.raises(RuntimeError, match="Metadata-Version must match the generated core metadata version exactly"):
        verify_extension_wheel(
            base_wheel=_write_minimal_base_wheel(tmp_path),
            extension_wheel=tampered_wheel,
            extension_name="sample",
            trust_identity=TEST_TRUST_IDENTITY,
        )


def test_clean_verifier_requires_the_exact_supported_python_range(tmp_path, synthetic_descriptor_factory):
    built = _build_sample_wheel(tmp_path)
    tampered_wheel = _rewrite_wheel_metadata(
        built.path,
        tmp_path / "tampered-python-range",
        lambda metadata: metadata.replace("Requires-Python: >=3.10,<3.15", "Requires-Python: >=3.11,<3.15"),
    )

    with pytest.raises(RuntimeError, match="Requires-Python must match the supported Python range exactly"):
        verify_extension_wheel(
            base_wheel=_write_minimal_base_wheel(tmp_path),
            extension_wheel=tampered_wheel,
            extension_name="sample",
            trust_identity=TEST_TRUST_IDENTITY,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "loose-vane",
        "loose-dependency",
        "missing-dependency",
        "wrong-dependency-version",
        "undeclared-dependency",
    ],
)
def test_clean_verifier_requires_exact_descriptor_bound_metadata(
    tmp_path,
    synthetic_descriptor_factory,
    mutation,
):
    dependency_built = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency"),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    root_built = _build_sample_wheel(tmp_path, dependencies=(dependency_built.path,))
    vane_requirement = f"Requires-Dist: vane-ai==={vane.__version__}"
    dependency_requirement = f"Requires-Dist: vane-extension-dependency==={dependency_built.distribution_version}"

    def transform(metadata: str) -> str:
        if mutation == "loose-vane":
            return metadata.replace(vane_requirement, vane_requirement.replace("===", "=="))
        if mutation == "loose-dependency":
            return metadata.replace(dependency_requirement, dependency_requirement.replace("===", "=="))
        if mutation == "missing-dependency":
            return metadata.replace(dependency_requirement + "\n", "")
        if mutation == "wrong-dependency-version":
            return metadata.replace(dependency_requirement, "Requires-Dist: vane-extension-dependency===0")
        assert mutation == "undeclared-dependency"
        return metadata.replace(
            dependency_requirement,
            dependency_requirement + "\nRequires-Dist: vane-extension-extra===1",
        )

    tampered_wheel = _rewrite_wheel_metadata(
        root_built.path,
        tmp_path / f"tampered-{mutation}",
        transform,
    )

    with pytest.raises(RuntimeError, match="Requires-Dist must match its descriptor exactly"):
        verify_extension_wheel(
            base_wheel=_write_minimal_base_wheel(tmp_path),
            extension_wheel=tampered_wheel,
            extension_name="sample",
            trust_identity=TEST_TRUST_IDENTITY,
            dependency_wheels=(dependency_built.path,),
            dependency_trust_identities=(TEST_TRUST_IDENTITY,),
        )


def test_clean_verifier_requires_the_exact_descriptor_dependency_wheel(
    tmp_path,
    synthetic_descriptor_factory,
):
    expected_dependency = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"expected"),
        extension_name="dependency",
        output_directory=tmp_path / "expected-dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    root_built = _build_sample_wheel(tmp_path, dependencies=(expected_dependency.path,))
    different_dependency = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"different"),
        extension_name="dependency",
        output_directory=tmp_path / "different-dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )

    with pytest.raises(RuntimeError, match="has no exact supplied wheel"):
        verify_extension_wheel(
            base_wheel=_write_minimal_base_wheel(tmp_path),
            extension_wheel=root_built.path,
            extension_name="sample",
            trust_identity=TEST_TRUST_IDENTITY,
            dependency_wheels=(different_dependency.path,),
            dependency_trust_identities=(TEST_TRUST_IDENTITY,),
        )


def test_clean_verifier_requires_explicit_dependency_signer_trust(
    tmp_path,
    synthetic_descriptor_factory,
):
    dependency = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency"),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    root = _build_sample_wheel(tmp_path, dependencies=(dependency.path,))

    with pytest.raises(RuntimeError, match="dependency trust identities must be supplied explicitly"):
        verify_extension_wheel(
            base_wheel=_write_minimal_base_wheel(tmp_path),
            extension_wheel=root.path,
            extension_name="sample",
            trust_identity=TEST_TRUST_IDENTITY,
            dependency_wheels=(dependency.path,),
        )


def test_clean_verifier_rejects_a_dependency_with_a_narrower_platform_policy(tmp_path, monkeypatch):
    def create_descriptor(path, *, name, trust_identity, dependencies):
        return _descriptor(
            path,
            name=name,
            trust_identity=trust_identity,
            platform="linux_amd64",
            dependencies=dependencies,
        )

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_descriptor)
    dependency = build_extension_wheel(
        artifact=_write_artifact(
            tmp_path / "dependency.duckdb_extension",
            b"dependency",
            architecture="x86_64",
        ),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag="manylinux_2_28_x86_64",
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    root = _build_sample_wheel(
        tmp_path,
        platform_tag="manylinux_2_28_x86_64",
        dependencies=(dependency.path,),
    )
    narrower_dependency = _relabel_wheel_platform(
        dependency.path,
        tmp_path / "narrower-policy",
        original="manylinux_2_28_x86_64",
        replacement="manylinux_2_39_x86_64",
    )

    with pytest.raises(RuntimeError, match="is not compatible with parent wheel platform tag"):
        verify_extension_wheel(
            base_wheel=_write_minimal_base_wheel(tmp_path, platform_tag="manylinux_2_28_x86_64"),
            extension_wheel=root.path,
            extension_name="sample",
            trust_identity=TEST_TRUST_IDENTITY,
            dependency_wheels=(narrower_dependency,),
            dependency_trust_identities=(TEST_TRUST_IDENTITY,),
        )


def test_clean_verifier_invokes_pip_in_isolated_mode(
    tmp_path,
    monkeypatch,
    synthetic_descriptor_factory,
):
    base_wheel = _write_minimal_base_wheel(tmp_path)
    extension_wheel = _build_sample_wheel(tmp_path).path
    commands: list[list[str]] = []
    environments: list[dict[str, str] | None] = []

    poisoned_variables = {
        "PYTHONHOME",
        "PYTHONMALLOC",
        "PYTHONOPTIMIZE",
        "PYTHONPATH",
        "PYTHONWARNINGS",
        "VIRTUAL_ENV",
        "__PYVENV_LAUNCHER__",
    }
    for variable in poisoned_variables:
        monkeypatch.setenv(variable, f"poisoned-{variable.casefold()}")
    monkeypatch.setenv("PIP_CONFIG_FILE", str(tmp_path / "hostile-pip.conf"))

    def record_command(command, *, cwd, environment=None):
        commands.append(command)
        environments.append(environment)

    monkeypatch.setattr(verify_extension_wheel_module, "_run", record_command)
    verify_extension_wheel(
        base_wheel=base_wheel,
        extension_wheel=extension_wheel,
        extension_name="sample",
        trust_identity=TEST_TRUST_IDENTITY,
    )

    assert len(commands) == 4
    assert commands[0][0] == sys.executable
    assert commands[0][1:6] == ["-I", "-m", "venv", "--clear", "--copies"]
    assert commands[1][1:4] == ["-m", "pip", "--isolated"]
    assert commands[2][1:5] == ["-m", "pip", "--isolated", "check"]
    for environment in environments:
        assert environment is not None
        assert not poisoned_variables & environment.keys()
        assert environment["PIP_CONFIG_FILE"] == os.devnull
        assert environment["PYTHONSAFEPATH"] == "1"


def test_clean_verifier_validates_and_installs_private_snapshots(
    tmp_path,
    monkeypatch,
    synthetic_descriptor_factory,
):
    dependency = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency"),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    root = _build_sample_wheel(tmp_path, dependencies=(dependency.path,))
    base = _write_minimal_base_wheel(tmp_path)
    source_paths = {base, root.path, dependency.path}
    recorded_snapshots: list[Path] = []
    installed_snapshots: list[Path] = []
    snapshot_archive = verify_extension_wheel_module.snapshot_archive

    @contextmanager
    def snapshot_then_replace(path, **kwargs):
        source_path = Path(path)
        with snapshot_archive(source_path, **kwargs) as snapshot:
            recorded_snapshots.append(snapshot.path)
            replacement = source_path.with_name(f"replacement-{source_path.name}")
            replacement.write_bytes(b"replacement after snapshot")
            replacement.replace(source_path)
            yield snapshot

    def inspect_command(command, *, cwd, environment=None):
        if "install" not in command:
            return
        installed_snapshots.extend(Path(argument) for argument in command if argument.endswith(".whl"))
        for snapshot_path in installed_snapshots:
            assert snapshot_path.is_file()
            assert snapshot_path.read_bytes() != b"replacement after snapshot"
            with zipfile.ZipFile(snapshot_path) as wheel:
                assert wheel.namelist()

    def reject_release_resnapshot(*_args, **_kwargs):
        raise AssertionError("base-wheel release validation must reuse the verifier snapshot")

    monkeypatch.setattr(verify_extension_wheel_module, "snapshot_archive", snapshot_then_replace)
    monkeypatch.setattr(check_release_artifacts, "snapshot_archive", reject_release_resnapshot)
    monkeypatch.setattr(verify_extension_wheel_module, "_run", inspect_command)

    verify_extension_wheel(
        base_wheel=base,
        extension_wheel=root.path,
        extension_name="sample",
        trust_identity=TEST_TRUST_IDENTITY,
        dependency_wheels=(dependency.path,),
        dependency_trust_identities=(TEST_TRUST_IDENTITY,),
    )

    assert len(recorded_snapshots) == 3
    assert set(installed_snapshots) == set(recorded_snapshots)
    assert all(path.read_bytes() == b"replacement after snapshot" for path in source_paths)
    assert all(not path.exists() and not path.parent.exists() for path in recorded_snapshots)


def test_clean_verifier_bounds_aggregate_snapshot_storage_and_cleans_prior_snapshots(
    tmp_path,
    monkeypatch,
    synthetic_descriptor_factory,
):
    dependency = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency"),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    root = _build_sample_wheel(tmp_path)
    base = _write_minimal_base_wheel(tmp_path)
    snapshot_attempts: list[tuple[Path, int, str]] = []
    retained_snapshot_paths: list[Path] = []
    snapshot_archive = verify_extension_wheel_module.snapshot_archive

    @contextmanager
    def record_snapshot(path, **kwargs):
        snapshot_attempts.append((Path(path), kwargs["max_bytes"], kwargs["size_limit_description"]))
        with snapshot_archive(path, **kwargs) as snapshot:
            retained_snapshot_paths.append(snapshot.path)
            yield snapshot

    monkeypatch.setattr(
        verify_extension_wheel_module,
        "_MAX_CLEAN_VERIFICATION_SNAPSHOT_BYTES",
        root.path.stat().st_size + base.stat().st_size,
    )
    monkeypatch.setattr(verify_extension_wheel_module, "snapshot_archive", record_snapshot)

    with pytest.raises(RuntimeError, match="aggregate 1 GiB snapshot limit"):
        verify_extension_wheel(
            base_wheel=base,
            extension_wheel=root.path,
            extension_name="sample",
            trust_identity=TEST_TRUST_IDENTITY,
            dependency_wheels=(dependency.path,),
            dependency_trust_identities=(TEST_TRUST_IDENTITY,),
        )

    assert [attempt[0] for attempt in snapshot_attempts] == [root.path, base, dependency.path]
    assert snapshot_attempts[-1][1:] == (0, "the clean verifier's aggregate 1 GiB snapshot limit")
    assert len(retained_snapshot_paths) == 2
    assert all(not path.exists() and not path.parent.exists() for path in retained_snapshot_paths)


def test_clean_verifier_validates_each_dependency_before_snapshotting_the_next(
    tmp_path,
    monkeypatch,
    synthetic_descriptor_factory,
):
    first_dependency = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency"),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    later_dependency = tmp_path / f"later-{first_dependency.path.name}"
    later_dependency.write_bytes(first_dependency.path.read_bytes())
    root = _build_sample_wheel(tmp_path)
    base = _write_minimal_base_wheel(tmp_path)
    attempted_sources: list[Path] = []
    retained_snapshot_paths: list[Path] = []
    snapshot_archive = verify_extension_wheel_module.snapshot_archive
    extension_name_from_artifact_path = verify_extension_wheel_module._extension_name_from_artifact_path

    @contextmanager
    def record_snapshot(path, **kwargs):
        attempted_sources.append(Path(path))
        with snapshot_archive(path, **kwargs) as snapshot:
            retained_snapshot_paths.append(snapshot.path)
            yield snapshot

    def reject_first_dependency(snapshot):
        if snapshot.source_path == first_dependency.path:
            raise RuntimeError("first dependency is invalid")
        return extension_name_from_artifact_path(snapshot)

    monkeypatch.setattr(verify_extension_wheel_module, "snapshot_archive", record_snapshot)
    monkeypatch.setattr(
        verify_extension_wheel_module,
        "_extension_name_from_artifact_path",
        reject_first_dependency,
    )

    with pytest.raises(RuntimeError, match="first dependency is invalid"):
        verify_extension_wheel(
            base_wheel=base,
            extension_wheel=root.path,
            extension_name="sample",
            trust_identity=TEST_TRUST_IDENTITY,
            dependency_wheels=(first_dependency.path, later_dependency),
            dependency_trust_identities=(TEST_TRUST_IDENTITY,),
        )

    assert attempted_sources == [root.path, base, first_dependency.path]
    assert later_dependency not in attempted_sources
    assert all(not path.exists() and not path.parent.exists() for path in retained_snapshot_paths)


def test_clean_verifier_removes_snapshots_when_environment_setup_fails(
    tmp_path,
    monkeypatch,
    synthetic_descriptor_factory,
):
    root = _build_sample_wheel(tmp_path)
    base = _write_minimal_base_wheel(tmp_path)
    recorded_snapshots: list[Path] = []
    snapshot_archive = verify_extension_wheel_module.snapshot_archive

    @contextmanager
    def record_snapshot(path, **kwargs):
        with snapshot_archive(path, **kwargs) as snapshot:
            recorded_snapshots.append(snapshot.path)
            yield snapshot

    def fail_environment_setup(*_args, **_kwargs):
        raise RuntimeError("environment setup failed")

    monkeypatch.setattr(verify_extension_wheel_module, "snapshot_archive", record_snapshot)
    monkeypatch.setattr(verify_extension_wheel_module, "_run", fail_environment_setup)

    with pytest.raises(RuntimeError, match="environment setup failed"):
        verify_extension_wheel(
            base_wheel=base,
            extension_wheel=root.path,
            extension_name="sample",
            trust_identity=TEST_TRUST_IDENTITY,
        )

    assert len(recorded_snapshots) == 2
    assert all(not path.exists() and not path.parent.exists() for path in recorded_snapshots)


def test_platform_wheel_rejects_an_artifact_changed_after_descriptor_creation(tmp_path, monkeypatch):
    artifact_path = _write_artifact(tmp_path / "sample.duckdb_extension")

    def create_then_replace(path, *, name, trust_identity, dependencies):
        descriptor = _descriptor(path, name=name, trust_identity=trust_identity, dependencies=dependencies)
        path.write_bytes(b"replacement")
        return descriptor

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_then_replace)
    with pytest.raises(RuntimeError, match="artifact changed"):
        _build_sample_wheel(tmp_path, artifact_path=artifact_path)


def test_platform_wheel_does_not_replace_a_different_existing_artifact(
    tmp_path,
    synthetic_descriptor_factory,
):
    artifact_path = _write_artifact(tmp_path / "sample.duckdb_extension")
    _build_sample_wheel(tmp_path, artifact_path=artifact_path)
    existing_wheel = next((tmp_path / "dist").glob("*.whl"))
    existing_wheel.write_bytes(b"different existing wheel")

    with pytest.raises(FileExistsError, match="refusing to replace"):
        _build_sample_wheel(tmp_path, artifact_path=artifact_path)

    assert existing_wheel.read_bytes() == b"different existing wheel"


@pytest.mark.skipif(os.name == "nt", reason="symbolic-link publication semantics differ on Windows")
def test_platform_wheel_does_not_reuse_an_identical_symbolic_link(
    tmp_path,
    synthetic_descriptor_factory,
):
    artifact_path = _write_artifact(tmp_path / "sample.duckdb_extension")
    source = _build_sample_wheel(tmp_path, artifact_path=artifact_path, output_name="source-dist")
    linked_directory = tmp_path / "linked-dist"
    linked_directory.mkdir()
    (linked_directory / source.path.name).symlink_to(source.path)

    with pytest.raises(FileExistsError, match="non-regular or multiply linked"):
        _build_sample_wheel(tmp_path, artifact_path=artifact_path, output_name="linked-dist")


@pytest.mark.skipif(os.name == "nt", reason="POSIX wheel modes are not available on Windows")
def test_platform_wheel_normalizes_permissions_on_an_identical_existing_artifact(
    tmp_path,
    synthetic_descriptor_factory,
):
    artifact_path = _write_artifact(tmp_path / "sample.duckdb_extension")
    first = _build_sample_wheel(tmp_path, artifact_path=artifact_path)
    first.path.chmod(0o600)

    second = _build_sample_wheel(tmp_path, artifact_path=artifact_path)

    assert second.path == first.path
    assert stat.S_IMODE(second.path.stat().st_mode) == 0o644


def test_platform_wheel_rejects_an_oversized_archive_before_publication(
    tmp_path,
    synthetic_descriptor_factory,
    monkeypatch,
):
    monkeypatch.setattr(extension_wheel_module, "_MAX_EXTENSION_WHEEL_BYTES", 0)

    with pytest.raises(ValueError, match="128 MiB publication limit"):
        _build_sample_wheel(tmp_path)

    assert not list((tmp_path / "dist").glob("*.whl"))


def test_platform_wheel_rejects_an_oversized_artifact_before_descriptor_inspection(
    tmp_path,
    monkeypatch,
):
    artifact_path = _write_artifact(tmp_path / "sample.duckdb_extension")
    monkeypatch.setattr(extension_wheel_module, "_MAX_EXTENSION_ARTIFACT_BYTES", 0)

    def reject_descriptor_inspection(*args, **kwargs):
        raise AssertionError("oversized artifact reached descriptor inspection")

    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", reject_descriptor_inspection)

    with pytest.raises(ValueError, match="extension artifact exceeds.*384 MiB extension-artifact limit"):
        _build_sample_wheel(tmp_path, artifact_path=artifact_path)


def test_platform_wheel_rechecks_the_artifact_bound_immediately_before_descriptor_inspection(
    tmp_path,
    monkeypatch,
):
    artifact_path = _write_artifact(tmp_path / "sample.duckdb_extension", b"initial")

    def expand_after_dependency_validation(_dependencies):
        artifact_path.write_bytes(b"expanded")

    def reject_descriptor_inspection(*args, **kwargs):
        raise AssertionError("expanded artifact reached descriptor inspection")

    monkeypatch.setattr(extension_wheel_module, "_MAX_EXTENSION_ARTIFACT_BYTES", len(b"initial"))
    monkeypatch.setattr(
        extension_wheel_module,
        "_validate_dependency_wheel_requirements",
        expand_after_dependency_validation,
    )
    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", reject_descriptor_inspection)

    with pytest.raises(ValueError, match="extension artifact exceeds.*384 MiB extension-artifact limit"):
        _build_sample_wheel(tmp_path, artifact_path=artifact_path)


def test_platform_wheel_rechecks_the_artifact_bound_after_descriptor_inspection(tmp_path, monkeypatch):
    artifact_path = _write_artifact(tmp_path / "sample.duckdb_extension", b"initial")

    def create_then_expand(path, *, name, trust_identity, dependencies):
        descriptor = _descriptor(
            path,
            name=name,
            trust_identity=trust_identity,
            dependencies=dependencies,
        )
        path.write_bytes(b"expanded")
        return descriptor

    monkeypatch.setattr(extension_wheel_module, "_MAX_EXTENSION_ARTIFACT_BYTES", len(b"initial"))
    monkeypatch.setattr(extension_wheel_module, "_create_descriptor", create_then_expand)

    with pytest.raises(ValueError, match="extension artifact exceeds.*384 MiB extension-artifact limit"):
        _build_sample_wheel(tmp_path, artifact_path=artifact_path)


def test_platform_wheel_rejects_an_oversized_license_before_reading_it(
    tmp_path,
    synthetic_descriptor_factory,
    monkeypatch,
):
    artifact_path = _write_artifact(tmp_path / "sample.duckdb_extension")
    monkeypatch.setattr(
        extension_wheel_module,
        "_MAX_EXTENSION_WHEEL_MEMBER_BYTES",
        0,
    )

    with pytest.raises(ValueError, match="license file exceeds.*384 MiB per-member uncompressed limit"):
        _build_sample_wheel(tmp_path, artifact_path=artifact_path)


def test_platform_wheel_rejects_oversized_decompressed_contents_before_publication(
    tmp_path,
    synthetic_descriptor_factory,
    monkeypatch,
):
    monkeypatch.setattr(extension_wheel_module, "_MAX_EXTENSION_WHEEL_UNCOMPRESSED_BYTES", 0)

    with pytest.raises(ValueError, match="512 MiB total uncompressed limit"):
        _build_sample_wheel(tmp_path)

    assert not list((tmp_path / "dist").glob("*.whl"))


def test_platform_wheel_rejects_oversized_aggregate_contents_before_creating_archive(
    tmp_path,
    synthetic_descriptor_factory,
    monkeypatch,
):
    artifact_path = _write_artifact(tmp_path / "sample.duckdb_extension", b"a" * 4096)
    license_path = tmp_path / "LICENSE"
    license_path.write_bytes(b"l" * 4096)
    monkeypatch.setattr(extension_wheel_module, "_MAX_EXTENSION_WHEEL_UNCOMPRESSED_BYTES", 6000)

    def reject_archive_creation(*args, **kwargs):
        raise AssertionError("oversized aggregate contents reached temporary archive creation")

    monkeypatch.setattr(extension_wheel_module.tempfile, "mkstemp", reject_archive_creation)

    with pytest.raises(ValueError, match="decompressed contents exceed.*512 MiB total uncompressed limit"):
        build_extension_wheel(
            artifact=artifact_path,
            extension_name="sample",
            output_directory=tmp_path / "dist",
            platform_tag=_wheel_platform_tag(),
            trust_identity=TEST_TRUST_IDENTITY,
            license_expression="Apache-2.0",
            license_files=[license_path],
        )

    assert not (tmp_path / "dist").exists()


def test_platform_wheel_rejects_oversized_decompressed_dependency_before_reading_members(
    tmp_path,
    synthetic_descriptor_factory,
    monkeypatch,
):
    dependency = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency"),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    root_artifact = _write_artifact(tmp_path / "sample.duckdb_extension", b"root")

    def require_size_check_before_read(wheel, *args, **kwargs):
        raise AssertionError("oversized dependency member was read before validation")

    monkeypatch.setattr(
        extension_wheel_module,
        "_MAX_EXTENSION_WHEEL_MEMBER_BYTES",
        root_artifact.stat().st_size,
    )
    monkeypatch.setattr(extension_wheel_module.zipfile.ZipFile, "read", require_size_check_before_read)

    with pytest.raises(ValueError, match="dependency extension wheel.*384 MiB per-member uncompressed limit"):
        _build_sample_wheel(
            tmp_path,
            artifact_path=root_artifact,
            output_name="root-dist",
            dependencies=(dependency.path,),
        )


def test_platform_wheel_rejects_too_many_dependency_members_before_opening_the_wheel(
    tmp_path,
    synthetic_descriptor_factory,
    monkeypatch,
):
    dependency = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency"),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    open_wheel = extension_wheel_module.zipfile.ZipFile

    def require_member_count_check_before_open(path, *args, **kwargs):
        if Path(path) == dependency.path:
            raise AssertionError("dependency wheel was opened before its member-count preflight")
        return open_wheel(path, *args, **kwargs)

    monkeypatch.setattr(extension_wheel_module, "_MAX_EXTENSION_WHEEL_MEMBERS", 1)
    monkeypatch.setattr(extension_wheel_module.zipfile, "ZipFile", require_member_count_check_before_open)

    with pytest.raises(ValueError, match="dependency extension wheel contains more than 1 archive members"):
        _build_sample_wheel(tmp_path, output_name="root-dist", dependencies=(dependency.path,))


def test_platform_wheel_dependency_preflight_and_parser_use_the_same_snapshot(
    tmp_path,
    synthetic_descriptor_factory,
    monkeypatch,
):
    dependency = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency"),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    validate_snapshot = archive_safety_module.validate_zip_member_count
    replacement = dependency.path.with_name(f"replacement-{dependency.path.name}")
    replacement.write_bytes(b"replacement after preflight")
    replaced = False

    def validate_then_replace(*args, **kwargs):
        nonlocal replaced
        member_count = validate_snapshot(*args, **kwargs)
        if kwargs["description"] == "dependency extension wheel" and not replaced:
            replacement.replace(dependency.path)
            replaced = True
        return member_count

    monkeypatch.setattr(archive_safety_module, "validate_zip_member_count", validate_then_replace)

    built = _build_sample_wheel(tmp_path, output_name="root-dist", dependencies=(dependency.path,))

    assert built.path.is_file()
    assert replaced
    assert dependency.path.read_bytes() == b"replacement after preflight"


def test_platform_wheel_bounds_dependency_wheel_count_before_reading_wheels(
    tmp_path,
    synthetic_descriptor_factory,
    monkeypatch,
):
    dependency = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency"),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )

    def reject_dependency_wheel_read(_path):
        raise AssertionError("dependency wheel was read before the graph-size bound")

    monkeypatch.setattr(extension_wheel_module, "_MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES", 0)
    monkeypatch.setattr(extension_wheel_module, "_read_dependency_wheel", reject_dependency_wheel_read)

    with pytest.raises(ValueError, match="dependency_wheels contains more than 0 wheel paths"):
        _build_sample_wheel(tmp_path, output_name="root-dist", dependencies=(dependency.path,))


def test_platform_wheel_bounds_dependency_descriptor_before_json_parsing(
    tmp_path,
    synthetic_descriptor_factory,
    monkeypatch,
):
    dependency = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency"),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    with zipfile.ZipFile(dependency.path) as wheel:
        descriptor_member = next(name for name in wheel.namelist() if name.endswith(".dynamic-extension.json"))
    tampered_directory = tmp_path / "oversized-dependency-descriptor"
    tampered_directory.mkdir()
    oversized_descriptor = b'{"unknown":"' + b"x" * extension_wheel_module._MAX_EXTENSION_DESCRIPTOR_BYTES + b'"}\n'
    tampered_dependency = _rewrite_wheel(
        dependency.path,
        tampered_directory / dependency.path.name,
        transforms={descriptor_member: lambda _contents: oversized_descriptor},
    )
    read_member = extension_wheel_module.zipfile.ZipFile.read

    def reject_whole_descriptor_read(wheel, member, *args, **kwargs):
        member_name = member.filename if isinstance(member, zipfile.ZipInfo) else member
        if member_name == descriptor_member:
            raise AssertionError("oversized dependency descriptor was whole-read before its byte bound")
        return read_member(wheel, member, *args, **kwargs)

    def reject_descriptor_parsing(_cls, _value):
        raise AssertionError("dependency descriptor reached JSON object parsing before its byte bound")

    monkeypatch.setattr(extension_wheel_module.zipfile.ZipFile, "read", reject_whole_descriptor_read)
    monkeypatch.setattr(DynamicExtensionDescriptor, "from_json", classmethod(reject_descriptor_parsing))

    with pytest.raises(ValueError, match="bounded 64 KiB descriptor limit"):
        _build_sample_wheel(
            tmp_path,
            output_name="root-dist",
            dependencies=(tampered_dependency,),
        )


@pytest.mark.parametrize(
    ("contents", "limit_name", "expected_message"),
    [
        (b'{"a":"b"}', "_MAX_EXTENSION_DESCRIPTOR_JSON_STRINGS", "more than 1 JSON strings"),
        (
            b'{"dependencies":[{}]}',
            "_MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES",
            "more than 0 nested dependency objects",
        ),
    ],
    ids=["string-count", "dependency-object-count"],
)
def test_extension_descriptor_structure_is_bounded_before_json_parsing(
    monkeypatch,
    contents,
    limit_name,
    expected_message,
):
    monkeypatch.setattr(extension_wheel_module, limit_name, 1 if limit_name.endswith("STRINGS") else 0)

    with pytest.raises(ValueError, match=expected_message):
        extension_wheel_module._validate_extension_descriptor_json_bounds(
            contents,
            description="test descriptor",
        )


def test_extension_descriptor_requires_ascii_before_json_parsing():
    with pytest.raises(ValueError, match="canonical ASCII JSON encoding"):
        extension_wheel_module._validate_extension_descriptor_json_bounds(
            b'\xff{"dependencies":[]}',
            description="test descriptor",
        )


def test_platform_wheel_rejects_too_many_generated_members_before_creating_record(
    tmp_path,
    synthetic_descriptor_factory,
    monkeypatch,
):
    def reject_record_creation(*args, **kwargs):
        raise AssertionError("generated wheel RECORD was created before its member-count validation")

    monkeypatch.setattr(extension_wheel_module, "_MAX_EXTENSION_WHEEL_MEMBERS", 9)
    monkeypatch.setattr(extension_wheel_module, "_record", reject_record_creation)

    with pytest.raises(ValueError, match="generated extension wheel contains more than 9 archive members"):
        _build_sample_wheel(tmp_path)

    assert not (tmp_path / "dist").exists()


def test_platform_wheel_rejects_oversized_total_decompressed_dependency_contents(
    tmp_path,
    synthetic_descriptor_factory,
    monkeypatch,
):
    dependency = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency"),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    with zipfile.ZipFile(dependency.path) as wheel:
        member_sizes = tuple(member.file_size for member in wheel.infolist())
    largest_member = max(member_sizes)
    assert sum(member_sizes) > largest_member
    monkeypatch.setattr(
        extension_wheel_module,
        "_MAX_EXTENSION_WHEEL_UNCOMPRESSED_BYTES",
        largest_member,
    )

    with pytest.raises(ValueError, match="decompressed contents exceed.*512 MiB total uncompressed limit"):
        _build_sample_wheel(tmp_path, output_name="root-dist", dependencies=(dependency.path,))


def test_clean_verifier_rejects_an_oversized_extension_wheel(
    tmp_path,
    synthetic_descriptor_factory,
    monkeypatch,
):
    built = _build_sample_wheel(tmp_path)
    snapshot_archive = verify_extension_wheel_module.snapshot_archive

    @contextmanager
    def reject_extension(path, **kwargs):
        if kwargs["description"] == "extension wheel":
            raise ValueError("extension wheel exceeds the project's 128 MiB publication limit")
        with snapshot_archive(path, **kwargs) as snapshot:
            yield snapshot

    monkeypatch.setattr(verify_extension_wheel_module, "snapshot_archive", reject_extension)

    with pytest.raises(RuntimeError, match="128 MiB publication limit"):
        verify_extension_wheel(
            base_wheel=_write_minimal_base_wheel(tmp_path),
            extension_wheel=built.path,
            extension_name="sample",
            trust_identity=TEST_TRUST_IDENTITY,
        )


def test_clean_verifier_bounds_dependency_wheel_count_before_reading_wheels(
    tmp_path,
    synthetic_descriptor_factory,
    monkeypatch,
):
    root = _build_sample_wheel(tmp_path)
    dependency = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency"),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )

    def reject_extension_wheel_layout(*_args, **_kwargs):
        raise AssertionError("extension wheel was read before the dependency graph-size bound")

    monkeypatch.setattr(verify_extension_wheel_module, "_MAX_EXTENSION_DESCRIPTOR_DEPENDENCIES", 0)
    monkeypatch.setattr(verify_extension_wheel_module, "_assert_extension_wheel_layout", reject_extension_wheel_layout)

    with pytest.raises(RuntimeError, match="dependency wheels contain more than 0 wheel paths"):
        verify_extension_wheel(
            base_wheel=_write_minimal_base_wheel(tmp_path),
            extension_wheel=root.path,
            extension_name="sample",
            trust_identity=TEST_TRUST_IDENTITY,
            dependency_wheels=(dependency.path,),
            dependency_trust_identities=(TEST_TRUST_IDENTITY,),
        )


def test_clean_verifier_bounds_descriptor_before_json_parsing(
    tmp_path,
    synthetic_descriptor_factory,
    monkeypatch,
):
    built = _build_sample_wheel(tmp_path)
    with zipfile.ZipFile(built.path) as wheel:
        descriptor_member = next(name for name in wheel.namelist() if name.endswith(".dynamic-extension.json"))
    tampered_directory = tmp_path / "oversized-root-descriptor"
    tampered_directory.mkdir()
    oversized_descriptor = b'{"unknown":"' + b"x" * extension_wheel_module._MAX_EXTENSION_DESCRIPTOR_BYTES + b'"}\n'
    tampered_wheel = _rewrite_wheel(
        built.path,
        tampered_directory / built.path.name,
        transforms={descriptor_member: lambda _contents: oversized_descriptor},
    )
    read_member = verify_extension_wheel_module.zipfile.ZipFile.read

    def reject_whole_descriptor_read(wheel, member, *args, **kwargs):
        member_name = member.filename if isinstance(member, zipfile.ZipInfo) else member
        if member_name == descriptor_member:
            raise AssertionError("oversized root descriptor was whole-read before its byte bound")
        return read_member(wheel, member, *args, **kwargs)

    def reject_json_parsing(_value):
        raise AssertionError("root descriptor reached JSON object parsing before its byte bound")

    monkeypatch.setattr(verify_extension_wheel_module.zipfile.ZipFile, "read", reject_whole_descriptor_read)
    monkeypatch.setattr(verify_extension_wheel_module.json, "loads", reject_json_parsing)

    with pytest.raises(RuntimeError, match="bounded 64 KiB descriptor limit"):
        _extension_name_from_wheel(tampered_wheel)


def test_clean_verifier_rejects_oversized_decompressed_extension_contents(
    tmp_path,
    synthetic_descriptor_factory,
    monkeypatch,
):
    built = _build_sample_wheel(tmp_path)
    monkeypatch.setattr(extension_wheel_module, "_MAX_EXTENSION_WHEEL_UNCOMPRESSED_BYTES", 0)

    with pytest.raises(RuntimeError, match="512 MiB total uncompressed limit"):
        verify_extension_wheel(
            base_wheel=_write_minimal_base_wheel(tmp_path),
            extension_wheel=built.path,
            extension_name="sample",
            trust_identity=TEST_TRUST_IDENTITY,
        )


def test_clean_verifier_rejects_an_oversized_base_wheel_before_opening_it(
    tmp_path,
    synthetic_descriptor_factory,
    monkeypatch,
):
    root = _build_sample_wheel(tmp_path)
    base_wheel = _write_minimal_base_wheel(tmp_path)
    snapshot_archive = verify_extension_wheel_module.snapshot_archive

    @contextmanager
    def reject_base(path, **kwargs):
        if kwargs["description"] == "base Vane wheel":
            raise ValueError("base Vane wheel exceeds the project's 128 MiB publication limit")
        with snapshot_archive(path, **kwargs) as snapshot:
            yield snapshot

    monkeypatch.setattr(verify_extension_wheel_module, "snapshot_archive", reject_base)

    with pytest.raises(RuntimeError, match="base Vane wheel exceeds.*128 MiB publication limit"):
        verify_extension_wheel(
            base_wheel=base_wheel,
            extension_wheel=root.path,
            extension_name="sample",
            trust_identity=TEST_TRUST_IDENTITY,
        )


def test_clean_verifier_rejects_an_oversized_dependency_before_opening_it(
    tmp_path,
    synthetic_descriptor_factory,
    monkeypatch,
):
    root = _build_sample_wheel(tmp_path)
    dependency = build_extension_wheel(
        artifact=_write_artifact(tmp_path / "dependency.duckdb_extension", b"dependency"),
        extension_name="dependency",
        output_directory=tmp_path / "dependency-dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=TEST_TRUST_IDENTITY,
        license_expression="Apache-2.0",
        license_files=[REPOSITORY_ROOT / "LICENSE"],
    )
    snapshot_archive = verify_extension_wheel_module.snapshot_archive

    @contextmanager
    def reject_dependency(path, **kwargs):
        if kwargs["description"] == "dependency extension wheel":
            raise ValueError("dependency extension wheel exceeds the project's 128 MiB publication limit")
        with snapshot_archive(path, **kwargs) as snapshot:
            yield snapshot

    monkeypatch.setattr(verify_extension_wheel_module, "snapshot_archive", reject_dependency)

    with pytest.raises(RuntimeError, match="128 MiB publication limit"):
        verify_extension_wheel(
            base_wheel=_write_minimal_base_wheel(tmp_path),
            extension_wheel=root.path,
            extension_name="sample",
            trust_identity=TEST_TRUST_IDENTITY,
            dependency_wheels=(dependency.path,),
            dependency_trust_identities=(TEST_TRUST_IDENTITY,),
        )


def test_platform_wheel_uses_distinct_versions_for_changed_artifacts(
    tmp_path,
    synthetic_descriptor_factory,
):
    artifact_path = _write_artifact(tmp_path / "sample.duckdb_extension")
    first = _build_sample_wheel(tmp_path, artifact_path=artifact_path)
    first_contents = first.path.read_bytes()
    _write_artifact(artifact_path, b"changed extension wheel test payload")

    second = _build_sample_wheel(tmp_path, artifact_path=artifact_path)

    assert second.path != first.path
    assert second.distribution_version != first.distribution_version
    assert first.path.read_bytes() == first_contents


def test_installed_platform_wheel_resolves_a_signed_artifact_in_a_clean_environment(tmp_path):
    staged_artifact = os.environ.get("VANE_TEST_SIGNED_DYNAMIC_EXTENSION_PATH")
    base_wheel = os.environ.get("VANE_TEST_BASE_WHEEL")
    if staged_artifact is None or base_wheel is None:
        pytest.skip("set VANE_TEST_SIGNED_DYNAMIC_EXTENSION_PATH and VANE_TEST_BASE_WHEEL for clean validation")

    extension_name = "loadable_extension_demo"
    trust_identity = "vane-ci-test-key"
    built = build_extension_wheel(
        artifact=Path(staged_artifact),
        extension_name=extension_name,
        output_directory=tmp_path / "dist",
        platform_tag=_wheel_platform_tag(),
        trust_identity=trust_identity,
        license_expression="Apache-2.0 AND MIT",
        license_files=[
            REPOSITORY_ROOT / "LICENSE",
            REPOSITORY_ROOT / "NOTICE",
            REPOSITORY_ROOT / "external" / "duckdb" / "LICENSE",
        ],
    )

    verify_extension_wheel(
        base_wheel=Path(base_wheel),
        extension_wheel=built.path,
        extension_name=extension_name,
        trust_identity=trust_identity,
    )

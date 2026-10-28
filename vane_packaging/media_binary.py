# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Platform-specific media library relocation; never load inspected libraries."""

from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path

from vane_packaging import extension_wheel as binary


def macho_linkage(contents: bytes) -> tuple[str | None, tuple[str, ...], tuple[str, ...]]:
    """Read names after the caller has validated the bounded Mach-O commands."""
    if len(contents) < binary._MACHO_HEADER_64.size or not contents.startswith(binary._MACHO_MAGIC_64):
        raise ValueError("media library must contain thin 64-bit Mach-O data")
    header = binary._MACHO_HEADER_64.unpack_from(contents)
    count, size = header[4:6]
    offset = binary._MACHO_HEADER_64.size
    end = offset + size
    if count > binary._MAX_MACHO_LOAD_COMMANDS or end > len(contents):
        raise ValueError("media Mach-O load commands exceed their bounds")
    identity = None
    dependencies = []
    rpaths = []
    for _ in range(count):
        if offset + 8 > end:
            raise ValueError("truncated media Mach-O command")
        command, length = struct.unpack_from("<II", contents, offset)
        if length < 8 or length % 8 or offset + length > end:
            raise ValueError("invalid media Mach-O command size")
        if command in binary._MACHO_DYLIB_COMMANDS or command == 0xD:
            minimum = binary._MACHO_DYLIB_COMMAND.size
        elif command == binary._MACHO_RPATH:
            minimum = binary._MACHO_RPATH_COMMAND.size
        else:
            offset += length
            continue
        if length < minimum:
            raise ValueError("truncated media Mach-O name command")
        name = binary._macho_load_command_string(
            contents,
            command_offset=offset,
            command_size=length,
            string_offset=struct.unpack_from("<I", contents, offset + 8)[0],
            minimum_string_offset=minimum,
            description="media Mach-O",
            field_description="library name",
        )
        if command == 0xD:
            if identity is not None:
                raise ValueError("duplicate media Mach-O install name")
            identity = name
        elif command == binary._MACHO_RPATH:
            rpaths.append(name)
        else:
            dependencies.append(name)
        offset += length
    if offset != end:
        raise ValueError("media Mach-O command sizes differ from the header")
    return identity, tuple(dependencies), tuple(rpaths)


def validate_non_elf_graph(libraries: dict[str, bytes], platform: str) -> dict[str, tuple[str, ...]]:
    family = binary._wheel_platform_policy(platform).family
    names = frozenset(libraries)
    result = {}
    for name, contents in libraries.items():
        if family == "windows":
            if name != name.casefold() or not name.endswith(".dll"):
                raise ValueError("media DLL names must be canonical lowercase filenames")
            result[name] = binary._validate_windows_pe_platform(
                contents, platform, description=name, interpreter_tag="cp312", bundled_libraries=names
            )
            if any(item.startswith("python") for item in result[name]):
                raise ValueError("media runtime libraries must not depend on Python")
        elif family == "macosx":
            binary._validate_macos_binary_platform(contents, platform, description=name, bundled_libraries=names)
            identity, dependencies, _ = macho_linkage(contents)
            if identity != f"@loader_path/{name}" or not name.endswith(".dylib"):
                raise ValueError("media dylib install name must match its namespaced filename")
            result[name] = tuple(Path(item).name for item in dependencies)
        else:
            raise ValueError("unsupported native media binary platform")
    return result


def stage_non_elf(prefix: Path, destination: Path, *, namespace: str, platform: str) -> dict[str, str]:
    family = binary._wheel_platform_policy(platform).family
    if family == "windows":
        candidates = sorted((prefix / "bin").glob("*.dll"))
    elif family == "macosx":
        candidates = sorted((prefix / "lib").glob("*.dylib"))
    else:
        raise ValueError("unsupported media library platform")
    sources = {}
    for candidate in candidates:
        if candidate.is_symlink():
            continue
        if family == "windows":
            name = candidate.name.casefold()
        else:
            identity, _, _ = macho_linkage(candidate.read_bytes())
            if identity is None:
                raise ValueError("media SDK dylib is missing its install name")
            name = Path(identity).name
        if name in sources or not 0 < candidate.stat().st_size <= 128 * 1024**2:
            raise ValueError("duplicate or oversized media library")
        if family == "windows" and name in binary._windows_policy_external_libraries("cp312"):
            raise ValueError("media SDK must not bundle Windows system libraries")
        sources[name] = candidate
    if not 1 <= len(sources) <= 128:
        raise ValueError("media SDK has no libraries or too many libraries")
    mapping = {name: f"{namespace}_{name}" for name in sources}
    binary._bundled_library_names(mapping.values())
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for name, source in sources.items():
            output = destination / mapping[name]
            shutil.copyfile(source, output)
            relocate_non_elf(output, mapping, platform, library_name=mapping[name])
        validate_non_elf_graph({path.name: path.read_bytes() for path in destination.iterdir()}, platform)
    except BaseException:
        shutil.rmtree(destination)
        raise
    return mapping


def relocate_non_elf(target: Path, mapping: dict[str, str], platform: str, *, library_name: str | None = None) -> None:
    """Rewrite only declared codec dependencies, preserving all import slots."""
    family = binary._wheel_platform_policy(platform).family
    if family == "windows":
        import lief

        # Parse a byte snapshot so no file-backed parser view prevents replacing
        # the input DLL on Windows after its imports have been rewritten.
        image = lief.PE.parse(target.read_bytes())
        if image is None:
            raise ValueError("media DLL could not be parsed for relocation")
        names = {name.casefold(): replacement for name, replacement in mapping.items()}
        for entry in image.delay_imports:
            if entry.name.casefold() in names:
                raise ValueError("namespacing delayed media imports is not supported")
        for entry in image.imports:
            if entry.name.casefold() in names:
                entry.name = names[entry.name.casefold()]
        config = lief.PE.Builder.config_t()
        config.imports = True
        output = target.with_name(target.name + ".relocated")
        try:
            image.write(str(output), config)
            output.replace(target)
        finally:
            output.unlink(missing_ok=True)
    elif family == "macosx":
        _, dependencies, rpaths = macho_linkage(target.read_bytes())
        arguments = ["install_name_tool"]
        if library_name is not None:
            arguments.extend(("-id", f"@loader_path/{library_name}"))
        prefix = "@loader_path" if library_name is not None else "@loader_path/.libs"
        for name in dependencies:
            replacement = mapping.get(Path(name).name)
            if replacement is not None:
                arguments.extend(("-change", name, f"{prefix}/{replacement}"))
        for rpath in rpaths:
            arguments.extend(("-delete_rpath", rpath))
        if len(arguments) > 1:
            subprocess.run([*arguments, str(target)], check=True, timeout=60)
        subprocess.run(["codesign", "--force", "--sign", "-", str(target)], check=True, timeout=60)
    else:
        raise ValueError("unsupported native media relocation platform")

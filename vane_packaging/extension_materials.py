# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Deliver source and relinking materials with LGPL extension wheels.

These checks establish completeness of the declared inventory and byte
identities, not legal approval or semantic equivalence of source and binaries.
Validation never executes a supplied build script or extracts a source archive.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

from packaging.licenses import canonicalize_license_expression

MANIFEST_NAME = "vane-extension-materials.json"
MATERIALS_DIRECTORY = "vane-extension-materials"
PRIVATE_CLASSIFIER = "Private :: Do Not Upload"
MAX_MANIFEST_BYTES = 64 * 1024
MAX_MATERIAL_BYTES = 128 * 1024 * 1024
MAX_MATERIALS_BYTES = 256 * 1024 * 1024
MAX_MATERIAL_FILES = 256
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PATH_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_RESERVED_NAMES = {"aux", "con", "conin$", "conout$", "nul", "prn"} | {
    f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10)
}
_NATIVE_LIBRARIES = {
    "image": {"ffmpeg"},
    "video": {"ffmpeg"},
    "audio": {"ffmpeg", "libsndfile", "soxr", "mpg123", "mp3lame"},
}
_INVENTORY_KEYS = {
    "libraries",
    "application",
    "materials_license_expression",
    "build_instructions",
    "relink_instructions",
    "relink_verification",
}
_MANIFEST_KEYS = _INVENTORY_KEYS | {
    "schema_version",
    "extension_name",
    "artifact_sha256",
    "license_expression",
    "files",
}


def needs_materials(name: str, license_expression: str) -> bool:
    """Treat every declared LGPL alternative conservatively as requiring materials."""
    return name in _NATIVE_LIBRARIES or bool(re.search(r"\bLGPL-", license_expression))


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate materials JSON key: {key!r}")
        result[key] = value
    return result


def parse_json(contents: bytes) -> dict[str, Any]:
    if len(contents) > MAX_MANIFEST_BYTES:
        raise ValueError("extension materials manifest exceeds 64 KiB")
    try:
        result = json.loads(contents, object_pairs_hook=_object)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ValueError("invalid extension materials JSON") from error
    if not isinstance(result, dict):
        raise ValueError("extension materials JSON must be an object")
    return result


def _text(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512 or any(ord(c) < 32 or ord(c) > 126 for c in value):
        raise ValueError(f"invalid extension materials {description}")
    return value


def _path(value: Any) -> str:
    value = _text(value, "file path")
    for part in value.split("/"):
        if not _PATH_PART.fullmatch(part) or part.endswith(".") or part.split(".", 1)[0].lower() in _RESERVED_NAMES:
            raise ValueError(f"unsafe extension materials path: {value!r}")
    return value


def _paths(value: Any, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or len(value) > MAX_MATERIAL_FILES or (not value and not allow_empty):
        raise ValueError("extension materials require bounded file lists")
    paths = [_path(item) for item in value]
    if len(set(paths)) != len(paths):
        raise ValueError("extension materials file lists must not repeat paths")
    return paths


def _mandatory_license_atoms(expression: str) -> set[str]:
    """Find licenses required by every alternative in a canonical SPDX expression."""
    if len(expression) > 4096:
        raise ValueError("extension materials wheel license expression exceeds 4096 characters")
    tokens = re.findall(r"[A-Za-z0-9.+-]+(?: WITH [A-Za-z0-9.+-]+)?|[()]", expression)
    values: list[set[str]] = []
    operators: list[str] = []
    precedence = {"OR": 1, "AND": 2}

    def reduce_operator() -> None:
        right, left = values.pop(), values.pop()
        values.append(left | right if operators.pop() == "AND" else left & right)

    for token in tokens:
        if token == "(":
            operators.append(token)
        elif token == ")":
            while operators[-1] != "(":
                reduce_operator()
            operators.pop()
        elif token in precedence:
            while operators and operators[-1] != "(" and precedence[operators[-1]] >= precedence[token]:
                reduce_operator()
            operators.append(token)
        else:
            values.append({token})
    while operators:
        reduce_operator()
    return values[0]


def inventory_paths(inventory: dict[str, Any], *, name: str, license_expression: str) -> set[str]:
    if set(inventory) != _INVENTORY_KEYS:
        raise ValueError("extension materials inventory has missing or unknown fields")
    material_expression = canonicalize_license_expression(
        _text(inventory["materials_license_expression"], "materials license expression")
    )
    atoms = r"[A-Za-z0-9.+-]+(?: WITH [A-Za-z0-9.+-]+)?"
    material_licenses = set(re.findall(atoms, material_expression)) - {"AND", "OR"}
    mandatory_material_licenses = _mandatory_license_atoms(material_expression)
    wheel_licenses = _mandatory_license_atoms(canonicalize_license_expression(license_expression))
    if not material_licenses <= wheel_licenses:
        raise ValueError("extension wheel License-Expression must include its source/build materials licenses")
    paths = set(_paths(inventory["application"]))
    documents = {_path(inventory[key]) for key in ("build_instructions", "relink_instructions", "relink_verification")}
    if len(documents) != 3:
        raise ValueError("build instructions, relink instructions and verification must use distinct files")
    libraries = inventory["libraries"]
    if not isinstance(libraries, list) or not 1 <= len(libraries) <= 64:
        raise ValueError("extension materials require 1..64 library source records")
    names: set[str] = set()
    for library in libraries:
        if not isinstance(library, dict) or set(library) != {
            "name",
            "version",
            "license",
            "source",
            "build_recipe",
            "patches",
        }:
            raise ValueError("invalid extension materials library record")
        library_name = _text(library["name"], "library name")
        if not re.fullmatch(r"[a-z][a-z0-9-]*", library_name) or library_name in names:
            raise ValueError("extension materials library names must be unique and canonical")
        names.add(library_name)
        _text(library["version"], "library version")
        library_license = canonicalize_license_expression(_text(library["license"], "library license"))
        if not re.fullmatch(r"LGPL-(?:2\.0|2\.1|3\.0)-(?:only|or-later)", library_license):
            raise ValueError("extension materials library records must identify their LGPL license")
        if library_license not in wheel_licenses:
            raise ValueError("extension wheel License-Expression must include its LGPL dependencies")
        if library_license not in mandatory_material_licenses:
            raise ValueError("materials_license_expression must include its LGPL library sources")
        paths.add(_path(library["source"]))
        paths.add(_path(library["build_recipe"]))
        paths.update(_paths(library["patches"], allow_empty=True))
    missing = _NATIVE_LIBRARIES.get(name, set()) - names
    if missing:
        raise ValueError(f"extension materials are missing LGPL libraries: {sorted(missing)}")
    # Code archives may contain multiple libraries, application code, recipes,
    # and patches. Instructions/evidence must be independent deliverables.
    if documents & paths:
        raise ValueError("extension materials instructions/evidence must not alias source or recipe files")
    paths.update(documents)
    if len(paths) > MAX_MATERIAL_FILES or len({p.casefold() for p in paths}) != len(paths):
        raise ValueError("extension materials require at most 256 distinct file paths, including case")
    folded = {p.casefold() for p in paths}
    for path in folded:
        if any("/".join(path.split("/")[:i]) in folded for i in range(1, len(path.split("/")))):
            raise ValueError("extension materials contain a file/parent conflict")
    return paths


def encode_manifest(manifest: dict[str, Any]) -> bytes:
    result = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if len(result) > MAX_MANIFEST_BYTES:
        raise ValueError("extension materials manifest exceeds 64 KiB")
    return result


def prepare_manifest(
    inventory: dict[str, Any],
    read_file: Callable[[str], bytes],
    *,
    name: str,
    artifact_sha256: str,
    license_expression: str,
) -> bytes:
    """Hash an explicitly reviewed inventory; the caller supplies completed relink evidence."""
    if not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", name) or not _SHA256.fullmatch(artifact_sha256):
        raise ValueError("invalid extension materials artifact identity")
    expression = canonicalize_license_expression(license_expression)
    paths = inventory_paths(inventory, name=name, license_expression=expression)
    files = {}
    total = 0
    for path in sorted(paths):
        contents = read_file(path)
        total += len(contents)
        if not 0 < len(contents) <= MAX_MATERIAL_BYTES or total > MAX_MATERIALS_BYTES:
            raise ValueError("extension materials exceed their file/aggregate size budget or contain an empty file")
        files[path] = {"sha256": hashlib.sha256(contents).hexdigest(), "size": len(contents)}
    return encode_manifest(
        {
            **inventory,
            "schema_version": 1,
            "extension_name": name,
            "artifact_sha256": artifact_sha256,
            "license_expression": expression,
            "files": files,
        }
    )


def validate_materials(
    contents: bytes,
    read_file: Callable[[str], bytes],
    *,
    name: str,
    artifact_sha256: str,
    license_expression: str,
) -> dict[str, bytes]:
    manifest = parse_json(contents)
    if (
        set(manifest) != _MANIFEST_KEYS
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
    ):
        raise ValueError("unsupported extension materials manifest schema")
    if (
        manifest["extension_name"] != name
        or manifest["artifact_sha256"] != artifact_sha256
        or not _SHA256.fullmatch(artifact_sha256)
        or manifest["license_expression"] != license_expression
    ):
        raise ValueError("extension materials do not match the artifact identity and license expression")
    inventory = {key: manifest[key] for key in _INVENTORY_KEYS}
    paths = inventory_paths(inventory, name=name, license_expression=license_expression)
    records = manifest["files"]
    if not isinstance(records, dict) or set(records) != paths:
        raise ValueError("extension materials file inventory does not match its declared roles")
    if contents != encode_manifest(manifest):
        raise ValueError("extension materials manifest must use canonical JSON")
    total = 0
    for path, record in records.items():
        if (
            not isinstance(record, dict)
            or set(record) != {"size", "sha256"}
            or type(record["size"]) is not int
            or not 0 < record["size"] <= MAX_MATERIAL_BYTES
            or not isinstance(record["sha256"], str)
            or not _SHA256.fullmatch(record["sha256"])
        ):
            raise ValueError(f"invalid extension materials file record: {path}")
        total += record["size"]
    if total > MAX_MATERIALS_BYTES:
        raise ValueError("extension materials exceed their aggregate size budget")
    result = {}
    for path, record in records.items():
        value = read_file(path)
        if len(value) != record["size"] or hashlib.sha256(value).hexdigest() != record["sha256"]:
            raise ValueError(f"extension materials file size/SHA-256 mismatch: {path}")
        result[path] = value
    return result


def material_path(directory: Path, name: str) -> Path:
    """Resolve only declared relative regular files inside the materials directory."""
    path = directory / _path(name)
    resolved_directory = directory.resolve(strict=True)
    cursor = path
    while cursor != directory:
        if cursor.is_symlink():
            raise ValueError(f"extension materials must not use symlinks: {name}")
        cursor = cursor.parent
    if not path.resolve(strict=True).is_relative_to(resolved_directory):
        raise ValueError(f"extension materials escape their directory: {name}")
    return path


def read_material_file(directory: Path, name: str, *, max_bytes: int = MAX_MATERIAL_BYTES) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        path = material_path(directory, name)
        with os.fdopen(os.open(path, flags), "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= max_bytes:
                raise ValueError("extension material must be a nonempty bounded regular file")
            contents = source.read(max_bytes + 1)
    except OSError as error:
        raise ValueError(f"cannot read extension material: {name}") from error
    if len(contents) > max_bytes:
        raise ValueError("extension material exceeds its bounded maximum size")
    return contents

# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded, deterministic media-runtime documents; importing this loads no native code.

The standalone media source distribution includes this module as build support.
Keep it independent of Vane and optional Python packages.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

DISTRIBUTION = "vane-media-runtime"
PACKAGE = "vane_media_runtime"
MANIFEST = "runtime-manifest.json"
SIGNATURE = "runtime-manifest.sig"
SIGNING_DOMAIN = b"VANE_NATIVE_RUNTIME_MANIFEST_V1\x00"
TRAILER_MAGIC = b"VANE_NATIVE_RUNTIME_V1\x00"
TRAILER_SIZE = len(TRAILER_MAGIC) + 64
DUCKDB_FOOTER_SIZE = 512
MAX_MANIFEST_BYTES = 64 * 1024
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_FILES = 128
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}\Z")
_VERSION = re.compile(r"[0-9]+(?:\.[0-9]+)+(?:a[0-9]+|b[0-9]+|rc[0-9]+)?(?:\.post[0-9]+)?(?:\.dev[0-9]+)?\Z")
_MANIFEST_KEYS = {
    "schema_version",
    "distribution",
    "version",
    "git_commit",
    "git_dirty",
    "vane_version",
    "platform",
    "namespace",
    "license_expression",
    "source",
    "components",
    "files",
}


def digest(value: object) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError("native runtime requires a lowercase SHA-256 digest")
    return value


def filename(value: object) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value) or value.endswith("."):
        raise ValueError("native runtime requires a bounded relative filename")
    if value.split(".", 1)[0].casefold() in {
        "con",
        "aux",
        "nul",
        "prn",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }:
        raise ValueError("native runtime filename is reserved")
    return value


def version(value: object) -> str:
    if not isinstance(value, str) or len(value) > 200 or not _VERSION.fullmatch(value):
        raise ValueError("native runtime version must be a canonical public numeric release")
    if any(len(part) > 1 and part.startswith("0") for part in re.findall(r"[0-9]+", value)):
        raise ValueError("native runtime version must not contain leading zeroes")
    return value


def git_commit(commit: object) -> str:
    """Validate the complete Git provenance carried by an exported source SDK."""
    if not isinstance(commit, str) or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit) is None:
        raise ValueError("native runtime requires a full lowercase Git commit hash")
    return commit


def _text(value: object, maximum: int = 1024) -> str:
    if not isinstance(value, str) or not 0 < len(value) <= maximum or any(not 32 <= ord(c) < 127 for c in value):
        raise ValueError("invalid native runtime text")
    return value


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate native runtime JSON key")
        result[key] = value
    return result


def parse_json(contents: bytes) -> dict[str, Any]:
    if not 0 < len(contents) <= MAX_MANIFEST_BYTES:
        raise ValueError("native runtime manifest exceeds its size bound")
    try:
        value = json.loads(contents, object_pairs_hook=_object)
        if not isinstance(value, dict) or canonical_json(value) != contents:
            raise ValueError("native runtime manifest must be a canonical JSON object")
        return value
    except (UnicodeError, RecursionError, OverflowError) as exc:
        raise ValueError("invalid native runtime JSON") from exc


def validate_reference(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {
        "distribution",
        "version",
        "manifest_sha256",
    }:
        raise ValueError("invalid native runtime reference")
    if value["distribution"] != DISTRIBUTION:
        raise ValueError("unsupported native runtime distribution")
    return {
        "distribution": DISTRIBUTION,
        "version": version(value["version"]),
        "manifest_sha256": digest(value["manifest_sha256"]),
    }


def parse_manifest(contents: bytes) -> dict[str, Any]:
    value = parse_json(contents)
    if set(value) != _MANIFEST_KEYS or type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported native runtime manifest schema")
    if value["distribution"] != DISTRIBUTION:
        raise ValueError("unsupported native runtime distribution")
    release = version(value["version"])
    git_commit(value["git_commit"])
    version(value["vane_version"])
    if type(value["git_dirty"]) is not bool:
        raise ValueError("invalid native runtime Git identity")
    if not re.fullmatch(
        r"manylinux_(?:0|[1-9][0-9]*)_(?:0|[1-9][0-9]*)_x86_64",
        _text(value["platform"], 80),
    ):
        raise ValueError("native runtime currently supports manylinux x86-64 only")
    if re.fullmatch(r"vane_media_[a-z0-9_]{1,64}", filename(value["namespace"])) is None:
        raise ValueError("invalid native runtime namespace")
    if value["namespace"] != "vane_media_" + value["git_commit"]:
        raise ValueError("native runtime namespace differs from its Git identity")
    _text(value["license_expression"], 4096)
    source = value["source"]
    if not isinstance(source, dict) or set(source) != {"filename", "sha256", "url"}:
        raise ValueError("invalid native runtime source reference")
    if source["filename"] != f"vane_media_runtime-{release}.tar.gz":
        raise ValueError("native runtime source filename does not match its version")
    digest(source["sha256"])
    source_url = source["url"]
    if not isinstance(source_url, str) or not 1 <= len(source_url) <= 2048 or not source_url.isascii():
        raise ValueError("native runtime source URL must be bounded ASCII text")
    if any(character.isspace() or ord(character) < 32 for character in source_url):
        raise ValueError("native runtime source URL contains whitespace or control characters")
    location = urlsplit(source_url)
    if location.scheme != "https" or not location.hostname or location.username or location.password or location.port:
        raise ValueError("native runtime source URL must use HTTPS without credentials or a custom port")
    components = value["components"]
    if not isinstance(components, dict) or not 1 <= len(components) <= MAX_FILES:
        raise ValueError("native runtime requires a bounded component inventory")
    for name, component in components.items():
        filename(name)
        if not isinstance(component, dict) or set(component) != {
            "version",
            "license",
            "notice_sha256",
        }:
            raise ValueError("invalid native runtime component")
        _text(component["version"], 128)
        _text(component["license"], 512)
        digest(component["notice_sha256"])
    files = value["files"]
    if not isinstance(files, dict) or not 1 <= len(files) <= MAX_FILES:
        raise ValueError("native runtime requires a bounded library inventory")
    if len({filename(name).casefold() for name in files}) != len(files):
        raise ValueError("native runtime filenames collide")
    total = 0
    for name, record in files.items():
        filename(name)
        if not name.startswith(value["namespace"] + "_"):
            raise ValueError("native runtime filename does not use its declared namespace")
        if not isinstance(record, dict) or set(record) != {
            "sha256",
            "size",
            "needed",
            "component",
        }:
            raise ValueError("invalid native runtime library record")
        digest(record["sha256"])
        size = record["size"]
        if type(size) is not int or not 0 < size <= MAX_FILE_BYTES:
            raise ValueError("native runtime library exceeds its size bound")
        total += size
        needed = record["needed"]
        if not isinstance(needed, list) or len(needed) > MAX_FILES:
            raise ValueError("invalid native runtime dependencies")
        if len(set(map(filename, needed))) != len(needed) or name in needed:
            raise ValueError("duplicate or self-referencing native runtime dependency")
        if filename(record["component"]) not in components:
            raise ValueError("native runtime library has an unknown source component")
    if total > MAX_TOTAL_BYTES:
        raise ValueError("native runtime exceeds its total size bound")
    return value


def reference(contents: bytes) -> dict[str, str]:
    value = parse_manifest(contents)
    return {
        "distribution": DISTRIBUTION,
        "version": value["version"],
        "manifest_sha256": hashlib.sha256(contents).hexdigest(),
    }


def read_file(directory: Path, name: str, maximum: int = MAX_FILE_BYTES) -> bytes:
    name = filename(name)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(directory / name, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
            raise ValueError("native runtime files must be bounded nonempty regular files")
        value = stream.read(maximum + 1)
        if len(value) != info.st_size or len(value) > maximum:
            raise ValueError("native runtime file changed during reading")
        return value


def verify_files(directory: Path, manifest: dict[str, Any]) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("native runtime libraries must be a regular directory")
    if set(item.name for item in directory.iterdir()) != set(manifest["files"]):
        raise ValueError("native runtime library directory differs from its manifest")
    for name, record in manifest["files"].items():
        contents = read_file(directory, name)
        if len(contents) != record["size"] or hashlib.sha256(contents).hexdigest() != record["sha256"]:
            raise ValueError(f"native runtime library digest mismatch: {name}")


def trailer_digest(contents: bytes) -> str | None:
    if len(contents) < DUCKDB_FOOTER_SIZE + TRAILER_SIZE:
        return None
    trailer = contents[-DUCKDB_FOOTER_SIZE - TRAILER_SIZE : -DUCKDB_FOOTER_SIZE]
    if not trailer.startswith(TRAILER_MAGIC):
        return None
    try:
        return digest(trailer[len(TRAILER_MAGIC) :].decode("ascii"))
    except UnicodeError as exc:
        raise ValueError("invalid native runtime extension trailer") from exc


def attach_trailer(contents: bytes, manifest_sha256: str) -> bytes:
    digest(manifest_sha256)
    if len(contents) < DUCKDB_FOOTER_SIZE:
        raise ValueError("native runtime requires a DuckDB extension footer")
    if contents[-256:] != bytes(256):
        raise ValueError("attach native runtime metadata before signing the extension")
    if trailer_digest(contents) is not None:
        contents = contents[: -DUCKDB_FOOTER_SIZE - TRAILER_SIZE] + contents[-DUCKDB_FOOTER_SIZE:]
    return (
        contents[:-DUCKDB_FOOTER_SIZE]
        + TRAILER_MAGIC
        + manifest_sha256.encode("ascii")
        + contents[-DUCKDB_FOOTER_SIZE:]
    )

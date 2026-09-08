# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Check reviewed GPL-family/SSPL source notices and the supported native profile.

This is a change detector for an explicit inventory, not a license detector or
a legal compatibility decision. Original notices must remain intact.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path

NOTICE_PATH = "LICENSES/Bison-parser-notice.txt"
POLICY_PATH = "LICENSES/copyleft-review.json"
_MARKER = re.compile(
    rb"\b(?:A?GPL|LGPL|SSPL)(?:[- +v]|\b)|GNU\s+(?:Lesser\s+|Library\s+|Affero\s+)?General\s+Public\s+License"
    rb"|Server\s+Side\s+Public\s+License",
    re.I,
)
_SOURCE_ROOTS = ("external/duckdb/", "src/", "vane/", "cmake/", "scripts/", "vane_packaging/")
_SOURCE_FILES = {
    "build_backend.py",
    "CMakeLists.txt",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY.md",
    "external/duckdb/tools/CMakeLists.txt",
    "external/duckdb/tools/utils/test_platform.cpp",
    "external/duckdb/scripts/append_metadata.cmake",
    "external/duckdb/scripts/null.txt",
}
_EXCLUDED_ROOTS = (
    "external/duckdb/extension/tpch/",
    "external/duckdb/extension/tpcds/",
    "external/duckdb/third_party/tpce-tool/",
    "external/duckdb/test/",
    "external/duckdb/data/",
    "external/duckdb/benchmark/",
    "external/duckdb/tools/",
    "external/duckdb/scripts/",
)
_FFMPEG_FEATURES = {"avcodec", "avformat", "swscale", "swresample", "zlib"}
_FORBIDDEN_PORTS = {"x264", "x265", "xvidcore", "libx264", "libx265", "libxvid"}


def source_candidate(path: str) -> bool:
    """Select release source and license records; exclude test/data fixtures."""
    if path == POLICY_PATH:
        # This trusted input cannot hash itself. Sdist validation compares its
        # complete bytes against the reviewed checkout before scanning sources.
        return False
    if path in _SOURCE_FILES:
        return True
    if path.startswith(_EXCLUDED_ROOTS):
        return False
    return path.startswith(("LICENSES/", *_SOURCE_ROOTS))


def has_copyleft_marker(contents: bytes) -> bool:
    return bool(_MARKER.search(contents))


def check_source_inventory(files: Iterable[tuple[str, bytes]], reviewed: dict[str, str]) -> None:
    actual = {
        path: hashlib.sha256(contents).hexdigest()
        for path, contents in files
        if source_candidate(path) and has_copyleft_marker(contents)
    }
    changed = sorted(path for path in actual.keys() | reviewed.keys() if actual.get(path) != reviewed.get(path))
    if changed:
        raise ValueError(f"GPL-family source inventory needs review: {changed}")


def check_native_manifest(manifest: dict) -> None:
    """Keep optional FFmpeg defaults and GPL/nonfree feature additions explicit."""
    groups = [manifest.get("dependencies", [])]
    groups.extend(feature.get("dependencies", []) for feature in manifest.get("features", {}).values())
    for dependencies in groups:
        for dependency in dependencies:
            name = dependency if isinstance(dependency, str) else dependency["name"]
            if name in _FORBIDDEN_PORTS:
                raise ValueError(f"unsupported GPL codec in native release profile: {name}")
            if name == "ffmpeg":
                if isinstance(dependency, str) or dependency.get("default-features") is not False:
                    raise ValueError("native FFmpeg must explicitly disable default features")
                unknown = set(dependency.get("features", [])) - _FFMPEG_FEATURES
                if unknown:
                    raise ValueError(f"native FFmpeg features need license review: {sorted(unknown)}")


def expected_installed_notices(
    manifest: dict, selected_features: Iterable[str], dependency_notices: dict[str, list[str]]
) -> set[str]:
    """Resolve reviewed notice requirements from explicitly selected dependencies."""
    groups = [manifest.get("dependencies", [])]
    for feature in selected_features:
        if feature not in manifest.get("features", {}):
            raise ValueError(f"unknown native dependency feature: {feature}")
        groups.append(manifest["features"][feature].get("dependencies", []))
    required = set()
    for dependencies in groups:
        for dependency in dependencies:
            name = dependency if isinstance(dependency, str) else dependency["name"]
            required.update(dependency_notices.get(name, []))
    return required


def _check_installed_version(directory: Path, name: str, expected_version: str) -> None:
    """Check the vcpkg port version, including its optional #port-revision."""
    metadata_path = directory / "vcpkg.spdx.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"installed dependency version metadata needs review: {name}") from exc
    packages = metadata.get("packages") if isinstance(metadata, dict) else None
    ports = (
        [item for item in packages if isinstance(item, dict) and item.get("SPDXID") == "SPDXRef-port"]
        if isinstance(packages, list)
        else []
    )
    if len(ports) != 1 or ports[0].get("name") != name or ports[0].get("versionInfo") != expected_version:
        raise ValueError(f"installed dependency version needs review: {name}; expected {expected_version}")


def check_installed_notices(
    share_dir: Path, reviewed: dict[str, dict[str, str]], *, expected: Iterable[str]
) -> list[str]:
    """Reject unreviewed GPL-family records in the installed dependency graph."""
    required = set(expected)
    if unknown := required - reviewed.keys():
        raise ValueError(f"expected dependency notices have no review: {sorted(unknown)}")
    records = sorted(share_dir.glob("*/copyright"))
    if not records:
        raise ValueError(f"no installed dependency copyright records below {share_dir}")
    checked = []
    for path in records:
        name = path.parent.name
        if name.startswith("vcpkg-"):
            continue  # Build-only ports are not redistributed with Vane.
        contents = path.read_bytes()
        if name not in reviewed and not has_copyleft_marker(contents):
            continue
        record = reviewed.get(name)
        if record is None or hashlib.sha256(contents).hexdigest() != record["copyright_sha256"]:
            raise ValueError(f"installed GPL-family dependency notice needs review: {name}")
        _check_installed_version(path.parent, name, record["version"])
        checked.append(name)
    if missing := required - set(checked):
        raise ValueError(f"missing expected dependency copyright records: {sorted(missing)}")
    return checked


def load_policy(root: Path) -> dict:
    return json.loads((root / POLICY_PATH).read_text(encoding="utf-8"))

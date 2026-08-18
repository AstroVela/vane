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
import os
import re
import stat
import tempfile
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from packaging.licenses import InvalidLicenseExpression, canonicalize_license_expression

if TYPE_CHECKING:
    from vane.extensions import DynamicExtensionDescriptor

ENTRY_POINT_GROUP = "vane.dynamic_extension_providers"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_EXTENSION_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_WHEEL_PLATFORM_TAG_RE = re.compile(r"^[a-z0-9_]+$")
_WHEEL_VERSION_RE = re.compile(r"^[A-Za-z0-9.]+$")
_ARTIFACT_PLATFORM_TAGS = {
    "linux_amd64": re.compile(r"^(?:linux_x86_64|manylinux_[0-9]+_[0-9]+_x86_64)$"),
    "linux_amd64_musl": re.compile(r"^musllinux_[0-9]+_[0-9]+_x86_64$"),
    "linux_arm64": re.compile(r"^(?:linux_aarch64|manylinux_[0-9]+_[0-9]+_aarch64)$"),
    "linux_arm64_musl": re.compile(r"^musllinux_[0-9]+_[0-9]+_aarch64$"),
    "osx_amd64": re.compile(r"^macosx_[0-9]+_[0-9]+_x86_64$"),
    "osx_arm64": re.compile(r"^macosx_[0-9]+_[0-9]+_arm64$"),
    "windows_amd64": re.compile(r"^win_amd64$"),
    "windows_arm64": re.compile(r"^win_arm64$"),
}


@dataclass(frozen=True)
class BuiltExtensionWheel:
    """The generated wheel and the descriptor embedded in it."""

    path: Path
    descriptor: DynamicExtensionDescriptor
    distribution_name: str
    wheel_tag: str


def build_extension_wheel(
    *,
    artifact: str | Path,
    extension_name: str,
    output_directory: str | Path,
    platform_tag: str,
    trust_identity: str,
    license_expression: str,
    license_files: Iterable[str | Path],
) -> BuiltExtensionWheel:
    """Build one platform-specific wheel from an already-built local artifact.

    ``artifact`` must be the exact, self-contained
    ``<extension_name>.duckdb_extension`` emitted by the Vane build.
    ``license_files`` must explicitly cover that artifact's redistributed
    source and binary dependencies, and ``license_expression`` must be the
    corresponding SPDX expression. The descriptor is generated from the
    artifact using the currently installed Vane runtime, which pins the wheel
    to its Vane and DuckDB identities.
    """
    name = _validate_extension_name(extension_name)
    normalized_platform_tag = _validate_platform_tag(platform_tag)
    normalized_license_expression = _validate_license_expression(license_expression)
    artifact_path = Path(artifact).expanduser().resolve()
    expected_artifact_name = f"{name}.duckdb_extension"
    if artifact_path.name != expected_artifact_name:
        raise ValueError(f"artifact must be named {expected_artifact_name!r}: {artifact_path}")
    import vane
    from vane.extensions import create_dynamic_extension_descriptor

    descriptor = create_dynamic_extension_descriptor(
        artifact_path,
        name=name,
        trust_identity=trust_identity,
    )
    if descriptor.duckdb_source_id != vane.__git_revision__:
        raise RuntimeError(
            f"artifact SourceID {descriptor.duckdb_source_id} does not match installed Vane runtime "
            f"SourceID {vane.__git_revision__}"
        )
    _validate_artifact_platform_tag(descriptor.platform, normalized_platform_tag)
    artifact_contents = artifact_path.read_bytes()
    if hashlib.sha256(artifact_contents).hexdigest() != descriptor.sha256:
        raise RuntimeError(f"artifact changed while creating its extension wheel: {artifact_path}")

    vane_version = _validate_wheel_version(vane.__version__)
    distribution_name = f"vane-extension-{name}"
    distribution_root = f"vane_extension_{name}-{vane_version}"
    wheel_tag = f"py3-none-{normalized_platform_tag}"
    output_path = Path(output_directory).expanduser().resolve() / f"{distribution_root}-{wheel_tag}.whl"
    package_root = f"vane_extensions/{name}"
    descriptor_name = f"{package_root}/{name}.dynamic-extension.json"
    artifact_name = f"{package_root}/{name}.duckdb_extension"
    dist_info_root = f"{distribution_root}.dist-info"
    license_entries = _license_entries(license_files, dist_info_root)

    entries = {
        f"{package_root}/__init__.py": _provider_module_source(name).encode("utf-8"),
        descriptor_name: (descriptor.to_json() + "\n").encode("utf-8"),
        artifact_name: artifact_contents,
        f"{dist_info_root}/METADATA": _metadata(
            distribution_name,
            vane_version,
            normalized_license_expression,
            tuple(sorted(license_entries)),
            dist_info_root,
        ).encode("utf-8"),
        f"{dist_info_root}/WHEEL": _wheel_metadata(wheel_tag).encode("utf-8"),
        f"{dist_info_root}/entry_points.txt": _entry_points(name).encode("utf-8"),
    }
    entries.update(license_entries)
    record_name = f"{dist_info_root}/RECORD"
    entries[record_name] = _record(entries, record_name).encode("utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor_handle, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(descriptor_handle)
    temporary_path = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary_path, mode="w", compression=zipfile.ZIP_DEFLATED) as wheel:
            for member_name, contents in sorted(entries.items()):
                wheel.writestr(_zip_info(member_name), contents)
        try:
            os.link(temporary_path, output_path)
        except FileExistsError:
            if output_path.read_bytes() != temporary_path.read_bytes():
                raise FileExistsError(
                    f"refusing to replace a different extension wheel at {output_path}; "
                    "choose a new output directory or version"
                ) from None
    finally:
        temporary_path.unlink(missing_ok=True)

    return BuiltExtensionWheel(
        path=output_path,
        descriptor=descriptor,
        distribution_name=distribution_name,
        wheel_tag=wheel_tag,
    )


def _validate_extension_name(value: str) -> str:
    if _EXTENSION_NAME_RE.fullmatch(value) is None:
        raise ValueError("extension_name must contain lowercase ASCII letters, digits, and underscores")
    return value


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


def _validate_wheel_version(value: str) -> str:
    if _WHEEL_VERSION_RE.fullmatch(value) is None:
        raise RuntimeError(f"installed Vane version cannot be used in a wheel filename: {value!r}")
    return value


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
    vane_version: str,
    license_expression: str,
    license_members: tuple[str, ...],
    dist_info_root: str,
) -> str:
    lines = [
        "Metadata-Version: 2.4",
        f"Name: {distribution_name}",
        f"Version: {vane_version}",
        "Summary: Platform-specific Vane dynamic extension artifact",
        f"License-Expression: {license_expression}",
        "Requires-Python: >=3.10",
        f"Requires-Dist: vane-ai (=={vane_version})",
    ]
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


def _entry_points(name: str) -> str:
    return "\n".join(
        (
            f"[{ENTRY_POINT_GROUP}]",
            f"{name} = vane_extensions.{name}:provider",
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


def _license_entries(license_files: Iterable[str | Path], dist_info_root: str) -> dict[str, bytes]:
    entries: dict[str, bytes] = {}
    for license_file in license_files:
        path = Path(license_file).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"license file is not a regular file: {path}")
        relative_path = _license_member_path(path)
        member_name = f"{dist_info_root}/licenses/{relative_path}"
        if member_name in entries:
            raise ValueError(f"license files must have unique wheel paths: {relative_path}")
        entries[member_name] = path.read_bytes()
    if not entries:
        raise ValueError("license_files must contain every license required by the extension artifact")
    return entries


def _license_member_path(path: Path) -> str:
    try:
        relative_path = path.relative_to(REPOSITORY_ROOT)
    except ValueError:
        relative_path = Path(path.name)
    member_path = relative_path.as_posix()
    if (
        not member_path
        or "\\" in member_path
        or any(part in {"", ".", ".."} for part in relative_path.parts)
        or any(ord(character) < 32 or ord(character) > 126 for character in member_path)
    ):
        raise ValueError(f"license file must have a safe ASCII relative path: {path}")
    return member_path


def _zip_info(member_name: str) -> zipfile.ZipInfo:
    source_date_epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "315532800"))
    timestamp = datetime.fromtimestamp(max(source_date_epoch, 315532800), tz=timezone.utc)
    info = zipfile.ZipInfo(member_name, date_time=timestamp.timetuple()[:6])
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


__all__ = ["ENTRY_POINT_GROUP", "BuiltExtensionWheel", "build_extension_wheel"]

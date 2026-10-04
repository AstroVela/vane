# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bundle verified media libraries into their extension's single installable wheel."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from pathlib import Path

from vane_packaging.archive_safety import open_zip_snapshot, snapshot_archive
from vane_packaging.artifact_limits import MAX_PUBLICATION_FILE_BYTES, PUBLICATION_FILE_LIMIT_DESCRIPTION
from vane_packaging.extension_wheel import (
    _MAX_EXTENSION_WHEEL_MEMBERS,
    _read_bounded_wheel_member,
    _validate_metadata_license_expression,
)
from vane_packaging.media_runtime import (
    PROJECT_NOTICES,
    _read_runtime_wheel_snapshot,
    runtime_license_expression,
    validate_library_graph,
)
from vane_packaging.media_version import identity_version, runtime_format


def validate_runtime_graph(references: Iterable[dict[str, str] | None]) -> None:
    """All media extensions in one load graph must select the same signed runtime."""
    selected = None
    for reference in references:
        if reference is None:
            continue
        runtime_format().validate_reference(reference)
        if selected is not None and reference != selected:
            raise ValueError("extension graph must use the same exact native media runtime")
        selected = reference


def validate_bundled_metadata(metadata, manifest) -> None:
    """Keep source discovery and binary license metadata bound to the signed runtime."""
    from packaging.licenses import canonicalize_license_expression

    source_url = manifest["source"]["url"]
    if metadata.get_all("Project-URL", []) != [f"Native media corresponding sources, {source_url}"]:
        raise ValueError("provider metadata must expose its signed corresponding-source URL")
    expression = _validate_metadata_license_expression(metadata)
    suffix = f" AND ({manifest['license_expression']})"
    if not expression.endswith(suffix):
        raise ValueError("provider license expression must include its bundled runtime licenses")
    provider = expression[: -len(suffix)]
    if not (provider.startswith("(") and provider.endswith(")")) or (
        canonicalize_license_expression(provider[1:-1]) != provider[1:-1]
    ):
        raise ValueError("provider license expression must preserve the generated runtime conjunction")


def read_build_runtime(path: Path, *, test_only: bool = False):
    """Read one private build input; keep verification and copied bytes on one snapshot."""
    fmt = runtime_format()
    with snapshot_archive(
        path, max_bytes=100 * 1024 * 1024, description="build runtime", size_limit_description="100 MiB"
    ) as snapshot:
        info = _read_runtime_wheel_snapshot(snapshot, test_only=test_only)
        _, manifest, libraries, document, signature = info
        payload = {fmt.MANIFEST: document, fmt.SIGNATURE: signature}
        payload.update({f".libs/{name}": contents for name, contents in libraries.items()})
        notice_names = (*PROJECT_NOTICES, *(f"{name}.txt" for name in manifest["components"]))
        with open_zip_snapshot(snapshot, max_members=512, description="build runtime") as wheel:
            root = f"vane_media_runtime-{manifest['version']}.dist-info/licenses"
            notices = {name: wheel.read(f"{root}/{name}") for name in notice_names}
    return info, payload, notices


def read_bundled_runtime(
    wheel,
    *,
    package_root: str,
    dist_info_root: str,
    reference: dict,
    platform: str,
    signature_verifier: Callable[[bytes, bytes], bool] | None,
    test_only: bool = False,
):
    """Check manifest data, every bundled library, its ELF graph and notices.

    The caller also validates the complete provider layout, RECORD, metadata,
    descriptor and native signature. Returned member names extend that exact
    owned layout; arbitrary extra files are never accepted as runtime content.

    Builders pass their installed Vane's native signature verifier. Data-only
    inspection must explicitly pass None; it does not authenticate the manifest.
    Clean verification authenticates the returned bytes with the supplied base
    wheel in its isolated environment before executing any provider or extension.
    """
    fmt = runtime_format()
    fmt.validate_reference(reference)
    prefix = f"{package_root}/runtime"
    license_prefix = f"{dist_info_root}.dist-info/licenses/runtime"

    def read(name: str, limit: int) -> bytes:
        return _read_bounded_wheel_member(wheel, name, max_bytes=limit, description="bundled native media runtime")

    document = read(f"{prefix}/{fmt.MANIFEST}", fmt.MAX_MANIFEST_BYTES)
    manifest = fmt.parse_manifest(document)
    if fmt.reference(document) != reference:
        raise ValueError("bundled media runtime differs from the extension's exact reference")
    if manifest["platform"] != platform:
        raise ValueError("extension and media runtime must use the same platform policy")
    if manifest["version"] != identity_version(manifest):
        raise ValueError("bundled media runtime version differs from its Git identity")
    if manifest["git_dirty"] and not test_only:
        raise ValueError("development runtime Git snapshots cannot be released")
    if manifest["license_expression"] != runtime_license_expression(manifest["components"]):
        raise ValueError("runtime license expression must cover the project and native components")
    signature = read(f"{prefix}/{fmt.SIGNATURE}", 256)
    if len(signature) != 256:
        raise ValueError("invalid bundled runtime manifest signature length")
    if signature_verifier is not None and not signature_verifier(document, signature):
        raise ValueError("bundled media runtime manifest signature is not trusted by the build runtime")
    notice_hashes = {
        **PROJECT_NOTICES,
        **{f"{name}.txt": r["notice_sha256"] for name, r in manifest["components"].items()},
    }
    notice_members = tuple(f"{license_prefix}/{name}" for name in notice_hashes)
    for name, digest in notice_hashes.items():
        if hashlib.sha256(read(f"{license_prefix}/{name}", 1024 * 1024)).hexdigest() != digest:
            raise ValueError(f"bundled runtime license notice digest differs: {name}")
    libraries = {}
    for name, record in manifest["files"].items():
        contents = read(f"{prefix}/.libs/{name}", fmt.MAX_FILE_BYTES)
        if len(contents) != record["size"] or hashlib.sha256(contents).hexdigest() != record["sha256"]:
            raise ValueError(f"bundled runtime library digest differs: {name}")
        libraries[name] = contents
    graph = validate_library_graph(libraries, platform)
    for name, needed in graph.items():
        if list(needed) != manifest["files"][name]["needed"]:
            raise ValueError(f"bundled runtime dependency graph differs from ELF metadata: {name}")
    members = (f"{prefix}/{fmt.MANIFEST}", f"{prefix}/{fmt.SIGNATURE}", *(f"{prefix}/.libs/{n}" for n in libraries))
    return (reference, manifest, libraries, document, signature), members, notice_members


def read_native_media_wheel(path: Path):
    """Inspect provider data; release acceptance separately verifies trust with its base wheel."""
    from scripts.verify_extension_wheel import _assert_extension_wheel_layout

    with snapshot_archive(
        path,
        max_bytes=MAX_PUBLICATION_FILE_BYTES,
        description="native media wheel",
        size_limit_description=PUBLICATION_FILE_LIMIT_DESCRIPTION,
    ) as snapshot:
        layout = _assert_extension_wheel_layout(snapshot, "native_media")
        if layout.native_runtime is None:
            raise ValueError("native_media delivery requires bundled dynamic libraries")
        with open_zip_snapshot(
            snapshot, max_members=_MAX_EXTENSION_WHEEL_MEMBERS, description="native media wheel"
        ) as wheel:
            package_root = next(
                name.rpartition("/")[0] for name in wheel.namelist() if name.endswith(".dynamic-extension.json")
            )
            info_root = f"vane_extension_native_media-{layout.distribution_version}"
            info, _, _ = read_bundled_runtime(
                wheel,
                package_root=package_root,
                dist_info_root=info_root,
                reference=layout.native_runtime,
                platform=layout.platform_tag,
                signature_verifier=None,
            )
            return info

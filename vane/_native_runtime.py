# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Private, process-wide media runtime snapshots. No downloads or import-time loads."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
import threading
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from vane import _native_runtime_format as fmt

if TYPE_CHECKING:
    from vane.extensions import DynamicExtensionDescriptor, NativeRuntimeReference

_lock = threading.RLock()
_override: Path | None = None
_override_manifest_sha256: str | None = None
_override_distributed = False
_selected: str | None = None


def use_native_media_runtime(directory: str | Path, *, allow_distributed: bool = False) -> None:
    """Select a locally rebuilt runtime before preparing any native media artifact.

    The directory contains runtime-manifest.json and .libs. This explicitly
    trusts the local code for this process. It does not change extension signer
    policy. ``allow_distributed`` also permits this exact content identity in
    Ray queries. Each node must independently authorize its local runtime with
    VANE_NATIVE_MEDIA_RUNTIME before starting Ray. Paths and library bytes are
    never propagated in query snapshots. Start a new process to switch runtimes.
    """
    global _override, _override_manifest_sha256, _override_distributed
    if not isinstance(allow_distributed, bool):
        raise TypeError("allow_distributed must be a bool")
    path = Path(directory).expanduser().resolve(strict=True)
    document = fmt.read_file(path, fmt.MANIFEST, fmt.MAX_MANIFEST_BYTES)
    manifest = fmt.parse_manifest(document)
    fmt.verify_files(path / ".libs", manifest)
    digest = hashlib.sha256(document).hexdigest()
    with _lock:
        if _selected is not None:
            raise ValueError("select a custom native media runtime before preparing media extensions")
        if _override is not None and (
            _override != path or _override_manifest_sha256 != digest or _override_distributed != allow_distributed
        ):
            raise ValueError("a native media runtime override is already selected; start a new process")
        _override = path
        _override_manifest_sha256 = digest
        _override_distributed = allow_distributed


def _runtime_source(reference: NativeRuntimeReference) -> tuple[Path, bytes, bytes, bytes, dict[str, Any]]:
    try:
        installed = distribution(fmt.DISTRIBUTION)
    except PackageNotFoundError as exception:
        raise ValueError(
            f"preinstall {fmt.DISTRIBUTION}=={reference.version} before preparing native media"
        ) from exception
    if installed.version != reference.version:
        raise ValueError(f"native media requires {fmt.DISTRIBUTION}=={reference.version}")
    root = Path(cast(os.PathLike[str], installed.locate_file(fmt.PACKAGE)))
    document = fmt.read_file(root, fmt.MANIFEST, fmt.MAX_MANIFEST_BYTES)
    manifest = fmt.parse_manifest(document)
    if fmt.reference(document) != reference.to_dict():
        raise ValueError("installed native media runtime differs from the extension's exact reference")
    signature = fmt.read_file(root, fmt.SIGNATURE, 256)
    if len(signature) != 256:
        raise ValueError("native media runtime requires an RSA-2048 manifest signature")
    if _override is None:
        return root, document, signature, document, manifest
    effective = fmt.read_file(_override, fmt.MANIFEST, fmt.MAX_MANIFEST_BYTES)
    if _override_manifest_sha256 is not None and hashlib.sha256(effective).hexdigest() != _override_manifest_sha256:
        raise ValueError("custom native media runtime changed after selection")
    replacement = fmt.parse_manifest(effective)
    for key in ("distribution", "version", "platform", "namespace"):
        if replacement[key] != manifest[key]:
            raise ValueError(f"custom native media runtime changes its required {key}")
    if set(replacement["files"]) != set(manifest["files"]):
        raise ValueError("custom native media runtime must preserve the library filenames")
    return _override, document, signature, effective, replacement


def _validate_snapshot_layout(target: Path, manifest: dict[str, Any]) -> None:
    expected = {target.name, fmt.MANIFEST, fmt.SIGNATURE, "effective-runtime.json", ".libs"}
    if {entry.name for entry in target.parent.iterdir()} != expected:
        raise ValueError("native media snapshot has missing or unexpected entries")
    for path in (target.parent, *target.parent.iterdir(), *(target.parent / ".libs").iterdir()):
        info = path.lstat()
        if info.st_uid not in {0, os.geteuid()}:
            raise ValueError("native media snapshot has an untrusted owner")
        if stat.S_ISDIR(info.st_mode):
            if info.st_mode & 0o077:
                raise ValueError("native media snapshot directories must be private")
        elif not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_mode & 0o222:
            raise ValueError(
                "native media snapshot files must be regular, read-only, and free of additional hard links"
            )
    fmt.verify_files(target.parent / ".libs", manifest)


def _validate_snapshot(
    target: Path,
    descriptor: DynamicExtensionDescriptor,
    official: bytes,
    signature: bytes,
    effective: bytes,
    manifest: dict[str, Any],
) -> None:
    if fmt.read_file(target.parent, fmt.MANIFEST, fmt.MAX_MANIFEST_BYTES) != official:
        raise ValueError("cached native media manifest differs from its reference")
    if fmt.read_file(target.parent, fmt.SIGNATURE, 256) != signature:
        raise ValueError("cached native media manifest signature differs")
    if fmt.read_file(target.parent, "effective-runtime.json", fmt.MAX_MANIFEST_BYTES) != effective:
        raise ValueError("cached effective native media manifest differs")
    _validate_snapshot_layout(target, manifest)
    contents = target.read_bytes()
    if hashlib.sha256(contents).hexdigest() != descriptor.sha256:
        raise ValueError("cached native media extension digest differs")
    if descriptor.native_runtime is None or fmt.trailer_digest(contents) != descriptor.native_runtime.manifest_sha256:
        raise ValueError("cached native media extension trailer differs from its descriptor")


def prepare_snapshot(artifact: Path, descriptor: DynamicExtensionDescriptor, cache_root: Path) -> Path:
    """Atomically publish the extension and its complete effective runtime together."""
    global _selected
    from vane import _native
    from vane.extensions import (
        DynamicExtensionResolver,
        _copy_and_hash_artifact,
        _make_snapshot_read_only,
        _sha256_file,
    )

    if descriptor.native_runtime is None:
        raise ValueError("native media snapshot requires a runtime reference")
    with _lock:
        source, official, signature, effective, manifest = _runtime_source(descriptor.native_runtime)
        if not _native._verify_native_runtime_signature(official, signature, False):
            raise ValueError("native media runtime manifest signature is not trusted")
        fmt.verify_files(source / ".libs", manifest)
        effective_id = hashlib.sha256(effective).hexdigest()
        if _selected is not None and _selected != effective_id:
            raise ValueError("another native media runtime is prepared; start a new process")
        parent = cache_root / descriptor.sha256 / effective_id
        DynamicExtensionResolver._prepare_cache_directory(parent)
        destination = parent / descriptor.name
        target = destination / artifact.name
        if destination.exists() or destination.is_symlink():
            if _sha256_file(artifact) != descriptor.sha256:
                raise ValueError("native media extension digest differs from its descriptor")
            _validate_snapshot(target, descriptor, official, signature, effective, manifest)
            _selected = effective_id
            return target
        staging = Path(tempfile.mkdtemp(prefix=".media-", dir=parent))
        try:
            DynamicExtensionResolver._prepare_created_private_directory(staging, description="native media staging")
            staged_artifact = staging / artifact.name
            actual_digest = _copy_and_hash_artifact(artifact, staged_artifact)
            if actual_digest != descriptor.sha256:
                raise ValueError("native media extension digest differs from its descriptor")
            _make_snapshot_read_only(staged_artifact, description="native media extension")
            if fmt.trailer_digest(staged_artifact.read_bytes()) != descriptor.native_runtime.manifest_sha256:
                raise ValueError("native media extension trailer differs from its descriptor")
            (staging / fmt.MANIFEST).write_bytes(official)
            (staging / fmt.SIGNATURE).write_bytes(signature)
            (staging / "effective-runtime.json").write_bytes(effective)
            library_directory = staging / ".libs"
            library_directory.mkdir(mode=0o700)
            DynamicExtensionResolver._prepare_created_private_directory(
                library_directory, description="native media libraries"
            )
            for name, record in manifest["files"].items():
                contents = fmt.read_file(source / ".libs", name)
                if len(contents) != record["size"] or hashlib.sha256(contents).hexdigest() != record["sha256"]:
                    raise ValueError(f"native media library differs from its manifest: {name}")
                (library_directory / name).write_bytes(contents)
            for item in staging.rglob("*"):
                item.chmod(0o700 if item.is_dir() else 0o400)
            try:
                os.rename(staging, destination)
            except OSError:
                if not destination.is_dir() or destination.is_symlink():
                    raise
            # A concurrent publisher or old cache is checked just as strictly.
            _validate_snapshot(target, descriptor, official, signature, effective, manifest)
            # Reserve one runtime identity per process; the OS loads dependencies
            # through the extension RUNPATH when DuckDB later loads this path.
            _selected = effective_id
            return target
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def distributed_runtime_selection(
    descriptors: Iterable[DynamicExtensionDescriptor], *, required: bool = True
) -> str | None:
    """Capture the transport-authorized identity, requiring opt-in for workers."""
    references = [descriptor.native_runtime for descriptor in descriptors if descriptor.native_runtime is not None]
    if not references or _override is None:
        return None
    with _lock:
        if not _override_distributed:
            if required:
                raise ValueError("Ray custom native media requires allow_distributed=True; this runtime is local-only")
            return None
        for reference in references:
            source, _official, _signature, effective, manifest = _runtime_source(reference)
            fmt.verify_files(source / ".libs", manifest)
            digest = hashlib.sha256(effective).hexdigest()
            if digest != _override_manifest_sha256 or (_selected is not None and digest != _selected):
                raise ValueError("custom native media runtime differs from the selected process identity")
        return _override_manifest_sha256


def prepare_distributed_runtime(
    descriptors: Iterable[DynamicExtensionDescriptor], expected_manifest_sha256: str | None
) -> None:
    """Match the coordinator identity against independently authorized local code."""
    descriptors = tuple(descriptors)
    if not any(descriptor.native_runtime is not None for descriptor in descriptors):
        return
    with _lock:
        if expected_manifest_sha256 is not None and _override is None:
            directory = os.environ.get("VANE_NATIVE_MEDIA_RUNTIME")
            if not directory:
                raise ValueError(
                    "Ray custom native media requires VANE_NATIVE_MEDIA_RUNTIME in each node's environment; "
                    "preinstall and explicitly authorize the matching runtime before starting Ray"
                )
            use_native_media_runtime(directory, allow_distributed=True)
        if distributed_runtime_selection(descriptors) != expected_manifest_sha256:
            raise ValueError("Ray native media runtime differs from the coordinator's exact content identity")

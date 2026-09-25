# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bound archive metadata parsing before standard readers retain all members."""

from __future__ import annotations

import gzip
import os
import stat
import struct
import tarfile
import tempfile
import zipfile
import zlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, cast

_END_OF_CENTRAL_DIRECTORY = struct.Struct("<4s4H2LH")
_CENTRAL_DIRECTORY_HEADER = struct.Struct("<4s6H3L5H2L")
_END_OF_CENTRAL_DIRECTORY_SIGNATURE = b"PK\x05\x06"
_CENTRAL_DIRECTORY_HEADER_SIGNATURE = b"PK\x01\x02"
_ZIP64_END_OF_CENTRAL_DIRECTORY_LOCATOR_SIGNATURE = b"PK\x06\x07"
_ZIP16_MAX = (1 << 16) - 1
_ZIP32_MAX = (1 << 32) - 1
_TAR_BLOCK_BYTES = tarfile.BLOCKSIZE
_TAR_ZERO_BLOCK = b"\0" * _TAR_BLOCK_BYTES
_TAR_READ_CHUNK_BYTES = 64 * 1024
_MAX_TAR_TRAILING_ZERO_BYTES = tarfile.RECORDSIZE
_MAX_TAR_EXTENSION_HEADER_BYTES = 1024 * 1024
_MAX_TAR_PAX_RECORDS = 10_000
_PAX_HEADER_TYPES = frozenset({tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.SOLARIS_XHDTYPE})
_GNU_LONG_HEADER_TYPES = frozenset({tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK})
_REGULAR_TAR_MEMBER_TYPES = frozenset({tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.CONTTYPE})
_ALLOWED_TAR_MEMBER_TYPES = _REGULAR_TAR_MEMBER_TYPES | {tarfile.DIRTYPE}
_ARCHIVE_SNAPSHOT_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ArchiveSnapshot:
    """One private, bounded copy of an untrusted archive input."""

    source_path: Path
    path: Path
    size: int
    file: BinaryIO = field(repr=False, compare=False)

    def validate_named_path(self, *, description: str) -> None:
        """Require the private path to still name the retained snapshot file."""
        try:
            path_metadata = self.path.lstat()
            file_metadata = os.fstat(self.file.fileno())
        except OSError as exception:
            raise ValueError(
                f"private {description} snapshot is no longer available: {self.source_path}"
            ) from exception
        if (
            not stat.S_ISREG(path_metadata.st_mode)
            or path_metadata.st_nlink != 1
            or path_metadata.st_size != self.size
            or not os.path.samestat(path_metadata, file_metadata)
        ):
            raise ValueError(f"private {description} snapshot changed after validation: {self.source_path}")


@contextmanager
def snapshot_archive(
    path: str | Path,
    *,
    max_bytes: int,
    description: str,
    size_limit_description: str,
) -> Iterator[ArchiveSnapshot]:
    """Copy one regular input into a private file that outlives all readers."""
    archive_path = Path(path)
    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")

    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    snapshot_file: BinaryIO | None = None
    try:
        source_path_metadata = archive_path.lstat()
        if not stat.S_ISREG(source_path_metadata.st_mode):
            raise ValueError(f"{description} must be a regular file: {archive_path}")
        source_flags = os.O_RDONLY
        for flag_name in ("O_BINARY", "O_CLOEXEC", "O_NONBLOCK", "O_NOFOLLOW"):
            source_flags |= getattr(os, flag_name, 0)
        source_descriptor = os.open(archive_path, source_flags)
        with _binary_file_from_descriptor(source_descriptor, "rb") as source_file:
            metadata = os.fstat(source_file.fileno())
            if not stat.S_ISREG(metadata.st_mode) or not os.path.samestat(source_path_metadata, metadata):
                raise ValueError(f"{description} must be a regular file: {archive_path}")
            if metadata.st_size > max_bytes:
                raise ValueError(f"{archive_path}: {description} exceeds {size_limit_description}")

            temporary_directory = tempfile.TemporaryDirectory(prefix="vane-archive-snapshot-")
            snapshot_directory = Path(temporary_directory.name)
            snapshot_directory_metadata = snapshot_directory.lstat()
            if not stat.S_ISDIR(snapshot_directory_metadata.st_mode) or (
                os.name != "nt" and stat.S_IMODE(snapshot_directory_metadata.st_mode) & 0o077
            ):
                raise ValueError(f"could not create a private snapshot directory for {description}: {archive_path}")
            snapshot_path = snapshot_directory / archive_path.name
            destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            for flag_name in ("O_BINARY", "O_CLOEXEC"):
                destination_flags |= getattr(os, flag_name, 0)
            destination_descriptor = os.open(snapshot_path, destination_flags, 0o600)
            with _binary_file_from_descriptor(destination_descriptor, "wb") as destination_file:
                snapshot_size = _copy_archive_snapshot(
                    source_file,
                    destination_file,
                    archive_path=archive_path,
                    max_bytes=max_bytes,
                    description=description,
                    size_limit_description=size_limit_description,
                )
                destination_file.flush()
                if os.name != "nt":
                    os.fchmod(destination_file.fileno(), 0o400)
                written_snapshot_metadata = os.fstat(destination_file.fileno())

        snapshot_file = _binary_file_from_descriptor(os.open(snapshot_path, source_flags), "rb")
        snapshot_metadata = snapshot_path.lstat()
        retained_snapshot_metadata = os.fstat(snapshot_file.fileno())
        if (
            not stat.S_ISREG(snapshot_metadata.st_mode)
            or not stat.S_ISREG(retained_snapshot_metadata.st_mode)
            or snapshot_metadata.st_nlink != 1
            or snapshot_metadata.st_size != snapshot_size
            or retained_snapshot_metadata.st_size != snapshot_size
            or not os.path.samestat(written_snapshot_metadata, retained_snapshot_metadata)
            or not os.path.samestat(snapshot_metadata, retained_snapshot_metadata)
            or (os.name != "nt" and stat.S_IMODE(retained_snapshot_metadata.st_mode) != 0o400)
        ):
            raise ValueError(f"could not create a private regular snapshot for {description}: {archive_path}")
        snapshot = ArchiveSnapshot(
            source_path=archive_path,
            path=snapshot_path,
            size=snapshot_size,
            file=snapshot_file,
        )
    except OSError as exception:
        try:
            if snapshot_file is not None:
                snapshot_file.close()
        finally:
            if temporary_directory is not None:
                temporary_directory.cleanup()
        raise ValueError(f"could not snapshot {description}: {archive_path}") from exception
    except BaseException:
        try:
            if snapshot_file is not None:
                snapshot_file.close()
        finally:
            if temporary_directory is not None:
                temporary_directory.cleanup()
        raise

    try:
        yield snapshot
    finally:
        if temporary_directory is None:  # pragma: no cover - assigned before every successful yield
            raise RuntimeError("archive snapshot lost its temporary-directory owner")
        try:
            snapshot.file.close()
        finally:
            temporary_directory.cleanup()


def _binary_file_from_descriptor(descriptor: int, mode: str) -> BinaryIO:
    try:
        return cast(BinaryIO, os.fdopen(descriptor, mode))
    except BaseException:
        os.close(descriptor)
        raise


def _copy_archive_snapshot(
    source_file: BinaryIO,
    destination_file: BinaryIO,
    *,
    archive_path: Path,
    max_bytes: int,
    description: str,
    size_limit_description: str,
) -> int:
    copied_bytes = 0
    while True:
        remaining_with_overflow = max_bytes - copied_bytes + 1
        chunk = source_file.read(min(_ARCHIVE_SNAPSHOT_CHUNK_BYTES, remaining_with_overflow))
        if not chunk:
            return copied_bytes
        copied_bytes += len(chunk)
        if copied_bytes > max_bytes:
            raise ValueError(f"{archive_path}: {description} exceeds {size_limit_description}")
        destination_file.write(chunk)


def validate_zip_member_count(
    archive_file: BinaryIO,
    *,
    archive_path: str | Path,
    file_size: int,
    max_members: int,
    description: str,
) -> int:
    """Validate and count central-directory entries without creating ``ZipInfo`` objects."""
    archive_path = Path(archive_path)
    if max_members < 0:
        raise ValueError("max_members must be non-negative")
    try:
        archive_file.seek(0)
        return _validate_zip_central_directory(
            archive_file,
            file_size=file_size,
            max_members=max_members,
            description=description,
        )
    except OSError as exception:
        raise ValueError(f"could not inspect {description}: {archive_path}") from exception


def validate_tar_member_count(
    archive_file: BinaryIO,
    *,
    archive_path: str | Path,
    max_members: int,
    max_member_bytes: int,
    max_total_bytes: int,
    member_limit_description: str,
    total_limit_description: str,
    description: str,
    metadata_chunk_callback: Callable[[bytes, bool], None] | None = None,
) -> int:
    """Stream and bound TAR metadata without retaining the complete member list.

    ``metadata_chunk_callback`` receives raw TAR headers and extension-header
    payload chunks. Its second argument is true when a header starts a new,
    physically contiguous metadata region.
    """
    archive_path = Path(archive_path)
    if max_members < 0 or max_member_bytes < 0 or max_total_bytes < 0:
        raise ValueError("TAR member and size limits must be non-negative")
    try:
        archive_file.seek(0)
        magic = archive_file.read(2)
        archive_file.seek(0)
        if magic == b"\x1f\x8b":
            with gzip.GzipFile(fileobj=archive_file, mode="rb") as tar_stream:
                return _validate_tar_stream(
                    tar_stream,
                    archive_path=archive_path,
                    max_members=max_members,
                    max_member_bytes=max_member_bytes,
                    max_total_bytes=max_total_bytes,
                    member_limit_description=member_limit_description,
                    total_limit_description=total_limit_description,
                    description=description,
                    metadata_chunk_callback=metadata_chunk_callback,
                )
        return _validate_tar_stream(
            archive_file,
            archive_path=archive_path,
            max_members=max_members,
            max_member_bytes=max_member_bytes,
            max_total_bytes=max_total_bytes,
            member_limit_description=member_limit_description,
            total_limit_description=total_limit_description,
            description=description,
            metadata_chunk_callback=metadata_chunk_callback,
        )
    except (EOFError, OSError, tarfile.TarError, zlib.error) as exception:
        raise ValueError(f"could not inspect {description}: {archive_path}") from exception


@contextmanager
def open_zip_snapshot(
    snapshot: ArchiveSnapshot,
    *,
    max_members: int,
    description: str,
) -> Iterator[zipfile.ZipFile]:
    """Preflight and parse one ZIP from the same snapshotted file descriptor."""
    validate_zip_member_count(
        snapshot.file,
        archive_path=snapshot.source_path,
        file_size=snapshot.size,
        max_members=max_members,
        description=description,
    )
    snapshot.file.seek(0)
    with zipfile.ZipFile(snapshot.file) as archive:
        yield archive


@contextmanager
def open_tar_snapshot(
    snapshot: ArchiveSnapshot,
    *,
    max_members: int,
    max_member_bytes: int,
    max_total_bytes: int,
    member_limit_description: str,
    total_limit_description: str,
    description: str,
    metadata_chunk_callback: Callable[[bytes, bool], None] | None = None,
    mode: str = "r:*",
) -> Iterator[tarfile.TarFile]:
    """Preflight and parse one TAR from the same snapshotted file descriptor."""
    validate_tar_member_count(
        snapshot.file,
        archive_path=snapshot.source_path,
        max_members=max_members,
        max_member_bytes=max_member_bytes,
        max_total_bytes=max_total_bytes,
        member_limit_description=member_limit_description,
        total_limit_description=total_limit_description,
        description=description,
        metadata_chunk_callback=metadata_chunk_callback,
    )
    snapshot.file.seek(0)
    with tarfile.open(fileobj=snapshot.file, mode=mode) as archive:
        yield archive


def _validate_tar_stream(
    tar_stream: BinaryIO,
    *,
    archive_path: Path,
    max_members: int,
    max_member_bytes: int,
    max_total_bytes: int,
    member_limit_description: str,
    total_limit_description: str,
    description: str,
    metadata_chunk_callback: Callable[[bytes, bool], None] | None,
) -> int:
    member_count = 0
    total_size = 0
    zero_blocks = 0
    pending_extension_header = False
    pending_pax_size: bytes | None = None
    while True:
        header = tar_stream.read(_TAR_BLOCK_BYTES)
        if not header:
            raise ValueError(f"{description} ended before its two-block TAR terminator")
        if len(header) != _TAR_BLOCK_BYTES:
            raise ValueError(f"{description} has a truncated TAR header")
        if metadata_chunk_callback is not None:
            metadata_chunk_callback(header, True)
        if header == _TAR_ZERO_BLOCK:
            zero_blocks += 1
            if zero_blocks == 2:
                if pending_extension_header:
                    raise ValueError(f"{description} ends with an unused TAR extension header")
                _validate_tar_trailing_padding(tar_stream, description=description)
                return member_count
            continue
        if zero_blocks:
            raise ValueError(f"{description} contains data after an incomplete TAR terminator")

        try:
            member = tarfile.TarInfo.frombuf(header, encoding="utf-8", errors="surrogateescape")
        except tarfile.HeaderError as exception:
            raise ValueError(f"{description} contains an invalid TAR header") from exception
        member_count += 1
        if member_count > max_members:
            raise ValueError(f"{description} contains more than {max_members} archive members")
        _validate_tar_payload_size(
            member.size,
            member_name=member.name,
            archive_path=archive_path,
            max_member_bytes=max_member_bytes,
            member_limit_description=member_limit_description,
        )

        if member.type in _PAX_HEADER_TYPES | _GNU_LONG_HEADER_TYPES:
            pending_extension_header = True
            if member.size > _MAX_TAR_EXTENSION_HEADER_BYTES:
                raise ValueError(
                    f"{archive_path}: TAR extension header {member.name!r} exceeds the bounded 1 MiB metadata limit"
                )
            total_size = _add_tar_payload_size(
                total_size,
                member.size,
                archive_path=archive_path,
                max_total_bytes=max_total_bytes,
                total_limit_description=total_limit_description,
            )
            payload = _read_tar_payload(
                tar_stream,
                member.size,
                collect=member.type in _PAX_HEADER_TYPES,
                chunk_callback=metadata_chunk_callback,
            )
            if member.type in _PAX_HEADER_TYPES:
                overrides = _parse_pax_overrides(payload, description=description)
                if member.type == tarfile.XGLTYPE:
                    if b"size" in overrides:
                        raise ValueError(f"{description} contains an unsupported global PAX size override")
                elif pending_pax_size is None:
                    # TarInfo processes chained local PAX headers recursively,
                    # then applies the first header last. Preserve that
                    # first-header-wins behavior for physical size overrides.
                    pending_pax_size = overrides.get(b"size")
            continue

        if member.type not in _ALLOWED_TAR_MEMBER_TYPES:
            raise ValueError(f"{archive_path}: unsupported TAR member type for {member.name!r}")

        effective_size = _pax_member_size(pending_pax_size, default=member.size, description=description)
        pending_extension_header = False
        pending_pax_size = None
        _validate_tar_payload_size(
            effective_size,
            member_name=member.name,
            archive_path=archive_path,
            max_member_bytes=max_member_bytes,
            member_limit_description=member_limit_description,
        )
        total_size = _add_tar_payload_size(
            total_size,
            effective_size,
            archive_path=archive_path,
            max_total_bytes=max_total_bytes,
            total_limit_description=total_limit_description,
        )
        physical_size = effective_size if member.type in _REGULAR_TAR_MEMBER_TYPES else 0
        _read_tar_payload(tar_stream, physical_size, collect=False)


def _validate_tar_payload_size(
    size: int,
    *,
    member_name: str,
    archive_path: Path,
    max_member_bytes: int,
    member_limit_description: str,
) -> None:
    if size < 0:
        raise ValueError(f"{archive_path}: archive member {member_name!r} has an invalid negative size")
    if size > max_member_bytes:
        raise ValueError(f"{archive_path}: archive member {member_name!r} exceeds {member_limit_description}")


def _add_tar_payload_size(
    total_size: int,
    member_size: int,
    *,
    archive_path: Path,
    max_total_bytes: int,
    total_limit_description: str,
) -> int:
    updated_size = total_size + member_size
    if updated_size > max_total_bytes:
        raise ValueError(f"{archive_path}: archive decompressed contents exceed {total_limit_description}")
    return updated_size


def _read_tar_payload(
    tar_stream: BinaryIO,
    size: int,
    *,
    collect: bool,
    chunk_callback: Callable[[bytes, bool], None] | None = None,
) -> bytes:
    remaining = size
    collected = bytearray() if collect else None
    while remaining:
        chunk = tar_stream.read(min(remaining, _TAR_READ_CHUNK_BYTES))
        if not chunk:
            raise ValueError("TAR payload is truncated")
        remaining -= len(chunk)
        if chunk_callback is not None:
            chunk_callback(chunk, False)
        if collected is not None:
            collected.extend(chunk)
    padding = (-size) % _TAR_BLOCK_BYTES
    while padding:
        chunk = tar_stream.read(min(padding, _TAR_READ_CHUNK_BYTES))
        if not chunk:
            raise ValueError("TAR payload padding is truncated")
        if chunk.strip(b"\0"):
            raise ValueError("TAR payload padding must contain only zero bytes")
        padding -= len(chunk)
    return bytes(collected) if collected is not None else b""


def _validate_tar_trailing_padding(tar_stream: BinaryIO, *, description: str) -> None:
    """Consume the decompressed stream and permit only one bounded zero record."""
    trailing_bytes = 0
    while True:
        remaining_with_overflow = _MAX_TAR_TRAILING_ZERO_BYTES - trailing_bytes + 1
        chunk = tar_stream.read(min(_TAR_READ_CHUNK_BYTES, remaining_with_overflow))
        if not chunk:
            return
        if chunk.strip(b"\0"):
            raise ValueError(f"{description} contains nonzero data after its TAR terminator")
        trailing_bytes += len(chunk)
        if trailing_bytes > _MAX_TAR_TRAILING_ZERO_BYTES:
            raise ValueError(
                f"{description} contains more than {_MAX_TAR_TRAILING_ZERO_BYTES} bytes of zero padding "
                "after its TAR terminator"
            )


def _parse_pax_overrides(payload: bytes, *, description: str) -> dict[bytes, bytes]:
    overrides: dict[bytes, bytes] = {}
    offset = 0
    record_count = 0
    while offset < len(payload):
        separator = payload.find(b" ", offset, min(len(payload), offset + 32))
        if separator < 0:
            raise ValueError(f"{description} contains an invalid PAX record length")
        raw_length = payload[offset:separator]
        if not raw_length.isdigit() or len(raw_length) > 20:
            raise ValueError(f"{description} contains an invalid PAX record length")
        record_length = int(raw_length)
        record_end = offset + record_length
        if record_length <= separator - offset + 2 or record_end > len(payload):
            raise ValueError(f"{description} contains an out-of-bounds PAX record")
        record = payload[separator + 1 : record_end]
        if not record.endswith(b"\n") or b"=" not in record[:-1]:
            raise ValueError(f"{description} contains an invalid PAX record")
        key, value = record[:-1].split(b"=", 1)
        if not key or b"\0" in key or key.startswith(b"GNU.sparse."):
            raise ValueError(f"{description} contains an unsupported PAX record")
        overrides[key] = value
        record_count += 1
        if record_count > _MAX_TAR_PAX_RECORDS:
            raise ValueError(f"{description} contains more than {_MAX_TAR_PAX_RECORDS} PAX records")
        offset = record_end
    return overrides


def _pax_member_size(
    value: bytes | None,
    *,
    default: int,
    description: str,
) -> int:
    if value is None:
        return default
    if not value.isdigit() or len(value) > 20:
        raise ValueError(f"{description} contains an invalid PAX size override")
    return int(value)


def _validate_zip_central_directory(
    archive_file: BinaryIO,
    *,
    file_size: int,
    max_members: int,
    description: str,
) -> int:
    if file_size < _END_OF_CENTRAL_DIRECTORY.size:
        raise ValueError(f"{description} has no valid end-of-central-directory record")
    eocd_offset = file_size - _END_OF_CENTRAL_DIRECTORY.size
    archive_file.seek(eocd_offset)
    eocd = _read_exact(
        archive_file,
        _END_OF_CENTRAL_DIRECTORY.size,
        description=description,
    )
    (
        signature,
        disk_number,
        central_directory_disk,
        entries_on_disk,
        declared_members,
        central_directory_size,
        central_directory_offset,
        comment_size,
    ) = _END_OF_CENTRAL_DIRECTORY.unpack(eocd)
    if signature != _END_OF_CENTRAL_DIRECTORY_SIGNATURE or comment_size != 0:
        raise ValueError(f"{description} must end with an un-commented ZIP central directory")
    if (
        entries_on_disk == _ZIP16_MAX
        or declared_members == _ZIP16_MAX
        or central_directory_size == _ZIP32_MAX
        or central_directory_offset == _ZIP32_MAX
        or _has_zip64_locator(archive_file, eocd_offset)
    ):
        raise ValueError(f"{description} must not use ZIP64 end records")
    if disk_number != 0 or central_directory_disk != 0 or entries_on_disk != declared_members:
        raise ValueError(f"{description} must contain one non-spanned ZIP archive")
    if declared_members > max_members:
        raise ValueError(f"{description} contains more than {max_members} archive members")
    if central_directory_offset + central_directory_size != eocd_offset:
        raise ValueError(f"{description} has invalid central-directory bounds")

    archive_file.seek(central_directory_offset)
    remaining = central_directory_size
    actual_members = 0
    while remaining:
        if remaining < _CENTRAL_DIRECTORY_HEADER.size:
            raise ValueError(f"{description} has a truncated central-directory header")
        header = _read_exact(
            archive_file,
            _CENTRAL_DIRECTORY_HEADER.size,
            description=description,
        )
        remaining -= _CENTRAL_DIRECTORY_HEADER.size
        fields = _CENTRAL_DIRECTORY_HEADER.unpack(header)
        if fields[0] != _CENTRAL_DIRECTORY_HEADER_SIGNATURE:
            raise ValueError(f"{description} has an invalid central-directory member header")
        if fields[13] != 0:
            raise ValueError(f"{description} must contain one non-spanned ZIP archive")
        variable_size = fields[10] + fields[11] + fields[12]
        if variable_size > remaining:
            raise ValueError(f"{description} has a truncated central-directory member")
        archive_file.seek(variable_size, os.SEEK_CUR)
        remaining -= variable_size
        actual_members += 1
        if actual_members > max_members:
            raise ValueError(f"{description} contains more than {max_members} archive members")
    if actual_members != declared_members:
        raise ValueError(
            f"{description} central-directory member count does not match its end record: "
            f"declared={declared_members}, actual={actual_members}"
        )
    return actual_members


def _has_zip64_locator(archive_file: BinaryIO, eocd_offset: int) -> bool:
    locator_size = 20
    if eocd_offset < locator_size:
        return False
    archive_file.seek(eocd_offset - locator_size)
    return archive_file.read(4) == _ZIP64_END_OF_CENTRAL_DIRECTORY_LOCATOR_SIGNATURE


def _read_exact(archive_file: BinaryIO, size: int, *, description: str) -> bytes:
    contents = archive_file.read(size)
    if len(contents) != size:
        raise ValueError(f"{description} ended while its central directory was being inspected")
    return contents


__all__ = [
    "ArchiveSnapshot",
    "open_tar_snapshot",
    "open_zip_snapshot",
    "snapshot_archive",
    "validate_tar_member_count",
    "validate_zip_member_count",
]

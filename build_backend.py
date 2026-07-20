# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""PEP 517 backend that adds DuckDB's generated identity to source archives."""

from __future__ import annotations

import gzip
import io
import os
import struct
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from scikit_build_core import build as _backend

from scripts.sync_duckdb_source_id import REPOSITORY_ROOT, source_tree_id, validate_source_id

SOURCE_ID_PATH = "DUCKDB_SOURCE_ID"
SOURCE_ID_FILE = REPOSITORY_ROOT / SOURCE_ID_PATH

build_editable = _backend.build_editable
build_wheel = _backend.build_wheel
get_requires_for_build_editable = _backend.get_requires_for_build_editable
get_requires_for_build_sdist = _backend.get_requires_for_build_sdist
get_requires_for_build_wheel = _backend.get_requires_for_build_wheel
prepare_metadata_for_build_editable = _backend.prepare_metadata_for_build_editable
prepare_metadata_for_build_wheel = _backend.prepare_metadata_for_build_wheel


def _source_id_for_sdist() -> str:
    if (REPOSITORY_ROOT / ".git").exists():
        return source_tree_id()
    if SOURCE_ID_FILE.is_file():
        return validate_source_id(SOURCE_ID_FILE.read_text(encoding="ascii").strip())
    return source_tree_id()


def _gzip_timestamp(path: Path) -> int:
    with path.open("rb") as archive_file:
        header = archive_file.read(10)
    if len(header) != 10 or header[:2] != b"\x1f\x8b":
        raise RuntimeError(f"build backend produced an invalid gzip archive: {path}")
    return struct.unpack("<I", header[4:8])[0]


def _inject_source_id(archive_path: Path, source_id: str) -> None:
    """Atomically add or replace DUCKDB_SOURCE_ID in a generated sdist."""
    expected = (validate_source_id(source_id) + "\n").encode("ascii")
    timestamp = _gzip_timestamp(archive_path)

    with tarfile.open(archive_path, mode="r:gz") as source_archive:
        members = source_archive.getmembers()
        roots = {PurePosixPath(member.name).parts[0] for member in members if PurePosixPath(member.name).parts}
        if len(roots) != 1:
            raise RuntimeError(f"sdist must contain exactly one root directory, found {sorted(roots)}")
        target_name = str(PurePosixPath(roots.pop()) / SOURCE_ID_PATH)
        existing = [member for member in members if member.name == target_name]
        if len(existing) == 1 and existing[0].isfile():
            source_file = source_archive.extractfile(existing[0])
            if source_file is not None and source_file.read() == expected:
                return

        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{archive_path.name}.", dir=archive_path.parent)
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            target = tarfile.TarInfo(target_name)
            target.size = len(expected)
            target.mode = 0o644
            target.mtime = timestamp
            target.uid = 0
            target.gid = 0
            target.uname = ""
            target.gname = ""

            entries = [(member.name, member, None) for member in members if member.name != target_name]
            entries.append((target_name, target, expected))
            with temporary_path.open("wb") as raw_archive:
                gzip_name = archive_path.name.removesuffix(".gz")
                with gzip.GzipFile(
                    filename=gzip_name,
                    mode="wb",
                    compresslevel=9,
                    fileobj=raw_archive,
                    mtime=timestamp,
                ) as gzip_archive:
                    with tarfile.open(fileobj=gzip_archive, mode="w", format=tarfile.PAX_FORMAT) as output_archive:
                        for _, member, data in sorted(entries, key=lambda entry: entry[0]):
                            if data is not None:
                                output_archive.addfile(member, fileobj=io.BytesIO(data))
                            elif member.isfile():
                                source_file = source_archive.extractfile(member)
                                if source_file is None:
                                    raise RuntimeError(f"unable to read {member.name!r} from {archive_path}")
                                output_archive.addfile(member, fileobj=source_file)
                            else:
                                output_archive.addfile(member)

            temporary_path.chmod(archive_path.stat().st_mode & 0o777)
            os.replace(temporary_path, archive_path)
        finally:
            temporary_path.unlink(missing_ok=True)


def build_sdist(
    sdist_directory: str,
    config_settings: dict[str, list[str] | str] | None = None,
) -> str:
    """Build an sdist containing the exact DuckDB tree ID for this checkout."""
    source_id = _source_id_for_sdist()
    filename = _backend.build_sdist(sdist_directory, config_settings)
    _inject_source_id(Path(sdist_directory) / filename, source_id)
    return filename

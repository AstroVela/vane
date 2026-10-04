# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Inventory exact redistributed Python wheels without treating metadata as legal approval."""

from __future__ import annotations

import hashlib
from email.parser import BytesParser
from email.policy import default
from pathlib import Path, PurePosixPath

from packaging.utils import canonicalize_name, parse_wheel_filename

from vane_packaging.archive_safety import open_zip_snapshot, snapshot_archive
from vane_packaging.extension_wheel import _validate_wheel_record


def inventory_wheel(path: Path) -> dict:
    distribution, version, _, tags = parse_wheel_filename(path.name)
    with snapshot_archive(
        path, max_bytes=512 * 1024 * 1024, description="Python delivery wheel", size_limit_description="512 MiB"
    ) as snapshot:
        digest = hashlib.sha256()
        while data := snapshot.file.read(1024 * 1024):
            digest.update(data)
        snapshot.file.seek(0)
        with open_zip_snapshot(snapshot, max_members=20000, description="Python delivery wheel") as archive:
            names = archive.namelist()
            entries = archive.infolist()
            if (
                len({name.casefold() for name in names}) != len(names)
                or sum(entry.file_size for entry in entries) > 1024 * 1024 * 1024
                or any(entry.file_size > 256 * 1024 * 1024 for entry in entries)
            ):
                raise ValueError("Python delivery wheel has colliding members or exceeds its bounds")
            for name in names:
                member = PurePosixPath(name)
                if member.is_absolute() or ".." in member.parts or "\\" in name or str(member) != name.rstrip("/"):
                    raise ValueError("Python delivery wheel contains a non-canonical member")
            metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
            if len(metadata_names) != 1 or archive.getinfo(metadata_names[0]).file_size > 1024 * 1024:
                raise ValueError("Python delivery wheel requires bounded, unique distribution metadata")
            info = metadata_names[0].removesuffix("/METADATA")
            metadata = BytesParser(policy=default).parsebytes(archive.read(metadata_names[0]))
            if (
                len(metadata.get_all("Name", [])) != 1
                or len(metadata.get_all("Version", [])) != 1
                or canonicalize_name(metadata["Name"]) != distribution
                or metadata["Version"] != str(version)
            ):
                raise ValueError("Python delivery wheel metadata differs from its filename")
            # Ordinary wheels can contain explicit ZIP directory entries; RECORD
            # inventories files, unlike our deliberately directory-free providers.
            _validate_wheel_record(
                archive, names=[entry.filename for entry in entries if not entry.is_dir()], record_name=info + "/RECORD"
            )
            notices = {}
            native = {}
            for entry in entries:
                if entry.is_dir():
                    continue
                name = entry.filename
                basename = PurePosixPath(name).name.lower()
                with archive.open(entry) as stream:
                    magic = stream.read(4)
                if (
                    magic
                    in {
                        b"\x7fELF",
                        b"!<ar",
                        b"!<th",
                        b"\xce\xfa\xed\xfe",
                        b"\xfe\xed\xfa\xce",
                        b"\xcf\xfa\xed\xfe",
                        b"\xfe\xed\xfa\xcf",
                        b"\xca\xfe\xba\xbe",
                        b"\xbe\xba\xfe\xca",
                        b"\xca\xfe\xba\xbf",
                        b"\xbf\xba\xfe\xca",
                    }
                    or magic[:2] == b"MZ"
                ):
                    native[name] = {"size": entry.file_size, "sha256": hashlib.sha256(archive.read(entry)).hexdigest()}
                if "/licenses/" in name.lower() or basename.startswith(
                    ("license", "licence", "copying", "copyright", "notice")
                ):
                    if entry.file_size > 4 * 1024 * 1024:
                        raise ValueError("Python delivery license notice exceeds 4 MiB")
                    notices[name] = hashlib.sha256(archive.read(entry)).hexdigest()
            return {
                "filename": path.name,
                "sha256": digest.hexdigest(),
                "size": snapshot.size,
                "distribution": distribution,
                "version": str(version),
                "tags": sorted(map(str, tags)),
                "license_expression": metadata.get("License-Expression"),
                "legacy_license": metadata.get("License"),
                "declared_license_files": metadata.get_all("License-File", []),
                "license_notices": notices,
                "native_binaries": native,
                "review_status": "required",
            }


def inventory_delivery(wheels: list[Path]) -> dict:
    if not 1 <= len(wheels) <= 512:
        raise ValueError("supply between 1 and 512 actually redistributed wheels")
    records = [inventory_wheel(path) for path in wheels]
    if len({record["filename"].casefold() for record in records}) != len(records):
        raise ValueError("Python delivery wheel filenames must be unique")
    return {"schema_version": 1, "wheels": sorted(records, key=lambda record: record["filename"])}

# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0
"""Sign a DuckDB extension with the repository's integration-test key."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

_DUCKDB_SIGNATURE_SIZE = 256
_HASH_CHUNK_SIZE = 1024 * 1024


def _extension_hash(payload: bytes) -> bytes:
    chunk_hashes = b"".join(
        hashlib.sha256(payload[offset : offset + _HASH_CHUNK_SIZE]).digest()
        for offset in range(0, len(payload), _HASH_CHUNK_SIZE)
    )
    if not chunk_hashes:
        raise ValueError("DuckDB extension payload before its signature must not be empty")
    return hashlib.sha256(chunk_hashes).digest()


def sign_test_extension(source: Path, destination: Path, private_key: Path) -> None:
    artifact = source.read_bytes()
    if len(artifact) <= _DUCKDB_SIGNATURE_SIZE:
        raise ValueError(f"DuckDB extension is too small to contain a signature footer: {source}")
    payload = artifact[:-_DUCKDB_SIGNATURE_SIZE]
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="vane-test-extension-signature-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        digest_path = temporary_root / "extension.sha256"
        signature_path = temporary_root / "extension.signature"
        digest_path.write_bytes(_extension_hash(payload))
        subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-sign",
                "-in",
                str(digest_path),
                "-inkey",
                str(private_key),
                "-pkeyopt",
                "digest:sha256",
                "-out",
                str(signature_path),
            ],
            check=True,
        )
        signature = signature_path.read_bytes()

    if len(signature) != _DUCKDB_SIGNATURE_SIZE:
        raise ValueError(f"test signing key produced {len(signature)} bytes; expected {_DUCKDB_SIGNATURE_SIZE}")
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as output_file:
        temporary_output = Path(output_file.name)
        output_file.write(payload)
        output_file.write(signature)
    try:
        temporary_output.chmod(source.stat().st_mode & 0o777)
        os.replace(temporary_output, destination)
    finally:
        temporary_output.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--private-key", required=True, type=Path)
    arguments = parser.parse_args()
    sign_test_extension(arguments.source, arguments.destination, arguments.private_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Materialize one signed Vane extension into an explicit local cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

sys.path[:] = [import_path for import_path in sys.path if Path(import_path or ".").resolve() != REPOSITORY_ROOT]

from vane.extension_repository import SignedExtensionRepository, read_ed25519_public_key


def main() -> int:
    """Materialize and validate an extension without loading it into DuckDB."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-url", required=True, help="Pinned https:// or file:// repository URL")
    parser.add_argument("--repository-id", required=True, help="Pinned repository identity")
    parser.add_argument("--signer-identity", required=True, help="Pinned Ed25519 signer identity")
    parser.add_argument("--public-key-file", required=True, type=Path, help="Raw, base64, or PEM Ed25519 public key")
    parser.add_argument("--cache-directory", required=True, type=Path, help="Explicit local cache directory")
    parser.add_argument("--extension-name", required=True, help="Extension name")
    parser.add_argument("--extension-version", help="Exact extension version; reject ambiguity when omitted")
    arguments = parser.parse_args()

    repository = SignedExtensionRepository(
        repository_url=arguments.repository_url,
        repository_id=arguments.repository_id,
        signer_identity=arguments.signer_identity,
        trusted_public_key=read_ed25519_public_key(arguments.public_key_file),
        cache_directory=arguments.cache_directory,
    )
    installed = repository.install(arguments.extension_name, extension_version=arguments.extension_version)
    print(
        json.dumps(
            {
                "artifacts": [str(artifact.path) for artifact in installed.artifacts],
                "identity": installed.descriptor.identity,
                "trust_identity": installed.provider.trust_identity,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Publish immutable extension-wheel contents to a signed Vane repository."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

sys.path[:] = [import_path for import_path in sys.path if Path(import_path or ".").resolve() != REPOSITORY_ROOT]

from vane.extension_repository import publish_extension_repository, read_ed25519_private_key


def main() -> int:
    """Publish one or more independently built extension wheels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extension-wheel",
        action="append",
        required=True,
        type=Path,
        help="One independent Vane extension platform wheel; pass once per artifact",
    )
    parser.add_argument("--output-directory", required=True, type=Path, help="Signed repository output directory")
    parser.add_argument("--repository-id", required=True, help="Pinned repository identity")
    parser.add_argument("--signer-identity", required=True, help="Descriptor and Ed25519 signer identity")
    parser.add_argument("--signing-key-file", required=True, type=Path, help="Raw, base64, or PEM Ed25519 private key")
    arguments = parser.parse_args()

    published = publish_extension_repository(
        extension_wheels=arguments.extension_wheel,
        output_directory=arguments.output_directory,
        repository_id=arguments.repository_id,
        signer_identity=arguments.signer_identity,
        private_key=read_ed25519_private_key(arguments.signing_key_file),
    )
    print(
        json.dumps(
            {
                "public_key": published.public_key_base64,
                "repository_id": published.repository_id,
                "repository_path": str(published.path),
                "signer_identity": published.signer_identity,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Package one staged Vane extension artifact as an independent platform wheel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ORIGINAL_SYS_PATH = sys.path.copy()
try:
    sys.path.insert(0, str(REPOSITORY_ROOT))
    from vane_packaging.extension_wheel import build_extension_wheel
finally:
    sys.path[:] = _ORIGINAL_SYS_PATH
    del _ORIGINAL_SYS_PATH


def main() -> int:
    """Build an extension wheel from one explicit self-contained artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path, help="Path to <extension>.duckdb_extension")
    parser.add_argument("--extension-name", required=True, help="Lowercase DuckDB extension name")
    parser.add_argument("--output-directory", required=True, type=Path, help="Directory for the generated wheel")
    parser.add_argument(
        "--platform-tag",
        required=True,
        help="Exact wheel platform tag, such as manylinux_2_28_x86_64 or macosx_11_0_arm64",
    )
    parser.add_argument("--trust-identity", required=True, help="Descriptor trust identity")
    parser.add_argument(
        "--dependency-wheel",
        action="append",
        default=[],
        type=Path,
        help="Complete dependency-wheel closure in load order; repeat once per wheel",
    )
    parser.add_argument(
        "--dependency-trust-identity",
        action="append",
        default=[],
        help="Explicitly trusted dependency signer identity; repeat once per unique identity",
    )
    parser.add_argument(
        "--license-expression",
        required=True,
        help="SPDX license expression covering the extension wheel",
    )
    parser.add_argument(
        "--license-file",
        action="append",
        required=True,
        type=Path,
        help="License file required by the artifact; pass once for each file",
    )
    arguments = parser.parse_args()

    built = build_extension_wheel(
        artifact=arguments.artifact,
        extension_name=arguments.extension_name,
        output_directory=arguments.output_directory,
        platform_tag=arguments.platform_tag,
        trust_identity=arguments.trust_identity,
        license_expression=arguments.license_expression,
        license_files=arguments.license_file,
        dependency_wheels=arguments.dependency_wheel,
        dependency_trust_identities=arguments.dependency_trust_identity,
    )
    print(built.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

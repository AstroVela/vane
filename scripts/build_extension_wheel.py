#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Package one staged Vane extension artifact as an independent platform wheel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from vane_packaging.extension_wheel import build_extension_wheel

sys.path[:] = [import_path for import_path in sys.path if Path(import_path or ".").resolve() != REPOSITORY_ROOT]


def main() -> int:
    """Build an extension wheel from one explicit self-contained artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path, help="Path to <extension>.duckdb_extension")
    parser.add_argument("--extension-name", required=True, help="Lowercase DuckDB extension name")
    parser.add_argument("--output-directory", required=True, type=Path, help="Directory for the generated wheel")
    parser.add_argument(
        "--platform-tag",
        required=True,
        help="Exact wheel platform tag, such as linux_x86_64 or manylinux_2_28_x86_64",
    )
    parser.add_argument("--trust-identity", required=True, help="Descriptor trust identity")
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
    )
    print(built.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

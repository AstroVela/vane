#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Install a base Vane wheel and extension wheel in a clean environment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import textwrap
import venv
import zipfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from vane_packaging.extension_wheel import ENTRY_POINT_GROUP

sys.path[:] = [import_path for import_path in sys.path if Path(import_path or ".").resolve() != REPOSITORY_ROOT]


def _run(command: list[str], *, cwd: Path, environment: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def _python_path(environment_directory: Path) -> Path:
    executable_name = "python.exe" if os.name == "nt" else "python"
    return environment_directory / ("Scripts" if os.name == "nt" else "bin") / executable_name


def _assert_base_wheel_is_artifact_free(base_wheel: Path) -> None:
    with zipfile.ZipFile(base_wheel) as wheel:
        artifacts = [name for name in wheel.namelist() if name.endswith(".duckdb_extension")]
    if artifacts:
        raise RuntimeError(f"base Vane wheel must not contain dynamic extension artifacts: {artifacts}")


def verify_extension_wheel(
    *,
    base_wheel: str | Path,
    extension_wheel: str | Path,
    extension_name: str,
    trust_identity: str,
) -> None:
    """Verify clean installation, metadata discovery, and local artifact loading."""
    resolved_base_wheel = Path(base_wheel).expanduser().resolve(strict=True)
    resolved_extension_wheel = Path(extension_wheel).expanduser().resolve(strict=True)
    _assert_base_wheel_is_artifact_free(resolved_base_wheel)

    with tempfile.TemporaryDirectory(prefix="vane-extension-wheel-") as temporary_directory:
        workspace = Path(temporary_directory)
        environment_directory = workspace / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment_directory)
        python = _python_path(environment_directory)
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONOPTIMIZE", None)
        environment.pop("VIRTUAL_ENV", None)
        environment["PYTHONSAFEPATH"] = "1"
        _run(
            [
                str(python),
                "-m",
                "pip",
                "--disable-pip-version-check",
                "install",
                str(resolved_base_wheel),
                str(resolved_extension_wheel),
            ],
            cwd=workspace,
            environment=environment,
        )

        validation = textwrap.dedent(
            f"""
            from importlib import import_module
            from importlib.metadata import entry_points

            import vane
            from vane.extensions import DynamicExtensionResolver

            providers = [
                entry_point
                for entry_point in entry_points(group={ENTRY_POINT_GROUP!r})
                if entry_point.name == {extension_name!r}
            ]
            assert len(providers) == 1, providers
            provider_entry_point = providers[0]
            provider = provider_entry_point.load()()
            descriptor = import_module(provider_entry_point.module).descriptor()
            assert descriptor.name == {extension_name!r}
            artifact = provider.find(descriptor.identity)
            assert artifact is not None
            assert artifact.path.name == {extension_name + ".duckdb_extension"!r}

            connection = vane.connect(config={{"allow_unsigned_extensions": "true"}})
            try:
                resolved = DynamicExtensionResolver(
                    trusted_identities={{{trust_identity!r}}},
                    providers=(provider,),
                ).load(connection, descriptor)
                assert resolved.identity == descriptor.identity
                assert connection.execute(
                    "SELECT loaded FROM duckdb_extensions() WHERE extension_name = ?", [descriptor.name]
                ).fetchone() == (True,)
            finally:
                connection.close()
            """
        )
        program = f"exec(compile({validation!r}, '<extension-wheel-validation>', 'exec', optimize=0))"
        _run([str(python), "-I", "-c", program], cwd=workspace, environment=environment)


def main() -> int:
    """Run clean-install verification for one extension wheel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-wheel", required=True, type=Path)
    parser.add_argument("--extension-wheel", required=True, type=Path)
    parser.add_argument("--extension-name", required=True)
    parser.add_argument("--trust-identity", required=True)
    arguments = parser.parse_args()
    verify_extension_wheel(
        base_wheel=arguments.base_wheel,
        extension_wheel=arguments.extension_wheel,
        extension_name=arguments.extension_name,
        trust_identity=arguments.trust_identity,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

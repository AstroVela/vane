#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise install and uninstall ordering with the official DuckDB wheel."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import textwrap
import venv
from pathlib import Path

OFFICIAL_DUCKDB_REQUIREMENT = "duckdb==1.5.4"
LEGACY_CONFLICTING_VANE_REQUIREMENT = "vane-ai==0.1.0a1"


def _environment_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def _run(python: Path, *arguments: str, cwd: Path) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("VANE_RUNNER", None)
    environment["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        [str(python), *arguments],
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        command = " ".join((str(python), *arguments))
        raise RuntimeError(f"command failed ({result.returncode}): {command}\n{result.stdout}{result.stderr}")


def _install(python: Path, requirement: str, *, cwd: Path, force_reinstall: bool = False) -> None:
    arguments = [
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--no-deps",
        "--quiet",
    ]
    if force_reinstall:
        arguments.append("--force-reinstall")
    if requirement.startswith(("duckdb==", "vane-ai==")):
        arguments.append("--only-binary=:all:")
    _run(python, *arguments, requirement, cwd=cwd)


def _uninstall(python: Path, distribution_name: str, *, cwd: Path) -> None:
    _run(
        python,
        "-m",
        "pip",
        "uninstall",
        "--yes",
        "--disable-pip-version-check",
        distribution_name,
        cwd=cwd,
    )


def _probe_both(python: Path, *, cwd: Path, vane_first: bool) -> None:
    imports = "import vane, duckdb" if vane_first else "import duckdb, vane"
    code = textwrap.dedent(
        f"""
        from importlib.metadata import distribution, version
        {imports}
        import _vane_duckdb
        from vane import _duckdb_func

        def owned_files(name):
            return {{str(path).replace('\\\\', '/') for path in distribution(name).files or []}}

        vane_files = owned_files('vane-ai')
        duckdb_files = owned_files('duckdb')
        overlap = vane_files & duckdb_files
        assert not overlap, sorted(overlap)

        vane_top_level = {{path.split('/', 1)[0] for path in vane_files}}
        assert {{'vane', '_vane_duckdb-stubs', 'vane_adbc_driver_duckdb'}} <= vane_top_level
        assert not ({{'duckdb', '_duckdb-stubs', 'adbc_driver_duckdb'}} & vane_top_level)
        assert not any(name == '_duckdb' or name.startswith('_duckdb.') for name in vane_top_level)

        assert vane.__version__ == version('vane-ai')
        assert vane.__duckdb_version__ == _vane_duckdb.__version__
        assert callable(vane.func)
        assert _duckdb_func.NATIVE is not None
        assert vane.STRING is not None and vane.NUMBER is not None
        assert duckdb.__version__ == version('duckdb')
        assert vane.DuckDBPyConnection is not duckdb.DuckDBPyConnection
        assert vane.ray_cxx is _vane_duckdb.ray_cxx
        assert vane.vane_runners_cpp is _vane_duckdb
        assert vane.sql('select 40 + 2').fetchone() == (42,)
        assert duckdb.sql('select 40 + 2').fetchone() == (42,)
        """
    )
    _run(python, "-c", code, cwd=cwd)


def _probe_vane(python: Path, *, cwd: Path) -> None:
    code = textwrap.dedent(
        """
        from importlib.metadata import PackageNotFoundError, version
        import vane
        import _vane_duckdb

        assert vane.__version__ == version('vane-ai')
        assert vane.__duckdb_version__ == _vane_duckdb.__version__
        assert vane.sql('select 42').fetchone() == (42,)
        try:
            version('duckdb')
        except PackageNotFoundError:
            pass
        else:
            raise AssertionError('official DuckDB metadata survived uninstall')
        """
    )
    _run(python, "-c", code, cwd=cwd)


def _probe_private_extension_first(python: Path, *, cwd: Path) -> None:
    code = textwrap.dedent(
        """
        import importlib

        import _vane_duckdb
        import vane

        assert vane.ray_cxx is _vane_duckdb.ray_cxx
        assert vane.vane_runners_cpp is _vane_duckdb
        assert vane.vane_runners is _vane_duckdb
        assert importlib.import_module('vane.ray_cxx') is _vane_duckdb.ray_cxx
        assert importlib.import_module('vane.vane_runners_cpp') is _vane_duckdb
        assert callable(vane.vane_runners_cpp.get_or_infer_runner_type)
        """
    )
    _run(python, "-c", code, cwd=cwd)


def _probe_official_duckdb(python: Path, *, cwd: Path) -> None:
    code = textwrap.dedent(
        """
        from importlib.metadata import PackageNotFoundError, version
        import duckdb

        assert duckdb.__version__ == version('duckdb')
        assert duckdb.sql('select 42').fetchone() == (42,)
        try:
            version('vane-ai')
        except PackageNotFoundError:
            pass
        else:
            raise AssertionError('Vane metadata survived uninstall')
        """
    )
    _run(python, "-c", code, cwd=cwd)


def _run_scenario(
    vane_wheel: Path,
    *,
    install_vane_first: bool,
    uninstall_vane: bool,
) -> None:
    with tempfile.TemporaryDirectory(prefix="vane-duckdb-coexist-") as temporary_directory:
        root = Path(temporary_directory)
        environment = root / "environment"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        python = _environment_python(environment)

        packages = (str(vane_wheel), OFFICIAL_DUCKDB_REQUIREMENT)
        if not install_vane_first:
            packages = tuple(reversed(packages))
        for package in packages:
            _install(python, package, cwd=root)

        _probe_both(python, cwd=root, vane_first=install_vane_first)
        _probe_private_extension_first(python, cwd=root)
        if uninstall_vane:
            _uninstall(python, "vane-ai", cwd=root)
            _probe_official_duckdb(python, cwd=root)
        else:
            _uninstall(python, "duckdb", cwd=root)
            _probe_vane(python, cwd=root)


def _run_legacy_upgrade_recovery(vane_wheel: Path) -> None:
    """Verify the documented repair after upgrading the conflicting alpha wheel."""
    with tempfile.TemporaryDirectory(prefix="vane-duckdb-legacy-upgrade-") as temporary_directory:
        root = Path(temporary_directory)
        environment = root / "environment"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        python = _environment_python(environment)

        _install(python, LEGACY_CONFLICTING_VANE_REQUIREMENT, cwd=root)
        _install(python, OFFICIAL_DUCKDB_REQUIREMENT, cwd=root)

        # Upgrading removes files listed in the legacy Vane RECORD, including
        # paths now owned by the official DuckDB wheel. Reinstalling DuckDB is
        # the documented repair after the private-name Vane wheel is installed.
        _install(python, str(vane_wheel), cwd=root, force_reinstall=True)
        _install(python, OFFICIAL_DUCKDB_REQUIREMENT, cwd=root, force_reinstall=True)
        _probe_both(python, cwd=root, vane_first=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vane_wheel", type=Path)
    args = parser.parse_args()
    vane_wheel = args.vane_wheel.resolve(strict=True)
    if vane_wheel.suffix != ".whl":
        parser.error(f"expected a wheel, got {vane_wheel}")

    for install_vane_first in (False, True):
        for uninstall_vane in (False, True):
            _run_scenario(
                vane_wheel,
                install_vane_first=install_vane_first,
                uninstall_vane=uninstall_vane,
            )
            install_order = "vane,duckdb" if install_vane_first else "duckdb,vane"
            removed = "vane-ai" if uninstall_vane else "duckdb"
            print(f"passed install order {install_order}; removed {removed}")
    _run_legacy_upgrade_recovery(vane_wheel)
    print(f"passed upgrade recovery from {LEGACY_CONFLICTING_VANE_REQUIREMENT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

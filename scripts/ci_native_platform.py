#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Platform adapters for the shared native Python CI test recipe."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from packaging.tags import sys_tags


def platform_tag() -> str:
    prefixes = {"linux": "manylinux_", "darwin": "macosx_", "win32": "win_"}
    return next(tag.platform for tag in sys_tags() if tag.platform.startswith(prefixes[sys.platform]))


def artifact(directory: Path, name: str) -> Path:
    paths = sorted(path for path in directory.rglob(name) if path.is_file())
    if len(paths) != 1:
        raise ValueError(f"expected exactly one {name} under {directory}: {paths}")
    return paths[0].resolve()


def windows_build_settings(prefix: Path) -> list[str]:
    if sys.platform != "win32":
        return []
    root = Path(os.environ["VCPKG_ROOT"])
    return [
        f"-Ccmake.define.CMAKE_TOOLCHAIN_FILE={root.as_posix()}/scripts/buildsystems/vcpkg.cmake",
        f"-Ccmake.define.VCPKG_INSTALLED_DIR={prefix.resolve().as_posix()}",
        f"-Ccmake.define.VCPKG_TARGET_TRIPLET={os.environ['VCPKG_TARGET_TRIPLET']}",
        "-Ccmake.define.VCPKG_MANIFEST_MODE=OFF",
        "-Ccmake.define.CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded",
        "-Ccmake.define.DISABLE_UNITY=1",
        "-Cbuild.tool-args=/m",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("platform-tag")
    locate = commands.add_parser("artifact")
    locate.add_argument("directory", type=Path)
    locate.add_argument("name")
    commands.add_parser("build-release")
    commands.add_parser("install-vanefs")
    repair = commands.add_parser("repair-base")
    repair.add_argument("wheel", type=Path)
    repair.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.command == "platform-tag":
        print(platform_tag())
    elif args.command == "artifact":
        print(artifact(args.directory, args.name).as_posix())
    elif args.command == "build-release":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--sdist",
                "--wheel",
                "--no-isolation",
                "--outdir",
                "dist",
                *windows_build_settings(Path("vcpkg_installed")),
            ],
            check=True,
        )
    elif args.command == "install-vanefs":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "./vane_fs[test]",
                "--no-build-isolation",
                *windows_build_settings(Path("vane_fs/vcpkg_installed")),
            ],
            check=True,
        )
    elif args.command == "repair-base":
        args.destination.mkdir(parents=True, exist_ok=True)
        if sys.platform == "linux":
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "auditwheel",
                    "repair",
                    "--plat",
                    platform_tag(),
                    "--wheel-dir",
                    str(args.destination),
                    str(args.wheel),
                ],
                check=True,
            )
        elif sys.platform == "darwin":
            subprocess.run(
                ["delocate-wheel", "--require-archs", "arm64", "--wheel-dir", str(args.destination), str(args.wheel)],
                check=True,
            )
        else:
            # The base Windows build uses the pinned static CRT/dependency
            # triplet. The clean extension verifier still inspects its imports.
            shutil.copyfile(args.wheel, args.destination / args.wheel.name)


if __name__ == "__main__":
    main()

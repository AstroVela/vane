# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the recorded 16 MiB admission experiment without changing the checkout."""

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

BASELINE = "f4f76ddce4f52e8818316c09e7378899d7c8c74b"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variant", choices=("admission-16",), default="admission-16")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must not exist")
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    artifacts = here
    expected = json.loads((artifacts / "source-inputs.json").read_text())
    names = (
        "src",
        "include",
        "python",
        "tests",
        "triplets",
        "LICENSES",
        "CMakeLists.txt",
        "pyproject.toml",
        "vcpkg.json",
        "README.md",
        "DESIGN.md",
        "LICENSE",
    )
    archive = subprocess.check_output(["git", "archive", BASELINE, "--", *[f"vane_fs/{n}" for n in names]], cwd=repo)
    output = args.output.resolve()
    baseline = output / "baseline"
    baseline.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for member in source:
            parts = Path(member.name).parts
            assert parts and parts[0] == "vane_fs" and ".." not in parts
            path = baseline.joinpath(*parts[1:])
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                assert member.isfile(), member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as data:
                    path.write_bytes(data.read())
                path.chmod(member.mode & 0o777)
    candidate = output / args.variant
    shutil.copytree(baseline, candidate)
    patch = subprocess.run(
        ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(artifacts / f"{args.variant}.patch")],
        cwd=candidate,
        check=True,
        capture_output=True,
        text=True,
    )
    (output / "patch.log").write_text(patch.stdout + patch.stderr)
    report = {"baseline_revision": BASELINE, "files": {}}
    for variant in ("baseline", args.variant):
        actual = {}
        target = output / variant
        for directory in ("src", "include", "python", "tests"):
            for path in (target / directory).rglob("*"):
                if path.is_file():
                    actual[str(path.relative_to(target))] = digest(path)
        for name in ("CMakeLists.txt", "pyproject.toml", "vcpkg.json"):
            actual[name] = digest(target / name)
        assert actual == expected[variant], variant
        report["files"][variant] = actual
    (output / "source-inputs.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

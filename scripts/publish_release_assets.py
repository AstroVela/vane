#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate the nested base-release assets and attach exact bytes to a draft."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

SUPPLEMENTAL = {"SHA256SUMS", "vane-ai-sbom.cdx.json", "vane-ai-build-provenance.sigstore.json"}
# Keep the base release contract aligned with release.yml's assemble job.
WHEEL_COUNT = 5


@dataclass(frozen=True)
class Asset:
    path: Path
    size: int
    sha256: str


def collect_assets(directory: Path) -> dict[str, Asset]:
    """Flatten regular files, preserving their paths for upload and rejecting links."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("release asset root must be a real directory")
    pending = [directory]
    assets = {}
    while pending:
        for path in sorted(pending.pop().iterdir()):
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                pending.append(path)
                continue
            if not stat.S_ISREG(mode):
                raise ValueError(f"release asset is not a regular file: {path}")
            if path.name in assets:
                raise ValueError(f"duplicate release asset basename: {path.name}")
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    size += len(chunk)
                    digest.update(chunk)
            if not size:
                raise ValueError(f"empty release asset: {path.name}")
            assets[path.name] = Asset(path, size, digest.hexdigest())
    return assets


def validate_assets(assets: dict[str, Asset]) -> str:
    """Require all distributions, signatures and supplemental files from assemble."""
    sdists = {name for name in assets if name.startswith("vane_ai-") and name.endswith(".tar.gz")}
    wheels = {name for name in assets if name.startswith("vane_ai-") and name.endswith(".whl")}
    if len(sdists) != 1 or len(wheels) != WHEEL_COUNT:
        raise ValueError(f"expected one source archive and {WHEEL_COUNT} wheels")
    version = next(iter(sdists)).removeprefix("vane_ai-").removesuffix(".tar.gz")
    if not re.fullmatch(r"[0-9][A-Za-z0-9.+]*", version) or any(
        not name.startswith(f"vane_ai-{version}-") for name in wheels
    ):
        raise ValueError("release distributions must use the same version")
    distributions = sdists | wheels
    expected = distributions | {f"{name}.sigstore.json" for name in distributions} | SUPPLEMENTAL
    if set(assets) != expected:
        raise ValueError(
            f"incomplete release inventory: missing={sorted(expected - assets.keys())}, "
            f"unexpected={sorted(assets.keys() - expected)}"
        )
    checksums = {}
    for line in assets["SHA256SUMS"].path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *]([^/\\\r\n]+)", line)
        if not match or match[2] in checksums:
            raise ValueError("invalid or duplicate SHA256SUMS entry")
        checksums[match[2]] = match[1]
    if set(checksums) != distributions or any(assets[name].sha256 != digest for name, digest in checksums.items()):
        raise ValueError("SHA256SUMS does not match the complete distribution set")
    return version


def _gh_json(*arguments: str):
    result = subprocess.run(["gh", *arguments], check=True, stdout=subprocess.PIPE, text=True)
    return json.loads(result.stdout)


def _release(repository: str, tag: str) -> dict:
    description = _gh_json("release", "view", tag, "--repo", repository, "--json", "apiUrl,isDraft")
    if description["isDraft"] is not True:
        raise ValueError("release must still be a draft")
    prefix = f"https://api.github.com/repos/{repository}/releases/"
    url = description["apiUrl"]
    if not url.lower().startswith(prefix.lower()) or not url[len(prefix) :].isdigit():
        raise ValueError("unexpected release API location")
    release = _gh_json("api", f"repos/{repository}/releases/{url[len(prefix) :]}")
    if release["draft"] is not True or release["tag_name"] != tag:
        raise ValueError("release draft identity changed")
    return release


def verify_remote(assets: dict[str, Asset], release: dict, *, complete: bool) -> set[str]:
    existing = {}
    for remote in release["assets"]:
        name = remote["name"]
        if name in existing or name not in assets:
            raise ValueError(f"unexpected or duplicate remote release asset: {name}")
        expected = assets[name]
        if (
            remote.get("state") != "uploaded"
            or remote.get("size") != expected.size
            or remote.get("digest") != f"sha256:{expected.sha256}"
        ):
            raise ValueError(f"remote release asset differs from accepted bytes: {name}")
        existing[name] = remote
    missing = assets.keys() - existing.keys()
    if complete and missing:
        raise ValueError(f"release is missing uploaded assets: {sorted(missing)}")
    return missing


def publish_assets(directory: Path, repository: str, tag: str) -> int:
    assets = collect_assets(directory)
    version = validate_assets(assets)
    if tag != f"v{version}":
        raise ValueError("release tag differs from the distribution version")
    release = _release(repository, tag)
    missing = verify_remote(assets, release, complete=False)
    # Keep the original hashes through upload and verification. Never replace
    # an asset already attached by an earlier attempt at the same release.
    for name in sorted(missing):
        subprocess.run(["gh", "release", "upload", tag, str(assets[name].path), "--repo", repository], check=True)
    uploaded = _release(repository, tag)
    if uploaded["id"] != release["id"]:
        raise ValueError("release changed during asset upload")
    verify_remote(assets, uploaded, complete=True)
    return len(missing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check")
    check.add_argument("--directory", type=Path, required=True)
    publish = subparsers.add_parser("publish")
    publish.add_argument("--directory", type=Path, required=True)
    publish.add_argument("--repository", required=True)
    publish.add_argument("--tag", required=True)
    args = parser.parse_args()
    if args.command == "check":
        assets = collect_assets(args.directory)
        version = validate_assets(assets)
        print(json.dumps({"version": version, "assets": len(assets), "verified": True}))
    else:
        uploaded = publish_assets(args.directory, args.repository, args.tag)
        print(json.dumps({"uploaded": uploaded, "verified": True}))


if __name__ == "__main__":
    main()

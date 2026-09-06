#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Collect and verify the GitHub Release asset set.

The release job downloads two artifacts into one directory and attaches every
regular file under it to the draft release. Two things have to hold before the
draft is published, and neither is expressible as a one-line shell test:

* The local tree must carry the complete expected inventory -- one sdist, five
  wheels, a Sigstore bundle for each of those six distributions, ``SHA256SUMS``,
  the CycloneDX SBOM and the provenance bundle. Fifteen assets, no more.
* Every one of them must be on the release *with the bytes it has locally*.
  Name equality is not enough: an upload truncated by a dropped connection, or
  a stale asset left over from a re-run of an earlier build, both produce a
  release whose asset *names* look right.

Release assets are a flat namespace keyed by basename, so the nested tree is
flattened on upload; a basename collision would silently overwrite under
``--clobber`` and publish a release quietly missing an asset. That is checked
here too, before anything is uploaded.

Usage::

    check_release_assets.py collect release-files --output assets.txt
    check_release_assets.py verify release-files --remote remote.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

# One sdist plus five wheels, each with its own Sigstore bundle, plus the three
# supplemental assets below.
EXPECTED_SDISTS = 1
EXPECTED_WHEELS = 5
EXPECTED_DISTRIBUTIONS = EXPECTED_SDISTS + EXPECTED_WHEELS
SHA256SUMS = "SHA256SUMS"
SBOM = "vane-ai-sbom.cdx.json"
PROVENANCE = "vane-ai-build-provenance.sigstore.json"
SUPPLEMENTAL = (SHA256SUMS, SBOM, PROVENANCE)
EXPECTED_ASSET_COUNT = EXPECTED_DISTRIBUTIONS * 2 + len(SUPPLEMENTAL)

SIGSTORE_SUFFIX = ".sigstore.json"
READ_CHUNK = 1024 * 1024


class AssetError(Exception):
    """A release asset problem worth failing the workflow for."""


@dataclass(frozen=True)
class Asset:
    """One asset reduced to the three fields both sides can agree on."""

    name: str
    size: int
    digest: str  # lowercase hex sha256, no algorithm prefix

    def describe(self) -> str:
        return f"{self.name} ({self.size} bytes, sha256:{self.digest})"


def sha256_of(path: Path) -> str:
    """Hash a file without reading it entirely into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(READ_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(root: Path) -> list[Path]:
    """Return every regular file under ``root``, sorted, following no symlinks.

    ``Path.rglob`` yields directories too, and the bug this script exists to
    prevent was precisely a directory being handed to ``gh release upload``.
    Symlinks are excluded rather than resolved: an asset that is a link out of
    the tree is not something the release should publish silently.
    """

    if not root.is_dir():
        raise AssetError(f"{root} is not a directory")
    files = [path for path in root.rglob("*") if path.is_file() and not path.is_symlink()]
    return sorted(files)


def local_manifest(paths: Sequence[Path]) -> dict[str, Asset]:
    """Reduce local files to a basename-keyed manifest, rejecting collisions."""

    if not paths:
        raise AssetError("no regular files found to upload")

    seen: dict[str, list[Path]] = {}
    for path in paths:
        seen.setdefault(path.name, []).append(path)

    collisions = {name: found for name, found in seen.items() if len(found) > 1}
    if collisions:
        lines = ["asset basenames collide, refusing to upload:"]
        for name in sorted(collisions):
            joined = ", ".join(str(path) for path in sorted(collisions[name]))
            lines.append(f"  {name}: {joined}")
        raise AssetError("\n".join(lines))

    return {path.name: Asset(name=path.name, size=path.stat().st_size, digest=sha256_of(path)) for path in paths}


def remote_manifest(payload: str) -> dict[str, Asset]:
    """Parse ``gh release view --json assets`` output into the same shape.

    ``gh`` reports the digest as ``sha256:<hex>``; older releases and assets
    still processing report an empty string. An asset we cannot compare is
    reported rather than skipped -- silently passing it is the fail-open this
    check exists to close.
    """

    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AssetError(f"could not parse the release JSON: {exc}") from exc

    assets = document.get("assets") if isinstance(document, dict) else document
    if not isinstance(assets, list):
        raise AssetError("release JSON has no 'assets' array")

    manifest: dict[str, Asset] = {}
    for entry in assets:
        name = entry.get("name")
        if not name:
            raise AssetError(f"release asset without a name: {entry!r}")
        digest = str(entry.get("digest") or "")
        if not digest.startswith("sha256:"):
            raise AssetError(f"release asset {name} reports digest {digest!r}, expected 'sha256:<hex>'")
        manifest[name] = Asset(
            name=name,
            size=int(entry.get("size", -1)),
            digest=digest[len("sha256:") :].lower(),
        )
    return manifest


def check_inventory(names: Iterable[str]) -> None:
    """Assert the expected 15-asset inventory, naming what is off.

    Checked by shape rather than by pinned filename so a version bump does not
    have to touch this script, but strictly enough that a missing wheel or an
    unsigned distribution fails.
    """

    names = set(names)
    problems: list[str] = []

    sdists = {name for name in names if name.endswith(".tar.gz")}
    wheels = {name for name in names if name.endswith(".whl")}
    distributions = sdists | wheels

    if len(sdists) != EXPECTED_SDISTS:
        problems.append(f"expected {EXPECTED_SDISTS} sdist, found {len(sdists)}: {sorted(sdists)}")
    if len(wheels) != EXPECTED_WHEELS:
        problems.append(f"expected {EXPECTED_WHEELS} wheels, found {len(wheels)}: {sorted(wheels)}")

    # Every distribution must carry its own bundle: sigstore-python names the
    # bundle after its input, so this pairs them exactly rather than counting.
    for distribution in sorted(distributions):
        bundle = f"{distribution}{SIGSTORE_SUFFIX}"
        if bundle not in names:
            problems.append(f"missing Sigstore bundle for {distribution}: expected {bundle}")

    for required in SUPPLEMENTAL:
        if required not in names:
            problems.append(f"missing supplemental asset: {required}")

    expected = (
        distributions | {f"{distribution}{SIGSTORE_SUFFIX}" for distribution in distributions} | set(SUPPLEMENTAL)
    )
    unexpected = names - expected
    if unexpected:
        problems.append(f"unexpected assets: {sorted(unexpected)}")

    if len(names) != EXPECTED_ASSET_COUNT:
        problems.append(f"expected {EXPECTED_ASSET_COUNT} unique assets, found {len(names)}")

    if problems:
        raise AssetError("release asset inventory is wrong:\n" + "\n".join(f"  {p}" for p in problems))


def compare_manifests(local: dict[str, Asset], remote: dict[str, Asset]) -> None:
    """Require the two manifests to be equal, reporting every difference.

    A subset check would pass with stale extra assets left on the release by an
    earlier run, and a name-only check would pass with a truncated upload.
    """

    problems: list[str] = []

    for name in sorted(set(local) - set(remote)):
        problems.append(f"missing from the release: {local[name].describe()}")
    for name in sorted(set(remote) - set(local)):
        problems.append(f"unexpected asset on the release: {remote[name].describe()}")

    for name in sorted(set(local) & set(remote)):
        want, got = local[name], remote[name]
        if want.size != got.size:
            problems.append(f"size mismatch for {name}: local {want.size}, release {got.size}")
        if want.digest != got.digest:
            problems.append(f"digest mismatch for {name}: local sha256:{want.digest}, release sha256:{got.digest}")

    if problems:
        raise AssetError(
            "the release asset set does not match what was built:\n" + "\n".join(f"  {p}" for p in problems)
        )


def _emit_error(message: str) -> None:
    for line in message.splitlines():
        print(f"::error::{line}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    collect = subcommands.add_parser("collect", help="list the files to upload")
    collect.add_argument("root", type=Path)
    collect.add_argument(
        "--output",
        type=Path,
        required=True,
        help="write the NUL-delimited file list here for the caller to read",
    )

    verify = subcommands.add_parser("verify", help="compare local files against the release")
    verify.add_argument("root", type=Path)
    verify.add_argument(
        "--remote",
        type=Path,
        required=True,
        help="'gh release view --json assets' output ('-' for stdin)",
    )

    args = parser.parse_args(argv)

    try:
        files = collect_files(args.root)
        manifest = local_manifest(files)
        check_inventory(manifest)

        if args.command == "collect":
            # NUL-delimited: an asset name may contain anything but a slash,
            # and a newline-delimited list would split it in half.
            args.output.write_bytes(b"".join(f"{path}\0".encode() for path in files))
            print(f"collected {len(files)} asset(s):")
            for path in files:
                print(f"  {path}")
            return 0

        payload = sys.stdin.read() if str(args.remote) == "-" else args.remote.read_text(encoding="utf-8")
        compare_manifests(manifest, remote_manifest(payload))
        print(f"verified {len(manifest)} asset(s) on the release by name, size and sha256")
        return 0
    except AssetError as exc:
        _emit_error(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

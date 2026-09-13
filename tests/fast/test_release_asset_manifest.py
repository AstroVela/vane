# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Cover the release asset collection and verification contract.

The bug in #550 was that ``release-files/*`` handed ``gh release upload`` a
directory, which it rejects only after uploading the assets ahead of it. The
checks here pin the shape of the fixed tree walk and, more importantly, the
verification that follows it -- the part that has to fail closed, since every
one of these cases previously produced a green step.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import check_release_assets
from scripts.check_release_assets import AssetError

SDIST = "vane_ai-0.1.0.tar.gz"
WHEELS = (
    "vane_ai-0.1.0-cp310-cp310-manylinux_2_28_x86_64.whl",
    "vane_ai-0.1.0-cp311-cp311-manylinux_2_28_x86_64.whl",
    "vane_ai-0.1.0-cp312-cp312-manylinux_2_28_x86_64.whl",
    "vane_ai-0.1.0-cp313-cp313-manylinux_2_28_x86_64.whl",
    "vane_ai-0.1.0-cp314-cp314-manylinux_2_28_x86_64.whl",
)
DISTRIBUTIONS = (SDIST, *WHEELS)
SUPPLEMENTAL = (
    check_release_assets.SHA256SUMS,
    check_release_assets.SBOM,
    check_release_assets.PROVENANCE,
)


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _release_tree(root: Path) -> Path:
    """Build the real post-download layout: two artifacts, nested directories.

    ``release-distributions`` unpacks flat into ``release-files/``, while
    ``release-supplemental`` keeps the ``packages/`` and ``release-assets/``
    prefixes it was packed with -- the nesting that broke the original glob.
    """

    files = root / "release-files"
    for distribution in DISTRIBUTIONS:
        _write(files / distribution, f"payload of {distribution}".encode())
        _write(
            files / "packages" / f"{distribution}.sigstore.json",
            f"bundle for {distribution}".encode(),
        )
    for name in SUPPLEMENTAL:
        _write(files / "release-assets" / name, f"contents of {name}".encode())
    return files


def _remote_json(manifest: dict[str, check_release_assets.Asset]) -> str:
    return json.dumps(
        {"assets": [{"name": a.name, "size": a.size, "digest": f"sha256:{a.digest}"} for a in manifest.values()]}
    )


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    return _release_tree(tmp_path)


def test_collect_walks_into_nested_artifact_directories(tree: Path) -> None:
    """The regression itself: nested files are collected, directories are not."""

    collected = check_release_assets.collect_files(tree)

    assert len(collected) == check_release_assets.EXPECTED_ASSET_COUNT
    assert all(path.is_file() for path in collected)
    names = {path.name for path in collected}
    assert names == set(DISTRIBUTIONS) | {f"{d}.sigstore.json" for d in DISTRIBUTIONS} | set(SUPPLEMENTAL)
    # The two directories the old glob tried to upload.
    assert not any(path.name in {"packages", "release-assets"} for path in collected)


def test_collect_rejects_a_directory_that_is_not_one(tmp_path: Path) -> None:
    with pytest.raises(AssetError, match="is not a directory"):
        check_release_assets.collect_files(tmp_path / "absent")


def test_an_empty_tree_is_refused(tmp_path: Path) -> None:
    """An empty download would otherwise publish a release with no assets."""

    empty = tmp_path / "release-files"
    empty.mkdir()

    with pytest.raises(AssetError, match="no regular files"):
        check_release_assets.local_manifest(check_release_assets.collect_files(empty))


def test_colliding_basenames_are_refused_before_upload(tree: Path) -> None:
    """Assets are a flat namespace; --clobber would drop one of these silently."""

    _write(tree / "release-assets" / SDIST, b"a different file with the same name")

    with pytest.raises(AssetError, match="collide") as caught:
        check_release_assets.local_manifest(check_release_assets.collect_files(tree))

    message = str(caught.value)
    assert SDIST in message
    # Both paths are named, so the operator can tell which one to fix.
    assert message.count(SDIST) >= 2


def test_the_expected_inventory_passes(tree: Path) -> None:
    manifest = check_release_assets.local_manifest(check_release_assets.collect_files(tree))

    check_release_assets.check_inventory(manifest)
    assert len(manifest) == 15


def test_a_missing_wheel_fails_the_inventory(tree: Path) -> None:
    (tree / WHEELS[0]).unlink()
    (tree / "packages" / f"{WHEELS[0]}.sigstore.json").unlink()

    manifest = check_release_assets.local_manifest(check_release_assets.collect_files(tree))
    with pytest.raises(AssetError, match="expected 5 wheels, found 4"):
        check_release_assets.check_inventory(manifest)


def test_an_unsigned_distribution_fails_the_inventory(tree: Path) -> None:
    """A distribution without its Sigstore bundle must not reach the release."""

    (tree / "packages" / f"{SDIST}.sigstore.json").unlink()

    manifest = check_release_assets.local_manifest(check_release_assets.collect_files(tree))
    with pytest.raises(AssetError, match=f"missing Sigstore bundle for {SDIST}"):
        check_release_assets.check_inventory(manifest)


@pytest.mark.parametrize("missing", SUPPLEMENTAL)
def test_a_missing_supplemental_asset_fails_the_inventory(tree: Path, missing: str) -> None:
    (tree / "release-assets" / missing).unlink()

    manifest = check_release_assets.local_manifest(check_release_assets.collect_files(tree))
    with pytest.raises(AssetError, match=f"missing supplemental asset: {missing}"):
        check_release_assets.check_inventory(manifest)


def test_an_unexpected_extra_asset_fails_the_inventory(tree: Path) -> None:
    _write(tree / "release-assets" / "internal-debug.log", b"not for publication")

    manifest = check_release_assets.local_manifest(check_release_assets.collect_files(tree))
    with pytest.raises(AssetError, match="unexpected assets"):
        check_release_assets.check_inventory(manifest)


def test_a_matching_release_verifies(tree: Path) -> None:
    manifest = check_release_assets.local_manifest(check_release_assets.collect_files(tree))

    remote = check_release_assets.remote_manifest(_remote_json(manifest))

    check_release_assets.compare_manifests(manifest, remote)


def test_an_asset_missing_from_the_release_is_reported(tree: Path) -> None:
    manifest = check_release_assets.local_manifest(check_release_assets.collect_files(tree))
    remote = check_release_assets.remote_manifest(_remote_json(manifest))
    del remote[SDIST]

    with pytest.raises(AssetError, match=f"missing from the release: {SDIST}"):
        check_release_assets.compare_manifests(manifest, remote)


def test_a_stale_extra_asset_on_the_release_is_reported(tree: Path) -> None:
    """comm -23 passed this: extra remote names are a superset, not a mismatch."""

    manifest = check_release_assets.local_manifest(check_release_assets.collect_files(tree))
    remote = check_release_assets.remote_manifest(_remote_json(manifest))
    remote["vane_ai-0.0.9.tar.gz"] = check_release_assets.Asset(name="vane_ai-0.0.9.tar.gz", size=10, digest="0" * 64)

    with pytest.raises(AssetError, match="unexpected asset on the release"):
        check_release_assets.compare_manifests(manifest, remote)


def test_a_truncated_upload_is_caught_by_size(tree: Path) -> None:
    """Same name, fewer bytes -- the dropped-connection case."""

    manifest = check_release_assets.local_manifest(check_release_assets.collect_files(tree))
    remote = check_release_assets.remote_manifest(_remote_json(manifest))
    intact = remote[SDIST]
    remote[SDIST] = check_release_assets.Asset(name=SDIST, size=intact.size - 1, digest=intact.digest)

    with pytest.raises(AssetError, match=f"size mismatch for {SDIST}"):
        check_release_assets.compare_manifests(manifest, remote)


def test_a_substituted_asset_is_caught_by_digest(tree: Path) -> None:
    """Right name, right length, different bytes."""

    manifest = check_release_assets.local_manifest(check_release_assets.collect_files(tree))
    remote = check_release_assets.remote_manifest(_remote_json(manifest))
    intact = remote[SDIST]
    remote[SDIST] = check_release_assets.Asset(
        name=SDIST,
        size=intact.size,
        digest=hashlib.sha256(b"substituted").hexdigest(),
    )

    with pytest.raises(AssetError, match=f"digest mismatch for {SDIST}"):
        check_release_assets.compare_manifests(manifest, remote)


def test_every_difference_is_reported_at_once(tree: Path) -> None:
    """One run should list everything wrong, not fail on the first problem."""

    manifest = check_release_assets.local_manifest(check_release_assets.collect_files(tree))
    remote = check_release_assets.remote_manifest(_remote_json(manifest))
    del remote[WHEELS[0]]
    remote["leftover.whl"] = check_release_assets.Asset("leftover.whl", 1, "f" * 64)
    intact = remote[SDIST]
    remote[SDIST] = check_release_assets.Asset(SDIST, intact.size + 5, intact.digest)

    with pytest.raises(AssetError) as caught:
        check_release_assets.compare_manifests(manifest, remote)

    message = str(caught.value)
    assert WHEELS[0] in message
    assert "leftover.whl" in message
    assert f"size mismatch for {SDIST}" in message


def test_an_asset_without_a_digest_is_refused_not_skipped() -> None:
    """gh reports '' for an asset still processing; comparing nothing must not pass."""

    payload = json.dumps({"assets": [{"name": SDIST, "size": 10, "digest": ""}]})

    with pytest.raises(AssetError, match="expected 'sha256:<hex>'"):
        check_release_assets.remote_manifest(payload)


def test_malformed_release_json_is_reported() -> None:
    with pytest.raises(AssetError, match="could not parse"):
        check_release_assets.remote_manifest("{not json")


def test_release_json_without_an_assets_array_is_reported() -> None:
    with pytest.raises(AssetError, match="no 'assets' array"):
        check_release_assets.remote_manifest(json.dumps({"tagName": "v0.1.0"}))


def test_collect_writes_a_nul_delimited_list(tree: Path, tmp_path: Path) -> None:
    """The workflow reads this with `mapfile -d ''`, so the delimiter matters."""

    output = tmp_path / "upload-list"

    assert check_release_assets.main(["collect", str(tree), "--output", str(output)]) == 0

    raw = output.read_bytes()
    assert raw.endswith(b"\0")
    entries = [chunk for chunk in raw.split(b"\0") if chunk]
    assert len(entries) == check_release_assets.EXPECTED_ASSET_COUNT
    assert not any(b"\n" in entry for entry in entries)


def test_collect_fails_without_writing_a_list_when_the_inventory_is_wrong(tree: Path, tmp_path: Path) -> None:
    """A partial upload list is worse than none: the step must stop first."""

    (tree / WHEELS[0]).unlink()
    output = tmp_path / "upload-list"

    assert check_release_assets.main(["collect", str(tree), "--output", str(output)]) == 1
    assert not output.exists()


def test_verify_exits_non_zero_on_a_mismatch(tree: Path, tmp_path: Path) -> None:
    manifest = check_release_assets.local_manifest(check_release_assets.collect_files(tree))
    remote = check_release_assets.remote_manifest(_remote_json(manifest))
    del remote[SDIST]
    payload = tmp_path / "remote.json"
    payload.write_text(_remote_json(remote), encoding="utf-8")

    assert check_release_assets.main(["verify", str(tree), "--remote", str(payload)]) == 1


def test_verify_exits_zero_on_a_complete_release(tree: Path, tmp_path: Path) -> None:
    manifest = check_release_assets.local_manifest(check_release_assets.collect_files(tree))
    payload = tmp_path / "remote.json"
    payload.write_text(_remote_json(manifest), encoding="utf-8")

    assert check_release_assets.main(["verify", str(tree), "--remote", str(payload)]) == 0

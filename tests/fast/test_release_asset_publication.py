# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts import publish_release_assets as publishing

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "AstroVela/vane"
TAG = "v0.2.0"
SDIST = "vane_ai-0.2.0.tar.gz"
WHEELS = {f"vane_ai-0.2.0-cp3{minor}-cp3{minor}-manylinux_2_28_x86_64.whl" for minor in range(10, 15)}


@pytest.fixture
def asset_tree(tmp_path):
    root = tmp_path / "release files"
    (root / "packages").mkdir(parents=True)
    (root / "release-assets").mkdir()
    checksums = []
    for name in sorted(WHEELS | {SDIST}):
        content = f"distribution: {name}".encode()
        (root / name).write_bytes(content)
        checksums.append(f"{hashlib.sha256(content).hexdigest()}  {name}\n")
        (root / "packages" / f"{name}.sigstore.json").write_text('{"bundle":"fixture"}')
    for name in publishing.SUPPLEMENTAL:
        (root / "release-assets" / name).write_text("".join(checksums) if name == "SHA256SUMS" else "{}")
    return root


def remote_asset(path):
    content = path.read_bytes()
    return {
        "name": path.name,
        "size": len(content),
        "digest": f"sha256:{hashlib.sha256(content).hexdigest()}",
        "state": "uploaded",
    }


def starter_asset(name, asset_id=101):
    return {"id": asset_id, "name": name, "size": 0, "digest": None, "state": "starter"}


@pytest.fixture
def github(monkeypatch):
    state = {"id": 17, "draft": True, "tag_name": TAG, "assets": []}
    calls = []
    uploads = []
    after_upload = []

    def read(*arguments):
        calls.append(arguments)
        if arguments[0] == "release":
            assert arguments == ("release", "view", TAG, "--repo", REPOSITORY, "--json", "apiUrl,isDraft")
            return {
                "apiUrl": f"https://api.github.com/repos/{REPOSITORY}/releases/{state['id']}",
                "isDraft": state["draft"],
            }
        if arguments == ("api", f"repos/{REPOSITORY}/releases/{state['id']}"):
            return copy.deepcopy(state)
        assert arguments[0] == "api"
        asset = next(
            asset
            for asset in state["assets"]
            if arguments[1] == f"repos/{REPOSITORY}/releases/assets/{asset.get('id')}"
        )
        return copy.deepcopy(asset)

    def upload(command, *, check):
        assert check
        if command[:4] == ["gh", "api", "--method", "DELETE"]:
            calls.append(tuple(command[1:]))
            assert len(command) == 5
            asset = next(
                asset
                for asset in state["assets"]
                if command[4] == f"repos/{REPOSITORY}/releases/assets/{asset.get('id')}"
            )
            assert asset["state"] == "starter" and asset["size"] == 0
            state["assets"].remove(asset)
            return subprocess.CompletedProcess(command, 0)
        assert command[:4] == ["gh", "release", "upload", TAG]
        assert command[5:] == ["--repo", REPOSITORY]
        path = Path(command[4])
        assert path.is_file() and not path.is_symlink()
        assert path.name not in {asset["name"] for asset in state["assets"]}
        uploads.append(path)
        state["assets"].append({"id": 1000 + len(uploads), **remote_asset(path)})
        for callback in after_upload:
            callback(path, state)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(publishing, "_gh_json", read)
    monkeypatch.setattr(publishing.subprocess, "run", upload)
    return state, calls, uploads, after_upload


def test_nested_artifacts_upload_all_fifteen_files(asset_tree, github):
    state, _, uploads, _ = github
    assert any(path.is_dir() for path in asset_tree.glob("*"))  # The old upload glob includes these directories.
    assert publishing.publish_assets(asset_tree, REPOSITORY, TAG) == 15
    assert len(uploads) == len(state["assets"]) == 15
    assert {path.name for path in uploads} == (
        WHEELS | {SDIST} | {f"{name}.sigstore.json" for name in WHEELS | {SDIST}} | publishing.SUPPLEMENTAL
    )
    assert state["draft"]  # The workflow publishes only after the verified upload returns.


@pytest.mark.parametrize(
    "missing", [SDIST, sorted(WHEELS)[0], f"{SDIST}.sigstore.json", *sorted(publishing.SUPPLEMENTAL)]
)
def test_missing_asset_stops_before_github_access(asset_tree, github, missing):
    _, calls, uploads, _ = github
    next(path for path in asset_tree.rglob("*") if path.name == missing).unlink()
    with pytest.raises(ValueError):
        publishing.publish_assets(asset_tree, REPOSITORY, TAG)
    assert not calls and not uploads


@pytest.mark.parametrize("damage", ["empty", "duplicate", "extra", "wrong_hash", "missing_hash", "duplicate_hash"])
def test_bad_local_inventory_stops_before_upload(asset_tree, github, damage):
    _, calls, uploads, _ = github
    checksums = asset_tree / "release-assets" / "SHA256SUMS"
    if damage == "empty":
        (asset_tree / SDIST).write_bytes(b"")
    elif damage == "duplicate":
        (asset_tree / "packages" / SDIST).write_bytes((asset_tree / SDIST).read_bytes())
    elif damage == "extra":
        (asset_tree / "release-assets" / "private.log").write_text("not a release asset")
    elif damage == "wrong_hash":
        (asset_tree / SDIST).write_text("changed bytes")
    elif damage == "missing_hash":
        checksums.write_text("".join(checksums.read_text().splitlines(keepends=True)[1:]))
    else:
        checksums.write_text(checksums.read_text() + checksums.read_text().splitlines(keepends=True)[0])
    with pytest.raises(ValueError):
        publishing.publish_assets(asset_tree, REPOSITORY, TAG)
    assert not calls and not uploads


@pytest.mark.parametrize("kind", ["file_link", "directory_link", "root_link", "fifo"])
def test_non_regular_assets_are_rejected(asset_tree, tmp_path, github, kind):
    _, calls, uploads, _ = github
    if kind == "file_link":
        (asset_tree / "extra").symlink_to(asset_tree / SDIST)
    elif kind == "directory_link":
        (asset_tree / "extra").symlink_to(asset_tree / "packages", target_is_directory=True)
    elif kind == "root_link":
        link = tmp_path / "linked-root"
        link.symlink_to(asset_tree, target_is_directory=True)
        asset_tree = link
    else:
        os.mkfifo(asset_tree / "extra")
    with pytest.raises(ValueError):
        publishing.publish_assets(asset_tree, REPOSITORY, TAG)
    assert not calls and not uploads


@pytest.mark.parametrize("damage", ["digest", "size", "pending", "missing_digest", "duplicate", "extra"])
def test_existing_conflicts_stop_without_overwriting(asset_tree, github, damage):
    state, _, uploads, _ = github
    existing = remote_asset(asset_tree / SDIST)
    state["assets"].append(existing)
    if damage == "digest":
        existing["digest"] = "sha256:" + "0" * 64
    elif damage == "size":
        existing["size"] -= 1
    elif damage == "pending":
        existing["state"] = "starter"
    elif damage == "missing_digest":
        existing.pop("digest")
    elif damage == "duplicate":
        state["assets"].append(existing.copy())
    else:
        existing["name"] = "old-version.tar.gz"
    with pytest.raises(ValueError):
        publishing.publish_assets(asset_tree, REPOSITORY, TAG)
    assert not uploads


@pytest.mark.parametrize("present", [3, 15])
def test_retry_uploads_only_missing_matching_assets(asset_tree, github, present):
    state, _, uploads, _ = github
    paths = [path for path in asset_tree.rglob("*") if path.is_file()]
    state["assets"] = [remote_asset(path) for path in paths[:present]]
    assert publishing.publish_assets(asset_tree, REPOSITORY, TAG) == 15 - present
    assert len(uploads) == 15 - present
    assert len(state["assets"]) == 15


def test_interrupted_upload_resumes_the_original_bytes(asset_tree, github):
    state, _, uploads, callbacks = github

    def fail_after_first_upload(path, release):
        raise subprocess.CalledProcessError(1, ["gh", "release", "upload"])

    callbacks.append(fail_after_first_upload)
    with pytest.raises(subprocess.CalledProcessError):
        publishing.publish_assets(asset_tree, REPOSITORY, TAG)
    assert len(state["assets"]) == 1
    callbacks.clear()
    assert publishing.publish_assets(asset_tree, REPOSITORY, TAG) == 14
    assert len(uploads) == 15


def test_502_starter_is_deleted_on_retry_without_replacing_completed_assets(asset_tree, github):
    state, calls, uploads, callbacks = github

    def fail_with_starter(path, release):
        if len(uploads) == 2:
            release["assets"][-1] = starter_asset(path.name, release["assets"][-1]["id"])
            raise subprocess.CalledProcessError(1, ["gh", "release", "upload"], stderr="HTTP 502: Bad Gateway")

    callbacks.append(fail_with_starter)
    with pytest.raises(subprocess.CalledProcessError):
        publishing.publish_assets(asset_tree, REPOSITORY, TAG)
    completed, failed = copy.deepcopy(state["assets"])
    callbacks.clear()
    assert publishing.publish_assets(asset_tree, REPOSITORY, TAG) == 14
    assert len(uploads) == 16 and len(state["assets"]) == 15
    assert completed in state["assets"] and failed not in state["assets"]
    assert [call for call in calls if "DELETE" in call] == [
        ("api", "--method", "DELETE", f"repos/{REPOSITORY}/releases/assets/{failed['id']}")
    ]
    assert state["draft"]


def test_multiple_empty_starters_are_recovered(asset_tree, github):
    state, calls, uploads, _ = github
    state["assets"] = [starter_asset(SDIST), starter_asset("SHA256SUMS", 102)]
    assert publishing.publish_assets(asset_tree, REPOSITORY, TAG) == 15
    assert len(uploads) == len(state["assets"]) == 15
    assert len([call for call in calls if "DELETE" in call]) == 2


@pytest.mark.parametrize("damage", ["missing_id", "invalid_id", "nonempty", "duplicate", "extra", "uploaded"])
def test_invalid_starters_are_never_deleted(asset_tree, github, damage):
    state, calls, uploads, _ = github
    starter = starter_asset(SDIST)
    state["assets"] = [starter]
    if damage == "missing_id":
        starter.pop("id")
    elif damage == "invalid_id":
        starter["id"] = "../17"
    elif damage == "nonempty":
        starter["size"] = 1
    elif damage == "duplicate":
        state["assets"].append(starter.copy())
    elif damage == "extra":
        starter["name"] = "unexpected.log"
    else:
        starter["state"] = "uploaded"
    before = copy.deepcopy(state)
    with pytest.raises(ValueError):
        publishing.publish_assets(asset_tree, REPOSITORY, TAG)
    assert not uploads and state == before
    assert not any("DELETE" in call for call in calls)


def test_complete_inventory_is_checked_before_starter_cleanup(asset_tree, github):
    state, calls, uploads, _ = github
    state["assets"] = [starter_asset("SHA256SUMS"), remote_asset(asset_tree / SDIST)]
    state["assets"][-1]["digest"] = "sha256:" + "0" * 64
    before = copy.deepcopy(state)
    with pytest.raises(ValueError):
        publishing.publish_assets(asset_tree, REPOSITORY, TAG)
    assert not uploads and state == before
    assert not any("DELETE" in call for call in calls)


@pytest.mark.parametrize("change", ["completed", "renamed", "replaced", "nonempty"])
def test_starter_is_rechecked_by_id_before_deletion(asset_tree, github, monkeypatch, change):
    state, calls, uploads, _ = github
    starter = starter_asset(SDIST)
    state["assets"] = [starter]
    read = publishing._gh_json

    def changed_asset(*arguments):
        if arguments == ("api", f"repos/{REPOSITORY}/releases/assets/{starter['id']}"):
            if change == "completed":
                starter.update(remote_asset(asset_tree / SDIST))
            elif change == "renamed":
                starter["name"] = "renamed.tar.gz"
            elif change == "replaced":
                response = read(*arguments)
                response["id"] += 1
                return response
            else:
                starter["size"] = 1
        return read(*arguments)

    monkeypatch.setattr(publishing, "_gh_json", changed_asset)
    with pytest.raises(ValueError):
        publishing.publish_assets(asset_tree, REPOSITORY, TAG)
    assert not uploads and len(state["assets"]) == 1
    assert not any("DELETE" in call for call in calls)


@pytest.mark.parametrize("deleted", [False, True])
def test_failed_starter_cleanup_can_be_retried(asset_tree, github, monkeypatch, deleted):
    state, _, uploads, _ = github
    state["assets"] = [starter_asset(SDIST)]
    run = publishing.subprocess.run

    def fail_delete(command, *, check):
        assert "DELETE" in command
        if deleted:
            run(command, check=check)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(publishing.subprocess, "run", fail_delete)
    with pytest.raises(subprocess.CalledProcessError):
        publishing.publish_assets(asset_tree, REPOSITORY, TAG)
    assert not uploads and state["draft"]
    monkeypatch.setattr(publishing.subprocess, "run", run)
    assert publishing.publish_assets(asset_tree, REPOSITORY, TAG) == 15
    assert len(uploads) == len(state["assets"]) == 15


@pytest.mark.parametrize(
    "damage", ["truncated", "substituted", "missing", "extra", "release_changed", "published", "starter"]
)
def test_post_upload_verification_prevents_incomplete_publication(asset_tree, github, damage):
    state, calls, uploads, callbacks = github

    def damage_last_upload(path, release):
        if len(uploads) != 15:
            return
        if damage == "truncated":
            release["assets"][-1]["size"] -= 1
        elif damage == "substituted":
            release["assets"][-1]["digest"] = "sha256:" + "0" * 64
        elif damage == "missing":
            release["assets"].pop()
        elif damage == "extra":
            release["assets"].append({"name": "unexpected.log"})
        elif damage == "release_changed":
            release["id"] += 1
        elif damage == "starter":
            release["assets"][-1] = starter_asset(path.name, release["assets"][-1]["id"])
        else:
            release["draft"] = False

    callbacks.append(damage_last_upload)
    with pytest.raises(ValueError):
        publishing.publish_assets(asset_tree, REPOSITORY, TAG)
    assert len(uploads) == 15
    assert not any("DELETE" in call for call in calls)


def test_tag_mismatch_does_not_access_github(asset_tree, github):
    _, calls, uploads, _ = github
    with pytest.raises(ValueError, match="tag differs"):
        publishing.publish_assets(asset_tree, REPOSITORY, "v0.2.1")
    assert not calls and not uploads


def test_published_release_is_not_modified(asset_tree, github):
    state, _, uploads, _ = github
    state["draft"] = False
    with pytest.raises(ValueError, match="must still be a draft"):
        publishing.publish_assets(asset_tree, REPOSITORY, TAG)
    assert not uploads


def test_build_only_checks_the_real_download_layout_before_index_upload(asset_tree):
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    jobs = workflow["jobs"]
    validation = jobs["verify-github-assets"]
    assert validation["needs"] == "assemble" and "if" not in validation
    assert "verify-github-assets" in jobs["publish-testpypi"]["needs"]
    assert [
        step["with"]["path"]
        for step in validation["steps"]
        if step.get("uses", "").startswith("actions/download-artifact@")
    ] == [
        "release-files",
        "release-files",
    ]
    for name in ("verify-github-assets", "publish-github-release"):
        assert any(step.get("uses", "").startswith("actions/checkout@") for step in jobs[name]["steps"])
        assert any(step.get("uses", "").startswith("actions/setup-python@") for step in jobs[name]["steps"])
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(ROOT / "scripts/publish_release_assets.py"),
            "check",
            "--directory",
            str(asset_tree),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=asset_tree.parent,
        env={**os.environ, "PATH": ""},
    )
    assert '"assets": 15' in result.stdout
    assert '"verified": true' in result.stdout

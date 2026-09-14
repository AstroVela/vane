# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Publication gates for the first, single-platform native_media release profile."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import Request, build_opener
from xml.etree import ElementTree

from packaging.utils import parse_wheel_filename
from packaging.version import Version

from vane_packaging.media_bundle import read_native_media_wheel
from vane_packaging.media_release import (
    _LIMITS,
    MANIFEST,
    _copy_file,
    _download,
    _file_record,
    _HTTPSRedirects,
    _output_directory,
    read_manifest,
    verify_release,
)
from vane_packaging.media_version import runtime_format, source_version

REPOSITORY = "AstroVela/vane"
TRUST = "astrovela/vane"
PLATFORM = "manylinux_2_28_x86_64"
INDEXES = {"testpypi": "https://test.pypi.org", "pypi": "https://pypi.org"}


def _json(url: str, *, missing: bool = False):
    try:
        with build_opener(_HTTPSRedirects()).open(
            Request(url, headers={"Accept-Encoding": "identity"}), timeout=60
        ) as response:
            contents = response.read(4 * 1024 * 1024 + 1)
    except HTTPError as error:
        if missing and error.code == 404:
            return None
        raise
    if len(contents) > 4 * 1024 * 1024:
        raise ValueError("index response exceeds 4 MiB")
    return json.loads(contents)


def index_files(channel: str, distribution: str, version: str) -> dict[str, dict] | None:
    document = _json(
        f"{INDEXES[channel]}/pypi/{quote(distribution, safe='')}/{quote(version, safe='')}/json", missing=True
    )
    if document is None:
        return None
    if document["info"]["version"] != version:
        raise ValueError("index returned a different release version")
    files = {}
    for record in document["urls"]:
        name = record["filename"]
        if name in files or record.get("yanked"):
            raise ValueError("index release contains duplicate or yanked files")
        url = urlsplit(record["url"])
        if (
            url.scheme != "https"
            or url.hostname not in {"files.pythonhosted.org", "test-files.pythonhosted.org"}
            or url.username
            or url.password
            or url.fragment
        ):
            raise ValueError("index artifact must come from the public Python package file host")
        files[name] = record
    return files


def _match_index(record: dict, expected: dict) -> None:
    if (
        record["filename"] != expected["filename"]
        or type(record["size"]) is not int
        or record["size"] != expected["size"]
        or record["digests"]["sha256"] != expected["sha256"]
    ):
        raise ValueError("indexed artifact differs from the accepted release bytes")


def preflight(output: Path, *, release: bool) -> dict:
    root = Path(__file__).resolve().parents[1]
    identity = source_version(root / "packages/vane-media-runtime", from_checkout=True)
    if identity["git_dirty"]:
        raise ValueError("media pipeline requires a clean source commit")
    commit = identity["git_commit"]
    version = identity["vane_version"]
    if release:
        from scripts.sign_media_release import require_context

        if require_context(dict(os.environ)) != (commit, version):
            raise ValueError("release tag differs from the actual source version")
        parsed = Version(version)
        branch = "main" if parsed.micro == 0 and parsed.post is None else f"release/{parsed.major}.{parsed.minor}"
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "refs/remotes/origin/" + branch], cwd=root, check=True
        )
        candidate_tag = "native-media-" + commit
        refs = _gh(f"repos/{REPOSITORY}/git/matching-refs/tags/{candidate_tag}")
        if any(record["ref"] == "refs/tags/" + candidate_tag for record in refs):
            raise ValueError(
                "media candidate already exists; resume failed jobs using original artifacts, never rebuild"
            )
    plan = {
        "git_commit": commit,
        "vane_version": version,
        "runtime_version": identity["version"],
        "tag": "native-media-" + commit,
        "release_url": f"https://github.com/{REPOSITORY}/releases/download/native-media-{commit}",
        "profile": "cp312-manylinux_2_28_x86_64",
    }
    output.mkdir(parents=True, exist_ok=False)
    if release:
        files = index_files("pypi", "vane-ai", version)
        if files is None:
            raise ValueError("publish the matching Vane base release to PyPI before releasing native_media")
        matches = []
        for name, record in files.items():
            if not name.endswith(".whl"):
                continue
            distribution, wheel_version, build, tags = parse_wheel_filename(name)
            if (
                distribution == "vane-ai"
                and str(wheel_version) == version
                and not build
                and tags
                and all(tag.interpreter == "cp312" and tag.abi == "cp312" and tag.platform == PLATFORM for tag in tags)
            ):
                matches.append(record)
        if len(matches) != 1:
            raise ValueError("published base release requires exactly one CPython 3.12 manylinux_2_28 x86-64 wheel")
        record = matches[0]
        expected = {"filename": record["filename"], "size": record["size"], "sha256": record["digests"]["sha256"]}
        if type(expected["size"]) is not int or not 0 < expected["size"] <= _LIMITS["base"]:
            raise ValueError("indexed base exceeds the publication budget")
        runtime_format().digest(expected["sha256"])
        _download(record["url"], output / expected["filename"], expected)
        from scripts.check_release_artifacts import check_artifact

        check_artifact(output / expected["filename"], expected_version=Version(version))
        plan["base"] = expected
    (output / "release-plan.json").write_bytes(runtime_format().canonical_json(plan))
    return plan


def checked_delivery(directory: Path, digest: str) -> dict:
    """Data-only validation for privileged jobs; clean/native validation ran earlier."""
    _, manifest = read_manifest(directory / MANIFEST, trust_identity=TRUST, sha256=digest)
    if {p.name for p in directory.iterdir()} != {MANIFEST, *(r["filename"] for r in manifest["artifacts"].values())}:
        raise ValueError("publication requires the exact five-file delivery")
    for role, record in manifest["artifacts"].items():
        if _file_record(directory / record["filename"], _LIMITS[role]) != record:
            raise ValueError("publication input differs from the accepted manifest")
    return manifest


def _provider_files(manifest: dict) -> tuple[str, str, dict]:
    record = manifest["artifacts"]["provider"]
    name = record["filename"]
    files = {name: record}
    distribution, version, build, tags = parse_wheel_filename(name)
    if (
        distribution != "vane-extension-native-media"
        or build
        or not tags
        or any(t.interpreter != "cp312" or t.abi != "none" or t.platform != PLATFORM for t in tags)
    ):
        raise ValueError("media publication only supports the reviewed CPython 3.12 Linux profile")
    return distribution, str(version), files


def stage_index(directory: Path, digest: str, *, channel: str, output: Path) -> bool:
    manifest = checked_delivery(directory, digest)
    distribution, version, expected = _provider_files(manifest)
    existing = index_files(channel, distribution, version) or {}
    if not existing.keys() <= expected.keys():
        raise ValueError("index contains unexpected release artifacts")
    with _output_directory(output) as stage:
        with tempfile.TemporaryDirectory(prefix="vane-index-resume-") as value:
            for name, record in existing.items():
                _match_index(record, expected[name])
                # A failed upload may be resumed only after actually retrieving
                # and matching existing bytes, not by blindly skipping filenames.
                _download(record["url"], Path(value) / name, expected[name])
            for record in expected.values():
                if record["filename"] not in existing:
                    copied = _copy_file(directory / record["filename"], stage / record["filename"], _LIMITS["provider"])
                    if copied != record:
                        raise ValueError("publication input changed while staging the accepted bytes")
    return bool(expected.keys() - existing.keys())


def verify_index(directory: Path, digest: str, *, channel: str, output: Path, attempts: int = 8) -> dict:
    manifest = checked_delivery(directory, digest)
    with _output_directory(output) as stage:
        _copy_file(directory / MANIFEST, stage / MANIFEST, 64 * 1024)
        for key in ("base", "instructions"):
            record = manifest["artifacts"][key]
            _copy_file(directory / record["filename"], stage / record["filename"], _LIMITS[key])
        distribution, version, expected = _provider_files(manifest)
        for attempt in range(attempts):
            files = index_files(channel, distribution, version)
            if files is not None and files.keys() == expected.keys():
                break
            if files and not files.keys() <= expected.keys():
                raise ValueError("index contains unexpected release artifacts")
            if attempt + 1 == attempts:
                raise ValueError("index has not exposed the complete media release")
            time.sleep(15)
        for name, record in files.items():
            _match_index(record, expected[name])
            _download(record["url"], stage / name, expected[name])
        # The single PyPI wheel names its immutable public source SDK. Retrieve
        # those exact source bytes again for each index's acceptance receipt.
        runtime = read_native_media_wheel(stage / manifest["artifacts"]["provider"]["filename"])
        source = manifest["artifacts"]["source"]
        _download(runtime[1]["source"]["url"], stage / source["filename"], source)
        verify_release(stage, trust_identity=TRUST, manifest_sha256=digest)
    return {"schema_version": 1, "index": INDEXES[channel], "manifest_sha256": digest, "verified": True}


def _gh(*arguments: str, payload: dict | None = None):
    command = ["gh", "api", *arguments]
    if payload is not None:
        command.extend(("--input", "-"))
    result = subprocess.run(
        command, input=json.dumps(payload) if payload is not None else None, text=True, capture_output=True, check=True
    )
    return json.loads(result.stdout) if result.stdout.strip() else None


def _check_github_tag(tag: str, commit: str, *, required: bool) -> None:
    references = _gh(f"repos/{REPOSITORY}/git/matching-refs/tags/{tag}")
    matches = [record["object"] for record in references if record["ref"] == "refs/tags/" + tag]
    if not matches and not required:
        return
    if len(matches) != 1:
        raise ValueError("media GitHub tag is missing or ambiguous")
    target = matches[0]
    for _ in range(5):
        if target["type"] == "commit":
            if target["sha"] != commit:
                raise ValueError("media GitHub tag points to a different source commit")
            return
        if target["type"] != "tag":
            break
        target = _gh(f"repos/{REPOSITORY}/git/tags/{target['sha']}")["object"]
    raise ValueError("media GitHub tag must resolve to the reviewed commit")


def publish_github(directory: Path, digest: str, *, evidence: bool = False) -> str:
    from scripts.sign_media_release import require_context

    commit, version = require_context(dict(os.environ))
    if not evidence:
        checked_delivery(directory, digest)
    tag = ("native-media-evidence-" if evidence else "native-media-") + commit
    _check_github_tag(tag, commit, required=False)
    # List by tag through the API (including drafts with this job's token). A
    # network/authentication failure must not masquerade as an unused tag.
    matching = []
    page = 1
    while True:
        releases = _gh(f"repos/{REPOSITORY}/releases?per_page=100&page={page}")
        matching.extend(r for r in releases if r["tag_name"] == tag)
        if len(releases) < 100:
            break
        page += 1
    if len(matching) > 1:
        raise ValueError("ambiguous media GitHub release")
    expected = {path.name: _file_record(path, 512 * 1024 * 1024) for path in directory.iterdir()}
    if not expected:
        raise ValueError("refusing to publish an empty release")
    title = f"native_media {'acceptance evidence' if evidence else 'candidate'} for Vane {version}"
    body = (
        f"Vane commit: `{commit}`\n\nDelivery manifest SHA-256: `{digest}`\n\n"
        + (
            "Source rebuild, modified-SoXR execution, and index verification evidence.\n"
            if evidence
            else "Candidate awaiting source rebuild and index acceptance. The complete corresponding source SDK and replacement instructions accompany the binaries.\n"
        )
        + f"\nWorkflow: https://github.com/{REPOSITORY}/actions/runs/{os.environ['GITHUB_RUN_ID']}\n"
    )
    release = (
        matching[0]
        if matching
        else _gh(
            f"repos/{REPOSITORY}/releases",
            "--method",
            "POST",
            payload={
                "tag_name": tag,
                "target_commitish": commit,
                "name": title,
                "body": body,
                "draft": True,
                "prerelease": True,
                "make_latest": "false",
            },
        )
    )
    if release["target_commitish"] != commit:
        raise ValueError("GitHub release targets another source commit")
    existing = {asset["name"]: asset for asset in release["assets"]}
    if len(existing) != len(release["assets"]) or not existing.keys() <= expected.keys():
        raise ValueError("GitHub release contains unexpected assets")
    for name, asset in existing.items():
        if asset["size"] != expected[name]["size"] or asset.get("digest") != "sha256:" + expected[name]["sha256"]:
            raise ValueError("refusing to replace an existing GitHub release asset")
    if not release["draft"] and existing.keys() != expected.keys():
        raise ValueError("published GitHub release is incomplete")
    for name in sorted(expected.keys() - existing.keys()):
        subprocess.run(["gh", "release", "upload", tag, str(directory / name), "--repo", REPOSITORY], check=True)
    if release["draft"]:
        uploaded = _gh(f"repos/{REPOSITORY}/releases/{release['id']}")["assets"]
        if (
            len(uploaded) != len(expected)
            or {asset["name"] for asset in uploaded} != expected.keys()
            or any(
                asset["size"] != expected[asset["name"]]["size"]
                or asset.get("digest") != "sha256:" + expected[asset["name"]]["sha256"]
                for asset in uploaded
            )
        ):
            raise ValueError("uploaded GitHub bytes differ from the accepted files; keep the release draft")
        release = _gh(f"repos/{REPOSITORY}/releases/{release['id']}", "--method", "PATCH", payload={"draft": False})
    if release.get("immutable") is not True:
        raise ValueError("enable GitHub immutable releases; a mutable candidate cannot be promoted")
    _check_github_tag(tag, commit, required=True)
    return f"https://github.com/{REPOSITORY}/releases/download/{tag}"


def promote_github(directory: Path, digest: str) -> None:
    from scripts.sign_media_release import require_context

    commit, version = require_context(dict(os.environ))
    validate_evidence(directory, digest)
    release = _gh(f"repos/{REPOSITORY}/releases/tags/native-media-{commit}")
    if release.get("immutable") is not True or release["draft"] or release["target_commitish"] != commit:
        raise ValueError("only the immutable candidate can be promoted")
    manifests = [asset for asset in release["assets"] if asset["name"] == MANIFEST]
    if len(manifests) != 1 or manifests[0].get("digest") != "sha256:" + digest:
        raise ValueError("candidate manifest differs from the qualified evidence")
    _check_github_tag("native-media-" + commit, commit, required=True)
    # Evidence is published as a second immutable release because candidate assets
    # are already frozen before acceptance starts.
    evidence_url = publish_github(directory, digest, evidence=True)
    _gh(
        f"repos/{REPOSITORY}/releases/{release['id']}",
        "--method",
        "PATCH",
        payload={
            "prerelease": False,
            "make_latest": "false",
            "name": f"native_media for Vane {version}",
            "body": f"Accepted delivery for Vane `{version}` at `{commit}`.\n\nManifest SHA-256: `{digest}`\n\n"
            f"[Source rebuild, replacement and index evidence]({evidence_url.replace('/download/', '/tag/')}).\n\n"
            "The native_media wheel, including its dynamic libraries, was promoted from TestPyPI to PyPI without rebuilding or re-signing.\n",
        },
    )


def validate_evidence(directory: Path, digest: str) -> None:
    """Do not mark a release qualified when a proof is missing, skipped or stale."""
    fmt = runtime_format()
    _, manifest = read_manifest(directory / MANIFEST, trust_identity=TRUST, sha256=digest)
    expected_names = {
        MANIFEST,
        "download.log",
        "rebuild.log",
        "ray.log",
        "ray.xml",
        "python-delivery.json",
        "rebuild-verification.json",
        "testpypi-verification.json",
        "testpypi-verification.log",
        "pypi-verification.json",
        "pypi-verification.log",
    }
    if {path.name for path in directory.iterdir()} != expected_names:
        raise ValueError("media qualification requires the complete, exact acceptance evidence set")
    for name in expected_names:
        _file_record(directory / name, 128 * 1024 * 1024)
    rebuild = json.loads(fmt.read_file(directory, "rebuild-verification.json", 64 * 1024))
    if (
        rebuild.get("release_manifest_sha256") != digest
        or rebuild.get("resampler_version_string") != "libsoxr-local-rebuild-proof"
    ):
        raise ValueError("source rebuild evidence does not prove this delivery's modified SoXR execution")
    fmt.digest(rebuild.get("extension_sha256"))
    fmt.digest(rebuild.get("effective_runtime_sha256"))
    for channel in INDEXES:
        receipt = json.loads(fmt.read_file(directory, f"{channel}-verification.json", 64 * 1024))
        if receipt != {"schema_version": 1, "index": INDEXES[channel], "manifest_sha256": digest, "verified": True}:
            raise ValueError("index acceptance evidence differs from this delivery")
    cases = ElementTree.fromstring(fmt.read_file(directory, "ray.xml", 1024 * 1024)).findall(".//testcase")
    names = {
        "test_ray_nodes_admit_only_the_exact_authorized_replacement[False]",
        "test_ray_nodes_admit_only_the_exact_authorized_replacement[True]",
    }
    if (
        len(cases) != 2
        or {case.get("name") for case in cases} != names
        or any(case.find(name) is not None for case in cases for name in ("skipped", "failure", "error"))
    ):
        raise ValueError("both real-Ray replacement and mismatch cases must pass")
    inventory = json.loads(fmt.read_file(directory, "python-delivery.json", 4 * 1024 * 1024))
    records = inventory["wheels"]
    expected_wheels = [manifest["artifacts"][role] for role in ("base", "provider")]
    if len(records) != 2 or {record["filename"] for record in records} != {r["filename"] for r in expected_wheels}:
        raise ValueError("Python inventory does not describe the two redistributed wheels")
    by_name = {record["filename"]: record for record in records}
    if any(any(by_name[r["filename"]][key] != r[key] for key in ("size", "sha256")) for r in expected_wheels):
        raise ValueError("Python inventory hashes differ from the released wheels")

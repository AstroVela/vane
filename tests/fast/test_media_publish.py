# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts import sign_media_release as signing
from tests.fast.test_media_release import release_inputs as release_inputs
from tests.fast.test_native_runtime_extension_wheels import TRUST_IDENTITY
from tests.fast.test_native_runtime_extension_wheels import media_dependency as media_dependency
from tests.fast.test_native_runtime_extension_wheels import release_runtime as release_runtime
from tests.fast.test_native_runtime_extension_wheels import runtime_wheel as runtime_wheel
from tests.fast.test_native_runtime_extension_wheels import source_sdk as source_sdk
from vane_packaging import media_publish as publishing
from vane_packaging.media_release import prepare_release, read_manifest
from vane_packaging.media_runtime import read_runtime_wheel
from vane_packaging.media_version import runtime_format

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def delivery(release_inputs, tmp_path, monkeypatch):
    inputs, _ = release_inputs
    path = tmp_path / "delivery"
    digest = prepare_release(**inputs, output=path)
    _, manifest = read_manifest(path / "media-release.json", trust_identity=TRUST_IDENTITY)
    monkeypatch.setattr(publishing, "TRUST", TRUST_IDENTITY)
    return path, digest, manifest


def indexed(record):
    return {
        "filename": record["filename"],
        "size": record["size"],
        "digests": {"sha256": record["sha256"]},
        "url": "https://files.pythonhosted.org/packages/" + record["filename"],
    }


def test_publication_sends_the_single_provider_wheel(delivery, tmp_path, monkeypatch):
    path, digest, manifest = delivery
    monkeypatch.setattr(publishing, "index_files", lambda *args: None)
    output = tmp_path / "dist"
    assert publishing.stage_index(path, digest, channel="pypi", output=output)
    assert {p.name for p in output.iterdir()} == {manifest["artifacts"]["provider"]["filename"]}
    assert all(p.read_bytes() == (path / p.name).read_bytes() for p in output.iterdir())


def test_resume_downloads_existing_wheel_and_avoids_duplicate_publication(delivery, tmp_path, monkeypatch):
    path, digest, manifest = delivery
    record = manifest["artifacts"]["provider"]
    monkeypatch.setattr(publishing, "index_files", lambda *args: {record["filename"]: indexed(record)})
    downloads = []

    def download(url, output, expected):
        downloads.append(expected)
        shutil.copyfile(path / expected["filename"], output)

    monkeypatch.setattr(publishing, "_download", download)
    output = tmp_path / "dist"
    assert not publishing.stage_index(path, digest, channel="testpypi", output=output)
    assert downloads == [record]
    assert not list(output.iterdir())


def test_staging_rejects_a_file_changed_after_initial_verification(delivery, tmp_path, monkeypatch):
    path, digest, manifest = delivery
    monkeypatch.setattr(publishing, "index_files", lambda *args: None)
    copy = publishing._copy_file

    def changed_copy(source, destination, limit):
        source.write_bytes(b"changed between verification and copy")
        return copy(source, destination, limit)

    monkeypatch.setattr(publishing, "_copy_file", changed_copy)
    with pytest.raises(ValueError, match="changed while staging"):
        publishing.stage_index(path, digest, channel="pypi", output=tmp_path / "dist")
    assert not (tmp_path / "dist").exists()


@pytest.mark.parametrize("damage", ["hash", "extra", "local", "manifest"])
def test_rejects_collisions_and_changed_delivery_before_upload(delivery, tmp_path, monkeypatch, damage):
    path, digest, manifest = delivery
    record = manifest["artifacts"]["provider"]
    files = {record["filename"]: indexed(record)}
    if damage == "hash":
        files[record["filename"]]["digests"]["sha256"] = "a" * 64
    elif damage == "extra":
        files["unexpected.whl"] = indexed(record)
    elif damage == "local":
        (path / record["filename"]).write_bytes(b"modified after acceptance")
    else:
        digest = "b" * 64
    monkeypatch.setattr(publishing, "index_files", lambda *args: files)
    monkeypatch.setattr(publishing, "_download", lambda *args: pytest.fail("invalid candidate reached download"))
    with pytest.raises(ValueError):
        publishing.stage_index(path, digest, channel="pypi", output=tmp_path / "dist")


@pytest.mark.parametrize("damage", [None, "source-corrupt", "source-missing"])
def test_index_acceptance_downloads_wheel_and_its_public_sources(delivery, tmp_path, monkeypatch, damage):
    path, digest, manifest = delivery
    downloads = []

    def index(channel, distribution, version):
        assert distribution == "vane-extension-native-media"
        record = manifest["artifacts"]["provider"]
        return {record["filename"]: indexed(record)}

    def download(url, output, expected):
        downloads.append((url, expected["filename"]))
        if expected == manifest["artifacts"]["source"] and damage == "source-missing":
            raise OSError("public corresponding source is unavailable")
        shutil.copyfile(path / expected["filename"], output)
        if expected == manifest["artifacts"]["source"] and damage == "source-corrupt":
            output.write_bytes(b"public source returned different bytes")

    monkeypatch.setattr(publishing, "index_files", index)
    monkeypatch.setattr(publishing, "_download", download)
    output = tmp_path / "indexed"
    if damage:
        with pytest.raises((ValueError, OSError)):
            publishing.verify_index(path, digest, channel="testpypi", output=output, attempts=1)
        assert not output.exists()
    else:
        receipt = publishing.verify_index(path, digest, channel="testpypi", output=output, attempts=1)
        assert receipt["verified"] is True
        assert receipt["manifest_sha256"] == digest
        assert {name for _, name in downloads} == {
            manifest["artifacts"][key]["filename"] for key in ("source", "provider")
        }
        runtime = publishing.read_native_media_wheel(path / manifest["artifacts"]["provider"]["filename"])
        assert downloads[-1][0] == runtime[1]["source"]["url"]


@pytest.mark.parametrize(
    "url",
    [
        "http://files.pythonhosted.org/f.whl",
        "https://evil.example/f.whl",
        "https://user:secret@files.pythonhosted.org/f.whl",
    ],
)
def test_index_cannot_redirect_delivery_to_an_unreviewed_host(monkeypatch, url):
    record = {"filename": "f.whl", "url": url, "yanked": False}
    monkeypatch.setattr(publishing, "_json", lambda *args, **kwargs: {"info": {"version": "1.0"}, "urls": [record]})
    with pytest.raises(ValueError, match="public Python package file host"):
        publishing.index_files("pypi", "vane-extension-native-media", "1.0")


def release_environment():
    return {
        "GITHUB_REPOSITORY": "AstroVela/vane",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF_PROTECTED": "true",
        "GITHUB_REF_NAME": "v0.2.0",
        "GITHUB_REF": "refs/tags/v0.2.0",
        "GITHUB_SHA": "a" * 40,
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("GITHUB_REPOSITORY", "fork/vane"),
        ("GITHUB_EVENT_NAME", "pull_request"),
        ("GITHUB_REF_PROTECTED", "false"),
        ("GITHUB_REF", "refs/heads/main"),
        ("GITHUB_REF_NAME", "v0.2.0.dev12"),
        ("GITHUB_REF_NAME", "v00.2.0"),
        ("GITHUB_SHA", "a" * 7),
    ],
)
def test_signing_and_publication_reject_unreviewed_context(key, value):
    environment = release_environment()
    assert signing.require_context(environment) == ("a" * 40, "0.2.0")
    environment[key] = value
    with pytest.raises(ValueError):
        signing.require_context(environment)


def test_integration_key_cannot_sign_a_production_release():
    key = bytearray((ROOT / "external/duckdb/test/mbedtls/private.pem").read_bytes())
    with pytest.raises(ValueError, match="production public fingerprint"):
        signing.require_key(key)


@pytest.fixture
def signing_data(release_runtime, tmp_path):
    _, manifest, _, _, _ = read_runtime_wheel(release_runtime[0])
    fmt = runtime_format()
    manifest["source"]["url"] = (
        f"https://github.com/AstroVela/vane/releases/download/native-media-{manifest['git_commit']}/{manifest['source']['filename']}"
    )
    document = fmt.canonical_json(manifest)
    artifact = fmt.attach_trailer(b"\x7fELF" + b"native code" * 60 + bytes(512), hashlib.sha256(document).hexdigest())
    directory = tmp_path / "signing"
    directory.mkdir()
    (directory / fmt.MANIFEST).write_bytes(document)
    (directory / "native_media.duckdb_extension").write_bytes(artifact)
    return directory, manifest, artifact


@pytest.mark.parametrize("damage", ["none", "commit", "version", "trailer", "signed", "symlink", "extra"])
def test_signer_binds_the_native_payload_to_the_exact_source_and_manifest(signing_data, damage):
    directory, manifest, artifact = signing_data
    commit, version = manifest["git_commit"], manifest["vane_version"]
    path = directory / "native_media.duckdb_extension"
    if damage == "commit":
        commit = "d" * 40
    elif damage == "version":
        version = "0.3.0"
    elif damage == "trailer":
        artifact = runtime_format().attach_trailer(artifact, "d" * 64)
        path.write_bytes(artifact)
    elif damage == "signed":
        path.write_bytes(artifact[:-256] + b"x" * 256)
    elif damage == "symlink":
        path.rename(directory.parent / "outside")
        path.symlink_to(directory.parent / "outside")
    elif damage == "extra":
        (directory / "backend.py").write_text("raise SystemExit('never import this')")
    if damage == "none":
        assert signing.signing_inputs(directory, commit=commit, version=version)[0] == artifact
    else:
        with pytest.raises((ValueError, OSError)):
            signing.signing_inputs(directory, commit=commit, version=version)


def test_unsigned_checker_runs_with_isolated_stdlib_python_and_no_key(signing_data):
    directory, manifest, _ = signing_data
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(ROOT / "scripts/sign_media_release.py"),
            "check",
            "--input-directory",
            str(directory),
            "--commit",
            manifest["git_commit"],
            "--version",
            manifest["vane_version"],
        ],
        check=True,
        env={},
    )


def test_signing_produces_verifiable_rsa_signatures_without_inheriting_the_secret(tmp_path, monkeypatch):
    key = ROOT / "external/duckdb/test/mbedtls/private.pem"
    digest = hashlib.sha256(b"runtime manifest signing fixture").digest()
    monkeypatch.setenv("VANE_SIGNING_PRIVATE_KEY", "must not reach openssl")
    run = subprocess.run

    def checked_run(*args, **kwargs):
        assert "VANE_SIGNING_PRIVATE_KEY" not in kwargs["env"]
        return run(*args, **kwargs)

    monkeypatch.setattr(signing.subprocess, "run", checked_run)
    signature = signing.sign_digest(key, digest, tmp_path)
    assert len(signature) == 256
    public = run(["openssl", "pkey", "-in", str(key), "-pubout"], capture_output=True, check=True).stdout
    (tmp_path / "public.pem").write_bytes(public)
    run(
        [
            "openssl",
            "pkeyutl",
            "-verify",
            "-pubin",
            "-inkey",
            str(tmp_path / "public.pem"),
            "-in",
            str(tmp_path / "digest"),
            "-sigfile",
            str(tmp_path / "signature"),
            "-pkeyopt",
            "digest:sha256",
        ],
        check=True,
    )


@pytest.mark.parametrize("size", [129 * 1024 * 1024, 384 * 1024 * 1024 + 1])
def test_signer_uses_the_native_member_budget_not_the_compressed_wheel_budget(signing_data, size):
    directory, manifest, artifact = signing_data
    tail = artifact[-(512 + runtime_format().TRAILER_SIZE) :]
    with (directory / "native_media.duckdb_extension").open("wb") as stream:
        stream.write(b"\x7fELF")
        stream.seek(size - len(tail))
        stream.write(tail)
    if size <= 384 * 1024 * 1024:
        contents, _ = signing.signing_inputs(directory, commit=manifest["git_commit"], version=manifest["vane_version"])
        assert len(contents) == size
    else:
        with pytest.raises(ValueError, match="bounded"):
            signing.signing_inputs(directory, commit=manifest["git_commit"], version=manifest["vane_version"])


@pytest.mark.parametrize("immutable,changed", [(True, False), (False, False), (True, True)])
def test_candidate_resume_never_overwrites_published_assets(delivery, monkeypatch, immutable, changed):
    path, digest, manifest = delivery
    environment = release_environment()
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    assets = []
    for item in path.iterdir():
        assets.append(
            {
                "name": item.name,
                "size": item.stat().st_size,
                "digest": "sha256:" + hashlib.sha256(item.read_bytes()).hexdigest(),
            }
        )
    if changed:
        assets[0]["digest"] = "sha256:" + "f" * 64
    release = {
        "tag_name": "native-media-" + "a" * 40,
        "id": 1,
        "target_commitish": "a" * 40,
        "draft": False,
        "immutable": immutable,
        "assets": assets,
    }

    def github(endpoint, *args, **kwargs):
        if "/git/matching-refs/" in endpoint:
            return [{"ref": "refs/tags/" + release["tag_name"], "object": {"type": "commit", "sha": "a" * 40}}]
        return [release]

    monkeypatch.setattr(publishing, "_gh", github)
    monkeypatch.setattr(
        publishing.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("must not replace or upload any published asset"),
    )
    if immutable and not changed:
        assert publishing.publish_github(path, digest).endswith("native-media-" + "a" * 40)
    else:
        with pytest.raises(ValueError):
            publishing.publish_github(path, digest)


@pytest.mark.parametrize("corrupt", [False, True])
def test_candidate_checks_server_asset_hashes_before_exposing_the_draft(delivery, monkeypatch, corrupt):
    path, digest, _ = delivery
    for key, value in release_environment().items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    release = {
        "tag_name": "native-media-" + "a" * 40,
        "id": 1,
        "target_commitish": "a" * 40,
        "draft": True,
        "immutable": False,
        "assets": [],
    }
    published = []

    def github(endpoint, *args, payload=None):
        if "/git/matching-refs/" in endpoint:
            return [{"ref": "refs/tags/" + release["tag_name"], "object": {"type": "commit", "sha": "a" * 40}}]
        if "?per_page=" in endpoint:
            return [release]
        if payload is not None:
            assert payload == {"draft": False}
            published.append(True)
            release.update(draft=False, immutable=True)
        return release

    def upload(command, **kwargs):
        asset = Path(command[4])
        assert command[:3] == ["gh", "release", "upload"] and "--clobber" not in command
        release["assets"].append(
            {
                "name": asset.name,
                "size": asset.stat().st_size,
                "digest": "sha256:" + ("f" * 64 if corrupt else hashlib.sha256(asset.read_bytes()).hexdigest()),
            }
        )

    monkeypatch.setattr(publishing, "_gh", github)
    monkeypatch.setattr(publishing.subprocess, "run", upload)
    if corrupt:
        with pytest.raises(ValueError, match="keep the release draft"):
            publishing.publish_github(path, digest)
        assert not published
    else:
        publishing.publish_github(path, digest)
        assert published == [True]


def test_workflow_cannot_publish_without_source_ray_and_index_acceptance():
    workflow = yaml.safe_load((ROOT / ".github/workflows/media-release.yml").read_text())
    jobs = workflow["jobs"]
    assert jobs["acceptance"]["uses"] == "./.github/workflows/media-release-verify.yml"
    assert "acceptance" in jobs["publish-testpypi"]["needs"]
    assert {"acceptance", "verify-testpypi"} <= set(jobs["publish-pypi"]["needs"])
    assert {"acceptance", "verify-pypi", "verify-testpypi"} <= set(jobs["complete"]["needs"])
    for channel in ("testpypi", "pypi"):
        job = jobs["publish-" + channel]
        assert job["environment"] == "native-media-" + channel
        assert "strategy" not in job
    signer = jobs["sign"]
    assert signer["environment"] == "media-production-signing"
    assert signer["if"] == "inputs.operation == 'release'"
    commands = "\n".join(step.get("run", "") for step in signer["steps"])
    assert "/usr/bin/python3 -I -S scripts/sign_media_release.py" in commands
    assert "pip" not in commands and "build_media" not in commands
    for name, job in jobs.items():
        for step in job.get("steps", []):
            if "download-artifact@" in step.get("uses", ""):
                assert step["with"]["artifact-ids"]
                assert step["with"]["digest-mismatch"] == "error"
        if name != "sign":
            assert "SIGNING_PRIVATE_KEY" not in json.dumps(job)
    acceptance = (ROOT / ".github/workflows/media-release-verify.yml").read_text()
    assert "test_ray_native_runtime_replacement.py" in acceptance
    assert "skips are not acceptance" in acceptance


@pytest.fixture
def evidence(delivery, tmp_path):
    from vane_packaging.python_delivery import inventory_delivery

    path, digest, manifest = delivery
    output = tmp_path / "evidence"
    output.mkdir()
    shutil.copyfile(path / "media-release.json", output / "media-release.json")
    for name in ("download", "rebuild", "ray", "testpypi-verification", "pypi-verification"):
        (output / (name + ".log")).write_text("passed\n")
    rebuild = {
        "release_manifest_sha256": digest,
        "resampler_version_string": "libsoxr-local-rebuild-proof",
        "extension_sha256": "1" * 64,
        "effective_runtime_sha256": "2" * 64,
        "schema_version": 1,
    }
    (output / "rebuild-verification.json").write_text(json.dumps(rebuild))
    for channel in publishing.INDEXES:
        (output / (channel + "-verification.json")).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "index": publishing.INDEXES[channel],
                    "manifest_sha256": digest,
                    "verified": True,
                }
            )
        )
    (output / "ray.xml").write_text(
        '<testsuites><testsuite><testcase name="test_ray_nodes_admit_only_the_exact_authorized_replacement[False]"/>'
        '<testcase name="test_ray_nodes_admit_only_the_exact_authorized_replacement[True]"/></testsuite></testsuites>'
    )
    inventory = inventory_delivery([path / manifest["artifacts"][role]["filename"] for role in ("base", "provider")])
    (output / "python-delivery.json").write_text(json.dumps(inventory))
    return output, digest


@pytest.mark.parametrize(
    "damage", ["none", "missing", "skipped", "wrong-tests", "old-rebuild", "failed-index", "different-wheels"]
)
def test_qualification_requires_all_proofs_for_the_same_delivery(evidence, damage):
    path, digest = evidence
    if damage == "missing":
        (path / "download.log").unlink()
    elif damage == "skipped":
        document = (path / "ray.xml").read_text().replace("/>", "><skipped/></testcase>", 1)
        (path / "ray.xml").write_text(document)
    elif damage == "wrong-tests":
        (path / "ray.xml").write_text('<testsuites><testcase name="unrelated"/><testcase name="another"/></testsuites>')
    elif damage == "old-rebuild":
        document = (path / "rebuild-verification.json").read_text().replace(digest, "a" * 64)
        (path / "rebuild-verification.json").write_text(document)
    elif damage == "failed-index":
        document = (path / "pypi-verification.json").read_text().replace("true", "false")
        (path / "pypi-verification.json").write_text(document)
    elif damage == "different-wheels":
        document = json.loads((path / "python-delivery.json").read_bytes())
        document["wheels"][0]["sha256"] = "a" * 64
        (path / "python-delivery.json").write_text(json.dumps(document))
    if damage == "none":
        publishing.validate_evidence(path, digest)
    else:
        with pytest.raises(ValueError):
            publishing.validate_evidence(path, digest)


@pytest.mark.parametrize("target", ["a" * 40, "b" * 40, None])
def test_candidate_tag_must_resolve_to_the_reviewed_commit(monkeypatch, target):
    references = [] if target is None else [{"ref": "refs/tags/candidate", "object": {"type": "tag", "sha": "c" * 40}}]

    def github(endpoint):
        if "matching-refs" in endpoint:
            return references
        return {"object": {"type": "commit", "sha": target}}

    monkeypatch.setattr(publishing, "_gh", github)
    if target == "a" * 40:
        publishing._check_github_tag("candidate", target, required=True)
    else:
        with pytest.raises(ValueError):
            publishing._check_github_tag("candidate", "a" * 40, required=True)


@pytest.mark.parametrize("state", ["dirty", "tag-mismatch", "candidate-exists", "missing-base"])
def test_release_preflight_fails_before_build_or_signing(tmp_path, monkeypatch, state):
    for key, value in release_environment().items():
        monkeypatch.setenv(key, value)
    identity = {"git_commit": "a" * 40, "git_dirty": state == "dirty", "vane_version": "0.2.0", "version": "0.2.0.1"}
    if state == "tag-mismatch":
        identity["vane_version"] = "0.2.0.dev1"
    monkeypatch.setattr(publishing, "source_version", lambda *args, **kwargs: identity)
    monkeypatch.setattr(publishing.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(publishing, "index_files", lambda *args: None)
    monkeypatch.setattr(
        publishing,
        "_gh",
        lambda *args: [{"ref": "refs/tags/native-media-" + "a" * 40}] if state == "candidate-exists" else [],
    )
    monkeypatch.setattr(publishing, "_download", lambda *args: pytest.fail("invalid release reached download"))
    with pytest.raises(ValueError):
        publishing.preflight(tmp_path / "preflight", release=True)

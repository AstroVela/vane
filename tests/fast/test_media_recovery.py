# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Recover an immutable delivery with newer workflow code and the original bytes."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import runpy
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from vane_packaging import media_publish as publishing
from vane_packaging.media_version import runtime_format

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def candidate(monkeypatch):
    commit = "a" * 40
    tag = "native-media-" + commit
    url = f"https://github.com/AstroVela/vane/releases/download/{tag}"
    names = {
        "base": "vane_ai-0.2.0-cp312-cp312-manylinux_2_28_x86_64.whl",
        "provider": "vane_extension_native_media-0.2.0.1-cp312-none-manylinux_2_28_x86_64.whl",
        "source": "vane_media_runtime-0.2.0.1.tar.gz",
        "instructions": "NATIVE_MEDIA_REPLACEMENT.md",
    }
    contents = {name: ("original " + role).encode() for role, name in names.items()}

    def record(name):
        return {"filename": name, "size": len(contents[name]), "sha256": hashlib.sha256(contents[name]).hexdigest()}

    manifest = {
        "schema_version": 1,
        "trust_identity": "astrovela/vane",
        "artifacts": {role: record(name) for role, name in names.items()},
    }
    contents[publishing.MANIFEST] = runtime_format().canonical_json(manifest)
    digest = record(publishing.MANIFEST)["sha256"]
    release = {
        "id": 17,
        "tag_name": tag,
        "target_commitish": commit,
        "draft": False,
        "immutable": True,
        "assets": [
            {"name": name, "size": len(data), "digest": "sha256:" + hashlib.sha256(data).hexdigest()}
            for name, data in contents.items()
        ],
    }
    targets = {"v0.2.0": commit, tag: commit}
    workflow = {"status": "ahead", "merge_base_commit": {"sha": publishing.RECOVERY_WORKFLOW_BASE}}
    github_calls = []

    def github(endpoint, *args, **kwargs):
        github_calls.append((endpoint, args, kwargs))
        assert not args and not kwargs, "recovery must not mutate the candidate"
        if endpoint == f"repos/AstroVela/vane/compare/{publishing.RECOVERY_WORKFLOW_BASE}...{os.environ['GITHUB_SHA']}":
            return workflow
        if "/git/matching-refs/tags/" in endpoint:
            name = endpoint.split("/git/matching-refs/tags/", 1)[1]
            target = targets.get(name)
            return [] if target is None else [{"ref": "refs/tags/" + name, "object": {"type": "commit", "sha": target}}]
        if endpoint in {
            f"repos/AstroVela/vane/releases/tags/{tag}",
            "repos/AstroVela/vane/releases?per_page=100&page=1",
        }:
            return [release] if "?" in endpoint else release
        raise AssertionError(endpoint)

    downloads = []

    def open_request(request, **kwargs):
        downloads.append(request.full_url)
        assert request.full_url.startswith(url + "/")
        response = io.BytesIO(contents[request.full_url.removeprefix(url + "/")])
        response.url = request.full_url
        return response

    base = manifest["artifacts"]["base"]
    indexes = {("pypi", "vane-ai", "0.2.0"): {base["filename"]: {**base, "digests": {"sha256": base["sha256"]}}}}
    runtime = {
        "git_commit": commit,
        "git_dirty": False,
        "vane_version": "0.2.0",
        "source": {key: value for key, value in manifest["artifacts"]["source"].items() if key != "size"},
    }
    runtime["source"]["url"] = url + "/" + names["source"]
    for key, value in {
        "GITHUB_REPOSITORY": "AstroVela/vane",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_REF_NAME": "main",
        "GITHUB_REF_PROTECTED": "true",
        "GITHUB_SHA": "b" * 40,
        "GITHUB_RUN_ID": "123",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(publishing, "_gh", github)
    # _download is imported from media_release; retain its real bounds and hash checks.
    monkeypatch.setattr("vane_packaging.media_release.build_opener", lambda *args: SimpleNamespace(open=open_request))
    monkeypatch.setattr(publishing, "index_files", lambda *args: indexes.get(args))
    # Native wheel parsing has its own signed-provider tests; this fixture tests
    # recovery's binding to that parser's result without requiring a test-key build.
    monkeypatch.setattr(publishing, "read_native_media_wheel", lambda path: (None, runtime))
    monkeypatch.setattr(
        publishing, "source_version", lambda *a, **k: pytest.fail("must not use the workflow's source identity")
    )
    monkeypatch.setattr(publishing.subprocess, "run", lambda *a, **k: pytest.fail("must not build, sign, or upload"))
    return SimpleNamespace(
        commit=commit,
        tag=tag,
        url=url,
        names=names,
        contents=contents,
        manifest=manifest,
        digest=digest,
        release=release,
        targets=targets,
        github_calls=github_calls,
        downloads=downloads,
        indexes=indexes,
        runtime=runtime,
        workflow=workflow,
    )


def recover(candidate, tmp_path, **kwargs):
    return publishing.resume(
        tmp_path / "recovered", release_tag="v0.2.0", digest=kwargs.get("digest", candidate.digest)
    )


def test_recovery_reuses_original_bytes_through_candidate_and_index_staging(candidate, tmp_path):
    plan = recover(candidate, tmp_path)
    assert plan["git_commit"] == candidate.commit != os.environ["GITHUB_SHA"]
    assert plan["workflow_commit"] == os.environ["GITHUB_SHA"]
    assert plan["release_tag"] == "v0.2.0"
    assert plan["manifest_sha256"] == candidate.digest
    assert plan["release_url"] == candidate.url
    assert json.loads((tmp_path / "recovered/release-plan.json").read_bytes()) == plan
    delivery = tmp_path / "recovered/delivery"
    assert {p.name: p.read_bytes() for p in delivery.iterdir()} == candidate.contents
    assert set(candidate.downloads) == {candidate.url + "/" + name for name in candidate.contents}
    assert publishing.publish_github(delivery, candidate.digest, release_tag="v0.2.0") == candidate.url
    assert publishing.stage_index(delivery, candidate.digest, channel="testpypi", output=tmp_path / "dist")
    assert {p.name: p.read_bytes() for p in (tmp_path / "dist").iterdir()} == {
        candidate.names["provider"]: candidate.contents[candidate.names["provider"]]
    }
    assert all(not args and not kwargs for _, args, kwargs in candidate.github_calls)


@pytest.mark.parametrize(
    "name,value",
    [
        ("GITHUB_REPOSITORY", "someone/vane"),
        ("GITHUB_EVENT_NAME", "pull_request"),
        ("GITHUB_REF", "refs/tags/v0.2.0"),
        ("GITHUB_REF_NAME", "feature"),
        ("GITHUB_REF_PROTECTED", "false"),
    ],
)
def test_recovery_requires_protected_main_dispatch(candidate, tmp_path, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="media recovery requires"):
        recover(candidate, tmp_path)
    assert not candidate.github_calls and not candidate.downloads


@pytest.mark.parametrize("tag", ["", "main", "v0.2.0rc1", "v0.2.0.dev1", "../v0.2.0"])
def test_recovery_requires_an_original_final_tag(candidate, tmp_path, tag):
    with pytest.raises(ValueError, match="original final Vane release tag"):
        publishing.resume(tmp_path / "recovered", release_tag=tag, digest=candidate.digest)
    assert not candidate.downloads


@pytest.mark.parametrize("commit", ["", "main", "b" * 39, "b" * 41, "../main"])
def test_recovery_requires_the_exact_workflow_commit(candidate, tmp_path, monkeypatch, commit):
    monkeypatch.setenv("GITHUB_SHA", commit)
    with pytest.raises(ValueError, match="exact workflow commit"):
        recover(candidate, tmp_path)
    assert not candidate.github_calls and not candidate.downloads


@pytest.mark.parametrize(
    "comparison",
    [
        {},
        {"status": "behind", "merge_base_commit": {"sha": publishing.RECOVERY_WORKFLOW_BASE}},
        {"status": "diverged", "merge_base_commit": {"sha": "c" * 40}},
        {"status": "ahead", "merge_base_commit": {"sha": "c" * 40}},
    ],
)
def test_recovery_rejects_a_workflow_without_the_acceptance_home_fix(candidate, tmp_path, comparison):
    candidate.workflow.clear()
    candidate.workflow.update(comparison)
    with pytest.raises(ValueError, match="reviewed acceptance HOME fix"):
        recover(candidate, tmp_path)
    assert len(candidate.github_calls) == 1
    assert not candidate.downloads
    assert not (tmp_path / "recovered").exists()


@pytest.mark.parametrize("status", ["ahead", "identical"])
def test_recovery_checks_the_dispatched_commit_instead_of_moving_main(candidate, tmp_path, monkeypatch, status):
    candidate.workflow["status"] = status
    if status == "identical":
        monkeypatch.setenv("GITHUB_SHA", publishing.RECOVERY_WORKFLOW_BASE)
    plan = recover(candidate, tmp_path)
    assert candidate.github_calls[0][0].endswith("..." + os.environ["GITHUB_SHA"])
    assert plan["workflow_commit"] == os.environ["GITHUB_SHA"]


@pytest.mark.parametrize(
    "damage",
    [
        "digest",
        "draft",
        "mutable",
        "commit",
        "tag",
        "missing-tag",
        "divergent-tag",
        "duplicate",
        "extra",
        "missing",
        "asset-hash",
        "asset-size",
    ],
)
def test_recovery_rejects_invalid_candidate_before_exposing_delivery(candidate, tmp_path, damage):
    if damage == "digest":
        candidate.digest = "f" * 64
    elif damage == "draft":
        candidate.release["draft"] = True
    elif damage == "mutable":
        candidate.release["immutable"] = False
    elif damage == "commit":
        candidate.release["target_commitish"] = "c" * 40
    elif damage == "tag":
        candidate.release["tag_name"] = "another-tag"
    elif damage == "missing-tag":
        candidate.targets.pop("v0.2.0")
    elif damage == "divergent-tag":
        candidate.targets[candidate.tag] = "c" * 40
    elif damage == "duplicate":
        candidate.release["assets"].append(candidate.release["assets"][0])
    elif damage == "extra":
        candidate.release["assets"].append({"name": "extra.txt"})
    elif damage == "missing":
        candidate.release["assets"].pop(0)
    elif damage == "asset-hash":
        candidate.release["assets"][0]["digest"] = "sha256:" + "f" * 64
    else:
        candidate.release["assets"][0]["size"] += 1
    with pytest.raises(ValueError):
        recover(candidate, tmp_path)
    assert not (tmp_path / "recovered").exists()


@pytest.mark.parametrize("role", ["manifest", "base", "provider", "source", "instructions"])
def test_recovery_rejects_changed_download_bytes(candidate, tmp_path, role):
    name = publishing.MANIFEST if role == "manifest" else candidate.names[role]
    candidate.contents[name] += b"changed"
    with pytest.raises(ValueError, match="published artifact"):
        recover(candidate, tmp_path)
    assert not (tmp_path / "recovered").exists()


@pytest.mark.parametrize("channel", ["testpypi", "pypi"])
@pytest.mark.parametrize("files", [{}, {"existing.whl": {}}])
def test_recovery_rejects_an_indexed_provider_version(candidate, tmp_path, channel, files):
    candidate.indexes[(channel, "vane-extension-native-media", "0.2.0.1")] = files
    with pytest.raises(ValueError, match="already published"):
        recover(candidate, tmp_path)
    assert candidate.downloads == [candidate.url + "/" + publishing.MANIFEST]


@pytest.mark.parametrize("damage", ["missing", "hash"])
def test_recovery_requires_the_original_pypi_base(candidate, tmp_path, damage):
    if damage == "missing":
        candidate.indexes.clear()
    else:
        candidate.indexes[("pypi", "vane-ai", "0.2.0")][candidate.names["base"]]["digests"]["sha256"] = "f" * 64
    with pytest.raises(ValueError):
        recover(candidate, tmp_path)
    assert not (tmp_path / "recovered").exists()


@pytest.mark.parametrize(
    "field,value", [("git_commit", "c" * 40), ("vane_version", "0.3.0"), ("git_dirty", True), ("source", {})]
)
def test_recovery_requires_the_original_runtime_identity(candidate, tmp_path, field, value):
    candidate.runtime[field] = value
    with pytest.raises(ValueError, match="original release source identity"):
        recover(candidate, tmp_path)
    assert not (tmp_path / "recovered").exists()


@pytest.mark.parametrize("channels", [(), ("testpypi",), ("pypi",), ("testpypi", "pypi")])
def test_recovery_promotion_retains_original_commit_and_digest(candidate, tmp_path, monkeypatch, channels):
    plan = recover(candidate, tmp_path)
    provider = candidate.manifest["artifacts"]["provider"]
    for channel in channels:
        candidate.indexes[(channel, "vane-extension-native-media", "0.2.0.1")] = {
            provider["filename"]: {**provider, "digests": {"sha256": provider["sha256"]}}
        }
    published = []
    updates = []
    github = publishing._gh

    def promote(endpoint, *args, **kwargs):
        if args == ("--method", "PATCH"):
            updates.append((endpoint, kwargs["payload"]))
            return None
        return github(endpoint, *args, **kwargs)

    def evidence(directory, digest, **kwargs):
        published.append((directory, digest, kwargs))
        assert publishing._publication_context(digest, kwargs["release_tag"]) == (candidate.commit, "0.2.0")
        return candidate.url.replace("native-media-", "native-media-evidence-")

    monkeypatch.setattr(publishing, "_gh", promote)
    monkeypatch.setattr(publishing, "validate_evidence", lambda path, digest: None)
    monkeypatch.setattr(publishing, "publish_github", evidence)
    publishing.promote_github(tmp_path, plan["manifest_sha256"], release_tag=plan["release_tag"])
    assert published == [(tmp_path, candidate.digest, {"evidence": True, "release_tag": "v0.2.0"})]
    assert updates[0][0] == "repos/AstroVela/vane/releases/17"
    assert candidate.commit in updates[0][1]["body"]
    assert os.environ["GITHUB_SHA"] not in updates[0][1]["body"]


def test_resume_cli_emits_the_original_identity(candidate, tmp_path, monkeypatch, capsys):
    output = tmp_path / "github-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "media_release_pipeline.py",
            "--github-output",
            str(output),
            "resume",
            "--release-tag",
            "v0.2.0",
            "--manifest-sha256",
            candidate.digest,
            "--output",
            str(tmp_path / "recovered"),
        ],
    )
    runpy.run_path(str(ROOT / "scripts/media_release_pipeline.py"), run_name="__main__")
    assert json.loads(capsys.readouterr().out)["git_commit"] == candidate.commit
    assert f"manifest_sha256={candidate.digest}\n" in output.read_text()
    assert "release_tag=v0.2.0\n" in output.read_text()


def _condition(expression, operation, results, *, cancelled=False):
    expression = expression.removeprefix("${{").removesuffix("}}").strip()
    expression = expression.replace("!cancelled()", str(not cancelled))
    expression = expression.replace("inputs.operation", repr(operation))
    expression = re.sub(r"needs\.([a-z-]+)\.result", lambda m: repr(results[m[1]]), expression)
    expression = expression.replace("&&", " and ").replace("||", " or ")
    return eval("(" + expression + ")", {"__builtins__": {}})


@pytest.mark.parametrize("operation", ["resume", "release", "build-only"])
@pytest.mark.parametrize("failure", [None, "preflight", "package", "acceptance", "verify-testpypi", "verify-pypi"])
def test_workflow_recovery_skips_build_but_keeps_all_acceptance_gates(operation, failure):
    jobs = yaml.safe_load((ROOT / ".github/workflows/media-release.yml").read_text())["jobs"]
    results = {}
    for name, job in jobs.items():
        dependencies = job.get("needs", [])
        if isinstance(dependencies, str):
            dependencies = [dependencies]
        expression = job.get("if", "")
        ready = all(results[dependency] == "success" for dependency in dependencies)
        if expression:
            selected = _condition(expression, operation, results)
            ready = selected and ("cancelled()" in expression or ready)
        results[name] = ("failure" if name == failure else "success") if ready else "skipped"
    if operation == "resume":
        assert results["build"] == results["sign"] == "skipped"
        if failure is None:
            assert results["package"] == results["complete"] == "success"
        package = jobs["package"]
        active = [step for step in package["steps"] if not step.get("if") or _condition(step["if"], operation, results)]
        assert not any("build_media_release.py" in step.get("run", "") for step in active)
        assert not any("needs.build." in str(step) or "needs.sign." in str(step) for step in active)
    if operation == "build-only" or failure:
        assert results["complete"] != "success"
    if failure in {"preflight", "package", "acceptance"}:
        assert results["publish-testpypi"] == results["publish-pypi"] == "skipped"
    if failure == "verify-testpypi":
        assert results["publish-pypi"] == "skipped"


def test_workflow_resume_dispatch_passes_pinned_inputs(tmp_path):
    jobs = yaml.safe_load((ROOT / ".github/workflows/media-release.yml").read_text())["jobs"]
    command = next(step["run"] for step in jobs["preflight"]["steps"] if step.get("id") == "plan")
    executable = tmp_path / "python"
    executable.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    executable.chmod(0o755)
    environment = dict(
        os.environ,
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        MEDIA_OPERATION="resume",
        MEDIA_RELEASE_TAG="v0.2.0",
        MEDIA_MANIFEST_SHA256="d" * 64,
        GITHUB_OUTPUT=str(tmp_path / "output"),
    )
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", command], env=environment, text=True, capture_output=True, check=True
    )
    arguments = json.loads(result.stdout)
    assert "resume" in arguments and "preflight" not in arguments
    assert arguments[arguments.index("--release-tag") + 1] == "v0.2.0"
    assert arguments[arguments.index("--manifest-sha256") + 1] == "d" * 64


@pytest.mark.skipif(sys.platform == "win32", reason="Acceptance runs in a Linux container with Unix HOME ownership")
def test_acceptance_isolates_home_before_running_any_tool(tmp_path):
    workflow = yaml.safe_load((ROOT / ".github/workflows/media-release-verify.yml").read_text())
    commands = [step for step in workflow["jobs"]["verify"]["steps"] if "run" in step]
    step = commands[0]
    assert step["name"] == "Isolate HOME for the verified artifact cache"
    assert not step.get("if") and not step.get("continue-on-error")
    output = tmp_path / "github-env"
    subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        env={**os.environ, "GITHUB_ENV": str(output)},
        check=True,
    )
    assignments = dict(line.split("=", 1) for line in output.read_text().splitlines())
    isolated = Path(assignments["HOME"])
    assert isolated.parent == Path("/tmp") and isolated.name.startswith("media-acceptance-home.")
    try:
        assert isolated.stat().st_uid == os.geteuid()
        assert stat.S_IMODE(isolated.stat().st_mode) == 0o700
        assert not list(isolated.iterdir())
    finally:
        isolated.rmdir()


@pytest.mark.parametrize("operation", ["release", "resume"])
def test_cancelled_workflow_cannot_resume_publication(operation):
    jobs = yaml.safe_load((ROOT / ".github/workflows/media-release.yml").read_text())["jobs"]
    results = dict.fromkeys(jobs, "success")
    for name in (
        "package",
        "candidate",
        "acceptance",
        "publish-testpypi",
        "verify-testpypi",
        "publish-pypi",
        "verify-pypi",
        "complete",
    ):
        assert not _condition(jobs[name]["if"], operation, results, cancelled=True)


@pytest.mark.parametrize(
    "command,function", [("publish-candidate", "publish_github"), ("promote-github", "promote_github")]
)
def test_publication_cli_forwards_recovery_identity(candidate, tmp_path, monkeypatch, command, function):
    calls = []
    monkeypatch.setattr(publishing, function, lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "media_release_pipeline.py",
            command,
            "--release-tag",
            "v0.2.0",
            "--manifest-sha256",
            candidate.digest,
            "--directory",
            str(tmp_path),
        ],
    )
    runpy.run_path(str(ROOT / "scripts/media_release_pipeline.py"), run_name="__main__")
    assert calls == [{"directory": tmp_path, "digest": candidate.digest, "release_tag": "v0.2.0"}]

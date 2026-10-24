# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Qualification must reject changed inputs and incomplete acceptance results."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("iceberg_qualification", ROOT / "scripts/qualify_iceberg.py")
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    runtime = tmp_path / "recipe"
    project = tmp_path / "project"
    directory = tmp_path / "bundle"
    for path in (runtime, project, directory / "context/wheels", directory / "test-wheels"):
        path.mkdir(parents=True)
    for name in ("Dockerfile", "runtime-requirements.txt"):
        (runtime / name).write_text("runtime input\n")
        (directory / "context" / name).write_text("runtime input\n")
    for name in ("requirements-test.txt", "requirements-qualification.txt"):
        (project / name).write_text("driver input\n")
        (directory / "test-wheels" / name).write_text("driver input\n")
    wheel = directory / "context/wheels/provider.whl"
    wheel.write_bytes(b"fixture wheel")
    monkeypatch.setattr(qualification, "RUNTIME", runtime)
    monkeypatch.setattr(qualification, "PROJECT", project)
    monkeypatch.setattr(qualification, "source_profile", lambda: {"source": "pinned"})
    files = [path for path in directory.rglob("*") if path.is_file()]
    record = {
        "format_version": 1,
        "test_only": True,
        "image": "sha256:" + "a" * 64,
        "profile": {"source": "pinned"},
        "files": [{"path": str(path.relative_to(directory)), "sha256": qualification.digest(path)} for path in files],
    }
    qualification.write_json(directory / "bundle.json", record)
    return directory, record


@pytest.mark.parametrize("change", ["bytes", "extra-wheel", "symlink", "source-pin", "mutable-image", "recipe"])
def test_replay_rejects_changed_bundle_before_starting_services(bundle, monkeypatch, change):
    directory, record = bundle
    wheel = directory / "context/wheels/provider.whl"
    if change == "bytes":
        wheel.write_bytes(b"changed")
    elif change == "extra-wheel":
        (wheel.parent / "unexpected.whl").write_bytes(b"extra")
    elif change == "symlink":
        target = directory.parent / "replacement"
        target.write_bytes(wheel.read_bytes())
        wheel.unlink()
        wheel.symlink_to(target)
    elif change == "source-pin":
        record["profile"] = {"source": "different"}
    elif change == "mutable-image":
        record["image"] = "vane:latest"
    else:
        (qualification.RUNTIME / "Dockerfile").write_text("new runtime recipe")
    qualification.write_json(directory / "bundle.json", record)
    monkeypatch.setattr(qualification, "Commands", lambda *_a, **_k: pytest.fail("must validate inputs first"))
    with pytest.raises(ValueError):
        qualification.qualify(directory, SimpleNamespace(docker_command="docker"))


def test_unchanged_bundle_replays_its_complete_inventory(bundle):
    directory, record = bundle
    assert qualification.verify_bundle(directory) == record


@pytest.mark.parametrize("problem", ["failure", "error", "skipped", "empty", "duplicate"])
def test_acceptance_rejects_failed_skipped_empty_or_duplicate_junit(tmp_path, problem):
    report = tmp_path / "report.xml"
    case = '<testcase classname="iceberg" name="read"/>'
    if problem in {"failure", "error", "skipped"}:
        contents = f'<testcase classname="iceberg" name="read"><{problem}/></testcase>'
    elif problem == "empty":
        contents = ""
    else:
        contents = case * 2
    report.write_text("<testsuite>" + contents + "</testsuite>")
    with pytest.raises(ValueError):
        qualification.junit_result(report)


def test_acceptance_counts_only_unique_passing_cases(tmp_path):
    report = tmp_path / "report.xml"
    report.write_text('<testsuites><testsuite><testcase classname="iceberg" name="read"/></testsuite></testsuites>')
    result = qualification.junit_result(report)
    assert result["passed"] == 1
    assert result["cases"] == ["iceberg::read"]


def test_runtime_probe_requires_exactly_one_identity_record():
    marker = 'VANE_QUALIFICATION={"version":"fixture"}\n'
    assert qualification.parse_probe("runtime log\n" + marker) == {"version": "fixture"}
    for output in ("", marker * 2):
        with pytest.raises(ValueError, match="exactly one"):
            qualification.parse_probe(output)


def test_qualification_does_not_inherit_writer_keys_or_runner_overrides(monkeypatch):
    for key in (
        "AWS_SECRET_ACCESS_KEY",
        "AWS_PROFILE",
        "RAY_ADDRESS",
        "VANE_TEST_RAY_OBJECT_STORE_BYTES",
        "PYTHONPATH",
    ):
        monkeypatch.setenv(key, "fixture-override")
    environment = qualification.clean_environment()
    assert not any(
        environment.get(key)
        for key in (
            "AWS_SECRET_ACCESS_KEY",
            "AWS_PROFILE",
            "RAY_ADDRESS",
            "VANE_TEST_RAY_OBJECT_STORE_BYTES",
            "PYTHONPATH",
        )
    )
    assert environment["AWS_EC2_METADATA_DISABLED"] == "true"


def test_wrong_sdk_baseline_fails_before_cloning_or_patching_sources(tmp_path, monkeypatch):
    import build_iceberg_fixture as builder

    vcpkg = tmp_path / "vcpkg"
    vcpkg.mkdir()
    monkeypatch.setattr(builder.subprocess, "check_output", lambda *_a, **_k: "0" * 40)
    args = SimpleNamespace(vcpkg_root=vcpkg)
    with pytest.raises(ValueError, match="baseline"):
        builder.build_inputs(tmp_path, args, lambda *_a, **_k: pytest.fail("must fail before mutation"), {})
    assert not (tmp_path / "sources").exists()


def test_host_services_receive_eof_and_exit_when_the_test_shard_raises(tmp_path, monkeypatch):
    helpers = tmp_path / "tests"
    helpers.mkdir()
    (helpers / "iceberg_qualification_services.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "print('VANE_SERVICES_READY', flush=True)\n"
        "sys.stdin.readline()\n"
        "Path(sys.argv[1], 'closed').touch()\n"
    )
    monkeypatch.setattr(qualification, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="test shard failed"):
        with qualification.host_services(
            Path(sys.executable), tmp_path, "fixture-image", qualification.clean_environment()
        ):
            raise RuntimeError("test shard failed")
    assert (tmp_path / "closed").is_file()

# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

import scripts.verify_extension_wheel as verifier
from tests.fast.test_extension_wheel import _write_minimal_base_wheel
from tests.fast.test_native_runtime_extension_wheels import TRUST_IDENTITY
from tests.fast.test_native_runtime_extension_wheels import media_dependency as media_dependency
from tests.fast.test_native_runtime_extension_wheels import release_runtime as release_runtime
from tests.fast.test_native_runtime_extension_wheels import runtime_wheel as runtime_wheel
from tests.fast.test_native_runtime_extension_wheels import source_sdk as source_sdk
from vane_packaging import media_release as delivery
from vane_packaging.artifact_limits import MAX_PUBLICATION_FILE_BYTES, MEBIBYTE
from vane_packaging.python_delivery import inventory_delivery


@pytest.fixture
def release_inputs(tmp_path, release_runtime, media_dependency, monkeypatch):
    calls = []

    def clean_install(**kwargs):
        # Native execution is covered by signed integration fixtures. Keep all
        # archive/layout/source/platform checks active for these synthetic ELFs.
        calls.append(kwargs["root_layout"].identity)
        assert kwargs["runtime_wheel"].path.is_file()
        assert kwargs["base_wheel"].path.is_file()
        assert kwargs["extension_wheel"].path.is_file()

    monkeypatch.setattr(verifier, "_verify_extension_wheel_snapshots", clean_install)
    return {
        "base": _write_minimal_base_wheel(tmp_path, platform_tag="manylinux_2_28_x86_64"),
        "provider": media_dependency.path,
        "runtime": release_runtime[0],
        "source": release_runtime[1],
        "trust_identity": TRUST_IDENTITY,
    }, calls


def test_complete_delivery_is_verified_before_exposure_and_can_be_rechecked(release_inputs, tmp_path):
    inputs, calls = release_inputs
    output = tmp_path / "delivery"
    digest = delivery.prepare_release(**inputs, output=output)
    manifest = delivery.verify_release(output, trust_identity=TRUST_IDENTITY, manifest_sha256=digest)
    assert len(calls) == 2
    assert set(manifest["artifacts"]) == {"base", "provider", "runtime", "source", "instructions"}
    assert len(list(output.iterdir())) == 6
    assert hashlib.sha256((output / delivery.MANIFEST).read_bytes()).hexdigest() == digest
    assert (output / delivery.INSTRUCTIONS).read_bytes() == (
        Path(__file__).resolve().parents[2] / delivery.INSTRUCTIONS
    ).read_bytes()
    with pytest.raises(ValueError, match="new directory"):
        delivery.prepare_release(**inputs, output=output)


@pytest.mark.parametrize("role", ["base", "provider"])
@pytest.mark.parametrize("size", [110 * MEBIBYTE, MAX_PUBLICATION_FILE_BYTES])
def test_delivery_stages_wheels_within_the_full_publication_budget(release_inputs, tmp_path, monkeypatch, role, size):
    inputs, _ = release_inputs
    # Isolate delivery byte limits from archive/native validation. Sparse input
    # padding exercises the real bounded copying and manifest verification.
    with inputs[role].open("r+b") as wheel:
        wheel.truncate(size)
    verified_sizes = []

    def verify_artifacts(directory, manifest, trust_identity):
        verified_sizes.append((directory / manifest["artifacts"][role]["filename"]).stat().st_size)

    monkeypatch.setattr(delivery, "_verify_contents", verify_artifacts)
    output = tmp_path / "delivery"
    digest = delivery.prepare_release(**inputs, output=output)
    manifest = delivery.verify_release(output, trust_identity=TRUST_IDENTITY, manifest_sha256=digest)
    assert manifest["artifacts"][role]["size"] == size
    assert verified_sizes == [size, size]


@pytest.mark.parametrize(
    ("role", "limit"),
    [
        ("base", MAX_PUBLICATION_FILE_BYTES),
        ("provider", MAX_PUBLICATION_FILE_BYTES),
        ("runtime", 100 * MEBIBYTE),
        ("source", 100 * MEBIBYTE),
    ],
)
def test_delivery_preserves_each_artifacts_publication_boundary(release_inputs, tmp_path, role, limit):
    inputs, calls = release_inputs
    output = tmp_path / "delivery"
    delivery.prepare_release(**inputs, output=output)
    path = output / delivery.MANIFEST
    manifest = json.loads(path.read_bytes())
    record = manifest["artifacts"][role]
    record["size"] = limit
    path.write_bytes(delivery.runtime_format().canonical_json(manifest))
    delivery.read_manifest(path, trust_identity=TRUST_IDENTITY)
    record["size"] += 1
    path.write_bytes(delivery.runtime_format().canonical_json(manifest))
    with pytest.raises(ValueError, match="invalid artifact size"):
        delivery.read_manifest(path, trust_identity=TRUST_IDENTITY)

    with inputs[role].open("r+b") as artifact:
        artifact.truncate(limit + 1)
    rejected = tmp_path / "oversized-delivery"
    with pytest.raises(ValueError, match="exceeds"):
        delivery.prepare_release(**inputs, output=rejected)
    assert not rejected.exists()
    assert len(calls) == 1


@pytest.mark.parametrize("damage", ["missing", "bytes", "symlink", "extra", "wrong-manifest", "wrong-trust"])
def test_delivery_rejects_missing_or_substituted_published_materials(release_inputs, tmp_path, damage):
    inputs, calls = release_inputs
    output = tmp_path / "delivery"
    digest = delivery.prepare_release(**inputs, output=output)
    source = output / inputs["source"].name
    if damage in {"missing", "symlink"}:
        source.unlink()
        if damage == "symlink":
            source.symlink_to(inputs["source"])
    elif damage == "bytes":
        source.write_bytes(b"not the corresponding sources")
    elif damage == "extra":
        (output / "unexpected.txt").write_text("unreviewed material")
    elif damage == "wrong-manifest":
        digest = "f" * 64
    with pytest.raises(ValueError):
        delivery.verify_release(
            output,
            trust_identity="untrusted" if damage == "wrong-trust" else TRUST_IDENTITY,
            manifest_sha256=digest,
        )
    assert len(calls) == 1


def test_staging_failure_never_publishes_partial_delivery(release_inputs, runtime_wheel, tmp_path):
    inputs, calls = release_inputs
    output = tmp_path / "delivery"
    with pytest.raises(ValueError, match="test-only runtime"):
        delivery.prepare_release(**dict(inputs, runtime=runtime_wheel), output=output)
    assert not output.exists()
    assert not list(tmp_path.glob(".media-release-*"))
    assert not calls


def test_staging_rejects_sources_even_if_their_filename_matches(release_inputs, tmp_path):
    inputs, calls = release_inputs
    inputs["source"].write_bytes(b"same filename, unrelated source")
    with pytest.raises(ValueError, match="corresponding-source archive digest"):
        delivery.prepare_release(**inputs, output=tmp_path / "delivery")
    assert not calls


@pytest.mark.parametrize("change", ["traversal", "oversize", "unknown", "duplicate", "no-instructions"])
def test_manifest_is_strict_before_resolving_paths(release_inputs, tmp_path, change):
    inputs, _ = release_inputs
    output = tmp_path / "delivery"
    delivery.prepare_release(**inputs, output=output)
    path = output / delivery.MANIFEST
    value = json.loads(path.read_bytes())
    records = value["artifacts"]
    if change == "traversal":
        records["source"]["filename"] = "../source.tar.gz"
    elif change == "oversize":
        records["source"]["size"] = 101 * 1024 * 1024
    elif change == "unknown":
        value["download_url"] = "https://unreviewed.invalid"
    elif change == "duplicate":
        records["runtime"] = records["provider"]
    else:
        records.pop("instructions")
    path.write_bytes(delivery.runtime_format().canonical_json(value))
    with pytest.raises(ValueError):
        delivery.read_manifest(path, trust_identity=TRUST_IDENTITY)


class _Response(io.BytesIO):
    def __init__(self, data, url):
        super().__init__(data)
        self.url = url


@pytest.mark.parametrize("damage", [None, "missing-source", "changed-source", "changed-manifest", "too-large", "http"])
def test_download_verifies_every_public_file_against_the_retained_manifest(
    release_inputs, tmp_path, monkeypatch, damage
):
    inputs, calls = release_inputs
    published = tmp_path / "published"
    digest = delivery.prepare_release(**inputs, output=published)
    retained = tmp_path / "retained.json"
    shutil.copyfile(published / delivery.MANIFEST, retained)
    requests = []

    class Opener:
        def open(self, request, timeout):
            name = unquote(urlsplit(request.full_url).path.rsplit("/", 1)[1])
            requests.append(name)
            if name == inputs["source"].name and damage == "missing-source":
                raise OSError("404: corresponding source was never published")
            contents = (published / name).read_bytes()
            if damage == "changed-manifest" and name == delivery.MANIFEST:
                contents = b" " + contents[1:]
            if name == inputs["source"].name:
                if damage == "changed-source":
                    contents = bytes(len(contents))
                if damage == "too-large":
                    contents += b"unreviewed trailing bytes"
            return _Response(contents, "http://downgraded.invalid" if damage == "http" else request.full_url)

    monkeypatch.setattr(delivery, "build_opener", lambda *args: Opener())
    arguments = dict(
        base_url="https://releases.example/media",
        expected_manifest=retained,
        trust_identity=TRUST_IDENTITY,
        output=tmp_path / "downloaded",
    )
    if damage:
        with pytest.raises((ValueError, OSError)):
            delivery.download_release(**arguments)
        assert not arguments["output"].exists()
        assert len(calls) == 1
    else:
        assert delivery.download_release(**arguments) == digest
        assert set(requests) == {path.name for path in published.iterdir()}
        assert len(calls) == 2


def test_manifest_and_http_redirect_cannot_select_credentials_or_insecure_transport():
    for value in ("http://example.com", "https://user:password@example.com", "https://example.com/#fragment"):
        with pytest.raises(ValueError, match="HTTPS"):
            delivery._https_url(value)
    with pytest.raises(ValueError, match="HTTPS"):
        delivery._HTTPSRedirects().redirect_request(None, None, 302, "redirect", {}, "http://example.com")


def test_python_delivery_inventory_keeps_wrapper_metadata_and_native_files_separate(release_inputs):
    inputs, _ = release_inputs
    report = inventory_delivery([inputs["runtime"], inputs["provider"]])
    assert len(report["wheels"]) == 2
    for record in report["wheels"]:
        assert record["review_status"] == "required"
        assert record["license_notices"]
        assert record["native_binaries"]
        assert record["license_expression"]
        path = next(path for path in inputs.values() if isinstance(path, Path) and path.name == record["filename"])
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="unique"):
        inventory_delivery([inputs["runtime"], inputs["runtime"]])


def test_python_delivery_accepts_zip_directories_without_record_entries(release_inputs, tmp_path):
    inputs, _ = release_inputs
    wheel = tmp_path / "ordinary" / inputs["runtime"].name
    wheel.parent.mkdir()
    shutil.copyfile(inputs["runtime"], wheel)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("vane_media_runtime/", b"")
        archive.writestr("vane_media_runtime/.libs/", b"")
    record = inventory_delivery([wheel])["wheels"][0]
    assert record["native_binaries"]
    assert record["review_status"] == "required"


def test_rebuild_rejects_unverified_delivery_before_executing_sources(release_inputs, tmp_path, monkeypatch):
    from vane_packaging import media_rebuild

    inputs, _ = release_inputs
    output = tmp_path / "delivery"
    digest = delivery.prepare_release(**inputs, output=output)
    (output / inputs["source"].name).write_bytes(b"unreviewed executable source")
    monkeypatch.setattr(media_rebuild.subprocess, "run", lambda *a, **k: pytest.fail("unverified source executed"))
    with pytest.raises(ValueError, match="source differs"):
        media_rebuild.rebuild_release(
            output, trust_identity=TRUST_IDENTITY, manifest_sha256=digest, output=tmp_path / "rebuilt"
        )

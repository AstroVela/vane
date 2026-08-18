# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

cryptography = pytest.importorskip("cryptography")
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import vane
from vane.extension_repository import (
    SignedExtensionRepository,
    publish_extension_repository,
)
from vane.extensions import DynamicExtensionError
from vane_packaging.extension_wheel import build_extension_wheel

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _runtime_platform() -> str:
    connection = vane.connect()
    try:
        return connection.execute("SELECT platform FROM pragma_platform()").fetchone()[0]
    finally:
        connection.close()


def _wheel_platform_tag() -> str:
    return {
        "linux_amd64": "linux_x86_64",
        "linux_amd64_musl": "musllinux_1_2_x86_64",
        "linux_arm64": "linux_aarch64",
        "linux_arm64_musl": "musllinux_1_2_aarch64",
        "osx_amd64": "macosx_10_9_x86_64",
        "osx_arm64": "macosx_11_0_arm64",
        "windows_amd64": "win_amd64",
        "windows_arm64": "win_arm64",
    }[_runtime_platform()]


def _write_extension_artifact(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    footer = bytearray(512)
    fields = ["", "", "", "CPP", "test-version", vane.__git_revision__, _runtime_platform(), "4"]
    for index, value in enumerate(fields):
        start = index * 32
        footer[start : start + len(value)] = value.encode("ascii")
    path.write_bytes(payload + footer)
    return path


def _build_extension_wheel(tmp_path: Path, name: str, payload: bytes) -> Path:
    artifact = _write_extension_artifact(tmp_path / f"{name}.duckdb_extension", payload)
    return build_extension_wheel(
        artifact=artifact,
        extension_name=name,
        output_directory=tmp_path / "wheels",
        platform_tag=_wheel_platform_tag(),
        trust_identity="vane-tests",
        license_expression="Apache-2.0 AND MIT",
        license_files=[REPOSITORY_ROOT / "LICENSE", REPOSITORY_ROOT / "LICENSES" / "DuckDB-MIT.txt"],
    ).path


def _private_key_bytes(key: Ed25519PrivateKey) -> bytes:
    return key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _public_key_bytes(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _repository(tmp_path: Path, wheel: Path, key: Ed25519PrivateKey) -> SignedExtensionRepository:
    repository_directory = tmp_path / "repository"
    publish_extension_repository(
        extension_wheels=[wheel],
        output_directory=repository_directory,
        repository_id="vane-tests/repository",
        signer_identity="vane-tests",
        private_key=_private_key_bytes(key),
    )
    return SignedExtensionRepository(
        repository_url=repository_directory.as_uri(),
        repository_id="vane-tests/repository",
        signer_identity="vane-tests",
        trusted_public_key=_public_key_bytes(key),
        cache_directory=tmp_path / "cache",
    )


def test_signed_repository_materializes_exact_platform_wheel_contents(tmp_path):
    extension_wheel = _build_extension_wheel(tmp_path, "sample", b"signed repository payload")
    signing_key = Ed25519PrivateKey.generate()
    repository = _repository(tmp_path, extension_wheel, signing_key)

    installed = repository.install("sample")

    with zipfile.ZipFile(extension_wheel) as wheel:
        wheel_artifact = wheel.read("vane_extensions/sample/sample.duckdb_extension")
        wheel_descriptor = wheel.read("vane_extensions/sample/sample.dynamic-extension.json")
    local_artifact = installed.provider.find(installed.descriptor.identity)
    assert local_artifact is not None
    assert local_artifact.path.read_bytes() == wheel_artifact
    assert local_artifact.path.with_suffix(".dynamic-extension.json").read_bytes() == wheel_descriptor
    assert hashlib.sha256(local_artifact.path.read_bytes()).hexdigest() == installed.descriptor.sha256

    connection = vane.connect()
    try:
        assert (
            installed.resolver().resolve(connection, installed.descriptor)[-1].identity == installed.descriptor.identity
        )
    finally:
        connection.close()


def test_signed_repository_revalidates_and_recovers_an_invalid_cached_artifact(tmp_path):
    extension_wheel = _build_extension_wheel(tmp_path, "sample", b"cache recovery payload")
    signing_key = Ed25519PrivateKey.generate()
    repository = _repository(tmp_path, extension_wheel, signing_key)

    first = repository.install("sample")
    first_artifact = first.provider.find(first.descriptor.identity)
    assert first_artifact is not None
    first_artifact.path.write_bytes(b"tampered cache")

    recovered = repository.install("sample")
    recovered_artifact = recovered.provider.find(recovered.descriptor.identity)
    assert recovered_artifact is not None
    assert hashlib.sha256(recovered_artifact.path.read_bytes()).hexdigest() == recovered.descriptor.sha256


def test_signed_repository_rejects_a_tampered_index_even_when_a_valid_cache_exists(tmp_path):
    extension_wheel = _build_extension_wheel(tmp_path, "sample", b"signature validation payload")
    signing_key = Ed25519PrivateKey.generate()
    repository = _repository(tmp_path, extension_wheel, signing_key)
    repository.install("sample")

    index_path = tmp_path / "repository" / "repository.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["artifacts"][0]["descriptor"]["sha256"] = "0" * 64
    index_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")

    with pytest.raises(DynamicExtensionError, match="VANE_DYNAMIC_EXTENSION_REPOSITORY_SIGNATURE_INVALID"):
        repository.install("sample")


def test_signed_repository_requires_a_pinned_signer_and_secure_repository_url(tmp_path):
    extension_wheel = _build_extension_wheel(tmp_path, "sample", b"pinned signer payload")
    signing_key = Ed25519PrivateKey.generate()
    repository = _repository(tmp_path, extension_wheel, signing_key)

    with pytest.raises(DynamicExtensionError, match="VANE_DYNAMIC_EXTENSION_REPOSITORY_SIGNER_UNTRUSTED"):
        SignedExtensionRepository(
            repository_url=(tmp_path / "repository").as_uri(),
            repository_id=repository.repository_id,
            signer_identity="another-signer",
            trusted_public_key=_public_key_bytes(signing_key),
            cache_directory=tmp_path / "cache",
        ).install("sample")
    with pytest.raises(DynamicExtensionError, match="VANE_DYNAMIC_EXTENSION_REPOSITORY_CONFIG_INVALID"):
        SignedExtensionRepository(
            repository_url="http://extensions.example.invalid/vane/",
            repository_id=repository.repository_id,
            signer_identity=repository.signer_identity,
            trusted_public_key=_public_key_bytes(signing_key),
            cache_directory=tmp_path / "cache",
        )


def test_publisher_retains_prior_immutable_entries_across_later_wheel_releases(tmp_path):
    sample_wheel = _build_extension_wheel(tmp_path / "sample", "sample", b"first release")
    other_wheel = _build_extension_wheel(tmp_path / "other", "other", b"second release")
    signing_key = Ed25519PrivateKey.generate()
    repository_directory = tmp_path / "repository"

    publish_extension_repository(
        extension_wheels=[sample_wheel],
        output_directory=repository_directory,
        repository_id="vane-tests/repository",
        signer_identity="vane-tests",
        private_key=_private_key_bytes(signing_key),
    )
    before = json.loads((repository_directory / "repository.json").read_text(encoding="utf-8"))
    publish_extension_repository(
        extension_wheels=[other_wheel],
        output_directory=repository_directory,
        repository_id="vane-tests/repository",
        signer_identity="vane-tests",
        private_key=_private_key_bytes(signing_key),
    )
    after = json.loads((repository_directory / "repository.json").read_text(encoding="utf-8"))

    assert len(before["artifacts"]) == 1
    assert len(after["artifacts"]) == 2
    assert before["artifacts"][0] in after["artifacts"]
    repository = SignedExtensionRepository(
        repository_url=repository_directory.as_uri(),
        repository_id="vane-tests/repository",
        signer_identity="vane-tests",
        trusted_public_key=_public_key_bytes(signing_key),
        cache_directory=tmp_path / "cache",
    )
    assert repository.install("sample").descriptor.name == "sample"
    assert repository.install("other").descriptor.name == "other"


def test_publisher_rejects_replacing_an_immutable_extension_release_coordinate(tmp_path):
    first_wheel = _build_extension_wheel(tmp_path / "first", "sample", b"first immutable payload")
    replacement_wheel = _build_extension_wheel(tmp_path / "replacement", "sample", b"replacement immutable payload")
    signing_key = Ed25519PrivateKey.generate()
    repository_directory = tmp_path / "repository"

    publish_extension_repository(
        extension_wheels=[first_wheel],
        output_directory=repository_directory,
        repository_id="vane-tests/repository",
        signer_identity="vane-tests",
        private_key=_private_key_bytes(signing_key),
    )
    with pytest.raises(DynamicExtensionError, match="VANE_DYNAMIC_EXTENSION_REPOSITORY_VERSION_CONFLICT"):
        publish_extension_repository(
            extension_wheels=[replacement_wheel],
            output_directory=repository_directory,
            repository_id="vane-tests/repository",
            signer_identity="vane-tests",
            private_key=_private_key_bytes(signing_key),
        )

    index = json.loads((repository_directory / "repository.json").read_text(encoding="utf-8"))
    assert len(index["artifacts"]) == 1


def test_repository_cli_scripts_import_the_installed_vane_runtime_from_outside_the_checkout(tmp_path):
    for script_name in ("publish_extension_repository.py", "install_extension_from_repository.py"):
        completed = subprocess.run(
            [sys.executable, "-I", str(REPOSITORY_ROOT / "scripts" / script_name), "--help"],
            check=True,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        )

        assert "usage:" in completed.stdout

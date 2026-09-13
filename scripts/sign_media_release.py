#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Sign two bounded data files using system Python -I -S; never install or load build outputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_KEY_SHA256 = "8729fbfbf5276be4b159c0b698c9e4214edd72eaad3e21bcefc03bcb36dffaeb"
RELEASE_TAG = re.compile(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:\.post(0|[1-9][0-9]*))?")


def runtime_format():
    spec = importlib.util.spec_from_file_location("_media_signing_format", ROOT / "vane/_native_runtime_format.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def artifact_limit() -> int:
    # This checked-in module contains only stdlib-independent byte constants.
    # Do not confuse the compressed wheel budget with the larger native member.
    spec = importlib.util.spec_from_file_location("_media_signing_limits", ROOT / "vane_packaging/artifact_limits.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MAX_EXTENSION_ARTIFACT_BYTES


def require_context(environment: dict[str, str]) -> tuple[str, str]:
    for name, expected in {
        "GITHUB_REPOSITORY": "AstroVela/vane",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF_PROTECTED": "true",
    }.items():
        if environment.get(name) != expected:
            raise ValueError(f"media publication requires {name}={expected}")
    tag = environment.get("GITHUB_REF_NAME", "")
    commit = environment.get("GITHUB_SHA", "")
    if (
        RELEASE_TAG.fullmatch(tag) is None
        or environment.get("GITHUB_REF") != "refs/tags/" + tag
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
    ):
        raise ValueError("media publication requires a protected final Vane version tag and exact commit")
    return commit, tag[1:]


def signing_inputs(directory: Path, *, commit: str, version: str) -> tuple[bytes, bytes]:
    fmt = runtime_format()
    if {path.name for path in directory.iterdir()} != {"native_media.duckdb_extension", fmt.MANIFEST}:
        raise ValueError("signer accepts only the native_media artifact and runtime manifest")
    document = fmt.read_file(directory, fmt.MANIFEST, fmt.MAX_MANIFEST_BYTES)
    manifest = fmt.parse_manifest(document)
    expected_url = (
        f"https://github.com/AstroVela/vane/releases/download/native-media-{commit}/{manifest['source']['filename']}"
    )
    if (
        manifest["git_commit"] != commit
        or manifest["git_dirty"]
        or manifest["vane_version"] != version
        or manifest["platform"] != "manylinux_2_28_x86_64"
        or manifest["source"]["url"] != expected_url
    ):
        raise ValueError("runtime signing input differs from the reviewed release source or download location")
    artifact = fmt.read_file(directory, "native_media.duckdb_extension", artifact_limit())
    if (
        len(artifact) <= fmt.DUCKDB_FOOTER_SIZE
        or not artifact.startswith(b"\x7fELF")
        or artifact[-256:] != bytes(256)
        or fmt.trailer_digest(artifact) != hashlib.sha256(document).hexdigest()
    ):
        raise ValueError("native signing input must be unsigned and bound to the exact runtime manifest")
    return artifact, document


def require_key(contents: bytearray) -> None:
    result = subprocess.run(
        ["/usr/bin/openssl", "pkey", "-pubout", "-outform", "DER", "-passin", "pass:"],
        input=contents,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode or hashlib.sha256(result.stdout).hexdigest() != PRODUCTION_KEY_SHA256:
        raise ValueError("media release key differs from the reviewed production public fingerprint")


def sign_digest(key: Path, digest: bytes, temporary: Path) -> bytes:
    (temporary / "digest").write_bytes(digest)
    subprocess.run(
        [
            "/usr/bin/openssl",
            "pkeyutl",
            "-sign",
            "-inkey",
            str(key),
            "-in",
            str(temporary / "digest"),
            "-out",
            str(temporary / "signature"),
            "-pkeyopt",
            "digest:sha256",
        ],
        check=True,
        timeout=30,
        env={"PATH": "/usr/bin:/bin"},
    )
    signature = (temporary / "signature").read_bytes()
    if len(signature) != 256:
        raise ValueError("media signing requires RSA-2048")
    return signature


def sign(directory: Path, output: Path) -> None:
    # Remove the secret before running any subprocess. Neither pip nor code from
    # the build/source archives runs in this job; imports come from the checkout.
    contents = bytearray(os.environ.pop("VANE_SIGNING_PRIVATE_KEY", "").encode())
    try:
        commit, version = require_context(dict(os.environ))
        actual = subprocess.check_output(
            ["/usr/bin/git", "-C", str(ROOT), "rev-parse", "HEAD"],
            env={"PATH": "/usr/bin:/bin"},
            text=True,
        ).strip()
        if actual != commit:
            raise ValueError("signing checkout differs from the dispatched commit")
        artifact, document = signing_inputs(directory, commit=commit, version=version)
        if not 0 < len(contents) <= 64 * 1024:
            raise ValueError("media release signing key must be nonempty and bounded")
        require_key(contents)
        output.mkdir(parents=True, exist_ok=False)
        with tempfile.TemporaryDirectory(prefix="vane-media-sign-", dir=os.environ["RUNNER_TEMP"]) as value:
            temporary = Path(value)
            key = temporary / "private.pem"
            try:
                with os.fdopen(os.open(key, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb") as stream:
                    stream.write(contents)
                payload = artifact[:-256]
                chunks = b"".join(
                    hashlib.sha256(payload[i : i + 1024 * 1024]).digest() for i in range(0, len(payload), 1024 * 1024)
                )
                signature = sign_digest(key, hashlib.sha256(chunks).digest(), temporary)
                (output / "native_media.duckdb_extension").write_bytes(payload + signature)
                signature = sign_digest(
                    key, hashlib.sha256(runtime_format().SIGNING_DOMAIN + document).digest(), temporary
                )
                (output / "runtime-manifest.sig").write_bytes(signature)
            finally:
                if key.exists():
                    with key.open("r+b") as stream:
                        stream.write(bytes(len(contents)))
                        stream.flush()
                        os.fsync(stream.fileno())
                    key.unlink()
    finally:
        contents[:] = bytes(len(contents))
        contents.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    sign(args.input_directory, args.output_directory)


if __name__ == "__main__":
    main()

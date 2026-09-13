# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage and retrieve an immutable native media delivery with corresponding sources."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from vane_packaging.archive_safety import snapshot_archive
from vane_packaging.artifact_limits import MAX_PUBLICATION_FILE_BYTES
from vane_packaging.media_runtime import read_runtime_wheel, verify_runtime_source
from vane_packaging.media_version import runtime_format

MANIFEST = "media-release.json"
INSTRUCTIONS = "NATIVE_MEDIA_REPLACEMENT.md"
_LIMITS = {
    "base": MAX_PUBLICATION_FILE_BYTES,
    "provider": MAX_PUBLICATION_FILE_BYTES,
    "runtime": 100 * 1024 * 1024,
    "source": 100 * 1024 * 1024,
    "instructions": 1024 * 1024,
}


def _filename(value: object) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,254}", value)
        or value.endswith(".")
    ):
        raise ValueError("release requires a bounded relative filename")
    runtime_format().filename(value.split("-", 1)[0])
    return value


@contextmanager
def _output_directory(destination: Path):
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("release output must be a new directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".media-release-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "delivery"
        stage.mkdir(mode=0o700)
        yield stage
        os.rename(stage, destination)


def _copy_file(source: Path, destination: Path, limit: int) -> dict:
    with snapshot_archive(
        source, max_bytes=limit, description="media release file", size_limit_description=f"the {limit} byte bound"
    ) as snapshot:
        with destination.open("xb") as output:
            shutil.copyfileobj(snapshot.file, output)
    return _file_record(destination, limit)


def _file_record(path: Path, limit: int) -> dict:
    with snapshot_archive(
        path, max_bytes=limit, description="media release file", size_limit_description=f"the {limit} byte bound"
    ) as snapshot:
        digest = hashlib.sha256()
        while contents := snapshot.file.read(1024 * 1024):
            digest.update(contents)
        return {"filename": path.name, "size": snapshot.size, "sha256": digest.hexdigest()}


def read_manifest(path: Path, *, trust_identity: str, sha256: str | None = None) -> tuple[bytes, dict]:
    fmt = runtime_format()
    with snapshot_archive(
        path, max_bytes=64 * 1024, description="media release manifest", size_limit_description="64 KiB"
    ) as snapshot:
        document = snapshot.file.read()
    if sha256 is not None and hashlib.sha256(document).hexdigest() != fmt.digest(sha256):
        raise ValueError("release manifest differs from the independently retained SHA-256")
    value = fmt.parse_json(document)
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "trust_identity", "artifacts"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["trust_identity"] != trust_identity
        or not trust_identity
        or not isinstance(value["artifacts"], dict)
        or set(value["artifacts"]) != set(_LIMITS)
        or fmt.canonical_json(value) != document
    ):
        raise ValueError("invalid media release manifest or explicit trust identity")
    names = {MANIFEST.casefold()}
    for role, record in value["artifacts"].items():
        if not isinstance(record, dict) or set(record) != {"filename", "size", "sha256"}:
            raise ValueError("invalid media release artifact record")
        name = _filename(record["filename"])
        fmt.digest(record["sha256"])
        if name.casefold() in names or type(record["size"]) is not int or not 0 < record["size"] <= _LIMITS[role]:
            raise ValueError("duplicate release filename or invalid artifact size")
        names.add(name.casefold())
        if role == "instructions" and name != INSTRUCTIONS:
            raise ValueError("release requires the replacement instructions")
        if role in {"base", "provider", "runtime"} and not name.endswith(".whl"):
            raise ValueError("release binary artifacts must be wheels")
        if role == "source" and not name.endswith(".tar.gz"):
            raise ValueError("release requires the source SDK distribution")
    return document, value


def _verify_contents(directory: Path, manifest: dict, trust_identity: str) -> None:
    from scripts.verify_extension_wheel import verify_extension_wheel

    records = manifest["artifacts"]
    expected = {MANIFEST, *(record["filename"] for record in records.values())}
    if {path.name for path in directory.iterdir()} != expected:
        raise ValueError("release directory has missing or unexpected files")
    paths = {}
    for role, record in records.items():
        path = directory / record["filename"]
        if _file_record(path, _LIMITS[role]) != record:
            raise ValueError(f"release {role} differs from its manifest")
        paths[role] = path
    # This checks the complete SDK inventory, notices and signed source hash.
    # The clean verifier also checks exact base/provider/runtime compatibility,
    # platform policy, immutable descriptors, native signatures and actual LOAD.
    runtime = read_runtime_wheel(paths["runtime"])
    verify_runtime_source(paths["source"], runtime[1])
    verify_extension_wheel(
        base_wheel=paths["base"],
        extension_wheel=paths["provider"],
        extension_name="native_media",
        trust_identity=trust_identity,
        runtime_wheel=paths["runtime"],
        runtime_source=paths["source"],
    )


def prepare_release(
    *, base: Path, provider: Path, runtime: Path, source: Path, trust_identity: str, output: Path
) -> str:
    """Verify private copies before exposing the complete directory for publication."""
    inputs = {"base": base, "provider": provider, "runtime": runtime, "source": source}
    inputs["instructions"] = Path(__file__).resolve().parents[1] / INSTRUCTIONS
    with _output_directory(output) as stage:
        records = {}
        for role, path in inputs.items():
            _filename(path.name)
            records[role] = _copy_file(path, stage / path.name, _LIMITS[role])
        document = runtime_format().canonical_json(
            {"schema_version": 1, "trust_identity": trust_identity, "artifacts": records}
        )
        (stage / MANIFEST).write_bytes(document)
        _, manifest = read_manifest(stage / MANIFEST, trust_identity=trust_identity)
        _verify_contents(stage, manifest, trust_identity)
    return hashlib.sha256(document).hexdigest()


@contextmanager
def verified_release(directory: Path, *, trust_identity: str, manifest_sha256: str):
    """Keep subsequent acceptance work on the same privately verified files."""
    document, manifest = read_manifest(directory / MANIFEST, trust_identity=trust_identity, sha256=manifest_sha256)
    # Keep hash verification and clean installation on the same private bytes.
    with tempfile.TemporaryDirectory(prefix="vane-media-verify-") as temporary:
        stage = Path(temporary)
        if {p.name for p in directory.iterdir()} != {
            MANIFEST,
            *(r["filename"] for r in manifest["artifacts"].values()),
        }:
            raise ValueError("release directory has missing or unexpected files")
        (stage / MANIFEST).write_bytes(document)
        for role, record in manifest["artifacts"].items():
            _copy_file(directory / record["filename"], stage / record["filename"], _LIMITS[role])
        _verify_contents(stage, manifest, trust_identity)
        yield stage, manifest


def verify_release(directory: Path, *, trust_identity: str, manifest_sha256: str) -> dict:
    """Check the delivery against a manifest digest retained outside the download."""
    with verified_release(directory, trust_identity=trust_identity, manifest_sha256=manifest_sha256) as (_, manifest):
        return manifest


def _https_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("media release downloads require HTTPS without embedded credentials or fragments")
    return url


class _HTTPSRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return super().redirect_request(request, fp, code, msg, headers, _https_url(newurl))


def _download(url: str, path: Path, expected: dict) -> None:
    opener = build_opener(_HTTPSRedirects())
    digest = hashlib.sha256()
    size = 0
    with opener.open(Request(_https_url(url), headers={"Accept-Encoding": "identity"}), timeout=60) as response:
        _https_url(response.url)
        with path.open("xb") as output:
            while contents := response.read(min(1024 * 1024, expected["size"] - size + 1)):
                size += len(contents)
                if size > expected["size"]:
                    raise ValueError("published artifact exceeds its expected size")
                digest.update(contents)
                output.write(contents)
    if size != expected["size"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError("published artifact differs from the retained release manifest")


def download_release(*, base_url: str, expected_manifest: Path, trust_identity: str, output: Path) -> str:
    """Retrieve every published file using a separately retained, reviewed manifest."""
    _https_url(base_url)
    if urlsplit(base_url).query:
        raise ValueError("release base URL cannot contain a query")
    document, manifest = read_manifest(expected_manifest, trust_identity=trust_identity)
    digest = hashlib.sha256(document).hexdigest()
    with _output_directory(output) as stage:
        records = [
            {"filename": MANIFEST, "size": len(document), "sha256": digest},
            *manifest["artifacts"].values(),
        ]
        for record in records:
            _download(base_url.rstrip("/") + "/" + quote(record["filename"]), stage / record["filename"], record)
        _verify_contents(stage, manifest, trust_identity)
    return digest

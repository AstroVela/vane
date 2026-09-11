# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Export the exact media sources and patched vcpkg recipes into a standalone SDK."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path, PurePosixPath

from vane_packaging.archive_safety import snapshot_archive, validate_tar_member_count
from vane_packaging.media_version import VERSION_FILE, identity_version, runtime_format, source_version

MAX_SOURCE_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_SOURCE_MEMBERS = 20000
MAX_SOURCE_MEMBER_BYTES = 256 * 1024 * 1024
MAX_SOURCE_TOTAL_BYTES = 512 * 1024 * 1024
MAX_SOURCE_METADATA_BYTES = 4 * 1024 * 1024
_SOURCE_METADATA_FILES = {"PKG-INFO", VERSION_FILE, "components.json", "source-inventory.json", "source-licenses.json"}

_REQUIRED_SOURCE_FILES = {
    "backend.py",
    "pyproject.toml",
    "components.json",
    "source-licenses.json",
    "LICENSE",
    "LICENSES/auditwheel-LICENSE.txt",
    "PKG-INFO",
    VERSION_FILE,
    "_native_runtime_format.py",
    "source-inventory.json",
    "vane_media_runtime/__init__.py",
    "vane_packaging/media_sources.py",
    "vane_packaging/archive_safety.py",
    "vane_packaging/media_runtime.py",
    "vane_packaging/media_version.py",
    "triplets/x64-linux-vane-media.cmake",
    "sdk/manifest/vcpkg.json",
    "sdk/vcpkg/.vcpkg-root",
    "sdk/vcpkg/LICENSE.txt",
    "sdk/vcpkg/bootstrap-vcpkg.sh",
    "sdk/vcpkg/scripts/bootstrap.sh",
    "sdk/vcpkg/scripts/buildsystems/vcpkg.cmake",
}


def read_source_file(path: Path) -> bytes:
    """Retain bounded, regular source bytes even if the supplied path changes."""
    with snapshot_archive(
        path,
        max_bytes=MAX_SOURCE_ARCHIVE_BYTES,
        description="runtime source archive",
        size_limit_description="the 100 MiB source publication limit",
    ) as snapshot:
        return snapshot.file.read()


def source_license_metadata(files, components, sources):
    """Bind the sdist's licensing to reviewed notices for its complete sources."""
    from packaging.licenses import canonicalize_license_expression

    from vane_packaging.media_runtime import PROJECT_LICENSE_SHA256, runtime_license_expression

    if hashlib.sha256(files["LICENSE"]).hexdigest() != PROJECT_LICENSE_SHA256:
        raise ValueError("unreviewed source SDK project license")
    notices = {"LICENSE", "sdk/vcpkg/LICENSE.txt", "LICENSES/auditwheel-LICENSE.txt"}
    expressions = {runtime_license_expression(components), "MIT"}
    reviewed = json.loads(files["source-licenses.json"])
    if not isinstance(reviewed, dict) or set(reviewed) != {record["filename"] for record in sources}:
        raise ValueError("source licenses must cover every corresponding source")
    for component, record in components.items():
        name = f"LICENSES/components/{component}.txt"
        if name not in files or hashlib.sha256(files[name]).hexdigest() != record["notice_sha256"]:
            raise ValueError(f"unreviewed source SDK component notice: {component}")
        notices.add(name)
    for source in sources:
        record = reviewed[source["filename"]]
        if (
            not isinstance(record, dict)
            or set(record) != {"sha512", "license", "notices"}
            or record["sha512"] != source["sha512"]
            or not isinstance(record["notices"], dict)
        ):
            raise ValueError("corresponding source differs from its license review")
        expressions.add(canonicalize_license_expression(record["license"]))
        for member, digest in record["notices"].items():
            path = PurePosixPath(member)
            if path.is_absolute() or ".." in path.parts or str(path) != member or "\\" in member:
                raise ValueError("invalid corresponding-source license path")
            name = f"LICENSES/sources/{source['filename']}/{member}"
            if name not in files or hashlib.sha256(files[name]).hexdigest() != digest:
                raise ValueError(f"unreviewed corresponding-source license notice: {member}")
            notices.add(name)
    if any(not files.get(name) for name in notices):
        raise ValueError("source SDK is missing a declared license file")
    expression = canonicalize_license_expression(" AND ".join(f"({value})" for value in sorted(expressions)))
    return expression, tuple(sorted(notices))


def source_metadata(release, expression, notices, *, private=False):
    # The sdist also carries unbuilt tools/documentation. Its broader licensing
    # and notice paths intentionally differ from the binary wheel's metadata.
    return (
        f"Metadata-Version: 2.4\nName: vane-media-runtime\nVersion: {release}\n"
        "Summary: Shared native media libraries for Vane extensions\nRequires-Python: >=3.10,<3.15\n"
        f"License-Expression: {expression}\nDynamic: License-Expression\nDynamic: License-File\n"
        "Dynamic: Classifier\n"
        + ("Classifier: Private :: Do Not Upload\n" if private else "")
        + "".join(f"License-File: {name}\n" for name in notices)
        + "\n"
    ).encode()


def read_source_archive(contents: bytes, filename: str) -> dict[str, bytes]:
    """Validate the complete source inventory before extracting or building it."""
    if not filename.endswith(".tar.gz") or not 0 < len(contents) <= MAX_SOURCE_ARCHIVE_BYTES:
        raise ValueError("invalid media source archive filename or size")
    stream = io.BytesIO(contents)
    validate_tar_member_count(
        stream,
        archive_path=filename,
        max_members=MAX_SOURCE_MEMBERS,
        max_member_bytes=MAX_SOURCE_MEMBER_BYTES,
        max_total_bytes=MAX_SOURCE_TOTAL_BYTES,
        member_limit_description="the 256 MiB source member limit",
        total_limit_description="the 512 MiB source total limit",
        description="media source archive",
    )
    stream.seek(0)
    files = {}
    folded = set()
    total = 0
    with tarfile.open(fileobj=stream, mode="r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or len(path.parts) < 2
                or path.parts[0] != filename[:-7]
                or ".." in path.parts
                or "\\" in member.name
                or str(path) != member.name
                or member.name.casefold() in folded
                or not 0 <= member.size <= MAX_SOURCE_MEMBER_BYTES
            ):
                raise ValueError("invalid media source distribution member")
            folded.add(member.name.casefold())
            total += member.size
            if total > MAX_SOURCE_TOTAL_BYTES or len(folded) > MAX_SOURCE_MEMBERS:
                raise ValueError("media source distribution exceeds its bounds")
            relative = path.relative_to(filename[:-7]).as_posix()
            if relative in _SOURCE_METADATA_FILES and member.size > MAX_SOURCE_METADATA_BYTES:
                raise ValueError("media source metadata member exceeds its size bound")
            files[relative] = archive.extractfile(member).read()
    missing = _REQUIRED_SOURCE_FILES - files.keys()
    if missing:
        raise ValueError(f"media source archive is missing required files: {sorted(missing)}")
    identity = runtime_format().parse_json(files[VERSION_FILE])
    if (
        set(identity) != {"git_commit", "git_dirty", "vane_version", "version"}
        or identity["version"] != identity_version(identity)
        or filename != f"vane_media_runtime-{identity['version']}.tar.gz"
    ):
        raise ValueError("media source archive differs from its Git identity")
    inventory = json.loads(files["source-inventory.json"])
    if not isinstance(inventory, dict) or set(inventory) != {"vcpkg_baseline", "ports", "sources", "files"}:
        raise ValueError("invalid media source inventory")
    records = inventory["files"]
    if not isinstance(records, dict) or set(records) != files.keys() - {"source-inventory.json"}:
        raise ValueError("media source archive differs from its complete file inventory")
    for name, digest in records.items():
        if hashlib.sha256(files[name]).hexdigest() != digest:
            raise ValueError(f"media source file differs from its inventory: {name}")
    ports = inventory["ports"]
    if not isinstance(ports, dict) or not ports:
        raise ValueError("media source inventory has no pinned recipes")
    for name, revision in ports.items():
        if not re.fullmatch(r"[a-z0-9-]+", name) or not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
            raise ValueError("invalid media source recipe identity")
        for relative in ("portfile.cmake", "vcpkg.json"):
            if f"sdk/ports/{name}/{relative}" not in files:
                raise ValueError(f"media source archive is missing recipe: {name}/{relative}")
    components = json.loads(files["components.json"])
    if not isinstance(components, dict) or not components or not components.keys() <= ports.keys():
        raise ValueError("media source archive is missing component recipes")
    sources = inventory["sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("media source inventory has no corresponding sources")
    names = set()
    for record in sources:
        if not isinstance(record, dict) or set(record) != {"filename", "sha512", "upstream"}:
            raise ValueError("invalid corresponding-source record")
        name = record["filename"]
        if not isinstance(name, str) or PurePosixPath(name).name != name or name in names:
            raise ValueError("invalid corresponding-source filename")
        names.add(name)
        value = files.get(f"sdk/downloads/{name}")
        if value is None or hashlib.sha512(value).hexdigest() != record["sha512"]:
            raise ValueError(f"media source archive is missing or changes corresponding sources: {name}")
    expression, notices = source_license_metadata(files, components, sources)
    metadata = BytesParser(policy=default).parsebytes(files["PKG-INFO"])
    for key, expected in {
        "Metadata-Version": ["2.4"],
        "Name": ["vane-media-runtime"],
        "Version": [identity["version"]],
        "Summary": ["Shared native media libraries for Vane extensions"],
        "Requires-Python": [">=3.10,<3.15"],
        "License-Expression": [expression],
        "License-File": list(notices),
        "Dynamic": ["License-Expression", "License-File", "Classifier"],
        "Classifier": ["Private :: Do Not Upload"] if identity["git_dirty"] else [],
    }.items():
        if metadata.get_all(key, []) != expected:
            raise ValueError(f"source SDK license/package metadata differs from its inventory: {key}")
    return files


def _git_files(repository: Path, tree: str, paths: tuple[str, ...] = ()) -> dict[str, bytes]:
    archive = subprocess.check_output(["git", "archive", tree, *paths], cwd=repository)
    result = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for member in stream:
            if member.isdir():
                continue
            if not member.isfile() or member.name.startswith("/") or ".." in Path(member.name).parts:
                raise ValueError("source SDK cannot export non-regular git archive members")
            result[member.name] = stream.extractfile(member).read()
    return result


def export_sdist(project: Path, repository: Path, installed: Path, downloads: Path, output: Path) -> Path:
    """Use installed SPDX checksums to include actual archives, never source URLs alone."""
    root = project.parents[1]
    manifest = json.loads((project / "vcpkg.json").read_bytes())
    identity = source_version(project)
    release = identity["version"]
    baseline = manifest.pop("builtin-baseline")
    prefix = f"vane_media_runtime-{release}"
    files = _git_files(
        repository,
        baseline,
        ("scripts", "triplets", "bootstrap-vcpkg.sh", "LICENSE.txt", ".vcpkg-root"),
    )
    files = {f"sdk/vcpkg/{name}": contents for name, contents in files.items()}
    # All port versions become local overlays: rebuilding an exported SDK needs
    # neither a Git checkout nor access to the vcpkg version registry.
    files["sdk/manifest/vcpkg.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    resources = {}
    ports = {}
    port_documents = {}
    source_records = []
    for metadata in sorted(installed.glob("*/share/*/vcpkg.spdx.json")):
        document = json.loads(metadata.read_bytes())
        package = next(value for value in document["packages"] if value["SPDXID"] == "SPDXRef-port")
        name = package["name"]
        match = re.fullmatch(
            r"git\+https://github.com/Microsoft/vcpkg@([0-9a-f]{40})",
            package["downloadLocation"],
        )
        if match is None:
            if package["downloadLocation"] != f"git+https://github.com/Microsoft/vcpkg#ports/{name}":
                raise ValueError(f"missing pinned vcpkg recipe provenance: {name}")
            tree = subprocess.check_output(
                ["git", "rev-parse", f"{baseline}:ports/{name}"],
                cwd=repository,
                text=True,
            ).strip()
        else:
            tree = match[1]
        if name in ports and ports[name] != tree:
            raise ValueError(f"conflicting host/target media recipe: {name}")
        ports[name] = tree
        port_documents[name] = document
        for resource in document["packages"]:
            if not resource["SPDXID"].startswith("SPDXRef-resource-"):
                continue
            hashes = [
                entry["checksumValue"] for entry in resource.get("checksums", []) if entry["algorithm"] == "SHA512"
            ]
            # vcpkg emits an unevaluated template for the Meson build tool.
            # Its pinned acquisition scripts are included; it is not a media library.
            if name == "vcpkg-tool-meson" and hashes == ["${download_sha512}"]:
                continue
            if len(hashes) != 1 or re.fullmatch(r"[0-9a-f]{128}", hashes[0]) is None:
                raise ValueError(f"source archive lacks an exact SHA512: {name}")
            resources[hashes[0]] = resource
    found = {}
    for candidate in sorted(downloads.iterdir()):
        if not candidate.is_file() or candidate.is_symlink() or candidate.name.endswith(".part"):
            continue
        with candidate.open("rb") as stream:
            hasher = hashlib.sha512()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
            checksum = hasher.hexdigest()
        if checksum in resources:
            found[checksum] = candidate
    if set(found) != set(resources):
        raise ValueError(f"missing corresponding source archives: {sorted(set(resources) - set(found))}")
    reviewed_sources = json.loads((project / "source-licenses.json").read_bytes())
    licensed_names = {record["sha512"]: name for name, record in reviewed_sources.items()}
    for checksum, candidate in found.items():
        if checksum not in licensed_names:
            raise ValueError(f"corresponding source requires a license review: {candidate.name}")
        # Caches can contain multiple filenames for identical source/license bytes.
        # Export the reviewed name, independent of the cache's incidental aliases.
        filename = licensed_names[checksum]
        files[f"sdk/downloads/{filename}"] = candidate.read_bytes()
        source_records.append(
            {
                "filename": filename,
                "sha512": checksum,
                "upstream": resources[checksum]["downloadLocation"],
            }
        )
    for name, tree in ports.items():
        port_files = _git_files(repository, tree)
        for record in port_documents[name]["files"]:
            if not record["SPDXID"].startswith("SPDXRef-port-file-"):
                continue
            relative = record["fileName"].removeprefix("./")
            hashes = [item["checksumValue"] for item in record["checksums"] if item["algorithm"] == "SHA256"]
            if (
                len(hashes) != 1
                or relative not in port_files
                or hashlib.sha256(port_files[relative]).hexdigest() != hashes[0]
            ):
                raise ValueError(f"exported recipe differs from the one actually built: {name}/{relative}")
        for relative, contents in port_files.items():
            files[f"sdk/ports/{name}/{relative}"] = contents
    for path in project.rglob("*"):
        if (
            path.is_file()
            and not path.is_symlink()
            and not set(path.relative_to(project).parts) & {"__pycache__", "dist", "build"}
        ):
            files[path.relative_to(project).as_posix()] = path.read_bytes()
    for path in (root / "vane_packaging").rglob("*"):
        if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    files["_native_runtime_format.py"] = (root / "vane/_native_runtime_format.py").read_bytes()
    for name in (
        "prepare_local_media_runtime.py",
        "prepare_dynamic_media_extension.py",
    ):
        files[f"scripts/{name}"] = (root / "scripts" / name).read_bytes()
    files["LICENSE"] = (root / "LICENSE").read_bytes()
    files["LICENSES/auditwheel-LICENSE.txt"] = (root / "LICENSES/auditwheel-LICENSE.txt").read_bytes()
    files[VERSION_FILE] = runtime_format().canonical_json(identity)
    components = json.loads(files["components.json"])
    for component in components:
        candidates = list(installed.glob(f"*/share/{component}/copyright"))
        values = {candidate.read_bytes() for candidate in candidates}
        if len(values) != 1:
            raise ValueError(f"missing or conflicting source SDK component notice: {component}")
        files[f"LICENSES/components/{component}.txt"] = values.pop()
    reviewed = json.loads(files["source-licenses.json"])
    for source in source_records:
        record = reviewed.get(source["filename"])
        if record is None or record["sha512"] != source["sha512"]:
            raise ValueError(f"corresponding source requires a license review: {source['filename']}")
        if not record["notices"]:
            continue
        contents = files[f"sdk/downloads/{source['filename']}"]
        if set(record["notices"]) == {source["filename"]}:
            extracted = {source["filename"]: contents}
        else:
            extracted = {}
            with tarfile.open(fileobj=io.BytesIO(contents)) as archive:
                for member in archive:
                    if member.name not in record["notices"]:
                        continue
                    if not member.isfile() or not 0 < member.size <= 2 * 1024 * 1024 or member.name in extracted:
                        raise ValueError("invalid corresponding-source license archive member")
                    extracted[member.name] = archive.extractfile(member).read()
        for member, notice in extracted.items():
            files[f"LICENSES/sources/{source['filename']}/{member}"] = notice
    expression, notices = source_license_metadata(files, components, source_records)
    files["PKG-INFO"] = source_metadata(release, expression, notices, private=identity["git_dirty"])
    files["source-inventory.json"] = (
        json.dumps(
            {
                "vcpkg_baseline": baseline,
                "ports": ports,
                "sources": source_records,
                "files": {name: hashlib.sha256(contents).hexdigest() for name, contents in files.items()},
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode()
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{prefix}.tar.gz"
    with tempfile.TemporaryDirectory(prefix=".media-sdist-", dir=output) as temporary:
        staged = Path(temporary) / target.name
        with (
            staged.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        ):
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for name, contents in sorted(files.items()):
                    info = tarfile.TarInfo(f"{prefix}/{name}")
                    info.size = len(contents)
                    info.mode = 0o755 if name.endswith(".sh") else 0o644
                    archive.addfile(info, io.BytesIO(contents))
        shutil.move(staged, target)
    return target

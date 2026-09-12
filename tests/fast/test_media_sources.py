# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import gzip
import hashlib
import importlib.util
import io
import json
import shutil
import struct
import subprocess
import tarfile
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

import pytest

import vane_packaging.archive_safety as archive_safety
import vane_packaging.media_sources as media_sources
from vane_packaging.media_runtime import exported_versions, read_runtime_wheel, verify_runtime_source
from vane_packaging.media_sources import read_source_archive, source_license_metadata, source_metadata
from vane_packaging.media_version import identity_version, runtime_format


@pytest.fixture
def source_sdk(tmp_path):
    root = Path(__file__).resolve().parents[2]
    project = tmp_path / "project"
    project.mkdir()
    identity = {"git_commit": "a" * 40, "git_dirty": False, "vane_version": "0.2.0.dev655"}
    identity["version"] = identity_version(identity)
    files = {
        name: (root / "packages/vane-media-runtime" / name).read_bytes()
        for name in (
            "backend.py",
            "pyproject.toml",
            "vane_media_runtime/__init__.py",
            "triplets/x64-linux-vane-media.cmake",
        )
    }
    for name in ("media_sources.py", "media_runtime.py", "media_version.py", "archive_safety.py"):
        files["vane_packaging/" + name] = (root / "vane_packaging" / name).read_bytes()
    files.update(
        {
            "LICENSE": (root / "LICENSE").read_bytes(),
            "NOTICE": (root / "NOTICE").read_bytes(),
            "LICENSES/auditwheel-LICENSE.txt": (root / "LICENSES/auditwheel-LICENSE.txt").read_bytes(),
            "LICENSES/components/soxr.txt": b"test component notice\n",
            "README.md": b"Source SDK\n",
            "PKG-INFO": b"Metadata-Version: 2.4\n",
            "runtime-version.json": runtime_format().canonical_json(identity),
            "_native_runtime_format.py": (root / "vane/_native_runtime_format.py").read_bytes(),
            "components.json": b'{"soxr": {}}',
            "sdk/manifest/vcpkg.json": b'{"dependencies": ["soxr"]}',
            "sdk/vcpkg/.vcpkg-root": b"",
            "sdk/vcpkg/LICENSE.txt": b"MIT vcpkg notice fixture\n",
            "sdk/vcpkg/bootstrap-vcpkg.sh": b"exit 0\n",
            "sdk/vcpkg/scripts/bootstrap.sh": b"exit 0\n",
            "sdk/vcpkg/scripts/buildsystems/vcpkg.cmake": b"# toolchain\n",
            "sdk/ports/soxr/portfile.cmake": b"# recipe\n",
            "sdk/ports/soxr/vcpkg.json": b'{"name": "soxr"}',
            "sdk/ports/soxr/fix.patch": b"upstream patch\n",
            "sdk/downloads/soxr.tar.gz": b"corresponding upstream source fixture\n",
        }
    )
    components = {
        "soxr": {
            "version": "0.1.3#8",
            "license": "LGPL-2.1-or-later",
            "notice_sha256": hashlib.sha256(files["LICENSES/components/soxr.txt"]).hexdigest(),
        }
    }
    files["components.json"] = json.dumps(components).encode()
    sources = [
        {
            "filename": "soxr.tar.gz",
            "sha512": hashlib.sha512(files["sdk/downloads/soxr.tar.gz"]).hexdigest(),
            "upstream": "https://example.org/soxr.tar.gz",
        }
    ]
    files["source-licenses.json"] = json.dumps(
        {"soxr.tar.gz": {"sha512": sources[0]["sha512"], "license": "LGPL-2.1-or-later", "notices": {}}}
    ).encode()
    expression, notices = source_license_metadata(files, components, sources)
    files["PKG-INFO"] = source_metadata(identity["version"], expression, notices)
    files["source-inventory.json"] = json.dumps(
        {
            "vcpkg_baseline": "b" * 40,
            "ports": {"soxr": "c" * 40},
            "sources": sources,
            "files": {name: hashlib.sha256(contents).hexdigest() for name, contents in files.items()},
        }
    ).encode()
    for name, contents in files.items():
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    archive = tmp_path / f"vane_media_runtime-{identity['version']}.tar.gz"
    spec = importlib.util.spec_from_file_location(
        "media_source_test_backend", root / "packages/vane-media-runtime/backend.py"
    )
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    backend.PROJECT = project
    return project, files, archive, backend


def _archive(path, files):
    with tarfile.open(path, "w:gz") as stream:
        for name, contents in files.items():
            member = tarfile.TarInfo(path.name[:-7] + "/" + name)
            member.size = len(contents)
            stream.addfile(member, io.BytesIO(contents))
    return path.read_bytes()


def _refresh_file_inventory(files):
    inventory = json.loads(files["source-inventory.json"])
    inventory["files"] = {
        name: hashlib.sha256(value).hexdigest() for name, value in files.items() if name != "source-inventory.json"
    }
    files["source-inventory.json"] = json.dumps(inventory).encode()


def test_source_archive_declares_its_project_component_and_build_notices(source_sdk):
    _, files, archive, _ = source_sdk
    read_source_archive(_archive(archive, files), archive.name)
    metadata = BytesParser(policy=default).parsebytes(files["PKG-INFO"])
    assert "Apache-2.0" in metadata["License-Expression"]
    assert "LGPL-2.1-or-later" in metadata["License-Expression"]
    assert "MIT" in metadata["License-Expression"]
    assert set(metadata.get_all("License-File")) == {
        "LICENSE",
        "NOTICE",
        "LICENSES/components/soxr.txt",
        "sdk/vcpkg/LICENSE.txt",
        "LICENSES/auditwheel-LICENSE.txt",
    }
    assert set(metadata.get_all("Dynamic")) == {"License-Expression", "License-File", "Classifier"}
    assert all(files[name] for name in metadata.get_all("License-File"))


@pytest.mark.parametrize(
    "damage", ["expression", "declaration", "notice", "project-license", "dynamic", "source-review"]
)
def test_source_archive_rejects_missing_or_tampered_license_metadata(source_sdk, damage):
    _, files, archive, _ = source_sdk
    if damage == "expression":
        files["PKG-INFO"] = b"\n".join(
            line for line in files["PKG-INFO"].split(b"\n") if not line.startswith(b"License-Expression:")
        )
    elif damage == "declaration":
        files["PKG-INFO"] = files["PKG-INFO"].replace(b"License-File: LICENSES/components/soxr.txt\n", b"")
    elif damage == "notice":
        files["LICENSES/components/soxr.txt"] = b"changed grant\n"
    elif damage == "project-license":
        files["LICENSE"] = b"different project grant\n"
    elif damage == "dynamic":
        files["PKG-INFO"] = files["PKG-INFO"].replace(b"Dynamic: License-Expression\n", b"")
    else:
        files["source-licenses.json"] = b"{}\n"
    # A self-consistent generic file inventory cannot substitute for licensing checks.
    _refresh_file_inventory(files)
    with pytest.raises(ValueError, match="license|notice"):
        read_source_archive(_archive(archive, files), archive.name)


def test_source_only_tool_license_is_separate_from_the_runtime_license(source_sdk):
    from vane_packaging.media_runtime import runtime_license_expression

    _, files, archive, _ = source_sdk
    inventory = json.loads(files["source-inventory.json"])
    source = {
        "filename": "tool.tar.gz",
        "sha512": hashlib.sha512(b"tool source").hexdigest(),
        "upstream": "https://example.org/tool.tar.gz",
    }
    inventory["sources"].append(source)
    files["source-inventory.json"] = json.dumps(inventory).encode()
    files["sdk/downloads/tool.tar.gz"] = b"tool source"
    notice = b"source-only GPL helper notice\n"
    files["LICENSES/sources/tool.tar.gz/tool/COPYING"] = notice
    reviewed = json.loads(files["source-licenses.json"])
    reviewed["tool.tar.gz"] = {
        "sha512": source["sha512"],
        "license": "GPL-3.0-or-later",
        "notices": {"tool/COPYING": hashlib.sha256(notice).hexdigest()},
    }
    files["source-licenses.json"] = json.dumps(reviewed).encode()
    components = json.loads(files["components.json"])
    expression, notices = source_license_metadata(files, components, inventory["sources"])
    assert "GPL-3.0-or-later" in expression
    assert "GPL-3.0-or-later" not in runtime_license_expression(components)
    identity = json.loads(files["runtime-version.json"])
    files["PKG-INFO"] = source_metadata(identity["version"], expression, notices)
    _refresh_file_inventory(files)
    read_source_archive(_archive(archive, files), archive.name)
    del files["LICENSES/sources/tool.tar.gz/tool/COPYING"]
    _refresh_file_inventory(files)
    with pytest.raises(ValueError, match="source license notice"):
        read_source_archive(_archive(archive, files), archive.name)


@pytest.mark.parametrize("underreported", [False, True])
def test_runtime_zip_member_limit_precedes_zipfile_construction(tmp_path, monkeypatch, underreported):
    path = tmp_path / "vane_media_runtime-0.1.0-py3-none-manylinux_2_28_x86_64.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        for index in range(513):
            wheel.writestr(f"entry-{index}", b"")
    if underreported:
        contents = bytearray(path.read_bytes())
        offset = contents.rfind(b"PK\x05\x06")
        contents[offset + 8 : offset + 12] = (1).to_bytes(2, "little") * 2
        path.write_bytes(contents)
    monkeypatch.setattr(archive_safety.zipfile, "ZipFile", lambda *a, **k: pytest.fail("ZIP parsed before preflight"))
    with pytest.raises(ValueError, match="more than 512 archive members"):
        read_runtime_wheel(path)


@pytest.mark.parametrize("member_type", [tarfile.XHDTYPE, tarfile.GNUTYPE_LONGNAME])
def test_source_pax_and_gnu_payload_limits_precede_tarfile_construction(monkeypatch, member_type):
    member = tarfile.TarInfo("oversized-metadata")
    member.type = member_type
    member.size = 2 * 1024 * 1024
    contents = gzip.compress(member.tobuf(format=tarfile.GNU_FORMAT) + b"payload must not be read")
    monkeypatch.setattr(media_sources.tarfile, "open", lambda *a, **k: pytest.fail("TAR parsed before preflight"))
    with pytest.raises(ValueError, match="TAR extension header.*metadata limit"):
        read_source_archive(contents, "vane_media_runtime-0.1.0.tar.gz")


def test_source_member_count_precedes_tarfile_construction(monkeypatch):
    contents = b"".join(tarfile.TarInfo(f"file-{index}").tobuf() for index in range(4))
    contents = gzip.compress(contents + bytes(1024))
    monkeypatch.setattr(media_sources, "MAX_SOURCE_MEMBERS", 3)
    monkeypatch.setattr(media_sources.tarfile, "open", lambda *a, **k: pytest.fail("TAR parsed before preflight"))
    with pytest.raises(ValueError, match="more than 3 archive members"):
        read_source_archive(contents, "vane_media_runtime-0.1.0.tar.gz")


def test_source_json_size_is_checked_before_reading_the_payload(tmp_path, monkeypatch):
    path = tmp_path / "vane_media_runtime-0.1.0.tar.gz"
    _archive(path, {"source-inventory.json": b" " * (media_sources.MAX_SOURCE_METADATA_BYTES + 1)})
    monkeypatch.setattr(tarfile.TarFile, "extractfile", lambda *a, **k: pytest.fail("oversized JSON was read"))
    with pytest.raises(ValueError, match="source metadata member exceeds"):
        read_source_archive(path.read_bytes(), path.name)


def test_runtime_metadata_size_is_checked_before_reading_the_payload(tmp_path, monkeypatch):
    path = tmp_path / "vane_media_runtime-0.1.0-py3-none-manylinux_2_28_x86_64.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr("vane_media_runtime/runtime-manifest.json", b" " * (64 * 1024 + 1))
    monkeypatch.setattr(zipfile.ZipFile, "read", lambda *a, **k: pytest.fail("oversized runtime manifest was read"))
    with pytest.raises(ValueError, match="runtime wheel metadata member exceeds"):
        read_runtime_wheel(path)


def test_runtime_preflight_and_parser_keep_the_same_snapshot(runtime_wheel, monkeypatch):
    expected = read_runtime_wheel(runtime_wheel, test_only=True)
    validate = archive_safety.validate_zip_member_count

    def replace_original_after_preflight(*args, **kwargs):
        result = validate(*args, **kwargs)
        runtime_wheel.write_bytes(b"replaced caller input")
        return result

    monkeypatch.setattr(archive_safety, "validate_zip_member_count", replace_original_after_preflight)
    assert read_runtime_wheel(runtime_wheel, test_only=True) == expected


def _version_definition_elf(name=b"FIXTURE_1"):
    header = bytearray(64)
    header[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<Q", header, 40, 64)
    struct.pack_into("<HH", header, 58, 64, 3)
    strings = b"\0" + name + b"\0"
    definition_offset = 256 + len(strings)
    section = struct.Struct("<IIQQQQIIQQ")
    headers = bytes(64)
    headers += section.pack(0, 3, 0, 0, 256, len(strings), 0, 0, 1, 0)
    headers += section.pack(0, 0x6FFFFFFD, 0, 0, definition_offset, 28, 1, 1, 4, 0)
    definition = struct.pack("<HHHHIII", 1, 0, 2, 1, 0, 20, 0) + struct.pack("<II", 1, 0)
    return bytearray(header + headers + strings + definition), definition_offset


def test_bounded_runtime_version_reader_accepts_a_definition():
    contents, _ = _version_definition_elf()
    assert exported_versions(bytes(contents)) == {"FIXTURE_1"}


@pytest.mark.parametrize(
    "damage", ["section-count", "auxiliary-count", "name-size", "definition-cycle", "string-bounds"]
)
def test_runtime_version_reader_rejects_unbounded_elf_metadata(damage):
    contents, definition = _version_definition_elf(b"x" * 129 if damage == "name-size" else b"FIXTURE_1")
    if damage == "section-count":
        struct.pack_into("<H", contents, 60, 65535)
    elif damage == "auxiliary-count":
        struct.pack_into("<H", contents, definition + 6, 4097)
    elif damage == "definition-cycle":
        struct.pack_into("<I", contents, 192 + 44, 2)
    elif damage == "string-bounds":
        struct.pack_into("<Q", contents, 128 + 24, 1 << 63)
    with pytest.raises(ValueError, match="runtime ELF"):
        exported_versions(bytes(contents))


@pytest.mark.parametrize(
    "missing",
    [
        "all",
        "readme-only",
        "source-inventory.json",
        "backend.py",
        "sdk/downloads/soxr.tar.gz",
        "sdk/ports/soxr/fix.patch",
    ],
)
def test_release_build_rejects_incomplete_source_archives_before_building(source_sdk, monkeypatch, missing):
    project, files, archive, backend = source_sdk
    if missing == "all":
        files.clear()
    elif missing == "readme-only":
        files = {"README.md": files["README.md"]}
    else:
        del files[missing]
    _archive(archive, files)
    monkeypatch.setattr(backend, "_build_wheel", lambda *args: pytest.fail("incomplete sources entered build"))
    # The missing inputs remain on disk: the archive must not rely on them.
    assert (project / "sdk/downloads/soxr.tar.gz").is_file()
    with pytest.raises(ValueError, match="missing required|file inventory"):
        backend.build_wheel(str(project / "dist"), {"source-archive": str(archive)})


@pytest.mark.parametrize(
    "damage",
    ["changed-file", "unlisted-file", "removed-recipe", "removed-source", "wrong-source-digest", "changed-version"],
)
def test_source_inventory_binds_all_files_and_corresponding_sources(source_sdk, damage):
    _, files, archive, _ = source_sdk
    inventory = json.loads(files["source-inventory.json"])
    if damage == "changed-file":
        files["sdk/ports/soxr/fix.patch"] += b"different patch\n"
    elif damage == "unlisted-file":
        files["sdk/ports/soxr/extra.patch"] = b"unrecorded patch\n"
    elif damage in {"removed-recipe", "removed-source"}:
        name = "sdk/ports/soxr/portfile.cmake" if damage == "removed-recipe" else "sdk/downloads/soxr.tar.gz"
        del files[name]
        del inventory["files"][name]
    elif damage == "wrong-source-digest":
        inventory["sources"][0]["sha512"] = "0" * 128
    else:
        files["runtime-version.json"] = b"{}\n"
    files["source-inventory.json"] = json.dumps(inventory).encode()
    with pytest.raises(ValueError, match="source|inventory|identity"):
        read_source_archive(_archive(archive, files), archive.name)


@pytest.mark.parametrize("name", ["../escape", "/escape", "sdk/../escape", "sdk\\escape", "sdk//escape", "README.MD"])
def test_source_archive_rejects_unsafe_or_ambiguous_paths(source_sdk, name):
    _, files, archive, _ = source_sdk
    files[name] = b"unexpected\n"
    with pytest.raises(ValueError, match="invalid media source distribution member"):
        read_source_archive(_archive(archive, files), archive.name)


def test_source_archive_rejects_links(source_sdk):
    _, files, archive, _ = source_sdk
    _archive(archive, files)
    # Create a fresh compressed archive with a link instead of a regular input.
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo(archive.name[:-7] + "/backend.py")
        member.type = tarfile.SYMTYPE
        member.linkname = "/outside/backend.py"
        stream.addfile(member)
    with pytest.raises(ValueError, match="invalid media source distribution member|unsupported TAR member type"):
        read_source_archive(archive.read_bytes(), archive.name)


def test_release_build_uses_only_the_verified_source_snapshot(source_sdk, monkeypatch):
    project, files, archive, backend = source_sdk
    _archive(archive, files)
    # Neither unarchived source nor an old installation may enter the snapshot.
    (project / "unarchived.c").write_text("unarchived source\n")
    (project / "build").mkdir()
    (project / "build/stale.so").write_bytes(b"old binary")
    snapshots = []

    def inspect_build(_output, _settings, _source, _contents, _identity, snapshot):
        snapshots.append(snapshot)
        assert snapshot != project
        assert {p.relative_to(snapshot).as_posix() for p in snapshot.rglob("*") if p.is_file()} == files.keys()
        original = files["sdk/ports/soxr/fix.patch"]
        (project / "sdk/ports/soxr/fix.patch").write_bytes(b"changed after validation")
        assert (snapshot / "sdk/ports/soxr/fix.patch").read_bytes() == original
        return "verified.whl"

    monkeypatch.setattr(backend, "_build_wheel", inspect_build)
    assert backend.build_wheel(str(project / "dist"), {"source-archive": str(archive)}) == "verified.whl"
    assert not snapshots[0].exists()


def test_sdk_build_rejects_a_previous_installation(source_sdk):
    project, _, _, backend = source_sdk
    (project / "build").mkdir()
    with pytest.raises(ValueError, match="fresh SDK extraction"):
        backend._build_sdk(project)


def test_runtime_source_verification_checks_structure_after_the_signed_digest(source_sdk):
    _, files, archive, _ = source_sdk
    contents = _archive(archive, files)
    manifest = {"source": {"filename": archive.name, "sha256": hashlib.sha256(contents).hexdigest()}}
    verify_runtime_source(archive, manifest)
    contents = _archive(archive, {"README.md": files["README.md"]})
    manifest["source"]["sha256"] = hashlib.sha256(contents).hexdigest()
    with pytest.raises(ValueError, match="missing required files"):
        verify_runtime_source(archive, manifest)


def test_sdk_build_uses_archived_recipes_and_disables_external_overlays(source_sdk, monkeypatch):
    project, _, _, backend = source_sdk
    monkeypatch.setenv("VCPKG_ROOT", "/external/vcpkg")
    monkeypatch.setenv("VCPKG_OVERLAY_PORTS", "/external/ports")
    monkeypatch.setenv("VCPKG_OVERLAY_TRIPLETS", "/external/triplets")
    monkeypatch.setenv("VCPKG_BINARY_SOURCES", "default,read")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        environment = kwargs["env"]
        assert environment["VCPKG_ROOT"] == str(project / "sdk/vcpkg")
        assert environment["VCPKG_DOWNLOADS"] == str(project / "sdk/downloads")
        assert environment["VCPKG_BINARY_SOURCES"] == "clear"
        assert "VCPKG_OVERLAY_PORTS" not in environment
        assert "VCPKG_OVERLAY_TRIPLETS" not in environment
        assert kwargs["cwd"] == project

    monkeypatch.setattr(backend.subprocess, "run", run)
    assert backend._build_sdk(project) == project / "build/installed/x64-linux-vane-media"
    assert len(calls) == 2
    assert f"--overlay-ports={project / 'sdk/ports'}" in calls[-1]
    assert f"--overlay-triplets={project / 'triplets'}" in calls[-1]


@pytest.fixture
def runtime_wheel(source_sdk):
    if not shutil.which("cc") or not shutil.which("patchelf"):
        pytest.skip("runtime wheel fixture requires a C compiler and patchelf")
    project, files, archive, backend = source_sdk
    prefix = project / "fixture-sdk"
    (prefix / "lib").mkdir(parents=True)
    (prefix / "source.c").write_text("int soxr_fixture(void) { return 42; }\n")
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            "-Wl,-soname,libsoxr.so.0",
            "-o",
            str(prefix / "lib/libsoxr.so.0"),
            str(prefix / "source.c"),
        ],
        check=True,
    )
    notice = b"test component notice\n"
    component = {
        "version": "0.1.3#8",
        "license": "LGPL-2.1-or-later",
        "notice_sha256": hashlib.sha256(notice).hexdigest(),
    }
    share = prefix / "share/soxr"
    share.mkdir(parents=True)
    (share / "copyright").write_bytes(notice)
    (share / "vcpkg.spdx.json").write_text(
        json.dumps(
            {
                "packages": [{"SPDXID": "SPDXRef-port", "versionInfo": component["version"]}],
                "files": [{"fileName": "./lib/libsoxr.so.0"}],
            }
        )
    )
    files["components.json"] = json.dumps({"soxr": component}).encode()
    inventory = json.loads(files["source-inventory.json"])
    inventory["files"]["components.json"] = hashlib.sha256(files["components.json"]).hexdigest()
    files["source-inventory.json"] = json.dumps(inventory).encode()
    for name in ("components.json", "source-inventory.json"):
        (project / name).write_bytes(files[name])
    _archive(archive, files)
    output = project / "dist"
    root = Path(__file__).resolve().parents[2]
    name = backend.build_wheel(
        str(output),
        {
            "source-archive": str(archive),
            "sdk-prefix": str(prefix),
            "test-only": "true",
            "platform-tag": "manylinux_2_28_x86_64",
            "signing-key": str(root / "external/duckdb/test/mbedtls/private.pem"),
        },
    )
    return output / name


def test_runtime_wheel_includes_project_and_component_licenses(runtime_wheel):
    _, manifest, _, _, _ = read_runtime_wheel(runtime_wheel, test_only=True)
    assert manifest["license_expression"] == "Apache-2.0 AND (LGPL-2.1-or-later)"
    with zipfile.ZipFile(runtime_wheel) as wheel:
        metadata_name = next(name for name in wheel.namelist() if name.endswith("/METADATA"))
        metadata = BytesParser(policy=default).parsebytes(wheel.read(metadata_name))
        assert metadata["License-Expression"] == manifest["license_expression"]
        assert set(metadata.get_all("License-File")) == {"LICENSE", "NOTICE", "soxr.txt"}
        project_license = metadata_name.removesuffix("METADATA") + "licenses/LICENSE"
        assert wheel.read(project_license) == (Path(__file__).resolve().parents[2] / "LICENSE").read_bytes()
        assert (
            wheel.read(project_license.removesuffix("LICENSE") + "NOTICE")
            == (Path(__file__).resolve().parents[2] / "NOTICE").read_bytes()
        )


@pytest.mark.parametrize("damage", ["missing-file", "changed-file", "missing-declaration", "missing-expression"])
def test_runtime_wheel_rejects_missing_or_changed_project_license(runtime_wheel, damage):
    with zipfile.ZipFile(runtime_wheel) as wheel:
        files = {name: wheel.read(name) for name in wheel.namelist()}
    metadata_name = next(name for name in files if name.endswith("/METADATA"))
    license_name = metadata_name.removesuffix("METADATA") + "licenses/LICENSE"
    if damage == "missing-file":
        del files[license_name]
    elif damage == "changed-file":
        files[license_name] = b"different license\n"
    elif damage == "missing-declaration":
        files[metadata_name] = files[metadata_name].replace(b"License-File: LICENSE\n", b"")
    else:
        manifest_name = "vane_media_runtime/runtime-manifest.json"
        manifest = json.loads(files[manifest_name])
        manifest["license_expression"] = "LGPL-2.1-or-later"
        files[manifest_name] = runtime_format().canonical_json(manifest)
        files[metadata_name] = files[metadata_name].replace(b"Apache-2.0 AND ", b"")
    with zipfile.ZipFile(runtime_wheel, "w") as wheel:
        for name, value in files.items():
            wheel.writestr(name, value)
    with pytest.raises(ValueError, match="missing or unowned|project license|license metadata|license expression"):
        read_runtime_wheel(runtime_wheel, test_only=True)


@pytest.mark.parametrize("damage", ["missing", "changed", "undeclared"])
def test_runtime_wheel_requires_the_reviewed_project_notice_with_valid_record(runtime_wheel, tmp_path, damage):
    from tests.fast.test_extension_wheel import _rewrite_wheel

    with zipfile.ZipFile(runtime_wheel) as wheel:
        metadata = next(name for name in wheel.namelist() if name.endswith("/METADATA"))
    notice = metadata.removesuffix("METADATA") + "licenses/NOTICE"
    options = {}
    if damage == "missing":
        options["removed_members"] = {notice}
    elif damage == "changed":
        options["transforms"] = {notice: lambda contents: b"removed project attribution\n"}
    else:
        options["transforms"] = {metadata: lambda contents: contents.replace(b"License-File: NOTICE\n", b"")}
    directory = tmp_path / "tampered"
    directory.mkdir()
    changed = _rewrite_wheel(runtime_wheel, directory / runtime_wheel.name, **options)
    with pytest.raises(ValueError, match="unowned members|project license/notice|license metadata"):
        read_runtime_wheel(changed, test_only=True)

# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import tarfile
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

import pytest

from vane_packaging.media_runtime import read_runtime_wheel, verify_runtime_source
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
    for name in ("media_sources.py", "media_runtime.py", "media_version.py"):
        files["vane_packaging/" + name] = (root / "vane_packaging" / name).read_bytes()
    files.update(
        {
            "LICENSE": (root / "LICENSE").read_bytes(),
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
    with pytest.raises(ValueError, match="invalid media source distribution member"):
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
        assert set(metadata.get_all("License-File")) == {"LICENSE", "soxr.txt"}
        project_license = metadata_name.removesuffix("METADATA") + "licenses/LICENSE"
        assert wheel.read(project_license) == (Path(__file__).resolve().parents[2] / "LICENSE").read_bytes()


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

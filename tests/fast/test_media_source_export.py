# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise SDK export from a real Git checkout and isolated download caches."""

import hashlib
import json
import shutil
import subprocess

import pytest

from tests.fast.test_media_sources import source_sdk as source_sdk
from vane_packaging import media_sources


@pytest.fixture
def checkout(source_sdk, tmp_path, monkeypatch):
    _, files, _, _ = source_sdk
    root = tmp_path / "checkout"
    project = root / "packages/vane-media-runtime"
    project.mkdir(parents=True)
    excluded = {"PKG-INFO", "runtime-version.json", "source-inventory.json"}
    for name, contents in files.items():
        if name in excluded or name.startswith(("sdk/", "LICENSES/components/")):
            continue
        if name == "_native_runtime_format.py":
            path = root / "vane/_native_runtime_format.py"
        elif name in {"LICENSE", "NOTICE", "LICENSES/auditwheel-LICENSE.txt"} or name.startswith("vane_packaging/"):
            path = root / name
        else:
            path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    (root / "pyproject.toml").write_text('[tool.setuptools_scm]\nlocal_scheme = "no-local-version"\n')
    (root / ".gitignore").write_text("*.pyc\n*.so\n*~\nbuild/\n")
    (project / "vcpkg.json").write_text(json.dumps({"builtin-baseline": "b" * 40, "dependencies": ["soxr"]}))
    resources = []
    review = {}
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    for name in ("alpha-source", "zeta-source"):
        contents = name.encode()
        checksum = hashlib.sha512(contents).hexdigest()
        (downloads / name).write_bytes(contents)
        resources.append(
            {
                "SPDXID": f"SPDXRef-resource-{name}",
                "checksums": [{"algorithm": "SHA512", "checksumValue": checksum}],
                "downloadLocation": f"https://example.org/{name}",
            }
        )
        review[name] = {"sha512": checksum, "license": "LGPL-2.1-or-later", "notices": {}}
    (project / "source-licenses.json").write_text(json.dumps(review))
    share = tmp_path / "installed/x64-linux-vane-media/share/soxr"
    share.mkdir(parents=True)
    (share / "copyright").write_bytes(files["LICENSES/components/soxr.txt"])
    (share / "vcpkg.spdx.json").write_text(
        json.dumps(
            {
                "packages": [
                    {
                        "SPDXID": "SPDXRef-port",
                        "name": "soxr",
                        "downloadLocation": "git+https://github.com/Microsoft/vcpkg@" + "c" * 40,
                    },
                    *resources,
                ],
                "files": [],
            }
        )
    )

    def git(*args):
        return subprocess.check_output(
            ["git", "-c", "user.name=Vane tests", "-c", "user.email=tests@example.invalid", *args],
            cwd=root,
        )

    git("init", "-q")
    git("add", ".")
    git("commit", "-qm", "source fixture")
    git("tag", "v0.1.0")

    def pinned_vcpkg_files(repository, tree, paths=()):
        # Only the upstream vcpkg object store is a fixture; Vane's tracked
        # source selection and working-tree provenance use the actual Git repo.
        assert repository == tmp_path / "vcpkg"
        prefix = "sdk/vcpkg/" if tree == "b" * 40 else "sdk/ports/soxr/"
        assert tree in {"b" * 40, "c" * 40}
        return {name.removeprefix(prefix): value for name, value in files.items() if name.startswith(prefix)}

    monkeypatch.setattr(media_sources, "_git_files", pinned_vcpkg_files)

    def export(name):
        archive = media_sources.export_sdist(
            project, tmp_path / "vcpkg", tmp_path / "installed", downloads, tmp_path / name
        )
        return archive, media_sources.read_source_archive(archive.read_bytes(), archive.name)

    return root, project, downloads, git, export


def test_export_ignores_local_editor_and_binary_files_in_both_source_roots(checkout):
    root, project, _, git, export = checkout
    for directory in (project, root / "vane_packaging"):
        for name in ("local.pyc", "local.so", "backend.py~"):
            (directory / name).write_bytes(b"private local fixture, never publish")
    assert git("status", "--porcelain") == b""
    _, files = export("ignored-files")
    assert not any(b"private local fixture" in contents for contents in files.values())
    assert json.loads(files["runtime-version.json"])["git_dirty"] is False


def test_download_aliases_do_not_change_the_exported_source_bytes(checkout):
    _, _, downloads, _, export = checkout
    first, _ = export("first")
    shutil.copyfile(downloads / "zeta-source", downloads / "000-alias")
    second, files = export("second")
    assert first.name == second.name
    assert first.read_bytes() == second.read_bytes()
    assert [item["filename"] for item in json.loads(files["source-inventory.json"])["sources"]] == [
        "alpha-source",
        "zeta-source",
    ]


def test_export_preserves_tracked_edits_and_binds_them_to_private_identity(checkout):
    _, project, _, git, export = checkout
    (project / "README.md").write_text("first local edit\n")
    first, before = export("first")
    assert before["README.md"] == b"first local edit\n"
    assert b"Private :: Do Not Upload" in before["PKG-INFO"]
    (project / "README.md").write_text("second local edit\n")
    second, after = export("second")
    assert first.name != second.name
    assert after["README.md"] == b"second local edit\n"
    git("add", ".")
    staged, _ = export("staged")
    assert second.name == staged.name
    assert second.read_bytes() == staged.read_bytes()


def test_checkout_export_does_not_use_a_stray_frozen_sdk_identity(checkout):
    _, project, _, git, export = checkout
    first, files = export("first")
    (project / "runtime-version.json").write_bytes(files["runtime-version.json"])
    (project / "README.md").write_text("changed checkout source\n")
    second, changed = export("second")
    identity = json.loads(changed["runtime-version.json"])
    assert identity["git_commit"] == git("rev-parse", "HEAD").decode().strip()
    assert identity["git_dirty"] is True
    assert identity["git_dirty_sha256"]
    assert second.name != first.name


@pytest.mark.parametrize("parent", [False, True])
def test_export_rejects_tracked_symlinks_and_symlink_ancestors(checkout, parent):
    root, project, _, _, export = checkout
    path = project / "vane_media_runtime" if parent else project / "backend.py"
    destination = root.parent / "outside"
    shutil.move(path, destination)
    path.symlink_to(destination, target_is_directory=parent)
    with pytest.raises(ValueError, match="symlink|Vane checkout"):
        export("symlink")


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_export_requires_the_reviewed_vane_notice(checkout, damage):
    root, _, _, _, export = checkout
    if damage == "missing":
        (root / "NOTICE").unlink()
    else:
        (root / "NOTICE").write_text("changed project attribution\n")
    with pytest.raises(ValueError, match="project license/notice"):
        export("notice")

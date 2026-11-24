# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real PEP 517 frontend/backend settings path used by native CI."""

import json
import sys
from pathlib import PurePosixPath, PureWindowsPath

import pytest

from scripts import ci_native_platform as ci


@pytest.mark.parametrize(
    "path_type,base",
    [(PureWindowsPath, r"D:\a\vane\checkout with spaces"), (PurePosixPath, "/tmp/vane checkout with spaces")],
)
def test_release_media_paths_survive_build_and_scikit_build_parsing(tmp_path, monkeypatch, path_type, base):
    pytest.importorskip("build")
    pytest.importorskip("scikit_build_core")
    (tmp_path / "pyproject.toml").write_text(
        '[build-system]\nrequires = []\nbuild-backend = "ci_path_probe_backend"\nbackend-path = ["."]\n'
    )
    (tmp_path / "ci_path_probe_backend.py").write_text(
        "import json, os\nfrom pathlib import Path\n"
        "from scikit_build_core.settings.skbuild_read_settings import SettingsReader\n"
        "def probe(kind, output, config):\n"
        "    settings = SettingsReader({}, config or {}, state=kind).settings\n"
        "    with open(os.environ['VANE_CI_PATH_PROBE'], 'a') as stream:\n"
        "        stream.write(json.dumps({'kind': kind, 'defines': settings.cmake.define}) + '\\n')\n"
        "    name = 'probe-0.0.0' + ('.tar.gz' if kind == 'sdist' else '-py3-none-any.whl')\n"
        "    Path(output, name).write_bytes(b'backend settings probe')\n"
        "    return name\n"
        "def build_sdist(output, config_settings=None):\n"
        "    return probe('sdist', output, config_settings)\n"
        "def build_wheel(output, config_settings=None, metadata_directory=None):\n"
        "    return probe('wheel', output, config_settings)\n"
    )
    receipt = tmp_path / "settings.jsonl"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VANE_CI_PATH_PROBE", str(receipt))
    monkeypatch.setenv("VANE_MEDIA_SDK", str(path_type(base) / "sdk"))
    monkeypatch.setenv("VANE_MEDIA_RUNTIME", str(path_type(base) / "runtime"))
    monkeypatch.delenv("CMAKE_ARGS", raising=False)
    monkeypatch.setattr(ci, "Path", path_type)
    monkeypatch.setattr(ci, "windows_build_settings", lambda _: [])
    monkeypatch.setattr(sys, "argv", ["ci_native_platform.py", "build-release"])
    ci.main()

    records = [json.loads(line) for line in receipt.read_text().splitlines()]
    assert [record["kind"] for record in records] == ["sdist", "wheel"]
    for record in records:
        assert record["defines"]["VANE_MEDIA_RUNTIME_SDK"] == (path_type(base) / "sdk").as_posix()
        assert record["defines"]["VANE_MEDIA_RUNTIME_DIRECTORY"] == (path_type(base) / "runtime").as_posix()
        assert record["defines"]["VANE_LOADABLE_EXTENSIONS"] == "native_media;gravitino"

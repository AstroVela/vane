# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest

from vane_packaging import copyleft_policy as policy


@pytest.mark.parametrize(
    "license_id", ["GPL-2.0-only", "GPL-3.0-or-later", "AGPL-3.0-or-later", "LGPL-2.1-only", "SSPL-1.0"]
)
def test_unreviewed_source_grants_are_rejected(license_id):
    contents = f"// SPDX-License-Identifier: {license_id}\n".encode()
    with pytest.raises(ValueError, match="source inventory needs review"):
        policy.check_source_inventory([("src/new.cpp", contents)], {})


@pytest.mark.parametrize("notice", [b"Server Side Public License", b"SERVER SIDE\nPUBLIC LICENSE"])
def test_unreviewed_full_sspl_notices_are_rejected(notice):
    with pytest.raises(ValueError, match="source inventory needs review"):
        policy.check_source_inventory([("src/new.cpp", notice)], {})


@pytest.mark.parametrize(
    "path",
    [
        "LICENSES/vendor.md",
        "LICENSES/COPYING",
        "LICENSES/vendor.json",
        "LICENSES/vendor/NOTICE.rst",
        "src/generated.hpp.in",
        "vane/vendor/protocol.proto",
        "cmake/vendor/custom-notice",
        "external/duckdb/tools/utils/test_platform.cpp",
        "external/duckdb/scripts/append_metadata.cmake",
        "THIRD_PARTY.md",
    ],
)
def test_packaged_notices_and_sources_are_scanned_without_suffix_restrictions(path):
    with pytest.raises(ValueError, match="source inventory needs review"):
        policy.check_source_inventory([(path, b"SPDX-License-Identifier: SSPL-1.0\n")], {})


def test_reviewed_dual_license_requires_identical_content():
    contents = b"// SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later\n"
    reviewed = {"src/library.cpp": hashlib.sha256(contents).hexdigest()}
    policy.check_source_inventory([("src/library.cpp", contents)], reviewed)
    with pytest.raises(ValueError, match="source inventory needs review"):
        policy.check_source_inventory([("src/library.cpp", contents.replace(b"Apache-2.0 OR ", b""))], reviewed)
    with pytest.raises(ValueError, match="source inventory needs review"):
        policy.check_source_inventory([], reviewed)


@pytest.mark.parametrize("feature", ["gpl", "nonfree", "version3", "x264", "new-codec"])
def test_ffmpeg_feature_expansion_requires_review(feature):
    dependency = {"name": "ffmpeg", "default-features": False, "features": ["avcodec", feature]}
    with pytest.raises(ValueError, match="features need license review"):
        policy.check_native_manifest({"features": {"native-audio": {"dependencies": [dependency]}}})


@pytest.mark.parametrize("dependency", ["ffmpeg", {"name": "ffmpeg"}, {"name": "ffmpeg", "default-features": True}])
def test_ffmpeg_defaults_cannot_enable_unreviewed_features(dependency):
    with pytest.raises(ValueError, match="disable default features"):
        policy.check_native_manifest({"dependencies": [dependency]})


@pytest.mark.parametrize("name", ["x264", "x265", "xvidcore"])
def test_direct_gpl_codec_additions_require_a_new_release_profile(name):
    with pytest.raises(ValueError, match="unsupported GPL codec"):
        policy.check_native_manifest({"dependencies": [name]})


def _installed_notice(share_dir, name, contents, version="1.0#2"):
    directory = share_dir / name
    directory.mkdir()
    (directory / "copyright").write_bytes(contents)
    (directory / "vcpkg.spdx.json").write_text(
        json.dumps({"packages": [{"name": name, "SPDXID": "SPDXRef-port", "versionInfo": version}]})
    )
    return {"copyright_sha256": hashlib.sha256(contents).hexdigest(), "version": version}


def test_installed_copyrights_require_a_reviewed_record(tmp_path):
    reviewed = {"mpg123": _installed_notice(tmp_path, "mpg123", b"LGPL-2.1-only\n")}
    assert policy.check_installed_notices(tmp_path, reviewed, expected=["mpg123"]) == ["mpg123"]
    with pytest.raises(ValueError, match="notice needs review"):
        policy.check_installed_notices(tmp_path, {}, expected=[])
    record = tmp_path / "mpg123" / "copyright"
    record.write_bytes(record.read_bytes() + b"Changed licensing terms.\n")
    with pytest.raises(ValueError, match="notice needs review"):
        policy.check_installed_notices(tmp_path, reviewed, expected=["mpg123"])


def test_empty_installed_dependency_tree_is_not_approval(tmp_path):
    with pytest.raises(ValueError, match="no installed dependency"):
        policy.check_installed_notices(tmp_path, {}, expected=[])


@pytest.mark.parametrize("replacement", [None, b"This package is licensed under MIT.\n"])
def test_missing_or_replaced_expected_notice_is_rejected(tmp_path, replacement):
    records = {}
    for name in ("ffmpeg", "mpg123"):
        records[name] = _installed_notice(tmp_path, name, b"LGPL-2.1-only\n")
    missing = tmp_path / "mpg123" / "copyright"
    if replacement is None:
        missing.unlink()
        message = "missing expected dependency"
    else:
        missing.write_bytes(replacement)
        message = "notice needs review"
    with pytest.raises(ValueError, match=message):
        policy.check_installed_notices(tmp_path, records, expected=records)


def test_base_profile_does_not_require_optional_notices(tmp_path):
    contents = b"Apache-2.0 OR GPL-2.0-or-later\n"
    records = {"arrow": _installed_notice(tmp_path, "arrow", contents)}
    records["ffmpeg"] = {"copyright_sha256": hashlib.sha256(contents).hexdigest(), "version": "1.0#2"}
    assert policy.check_installed_notices(tmp_path, records, expected=["arrow"]) == ["arrow"]
    with pytest.raises(ValueError, match="expected dependency notices have no review"):
        policy.check_installed_notices(tmp_path, records, expected=["new-library"])


@pytest.mark.parametrize("installed_version", ["1.1#2", "1.0#3", "1.0", None])
def test_identical_notice_does_not_approve_a_different_version(tmp_path, installed_version):
    reviewed = {"mpg123": _installed_notice(tmp_path, "mpg123", b"LGPL-2.1-only\n")}
    metadata = tmp_path / "mpg123" / "vcpkg.spdx.json"
    data = json.loads(metadata.read_text())
    data["packages"][0]["versionInfo"] = installed_version
    metadata.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="dependency version needs review"):
        policy.check_installed_notices(tmp_path, reviewed, expected=reviewed)


@pytest.mark.parametrize("metadata", [None, "not JSON", "[]", "{}", '{"packages": []}'])
def test_missing_or_invalid_installed_version_metadata_is_rejected(tmp_path, metadata):
    reviewed = {"mpg123": _installed_notice(tmp_path, "mpg123", b"LGPL-2.1-only\n")}
    path = tmp_path / "mpg123" / "vcpkg.spdx.json"
    if metadata is None:
        path.unlink()
    else:
        path.write_text(metadata)
    with pytest.raises(ValueError, match="dependency version.*needs review"):
        policy.check_installed_notices(tmp_path, reviewed, expected=reviewed)


@pytest.mark.parametrize("mismatch", ["name", "SPDXID", "duplicate"])
def test_installed_version_must_identify_one_matching_port(tmp_path, mismatch):
    reviewed = {"mpg123": _installed_notice(tmp_path, "mpg123", b"LGPL-2.1-only\n")}
    path = tmp_path / "mpg123" / "vcpkg.spdx.json"
    data = json.loads(path.read_text())
    if mismatch == "duplicate":
        data["packages"].append(dict(data["packages"][0]))
    else:
        data["packages"][0][mismatch] = "another-package"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="dependency version needs review"):
        policy.check_installed_notices(tmp_path, reviewed, expected=reviewed)


def test_selected_features_require_their_reviewed_transitive_notices():
    manifest = {
        "dependencies": [{"name": "arrow"}],
        "features": {
            "native-image": {"dependencies": ["ffmpeg"]},
            "native-audio": {"dependencies": ["ffmpeg", "libsndfile", "soxr"]},
        },
    }
    dependency_notices = {
        "arrow": ["arrow", "zstd"],
        "ffmpeg": ["ffmpeg"],
        "libsndfile": ["libsndfile", "mpg123", "mp3lame"],
        "soxr": ["soxr"],
    }
    base = {"arrow", "zstd"}
    image = base | {"ffmpeg"}
    audio = image | {"libsndfile", "mpg123", "mp3lame", "soxr"}
    assert policy.expected_installed_notices(manifest, [], dependency_notices) == base
    assert policy.expected_installed_notices(manifest, ["native-image"], dependency_notices) == image
    assert policy.expected_installed_notices(manifest, ["native-audio", "native-image"], dependency_notices) == audio
    with pytest.raises(ValueError, match="unknown native dependency feature"):
        policy.expected_installed_notices(manifest, ["native-typo"], dependency_notices)


def test_current_native_manifest_uses_the_reviewed_profile():
    root = Path(__file__).resolve().parents[2]
    policy.check_native_manifest(json.loads((root / "vcpkg.json").read_text()))

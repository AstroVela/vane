# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path

import pytest

from vane_packaging import copyleft_policy as policy


@pytest.mark.parametrize("license_id", ["GPL-2.0-only", "GPL-3.0-or-later", "AGPL-3.0-or-later", "LGPL-2.1-only"])
def test_unreviewed_source_grants_are_rejected(license_id):
    contents = f"// SPDX-License-Identifier: {license_id}\n".encode()
    with pytest.raises(ValueError, match="source inventory needs review"):
        policy.check_source_inventory([("src/new.cpp", contents)], {})


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


def test_installed_copyrights_require_a_reviewed_record(tmp_path):
    record = tmp_path / "mpg123" / "copyright"
    record.parent.mkdir()
    record.write_bytes(b"This library is distributed under LGPL-2.1-only.\n")
    reviewed = {"mpg123": {"copyright_sha256": hashlib.sha256(record.read_bytes()).hexdigest()}}
    assert policy.check_installed_notices(tmp_path, reviewed) == ["mpg123"]
    with pytest.raises(ValueError, match="notice needs review"):
        policy.check_installed_notices(tmp_path, {})
    record.write_bytes(record.read_bytes() + b"Changed licensing terms.\n")
    with pytest.raises(ValueError, match="notice needs review"):
        policy.check_installed_notices(tmp_path, reviewed)


def test_empty_installed_dependency_tree_is_not_approval(tmp_path):
    with pytest.raises(ValueError, match="no installed dependency"):
        policy.check_installed_notices(tmp_path, {})


def test_current_native_manifest_uses_the_reviewed_profile():
    import json

    root = Path(__file__).resolve().parents[2]
    policy.check_native_manifest(json.loads((root / "vcpkg.json").read_text()))

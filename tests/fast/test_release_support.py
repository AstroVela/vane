# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Release inventory checks that do not import the native package."""

import copy
import json
import runpy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALIDATE = runpy.run_path(str(ROOT / "scripts/check_release_support.py"))["validate"]
MANIFEST = json.loads((ROOT / "release-support.json").read_text())

# Policy regressions use a fixed synthetic release so publishing a new version
# or changing the real Stable selection does not change their preconditions.
FIXTURE = {
    "schema_version": 1,
    "reviewed_at": "2026-10-26",
    "latest_release": "0.2.0",
    "recommended_release": "0.2.0",
    "recommendation_scope": "evaluation",
    "stable_release": None,
    "releases": [
        {
            "version": "0.2.0",
            "tag": "v0.2.0",
            "source_commit": "1" * 40,
            "published_at": "2026-09-19T02:14:58Z",
            "release_url": "https://github.com/AstroVela/vane/releases/tag/v0.2.0",
            "pypi_url": "https://pypi.org/project/vane-ai/0.2.0/",
            "maturity": "alpha",
            "maintenance": "not_announced",
            "maintenance_until": None,
            "wheels": [
                {
                    "filename": "vane_ai-0.2.0-cp312-cp312-manylinux_2_28_x86_64.whl",
                    "python": "3.12",
                    "platform": "manylinux_2_28_x86_64",
                    "sha256": "2" * 64,
                }
            ],
            "extension_compatibility": {"rule": "exact_runtime_and_descriptor", "qualified_combinations": []},
            "upgrade_paths": [],
        }
    ],
}


class TestReleaseSupport(unittest.TestCase):
    def stable_inventory(self) -> dict:
        data = copy.deepcopy(FIXTURE)
        release = data["releases"][0]
        data["recommended_release"] = data["stable_release"] = release["version"]
        data["recommendation_scope"] = "production"
        release.update(
            maturity="stable",
            maintenance="maintained",
            maintenance_policy_url="https://example.org/maintenance",
            qualification_url="https://example.org/qualification",
        )
        return data

    def test_inventory_matches_published_asset_evidence(self) -> None:
        release = MANIFEST["releases"][0]
        evidence = {
            "tag_name": release["tag"],
            "published_at": release["published_at"],
            "draft": False,
            "prerelease": False,
            "assets": [
                {"name": wheel["filename"], "digest": f"sha256:{wheel['sha256']}"} for wheel in release["wheels"]
            ],
        }
        VALIDATE(MANIFEST, evidence)
        evidence["assets"][0]["digest"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ValueError, "digest differs"):
            VALIDATE(MANIFEST, evidence)

    def test_added_platform_cannot_reuse_another_wheels_identity(self) -> None:
        data = copy.deepcopy(FIXTURE)
        data["releases"][0]["wheels"][0]["platform"] = "win_amd64"
        with self.assertRaisesRegex(ValueError, "wheel tags disagree"):
            VALIDATE(data)

    def test_post_release_uses_its_exact_artifact_identity(self) -> None:
        data = copy.deepcopy(FIXTURE)
        release = data["releases"][0]
        previous = release["version"]
        version = "0.2.0.post1"
        release.update(
            version=version,
            tag=f"v{version}",
            release_url=f"https://github.com/AstroVela/vane/releases/tag/v{version}",
            pypi_url=f"https://pypi.org/project/vane-ai/{version}/",
        )
        for wheel in release["wheels"]:
            wheel["filename"] = wheel["filename"].replace(f"-{previous}-", f"-{version}-")
        data["latest_release"] = data["recommended_release"] = version
        VALIDATE(data)

    def test_alpha_cannot_be_promoted_by_changing_recommendation_only(self) -> None:
        data = copy.deepcopy(FIXTURE)
        data["releases"][0]["maturity"] = "alpha"
        data["recommended_release"] = data["releases"][0]["version"]
        data["recommendation_scope"] = "production"
        data["stable_release"] = data["recommended_release"]
        with self.assertRaisesRegex(ValueError, "not marked stable"):
            VALIDATE(data)

    def test_stable_requires_maintenance_and_usable_evidence_urls(self) -> None:
        VALIDATE(self.stable_inventory())
        for field, invalid, message in (
            ("maintenance", "not_announced", "maintenance commitment"),
            ("maintenance_policy_url", "", "maintenance policy URL"),
            ("qualification_url", " ", "qualification evidence URL"),
            ("qualification_url", "reviewed", "qualification evidence URL"),
        ):
            with self.subTest(field=field, invalid=invalid):
                data = self.stable_inventory()
                data["releases"][0][field] = invalid
                with self.assertRaisesRegex(ValueError, message):
                    VALIDATE(data)

    def test_expired_stable_cannot_remain_selected(self) -> None:
        data = self.stable_inventory()
        data["reviewed_at"] = "2027-02-01"
        data["releases"][0]["maintenance_until"] = "2027-01-31"
        with self.assertRaisesRegex(ValueError, "maintenance has expired"):
            VALIDATE(data)

    def test_historical_stable_can_remain_after_maintenance_ends(self) -> None:
        data = self.stable_inventory()
        data["stable_release"] = None
        data["recommendation_scope"] = "evaluation"
        data["releases"][0]["maintenance"] = "ended"
        VALIDATE(data)

    def test_upgrade_claim_needs_verification_evidence(self) -> None:
        data = copy.deepcopy(FIXTURE)
        data["releases"][0]["upgrade_paths"] = [{"from": "0.1.0", "verification_url": ""}]
        with self.assertRaisesRegex(ValueError, "require verification evidence"):
            VALIDATE(data)


if __name__ == "__main__":
    unittest.main()

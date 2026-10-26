# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate the public release inventory without importing Vane or using the network.

Use --github-release with a saved GitHub Releases API response to check the
published artifact inventory. Use --website to check the website's vendored
copy before submitting paired runtime/documentation changes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_evidence_url(value: Any, message: str) -> None:
    require(isinstance(value, str) and bool(value) and not any(char.isspace() for char in value), message)
    parsed = urlsplit(value)
    require(parsed.scheme == "https" and bool(parsed.hostname), message)


def validate(data: dict[str, Any], github_release: dict[str, Any] | None = None) -> None:
    require(data["schema_version"] == 1, "unsupported release support schema")
    reviewed = dt.date.fromisoformat(data["reviewed_at"])
    releases = data["releases"]
    require(bool(releases), "release inventory is empty")
    by_version = {release["version"]: release for release in releases}
    require(len(by_version) == len(releases), "duplicate release version")
    require(data["latest_release"] in by_version, "latest release is missing")
    require(data["recommended_release"] in by_version, "recommended release is missing")
    require(data["recommendation_scope"] in {"evaluation", "production"}, "invalid recommendation scope")
    stable = data["stable_release"]
    require(stable is None or stable in by_version, "stable release is missing")
    if data["recommendation_scope"] == "production":
        require(data["recommended_release"] == stable, "production recommendation must name a qualified stable release")

    for version, release in by_version.items():
        require(re.fullmatch(r"\d+\.\d+\.\d+(?:\.post\d+)?", version) is not None, f"invalid version: {version}")
        require(release["tag"] == f"v{version}", f"tag does not match {version}")
        require(re.fullmatch(r"[0-9a-f]{40}", release["source_commit"]) is not None, "invalid source commit")
        published = dt.datetime.fromisoformat(release["published_at"].replace("Z", "+00:00"))
        require(published.date() <= reviewed, "release is newer than review date")
        require(
            release["release_url"] == f"https://github.com/AstroVela/vane/releases/tag/v{version}", "wrong release URL"
        )
        require(release["pypi_url"] == f"https://pypi.org/project/vane-ai/{version}/", "wrong package URL")
        require(release["maturity"] in {"alpha", "beta", "stable"}, "invalid maturity")
        require(
            release["maintenance"] in {"not_announced", "best_effort", "maintained", "ended"}, "invalid maintenance"
        )
        maintenance_until = release["maintenance_until"]
        if maintenance_until is not None:
            maintenance_until = dt.date.fromisoformat(maintenance_until)
            require(maintenance_until >= published.date(), "maintenance ends before publication")
        if stable == version or release["maturity"] == "stable":
            require(release["maturity"] == "stable", "stable selection is not marked stable")
            require_evidence_url(
                release.get("maintenance_policy_url"), "stable requires an explicit maintenance policy URL"
            )
            require_evidence_url(release.get("qualification_url"), "stable requires qualification evidence URL")
        if stable == version:
            require(release["maintenance"] == "maintained", "stable requires a maintenance commitment")
            require(
                maintenance_until is None or maintenance_until >= reviewed, "selected stable maintenance has expired"
            )

        names: set[str] = set()
        for wheel in release["wheels"]:
            python_tag = wheel["python"].replace(".", "")
            expected = f"vane_ai-{version}-cp{python_tag}-cp{python_tag}-{wheel['platform']}.whl"
            require(wheel["filename"] == expected, f"wheel tags disagree: {wheel['filename']}")
            require(wheel["filename"] not in names, "duplicate wheel")
            names.add(wheel["filename"])
            require(re.fullmatch(r"[0-9a-f]{64}", wheel["sha256"]) is not None, "invalid wheel digest")
        require(bool(names), f"no published wheels recorded for {version}")
        extensions = release["extension_compatibility"]
        require(extensions["rule"] == "exact_runtime_and_descriptor", "extension compatibility rule changed")
        for combination in extensions["qualified_combinations"]:
            require(bool(combination["provider_wheel"]), "qualified extension needs an immutable wheel filename")
            require(re.fullmatch(r"[0-9a-f]{64}", combination["sha256"]) is not None, "invalid provider digest")
            require_evidence_url(
                combination["qualification_url"], "qualified extension needs verification evidence URL"
            )
        for upgrade in release["upgrade_paths"]:
            require(upgrade["from"] != version, "self-upgrade is not an upgrade path")
            require_evidence_url(upgrade["verification_url"], "upgrade paths require verification evidence URL")

    if github_release is not None:
        version = github_release["tag_name"].removeprefix("v")
        require(version in by_version, "GitHub release is absent from inventory")
        release = by_version[version]
        require(
            not github_release["draft"] and not github_release["prerelease"],
            "inventory must describe a published release",
        )
        require(release["published_at"] == github_release["published_at"], "publication date differs")
        assets = {asset["name"]: asset for asset in github_release["assets"] if asset["name"].endswith(".whl")}
        require(set(assets) == {wheel["filename"] for wheel in release["wheels"]}, "published wheel inventory differs")
        for wheel in release["wheels"]:
            require(
                assets[wheel["filename"]]["digest"] == f"sha256:{wheel['sha256']}", "published wheel digest differs"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "release-support.json")
    parser.add_argument("--github-release", type=Path, help="saved GitHub Releases API JSON")
    parser.add_argument("--website", type=Path, help="website checkout with src/data/release-support.json")
    args = parser.parse_args()
    try:
        data = json.loads(args.manifest.read_text())
        upstream = json.loads(args.github_release.read_text()) if args.github_release else None
        validate(data, upstream)
        if args.website:
            website_data = json.loads((args.website / "src/data/release-support.json").read_text())
            require(website_data == data, "website release support copy differs from the canonical inventory")
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"release support validation failed: {error}\n")
    print(f"Release support inventory valid: {len(data['releases'])} release(s)")


if __name__ == "__main__":
    main()

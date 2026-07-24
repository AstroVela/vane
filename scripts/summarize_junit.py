# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestTotals:
    tests: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    seconds: float = 0.0

    def add_suite(self, suite: ET.Element) -> None:
        self.tests += int(suite.get("tests", "0"))
        self.failures += int(suite.get("failures", "0"))
        self.errors += int(suite.get("errors", "0"))
        self.skipped += int(suite.get("skipped", "0"))
        self.seconds += float(suite.get("time", "0"))

    @property
    def passed(self) -> int:
        return self.tests - self.failures - self.errors - self.skipped


def _report_paths(inputs: list[Path]) -> list[Path]:
    reports: set[Path] = set()
    for input_path in inputs:
        if input_path.is_dir():
            reports.update(input_path.glob("*.xml"))
        elif input_path.is_file():
            reports.add(input_path)
    return sorted(reports)


def _top_level_suites(root: ET.Element) -> list[ET.Element]:
    if root.tag == "testsuite":
        return [root]
    return [child for child in root if child.tag == "testsuite"]


def summarize(inputs: list[Path]) -> tuple[TestTotals, list[Path]]:
    totals = TestTotals()
    reports = _report_paths(inputs)
    for report in reports:
        root = ET.parse(report).getroot()
        for suite in _top_level_suites(root):
            totals.add_suite(suite)
    return totals, reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Render JUnit totals as a GitHub Markdown summary.")
    parser.add_argument("--label", default="Fast tests", help="Summary row label")
    parser.add_argument("paths", nargs="+", type=Path, help="JUnit XML files or directories")
    args = parser.parse_args()

    try:
        totals, reports = summarize(args.paths)
    except (ET.ParseError, OSError, ValueError) as exc:
        print(f"Could not summarize JUnit reports: {exc}", file=sys.stderr)
        return 1

    print("### Fast-test results")
    if not reports:
        print()
        print(f"No JUnit reports were produced for `{args.label}`.")
        return 0

    print()
    print("| Shard | Reports | Tests | Passed | Failed | Errors | Skipped | Time |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    print(
        f"| {args.label} | {len(reports)} | {totals.tests} | {totals.passed} | "
        f"{totals.failures} | {totals.errors} | {totals.skipped} | {totals.seconds:.1f}s |"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

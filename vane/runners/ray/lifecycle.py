# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from os import PathLike, fspath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

BasePath = str | PathLike[str]


@dataclass(frozen=True)
class CopyDirectWriteLifecycleScan:
    base_path: str
    scanned_runs: int
    cleaned_runs: int
    committed_runs: int
    active_runs: int
    catalog_commit_pending_runs: int
    skipped_unregistered_runs: int
    errors: int
    cleaned_run_ids: list[str]
    error_messages: list[str]

    @classmethod
    def from_api_result(cls, base_path: str, result: Any) -> CopyDirectWriteLifecycleScan:
        return cls(
            base_path=str(base_path),
            scanned_runs=int(result.get("scanned_runs", 0)),
            cleaned_runs=int(result.get("cleaned_runs", 0)),
            committed_runs=int(result.get("committed_runs", 0)),
            active_runs=int(result.get("active_runs", 0)),
            catalog_commit_pending_runs=int(result["catalog_commit_pending_runs"]),
            skipped_unregistered_runs=int(result.get("skipped_unregistered_runs", 0)),
            errors=int(result.get("errors", 0)),
            cleaned_run_ids=[str(run_id) for run_id in result.get("cleaned_run_ids", [])],
            error_messages=[str(message) for message in result.get("error_messages", [])],
        )

    @classmethod
    def from_exception(cls, base_path: str, exc: BaseException) -> CopyDirectWriteLifecycleScan:
        return cls(
            base_path=str(base_path),
            scanned_runs=0,
            cleaned_runs=0,
            committed_runs=0,
            active_runs=0,
            catalog_commit_pending_runs=0,
            skipped_unregistered_runs=0,
            errors=1,
            cleaned_run_ids=[],
            error_messages=[str(exc)],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_path": self.base_path,
            "scanned_runs": self.scanned_runs,
            "cleaned_runs": self.cleaned_runs,
            "committed_runs": self.committed_runs,
            "active_runs": self.active_runs,
            "catalog_commit_pending_runs": self.catalog_commit_pending_runs,
            "skipped_unregistered_runs": self.skipped_unregistered_runs,
            "errors": self.errors,
            "cleaned_run_ids": list(self.cleaned_run_ids),
            "error_messages": list(self.error_messages),
        }


def _normalize_base_paths(base_paths: BasePath | Iterable[BasePath]) -> list[str]:
    if isinstance(base_paths, (str, PathLike)):
        paths = [fspath(base_paths)]
    else:
        paths = [fspath(path) for path in base_paths]
    paths = [path for path in paths if path]
    if not paths:
        raise ValueError("at least one direct-write COPY base path is required")
    return paths


def _aggregate_scans(scans: Sequence[CopyDirectWriteLifecycleScan]) -> dict[str, Any]:
    cleaned_runs: list[dict[str, str]] = []
    error_messages: list[str] = []
    for scan in scans:
        cleaned_runs.extend({"base_path": scan.base_path, "run_id": run_id} for run_id in scan.cleaned_run_ids)
        error_messages.extend(f"{scan.base_path}: {message}" for message in scan.error_messages)

    return {
        "base_path_count": len(scans),
        "scanned_runs": sum(scan.scanned_runs for scan in scans),
        "cleaned_runs": sum(scan.cleaned_runs for scan in scans),
        "committed_runs": sum(scan.committed_runs for scan in scans),
        "active_runs": sum(scan.active_runs for scan in scans),
        "catalog_commit_pending_runs": sum(scan.catalog_commit_pending_runs for scan in scans),
        "skipped_unregistered_runs": sum(scan.skipped_unregistered_runs for scan in scans),
        "errors": sum(scan.errors for scan in scans),
        "cleaned_run_ids": cleaned_runs,
        "error_messages": error_messages,
        "scans": [scan.to_dict() for scan in scans],
    }


def cleanup_copy_direct_write_lifecycle_once(
    base_paths: BasePath | Iterable[BasePath],
    *,
    min_age_ms: int,
    now_epoch_ms: int = 0,
    fail_fast: bool = False,
    conn: Any | None = None,
) -> dict[str, Any]:
    """Run one direct-write COPY lifecycle cleanup scan.

    This is the one-shot entry point for stale uncommitted direct-write COPY
    results. It delegates correctness decisions to the C++ manifest/marker
    aware scanner: committed runs and catalog-commit-pending extension runs are
    skipped, active runs are kept, and only lifecycle-registered stale
    uncommitted runs are removed. Pass the DuckDB connection that owns any
    connection-scoped filesystem or secret config.

    The caller must ensure that no COPY is running for any supplied base path.
    Elapsed time does not prove that an uncommitted run is abandoned.
    """
    import vane

    paths = _normalize_base_paths(base_paths)
    scans: list[CopyDirectWriteLifecycleScan] = []
    for base_path in paths:
        try:
            raw = vane.ray_cxx.cleanup_expired_copy_direct_write_runs(
                base_path,
                min_age_ms=int(min_age_ms),
                now_epoch_ms=int(now_epoch_ms),
                conn=conn,
            )
            scans.append(CopyDirectWriteLifecycleScan.from_api_result(base_path, raw))
        except Exception as exc:
            if fail_fast:
                raise
            scans.append(CopyDirectWriteLifecycleScan.from_exception(base_path, exc))
    return _aggregate_scans(scans)


def _format_summary(summary: dict[str, Any]) -> str:
    return (
        "direct-write lifecycle cleanup: "
        f"base_paths={summary['base_path_count']} "
        f"scanned={summary['scanned_runs']} "
        f"cleaned={summary['cleaned_runs']} "
        f"committed={summary['committed_runs']} "
        f"active={summary['active_runs']} "
        f"catalog_commit_pending={summary['catalog_commit_pending_runs']} "
        f"errors={summary['errors']}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Cleanup stale uncommitted Vane direct-write COPY runs. "
            "Run only while COPY is quiesced for every supplied base path."
        ),
    )
    parser.add_argument(
        "--base-path",
        action="append",
        required=True,
        help="Distributed COPY output base path. Repeat for multiple outputs.",
    )
    parser.add_argument(
        "--min-age-ms",
        type=int,
        default=24 * 60 * 60 * 1000,
        help="Minimum uncommitted run age before cleanup. Defaults to 24h.",
    )
    parser.add_argument(
        "--now-epoch-ms",
        type=int,
        default=0,
        help="Override the current epoch time for deterministic maintenance runs.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    summary = cleanup_copy_direct_write_lifecycle_once(
        args.base_path,
        min_age_ms=args.min_age_ms,
        now_epoch_ms=args.now_epoch_ms,
        fail_fast=args.fail_fast,
    )
    print(json.dumps(summary, sort_keys=True) if args.as_json else _format_summary(summary))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

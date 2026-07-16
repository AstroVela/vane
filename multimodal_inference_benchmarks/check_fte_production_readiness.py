from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

DEFAULT_SUITES = ("image", "document", "audio", "video")
@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    status: str
    message: str
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "status": self.status,
            "message": self.message,
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


def _check(name: str, status: str, message: str, **details: object) -> ReadinessCheck:
    return ReadinessCheck(name, status, message, details)


def _positive_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _check_fte_switches() -> list[ReadinessCheck]:
    checks: list[ReadinessCheck] = []
    checks.append(
        _check(
            "fte.enabled",
            "PASS",
            "FTE is mandatory for Ray distributed execution",
        )
    )
    return checks


def _check_admission(env: Mapping[str, str]) -> list[ReadinessCheck]:
    checks: list[ReadinessCheck] = []
    checks.append(
        _check(
            "admission.capacity_source",
            "PASS",
            "worker admission is derived from registered leases and runtime capacity",
        )
    )

    for key in (
        "VANE_FTE_SPLIT_QUEUE_MAX_BUFFERED_SPLITS",
        "VANE_FTE_TASK_UPDATE_MAX_SPLITS",
        "VANE_FTE_TASK_UPDATE_MAX_PAYLOAD_BYTES",
    ):
        value = env.get(key)
        parsed = _positive_int(value)
        checks.append(
            _check(
                f"backpressure.{key.lower()}",
                "PASS" if parsed is not None else "WARN",
                (
                    f"{key} has a positive cap"
                    if parsed is not None
                    else f"{key} is unset or non-positive; use an explicit cap for production"
                ),
                value=value,
            )
        )

    return checks


def _check_retry_and_reservation(env: Mapping[str, str]) -> list[ReadinessCheck]:
    checks: list[ReadinessCheck] = []
    for key, default in (
        ("VANE_FTE_RETRY_INITIAL_DELAY_S", "10"),
        ("VANE_FTE_RETRY_MAX_DELAY_S", "60"),
        ("VANE_FTE_RETRY_DELAY_SCALE_FACTOR", "2.0"),
    ):
        value = env.get(key, default)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = -1.0
        checks.append(
            _check(
                f"retry.{key.lower()}",
                "PASS" if parsed > 0 else "FAIL",
                f"{key} effective value must be positive",
                value=value,
                defaulted=key not in env,
            )
        )
    return checks


def _storage_paths(raw: str | None) -> list[str]:
    if raw is None or raw.strip() == "":
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _is_object_uri(path: str) -> bool:
    return "://" in path and not path.startswith("file://")


def _check_shuffle_storage(
    env: Mapping[str, str],
    *,
    min_free_bytes: int,
) -> list[ReadinessCheck]:
    raw = env.get("VANE_SHUFFLE_LOCAL_DIRS") or env.get("DUCKDB_SHUFFLE_DIRS")
    paths = _storage_paths(raw)
    if not paths:
        return [
            _check(
                "shuffle.path",
                "WARN",
                "VANE_SHUFFLE_LOCAL_DIRS/DUCKDB_SHUFFLE_DIRS is unset; verify durable exchange prefix manually",
            )
        ]
    checks: list[ReadinessCheck] = []
    for index, path_text in enumerate(paths):
        name = f"shuffle.path.{index}"
        if _is_object_uri(path_text):
            checks.append(
                _check(
                    name,
                    "WARN",
                    "object-store shuffle prefix cannot be fully verified locally; check credentials, lifecycle and fault tests",
                    path=path_text,
                )
            )
            continue
        path = Path(path_text)
        probe = path if path.exists() else path.parent
        if not probe.exists():
            checks.append(_check(name, "FAIL", "shuffle path parent does not exist", path=path_text))
            continue
        if not os.access(probe, os.W_OK):
            checks.append(_check(name, "FAIL", "shuffle path is not writable", path=path_text))
            continue
        free_bytes = shutil.disk_usage(probe).free
        status = "PASS" if free_bytes >= min_free_bytes else "WARN"
        checks.append(
            _check(
                name,
                status,
                (
                    "shuffle path is writable and has enough free space"
                    if status == "PASS"
                    else "shuffle path is writable but free space is below readiness threshold"
                ),
                path=path_text,
                free_bytes=free_bytes,
                min_free_bytes=min_free_bytes,
            )
        )
    return checks


def _read_manifest_events(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    events: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        events.append(json.loads(line))
    return events


def _check_full_matrix_manifest(
    manifest_path: Path | None,
    *,
    suites: Sequence[str],
    require_full_matrix: bool,
) -> list[ReadinessCheck]:
    if manifest_path is None:
        return [
            _check(
                "benchmark.full_matrix",
                "FAIL" if require_full_matrix else "WARN",
                "no full matrix manifest path was provided",
            )
        ]
    events = _read_manifest_events(manifest_path)
    latest: dict[str, dict[str, object]] = {}
    for event in events:
        if event.get("event") == "comparison_success" and event.get("suite"):
            latest[str(event["suite"])] = event
    missing = [suite for suite in suites if suite not in latest]
    if missing:
        return [
            _check(
                "benchmark.full_matrix",
                "FAIL" if require_full_matrix else "WARN",
                "full matrix manifest is missing comparison_success events",
                manifest_path=str(manifest_path),
                missing_suites=missing,
            )
        ]
    not_full = [suite for suite in suites if latest[suite].get("full") is False]
    if not_full and require_full_matrix:
        return [
            _check(
                "benchmark.full_matrix",
                "FAIL",
                "manifest contains smoke comparison events where full results are required",
                manifest_path=str(manifest_path),
                smoke_suites=not_full,
            )
        ]
    return [
        _check(
            "benchmark.full_matrix",
            "PASS",
            "full matrix manifest contains reference-vs-chaos comparison_success for all suites",
            manifest_path=str(manifest_path),
            suites=list(suites),
        )
    ]


def evaluate_readiness(
    *,
    env: Mapping[str, str] | None = None,
    manifest_path: Path | None = None,
    require_full_matrix: bool = False,
    suites: Sequence[str] = DEFAULT_SUITES,
    min_shuffle_free_bytes: int = 10 * 1024 * 1024 * 1024,
) -> list[ReadinessCheck]:
    env = dict(os.environ if env is None else env)
    checks: list[ReadinessCheck] = []
    checks.extend(_check_fte_switches())
    checks.extend(_check_admission(env))
    checks.extend(_check_retry_and_reservation(env))
    checks.extend(_check_shuffle_storage(env, min_free_bytes=min_shuffle_free_bytes))
    checks.extend(
        _check_full_matrix_manifest(
            manifest_path,
            suites=suites,
            require_full_matrix=require_full_matrix,
        )
    )
    return checks


def overall_status(checks: Sequence[ReadinessCheck]) -> str:
    statuses = {check.status for check in checks}
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def _print_text(checks: Sequence[ReadinessCheck]) -> None:
    for check in checks:
        print(f"[{check.status}] {check.name}: {check.message}")
        for key, value in sorted(check.details.items()):
            print(f"  {key}: {value}")
    print(f"overall: {overall_status(checks)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether the current environment is ready for Trino-style Vane FTE production runs."
    )
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument(
        "--require-full-matrix",
        action="store_true",
        help="Require image/document/audio/video comparison_success events in the manifest.",
    )
    parser.add_argument(
        "--suite",
        action="append",
        choices=(*DEFAULT_SUITES, "all"),
        default=None,
        help="Suites required in the full matrix manifest. Default: all.",
    )
    parser.add_argument(
        "--min-shuffle-free-bytes",
        type=int,
        default=10 * 1024 * 1024 * 1024,
        help="Minimum local free bytes expected for each shuffle path.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when any warning is present. Failures always return non-zero.",
    )
    return parser.parse_args()


def _requested_suites(values: list[str] | None) -> tuple[str, ...]:
    requested = values or ["all"]
    if "all" in requested:
        return DEFAULT_SUITES
    suites: list[str] = []
    for suite in requested:
        if suite not in suites:
            suites.append(suite)
    return tuple(suites)


def main() -> None:
    args = _parse_args()
    checks = evaluate_readiness(
        manifest_path=args.manifest_path,
        require_full_matrix=args.require_full_matrix,
        suites=_requested_suites(args.suite),
        min_shuffle_free_bytes=max(0, int(args.min_shuffle_free_bytes)),
    )
    status = overall_status(checks)
    if args.json:
        print(
            json.dumps(
                {
                    "overall": status,
                    "checks": [check.to_dict() for check in checks],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        _print_text(checks)
    if status == "FAIL":
        raise SystemExit(2)
    if args.strict and status == "WARN":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

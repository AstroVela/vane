from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multimodal_inference_benchmarks import check_fte_production_readiness as readiness
from multimodal_inference_benchmarks import run_fte_full_matrix as matrix_runner

DEFAULT_FAST_TESTS = (
    "tests/fast/test_fte_chaos_benchmark_runner.py",
    "tests/fast/test_fte_production_readiness.py",
    "tests/fast/test_ray_fte.py",
    "tests/fast/test_ray_fte_event_scheduler.py",
    "tests/fast/test_ray_fragment_submission.py",
    "tests/fast/test_ray_result_contract.py",
    "tests/fast/test_ray_fte_fault_injection.py",
)

FAST_TEST_ENV_PREFIXES_TO_CLEAR = (
    "VANE_FTE_",
    "VANE_RAY_",
    "VANE_EXCHANGE_",
    "VANE_SHUFFLE_",
    "VANE_DISTRIBUTED_",
    "VANE_PLAN_",
)
FAST_TEST_ENV_KEYS_TO_CLEAR = {
    "DUCKDB_SHUFFLE_DIRS",
    "NUM_GPU_NODES",
    "INPUT_PATH",
    "OUTPUT_PATH",
    "LOCAL_IMAGE_ROOT",
    "LOCAL_PDF_ROOT",
}


@dataclass
class GateStep:
    name: str
    status: str
    elapsed_s: float = 0.0
    command: list[str] | None = None
    log_path: str | None = None
    returncode: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "elapsed_s": round(float(self.elapsed_s), 3),
        }
        if self.command is not None:
            payload["command"] = list(self.command)
        if self.log_path is not None:
            payload["log_path"] = self.log_path
        if self.returncode is not None:
            payload["returncode"] = self.returncode
        if self.details:
            payload["details"] = dict(self.details)
        return payload


def _default_run_root() -> Path:
    return Path("/tmp") / time.strftime("vane_fte_release_gate_%Y%m%d_%H%M%S")


def _overall_status(steps: Sequence[GateStep]) -> str:
    statuses = {step.status for step in steps}
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def _write_reports(run_root: Path, steps: Sequence[GateStep]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    overall = _overall_status(steps)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overall": overall,
        "run_root": str(run_root),
        "steps": [step.to_dict() for step in steps],
    }
    (run_root / "release_gate_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Vane FTE Release Gate Report",
        "",
        f"- generated_at: {payload['created_at']}",
        f"- run_root: `{run_root}`",
        f"- overall: `{overall}`",
        "",
        "| Step | Status | Elapsed | Log |",
        "| --- | --- | ---: | --- |",
    ]
    for step in steps:
        log = "-" if step.log_path is None else f"`{step.log_path}`"
        lines.append(f"| {step.name} | {step.status} | {step.elapsed_s:.1f}s | {log} |")
    lines.append("")
    lines.append("## Details")
    for step in steps:
        lines.append("")
        lines.append(f"### {step.name}")
        lines.append("")
        lines.append(f"- status: `{step.status}`")
        if step.command is not None:
            lines.append(f"- command: `{' '.join(step.command)}`")
        if step.returncode is not None:
            lines.append(f"- returncode: `{step.returncode}`")
        if step.details:
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(step.details, indent=2, sort_keys=True))
            lines.append("```")
    (run_root / "release_gate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_command(
    *,
    name: str,
    command: list[str],
    log_path: Path,
    timeout_s: int | None,
    dry_run: bool,
    env: dict[str, str] | None = None,
) -> GateStep:
    if dry_run:
        return GateStep(
            name=name,
            status="SKIPPED",
            command=command,
            log_path=str(log_path),
            details={"reason": "dry_run"},
        )

    start = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=os.environ.copy() if env is None else env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
            )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - start
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[fte-release-gate] command timed out after {exc.timeout}s\n")
        return GateStep(
            name=name,
            status="FAIL",
            elapsed_s=elapsed,
            command=command,
            log_path=str(log_path),
            details={"timeout_s": exc.timeout},
        )

    elapsed = time.time() - start
    status = "PASS" if result.returncode == 0 else "FAIL"
    return GateStep(
        name=name,
        status=status,
        elapsed_s=elapsed,
        command=command,
        log_path=str(log_path),
        returncode=int(result.returncode),
    )


def _readiness_step(
    *,
    name: str,
    manifest_path: Path | None,
    require_full_matrix: bool,
    strict: bool,
    dry_run: bool,
) -> GateStep:
    if dry_run:
        return GateStep(
            name=name,
            status="SKIPPED",
            details={"reason": "dry_run"},
        )
    start = time.time()
    checks = readiness.evaluate_readiness(
        manifest_path=manifest_path,
        require_full_matrix=require_full_matrix,
        min_shuffle_free_bytes=0,
    )
    if manifest_path is None and not require_full_matrix:
        checks = [check for check in checks if check.name != "benchmark.full_matrix"]
    readiness_status = readiness.overall_status(checks)
    if readiness_status == "PASS":
        status = "PASS"
    elif strict:
        status = "FAIL"
    else:
        status = "WARN"
    return GateStep(
        name=name,
        status=status,
        elapsed_s=time.time() - start,
        details={
            "readiness_overall": readiness_status,
            "strict": strict,
            "checks": [check.to_dict() for check in checks],
        },
    )


def _manifest_step(
    *,
    manifest_path: Path,
    suites: Sequence[str],
    require_full_matrix: bool,
    dry_run: bool,
) -> GateStep:
    if dry_run:
        return GateStep(
            name="matrix-manifest",
            status="SKIPPED",
            details={"reason": "dry_run", "manifest_path": str(manifest_path)},
        )
    start = time.time()
    checks = readiness._check_full_matrix_manifest(
        manifest_path,
        suites=suites,
        require_full_matrix=require_full_matrix,
    )
    status = readiness.overall_status(checks)
    return GateStep(
        name="matrix-manifest",
        status=status,
        elapsed_s=time.time() - start,
        details={"checks": [check.to_dict() for check in checks]},
    )


def _fast_test_command(python: str, tests: Sequence[str]) -> list[str]:
    return [python, "-m", "pytest", "-q", *tests]


def _fast_test_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key in FAST_TEST_ENV_KEYS_TO_CLEAR or any(
            key.startswith(prefix) for prefix in FAST_TEST_ENV_PREFIXES_TO_CLEAR
        ):
            env.pop(key, None)
    return env


def _matrix_command(
    *,
    python: str,
    matrix_run_root: Path,
    smoke_limit: int,
    timeout_s: int,
    suites: Sequence[str],
    resume: bool,
    dry_run: bool,
) -> list[str]:
    command = [
        python,
        "multimodal_inference_benchmarks/run_fte_full_matrix.py",
        "--run-root",
        str(matrix_run_root),
        "--timeout-s",
        str(timeout_s),
    ]
    for suite in suites:
        command.extend(["--suite", suite])
    if smoke_limit > 0:
        command.extend(["--smoke-limit", str(smoke_limit)])
    if resume:
        command.append("--resume")
    if dry_run:
        command.append("--dry-run")
    return command


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Vane FTE release gate: readiness, fast regression, and benchmark matrix."
    )
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-readiness", action="store_true")
    parser.add_argument("--strict-readiness", action="store_true")
    parser.add_argument("--skip-fast-tests", action="store_true")
    parser.add_argument("--skip-matrix", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--suite",
        action="append",
        choices=(*matrix_runner.DEFAULT_SUITES, "all"),
        default=None,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the matrix in smoke mode. This is the default unless --full is set.",
    )
    parser.add_argument("--full", action="store_true", help="Run full benchmark matrix.")
    parser.add_argument("--smoke-limit", type=int, default=4)
    parser.add_argument("--matrix-timeout-s", type=int, default=900)
    parser.add_argument("--fast-timeout-s", type=int, default=900)
    parser.add_argument(
        "--fast-test",
        action="append",
        default=None,
        help="Override fast regression test list. Repeat for multiple paths.",
    )
    return parser.parse_args()


def _requested_suites(values: list[str] | None) -> list[str]:
    requested = values or ["all"]
    if "all" in requested:
        return list(matrix_runner.DEFAULT_SUITES)
    suites: list[str] = []
    for suite in requested:
        if suite not in suites:
            suites.append(suite)
    return suites


def main() -> None:
    args = _parse_args()
    run_root = args.run_root or _default_run_root()
    logs_dir = run_root / "logs"
    matrix_run_root = run_root / "full_matrix"
    matrix_manifest_path = matrix_run_root / "manifest.jsonl"
    suites = _requested_suites(args.suite)
    smoke_limit = 0 if args.full else max(1, int(args.smoke_limit))
    require_full_matrix = bool(args.full)
    fast_tests = tuple(args.fast_test or DEFAULT_FAST_TESTS)
    steps: list[GateStep] = []

    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[fte-release-gate] run_root={run_root}", flush=True)
    print(f"[fte-release-gate] suites={','.join(suites)}", flush=True)
    print(f"[fte-release-gate] smoke_limit={smoke_limit} full={args.full}", flush=True)

    try:
        if not args.skip_readiness:
            step = _readiness_step(
                name="readiness-preflight",
                manifest_path=None,
                require_full_matrix=False,
                strict=bool(args.strict_readiness),
                dry_run=bool(args.dry_run),
            )
            steps.append(step)
            if step.status == "FAIL":
                raise RuntimeError("readiness preflight failed")

        if not args.skip_fast_tests:
            step = _run_command(
                name="fast-regression",
                command=_fast_test_command(args.python, fast_tests),
                log_path=logs_dir / "fast-regression.log",
                timeout_s=max(1, int(args.fast_timeout_s)),
                dry_run=bool(args.dry_run),
                env=_fast_test_env(),
            )
            steps.append(step)
            if step.status == "FAIL":
                raise RuntimeError("fast regression failed")

        if not args.skip_matrix:
            step = _run_command(
                name="full-matrix-smoke" if smoke_limit > 0 else "full-matrix",
                command=_matrix_command(
                    python=args.python,
                    matrix_run_root=matrix_run_root,
                    smoke_limit=smoke_limit,
                    timeout_s=max(1, int(args.matrix_timeout_s)),
                    suites=suites,
                    resume=bool(args.resume),
                    dry_run=bool(args.dry_run),
                ),
                log_path=logs_dir / "full-matrix.log",
                timeout_s=None,
                dry_run=bool(args.dry_run),
                env=os.environ.copy(),
            )
            steps.append(step)
            if step.status == "FAIL":
                raise RuntimeError("full matrix step failed")
            steps.append(
                _manifest_step(
                    manifest_path=matrix_manifest_path,
                    suites=suites,
                    require_full_matrix=require_full_matrix,
                    dry_run=bool(args.dry_run),
                )
            )
            if steps[-1].status == "FAIL":
                raise RuntimeError("matrix manifest validation failed")

        if not args.skip_readiness and not args.skip_matrix:
            step = _readiness_step(
                name="readiness-post-matrix",
                manifest_path=matrix_manifest_path,
                require_full_matrix=require_full_matrix,
                strict=bool(args.strict_readiness),
                dry_run=bool(args.dry_run),
            )
            steps.append(step)
            if step.status == "FAIL":
                raise RuntimeError("post-matrix readiness failed")
    finally:
        _write_reports(run_root, steps)
        print(f"[fte-release-gate] wrote {run_root / 'release_gate_report.md'}", flush=True)
        print(f"[fte-release-gate] overall={_overall_status(steps)}", flush=True)

    if _overall_status(steps) == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

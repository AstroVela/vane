from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multimodal_inference_benchmarks import run_fte_chaos_benchmarks as runner

DEFAULT_SUITES = ("image", "document", "audio", "video")
DEFAULT_TIMEOUTS = {
    "image": 3600,
    "document": 3600,
    "audio": 7200,
    "video": 21600,
}
DEFAULT_CLEANUP_REFERENCE_SUITES = frozenset({"video"})


def _default_run_root() -> Path:
    return Path("/tmp") / time.strftime("vane_fte_full_matrix_%Y%m%d_%H%M%S")


def _default_audio_num_gpu_nodes() -> int:
    for key in ("AUDIO_NUM_GPU_NODES", "NUM_GPU_NODES"):
        raw = os.getenv(key)
        if raw is None or raw.strip() == "":
            continue
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 1
    return 1


def _parse_suite_timeout(raw: str) -> tuple[str, int]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("suite timeout must use suite=seconds")
    suite, value = raw.split("=", 1)
    suite = suite.strip()
    if suite not in DEFAULT_TIMEOUTS:
        raise argparse.ArgumentTypeError(f"unknown suite for timeout: {suite}")
    try:
        timeout_s = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"invalid timeout seconds: {value}") from exc
    if timeout_s <= 0:
        raise argparse.ArgumentTypeError("timeout seconds must be positive")
    return suite, timeout_s


def _requested_suites(values: list[str] | None) -> list[str]:
    requested = values or ["all"]
    if "all" in requested:
        return list(DEFAULT_SUITES)
    suites: list[str] = []
    for suite in requested:
        if suite not in suites:
            suites.append(suite)
    return suites


def _timeout_map(common_timeout_s: int | None, overrides: list[tuple[str, int]]) -> dict[str, int]:
    result = {
        suite: int(common_timeout_s) if common_timeout_s is not None else default
        for suite, default in DEFAULT_TIMEOUTS.items()
    }
    result.update(dict(overrides))
    return result


def _read_manifest_events(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        events.append(json.loads(line))
    return events


def _latest_events(
    events: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    comparisons: dict[str, dict[str, Any]] = {}
    label_events: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        suite = str(event.get("suite") or "")
        event_name = str(event.get("event") or "")
        if event_name in {"comparison_success", "comparison_failure"} and suite:
            comparisons[suite] = event
        label = str(event.get("label") or "")
        if event_name in {"success", "resume_success"} and suite and label:
            label_events[(suite, label)] = event
    return comparisons, label_events


def _summary_cell(event: dict[str, Any] | None, summary: dict[str, Any] | None) -> str:
    if summary is None:
        return "missing"
    elapsed = event.get("elapsed_s") if event else None
    elapsed_text = "-" if elapsed is None else f"{float(elapsed):.1f}s"
    return f"{elapsed_text}, {summary.get('row_count')} rows, {summary.get('column_count')} cols"


def _checksum(summary: dict[str, Any] | None) -> str:
    if summary is None:
        return "missing"
    return f"({summary.get('hash_sum')}, {summary.get('hash_xor')})"


def write_report(
    *,
    report_path: Path,
    run_root: Path,
    manifest_path: Path,
    suites: list[str],
    cleanup_reference_suites: set[str],
) -> None:
    events = _read_manifest_events(manifest_path)
    comparisons, label_events = _latest_events(events)
    lines: list[str] = [
        "# Vane FTE Full Benchmark Matrix Report",
        "",
        f"- generated_at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"- run_root: `{run_root}`",
        f"- manifest_path: `{manifest_path}`",
        f"- cleanup_reference_suites: `{','.join(sorted(cleanup_reference_suites)) or '-'}`",
        "",
        "| Suite | Status | Reference | Chaos | Checksum |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for suite in suites:
        comparison = comparisons.get(suite)
        reference = None if comparison is None else comparison.get("reference")
        chaos = None if comparison is None else comparison.get("chaos")
        if comparison is None:
            status = "missing"
        elif comparison.get("event") == "comparison_success":
            status = "passed"
        else:
            status = "failed"
        lines.append(
            "| {suite} | {status} | {reference} | {chaos} | `{checksum}` |".format(
                suite=suite,
                status=status,
                reference=_summary_cell(label_events.get((suite, "reference")), reference),
                chaos=_summary_cell(label_events.get((suite, "chaos")), chaos),
                checksum=_checksum(chaos),
            )
        )
    lines.append("")
    lines.append("## Schemas")
    for suite in suites:
        comparison = comparisons.get(suite)
        summary = None if comparison is None else comparison.get("chaos")
        schema = [] if summary is None else summary.get("schema") or []
        lines.append("")
        lines.append(f"### {suite}")
        if not schema:
            lines.append("")
            lines.append("missing")
            continue
        lines.append("")
        lines.append("```text")
        lines.extend(f"{column.get('name')} {column.get('type')}" for column in schema)
        lines.append("```")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full Vane FTE multimodal reference-vs-chaos benchmark matrix."
    )
    parser.add_argument(
        "--suite",
        action="append",
        choices=(*DEFAULT_SUITES, "all"),
        default=None,
        help="Suite to run. Repeat for multiple suites. Default: all.",
    )
    parser.add_argument("--mode", choices=("worker", "host"), default="host")
    parser.add_argument("--data-root", type=Path, default=runner.DEFAULT_DATA_ROOT)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke-limit",
        type=int,
        default=0,
        help=(
            "Run a smoke matrix by passing this input limit to every suite. Default 0 keeps full benchmark behavior."
        ),
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=None,
        help="Common timeout for every suite. Per-suite defaults are used when omitted.",
    )
    parser.add_argument(
        "--suite-timeout",
        action="append",
        type=_parse_suite_timeout,
        default=[],
        help="Override one suite timeout, e.g. --suite-timeout video=21600.",
    )
    parser.add_argument(
        "--audio-num-gpu-nodes",
        type=int,
        default=_default_audio_num_gpu_nodes(),
        help="Set AUDIO_NUM_GPU_NODES and NUM_GPU_NODES for the audio suite. Use 0 to leave env unchanged.",
    )
    parser.add_argument(
        "--keep-reference-output",
        action="store_true",
        help="Keep every reference_output directory. By default video reference_output is cleaned before chaos.",
    )
    parser.add_argument(
        "--cleanup-reference-suite",
        action="append",
        choices=DEFAULT_SUITES,
        default=None,
        help="Suite whose reference_output should be removed before chaos. Default: video.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_root = args.run_root or _default_run_root()
    manifest_path = args.manifest_path or (run_root / "manifest.jsonl")
    report_path = args.report_path or (run_root / "full_matrix_report.md")
    suites = _requested_suites(args.suite)
    timeouts = _timeout_map(args.timeout_s, list(args.suite_timeout))
    smoke_limit = max(0, int(args.smoke_limit))
    full = smoke_limit == 0
    cleanup_reference_suites = set()
    if not args.keep_reference_output:
        cleanup_reference_suites = set(args.cleanup_reference_suite or DEFAULT_CLEANUP_REFERENCE_SUITES)

    specs = runner._benchmark_specs(args.data_root)
    audio_num_gpu_nodes = max(0, int(args.audio_num_gpu_nodes))
    if audio_num_gpu_nodes > 0:
        audio = specs["audio"]
        specs["audio"] = replace(
            audio,
            full_extra_env={
                **audio.full_extra_env,
                "AUDIO_NUM_GPU_NODES": str(audio_num_gpu_nodes),
                "NUM_GPU_NODES": str(audio_num_gpu_nodes),
            },
        )

    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[fte-matrix] run_root={run_root}", flush=True)
    print(f"[fte-matrix] manifest_path={manifest_path}", flush=True)
    print(f"[fte-matrix] report_path={report_path}", flush=True)
    print(f"[fte-matrix] suites={','.join(suites)}", flush=True)
    print(f"[fte-matrix] full={full} smoke_limit={smoke_limit}", flush=True)
    print(f"[fte-matrix] cleanup_reference_suites={','.join(sorted(cleanup_reference_suites)) or '-'}", flush=True)

    try:
        for suite in suites:
            print(
                f"[fte-matrix] suite={suite} timeout_s={timeouts[suite]}",
                flush=True,
            )
            runner._run_one(
                specs[suite],
                mode=args.mode,
                run_root=run_root,
                smoke_limit=smoke_limit,
                full=full,
                timeout_s=timeouts[suite],
                dry_run=bool(args.dry_run),
                compare_reference=True,
                manifest_path=manifest_path,
                resume=bool(args.resume),
                label="all",
                cleanup_reference_before_chaos=suite in cleanup_reference_suites,
            )
    finally:
        write_report(
            report_path=report_path,
            run_root=run_root,
            manifest_path=manifest_path,
            suites=suites,
            cleanup_reference_suites=cleanup_reference_suites,
        )
        print(f"[fte-matrix] wrote report {report_path}", flush=True)


if __name__ == "__main__":
    main()

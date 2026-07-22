# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = REPO_ROOT / "multimodal_inference_benchmarks"
DEFAULT_DATA_ROOT = Path(os.getenv("VANE_BENCHMARK_DATA_ROOT", "~/.cache/vane/benchmarks")).expanduser()


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    script: Path
    local_env: dict[str, str]
    extra_env: dict[str, str] = field(default_factory=dict)
    full_extra_env: dict[str, str] = field(default_factory=dict)
    smoke_limit_env: str = "INPUT_LIMIT"
    smoke_extra_env: dict[str, str] = field(default_factory=dict)
    stable_smoke_input_env: str | None = None
    stable_smoke_input_order_by: str | None = None


@dataclass(frozen=True)
class OutputSummary:
    row_count: int
    column_count: int
    schema: tuple[tuple[str, str], ...]
    hash_sum: str
    hash_xor: str


def _summary_to_dict(summary: OutputSummary) -> dict[str, object]:
    return {
        "row_count": summary.row_count,
        "column_count": summary.column_count,
        "schema": [{"name": name, "type": column_type} for name, column_type in summary.schema],
        "hash_sum": summary.hash_sum,
        "hash_xor": summary.hash_xor,
    }


def _summary_from_dict(payload: Mapping[str, object]) -> OutputSummary:
    schema_payload = payload.get("schema") or ()
    schema: list[tuple[str, str]] = []
    for column in schema_payload:
        if isinstance(column, Mapping):
            schema.append((str(column.get("name")), str(column.get("type"))))
        else:
            name, column_type = column  # type: ignore[misc]
            schema.append((str(name), str(column_type)))
    return OutputSummary(
        row_count=int(payload.get("row_count", 0)),
        column_count=int(payload.get("column_count", len(schema))),
        schema=tuple(schema),
        hash_sum=str(payload.get("hash_sum", "0")),
        hash_xor=str(payload.get("hash_xor", "0")),
    )


def _summary_json_path(suite_root: Path, label: str) -> Path:
    return suite_root / f"{label}_summary.json"


def _write_summary_json(
    path: Path,
    *,
    spec: BenchmarkSpec,
    label: str,
    mode: str,
    full: bool,
    smoke_limit: int,
    chaos: bool,
    output_path: Path,
    summary: OutputSummary,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "suite": spec.name,
        "label": label,
        "mode": mode,
        "full": full,
        "smoke_limit": smoke_limit,
        "chaos": chaos,
        "output_path": str(output_path),
        "summary": _summary_to_dict(summary),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_summary_json(path: Path) -> OutputSummary:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary_payload = payload.get("summary", payload)
    if not isinstance(summary_payload, Mapping):
        raise TypeError(f"summary JSON {path} does not contain an object summary")
    return _summary_from_dict(summary_payload)


def _append_manifest_event(manifest_path: Path | None, event: dict[str, object]) -> None:
    if manifest_path is None:
        return
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **event,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _path_string(path: Path | str) -> str:
    return str(path)


def _quote_sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _normalize_parquet_input(path: str) -> str:
    if any(ch in path for ch in ("*", "?", "[")):
        return path
    lower = path.lower()
    if lower.endswith((".parquet", ".parquet.gz")):
        return path
    if path.startswith("s3://"):
        return path.rstrip("/") + "/*.parquet"
    input_path = Path(path)
    if input_path.is_dir():
        return str(input_path / "**" / "*.parquet")
    return path


def _materialize_stable_smoke_input(
    *,
    spec: BenchmarkSpec,
    suite_root: Path,
    input_path: str,
    smoke_limit: int,
) -> Path:
    if spec.stable_smoke_input_order_by is None:
        raise ValueError("stable smoke input order column is not configured")
    import vane

    output_path = suite_root / "smoke_input" / f"{spec.name}_{int(smoke_limit)}.parquet"
    if output_path.exists():
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    con = vane.connect()
    try:
        source = _quote_sql_string(_normalize_parquet_input(input_path))
        output = _quote_sql_string(str(output_path))
        order_by = _quote_identifier(spec.stable_smoke_input_order_by)
        con.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet({source}, union_by_name=true)
                ORDER BY {order_by}
                LIMIT {int(smoke_limit)}
            )
            TO {output} (FORMAT PARQUET)
            """
        )
    finally:
        con.close()
    return output_path


def _benchmark_specs(data_root: Path) -> dict[str, BenchmarkSpec]:
    return {
        "document": BenchmarkSpec(
            name="document",
            script=BENCH_ROOT / "document_embedding" / "vane_main.py",
            local_env={
                "INPUT_PATH": _path_string(data_root / "digitalcorpora" / "metadata"),
                "LOCAL_PDF_ROOT": _path_string(data_root / "digitalcorpora" / "pdf_dump"),
            },
        ),
        "image": BenchmarkSpec(
            name="image",
            script=BENCH_ROOT / "image_classification" / "vane_main.py",
            local_env={
                "INPUT_PATH": _path_string(data_root / "imagenet" / "metadata_file.parquet"),
                "LOCAL_IMAGE_ROOT": _path_string(data_root / "imagenet" / "train"),
            },
        ),
        "audio": BenchmarkSpec(
            name="audio",
            script=BENCH_ROOT / "audio_transcription" / "vane_main.py",
            local_env={
                "INPUT_PATH": _path_string(data_root / "common_voice_17" / "parquet"),
            },
            stable_smoke_input_env="INPUT_PATH",
            stable_smoke_input_order_by="path",
        ),
        "video": BenchmarkSpec(
            name="video",
            script=BENCH_ROOT / "video_object_detection" / "vane_main.py",
            local_env={
                "INPUT_PATH": _path_string(data_root / "hollywood2" / "AVIClips"),
            },
            smoke_extra_env={"VIDEO_FILE_LIMIT": "2"},
        ),
    }


def _base_chaos_env(mode: str, shuffle_dir: Path) -> dict[str, str]:
    env = _base_distributed_env(shuffle_dir)
    env.update(
        {
            "VANE_FTE_CHAOS_KILL_WORKER_ON_RUNNING": "1",
            "VANE_FTE_CHAOS_KILL_ATTEMPT_ID": "0",
            "VANE_FTE_CHAOS_KILL_WORKER_INDEX": "0",
        }
    )
    if mode == "host":
        env["VANE_FTE_CHAOS_FAIL_HOST_ON_WORKER_LOSS"] = "1"
    return env


def _base_reference_env(mode: str, shuffle_dir: Path) -> dict[str, str]:
    return _base_distributed_env(shuffle_dir)


def _base_distributed_env(shuffle_dir: Path) -> dict[str, str]:
    return {
        "VANE_FTE_STATUS_WAIT_TIMEOUT_S": "1",
        "VANE_FTE_CONTROL_RPC_INITIAL_BACKOFF_S": "0",
        "VANE_FTE_SPLIT_QUEUE_SPACE_WAIT_TIMEOUT_S": "0.1",
        "VANE_SHUFFLE_ALGORITHM": "flight_shuffle",
        "VANE_SHUFFLE_LOCAL_DIRS": str(shuffle_dir),
        "RAY_DEDUP_LOGS": "0",
    }


def _validate_inputs(spec: BenchmarkSpec) -> None:
    for key, value in spec.local_env.items():
        if key.startswith("VANE_"):
            continue
        if "**" in value or "*" in value:
            parent = Path(value.split("*", 1)[0]).parent
            if not parent.exists():
                raise FileNotFoundError(f"{spec.name}: input parent for {key} does not exist: {parent}")
            continue
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"{spec.name}: {key} does not exist: {path}")


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _validate_output(output_path: Path) -> OutputSummary:
    parquet_files = sorted(output_path.glob("**/*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"no parquet files written under {output_path}")
    import vane

    con = vane.connect()
    try:
        glob = str(output_path / "**" / "*.parquet")
        row_count = con.execute(
            "SELECT count(*)::BIGINT FROM read_parquet(?, union_by_name=true)",
            [glob],
        ).fetchone()[0]
        schema_rows = con.execute(
            "DESCRIBE SELECT * FROM read_parquet(?, union_by_name=true)",
            [glob],
        ).fetchall()
        schema = tuple((str(row[0]), str(row[1])) for row in schema_rows)
        if schema:
            row_hash_expr = "hash(%s)" % ", ".join(_quote_identifier(name) for name, _ in schema)
            hash_sum, hash_xor = con.execute(
                f"""
                WITH row_hashes AS (
                    SELECT {row_hash_expr} AS row_hash
                    FROM read_parquet(?, union_by_name=true)
                )
                SELECT
                    coalesce(sum(row_hash::HUGEINT), 0)::VARCHAR,
                    coalesce(bit_xor(row_hash), 0)::VARCHAR
                FROM row_hashes
                """,
                [glob],
            ).fetchone()
        else:
            hash_sum, hash_xor = "0", "0"
    finally:
        con.close()
    if int(row_count) <= 0:
        raise RuntimeError(f"output under {output_path} has no rows")
    column_count = len(schema)
    if column_count <= 0:
        raise RuntimeError(f"output under {output_path} has no columns")
    return OutputSummary(
        row_count=int(row_count),
        column_count=column_count,
        schema=schema,
        hash_sum=str(hash_sum),
        hash_xor=str(hash_xor),
    )


def _compare_output_summaries(
    spec: BenchmarkSpec,
    reference: OutputSummary,
    chaos: OutputSummary,
) -> None:
    if chaos.schema != reference.schema:
        raise RuntimeError(f"{spec.name}: output schema mismatch reference={reference.schema!r} chaos={chaos.schema!r}")
    if chaos.row_count != reference.row_count:
        raise RuntimeError(
            f"{spec.name}: output row count mismatch reference={reference.row_count} chaos={chaos.row_count}"
        )
    if (chaos.hash_sum, chaos.hash_xor) != (reference.hash_sum, reference.hash_xor):
        raise RuntimeError(
            f"{spec.name}: output checksum mismatch "
            f"reference=({reference.hash_sum}, {reference.hash_xor}) "
            f"chaos=({chaos.hash_sum}, {chaos.hash_xor})"
        )


def _run_one(
    spec: BenchmarkSpec,
    *,
    mode: str,
    run_root: Path,
    smoke_limit: int,
    full: bool,
    timeout_s: int,
    dry_run: bool,
    compare_reference: bool,
    manifest_path: Path | None,
    resume: bool,
    label: str = "all",
    cleanup_reference_before_chaos: bool = False,
) -> None:
    if label not in ("all", "reference", "chaos"):
        raise ValueError(f"unsupported benchmark label: {label}")
    _validate_inputs(spec)
    suite_root = run_root / spec.name
    output_path = suite_root / "output"
    shuffle_dir = suite_root / "shuffle"
    log_path = suite_root / "run.log"
    reference_output_path = suite_root / "reference_output"
    reference_shuffle_dir = suite_root / "reference_shuffle"
    reference_log_path = suite_root / "reference.log"
    reference_summary_path = _summary_json_path(suite_root, "reference")
    suite_root.mkdir(parents=True, exist_ok=True)
    shuffle_dir.mkdir(parents=True, exist_ok=True)
    reference_shuffle_dir.mkdir(parents=True, exist_ok=True)

    command = [sys.executable, str(spec.script)]
    reference_summary: OutputSummary | None = None
    stable_smoke_env: dict[str, str] = {}
    if not full and smoke_limit > 0 and spec.stable_smoke_input_env and spec.stable_smoke_input_order_by:
        source_input = spec.local_env.get(spec.stable_smoke_input_env) or os.environ.get(
            spec.stable_smoke_input_env,
            "",
        )
        if not source_input:
            raise RuntimeError(f"{spec.name}: stable smoke input env {spec.stable_smoke_input_env} is not set")
        stable_input_path = _materialize_stable_smoke_input(
            spec=spec,
            suite_root=suite_root,
            input_path=source_input,
            smoke_limit=smoke_limit,
        )
        stable_smoke_env[spec.stable_smoke_input_env] = str(stable_input_path)
        _append_manifest_event(
            manifest_path,
            {
                "event": "stable_smoke_input_prepared",
                "suite": spec.name,
                "mode": mode,
                "full": full,
                "smoke_limit": smoke_limit,
                "source_path": source_input,
                "output_path": str(stable_input_path),
                "input_env": spec.stable_smoke_input_env,
                "order_by": spec.stable_smoke_input_order_by,
            },
        )

    def build_env(output: Path, shuffle: Path, *, chaos: bool) -> dict[str, str]:
        env = os.environ.copy()
        if chaos:
            env.update(_base_chaos_env(mode, shuffle))
        else:
            env.update(_base_reference_env(mode, shuffle))
        env.update(spec.local_env)
        env.update(spec.extra_env)
        env.update(spec.full_extra_env if full else spec.smoke_extra_env)
        env.update(stable_smoke_env)
        env["OUTPUT_PATH"] = str(output)
        if not full and smoke_limit > 0:
            env[spec.smoke_limit_env] = str(smoke_limit)
        return env

    def run_command(label: str, output: Path, shuffle: Path, log: Path, *, chaos: bool) -> OutputSummary:
        env = build_env(output, shuffle, chaos=chaos)
        summary_path = _summary_json_path(suite_root, label)
        print(f"[fte-chaos] suite={spec.name} label={label} mode={mode} output={output}", flush=True)
        print(f"[fte-chaos] command={' '.join(command)}", flush=True)
        if dry_run:
            interesting = {
                key: env[key]
                for key in sorted(env)
                if key in spec.local_env
                or key in spec.extra_env
                or key in spec.full_extra_env
                or key in spec.smoke_extra_env
                or key in ("OUTPUT_PATH", spec.smoke_limit_env)
                or key.startswith("VANE_FTE_CHAOS")
            }
            for key, value in interesting.items():
                print(f"[fte-chaos][dry-run][{label}] {key}={value}", flush=True)
            summary = OutputSummary(0, 0, (), "0", "0")
            _append_manifest_event(
                manifest_path,
                {
                    "event": "dry_run",
                    "suite": spec.name,
                    "label": label,
                    "mode": mode,
                    "full": full,
                    "smoke_limit": smoke_limit,
                    "chaos": chaos,
                    "command": command,
                    "output_path": str(output),
                    "shuffle_dir": str(shuffle),
                    "log_path": str(log),
                    "env": interesting,
                    "summary": _summary_to_dict(summary),
                },
            )
            return summary

        if resume and output.exists():
            try:
                summary = _validate_output(output)
            except Exception as exc:
                _append_manifest_event(
                    manifest_path,
                    {
                        "event": "resume_invalid",
                        "suite": spec.name,
                        "label": label,
                        "mode": mode,
                        "full": full,
                        "smoke_limit": smoke_limit,
                        "chaos": chaos,
                        "command": command,
                        "output_path": str(output),
                        "shuffle_dir": str(shuffle),
                        "log_path": str(log),
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
            else:
                print(
                    f"[fte-chaos] suite={spec.name} label={label} resume passed "
                    f"rows={summary.row_count} columns={summary.column_count} "
                    f"checksum=({summary.hash_sum}, {summary.hash_xor})",
                    flush=True,
                )
                _append_manifest_event(
                    manifest_path,
                    {
                        "event": "resume_success",
                        "suite": spec.name,
                        "label": label,
                        "mode": mode,
                        "full": full,
                        "smoke_limit": smoke_limit,
                        "chaos": chaos,
                        "command": command,
                        "output_path": str(output),
                        "shuffle_dir": str(shuffle),
                        "log_path": str(log),
                        "summary": _summary_to_dict(summary),
                    },
                )
                _write_summary_json(
                    summary_path,
                    spec=spec,
                    label=label,
                    mode=mode,
                    full=full,
                    smoke_limit=smoke_limit,
                    chaos=chaos,
                    output_path=output,
                    summary=summary,
                )
                return summary

        start = time.monotonic()
        try:
            with log.open("w", encoding="utf-8") as log_file:
                proc = subprocess.run(
                    command,
                    cwd=str(REPO_ROOT),
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_s,
                    check=False,
                )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - start
            _append_manifest_event(
                manifest_path,
                {
                    "event": "timeout",
                    "suite": spec.name,
                    "label": label,
                    "mode": mode,
                    "full": full,
                    "smoke_limit": smoke_limit,
                    "chaos": chaos,
                    "command": command,
                    "output_path": str(output),
                    "shuffle_dir": str(shuffle),
                    "log_path": str(log),
                    "elapsed_s": round(elapsed, 3),
                    "timeout_s": timeout_s,
                    "error": f"benchmark timed out after {timeout_s}s",
                },
            )
            raise RuntimeError(
                f"{spec.name}: {label} benchmark timed out after {timeout_s}s elapsed={elapsed:.1f}s log={log}"
            ) from exc
        except KeyboardInterrupt:
            elapsed = time.monotonic() - start
            _append_manifest_event(
                manifest_path,
                {
                    "event": "interrupted",
                    "suite": spec.name,
                    "label": label,
                    "mode": mode,
                    "full": full,
                    "smoke_limit": smoke_limit,
                    "chaos": chaos,
                    "command": command,
                    "output_path": str(output),
                    "shuffle_dir": str(shuffle),
                    "log_path": str(log),
                    "elapsed_s": round(elapsed, 3),
                    "error": "benchmark interrupted",
                },
            )
            raise
        elapsed = time.monotonic() - start
        if proc.returncode != 0:
            _append_manifest_event(
                manifest_path,
                {
                    "event": "failure",
                    "suite": spec.name,
                    "label": label,
                    "mode": mode,
                    "full": full,
                    "smoke_limit": smoke_limit,
                    "chaos": chaos,
                    "command": command,
                    "output_path": str(output),
                    "shuffle_dir": str(shuffle),
                    "log_path": str(log),
                    "elapsed_s": round(elapsed, 3),
                    "returncode": proc.returncode,
                    "error": f"benchmark failed rc={proc.returncode}",
                },
            )
            raise RuntimeError(
                f"{spec.name}: {label} benchmark failed rc={proc.returncode} elapsed={elapsed:.1f}s log={log}"
            )
        summary = _validate_output(output)
        _write_summary_json(
            summary_path,
            spec=spec,
            label=label,
            mode=mode,
            full=full,
            smoke_limit=smoke_limit,
            chaos=chaos,
            output_path=output,
            summary=summary,
        )
        print(
            f"[fte-chaos] suite={spec.name} label={label} passed elapsed={elapsed:.1f}s "
            f"rows={summary.row_count} columns={summary.column_count} "
            f"checksum=({summary.hash_sum}, {summary.hash_xor}) log={log}",
            flush=True,
        )
        _append_manifest_event(
            manifest_path,
            {
                "event": "success",
                "suite": spec.name,
                "label": label,
                "mode": mode,
                "full": full,
                "smoke_limit": smoke_limit,
                "chaos": chaos,
                "command": command,
                "output_path": str(output),
                "shuffle_dir": str(shuffle),
                "log_path": str(log),
                "elapsed_s": round(elapsed, 3),
                "returncode": proc.returncode,
                "summary": _summary_to_dict(summary),
            },
        )
        return summary

    def load_reference_summary() -> OutputSummary:
        try:
            summary = _validate_output(reference_output_path)
        except Exception as exc:
            if reference_summary_path.exists():
                try:
                    summary = _load_summary_json(reference_summary_path)
                except Exception as summary_exc:
                    _append_manifest_event(
                        manifest_path,
                        {
                            "event": "reference_summary_invalid",
                            "suite": spec.name,
                            "label": "reference",
                            "mode": mode,
                            "full": full,
                            "smoke_limit": smoke_limit,
                            "chaos": False,
                            "command": command,
                            "output_path": str(reference_output_path),
                            "summary_path": str(reference_summary_path),
                            "error": f"{type(summary_exc).__name__}: {summary_exc}",
                        },
                    )
                else:
                    print(
                        f"[fte-chaos] suite={spec.name} label=reference summary loaded "
                        f"rows={summary.row_count} columns={summary.column_count} "
                        f"checksum=({summary.hash_sum}, {summary.hash_xor}) "
                        f"summary={reference_summary_path}",
                        flush=True,
                    )
                    _append_manifest_event(
                        manifest_path,
                        {
                            "event": "reference_summary_loaded",
                            "suite": spec.name,
                            "label": "reference",
                            "mode": mode,
                            "full": full,
                            "smoke_limit": smoke_limit,
                            "chaos": False,
                            "command": command,
                            "output_path": str(reference_output_path),
                            "summary_path": str(reference_summary_path),
                            "summary": _summary_to_dict(summary),
                        },
                    )
                    return summary
            _append_manifest_event(
                manifest_path,
                {
                    "event": "reference_invalid",
                    "suite": spec.name,
                    "label": "reference",
                    "mode": mode,
                    "full": full,
                    "smoke_limit": smoke_limit,
                    "chaos": False,
                    "command": command,
                    "output_path": str(reference_output_path),
                    "shuffle_dir": str(reference_shuffle_dir),
                    "log_path": str(reference_log_path),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise RuntimeError(
                f"{spec.name}: existing reference output is not valid for comparison: {reference_output_path}: {exc}"
            ) from exc
        print(
            f"[fte-chaos] suite={spec.name} label=reference loaded "
            f"rows={summary.row_count} columns={summary.column_count} "
            f"checksum=({summary.hash_sum}, {summary.hash_xor})",
            flush=True,
        )
        _write_summary_json(
            reference_summary_path,
            spec=spec,
            label="reference",
            mode=mode,
            full=full,
            smoke_limit=smoke_limit,
            chaos=False,
            output_path=reference_output_path,
            summary=summary,
        )
        _append_manifest_event(
            manifest_path,
            {
                "event": "reference_loaded",
                "suite": spec.name,
                "label": "reference",
                "mode": mode,
                "full": full,
                "smoke_limit": smoke_limit,
                "chaos": False,
                "command": command,
                "output_path": str(reference_output_path),
                "shuffle_dir": str(reference_shuffle_dir),
                "log_path": str(reference_log_path),
                "summary": _summary_to_dict(summary),
            },
        )
        return summary

    if compare_reference and label in ("all", "reference"):
        if (
            label == "all"
            and resume
            and cleanup_reference_before_chaos
            and not reference_output_path.exists()
            and reference_summary_path.exists()
            and not dry_run
        ):
            reference_summary = load_reference_summary()
        else:
            reference_summary = run_command(
                "reference",
                reference_output_path,
                reference_shuffle_dir,
                reference_log_path,
                chaos=False,
            )
        if label == "all" and cleanup_reference_before_chaos and not dry_run and reference_output_path.exists():
            shutil.rmtree(reference_output_path)
            _append_manifest_event(
                manifest_path,
                {
                    "event": "reference_output_cleaned",
                    "suite": spec.name,
                    "label": "reference",
                    "mode": mode,
                    "full": full,
                    "smoke_limit": smoke_limit,
                    "chaos": False,
                    "output_path": str(reference_output_path),
                    "summary_path": str(reference_summary_path),
                    "reason": "cleanup_reference_before_chaos",
                },
            )
    elif compare_reference and label == "chaos" and not dry_run:
        reference_summary = load_reference_summary()

    chaos_summary: OutputSummary | None = None
    if label in ("all", "chaos"):
        chaos_summary = run_command("chaos", output_path, shuffle_dir, log_path, chaos=True)

    if compare_reference and reference_summary is not None and chaos_summary is not None and not dry_run:
        try:
            _compare_output_summaries(spec, reference_summary, chaos_summary)
        except Exception as exc:
            _append_manifest_event(
                manifest_path,
                {
                    "event": "comparison_failure",
                    "suite": spec.name,
                    "mode": mode,
                    "full": full,
                    "smoke_limit": smoke_limit,
                    "reference": _summary_to_dict(reference_summary),
                    "chaos": _summary_to_dict(chaos_summary),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise
        _append_manifest_event(
            manifest_path,
            {
                "event": "comparison_success",
                "suite": spec.name,
                "mode": mode,
                "full": full,
                "smoke_limit": smoke_limit,
                "reference": _summary_to_dict(reference_summary),
                "chaos": _summary_to_dict(chaos_summary),
            },
        )
        print(
            f"[fte-chaos] suite={spec.name} reference comparison passed "
            f"rows={chaos_summary.row_count} columns={chaos_summary.column_count}",
            flush=True,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Vane multimodal benchmarks with FTE worker/host-loss chaos enabled."
    )
    parser.add_argument(
        "--suite",
        action="append",
        choices=("document", "image", "audio", "video", "all"),
        default=None,
        help="Suite to run. Repeat for multiple suites. Default: all.",
    )
    parser.add_argument("--mode", choices=("worker", "host"), default="host")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-root", type=Path, default=Path("/tmp/vane_fte_chaos_benchmarks") / uuid.uuid4().hex)
    parser.add_argument("--smoke-limit", type=int, default=16)
    parser.add_argument("--full", action="store_true", help="Do not set INPUT_LIMIT smoke limits.")
    parser.add_argument(
        "--compare-reference",
        action="store_true",
        help="Run the same distributed/FTE benchmark without chaos first and compare output schema, row count, and checksum.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="JSONL manifest path. Default: <run-root>/manifest.jsonl.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing validated output directories instead of rerunning completed suite labels.",
    )
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--label",
        choices=("all", "reference", "chaos"),
        default="all",
        help="Run both labels, only the reference label, or only the chaos label. Default: all.",
    )
    parser.add_argument(
        "--cleanup-reference-before-chaos",
        action="store_true",
        help=(
            "Persist reference_summary.json and remove reference_output before running chaos. "
            "This is useful for large local outputs such as video."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    specs = _benchmark_specs(args.data_root)
    requested = args.suite or ["all"]
    suites = list(specs) if "all" in requested else requested
    args.run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_path or (args.run_root / "manifest.jsonl")
    print(f"[fte-chaos] run_root={args.run_root}", flush=True)
    print(f"[fte-chaos] manifest_path={manifest_path}", flush=True)
    for suite in suites:
        _run_one(
            specs[suite],
            mode=args.mode,
            run_root=args.run_root,
            smoke_limit=max(0, int(args.smoke_limit)),
            full=bool(args.full),
            timeout_s=int(args.timeout_s),
            dry_run=bool(args.dry_run),
            compare_reference=bool(args.compare_reference),
            manifest_path=manifest_path,
            resume=bool(args.resume),
            label=str(args.label),
            cleanup_reference_before_chaos=bool(args.cleanup_reference_before_chaos),
        )


if __name__ == "__main__":
    main()

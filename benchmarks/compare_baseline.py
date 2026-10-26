#!/usr/bin/env python3
"""Committed-baseline instruction-count comparison for the benchmark suite. See benchmarks/README.md.

pytest-codspeed's hooks call `callgrind_dump_stats_at(<uri>)` per benchmark, so callgrind writes ONE dump each,
headed by `desc: Trigger: Client Request: <uri>` with the count on `totals:` (`events: Ir`). This parses those
raw dumps directly (no CodSpeed account/token/runner). Run-to-run noise is ~0.1%, so the 5% gate threshold sits
far above it (PYTHONHASHSEED pinned in CI).

Two modes (CI-only; no valgrind on macOS arm64):
  regen:   write baseline.json from a fresh run: counts + provenance + binding fractions + auto-move.
  compare: diff a fresh run against baseline.json. Gate benches over threshold are regressions; informational
           are reported only. Report-only by default; `--enforce` exits non-zero on a gate regression.

baseline.json and benchmarks/requirements-bench.txt are regenerated together so counts match the frozen pins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
GATE_DEFAULT_THRESHOLD_PCT = 5.0
BINDING_FRACTION_CUTOFF = 0.25  # a gate whose isolable binding fraction is below this is auto-moved to
#                                 informational (a threshold on its engine-diluted total is not meaningful).

# Floor map: the engine-control bench that is the "engine floor" of a numeric-produce gate.
# binding_fraction = 1 - floor_Ir / bench_Ir. ONLY numeric-produce benches are listed (their per-element binding
# is a bulk memcpy of ~engine magnitude); every other gate is high-binding and needs no fraction. Add a mapping
# (and, if needed, a floor) to evaluate more benches.
_E = "benchmarks/test_engine_control_perf.py"
FLOOR_MAP = {
    "benchmarks/test_produce_numpy_perf.py::test_df_numeric": f"{_E}::test_engine_sum_2col_500k",
    "benchmarks/test_produce_numpy_perf.py::test_fetchnumpy_numeric": f"{_E}::test_engine_sum_2col_500k",
    "benchmarks/test_types_roundtrip_perf.py::test_out_col_df[int64]": f"{_E}::test_engine_sum_1col_100k",
    "benchmarks/test_types_roundtrip_perf.py::test_out_col_df[double]": f"{_E}::test_engine_sum_1col_100k",
    "benchmarks/test_types_roundtrip_perf.py::test_out_col_df[bool]": f"{_E}::test_engine_sum_1col_100k",
    "benchmarks/test_types_roundtrip_perf.py::test_out_col_df[date]": f"{_E}::test_engine_sum_1col_100k",
}

_TRIGGER_RE = re.compile(r"^desc:\s*Trigger:\s*Client Request:\s*(?P<uri>.+?)\s*$")
_TOTALS_RE = re.compile(r"^totals:\s*(?P<ir>\d+)\s*$")


# --------------------------------------------------------------------------- #
# callgrind parsing
# --------------------------------------------------------------------------- #


def _normalize_uri(raw: str) -> str:
    """Return a repo-relative benchmark key (strip a leading absolute path if the run was outside a git repo)."""
    raw = raw.strip()
    if "::" not in raw:
        return raw
    path, _, rest = raw.partition("::")
    idx = path.find("benchmarks/")
    if idx > 0:
        path = path[idx:]
    return f"{path}::{rest}"


def parse_profiles(profile_dir: Path) -> dict[str, int]:
    """Parse every callgrind dump in `profile_dir`; return {benchmark_uri: instruction_count}.

    Keeps only dumps whose Trigger is a benchmark Client Request (contains `::`); skips metadata/termination
    dumps. If a uri appears more than once (should not happen) the max is kept.
    """
    counts: dict[str, int] = {}
    files = sorted(profile_dir.rglob("*")) if profile_dir.exists() else []
    for f in files:
        if not f.is_file():
            continue
        uri: str | None = None
        ir: int | None = None
        try:
            text = f.read_text(errors="replace")
        except (OSError, UnicodeError):
            continue
        for line in text.splitlines():
            m = _TRIGGER_RE.match(line)
            if m:
                uri = _normalize_uri(m.group("uri"))
                continue
            m = _TOTALS_RE.match(line)
            if m:
                ir = int(m.group("ir"))
        if uri and "::" in uri and ir is not None:
            counts[uri] = max(counts.get(uri, 0), ir)
    return counts


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _load_gate_set(gate_list: Path | None) -> set[str]:
    """Load the set of gate benchmark uris from a `pytest -m gate --collect-only -q` node-id list."""
    if not gate_list or not gate_list.exists():
        return set()
    out = set()
    for raw in gate_list.read_text().splitlines():
        line = raw.strip()
        if "::" in line:  # a pytest node-id (the workflow pre-filters the collect-only output to '::' lines)
            out.add(_normalize_uri(line))
    return out


def _pct(base: int, new: int) -> float:
    return 0.0 if base == 0 else (new - base) / base * 100.0


# --------------------------------------------------------------------------- #
# regen
# --------------------------------------------------------------------------- #


def regen(args: argparse.Namespace) -> int:
    """Write baseline.json from a valgrind run: counts + provenance + Option-B binding fractions/auto-move."""
    counts = parse_profiles(Path(args.profiles))
    if not counts:
        print(f"ERROR: no benchmark dumps found under {args.profiles}", file=sys.stderr)
        return 2
    gate_set = _load_gate_set(Path(args.gate_list) if args.gate_list else None)

    benches: dict[str, dict] = {}
    auto_moved: list[str] = []
    for uri, ir in sorted(counts.items()):
        source_marker = "gate" if uri in gate_set else "informational"
        marker = source_marker
        binding_fraction = None
        floor_uri = FLOOR_MAP.get(uri)
        if source_marker == "gate" and floor_uri and floor_uri in counts and ir > 0:
            binding_fraction = round(max(0.0, 1.0 - counts[floor_uri] / ir), 4)
            if binding_fraction < args.cutoff:
                marker = "informational"  # Option-B auto-move: engine-diluted, threshold not meaningful
                auto_moved.append(uri)
        benches[uri] = {
            "marker": marker,
            "source_marker": source_marker,
            "auto_moved": marker != source_marker,
            "instructions": ir,
            "binding_fraction": binding_fraction,
            "threshold_pct": GATE_DEFAULT_THRESHOLD_PCT if marker == "gate" else None,
        }

    baseline = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": args.git_commit,
            "duckdb_submodule_sha": args.submodule_sha,
            "requirements_bench_sha256": _sha256(Path(args.pins)) if args.pins else "",
            "measurement": {"tool": "valgrind callgrind", "event": "Ir", "pythonhashseed": "0"},
            "bench_scale": os.environ.get("BENCH_SCALE", ""),  # counts are only comparable at the same scale
            "gate_default_threshold_pct": GATE_DEFAULT_THRESHOLD_PCT,
            "binding_fraction_cutoff": args.cutoff,
            "noise_note": "callgrind Ir observed ~0.1% run-to-run; gate threshold set well above.",
        },
        "benchmarks": benches,
    }
    Path(args.out).write_text(json.dumps(baseline, indent=2) + "\n")
    n_gate = sum(1 for b in benches.values() if b["marker"] == "gate")
    n_info = len(benches) - n_gate
    print(f"Wrote {args.out}: {len(benches)} benchmarks ({n_gate} gate, {n_info} informational).")
    if auto_moved:
        print(f"Option-B auto-moved {len(auto_moved)} engine-diluted gate(s) to informational:")
        for uri in auto_moved:
            print(f"  {uri}  (binding_fraction={benches[uri]['binding_fraction']})")
        print("Recommend updating these benches' @pytest.mark.gate -> informational so code matches the baseline.")
    return 0


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #


def compare(args: argparse.Namespace) -> int:
    """Diff a fresh valgrind run against baseline.json and print a report (report-only unless --enforce)."""
    new_counts = parse_profiles(Path(args.profiles))
    if not new_counts:
        print(f"ERROR: no benchmark dumps found under {args.profiles}", file=sys.stderr)
        return 2
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        # Bootstrap state: no committed baseline yet. Report the run and instruct to regenerate; never fail.
        print(f"No baseline at {baseline_path} yet -- run the workflow with regen=true to create it.")
        print(f"This run produced {len(new_counts)} benchmark instruction counts.")
        return 0
    baseline = json.loads(baseline_path.read_text())
    meta = baseline.get("meta", {})
    base_benches = baseline.get("benchmarks", {})

    # scale guard: a baseline built at BENCH_SCALE=X is only comparable to a run at the same scale.
    run_scale = os.environ.get("BENCH_SCALE", "")
    base_scale = meta.get("bench_scale", "")
    if run_scale != base_scale:
        print(
            f"WARNING: BENCH_SCALE differs (run={run_scale!r}, baseline={base_scale!r}) -> instruction counts are "
            "not comparable. Regenerate the baseline at this scale."
        )

    # pin-drift guard: the baseline's counts only compare cleanly against the pinned data libs it was built with.
    if args.pins:
        cur = _sha256(Path(args.pins))
        base_pins = meta.get("requirements_bench_sha256", "")
        if cur and base_pins and cur != base_pins:
            print(
                "WARNING: benchmarks/requirements-bench.txt differs from the baseline's pins -> data-lib deltas "
                "may not be pure binding. Regenerate the baseline with the current pins."
            )

    # engine-bump guard: engine-inclusive counts shift when the DuckDB submodule changes. If the SHA differs from
    # the baseline's, don't treat gate deltas as hard failures (they may reflect the bump); warn to regenerate.
    engine_changed = bool(
        args.submodule_sha and meta.get("duckdb_submodule_sha") and args.submodule_sha != meta["duckdb_submodule_sha"]
    )

    regressions: list[str] = []
    rows: list[tuple[str, str, str]] = []  # (status, uri, detail)
    for uri, ir in sorted(new_counts.items()):
        b = base_benches.get(uri)
        if b is None:
            rows.append(("NEW", uri, f"{ir} Ir (no baseline)"))
            continue
        base_ir = b["instructions"]
        delta = _pct(base_ir, ir)
        marker = b.get("marker", "informational")
        thr = b.get("threshold_pct") or GATE_DEFAULT_THRESHOLD_PCT
        detail = f"{base_ir} -> {ir} Ir  ({delta:+.2f}%, thr {thr:.1f}%, {marker})"
        if marker == "gate" and delta > thr:
            if engine_changed:
                rows.append(("ENGINE?", uri, detail + "  [submodule changed -> not enforced]"))
            else:
                rows.append(("REGRESSION", uri, detail))
                regressions.append(uri)
        else:
            rows.append(("ok" if marker == "gate" else "info", uri, detail))
    rows.extend(
        ("MISSING", uri, "in baseline, absent from run (rename/removal?)")
        for uri in sorted(set(base_benches) - set(new_counts))
    )

    _print_report(meta, rows, engine_changed=engine_changed, enforce=args.enforce)

    if not args.enforce:
        return 0
    if engine_changed:
        print("\nNOT ENFORCING: DuckDB submodule differs from the baseline; regenerate the baseline.")
        return 0
    return 1 if regressions else 0


def _print_report(meta: dict, rows: list[tuple[str, str, str]], *, engine_changed: bool, enforce: bool) -> None:
    mode = "ENFORCING" if enforce else "REPORT-ONLY (not failing the job)"
    print("=" * 100)
    print(f"CodSpeed instruction-count baseline comparison  [{mode}]")
    print(
        f"baseline: commit {meta.get('git_commit', '?')[:12]}  submodule {str(meta.get('duckdb_submodule_sha'))[:12]}"
        f"  generated {meta.get('generated_at_utc', '?')}"
    )
    if engine_changed:
        print(
            "WARNING: DuckDB submodule SHA differs from the baseline -> engine-inclusive deltas may reflect the "
            "engine bump, not the binding. Regenerate the baseline for this engine."
        )
    print("=" * 100)
    order = {"REGRESSION": 0, "ENGINE?": 1, "MISSING": 2, "NEW": 3, "ok": 4, "info": 5}
    for status, uri, detail in sorted(rows, key=lambda r: (order.get(r[0], 9), r[1])):
        print(f"  [{status:>10}] {uri}\n               {detail}")
    n_reg = sum(1 for s, _, _ in rows if s == "REGRESSION")
    print("-" * 100)
    print(f"Summary: {len(rows)} benchmarks, {n_reg} gate regression(s)" + ("" if enforce else "  (report-only)"))


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: dispatch to the `regen` or `compare` subcommand."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("regen", help="write baseline.json from a valgrind run")
    r.add_argument("--profiles", required=True, help="CODSPEED_PROFILE_FOLDER with callgrind dumps")
    r.add_argument("--out", default="benchmarks/baseline.json")
    r.add_argument("--gate-list", help="file of gate node-ids (pytest -m gate --collect-only -q)")
    r.add_argument("--git-commit", default="")
    r.add_argument("--submodule-sha", default="")
    r.add_argument("--pins", default="benchmarks/requirements-bench.txt")
    r.add_argument("--cutoff", type=float, default=BINDING_FRACTION_CUTOFF)
    r.set_defaults(func=regen)

    c = sub.add_parser("compare", help="compare a valgrind run against baseline.json")
    c.add_argument("--profiles", required=True)
    c.add_argument("--baseline", default="benchmarks/baseline.json")
    c.add_argument("--submodule-sha", default="")
    c.add_argument(
        "--pins", default="benchmarks/requirements-bench.txt", help="warn if pins differ from the baseline's"
    )
    c.add_argument("--enforce", action="store_true", help="exit non-zero on a gate regression (default: report-only)")
    c.set_defaults(func=compare)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

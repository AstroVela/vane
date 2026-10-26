# Benchmark suite

CodSpeed micro-benchmarks for the binding hot paths (produce, ingest, UDF).
Design rationale: [PLAN.md](PLAN.md). CI: [../.github/workflows/codspeed.yml](../.github/workflows/codspeed.yml).

## Markers

Every benchmark carries exactly one (registered in `conftest.py`):

- **gate**: binding-dominated, GIL-held, deterministic under Callgrind. A threshold breach is a binding regression.
- **informational**: engine/library/streaming-diluted. Reported, never gated (would false-positive on engine bumps).

## Local A/B (walltime)

Only walltime runs locally (no Valgrind on macOS arm64; instruction-count gating is Linux/CI-only, and walltime is
noisy on sub-ms benches). Pin the data libs identically across both builds so the delta is pure binding:

```bash
for P in ../main/.venv-release/bin/python .venv-release/bin/python; do
  $P -m pytest benchmarks/<module>.py --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider
done
```

## Conventions

- READ aggregates real columns (`sum`/`length`), never `count(*)` (answered from metadata).
- WRITE fully materializes the result or drains the lazy reader.
- Warm once before measuring.
- `con` fixture pins `threads=1` (see `conftest.py`).

Two traps (a benchmark that skips these silently measures the wrong thing):

- OUT-col null benches need REAL nulls (`CASE WHEN ... THEN NULL`), else the cheap `std::move` path is taken.
- IN-numpy string benches need mixed ASCII + non-ASCII + a null sentinel, else the transcode/null ladder is skipped.

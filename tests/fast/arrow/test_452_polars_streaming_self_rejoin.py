"""Regression test for duckdb-python issue #452.

Silent row drop when two `db.sql(query).pl(lazy=True)` LazyFrames are joined,
the result is self-rejoined to derive grouping keys, a window expression is
applied downstream, and the plan is collected via `engine="streaming"`.

The streaming output is clamped to ~10.3M rows regardless of input size — at
20K / 30K / 50K variable-length groups (20M / 30M / 50M input rows) the
streaming output is ~10.30M / ~10.30M / ~10.31M.

This test is intentionally heavy: it must cross the bug threshold (>10M rows)
to trigger the failure. Runs in ~30 seconds at N_GROUPS=20_000.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import duckdb

if TYPE_CHECKING:
    from pathlib import Path

pl = pytest.importorskip("polars")
np = pytest.importorskip("numpy")
pytest.importorskip("pyarrow")


def test_452_polars_streaming_self_rejoin_does_not_drop_rows(tmp_path: Path) -> None:
    n_groups = 20_000
    rng = np.random.default_rng(42)
    group_lens = np.clip(rng.lognormal(mean=6.8, sigma=0.5, size=n_groups).astype(int), 30, 2900)
    g = np.repeat(np.arange(n_groups, dtype=np.int32), group_lens)
    t = np.concatenate([np.arange(n, dtype=np.int32) for n in group_lens])
    x = rng.uniform(-1.0, 1.0, int(group_lens.sum())).astype(np.float32)

    left_path = tmp_path / "left.parquet"
    right_path = tmp_path / "right.parquet"
    pl.DataFrame({"g": g, "t": t, "x": x}).write_parquet(left_path, row_group_size=200_000)
    pl.DataFrame({"g": g, "t": t}).write_parquet(right_path, row_group_size=200_000)
    del g, t, x

    def build(left_lf: pl.LazyFrame, right_lf: pl.LazyFrame) -> pl.LazyFrame:
        joined = left_lf.join(right_lf, on=["g", "t"], how="inner")
        keys = joined.select("g").unique()
        plan = joined.join(keys, on="g")
        return plan.sort(["g", "t"]).select(
            "g",
            "t",
            pl.col("x").rolling_sum(window_size=100).over("g").alias("y"),
        )

    ref = build(pl.scan_parquet(left_path), pl.scan_parquet(right_path)).collect()

    db_l = duckdb.connect(":memory:")
    db_r = duckdb.connect(":memory:")
    try:
        left_lf = db_l.sql(f"select * from read_parquet('{left_path}')").pl(lazy=True)
        right_lf = db_r.sql(f"select * from read_parquet('{right_path}')").pl(lazy=True)
        out = build(left_lf, right_lf).collect(engine="streaming")
    finally:
        db_l.close()
        db_r.close()

    assert out.shape == ref.shape, f"streaming output dropped rows: got {out.shape}, expected {ref.shape}"

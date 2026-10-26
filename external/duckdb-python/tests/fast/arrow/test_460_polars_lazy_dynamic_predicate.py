"""Regression test for duckdb-python issue #460 (half b).

A LazyFrame produced by `rel.pl(lazy=True)` panics polars when a `sort` +
`limit` (a TOP-N / slice) is collected. With Polars >= 1.39 the planner pushes a
*dynamic predicate* down to the IO source. It arrives not as a real expression
but as a `Display` node (`fmt_str: "dynamic_pred: <uuid>"`), and the fallback in
`duckdb/polars_io.py`'s `source_generator` tries to materialize it via
`pl.from_arrow(record_batch).filter(predicate)`. Polars's DSL->IR lowering hits
`unreachable!()` on that node shape and raises `pyo3_runtime.PanicException`
(`internal error: entered unreachable code`).

Only reproduces on Polars >= 1.39 — earlier planners don't push a dynamic
predicate to the IO source, so the test is skipped below that version.
"""

from __future__ import annotations

import pytest

import duckdb

# Dynamic-predicate pushdown to IO sources starts at Polars 1.39; below that the
# bug is unreachable, so skip rather than xfail.
pl = pytest.importorskip("polars", minversion="1.39")
pytest.importorskip("pyarrow")


@pytest.mark.parametrize("engine", ["streaming", "in-memory"])
def test_460_lazy_sort_limit_does_not_panic(engine: str) -> None:
    df = pl.DataFrame({"x": [3, 1, 2]})
    conn = duckdb.connect()
    try:
        conn.register("df_registered", df)
        lf = conn.sql("SELECT * FROM df_registered").pl(lazy=True)
        # sort + limit makes polars push a (pure) dynamic predicate into the source.
        out = lf.sort("x").limit(1).collect(engine=engine)
    finally:
        conn.close()

    assert out.to_dict(as_series=False) == {"x": [1]}


@pytest.mark.parametrize("engine", ["streaming", "in-memory"])
def test_460_user_filter_combined_with_dynamic_predicate_is_not_dropped(engine: str) -> None:
    # Polars ANDs the dynamic predicate onto the user's real filter into one
    # pushed predicate. Stripping the hint must NOT drop the real filter, and
    # polars does not re-filter above the source. Adversarial data: the global
    # min-x row (x=1) fails the y>15 filter, so dropping the filter yields a
    # different (wrong) row.
    df = pl.DataFrame({"x": [3, 1, 2], "y": [100, 5, 50]})
    conn = duckdb.connect()
    try:
        conn.register("t", df)
        lf = conn.sql("SELECT * FROM t").pl(lazy=True)
        out = lf.filter(pl.col("y") > 15).sort("x").limit(1).collect(engine=engine)
    finally:
        conn.close()

    ref = df.lazy().filter(pl.col("y") > 15).sort("x").limit(1).collect()
    assert out.to_dict(as_series=False) == ref.to_dict(as_series=False)
    assert out.to_dict(as_series=False) == {"x": [2], "y": [50]}


# --- unit tests for the stripping logic -------------------------------------

import json  # noqa: E402

from duckdb.polars_io import _strip_dynamic_predicates  # noqa: E402

_DYN = {"Display": {"inputs": [{"Column": "x"}], "fmt_str": "dynamic_pred: abc-123"}}


def _tree(expr: pl.Expr) -> dict:
    return json.loads(expr.meta.serialize(format="json"))


def test_strip_bare_dynamic_predicate_returns_none() -> None:
    assert _strip_dynamic_predicates(_DYN) == (None, True)


def test_strip_leaves_real_predicate_untouched() -> None:
    real = _tree(pl.col("y") > 15)
    assert _strip_dynamic_predicates(real) == (real, False)


def test_strip_and_of_real_and_dynamic_keeps_real() -> None:
    real = _tree(pl.col("y") > 15)
    for tree in (
        {"BinaryExpr": {"left": real, "op": "And", "right": _DYN}},
        {"BinaryExpr": {"left": _DYN, "op": "And", "right": real}},
    ):
        stripped, removed = _strip_dynamic_predicates(tree)
        assert removed is True
        assert stripped == real


def test_strip_raises_on_dynamic_outside_top_level_and() -> None:
    # An OR with a dynamic predicate cannot be safely dropped or applied.
    tree = {"BinaryExpr": {"left": _tree(pl.col("y") > 15), "op": "Or", "right": _DYN}}
    with pytest.raises(NotImplementedError):
        _strip_dynamic_predicates(tree)

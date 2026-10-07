# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Acceptance oracle checks and deterministic producer arrival permutations."""

import json
from itertools import permutations

import pyarrow as pa
import pytest

import vane
from tests.execution_acceptance import Case, Evidence, compare, corpus, native_reference
from tests.fast.test_direct_exchange import collect, submission, until
from vane.execution.direct_exchange import DirectExchangeLimits, InProcessTaskService


@pytest.mark.parametrize("ordered", [False, True])
def test_differential_oracle_preserves_duplicate_rows_and_order(ordered):
    case = Case("duplicates", "select 1", ordered)
    expected = pa.table({"v": [1, 1, 2, None]})
    reordered = pa.table({"v": [None, 2, 1, 1]})
    if ordered:
        with pytest.raises(AssertionError):
            compare(case, expected, reordered, expected.schema)
    else:
        compare(case, expected, reordered, expected.schema)
    with pytest.raises(AssertionError):
        compare(case, expected, pa.table({"v": [1, 2, 2, None]}), expected.schema)


def test_differential_oracle_rejects_schema_and_nested_value_changes():
    case = Case("nested", "select 1")
    expected = pa.table({"v": [[{"x": 1}, None], None, []]})
    for actual in (
        pa.table({"v": [[{"x": 2}, None], None, []]}),
        pa.table({"v": [[{"x": 1}, None], [], []]}),
        expected.rename_columns(["renamed"]),
        expected.cast(pa.schema([("v", pa.list_(pa.struct([("x", pa.int32())])))])),
    ):
        with pytest.raises(AssertionError):
            compare(case, expected, actual, expected.schema)


def test_float_tolerance_is_explicit_and_does_not_hide_nulls_or_nonfinite_values():
    expected = pa.table({"f": [1.0, float("nan"), float("inf"), -0.0, None]})
    case = Case("float", "select 1", ordered=True, approximate=True)
    compare(case, expected, pa.table({"f": [1 + 1e-13, float("nan"), float("inf"), 0.0, None]}), expected.schema)
    for values in (
        [1.1, float("nan"), float("inf"), 0.0, None],
        [None, float("nan"), float("inf"), 0.0, None],
        [1.0, 0.0, float("inf"), 0.0, None],
        [1.0, float("nan"), -float("inf"), 0.0, None],
    ):
        with pytest.raises(AssertionError):
            compare(case, expected, pa.table({"f": values}), expected.schema)
    with pytest.raises(AssertionError):
        compare(
            Case("exact", "select 1", True),
            expected,
            pa.table({"f": [1 + 1e-13, float("nan"), float("inf"), 0.0, None]}),
            expected.schema,
        )


@pytest.mark.parametrize("seed", [0, 970])
def test_seeded_corpus_has_reproducible_inputs_and_valid_native_reference(tmp_path, seed):
    from vane.execution.compiler import compile_fragment_graph

    first, second = tmp_path / "first", tmp_path / "second"
    cases = corpus(first, seed)
    replay = corpus(second, seed)
    inputs = json.loads((first / "inputs.json").read_text())
    copies = json.loads((second / "inputs.json").read_text())
    assert [f["sha256"] for f in inputs["files"]] == [f["sha256"] for f in copies["files"]]
    assert len(cases) == len({c.name for c in cases})
    with vane.connect(config={"threads": 1}) as connection:
        for case, other in zip(cases, replay):
            compile_fragment_graph(connection, case.sql, query_id=case.name)
            expected, schema = native_reference(connection, case)
            actual, other_schema = native_reference(connection, other)
            assert schema == other_schema
            # A native HUGEINT exporter declares precision 38 even for 39-digit
            # values. Widen only that known reference representation.
            compare(case, expected, actual.cast(schema), schema)


def test_failure_evidence_survives_a_broken_diagnostic_probe(tmp_path, monkeypatch):
    monkeypatch.delenv("VANE_TEST_DIAGNOSTICS_DIR", raising=False)
    evidence = Evidence(tmp_path / "failure", seed=970, partitions=3)
    evidence.begin(Case("tiny", "select 7"), mode="fte")

    class Broken:
        @property
        def query_runtime(self):
            raise RuntimeError("diagnostic probe failed")

    evidence.failed(ValueError("original failure"), Broken())
    assert json.loads((evidence.directory / "failure.json").read_text())["message"] == "original failure"
    assert (evidence.directory / "replay.sql").read_text() == "select 7;\n"
    assert (evidence.directory / "threads.txt").stat().st_size > 0


@pytest.mark.parametrize("git", ["missing", "unrelated"])
def test_source_archive_evidence_does_not_require_or_misidentify_git(tmp_path, monkeypatch, git):
    import tests.execution_acceptance as acceptance

    monkeypatch.delenv("VANE_TEST_DIAGNOSTICS_DIR", raising=False)

    def revision(*args, **kwargs):
        if git == "missing":
            raise FileNotFoundError("git unavailable")
        return f"{tmp_path}\nunrelated-commit\n"

    monkeypatch.setattr(acceptance.subprocess, "check_output", revision)
    evidence = Evidence(tmp_path / "source-archive")
    assert evidence.configuration["commit"] is None
    assert evidence.configuration["engine_identity"]


@pytest.mark.parametrize("kind", ["hugeint", "decimal(38,0)"])
@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_wide_aggregate_arrival_permutations(tmp_path, kind, order):
    # These three partitions contain negative, positive and zero values. Starting
    # one producer at a time forces its entire input to arrive before the next,
    # including prefixes that exceed a signed 128-bit accumulator.
    value = (
        f"case when range<3 then '-40000000000000000000000000000000000000'::{kind} "
        f"when range<6 then '60000000000000000000000000000000000000'::{kind} else 0::{kind} end"
    )
    sql = f"select sum(v)::varchar s, avg(v) a from (select {value} v from range(9))"
    evidence = Evidence(tmp_path / "arrival", kind=kind, order=order)
    case = Case("arrival", sql, ordered=True, approximate=True)
    evidence.begin(case)
    with vane.connect(config={"threads": 1}) as connection:
        expected = connection.execute(sql).fetchall()
        spec = submission(connection, sql, partitions=3)
        sources = {f.fragment_id for f in spec.graph.fragments if f.sources}
        with InProcessTaskService(connection, spec, DirectExchangeLimits(8192, 2048, 2, 2)) as service:
            try:
                for task in service.task_ids:
                    if task.rsplit("/", 1)[0] not in sources:
                        service.start(task)
                for partition in order:
                    started = [
                        t for t in service.task_ids if t.rsplit("/", 1)[0] in sources and t.endswith(f"/{partition}")
                    ]
                    assert started
                    for task in started:
                        service.start(task)
                    until(
                        service,
                        lambda: all(
                            t["state"] == "FINISHED" for t in service.snapshot()["tasks"] if t["task_id"] in started
                        ),
                    )
                actual = collect(service)
                assert actual[0][0] == expected[0][0]
                assert actual[0][1] == pytest.approx(expected[0][1], rel=1e-12)
                evidence.write("report.json", {"status": "passed", "order": order, "native": service.snapshot()})
            except BaseException as error:
                evidence.write("native.json", service.snapshot())
                evidence.failed(error, connection)
                raise

# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reproducible SQL corpus and independent result checks for execution acceptance."""

from __future__ import annotations

import faulthandler
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class Case:
    name: str
    sql: str
    ordered: bool = False
    approximate: bool = False


def corpus(directory: Path, seed: int) -> list[Case]:
    """Keep seeds, physical file order, empty files and duplicate references explicit."""
    rng = random.Random(seed)
    directory.mkdir(parents=True, exist_ok=False)
    schema = pa.schema(
        [
            ("id", pa.int64()),
            ("k", pa.int64()),
            ("v", pa.int64()),
            ("d", pa.decimal128(38, 3)),
            ("f", pa.float64()),
            ("tag", pa.string()),
        ]
    )
    rows = [
        dict(
            id=i,
            k=rng.choice([None, 0, 0, 0, 1, 2, 7]),
            v=None if i % 5 == 0 else rng.randint(-10000, 10000),
            d=None if i % 7 == 0 else Decimal(rng.randint(-(10**15), 10**15)).scaleb(-3),
            f=None if i % 11 == 0 else rng.uniform(-100, 100),
            tag=rng.choice([None, "", "a", "A", "中文", "ß", "repeated"]),
        )
        for i in range(59)
    ]
    rng.shuffle(rows)
    paths, offset = [], 0
    for part, size in enumerate((23, 0, 31, 5)):
        path = directory / f"part-{part}.parquet"
        pq.write_table(pa.Table.from_pylist(rows[offset : offset + size], schema=schema), path, row_group_size=7)
        paths.append(path)
        offset += size
    rng.shuffle(paths)
    # A duplicated reference is a duplicated scan, even in a frozen FTE source.
    paths.append(paths[0])
    names = ", ".join("'" + str(path).replace("'", "''") + "'" for path in paths)
    source = f"read_parquet([{names}])"
    inputs = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
    (directory / "inputs.json").write_text(json.dumps({"seed": seed, "files": inputs}, indent=2))
    nested = (
        "case when id % 4=0 then null else "
        "{'arrays': [[v, null]::bigint[2], [id, k]::bigint[2]], "
        "'empty': []::bigint[2][], 'decimals': [d, null], "
        "'map': map([id, id+100], [[v,k]::bigint[2], [id,null]::bigint[2]])} end payload"
    )
    cases = [
        Case(
            "scan_multiset",
            f"select k, v, d, tag, case when tag is null then null else 'binary\\x00\\xFF'::blob end b "
            f"from {source} where k is null or k<>2",
        ),
        Case(
            "group_filter_distinct",
            f"select k, count(*) n, sum(v) s, sum(d) ds, count(distinct tag) tags, "
            f"sum(v) filter(where id%3=0) filtered from {source} group by k",
        ),
        Case(
            "float_aggregate",
            f"select k, sum(f) s, avg(f) a from {source} group by k order by k nulls first",
            True,
            True,
        ),
        Case(
            "ordered_aggregate",
            f"select k, sum(f order by id, v) s, avg(f order by id, v) a "
            f"from {source} group by k order by k nulls last",
            True,
            True,
        ),
        Case("outer_join", f"select a.id, a.k, b.range as b, a.d from {source} a full join range(10) b on a.k=b.range"),
        Case("topn", f"select id,k,v,tag from {source} order by k nulls first, id, v limit 11 offset 3", True),
        Case("nested_arrays", f"select id, {nested} from {source} where id%3<>0 order by id", True),
        Case("empty_nested", f"select id, {nested} from {source} where id<0 order by id", True),
        Case("empty_aggregate", f"select count(*) n, sum(d) s, avg(v) a from {source} where id<0"),
        Case(
            "all_null",
            "select count(v) n, sum(v) s, avg(v) a, min(v) lo, max(v) hi "
            "from (select null::decimal(38,5) v from range(7))",
        ),
    ]
    wide = (
        "select range%2 k, sum(case when range%2=0 then '40000000000000000000000000000000000000'::decimal(38,0) "
        "else '-30000000000000000000000000000000000000'::decimal(38,0) end) s from range(6) group by k"
    )
    cases.extend(
        [
            Case("nested_decimal", f"select sum(s) s from (select k, max(s) s from ({wide}) group by k)"),
            Case(
                "hugeint_domain",
                "select range id, case range%3 when 0 then '-170141183460469231731687303715884105728'::hugeint "
                "when 1 then '170141183460469231731687303715884105727'::hugeint else null end h, "
                "[range::hugeint,null] hs from range(9) order by range desc",
                True,
            ),
            # Cast after aggregation so native TIME/INTERVAL values cross an exchange.
            # Native Arrow cannot represent their complete domain; native SQL text can.
            Case(
                "temporal_domain",
                "select min(t)::varchar lo, max(t)::varchar hi, min(i)::varchar span "
                "from (select case when range%2=0 then time '24:00:00' else time '00:00:00' end t, "
                "interval '2305843009213693952 microseconds' i from range(7))",
            ),
            Case(
                "special_floats",
                "select range id, case range%5 when 0 then 'NaN'::double when 1 then 'Infinity'::double "
                "when 2 then '-Infinity'::double when 3 then -0.0::double else null end f from range(13)",
            ),
        ]
    )
    return cases


def public_type(kind, arrow_type):
    """Map only documented representation differences; never cast the actual result."""
    if kind.id == "hugeint":
        return pa.decimal256(39, 0)
    if kind.id == "list":
        return pa.list_(public_type(kind.children[0][1], arrow_type.value_type))
    if kind.id == "array":
        return pa.list_(public_type(kind.children[0][1], arrow_type.value_type), kind.children[1][1])
    if kind.id == "struct":
        return pa.struct([(name, public_type(child, arrow_type.field(name).type)) for name, child in kind.children])
    if kind.id == "map":
        return pa.map_(
            public_type(kind.children[0][1], arrow_type.key_type),
            public_type(kind.children[1][1], arrow_type.item_type),
        )
    return arrow_type


def native_reference(connection, case):
    result = connection.execute(case.sql)
    kinds = [col[1] for col in result.description]
    table = result.to_arrow_table()
    expected_schema = pa.schema(
        [(field.name, public_type(kind, field.type)) for kind, field in zip(kinds, table.schema)]
    )
    return table, expected_schema


def canonical(value):
    if isinstance(value, float) and math.isnan(value):
        return ("nan",)
    if isinstance(value, dict):
        return tuple((name, canonical(child)) for name, child in value.items())
    if isinstance(value, (tuple, list)):
        return tuple(canonical(child) for child in value)
    return value


def compare(case, expected, actual, expected_schema):
    actual.validate(full=True)
    assert actual.schema.equals(expected_schema), (actual.schema, expected_schema)
    left, right = expected.to_pylist(), actual.to_pylist()
    assert len(left) == len(right), (len(left), len(right))
    if case.approximate:
        assert case.ordered, "approximate aggregates require deterministic group ordering"
        for index, (a, b) in enumerate(zip(left, right)):
            for name in a:
                if isinstance(a[name], float) and math.isfinite(a[name]):
                    assert b[name] is not None and math.isclose(a[name], b[name], rel_tol=1e-12, abs_tol=1e-12), (
                        index,
                        name,
                        a,
                        b,
                    )
                else:
                    assert canonical(a[name]) == canonical(b[name]), (index, name, a, b)
    elif case.ordered:
        assert [canonical(row) for row in left] == [canonical(row) for row in right]
    else:
        assert Counter(map(canonical, left)) == Counter(map(canonical, right))


class Evidence:
    """Write replay inputs before execution and diagnostics before caller cleanup."""

    def __init__(self, directory, **configuration):
        from vane._native.execution_plan import engine_identity

        source_root = Path(__file__).resolve().parents[1]
        try:
            git_root, revision = (
                subprocess.check_output(
                    ["git", "rev-parse", "--show-toplevel", "HEAD"],
                    cwd=source_root,
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                .strip()
                .splitlines()
            )
            commit = revision if Path(git_root).resolve() == source_root else None
        except (OSError, ValueError, subprocess.SubprocessError):
            commit = None  # Release tests also run from an unpacked source distribution.
        root = os.environ.get("VANE_TEST_DIAGNOSTICS_DIR")
        self.directory = (Path(root) / f"execution-{uuid.uuid4().hex}" if root else Path(directory)).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.configuration = {
            **configuration,
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {name: version(name) for name in ("vane-ai", "ray", "pyarrow")},
            "engine_identity": engine_identity(),
            "commit": commit,
        }
        self.write("configuration.json", self.configuration)
        print(f"Execution acceptance evidence: {self.directory}", file=sys.stderr, flush=True)

    def write(self, name, value):
        (self.directory / name).write_text(json.dumps(value, indent=2, default=str))

    def begin(self, case, **configuration):
        self.write("active.json", {**self.configuration, **configuration, **asdict(case)})
        (self.directory / "replay.sql").write_text(case.sql + ";\n")

    def failed(self, error, connection, result=None):
        # Persist the original exception before any diagnostic RPC. A failed
        # snapshot must not mask the failure or stop the caller's cleanup.
        try:
            self.write("failure.json", {"type": type(error).__name__, "message": str(error)})
            with (self.directory / "threads.txt").open("w") as output:
                faulthandler.dump_traceback(file=output, all_threads=True)
            if connection.query_runtime is not None:
                self.write("resources.json", connection.query_runtime.resource_snapshot())
            if result is not None:
                self.write("query.json", result.diagnostics())
        except Exception as diagnostic_error:
            print(f"Acceptance diagnostics failed: {diagnostic_error}", file=sys.stderr)


def assert_idle(connection):
    """Check all live ownership; monotonically increasing completion counters are excluded."""
    runtime = connection.query_runtime
    state = runtime.resource_snapshot()
    assert state["queries"] == {}, state
    for name in ("active_requests", "queued_requests"):
        assert state["request_admission"][name] == 0, state
    for name in ("active_results", "usage_bytes", "buffers", "waiting_bytes", "cleanup_pending_results"):
        assert state["result_delivery"][name] == 0, state
    if runtime.backend == "ray":
        import ray

        workers = runtime.pool.admission.snapshot()
        assert workers["reservations"] == {}, workers
        assert workers["waiting"] == [], workers
        for actor in runtime.pool.workers:
            assert ray.get(actor.resources_snapshot.remote(), timeout=5)["reservations"] == {}
        for store in runtime.stores.values():
            assert store.snapshot() == {"queries": 0, "reserved_bytes": 0}

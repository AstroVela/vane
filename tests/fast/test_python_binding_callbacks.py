# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded native binding probes, including implicit calls and exception cleanup."""

import subprocess
import sys
from pathlib import Path

import pytest

CONVERSIONS = [
    "arrow_protocol",
    "datasource_type",
    "project",
    "select_types",
    "aggregate",
    "repartition",
    "join",
    "sort_error",
    "statement_error",
    "replacement_type_error",
    "replacement_name_error",
    "parquet_partition",
    "csv_partition",
    "update_mapping",
    "merge_condition",
    "merge_clauses",
    "create_partition",
    "map_cpu",
    "map_batches_cpu",
    "flat_map_cpu",
]
LIFETIMES = [
    "tasks_return",
    "tasks_iteration_error",
    "tasks_pickle_error",
    "source_pickle_error",
    "registered_unregister",
    "registered_replace",
    "registered_close",
]


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("action", ["sql", "close", "propagate"])
@pytest.mark.parametrize("entry", CONVERSIONS)
def test_binding_conversions_reject_cross_connection_callbacks(monkeypatch, tmp_path, entry, action, configured):
    _run_probe(monkeypatch, tmp_path, entry, action, configured)


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("action", ["sql", "close"])
@pytest.mark.parametrize("entry", LIFETIMES)
def test_binding_owns_callback_cleanup_on_success_and_error(monkeypatch, tmp_path, entry, action, configured):
    if configured and entry.startswith("registered_"):
        pytest.importorskip("pandas")
    _run_probe(monkeypatch, tmp_path, entry, action, configured)


def _run_probe(monkeypatch, tmp_path, entry, action, configured):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    result = subprocess.run(
        [sys.executable, "-I", str(Path(__file__).resolve()), entry, action, str(configured), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_arrow_protocol_probe_preserves_descriptor_errors(monkeypatch):
    import vane

    monkeypatch.setenv("VANE_RUNNER", "local-fast")

    class Arrow:
        @property
        def __arrow_c_stream__(self):
            raise RuntimeError("protocol descriptor failure")

    with vane.connect() as con:
        with pytest.raises(RuntimeError, match="protocol descriptor failure"):
            con.from_arrow(Arrow())
        assert con.execute("SELECT 7").fetchall() == [(7,)]


def _probe(entry, action, configured, directory):
    import builtins
    import faulthandler
    import threading
    from collections.abc import Mapping as MappingABC
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow as pa

    import vane
    from vane.datasource import DataSource, DataSourceTask
    from vane.execution.request_admission import RequestAdmissionLimits

    faulthandler.dump_traceback_later(15, exit=True)
    connections = [vane.connect(config={"threads": 1}) for _ in range(2)]
    for con in connections:
        con.execute("CREATE TABLE target(x INTEGER)")
        con.execute("INSERT INTO target VALUES (1)")
    runtimes = []
    if configured:
        runtimes = [con.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1)) for con in connections]
    relations = [con.table("target") for con in connections]
    others = [con.sql("SELECT 1 AS x").set_alias("other") for con in connections]
    barrier = threading.Barrier(2, timeout=5)
    attempted = [False, False]
    rejected = [False, False]
    destroyed = [False, False]
    outer_errors = [None, None]

    def callback(index):
        if attempted[index] and action != "propagate" and not entry.startswith("replacement_"):
            return
        if not attempted[index]:
            attempted[index] = True
            # Each outer bind owns its cursor before entering this barrier.
            barrier.wait()
        try:
            if action == "close":
                connections[1 - index].close()
            else:
                connections[1 - index].sql("SELECT 42")
        except vane.InvalidInputException as error:
            assert "Python input callback" in str(error), str(error)
            rejected[index] = True
            if action == "propagate":
                raise
        else:
            raise AssertionError("binding callback entered another connection")

    def destroy(index):
        destroyed[index] = True
        callback(index)

    # Pickled task definitions must not capture the connections/barrier.
    builtins._vane_binding_callback = destroy

    class Task(DataSourceTask):
        def __init__(self, index):
            self.index = index

        def execute(self):
            import pyarrow as pa

            yield pa.record_batch({"x": [1]})

        def __del__(self):
            import builtins

            builtins._vane_binding_callback(self.index)

    class FailingTask(DataSourceTask):
        def execute(self):
            yield

        def __getstate__(self):
            raise RuntimeError("task pickle failure")

    class Source(DataSource):
        schema = {"x": "INTEGER"}

        def __init__(self, index, phase):
            self.index = index
            self.phase = phase

        def get_tasks(self):
            yield Task(self.index)
            if self.phase == "tasks_iteration_error":
                raise RuntimeError("task iteration failure")
            if self.phase == "tasks_pickle_error":
                yield FailingTask()

        def __getstate__(self):
            if self.phase == "source_pickle_error":
                raise RuntimeError("source pickle failure")
            return self.__dict__

    def run(index):
        class InputType(type):
            def __getattribute__(cls, name):
                if entry == "replacement_name_error" and name == "__name__":
                    callback(index)
                return super().__getattribute__(name)

        class InvalidInput(metaclass=InputType):
            @property
            def __class__(self):
                if entry == "replacement_type_error":
                    callback(index)
                return type(self)

        class DescribedSource:
            @property
            def __class__(self):
                callback(index)
                raise RuntimeError("stopped after source type probe")

        class Text(str):
            def __str__(self):
                callback(index)
                return super().__str__()

        class Sequence:
            def __len__(self):
                return 1

            def __iter__(self):
                return (self[item] for item in range(len(self)))

            def __getitem__(self, item):
                if item:
                    raise IndexError(item)
                callback(index)
                return "x"

        class Expressions:
            def __iter__(self):
                callback(index)
                yield vane.FunctionExpression("sum", vane.col("x"))

        class Mapping(MappingABC):
            def __len__(self):
                return 1

            def __iter__(self):
                return iter(self.keys())

            def keys(self):
                callback(index)
                return ["x"]

            def __getitem__(self, key):
                return vane.col("x") + 1

        class WrongType(type):
            def __str__(self):
                callback(index)
                return "WrongExpression"

        class WrongExpression(metaclass=WrongType):
            pass

        class CPU:
            def __float__(self):
                callback(index)
                return 1.0

        class Arrow:
            @property
            def __arrow_c_stream__(self):
                if entry == "arrow_protocol":
                    callback(index)
                return pa.table({"x": [1]}).__arrow_c_stream__

            def __del__(self):
                if entry.startswith("registered_"):
                    destroy(index)

        con = connections[index]
        rel = relations[index]
        try:
            if entry == "arrow_protocol":
                return con.from_arrow(Arrow())
            if entry == "datasource_type":
                return con.from_datasource(DescribedSource())
            if entry in LIFETIMES and not entry.startswith("registered_"):
                return con.from_datasource(Source(index, entry))
            if entry.startswith("registered_"):
                if configured:
                    # Configured runtimes deliberately reject opaque Arrow
                    # streams; a DataFrame still exercises retained input owners.
                    import pandas as pd

                    class Frame(pd.DataFrame):
                        def __del__(self):
                            destroy(index)

                    con.register("registered_input", Frame({"x": [1]}))
                else:
                    con.register("registered_input", Arrow())
                if entry == "registered_unregister":
                    con.unregister("registered_input")
                elif entry == "registered_replace":
                    con.register("registered_input", pa.table({"x": [2]}))
                else:
                    con.close()
                return None
            if entry == "project":
                return rel.project(Text("x"))
            if entry == "select_types":
                return rel.select_types([Text("INTEGER")])
            if entry == "aggregate":
                return rel.aggregate(Expressions())
            if entry == "repartition":
                return rel.repartition(Text("x"))
            if entry == "join":
                return rel.join(others[index], Sequence())
            if entry == "sort_error":
                return rel.sort(WrongExpression())
            if entry == "statement_error":
                return con.execute(WrongExpression())
            if entry == "replacement_name_error":
                return con.register("invalid_input", InvalidInput())
            if entry == "replacement_type_error":
                candidate = InvalidInput()  # noqa: F841 - native replacement scan
                return con.sql("SELECT * FROM candidate")
            if entry == "parquet_partition":
                return rel.write_parquet(
                    str(Path(directory) / f"part-{index}"), partition_by=[Text("x")], write_partition_columns=True
                )
            if entry == "csv_partition":
                return rel.write_csv(
                    str(Path(directory) / f"part-{index}"), partition_by=[Text("x")], write_partition_columns=True
                )
            if entry == "update_mapping":
                return rel.update(Mapping())
            if entry == "merge_condition":
                return rel.merge_into("target", Sequence(), when_clauses=["WHEN MATCHED THEN DELETE"])
            if entry == "merge_clauses":

                class Clauses(Sequence):
                    def __getitem__(self, item):
                        if item:
                            raise IndexError(item)
                        callback(index)
                        return "WHEN MATCHED THEN DELETE"

                return rel.merge_into("target", "target.x = source.x", when_clauses=Clauses())
            if entry == "create_partition":
                return rel.create("partitioned", partition_by=Sequence())
            if entry == "map_cpu":
                return rel.map(lambda x: x, return_type=vane.sqltypes.INTEGER, cpus=CPU())
            if entry in ("map_batches_cpu", "flat_map_cpu"):
                operation = rel.map_batches if entry == "map_batches_cpu" else rel.flat_map
                return operation(lambda batch: batch, schema={"x": vane.sqltypes.INTEGER}, cpus=CPU())
            raise AssertionError(entry)
        except Exception as error:
            outer_errors[index] = error
            return None

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(run, range(2)))
    assert attempted == rejected == [True, True], (entry, attempted, rejected, outer_errors)
    if entry in LIFETIMES:
        assert destroyed == [True, True], (entry, destroyed)
    if entry.endswith("_error") or action == "propagate":
        assert all(error is not None for error in outer_errors), outer_errors
        if action == "propagate" and not entry.endswith("_cpu") and entry not in {"sort_error", "statement_error"}:
            assert all("Python input callback" in str(error) for error in outer_errors), outer_errors
    elif entry == "datasource_type":
        assert all(isinstance(error, RuntimeError) for error in outer_errors), outer_errors
        assert all("stopped after source type probe" in str(error) for error in outer_errors), outer_errors
    elif entry == "create_partition" and not configured:
        assert all(isinstance(error, vane.CatalogException) for error in outer_errors), outer_errors
        assert all("PARTITIONED BY is not supported" in str(error) for error in outer_errors), outer_errors
    elif configured and entry == "arrow_protocol":
        assert all(isinstance(error, vane.InvalidInputException) for error in outer_errors), outer_errors
        assert all("opaque Arrow" in str(error) for error in outer_errors), outer_errors
    elif configured and entry in {
        "parquet_partition",
        "csv_partition",
        "update_mapping",
        "merge_condition",
        "merge_clauses",
        "create_partition",
    }:
        assert all(isinstance(error, vane.InvalidInputException) for error in outer_errors), outer_errors
        assert all("read-only" in str(error) for error in outer_errors), outer_errors
    else:
        assert outer_errors == [None, None], outer_errors
    results.clear()
    relations.clear()
    others.clear()
    for con in connections:
        if entry != "registered_close":
            assert con.execute("SELECT 7").fetchall() == [(7,)]
            con.close()
    for runtime in runtimes:
        runtime.close()
    del builtins._vane_binding_callback
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    _probe(sys.argv[1], sys.argv[2], sys.argv[3] == "True", sys.argv[4])

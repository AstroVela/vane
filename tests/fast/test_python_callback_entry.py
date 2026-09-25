# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Connection entry from Python callbacks, including before handles exist."""

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("fsspec")
pytest.importorskip("pyarrow")


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize(
    ("phase", "action", "worker"),
    [
        (phase, action, False)
        for phase in ["open", "glob", "info", "modified", "read", "seek", "close"]
        for action in ["close", "execute", "extract"]
    ]
    + [
        ("open", action, False)
        for action in [
            "parent_close",
            "relation",
            "file_open",
            "file_read",
            "file_close",
            "idle",
            "idle_close",
            "idle_bind",
            "other",
            "other_close",
            "connect",
        ]
    ]
    + [("open", action, True) for action in ["close", "parent_close", "execute", "extract", "idle"]],
)
def test_input_callbacks_check_connection_entry(tmp_path, monkeypatch, phase, action, worker, configured):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    script = textwrap.dedent(
        """
        import faulthandler
        import io
        import sys
        import threading
        from datetime import datetime, timezone
        from pathlib import Path
        import fsspec
        import pyarrow as pa
        import pyarrow.parquet as pq
        import vane
        from vane.execution.request_admission import RequestAdmissionLimits

        path, phase, action, worker, configured = sys.argv[1:]
        worker = worker == "True"
        configured = configured == "True"
        faulthandler.dump_traceback_later(15, exit=True)
        owner_thread = threading.get_ident()
        rows = 65536 if worker else 16
        paths = [f"callbackfs://part-{i}.parquet" for i in range(16 if worker else 1)]
        output = pa.BufferOutputStream()
        pq.write_table(pa.table({"x": range(rows)}), output, row_group_size=1024)
        payload = output.getvalue().to_pybytes()
        Path(path).write_bytes(b"abc")
        armed = False
        attempts = []

        def callback(name):
            if not armed or attempts or name != phase:
                return
            current_thread = threading.get_ident()
            if worker and current_thread == owner_thread:
                return
            attempts.append(current_thread)
            try:
                if action == "close":
                    cursor.close()
                elif action == "parent_close":
                    parent.close()
                elif action == "execute":
                    cursor.execute("SELECT 7").fetchall()
                elif action == "extract":
                    cursor.extract_statements("SELECT 7")
                elif action == "relation":
                    bound.project("x + 1")
                elif action == "file_open":
                    vane.File(path).open(connection=cursor)
                elif action == "file_read":
                    reader.read(1)
                elif action == "file_close":
                    reader.close()
                elif action == "idle":
                    assert sibling.execute("SELECT 7").fetchall() == [(7,)]
                elif action == "idle_close":
                    sibling.close()
                elif action == "idle_bind":
                    sibling.sql("SELECT 7")
                elif action == "other":
                    independent.execute("SELECT 7")
                elif action == "other_close":
                    independent.close()
                elif action == "connect":
                    vane.connect()
                else:
                    raise AssertionError(action)
            except vane.InvalidInputException as error:
                assert "Python input callback" in str(error), str(error)
            else:
                raise AssertionError("callback entered a connection API")

        class Reader(io.BytesIO):
            def read(self, size=-1):
                callback("read")
                return super().read(size)
            def seek(self, offset, whence=0):
                callback("seek")
                return super().seek(offset, whence)
            def close(self):
                callback("close")
                return super().close()

        class Filesystem(fsspec.AbstractFileSystem):
            protocol = "callbackfs"
            def _open(self, path, mode="rb", **kwargs):
                callback("open")
                return Reader(payload)
            def info(self, path, **kwargs):
                callback("info")
                return {"name": path, "size": len(payload), "type": "file"}
            def modified(self, path):
                callback("modified")
                return datetime(2026, 1, 1, tzinfo=timezone.utc)
            def glob(self, path, **kwargs):
                callback("glob")
                return paths

        with vane.connect(config={"threads": 4 if worker else 1}) as parent:
            parent.register_filesystem(Filesystem(skip_instance_cache=True))
            if configured:
                runtime = parent.configure_local_runtime(
                    request_limit=RequestAdmissionLimits(2, 1), execution_timeout=5
                )
            with parent.cursor() as cursor, parent.cursor() as sibling, vane.connect() as independent:
                bound = cursor.sql("SELECT 1 AS x")
                reader = vane.File(path).open(connection=cursor)
                sql = "SELECT sum(x) FROM read_parquet('callbackfs://part-*.parquet')"
                # Bind before arming worker cases, so the callback must originate
                # from actual parallel execution instead of main-thread binding.
                relation = cursor.sql(sql) if worker else None
                armed = True
                result = relation.fetchall() if worker else cursor.execute(sql).fetchall()
                armed = False
                assert result == [(len(paths) * rows * (rows - 1) // 2,)]
                assert len(attempts) == 1, (phase, action, worker, attempts)
                if worker:
                    assert attempts[0] != owner_thread
                assert reader.read(3) == b"abc", "rejected callback retired its reader"
                reader.close()
                assert cursor.execute("SELECT 7").fetchall() == [(7,)]
                assert parent.execute("SELECT 8").fetchall() == [(8,)]
                assert sibling.execute("SELECT 9").fetchall() == [(9,)]
                assert independent.execute("SELECT 10").fetchall() == [(10,)]
                if configured:
                    state = runtime.resource_snapshot()["request_admission"]
                    assert state["active_requests"] == state["cleanup_pending_requests"] == 0, state
                    assert not state["draining"], state
        faulthandler.cancel_dump_traceback_later()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(tmp_path / "source.bin"), phase, action, str(worker), str(configured)],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("action", ["execute", "bind", "close", "propagate"])
@pytest.mark.parametrize(
    "phase",
    [
        "csv_read",
        "json_read",
        "csv_text",
        "json_text",
        "csv_path",
        "json_path",
        "csv_columns",
        "json_columns",
        "schema_get",
        "schema_dtype",
        "schema_shape",
        "schema_dimension",
        "schema_type",
        "filesystem_protocol",
        "filesystem_capability",
    ],
)
def test_input_copy_rejects_cross_connection_callbacks(tmp_path, monkeypatch, configured, action, phase):
    """Both imports hold their cursor lock before querying the opposite cursor."""
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    script = textwrap.dedent(
        """
        import builtins
        import faulthandler
        import io
        import sys
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path
        import fsspec
        import vane
        from vane.datasource import DataSource, DataSourceTask

        phase, action, configured, path = sys.argv[1:]
        faulthandler.dump_traceback_later(15, exit=True)
        connections = [vane.connect(), vane.connect()]
        runtimes = []
        if configured == "True":
            from vane.execution.request_admission import RequestAdmissionLimits
            runtimes = [
                c.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1), execution_timeout=2)
                for c in connections
            ]
        barrier = threading.Barrier(2, timeout=5)
        attempts = [0, 0]
        armed = False

        def callback(current, index):
            if not armed or current != phase or attempts[index]:
                return
            attempts[index] += 1
            barrier.wait()
            target = connections[1 - index]
            try:
                if action == "close":
                    target.close()
                elif action == "bind":
                    target.sql("SELECT 42")
                else:
                    target.execute("SELECT 42")
            except vane.InvalidInputException as error:
                assert "Python input callback" in str(error), str(error)
                if action == "propagate":
                    raise
            else:
                raise AssertionError("input copying reentered the opposite connection")

        builtins._vane_copy_callback = callback

        class BinaryReader(io.BytesIO):
            def __init__(self, index, data):
                super().__init__(data)
                self.index = index
            def read(self, *args):
                callback(phase, self.index)
                return super().read(*args)

        class TextReader(io.StringIO):
            def __init__(self, index, data):
                super().__init__(data)
                self.index = index
            def read(self, *args):
                callback(phase, self.index)
                return super().read(*args)

        class InputPath(type(Path())):
            def __str__(self):
                callback(phase, self.index)
                return super().__str__()

        class Filesystem(fsspec.AbstractFileSystem):
            def __init__(self, index, **kwargs):
                self.index = index
                super().__init__(**kwargs)
            @property
            def protocol(self):
                callback("filesystem_protocol", self.index)
                return "callback-copy"
            @property
            def vane_directory_semantics(self):
                callback("filesystem_capability", self.index)
                return False

        class Entry(dict):
            def __init__(self, index, **values):
                super().__init__(values)
                self.index = index
            def get(self, key, default=None):
                import builtins
                if key == "kind":
                    builtins._vane_copy_callback("schema_get", self.index)
                return super().get(key, default)

        class Text(str):
            def __new__(cls, value, index, phase):
                obj = super().__new__(cls, value)
                obj.index, obj.phase = index, phase
                return obj
            def __str__(self):
                import builtins
                builtins._vane_copy_callback(self.phase, self.index)
                return self

        class Shape(list):
            def __init__(self, index):
                super().__init__([1])
                self.index = index
            def __iter__(self):
                import builtins
                builtins._vane_copy_callback("schema_shape", self.index)
                return super().__iter__()

        class Dimension:
            def __init__(self, index):
                self.index = index
            def __index__(self):
                import builtins
                builtins._vane_copy_callback("schema_dimension", self.index)
                return 1

        class Task(DataSourceTask):
            def __init__(self, tensor):
                self.tensor = tensor
            def execute(self):
                import numpy as np
                import pyarrow as pa
                if self.tensor:
                    values = pa.FixedShapeTensorArray.from_numpy_ndarray(np.array([[1.0], [2.0]]))
                else:
                    values = [1, 2]
                yield pa.record_batch({"x": values})

        class Source(DataSource):
            def __init__(self, index, phase):
                self.index, self.phase = index, phase
            @property
            def schema(self):
                if self.phase == "schema_type":
                    return {"x": {"type": Text("BIGINT", self.index, self.phase)}}
                dtype = Text("DOUBLE", self.index, "schema_dtype")
                shape = [Dimension(self.index)] if self.phase == "schema_dimension" else Shape(self.index)
                return {"x": Entry(self.index, kind="tensor", dtype=dtype, shape=shape)}
            def get_tasks(self):
                yield Task(self.phase != "schema_type")

        is_json = phase.startswith("json")
        data = '{"x": 1}\\n{"x": 2}\\n' if is_json else "x\\n1\\n2\\n"
        Path(path).write_text(data)
        filesystems = [Filesystem(index, skip_instance_cache=True) for index in range(2)]

        def create(index):
            connection = connections[index]
            if phase.startswith("filesystem"):
                connection.register_filesystem(filesystems[index])
                return connection.sql("SELECT * FROM (VALUES (1), (2)) t(x)")
            if phase.startswith("schema"):
                return connection.from_datasource(Source(index, phase))
            if phase.endswith("columns"):
                return (connection.read_json if is_json else connection.read_csv)(
                    path, columns={"x": Text("BIGINT", index, phase)}
                )
            if phase.endswith("path"):
                source = InputPath(path)
                source.index = index
            elif phase.endswith("text"):
                source = TextReader(index, data)
            else:
                source = BinaryReader(index, data.encode())
            return (connection.read_json if is_json else connection.read_csv)(source)

        armed = True
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(create, index) for index in range(2)]
            for future in futures:
                try:
                    relation = future.result(timeout=10)
                except vane.InvalidInputException as error:
                    assert action == "propagate", str(error)
                    assert "Python input callback" in str(error), str(error)
                else:
                    assert action != "propagate"
                    rows = relation.fetchall()
                    expected = [((1.0,),), ((2.0,),)] if phase.startswith("schema") and phase != "schema_type" else [(1,), (2,)]
                    assert rows == expected, rows
        assert attempts == [1, 1], attempts
        for connection in connections:
            assert connection.execute("SELECT 7").fetchall() == [(7,)]
        for runtime in runtimes:
            state = runtime.resource_snapshot()["request_admission"]
            assert not state["draining"], state
            assert state["active_requests"] == state["cleanup_pending_requests"] == 0, state
        for connection in connections:
            connection.close()
        del builtins._vane_copy_callback
        faulthandler.cancel_dump_traceback_later()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, phase, action, str(configured), str(tmp_path / "input.data")],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    ("source", "configured"),
    [(source, configured) for source in ["pandas", "numpy", "datasource"] for configured in [False, True]]
    + [("arrow", False)],
)
@pytest.mark.parametrize("action", ["execute", "bind", "close"])
def test_input_kinds_reject_idle_sibling_entry(monkeypatch, source, configured, action):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    script = textwrap.dedent(
        """
        import builtins
        import faulthandler
        import sys
        import numpy as np
        import pandas as pd
        import pyarrow as pa
        import vane
        from vane.datasource import DataSource, DataSourceTask
        from vane.execution.request_admission import RequestAdmissionLimits

        source, configured, action = sys.argv[1:]
        faulthandler.dump_traceback_later(15, exit=True)
        armed = False
        attempted = []

        def callback():
            if not armed or attempted:
                return
            attempted.append(action)
            try:
                if action == "execute":
                    sibling.execute("SELECT 7")
                elif action == "bind":
                    prebound.project("x + 1")
                else:
                    sibling.close()
            except vane.InvalidInputException as error:
                assert "Python input callback" in str(error), str(error)
            else:
                raise AssertionError("input callback entered an idle sibling")

        class Value:
            def __str__(self):
                callback()
                return "x"

        class Task(DataSourceTask):
            def execute(self):
                import builtins
                import pyarrow as pa
                builtins._vane_callback_entry_probe()
                return iter([pa.record_batch({"x": ["x"]})])

        class Source(DataSource):
            @property
            def schema(self):
                return {"x": "VARCHAR"}
            def get_tasks(self):
                yield Task()

        def batches():
            callback()
            yield pa.record_batch({"x": ["x"]})

        builtins._vane_callback_entry_probe = callback
        with vane.connect(config={"threads": 2}) as parent:
            if configured == "True":
                runtime = parent.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1))
            with parent.cursor() as cursor, parent.cursor() as sibling:
                prebound = sibling.sql("SELECT 7 AS x")
                if source == "datasource":
                    relation = cursor.from_datasource(Source())
                else:
                    if source == "arrow":
                        data = pa.RecordBatchReader.from_batches(pa.schema([("x", pa.string())]), batches())
                    else:
                        data = {"x": np.array([Value()], dtype=object)}
                        if source == "pandas":
                            data = pd.DataFrame(data)
                    cursor.register("items", data)
                    relation = cursor.sql("SELECT * FROM items")
                armed = True
                assert relation.fetchall() == [("x",)]
                armed = False
                assert attempted == [action]
                assert sibling.execute("SELECT 7").fetchall() == [(7,)]
                assert cursor.execute("SELECT 8").fetchall() == [(8,)]
                if configured == "True":
                    state = runtime.resource_snapshot()["request_admission"]
                    assert state["active_requests"] == state["cleanup_pending_requests"] == 0, state
                    assert not state["draining"], state
        del builtins._vane_callback_entry_probe
        faulthandler.cancel_dump_traceback_later()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, source, str(configured), action],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("materialized", [False, True])
def test_callback_fetch_distinguishes_live_and_materialized_readers(monkeypatch, materialized):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    script = textwrap.dedent(
        """
        import faulthandler
        import sys
        import numpy as np
        import pyarrow as pa
        import vane

        faulthandler.dump_traceback_later(15, exit=True)
        materialized = sys.argv[1] == "True"
        armed = False
        attempted = []
        class Value:
            def __str__(self):
                if armed and not attempted:
                    attempted.append(True)
                    try:
                        result = reader.read_all()
                    except (pa.ArrowInvalid, OSError) as error:
                        assert not materialized, str(error)
                        assert "Python input callback" in str(error), str(error)
                    else:
                        assert materialized, "callback executed a live native result"
                        assert result.num_rows == 25000
                return "x"

        with vane.connect(config={"threads": 1}) as parent, parent.cursor() as sibling:
            relation = sibling.sql("SELECT i AS x FROM range(25000) r(i)")
            if materialized:
                relation.execute()
            reader = relation.to_arrow_reader(batch_size=128)
            parent.register("items", {"x": np.array([Value()], dtype=object)})
            query = parent.sql("SELECT * FROM items")
            armed = True
            assert query.fetchall() == [("x",)]
            armed = False
            assert attempted == [True]
            reader.close()
            assert sibling.execute("SELECT 7").fetchall() == [(7,)]
        faulthandler.cancel_dump_traceback_later()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(materialized)], capture_output=True, text=True, timeout=25
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "source",
    [
        "pandas_copy",
        "pandas_keys",
        "numpy_keys",
        "datasource_schema",
        "datasource_schema_items",
        "datasource_tasks",
        "datasource_pickle",
        "datasource_task_pickle",
    ],
)
@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("target", ["cursor", "sibling"])
@pytest.mark.parametrize("action", ["execute", "close"])
def test_input_metadata_callbacks_reject_connection_entry(monkeypatch, source, configured, target, action):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    script = textwrap.dedent(
        """
        import builtins
        import faulthandler
        import sys
        import numpy as np
        import pandas as pd
        import pyarrow as pa
        import vane
        from vane.datasource import DataSource, DataSourceTask
        from vane.execution.request_admission import RequestAdmissionLimits

        source, configured, target, action = sys.argv[1:]
        faulthandler.dump_traceback_later(15, exit=True)
        armed = False
        attempted = []
        def callback(phase):
            if not armed or attempted or phase != source:
                return
            attempted.append(phase)
            connection = cursor if target == "cursor" else sibling
            try:
                if action == "execute":
                    connection.execute("SELECT 7")
                else:
                    connection.close()
            except vane.InvalidInputException as error:
                assert "Python input callback" in str(error), str(error)
            else:
                raise AssertionError("input metadata callback entered a connection")

        class Frame(pd.DataFrame):
            @property
            def _constructor(self):
                return Frame
            def copy(self, *args, **kwargs):
                callback("pandas_copy")
                return super().copy(*args, **kwargs)
            def keys(self):
                callback("pandas_keys")
                return super().keys()

        class Columns(dict):
            def keys(self):
                callback("numpy_keys")
                return super().keys()

        class Task(DataSourceTask):
            def __getstate__(self):
                import builtins
                builtins._vane_metadata_callback_probe("datasource_task_pickle")
                return {}
            def execute(self):
                import pyarrow as pa
                return iter([pa.record_batch({"x": [1]})])

        class Schema(dict):
            def items(self):
                import builtins
                builtins._vane_metadata_callback_probe("datasource_schema_items")
                return super().items()

        class Source(DataSource):
            def __getstate__(self):
                import builtins
                builtins._vane_metadata_callback_probe("datasource_pickle")
                return {}
            @property
            def schema(self):
                import builtins
                builtins._vane_metadata_callback_probe("datasource_schema")
                return Schema(x="BIGINT")
            def get_tasks(self):
                import builtins
                builtins._vane_metadata_callback_probe("datasource_tasks")
                yield Task()

        builtins._vane_metadata_callback_probe = callback
        with vane.connect(config={"threads": 2}) as parent:
            if configured == "True":
                runtime = parent.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1))
            with parent.cursor() as cursor, parent.cursor() as sibling:
                data = Frame({"x": [1]}) if source.startswith("pandas") else Columns(x=np.array([1]))
                armed = True
                if source.startswith("datasource"):
                    relation = cursor.from_datasource(Source())
                else:
                    cursor.register("items", data)
                    relation = cursor.sql("SELECT * FROM items")
                armed = False
                assert attempted == [source], attempted
                assert relation.fetchall() == [(1,)]
                assert cursor.execute("SELECT 7").fetchall() == [(7,)]
                assert sibling.execute("SELECT 8").fetchall() == [(8,)]
                if configured == "True":
                    state = runtime.resource_snapshot()["request_admission"]
                    assert not state["draining"], state
                    assert state["active_requests"] == 0, state
        del builtins._vane_metadata_callback_probe
        faulthandler.cancel_dump_traceback_later()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, source, str(configured), target, action],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

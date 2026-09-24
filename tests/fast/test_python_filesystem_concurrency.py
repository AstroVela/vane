# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Native filesystem concurrency checks run in bounded child processes."""

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("fsspec")


@pytest.mark.parametrize("phase", ["seek", "read"])
@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("fail_once", [False, True])
def test_shared_handle_position_survives_gil_release(tmp_path, monkeypatch, phase, configured, fail_once):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    script = textwrap.dedent(
        """
        import faulthandler
        import io
        import sys
        import threading
        import time
        from pathlib import Path
        import fsspec
        import vane
        from vane.execution.request_admission import RequestAdmissionLimits

        path, phase, configured, fail_once = sys.argv[1:]
        faulthandler.dump_traceback_later(15, exit=True)
        query = "SELECT sum(x % 97) FROM source.t"
        with vane.connect(path) as source:
            source.execute("CREATE TABLE t AS SELECT hash(i) AS x FROM range(1000000) r(i)")
            expected = source.execute("SELECT sum(x % 97) FROM t").fetchall()
        payload = Path(path).read_bytes()
        armed = False
        should_fail = fail_once == "True"
        threads = set()
        scanned_handles = set()

        class Reader(io.BytesIO):
            def checkpoint(self):
                global should_fail
                if not armed:
                    return
                threads.add(threading.get_ident())
                scanned_handles.add(id(self))
                offset = super().tell()
                if should_fail:
                    should_fail = False
                    raise RuntimeError("injected file callback failure")
                # Python I/O and callbacks may release the GIL while other
                # native scan workers read blocks from this same handle.
                time.sleep(0.005)
                assert super().tell() == offset, "shared file position changed during callback"

            def seek(self, offset, whence=0):
                result = super().seek(offset, whence)
                if phase == "seek":
                    self.checkpoint()
                return result

            def read(self, size=-1):
                if phase == "read":
                    self.checkpoint()
                return super().read(size)

        class Filesystem(fsspec.AbstractFileSystem):
            protocol = "http"

            def info(self, path, **kwargs):
                return {"name": path, "size": len(payload), "type": "file"}

            def _open(self, path, mode="rb", **kwargs):
                return Reader(payload)

        with vane.connect(config={"threads": 4}) as parent:
            parent.register_filesystem(Filesystem(skip_instance_cache=True))
            parent.execute("ATTACH 'http://test.invalid/source.db' AS source (READ_ONLY)")
            if configured == "True":
                parent.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1), execution_timeout=5)
            with parent.cursor() as cursor:
                armed = True
                if should_fail:
                    try:
                        cursor.execute(query).fetchall()
                    except vane.Error as error:
                        assert "injected file callback failure" in str(error), str(error)
                    else:
                        raise AssertionError("injected callback failure was swallowed")
                    threads.clear()
                    scanned_handles.clear()
                assert cursor.execute(query).fetchall() == expected
                assert len(threads) >= 2, "the shared-handle scan did not run in parallel"
                assert len(scanned_handles) == 1, "the scan must share one Python file object"
                assert cursor.execute("SELECT 7").fetchall() == [(7,)]
        faulthandler.cancel_dump_traceback_later()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(tmp_path / "source.db"), phase, str(configured), str(fail_once)],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_different_file_handles_can_read_concurrently(tmp_path, monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    script = textwrap.dedent(
        """
        import faulthandler
        import io
        import sys
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path
        import fsspec
        import vane

        path = sys.argv[1]
        faulthandler.dump_traceback_later(15, exit=True)
        with vane.connect(path) as source:
            source.execute("CREATE TABLE t AS SELECT hash(i) AS x FROM range(1000000) r(i)")
            expected = source.execute("SELECT sum(x % 97) FROM t").fetchall()
        payload = Path(path).read_bytes()
        armed = False
        rendezvous = threading.Barrier(2, timeout=5)
        entered = set()

        class Reader(io.BytesIO):
            def read(self, size=-1):
                if armed and id(self) not in entered:
                    entered.add(id(self))
                    # Holding a filesystem-wide lock would prevent the other
                    # query from reaching its independent handle's callback.
                    rendezvous.wait()
                return super().read(size)

        class Filesystem(fsspec.AbstractFileSystem):
            protocol = "http"

            def info(self, path, **kwargs):
                return {"name": path, "size": len(payload), "type": "file"}

            def _open(self, path, mode="rb", **kwargs):
                return Reader(payload)

        with vane.connect(config={"threads": 1}) as parent:
            parent.register_filesystem(Filesystem(skip_instance_cache=True))
            for name in ["a", "b"]:
                parent.execute(f"ATTACH 'http://test.invalid/{name}.db' AS {name} (READ_ONLY)")
            with parent.cursor() as first, parent.cursor() as second:
                armed = True
                def execute(cursor, name):
                    return cursor.execute(f"SELECT sum(x % 97) FROM {name}.t").fetchall()
                with ThreadPoolExecutor(max_workers=2) as workers:
                    queries = [workers.submit(execute, first, "a"), workers.submit(execute, second, "b")]
                    for query in queries:
                        assert query.result(timeout=10) == expected
                assert len(entered) == 2
        faulthandler.cancel_dump_traceback_later()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(tmp_path / "source.db")],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_same_handle_reentry_fails_without_deadlock_or_position_change(tmp_path, monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    script = textwrap.dedent(
        """
        import faulthandler
        import io
        import sys
        from pathlib import Path
        import fsspec
        import vane

        path = sys.argv[1]
        faulthandler.dump_traceback_later(15, exit=True)
        with vane.connect(path) as source:
            for name in ["a", "b"]:
                source.execute(f"CREATE TABLE {name} AS SELECT hash(i) AS x FROM range(100000) r(i)")
            expected = source.execute("SELECT sum(x % 97) FROM a").fetchall()
        payload = Path(path).read_bytes()
        armed = False
        attempted = False
        rejected = []

        class Reader(io.BytesIO):
            def read(self, size=-1):
                global attempted
                if armed and not attempted:
                    attempted = True
                    try:
                        sibling.execute("SELECT sum(x % 97) FROM source.b").fetchall()
                    except vane.Error as error:
                        assert "reentrant I/O" in str(error), str(error)
                        rejected.append(str(error))
                    else:
                        raise AssertionError("same-handle I/O reentry was not rejected")
                return super().read(size)

        class Filesystem(fsspec.AbstractFileSystem):
            protocol = "http"

            def info(self, path, **kwargs):
                return {"name": path, "size": len(payload), "type": "file"}

            def _open(self, path, mode="rb", **kwargs):
                return Reader(payload)

        with vane.connect(config={"threads": 1}) as parent:
            parent.register_filesystem(Filesystem(skip_instance_cache=True))
            parent.execute("ATTACH 'http://test.invalid/source.db' AS source (READ_ONLY)")
            with parent.cursor() as cursor, parent.cursor() as sibling:
                armed = True
                assert cursor.execute("SELECT sum(x % 97) FROM source.a").fetchall() == expected
                assert len(rejected) == 1
                assert sibling.execute("SELECT sum(x % 97) FROM source.b").fetchall() == expected
        faulthandler.cancel_dump_traceback_later()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(tmp_path / "source.db")],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("phase", ["read", "seek"])
@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("target", ["sibling", "owner", "self", "ancestor", "idle", "control"])
def test_file_callback_close_checks_active_siblings(tmp_path, monkeypatch, phase, configured, target):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    script = textwrap.dedent(
        """
        import faulthandler
        import io
        import sys
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
        from pathlib import Path
        import fsspec
        import vane
        from vane.execution.request_admission import RequestAdmissionLimits, RequestCancelled

        path, phase, configured, target = sys.argv[1:]
        configured = configured == "True"
        concurrent = target in {"sibling", "owner", "control"}
        faulthandler.dump_traceback_later(15, exit=True)
        with vane.connect(path) as source:
            for name in ["a", "b"]:
                source.execute(f"CREATE TABLE {name} AS SELECT hash(i) AS x FROM range(1000000) r(i)")
            expected = source.execute("SELECT sum(x % 97) FROM a").fetchall()
        payload = Path(path).read_bytes()
        armed = False
        attempted = False
        entered = threading.Event()
        release = threading.Event()
        rejected = []

        def wait_for_sibling():
            # Wait for native execution under the sibling's connection lock,
            # not just for its Python thread to have called execute().
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if configured:
                    if runtime.resource_snapshot()["request_admission"]["active_requests"] == 2:
                        return
                elif sibling.query_progress() >= 0:
                    return
                time.sleep(0.001)
            raise AssertionError("sibling did not enter native execution")

        def checkpoint():
            global attempted
            if not armed or attempted:
                return
            attempted = True
            entered.set()
            if target == "control":
                assert release.wait(5)
                return
            if concurrent:
                wait_for_sibling()
            closing = {
                "sibling": sibling, "owner": owner, "self": cursor,
                "ancestor": parent, "idle": sibling,
            }[target]
            try:
                closing.close()
            except vane.InvalidInputException as error:
                assert target != "idle", str(error)
                message = str(error)
                assert "cannot close a busy cursor" in message or "cannot close a cursor reentrantly" in message
                rejected.append(message)
            else:
                assert target == "idle", "callback closed an active cursor"

        class Reader(io.BytesIO):
            def read(self, size=-1):
                if phase == "read":
                    checkpoint()
                return super().read(size)

            def seek(self, offset, whence=0):
                result = super().seek(offset, whence)
                if phase == "seek":
                    checkpoint()
                return result

        class Filesystem(fsspec.AbstractFileSystem):
            protocol = "http"

            def info(self, path, **kwargs):
                return {"name": path, "size": len(payload), "type": "file"}

            def _open(self, path, mode="rb", **kwargs):
                return Reader(payload)

        with vane.connect(config={"threads": 1}) as parent:
            parent.register_filesystem(Filesystem(skip_instance_cache=True))
            parent.execute("ATTACH 'http://test.invalid/source.db' AS source (READ_ONLY)")
            if configured:
                runtime = parent.configure_local_runtime(
                    request_limit=RequestAdmissionLimits(2, 2), execution_timeout=2
                )
            with parent.cursor() as cursor, parent.cursor() as owner, owner.cursor() as sibling:
                if not configured:
                    sibling.execute("SET enable_progress_bar=true")
                    sibling.execute("SET enable_progress_bar_print=false")
                # Bind both tables before arming callbacks, so the sibling's
                # progress signal precedes its first blocked data read.
                first_relation = cursor.sql("SELECT sum(x % 97) FROM source.a")
                second_relation = sibling.sql("SELECT sum(x % 97) FROM source.b")
                armed = True
                with ThreadPoolExecutor(max_workers=3) as workers:
                    first = workers.submit(first_relation.fetchall)
                    assert entered.wait(5)
                    second = workers.submit(second_relation.fetchall) if concurrent else None
                    closing = None
                    if target == "control":
                        try:
                            wait_for_sibling()
                            closing = workers.submit(sibling.close)
                            try:
                                closing.result(timeout=0.1)
                            except FutureTimeoutError:
                                pass
                            # Cancellation may finish before the sibling reaches
                            # its read, or wait until this callback releases I/O.
                        finally:
                            release.set()
                    assert first.result(timeout=5) == expected
                    if second is not None:
                        try:
                            assert second.result(timeout=5) == expected
                        except RequestCancelled:
                            assert target == "control" and configured
                    if closing is not None:
                        closing.result(timeout=5)
                assert len(rejected) == (0 if target in {"idle", "control"} else 1), rejected
                assert cursor.execute("SELECT 7").fetchall() == [(7,)]
                assert owner.execute("SELECT 8").fetchall() == [(8,)]
                if target not in {"idle", "control"}:
                    assert sibling.execute("SELECT 9").fetchall() == [(9,)]
                if configured:
                    state = runtime.resource_snapshot()["request_admission"]
                    assert state["active_requests"] == state["cleanup_pending_requests"] == 0, state
        faulthandler.cancel_dump_traceback_later()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(tmp_path / "source.db"), phase, str(configured), target],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    ("delivery", "operation"),
    [
        (delivery, operation)
        for delivery in ["rows", "relation", "numpy", "arrow_table", "arrow_reader", "relation_reader"]
        for operation in ["close", "extract"]
    ]
    + [
        ("rows", operation)
        for operation in [
            "owner_close",
            "execute",
            "table",
            "project",
            "length",
            "file_open",
            "file_read",
            "uncaught_extract",
            "idle",
        ]
    ],
)
def test_file_callback_checks_streaming_and_cursor_operations(tmp_path, monkeypatch, delivery, operation):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    script = textwrap.dedent(
        """
        import faulthandler
        import io
        import sys
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path
        import fsspec
        import vane

        path, delivery, operation = sys.argv[1:]
        faulthandler.dump_traceback_later(15, exit=True)
        with vane.connect(path) as source:
            for name in ["a", "b"]:
                source.execute(f"CREATE TABLE {name} AS SELECT hash(i) AS x FROM range(1000000) r(i)")
            expected = source.execute("SELECT sum(x % 97) FROM a").fetchone()[0]
        payload = Path(path).read_bytes()
        armed = False
        attempted = False
        entered = threading.Event()
        fetching = threading.Event()
        rejected = []

        def callback():
            global attempted
            if not armed or attempted:
                return
            attempted = True
            entered.set()
            assert fetching.wait(5)
            # Streaming execution is already open before the callback is armed.
            # Give its fetch thread time to drain buffered chunks and reach the
            # shared file mutex, without another callback moving the offset.
            time.sleep(0.2)
            assert not second.done(), "the streaming query did not need another file read"
            if operation == "uncaught_extract":
                sibling.extract_statements("SELECT 7")
                raise AssertionError("busy cursor call was not rejected")
            try:
                if operation == "close":
                    sibling.close()
                elif operation == "owner_close":
                    owner.close()
                elif operation == "extract":
                    sibling.extract_statements("SELECT 7")
                elif operation == "execute":
                    sibling.execute("SELECT 7")
                elif operation == "table":
                    sibling.table("source.b")
                elif operation == "project":
                    sibling_relation.project("x + 1")
                elif operation == "length":
                    len(sibling_relation)
                elif operation == "file_open":
                    vane.File(path).open(connection=sibling)
                elif operation == "file_read":
                    file_reader.read(1)
                else:
                    assert idle.extract_statements("SELECT 7")
                    assert idle.execute("SELECT 7").fetchall() == [(7,)]
            except vane.InvalidInputException as error:
                assert operation != "idle", str(error)
                assert "busy cursor" in str(error), str(error)
                rejected.append(str(error))
            else:
                assert operation == "idle", "callback was allowed to enter a busy cursor"

        class Reader(io.BytesIO):
            def read(self, size=-1):
                callback()
                return super().read(size)

        class Filesystem(fsspec.AbstractFileSystem):
            protocol = "http"

            def info(self, path, **kwargs):
                return {"name": path, "size": len(payload), "type": "file"}

            def _open(self, path, mode="rb", **kwargs):
                return Reader(payload)

        with vane.connect(config={"threads": 1}) as parent:
            parent.register_filesystem(Filesystem(skip_instance_cache=True))
            parent.execute("ATTACH 'http://test.invalid/source.db' AS source (READ_ONLY)")
            with parent.cursor() as cursor, parent.cursor() as owner, owner.cursor() as sibling, parent.cursor() as idle:
                first_relation = cursor.sql("SELECT sum(x % 97) FROM source.a")
                sibling_relation = sibling.sql("SELECT x FROM source.b")
                file_reader = vane.File(path).open(connection=sibling)
                prefix = []
                if delivery == "relation":
                    prefix = [sibling_relation.fetchone()[0]]
                    consume = sibling_relation.fetchall
                elif delivery == "relation_reader":
                    reader = sibling_relation.to_arrow_reader(batch_size=2048)
                    consume = reader.read_all
                else:
                    sibling.execute("SELECT x FROM source.b")
                    if delivery == "rows":
                        prefix = [sibling.fetchone()[0]]
                        consume = sibling.fetchall
                    elif delivery == "numpy":
                        consume = sibling.fetchnumpy
                    elif delivery == "arrow_table":
                        consume = sibling.to_arrow_table
                    else:
                        reader = sibling.to_arrow_reader(batch_size=2048)
                        consume = reader.read_all
                armed = True

                def fetch_sibling():
                    fetching.set()
                    return consume()

                with ThreadPoolExecutor(max_workers=2) as workers:
                    first = workers.submit(first_relation.fetchall)
                    assert entered.wait(5)
                    second = workers.submit(fetch_sibling)
                    try:
                        assert first.result(timeout=5) == [(expected,)]
                    except vane.Error as error:
                        assert operation == "uncaught_extract", str(error)
                        assert "busy cursor" in str(error), str(error)
                        rejected.append(str(error))
                    else:
                        assert operation != "uncaught_extract", "callback failure was swallowed"
                    result = second.result(timeout=5)
                if delivery in {"rows", "relation"}:
                    values = [row[0] for row in result]
                elif delivery == "numpy":
                    values = result["x"].tolist()
                else:
                    values = result.column("x").to_pylist()
                assert len(prefix) + len(values) == 1000000
                assert sum(value % 97 for value in prefix + values) == expected
                assert len(rejected) == (0 if operation == "idle" else 1), rejected
                file_reader.close()
                assert cursor.execute("SELECT 8").fetchall() == [(8,)]
                assert sibling.execute("SELECT 9").fetchall() == [(9,)]
        faulthandler.cancel_dump_traceback_later()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(tmp_path / "source.db"), delivery, operation],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_interrupt_is_preserved_while_waiting_for_cursor_entry(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    script = textwrap.dedent(
        """
        import faulthandler
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor
        import pyarrow as pa
        import vane

        faulthandler.dump_traceback_later(15, exit=True)
        entered = threading.Event()
        waiting = threading.Event()
        release = threading.Event()
        batch = pa.record_batch({"x": [1, 2]})

        def batches():
            entered.set()
            assert release.wait(5)
            yield batch

        with vane.connect(config={"threads": 1}) as connection:
            connection.register("source", pa.RecordBatchReader.from_batches(batch.schema, batches()))
            def next_query():
                waiting.set()
                return connection.execute("SELECT 7").fetchall()

            with ThreadPoolExecutor(max_workers=2) as workers:
                first = workers.submit(connection.execute, "SELECT sum(x) FROM source")
                assert entered.wait(5)
                second = workers.submit(next_query)
                assert waiting.wait(5)
                time.sleep(0.1)
                assert not second.done()
                connection.interrupt()
                release.set()
                for query in [first, second]:
                    try:
                        query.result(timeout=5)
                    except vane.InterruptException:
                        pass
                    else:
                        raise AssertionError("interrupt was lost while waiting for the cursor lock")
            assert connection.execute("SELECT 9").fetchall() == [(9,)]
        faulthandler.cancel_dump_traceback_later()
        """
    )
    completed = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=25)
    assert completed.returncode == 0, completed.stdout + completed.stderr

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

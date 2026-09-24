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
        handles = []

        class Reader(io.BytesIO):
            def checkpoint(self):
                global should_fail
                if not armed:
                    return
                threads.add(threading.get_ident())
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
                reader = Reader(payload)
                handles.append(reader)
                return reader

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
                assert cursor.execute(query).fetchall() == expected
                assert len(threads) >= 2, "the shared-handle scan did not run in parallel"
                assert len(handles) == 1, "the scan must share one Python file object"
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

# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import io
import os
import sys
import threading
from array import array
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import vane
import vane._file as file_module


def _start_object_server(payload):
    class ObjectHandler(BaseHTTPRequestHandler):
        block_reads = False
        read_started = threading.Event()
        release_read = threading.Event()
        requests = []
        requests_lock = threading.Lock()

        def _record(self):
            with type(self).requests_lock:
                type(self).requests.append(
                    {
                        "authorization": self.headers.get("Authorization"),
                        "path": self.path,
                        "range": self.headers.get("Range"),
                    }
                )

        def _send_object(self, include_body):
            self._record()
            if self.path.split("?", 1)[0] != "/bucket/object.bin":
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            body = payload
            status = 200
            content_range = None
            range_header = self.headers.get("Range")
            if range_header is not None:
                unit, separator, bounds = range_header.partition("=")
                start_text, dash, end_text = bounds.partition("-")
                if (
                    unit != "bytes"
                    or not separator
                    or not dash
                    or not start_text.isdigit()
                    or (end_text and not end_text.isdigit())
                ):
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{len(payload)}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                start = int(start_text)
                end = len(payload) - 1 if not end_text else min(int(end_text), len(payload) - 1)
                if start >= len(payload) or end < start:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{len(payload)}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                status = 206
                body = payload[start : end + 1]
                content_range = f"bytes {start}-{end}/{len(payload)}"

            self.send_response(status)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "application/octet-stream")
            if content_range is not None:
                self.send_header("Content-Range", content_range)
            self.end_headers()
            if include_body:
                if type(self).block_reads:
                    type(self).read_started.set()
                    type(self).release_read.wait(timeout=10)
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def do_HEAD(self):
            self._send_object(False)

        def do_GET(self):
            self._send_object(True)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), ObjectHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, ObjectHandler


def test_file_reader_is_a_strict_read_only_logical_view(tmp_path):
    path = tmp_path / "window.bin"
    path.write_bytes(b"0123456789abcdef")
    reader = vane.File(str(path), position=2, size=8).open(buffer_size=3)

    assert isinstance(reader, vane.VaneFileReader)
    assert isinstance(reader, io.RawIOBase)
    assert str(reader) == str(path)
    assert repr(reader) == f"VaneFileReader(url={str(path)!r}, closed=False)"
    assert reader.readable()
    assert not reader.writable()
    assert reader.seekable()
    assert not reader.isatty()
    assert reader.size() == 8
    assert reader.tell() == 0
    assert reader.read(2) == b"23"

    target = bytearray(3)
    assert reader.readinto(target) == 3
    assert target == b"456"
    assert reader.tell() == 5
    assert reader.seek(-2, io.SEEK_CUR) == 3
    assert reader.read(2) == b"56"
    assert reader.seek(-2, io.SEEK_END) == 6
    assert reader.read() == b"89"
    assert reader.read() == b""
    assert reader.seek(100) == 100
    assert reader.read() == b""
    assert reader.seek(2**63 - 1) == 2**63 - 1
    with pytest.raises(ValueError, match="exceeds signed 64-bit range"):
        reader.seek(1, io.SEEK_CUR)
    with pytest.raises(ValueError, match="invalid whence"):
        reader.seek(0, 99)
    assert reader.seek(0) == 0
    assert reader.read() == b"23456789"
    with pytest.raises(io.UnsupportedOperation):
        reader.write(b"not allowed")

    reader.close()
    reader.close()
    assert reader.closed
    assert repr(reader) == f"VaneFileReader(url={str(path)!r}, closed=True)"
    for operation in (
        lambda: reader.read(),
        lambda: reader.write(b"not allowed"),
        lambda: reader.seek(0),
        reader.tell,
        reader.size,
        reader.guess_mime_type,
        reader.readable,
        reader.writable,
        reader.seekable,
        reader.isatty,
        reader.__enter__,
    ):
        with pytest.raises(ValueError, match="closed VaneFileReader"):
            operation()


def test_file_reader_readinto_supports_writable_contiguous_buffers(tmp_path):
    path = tmp_path / "readinto.bin"
    path.write_bytes(bytes(range(16)))

    with vane.File(str(path)).open(buffer_size=4) as reader:
        words = array("I", [0, 0])
        assert reader.readinto(words) == words.itemsize * len(words)
        assert memoryview(words).cast("B").tolist() == list(range(8))

        with pytest.raises(TypeError, match="writable bytes-like"):
            reader.readinto(b"readonly")
        noncontiguous = memoryview(bytearray(8))[::2]
        try:
            with pytest.raises(TypeError, match="contiguous writable"):
                reader.readinto(noncontiguous)
        finally:
            noncontiguous.release()


def test_file_reader_mime_detection_does_not_move_cursor(tmp_path):
    path = tmp_path / "image.data"
    path.write_bytes(b"prefix" + b"\x89PNG\r\n\x1a\n" + b"payload")

    with vane.File(str(path), position=6, size=15).open(buffer_size=4) as reader:
        assert reader.seek(3) == 3
        assert reader.guess_mime_type() == "image/png"
        assert reader.tell() == 3
        assert reader.read(5) == b"G\r\n\x1a\n"


def test_file_reader_zero_length_and_out_of_bounds_windows(tmp_path):
    path = tmp_path / "bounds.bin"
    path.write_bytes(b"value")

    with vane.File(str(path), position=5, size=0).open() as reader:
        assert reader.size() == 0
        assert reader.read() == b""
        assert reader.seek(7) == 7
        assert reader.read(1) == b""
        with pytest.raises(ValueError, match="negative seek"):
            reader.seek(-1)

    with pytest.raises(vane.IOException, match="exceeds the backing object size"):
        vane.File(str(path), position=4, size=2).open()


@pytest.mark.skipif(os.name == "nt", reason="open-file replacement semantics differ on Windows")
def test_file_reader_rejects_short_backing_object_without_poisoning_buffer(tmp_path):
    path = tmp_path / "short.bin"
    path.write_bytes(b"abcdef")

    with vane.File(str(path)).open(buffer_size=4) as reader:
        assert reader.read(3) == b"abc"
        path.write_bytes(b"abc")
        with pytest.raises(vane.IOException, match="read enough bytes"):
            reader.read(2)
        assert reader.tell() == 3

        path.write_bytes(b"abcdef")
        assert reader.read(2) == b"de"


@pytest.mark.parametrize("buffer_size", [0, -1])
def test_file_open_rejects_nonpositive_buffer_size(buffer_size):
    with pytest.raises(ValueError, match="greater than zero"):
        vane.File("missing").open(buffer_size=buffer_size)


@pytest.mark.parametrize("buffer_size", [True, 1.5, "1024"])
def test_file_open_rejects_noninteger_buffer_size(buffer_size):
    with pytest.raises(TypeError, match="buffer_size must be int or None"):
        vane.File("missing").open(buffer_size=buffer_size)


def test_file_open_rejects_oversized_buffer_before_io():
    with pytest.raises(OverflowError, match="Py_ssize_t"):
        vane.File("missing").open(buffer_size=sys.maxsize + 1)


@pytest.mark.parametrize("buffer_size", [None, True, 1.5, "1024"])
def test_file_to_tempfile_requires_integer_buffer_size(buffer_size):
    with pytest.raises(TypeError, match="buffer_size must be int"):
        vane.File("missing").to_tempfile(buffer_size=buffer_size)


def test_file_reader_uses_explicit_connection_and_retains_open_context(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "value.bin").write_bytes(b"connection-scoped")
    connection = vane.connect("")
    connection.execute("SET home_directory = ?", [str(home)])

    reader = vane.File("~/value.bin").open(connection=connection)
    connection.close()
    try:
        assert reader.read() == b"connection-scoped"
    finally:
        reader.close()


def test_file_reader_io_does_not_commit_or_invalidate_explicit_transaction(tmp_path):
    path = tmp_path / "transaction.bin"
    path.write_bytes(b"value")
    connection = vane.connect("")
    try:
        connection.execute("CREATE TABLE reader_transaction(value INTEGER)")
        connection.execute("BEGIN")
        connection.execute("INSERT INTO reader_transaction VALUES (42)")

        with vane.File(str(path)).open(connection=connection) as reader:
            assert reader.read() == b"value"
        with pytest.raises(vane.IOException):
            vane.File(str(tmp_path / "missing.bin")).open(connection=connection)

        assert connection.execute("SELECT * FROM reader_transaction").fetchall() == [(42,)]
        connection.execute("ROLLBACK")
        assert connection.execute("SELECT * FROM reader_transaction").fetchall() == []
    finally:
        connection.close()


def test_open_file_binary_and_text_modes(tmp_path):
    path = tmp_path / "text.txt"
    path.write_bytes("café\r\nnext".encode())

    with vane.open_file(str(path), "rb", buffering=0) as raw:
        assert isinstance(raw, vane.VaneFileReader)
        assert raw.read(5) == "café".encode()
    with vane.open_file(str(path), "rb", buffering=4) as buffered:
        assert isinstance(buffered, io.BufferedReader)
        assert buffered.read() == path.read_bytes()
    with vane.open_file(str(path), encoding="utf-8") as text:
        assert isinstance(text, io.TextIOWrapper)
        assert text.readline() == "café\n"
        assert text.read() == "next"
    with vane.open_file(str(path), "rt", buffering=1, encoding="utf-8", newline="") as text:
        assert text.readline() == "café\r\n"


@pytest.mark.skipif(os.name == "nt", reason="open-file replacement semantics differ on Windows")
def test_open_file_unbuffered_mode_does_not_readahead(tmp_path):
    path = tmp_path / "unbuffered.bin"
    path.write_bytes(b"abc")

    with vane.open_file(str(path), "rb", buffering=0) as raw:
        assert raw.read(1) == b"a"
        path.write_bytes(b"aXY")
        assert raw.read(2) == b"XY"


@pytest.mark.parametrize("mode", ["w", "r+", "br", ""])
def test_open_file_rejects_non_read_modes_before_io(mode):
    with pytest.raises(ValueError, match="only 'r', 'rt', and 'rb'"):
        vane.open_file("missing", mode)


@pytest.mark.parametrize("buffering", [-2, -10])
def test_open_file_rejects_invalid_buffering_before_io(buffering):
    with pytest.raises(ValueError, match="buffering must be"):
        vane.open_file("missing", buffering=buffering)


def test_open_file_rejects_binary_text_options_and_unbuffered_text():
    with pytest.raises(ValueError, match="binary mode"):
        vane.open_file("missing", "rb", encoding="utf-8")
    with pytest.raises(ValueError, match="unbuffered text"):
        vane.open_file("missing", "rt", buffering=0)


def test_file_to_tempfile_copies_only_the_logical_view(tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"outside-selected-outside")

    with vane.File(str(path), position=8, size=8).to_tempfile(buffer_size=3) as temporary:
        assert temporary.read() == b"selected"
        assert temporary.tell() == 8


def test_file_to_tempfile_closes_partial_file_after_failure(monkeypatch):
    created = []

    class RecordingTemporaryFile(io.BytesIO):
        pass

    class FailingReader(io.RawIOBase):
        def readable(self):
            return True

        def read(self, _size=-1):
            raise RuntimeError("read failed")

    def temporary_file(**_kwargs):
        result = RecordingTemporaryFile()
        created.append(result)
        return result

    monkeypatch.setattr(file_module.tempfile, "TemporaryFile", temporary_file)
    monkeypatch.setattr(file_module, "_file_open", lambda *_args, **_kwargs: FailingReader())

    with pytest.raises(RuntimeError, match="read failed"):
        vane.File("unused").to_tempfile(buffer_size=4)
    assert len(created) == 1
    assert created[0].closed


def test_file_reader_http_and_s3_reuse_connection_resolution():
    payload = b"prefix-remote-window-suffix"
    server, thread, handler = _start_object_server(payload)
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    connection = vane.connect()
    try:
        connection.execute("SET http_proxy = ''")
        connection.execute(f"SET s3_endpoint = '{endpoint.removeprefix('http://')}'")
        connection.execute("SET s3_access_key_id = 'reader-access-key'")
        connection.execute("SET s3_secret_access_key = 'reader-secret-key'")
        connection.execute("SET s3_region = 'us-east-1'")
        connection.execute("SET s3_use_ssl = false")
        connection.execute("SET s3_url_style = 'path'")
        http_url = f"{endpoint}/bucket/object.bin"
        for url in (http_url, "s3://bucket/object.bin"):
            with vane.File(url, position=7, size=13).open(buffer_size=4, connection=connection) as reader:
                assert reader.read() == b"remote-window"
    finally:
        try:
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    with handler.requests_lock:
        requests = list(handler.requests)
    ranges = [request["range"] for request in requests if request["range"] is not None]
    assert ranges
    assert set(ranges) == {"bytes=7-19"}
    assert any(
        request["range"] == "bytes=7-19"
        and request["authorization"]
        and "Credential=reader-access-key/" in request["authorization"]
        for request in requests
    )


def test_file_reader_interrupt_cancels_only_the_active_operation():
    payload = bytes(range(256)) * (2 * 1024 * 1024 // 256)
    server, server_thread, handler = _start_object_server(payload)
    connection = vane.connect()
    reader = None
    read_thread = None
    errors = []
    try:
        connection.execute("SET http_proxy = ''")
        url = f"http://127.0.0.1:{server.server_address[1]}/bucket/object.bin"
        reader = vane.File(url, position=1024 * 1024, size=1024 * 1024).open(
            buffer_size=4096,
            connection=connection,
        )
        handler.block_reads = True

        def read_file():
            try:
                reader.read()
            except BaseException as error:
                errors.append(error)

        read_thread = threading.Thread(target=read_file)
        read_thread.start()
        assert handler.read_started.wait(timeout=5)
        connection.interrupt()
        handler.release_read.set()
        read_thread.join(timeout=5)

        assert not read_thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], vane.InterruptException)
        assert reader.tell() == 0
        assert reader.read(4) == payload[1024 * 1024 : 1024 * 1024 + 4]
    finally:
        handler.release_read.set()
        if read_thread is not None:
            read_thread.join(timeout=5)
        if reader is not None:
            reader.close()
        connection.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)


def test_concurrent_file_reader_close_waits_for_complete_cleanup():
    payload = bytes(range(256)) * (2 * 1024 * 1024 // 256)
    server, server_thread, handler = _start_object_server(payload)
    connection = vane.connect()
    reader = None
    read_thread = None
    close_threads = []
    close_started = [threading.Event(), threading.Event()]
    close_finished = [threading.Event(), threading.Event()]
    try:
        connection.execute("SET http_proxy = ''")
        url = f"http://127.0.0.1:{server.server_address[1]}/bucket/object.bin"
        reader = vane.File(url).open(buffer_size=4096, connection=connection)
        handler.block_reads = True

        read_thread = threading.Thread(target=reader.read)
        read_thread.start()
        assert handler.read_started.wait(timeout=5)

        def close_reader(index):
            close_started[index].set()
            reader.close()
            close_finished[index].set()

        close_threads = [threading.Thread(target=close_reader, args=(index,)) for index in range(2)]
        for thread in close_threads:
            thread.start()
        assert all(started.wait(timeout=5) for started in close_started)

        assert not close_finished[0].wait(timeout=0.2)
        assert not close_finished[1].is_set()

        handler.release_read.set()
        read_thread.join(timeout=5)
        for thread in close_threads:
            thread.join(timeout=5)

        assert not read_thread.is_alive()
        assert all(not thread.is_alive() for thread in close_threads)
        assert all(finished.is_set() for finished in close_finished)
        assert reader.closed
    finally:
        handler.release_read.set()
        if read_thread is not None:
            read_thread.join(timeout=5)
        for thread in close_threads:
            thread.join(timeout=5)
        if reader is not None:
            reader.close()
        connection.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)

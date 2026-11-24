# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise benchmark HTTP startup without requiring native codec artifacts."""

import json
import multiprocessing
import socket
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from scripts import benchmark_native_media as benchmark


def _serve_without_reverse_dns(paths, counters, control):
    def unavailable(*_args):
        raise AssertionError("numeric loopback must not perform reverse DNS")

    socket.getfqdn = unavailable
    benchmark._http_server(paths, 0, counters, control)


def test_benchmark_http_server_starts_without_reverse_dns(tmp_path):
    source = tmp_path / "input.bin"
    source.write_bytes(b"benchmark input")
    context = multiprocessing.get_context("spawn")
    counters = context.Array("Q", 5)
    parent, child = context.Pipe()
    process = context.Process(target=_serve_without_reverse_dns, args=({"/input/0": source}, counters, child))
    process.start()
    child.close()
    try:
        assert parent.poll(10), "benchmark server did not announce its port"
        port = parent.recv()
        url = f"http://127.0.0.1:{port}/input/0"
        with urlopen(Request(url, method="HEAD"), timeout=5) as response:
            assert int(response.headers["Content-Length"]) == source.stat().st_size
        with urlopen(Request(url, headers={"Range": "bytes=2-6"}), timeout=5) as response:
            assert response.status == 206
            assert response.read() == b"nchma"
    finally:
        try:
            if process.is_alive():
                parent.send("stop")
        except (BrokenPipeError, EOFError, OSError):
            pass
        parent.close()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert process.exitcode == 0
    assert list(counters) == [5, 1, 1, 0, 0]


@pytest.mark.parametrize("operation", ["image_decode", "audio_resample"])
def test_python_media_benchmark_http_subprocess(tmp_path, operation):
    from tests.fast.test_native_media_extensions import _png, _wav

    if operation == "image_decode":
        pytest.importorskip("PIL")
        payload = _png()
    else:
        pytest.importorskip("soundfile")
        pytest.importorskip("soxr")
        payload = _wav()
    source = tmp_path / "input.bin"
    source.write_bytes(payload)
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(Path(benchmark.__file__).resolve()),
            operation,
            str(source),
            "--backend",
            "python",
            "--rows",
            "2",
            "--concurrency",
            "2",
            "--repetitions",
            "1",
            "--transport",
            "http",
            "--diagnostics",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    traffic = report["samples"]["python"][0]["http"]
    assert traffic["body_bytes"] > 0 and traffic["get_requests"] > 0
    assert traffic["errors"] == 0 and traffic["active_requests"] == 0
    assert report["diagnostics"]["python"]["python_temporary_files"]["written_bytes"] > 0

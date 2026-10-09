# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""The CLI owns its Ray connection and both public Flight listeners."""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pyarrow.flight as flight
import pytest

from tests.fast.test_flight_server import TOKEN, call
from vane.client import Client


@pytest.mark.real_ray
@pytest.mark.ray_cluster_owner
@pytest.mark.skipif(sys.platform == "win32", reason="subprocess SIGTERM shutdown uses POSIX signals")
def test_standalone_cli_handles_sessions_and_releases_database(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text(TOKEN)
    database = str(tmp_path / "server.db")
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-m",
            "vane.server",
            "--ray-address",
            "local",
            "--ray-cpus",
            "2",
            "--result-port",
            "0",
            "--port",
            "0",
            "--token-file",
            str(token_file),
            "--database",
            database,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # Read startup in a thread so a failed listener cannot hang this test.
        with ThreadPoolExecutor(1) as threads:
            ready = threads.submit(process.stdout.readline)
            try:
                info = json.loads(ready.result(timeout=60))
            except BaseException:
                process.kill()
                raise
        with flight.FlightClient(info["location"]) as client:
            assert call(client, "vane.session.open")["ok"]
        with Client(info["location"], token=TOKEN) as client:
            assert client.query("select 42").collect().column(0).to_pylist() == [42]
        process.terminate()
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, stdout + stderr
        reopened = subprocess.run(
            [sys.executable, "-I", "-c", "import vane,sys; vane.connect(sys.argv[1]).close()", database],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert reopened.returncode == 0, reopened.stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)

# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests.fast import test_native_media_extensions as media

video_path = media.video_path


@pytest.mark.parametrize("capacity", [None, "", "1", "4"])
def test_native_decoder_admission_configuration(capacity):
    env = {**os.environ, "VANE_RUNNER": "local-fast"}
    variable = "VANE_NATIVE_MEDIA_MAX_CONCURRENT_DECODES"
    if capacity is None:
        env.pop(variable, None)
    else:
        env[variable] = capacity
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import vane; "
            "a = vane.connect(); b = vane.connect(); "
            "assert a.execute('SELECT 42').fetchone() == (42,); "
            "assert b.execute('SELECT 43').fetchone() == (43,); "
            "a.close(); b.close()",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("capacity", ["0", "-1", "1.5", " 2", "2x", str(2**64)])
def test_native_decoder_admission_rejects_invalid_capacity(capacity):
    result = subprocess.run(
        [sys.executable, "-I", "-c", "import vane; vane.connect()"],
        env={**os.environ, "VANE_RUNNER": "local-fast", "VANE_NATIVE_MEDIA_MAX_CONCURRENT_DECODES": capacity},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "VANE_NATIVE_MEDIA_MAX_CONCURRENT_DECODES must be a positive integer" in result.stderr


def test_native_video_admission_parallel_files_and_cleanup(video_path, tmp_path):
    providers = os.environ.get("VANE_TEST_NATIVE_MEDIA_PROVIDERS") == "1"
    artifact = "" if providers else str(media._artifact("video"))
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    # The pool is fixed when the first database is created, including a module's
    # default connection. Configure the subprocess before importing Vane.
    script = r"""
import concurrent.futures
import sys
import vane
from vane.datasource.video_reader import VideoFrameSource

artifact, path, broken = sys.argv[1:]

def connect():
    con = (vane.connect() if not artifact else
           vane.connect(config={"allow_unsigned_extensions": "true"}))
    if artifact:
        con.load_extension(artifact)
    else:
        vane.load_installed_extension("native_media", connection=con)
    con.execute("SET video_backend='native'")
    con.execute("SET threads=4")
    return con

def frames(con, paths, **kwargs):
    return vane.read_video_frames(paths, 6, 8, connection=con,
                                  max_partition_bytes=2048, **kwargs)

def run(index):
    with connect() as con:
        for tasks in (1, 2, 5):
            rows = frames(con, [path] * 5, read_task_count=tasks).project("frame_index").fetchall()
            assert sorted(row[0] for row in rows) == sorted(list(range(12)) * 5)
        assert len(frames(con, [path] * 3, frame_limit=3).fetchall()) == 3
        assert frames(con, [broken], frame_limit=0).fetchall() == []
        assert len(frames(con, [broken, path], on_error="skip").fetchall()) == 12
        # LIMIT and a downstream expression error end the source before EOF.
        assert len(frames(con, path).limit(1).fetchall()) == 1
        try:
            frames(con, path).project("CAST('invalid' || frame_index::VARCHAR AS BIGINT)").fetchall()
        except vane.ConversionException:
            pass
        else:
            raise AssertionError("expected downstream conversion failure")
        try:
            frames(con, broken).fetchall()
        except vane.Error:
            pass
        else:
            raise AssertionError("expected codec failure")
        assert len(frames(con, path).fetchall()) == 12
        source = VideoFrameSource([path] * 3, width=8, height=6, read_task_count=3,
                                  max_partition_bytes=2048)
        assert con.table_function("native_video_frames", source._native_parameters()).count("*").fetchone() == (36,)

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    list(executor.map(run, range(3)))
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script, artifact, str(video_path), str(broken)],
        env={**os.environ, "VANE_RUNNER": "local-fast", "VANE_NATIVE_MEDIA_MAX_CONCURRENT_DECODES": "1"},
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )

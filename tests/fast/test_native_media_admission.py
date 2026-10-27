# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import os
import subprocess
import sys

import pytest

from tests.fast import test_native_media_extensions as media

video_path = media.video_path


def _assert_retained_video_pixels(read_frames):
    import numpy as np

    retained = read_frames().project("frame_index, data").fetchall()
    assert [row[0] for row in retained] == list(range(12))
    assert all(
        pixels.shape == (6, 8, 3) and pixels.dtype == np.uint8 and pixels.nbytes == 144 for _, pixels in retained
    )
    snapshots = [pixels.copy() for _, pixels in retained]
    repeated = read_frames().project("frame_index, data").fetchall()
    assert [row[0] for row in repeated] == list(range(12))
    for (_, retained_pixels), (_, repeated_pixels), snapshot in zip(retained, repeated, snapshots):
        np.testing.assert_array_equal(retained_pixels, snapshot)
        np.testing.assert_array_equal(repeated_pixels, snapshot)
    return retained


def test_video_admission_retained_image_contract():
    pytest.importorskip("numpy")
    import vane

    with vane.connect() as con:
        _assert_retained_video_pixels(
            lambda: con.sql(
                "SELECT i AS frame_index, "
                "image(repeat(chr(65 + i::INTEGER), 144)::BLOB, 8, 6, 3, 'RGB')::IMAGE('RGB', 6, 8) AS data "
                "FROM range(12) t(i) ORDER BY i"
            )
        )


@pytest.mark.parametrize("capacity", [None, "", "1", "4"])
@pytest.mark.parametrize("variable", ["VANE_NATIVE_MEDIA_MAX_CONCURRENT_DECODES", "VANE_NATIVE_MEDIA_MAX_OUTPUT_BYTES"])
def test_native_decoder_admission_configuration(capacity, variable):
    env = {**os.environ, "VANE_RUNNER": "local-fast"}
    if capacity is None:
        env.pop(variable, None)
    else:
        env[variable] = capacity
    result = subprocess.run(
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
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("capacity", ["0", "-1", "1.5", " 2", "2x", str(2**64)])
@pytest.mark.parametrize("variable", ["VANE_NATIVE_MEDIA_MAX_CONCURRENT_DECODES", "VANE_NATIVE_MEDIA_MAX_OUTPUT_BYTES"])
def test_native_decoder_admission_rejects_invalid_capacity(capacity, variable):
    result = subprocess.run(
        [sys.executable, "-I", "-c", "import vane; vane.connect()"],
        env={**os.environ, "VANE_RUNNER": "local-fast", variable: capacity},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert f"{variable} must be a positive integer" in result.stderr


@pytest.mark.parametrize("output_bytes", [None, "144", "288"])
def test_native_video_admission_parallel_files_and_cleanup(video_path, tmp_path, output_bytes):
    providers = os.environ.get("VANE_TEST_NATIVE_MEDIA_PROVIDERS") == "1"
    artifact = "" if providers else str(media._artifact("video"))
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    # The pool is fixed when the first database is created, including a module's
    # default connection. Configure the subprocess before importing Vane.
    script = (
        inspect.getsource(_assert_retained_video_pixels)
        + r"""
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
        assert len(frames(con, [path] * 3).order("frame_index DESC").fetchall()) == 36
        # Keep detached results alive through the later error/cleanup checks.
        retained = _assert_retained_video_pixels(lambda: frames(con, path))
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
    )
    env = {**os.environ, "VANE_RUNNER": "local-fast", "VANE_NATIVE_MEDIA_MAX_CONCURRENT_DECODES": "1"}
    if output_bytes is None:
        env.pop("VANE_NATIVE_MEDIA_MAX_OUTPUT_BYTES", None)
    else:
        env["VANE_NATIVE_MEDIA_MAX_OUTPUT_BYTES"] = output_bytes
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, artifact, str(video_path), str(broken)],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_video_output_budget_rejects_an_oversized_frame(video_path):
    providers = os.environ.get("VANE_TEST_NATIVE_MEDIA_PROVIDERS") == "1"
    artifact = "" if providers else str(media._artifact("video"))
    script = r"""
import sys
import vane
artifact, path = sys.argv[1:]
con = vane.connect() if not artifact else vane.connect(config={"allow_unsigned_extensions": "true"})
if artifact:
    con.load_extension(artifact)
else:
    vane.load_installed_extension("native_media", connection=con)
con.execute("SET video_backend='native'")
for policy in ('raise', 'skip'):
    try:
        vane.read_video_frames(path, 6, 8, connection=con, on_error=policy).fetchall()
    except vane.OutOfMemoryException as error:
        assert 'VANE_NATIVE_MEDIA_MAX_OUTPUT_BYTES' in str(error), str(error)
    else:
        raise AssertionError('an impossible output reservation must fail without waiting')
assert vane.read_video_frames(path, 6, 8, connection=con, frame_limit=0).fetchall() == []
con.close()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, artifact, str(video_path)],
        env={**os.environ, "VANE_RUNNER": "local-fast", "VANE_NATIVE_MEDIA_MAX_OUTPUT_BYTES": "143"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

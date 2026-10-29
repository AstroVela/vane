# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys

import pytest

from tests.fast import test_native_media_extensions as media

video_path = media.video_path


@pytest.mark.parametrize("window_bytes", ["144", "288"])
def test_native_video_generic_execution_flow_matches_ordinary_expressions(video_path, window_bytes):
    providers = os.environ.get("VANE_TEST_NATIVE_MEDIA_PROVIDERS") == "1"
    artifacts = [] if providers else [str(media._artifact("video"))]
    script = r"""
import faulthandler
import json
import sys
import numpy as np
import pyarrow as pa
import vane

faulthandler.dump_traceback_later(60, exit=True)
path, enabled, *artifacts = sys.argv[1:]
con = vane.connect(config={'allow_unsigned_extensions': 'true'}) if artifacts else vane.connect()
if artifacts:
    for artifact in artifacts:
        con.load_extension(artifact)
else:
    vane.load_installed_extension('native_media', connection=con)
con.execute("SET video_backend='native'")
con.execute('SET threads=4')
con.execute("SET streaming_buffer_size='512B'")

def projected():
    frames = vane.read_video_frames(path, 6, 8, connection=con, max_partition_bytes=288)
    return frames.project('''
        frame_index,
        native_resize(native_resize(data, 4, 3), 3, 2) AS nested,
        CASE WHEN frame_index%2=0 THEN native_resize(data, 3, 2) ELSE NULL END AS conditional
    ''')

assert ('NATIVE_PROJECT' in projected().explain()) == (enabled == '1')
rows = projected().fetchall()
assert [row[0] for row in rows] == list(range(12))
snapshots = [(a.copy(), None if b is None else b.copy()) for _, a, b in rows]
pixels = [(a.tolist(), None if b is None else b.tolist()) for _, a, b in rows]
for _ in range(2):
    repeated = projected().fetchall()
    for (_, a, b), (expected_a, expected_b) in zip(repeated, snapshots):
        np.testing.assert_array_equal(a, expected_a)
        if expected_b is None:
            assert b is None
        else:
            np.testing.assert_array_equal(b, expected_b)
assert len(projected().limit(2).fetchall()) == 2
arrow = projected().to_arrow_table()
capsule = pa.table(projected())
assert arrow.equals(capsule)
saved = arrow.to_pylist()
con.close()
for (_, a, b), (expected_a, expected_b) in zip(rows, snapshots):
    np.testing.assert_array_equal(a, expected_a)
    if b is not None:
        np.testing.assert_array_equal(b, expected_b)
assert arrow.to_pylist() == saved
print('FLOW_RESULT=' + json.dumps(pixels))
faulthandler.cancel_dump_traceback_later()
"""
    outputs = []
    for enabled in ("0", "1"):
        result = subprocess.run(
            [sys.executable, "-I", "-c", script, str(video_path), enabled, *artifacts],
            env={
                **os.environ,
                "VANE_RUNNER": "local-fast",
                "VANE_PROGRESS": "0",
                "VANE_NATIVE_ENABLE_EXECUTION_FLOW": enabled,
                "VANE_NATIVE_MEDIA_ENABLE_SCALAR_ADMISSION": "0",
                "VANE_NATIVE_MEDIA_MAX_CONCURRENT_DECODES": "1",
                "VANE_NATIVE_MEDIA_MAX_OUTPUT_BYTES": "800",
                "VANE_NATIVE_MEDIA_MAX_INFLIGHT_OUTPUT_BYTES": window_bytes,
            },
            capture_output=True,
            text=True,
            timeout=75,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        outputs.append(
            json.loads(
                next(
                    line.removeprefix("FLOW_RESULT=")
                    for line in result.stdout.splitlines()
                    if line.startswith("FLOW_RESULT=")
                )
            )
        )
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("window_bytes", ["144", "288"])
@pytest.mark.parametrize("antialias", [False, True])
def test_native_video_fixed_resize_admission_matches_unmanaged_execution(video_path, window_bytes, antialias):
    providers = os.environ.get("VANE_TEST_NATIVE_MEDIA_PROVIDERS") == "1"
    artifacts = [] if providers else [str(media._artifact("video"))]
    script = r"""
import faulthandler
import json
import os
import sys
import numpy as np
import pyarrow as pa
import vane

faulthandler.dump_traceback_later(60, exit=True)
path, antialias, managed, *artifacts = sys.argv[1:]
antialias = antialias == '1'
managed = managed == '1'
con = vane.connect(config={'allow_unsigned_extensions': 'true'}) if artifacts else vane.connect()
if artifacts:
    for artifact in artifacts:
        con.load_extension(artifact)
else:
    vane.load_installed_extension('native_media', connection=con)
# A self-contained extension must read the flag captured by its host runtime.
# Mutating the environment after database creation must not change its binding.
os.environ['VANE_NATIVE_MEDIA_ENABLE_SCALAR_ADMISSION'] = '0' if managed else '1'
con.execute("SET video_backend='native'")
con.execute('SET threads=1')
con.execute("SET streaming_buffer_size='512B'")

def resized():
    frames = vane.read_video_frames(path, 6, 8, connection=con, max_partition_bytes=288)
    return frames.project('frame_index, native_resize(data, 3, 2, ' + str(antialias).lower() + ') AS data')

plan = resized().explain()
assert ('MEDIA_PROJECT' in plan) == managed, plan
rows = resized().fetchall()
assert [index for index, _ in rows] == list(range(12))
pixels = [image.tolist() for _, image in rows]
snapshots = [image.copy() for _, image in rows]
assert len(resized().limit(2).fetchall()) == 2
for threads in (1, 4):
    con.execute('SET threads=' + str(threads))
    repeated = resized().fetchall()
    for (_, image), snapshot in zip(repeated, snapshots):
        np.testing.assert_array_equal(image, snapshot)
arrow = resized().to_arrow_table()
capsule = pa.table(resized())
assert arrow['frame_index'].to_pylist() == list(range(12))
assert capsule['frame_index'].to_pylist() == list(range(12))
assert arrow['data'].combine_chunks().storage.to_pylist() == capsule['data'].combine_chunks().storage.to_pylist()
arrow_pixels = arrow['data'].combine_chunks().storage.to_pylist()
con.close()
for (_, image), snapshot in zip(rows, snapshots):
    np.testing.assert_array_equal(image, snapshot)
assert arrow['data'].combine_chunks().storage.to_pylist() == arrow_pixels
print('RESIZE_RESULT=' + json.dumps(pixels))
faulthandler.cancel_dump_traceback_later()
"""
    outputs = []
    for managed in ("0", "1"):
        result = subprocess.run(
            [sys.executable, "-I", "-c", script, str(video_path), str(int(antialias)), managed, *artifacts],
            env={
                **os.environ,
                "VANE_RUNNER": "local-fast",
                "VANE_PROGRESS": "0",
                "VANE_NATIVE_MEDIA_ENABLE_SCALAR_ADMISSION": managed,
                "VANE_NATIVE_MEDIA_MAX_CONCURRENT_DECODES": "1",
                "VANE_NATIVE_MEDIA_MAX_OUTPUT_BYTES": "800",
                "VANE_NATIVE_MEDIA_MAX_INFLIGHT_OUTPUT_BYTES": window_bytes,
            },
            capture_output=True,
            text=True,
            timeout=75,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        outputs.append(
            json.loads(
                next(
                    line.removeprefix("RESIZE_RESULT=")
                    for line in result.stdout.splitlines()
                    if line.startswith("RESIZE_RESULT=")
                )
            )
        )
    assert outputs[0] == outputs[1]


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
@pytest.mark.parametrize(
    "variable",
    [
        "VANE_NATIVE_MEDIA_MAX_CONCURRENT_DECODES",
        "VANE_NATIVE_MEDIA_MAX_OUTPUT_BYTES",
        "VANE_NATIVE_MEDIA_MAX_INFLIGHT_OUTPUT_BYTES",
    ],
)
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
@pytest.mark.parametrize(
    "variable",
    [
        "VANE_NATIVE_MEDIA_MAX_CONCURRENT_DECODES",
        "VANE_NATIVE_MEDIA_MAX_OUTPUT_BYTES",
        "VANE_NATIVE_MEDIA_MAX_INFLIGHT_OUTPUT_BYTES",
    ],
)
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


@pytest.mark.parametrize("window_bytes", ["144", "288"])
@pytest.mark.parametrize("arrow_export", ["table", "capsule"])
def test_native_video_flow_sort_spills_and_preserves_pixels(video_path, tmp_path, window_bytes, arrow_export):
    providers = os.environ.get("VANE_TEST_NATIVE_MEDIA_PROVIDERS") == "1"
    artifact = "" if providers else str(media._artifact("video"))
    script = r"""
import faulthandler
from pathlib import Path
import sys
import numpy as np
import pyarrow as pa
import vane

faulthandler.dump_traceback_later(90, exit=True)
artifact, path, spill_path, arrow_export = sys.argv[1:]
with (vane.connect() if not artifact else
      vane.connect(config={'allow_unsigned_extensions': 'true'})) as con:
    if artifact:
        con.load_extension(artifact)
    else:
        vane.load_installed_extension('native_media', connection=con)
    con.execute("SET video_backend='native'")
    con.execute('SET threads=4')
    con.execute("SET temp_directory='" + spill_path.replace("'", "''") + "'")
    con.execute("SET memory_limit='2MiB'")
    con.execute("SET streaming_buffer_size='512B'")

    def frames(files=1):
        return vane.read_video_frames([path] * files, 6, 8, connection=con,
                                      read_task_count=files,
                                      max_partition_bytes=2048).project('frame_index, data')

    ordered = frames(3).order('frame_index DESC').execute()
    assert any(p.is_file() and p.stat().st_size for p in Path(spill_path).rglob('*'))
    retained = ordered.fetchall()
    assert [row[0] for row in retained] == sorted(list(range(12)) * 3, reverse=True)
    snapshots = [pixels.copy() for _, pixels in retained]
    filtered = frames().filter('frame_index % 2 = 0').order('frame_index DESC').fetchall()
    assert [row[0] for row in filtered] == [10, 8, 6, 4, 2, 0]
    mixed = frames().project(
        'frame_index, data, CASE WHEN frame_index % 5 = 0 THEN NULL '
        'ELSE frame_index % 3 END AS k'
    ).order('k DESC NULLS FIRST, frame_index ASC').fetchall()
    expected = sorted(range(12), key=lambda i: (i % 5 != 0, -(i % 3) if i % 5 else 0, i))
    assert [row[0] for row in mixed] == expected

    con.execute("SET max_temp_directory_size='0B'")
    try:
        frames(3).order('frame_index DESC').fetchall()
    except vane.OutOfMemoryException:
        pass
    else:
        raise AssertionError('a full media sort with spill disabled must fail')
    con.execute("SET max_temp_directory_size='100MiB'")
    con.execute("SET streaming_buffer_size='175B'")
    try:
        frames().order('frame_index').fetchall()
    except vane.OutOfMemoryException as error:
        assert 'streaming_buffer_size receiving capacity' in str(error), str(error)
    else:
        raise AssertionError('the sort must reserve encoded keys as well as pixels')
    con.execute("SET streaming_buffer_size='512B'")
    for relation, message in (
        (frames().order('frame_index').limit(3), 'materialization budget handoff'),
        (frames().project('frame_index, data, frame_index::VARCHAR AS k').order('k'),
         'fixed-width primitive sort keys'),
    ):
        try:
            relation.fetchall()
        except vane.NotImplementedException as error:
            assert message in str(error), str(error)
        else:
            raise AssertionError('a deferred sorting path must be rejected explicitly')
    try:
        frames().project("CAST('bad' || frame_index::VARCHAR AS BIGINT) AS frame_index, data").order(
            'frame_index'
        ).fetchall()
    except vane.ConversionException:
        pass
    else:
        raise AssertionError('expected a downstream expression failure')

    # Compare the spill/reload path with an independent larger-memory scan.
    con.execute("SET memory_limit='10MiB'")
    reference = dict(frames().fetchall())
    assert sorted(reference) == list(range(12))
    for index, pixels in retained:
        np.testing.assert_array_equal(pixels, reference[index])
    for index, pixels in filtered:
        np.testing.assert_array_equal(pixels, reference[index])
    for index, pixels, _ in mixed:
        np.testing.assert_array_equal(pixels, reference[index])
    relation = frames().order('frame_index DESC')
    arrow = relation.to_arrow_table() if arrow_export == 'table' else pa.table(relation)
    assert arrow['frame_index'].to_pylist() == list(range(11, -1, -1))
    arrow_pixels = arrow['data'].combine_chunks().storage.to_pylist()
    assert con.execute('SELECT 42').fetchone() == (42,)
for (index, pixels), snapshot in zip(retained, snapshots):
    np.testing.assert_array_equal(pixels, snapshot)
    np.testing.assert_array_equal(pixels, reference[index])
assert arrow['data'].combine_chunks().storage.to_pylist() == arrow_pixels
faulthandler.cancel_dump_traceback_later()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, artifact, str(video_path), str(tmp_path / "sort-spill"), arrow_export],
        env={
            **os.environ,
            "VANE_RUNNER": "local-fast",
            "VANE_PROGRESS": "0",
            "VANE_NATIVE_MEDIA_MAX_CONCURRENT_DECODES": "1",
            "VANE_NATIVE_MEDIA_MAX_OUTPUT_BYTES": "288",
            "VANE_NATIVE_MEDIA_MAX_INFLIGHT_OUTPUT_BYTES": window_bytes,
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("window_bytes", ["144", "288"])
def test_native_video_flow_materialized_result_spills_and_cleans_up(video_path, tmp_path, window_bytes):
    providers = os.environ.get("VANE_TEST_NATIVE_MEDIA_PROVIDERS") == "1"
    artifact = "" if providers else str(media._artifact("video"))
    script = r"""
import faulthandler
from pathlib import Path
import sys
import numpy as np
import vane

faulthandler.dump_traceback_later(60, exit=True)
artifact, path, spill_path = sys.argv[1:]
with (vane.connect() if not artifact else
      vane.connect(config={'allow_unsigned_extensions': 'true'})) as con:
    if artifact:
        con.load_extension(artifact)
    else:
        vane.load_installed_extension('native_media', connection=con)
    con.execute("SET video_backend='native'")
    con.execute('SET threads=4')
    con.execute("SET temp_directory='" + spill_path.replace("'", "''") + "'")
    con.execute("SET memory_limit='2MiB'")
    con.execute("SET streaming_buffer_size='512B'")

    def frames(files=3):
        return vane.read_video_frames([path] * files, 6, 8, connection=con,
                                      read_task_count=files,
                                      max_partition_bytes=2048).project('frame_index, data')

    # Execute materializes before returning. The producer must reach EOF even
    # though its window is only one/two frames and the collector has to spill.
    materialized = frames().execute()
    assert any(p.is_file() and p.stat().st_size for p in Path(spill_path).rglob('*'))
    retained = materialized.fetchall()
    assert sorted(row[0] for row in retained) == sorted(list(range(12)) * 3)
    assert all(pixels.shape == (6, 8, 3) and pixels.dtype == np.uint8 for _, pixels in retained)
    snapshots = [pixels.copy() for _, pixels in retained]
    filtered = frames(1).filter('frame_index % 2 = 0').limit(3).fetchall()
    assert [row[0] for row in filtered] == [0, 2, 4]

    # A receiving failure must not turn into an unbounded in-memory result or
    # leave the flow/pixel pools stuck for the next query.
    con.execute("SET max_temp_directory_size='0B'")
    try:
        frames().fetchall()
    except vane.OutOfMemoryException:
        pass
    else:
        raise AssertionError('a full BufferManager with spill disabled must fail')
    con.execute("SET max_temp_directory_size='100MiB'")
    con.execute("SET memory_limit='10MiB'")
    reference = {index: pixels for index, pixels in frames(1).fetchall()}
    assert sorted(reference) == list(range(12))
    for index, pixels in retained:
        np.testing.assert_array_equal(pixels, reference[index])
    con.execute("SET streaming_buffer_size='151B'")
    try:
        frames(1).fetchall()
    except vane.OutOfMemoryException as error:
        assert 'streaming_buffer_size receiving capacity' in str(error), str(error)
    else:
        raise AssertionError('an impossible materialized receiving row must fail')
    con.execute("SET streaming_buffer_size='512B'")
    try:
        frames(1).project("CAST('bad' || frame_index::VARCHAR AS BIGINT), data").fetchall()
    except vane.ConversionException:
        pass
    else:
        raise AssertionError('expected a downstream expression failure')
    assert len(frames(1).fetchall()) == 12
    assert con.execute('SELECT 42').fetchone() == (42,)
for (_, pixels), snapshot in zip(retained, snapshots):
    np.testing.assert_array_equal(pixels, snapshot)
faulthandler.cancel_dump_traceback_later()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, artifact, str(video_path), str(tmp_path / "spill")],
        env={
            **os.environ,
            "VANE_RUNNER": "local-fast",
            "VANE_PROGRESS": "0",
            "VANE_NATIVE_MEDIA_MAX_CONCURRENT_DECODES": "1",
            "VANE_NATIVE_MEDIA_MAX_OUTPUT_BYTES": "288",
            "VANE_NATIVE_MEDIA_MAX_INFLIGHT_OUTPUT_BYTES": window_bytes,
        },
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("window_bytes", ["144", "288"])
@pytest.mark.parametrize("ordered", [False, True])
def test_native_video_flow_streaming_result_handoff(video_path, window_bytes, ordered):
    providers = os.environ.get("VANE_TEST_NATIVE_MEDIA_PROVIDERS") == "1"
    artifact = "" if providers else str(media._artifact("video"))
    script = r"""
import faulthandler
import gc
import sys
import time
import vane
from vane.execution.request_admission import RequestAdmissionLimits
from vane.execution.result_delivery import ResultDeliveryLimits

faulthandler.dump_traceback_later(60, exit=True)
artifact, path, ordered = sys.argv[1:]
file_count = 1 if ordered == 'true' else 3
expected_frames = list(range(12)) * file_count

def configure(connection, receiving_bytes):
    if artifact:
        connection.load_extension(artifact)
    else:
        vane.load_installed_extension('native_media', connection=connection)
    connection.execute("SET video_backend='native'")
    connection.execute('SET threads=4')
    connection.execute('SET preserve_insertion_order=' + ordered)
    # Pixel bytes and receiving payload bytes are separate capacities. Each
    # selected row has 8 index bytes plus 144 fixed-image bytes.
    connection.execute("SET streaming_buffer_size='" + str(receiving_bytes) + "B'")

with (vane.connect() if not artifact else
      vane.connect(config={'allow_unsigned_extensions': 'true'})) as con:
    configure(con, 512)
    runtime = con.configure_local_runtime(
        request_limit=RequestAdmissionLimits(1, 1),
        result_limit=ResultDeliveryLimits(2, 1048576),
    )

    def frames(connection=con):
        return vane.read_video_frames([path] * file_count, 6, 8, connection=connection,
                                      read_task_count=file_count,
                                      max_partition_bytes=2048).project('frame_index, data')

    def pixels(batch):
        column = batch['data']
        array = column.combine_chunks() if hasattr(column, 'combine_chunks') else column
        return array.storage.to_pylist()

    def collect(relation):
        result = relation.execute_result(stream=True, rows_per_batch=1000000)
        values = []
        retained = []
        try:
            while True:
                try:
                    batch = result.take()
                except StopIteration:
                    break
                # A large preferred Arrow batch cannot prefetch the full scan.
                assert batch.num_rows == 1, batch.num_rows
                values.extend(batch['frame_index'].to_pylist())
                retained.append(batch)
        finally:
            result.close()
        return values, retained

    stream = frames().execute_result(stream=True, rows_per_batch=1000000)
    first = stream.take()
    assert first.num_rows == 1
    first_index = first['frame_index'].to_pylist()[0]
    assert 0 <= first_index < 12
    if ordered == 'true':
        assert first_index == 0
    saved = pixels(first)
    time.sleep(0.1)  # Leave the real decoder behind a non-consuming receiver.
    assert runtime.resource_snapshot()['request_admission']['active_requests'] == 1
    stream.close()
    assert runtime.resource_snapshot()['request_admission']['active_requests'] == 0
    assert pixels(first) == saved

    values, retained = collect(frames().filter('frame_index % 2 = 0'))
    assert sorted(values) == sorted(list(range(0, 12, 2)) * file_count)
    values, limited = collect(frames().filter('frame_index % 2 = 0').limit(3))
    if ordered == 'true':
        assert values == [0, 2, 4], values
    else:
        assert len(values) == 3 and all(value % 2 == 0 for value in values), values
    values, repeated = collect(frames())
    assert sorted(values) == sorted(expected_frames)
    assert pixels(first) == saved
    filtered_pixels = {batch['frame_index'].to_pylist()[0]: pixels(batch) for batch in retained}
    for batch in repeated:
        index = batch['frame_index'].to_pylist()[0]
        if index in filtered_pixels:
            assert pixels(batch) == filtered_pixels[index]

    # A failure after some accepted frames must retire the source and leave
    # the same connection usable while earlier Arrow buffers remain alive.
    bad = frames().project("CAST('bad' || frame_index::VARCHAR AS BIGINT) AS frame_index, data")
    try:
        collect(bad)
    except vane.Error:
        pass
    else:
        raise AssertionError('expected a conversion failure')
    assert runtime.resource_snapshot()['request_admission']['active_requests'] == 0
    assert con.execute('SELECT 42').fetchall() == [(42,)]
    assert pixels(first) == saved

    # An indivisible row is a permanent resource failure, not an empty-queue
    # oversize escape or a wait that needs EOF to become ready.
    # Configured local runtimes accept read-only queries. Configure the small
    # receiving queue before attaching its independent runtime.
    with (vane.connect() if not artifact else
          vane.connect(config={'allow_unsigned_extensions': 'true'})) as small:
        configure(small, 151)
        small_runtime = small.configure_local_runtime(
            request_limit=RequestAdmissionLimits(1, 1),
            result_limit=ResultDeliveryLimits(2, 1048576),
        )
        try:
            collect(frames(small))
        except vane.OutOfMemoryException as error:
            assert 'streaming_buffer_size receiving capacity' in str(error), str(error)
        else:
            raise AssertionError('an impossible receiving row must fail')
        assert small_runtime.resource_snapshot()['request_admission']['active_requests'] == 0
    values, final = collect(frames())
    assert sorted(values) == sorted(expected_frames)
    assert pixels(first) == saved
    del first, retained, limited, repeated, final, batch
    gc.collect()
    assert runtime.resource_snapshot()['request_admission']['active_requests'] == 0
    delivery = runtime.resource_snapshot()['result_delivery']
    assert delivery['active_results'] == delivery['usage_bytes'] == 0, delivery
faulthandler.cancel_dump_traceback_later()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, artifact, str(video_path), str(ordered).lower()],
        env={
            **os.environ,
            "VANE_RUNNER": "local-fast",
            "VANE_PROGRESS": "0",
            "VANE_NATIVE_MEDIA_MAX_CONCURRENT_DECODES": "1",
            "VANE_NATIVE_MEDIA_MAX_OUTPUT_BYTES": "288",
            "VANE_NATIVE_MEDIA_MAX_INFLIGHT_OUTPUT_BYTES": window_bytes,
        },
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("window_bytes", ["144", "288"])
@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
def test_native_video_flow_window_to_slow_udf(video_path, tmp_path, window_bytes, backend):
    providers = os.environ.get("VANE_TEST_NATIVE_MEDIA_PROVIDERS") == "1"
    artifact = "" if providers else str(media._artifact("video"))
    # This file is also in the explicit native_media provider CI test list.
    # Keep the producer window smaller than the preferred UDF compute batch;
    # its independent ingress byte window must flush a short batch to progress.
    script = r"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
import time
import pyarrow as pa
import vane

artifact, path, directory, backend = sys.argv[1:]
entered, proceed = (Path(directory) / name for name in ('entered', 'proceed'))

def consume(batch):
    entered.touch()
    deadline = time.monotonic() + 30
    while not proceed.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('slow consumer was never released')
        time.sleep(0.01)
    return pa.table({'frame_index': batch['frame_index']})

class Consume:
    def __call__(self, batch):
        return consume(batch)

def fail(batch):
    raise ValueError('injected flow consumer failure')

with (vane.connect() if not artifact else
      vane.connect(config={'allow_unsigned_extensions': 'true'})) as con:
    if artifact:
        con.load_extension(artifact)
    else:
        vane.load_installed_extension('native_media', connection=con)
    con.execute("SET video_backend='native'")
    con.execute('SET threads=4')
    watchdog = threading.Timer(75, con.interrupt)
    watchdog.daemon = True
    watchdog.start()

    def frames():
        return vane.read_video_frames(path, 6, 8, connection=con,
                                      max_partition_bytes=2048).project('frame_index, data')

    def mapped(relation, *, fn=None, capacity=304):
        options = {'actor_number': 1} if backend == 'subprocess_actor' and fn is None else {}
        return relation.map_batches(
            fn or (Consume if options else consume), schema={'frame_index': 'BIGINT'},
            execution_backend=backend if fn is None else 'subprocess_task',
            batch_size=4, min_task_batch_size=4, task_input_max_bytes=capacity,
            output_batch_size=2, output_target_max_bytes=1024, **options)

    try:
        with ThreadPoolExecutor(max_workers=1) as threads:
            future = threads.submit(lambda: mapped(frames().filter('frame_index % 2 = 0')).to_arrow_table())
            try:
                deadline = time.monotonic() + 30
                while not entered.exists():
                    if future.done():
                        future.result()  # expose a failure before the consumer starts
                        raise AssertionError('query completed before the consumer gate')
                    assert time.monotonic() < deadline, 'small native window failed to reach the UDF'
                    time.sleep(0.01)
                time.sleep(0.2)
                assert not future.done(), 'consumer gate did not pause the query'
            finally:
                proceed.touch()
            result = future.result(timeout=30)
        assert sorted(result['frame_index'].to_pylist()) == list(range(0, 12, 2))

        # LIMIT cancels the rest of the native/UDF chain. A new query must work
        # even while the prior detached Arrow result remains alive.
        assert len(mapped(frames()).limit(1).fetchall()) == 1
        for relation, expected in (
            (mapped(frames(), capacity=151), 'task_input_max_bytes receiving capacity'),
            (mapped(frames(), fn=fail), 'injected flow consumer failure'),
            (mapped(frames().project("CAST('bad' || frame_index::VARCHAR AS BIGINT) AS frame_index, data")),
             'Conversion'),
        ):
            try:
                relation.fetchall()
            except vane.Error as error:
                assert expected.lower() in str(error).lower(), str(error)
            else:
                raise AssertionError('expected rejection or downstream failure')
        assert len(frames().fetchall()) == 12
        ordered_frames = frames().order('frame_index DESC').fetchall()
        assert [row[0] for row in ordered_frames] == list(range(11, -1, -1))
        for relation in (mapped(frames().order('frame_index DESC')),):
            try:
                relation.fetchall()
            except vane.NotImplementedException as error:
                assert 'materialization budget handoff' in str(error), str(error)
            else:
                raise AssertionError('unsupported materialization must fail without waiting')
        repeated = mapped(frames()).to_arrow_table()
        assert sorted(repeated['frame_index'].to_pylist()) == list(range(12))
        assert sorted(result['frame_index'].to_pylist()) == list(range(0, 12, 2))
    finally:
        proceed.touch()
        watchdog.cancel()
        vane.teardown_runner()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, artifact, str(video_path), str(tmp_path), backend],
        env={
            **os.environ,
            "VANE_RUNNER": "local-fast",
            "VANE_PROGRESS": "0",
            "VANE_NATIVE_MEDIA_MAX_CONCURRENT_DECODES": "1",
            "VANE_NATIVE_MEDIA_MAX_OUTPUT_BYTES": "288",
            "VANE_NATIVE_MEDIA_MAX_INFLIGHT_OUTPUT_BYTES": window_bytes,
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_video_flow_window_rejects_an_oversized_frame(video_path):
    providers = os.environ.get("VANE_TEST_NATIVE_MEDIA_PROVIDERS") == "1"
    artifact = "" if providers else str(media._artifact("video"))
    script = r"""
import sys
import vane
artifact, path = sys.argv[1:]
with (vane.connect() if not artifact else
      vane.connect(config={'allow_unsigned_extensions': 'true'})) as con:
    if artifact:
        con.load_extension(artifact)
    else:
        vane.load_installed_extension('native_media', connection=con)
    con.execute("SET video_backend='native'")
    for policy in ('raise', 'skip'):
        try:
            vane.read_video_frames(path, 6, 8, connection=con, on_error=policy).fetchall()
        except vane.OutOfMemoryException as error:
            assert 'VANE_NATIVE_MEDIA_MAX_INFLIGHT_OUTPUT_BYTES' in str(error), str(error)
        else:
            raise AssertionError('impossible flow reservation must fail without waiting')
    assert vane.read_video_frames(path, 6, 8, connection=con, frame_limit=0).fetchall() == []
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, artifact, str(video_path)],
        env={**os.environ, "VANE_RUNNER": "local-fast", "VANE_NATIVE_MEDIA_MAX_INFLIGHT_OUTPUT_BYTES": "143"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from duckdb.datasource import _schema_to_arrow
from duckdb.datasource.video_reader import (
    LimitedVideoFrameTask,
    VideoFrameSource,
    VideoFrameTask,
    _coalesce_video_frame_batches,
    _decode_video_batches,
    _flush_frame_batch,
    _materialize_video_path,
    _resize_frame_batch,
    _s3_filesystem,
    _split_video_path_groups,
    _video_frame_source_manifest_sql,
    _video_frame_source_map_batches,
    _video_source_udf_output_batch_size,
)

_S3_ENV_NAMES = (
    "S3FS_ANON",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "AWS_ROLE_ARN",
    "AWS_ROLE_SESSION_NAME",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_ENDPOINT_URL",
)


@pytest.fixture
def recording_s3_filesystem(monkeypatch):
    import pyarrow.fs as pa_fs

    recorded = {}

    class RecordingS3FileSystem:
        def __init__(self, **kwargs):
            recorded["kwargs"] = kwargs

        def get_file_info(self, _path):
            return type("FileInfo", (), {"size": len(b"video-bytes")})()

        def open_input_file(self, path):
            recorded["path"] = path
            return io.BytesIO(b"video-bytes")

    monkeypatch.setattr(pa_fs, "S3FileSystem", RecordingS3FileSystem)
    return recorded


def _clear_s3_environment(monkeypatch):
    for name in _S3_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_video_s3_reader_uses_default_aws_sdk_chain(
    monkeypatch,
    recording_s3_filesystem,
    tmp_path,
):
    _clear_s3_environment(monkeypatch)

    with _materialize_video_path(
        "s3://media-bucket/clips/example.mp4",
        max_remote_video_bytes=1024,
        remote_temp_dir=str(tmp_path),
    ) as local_path:
        result = Path(local_path).read_bytes()

    assert result == b"video-bytes"
    assert recording_s3_filesystem == {
        "kwargs": {},
        "path": "media-bucket/clips/example.mp4",
    }


def test_remote_video_materialization_reads_bounded_chunks_and_cleans_up(monkeypatch, tmp_path):
    import pyarrow.fs as pa_fs

    import duckdb.datasource.video_reader as video_reader

    chunk_bytes = video_reader._REMOTE_VIDEO_READ_CHUNK_BYTES
    object_size = 2 * chunk_bytes + 17
    read_sizes = []

    class LazyRemoteFile:
        def __init__(self):
            self.remaining = object_size

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, size=-1):
            read_sizes.append(size)
            assert 0 < size <= chunk_bytes
            count = min(size, self.remaining)
            self.remaining -= count
            return b"x" * count

    class FakeFileInfo:
        size = object_size

    class FakeS3FileSystem:
        def __init__(self, **_kwargs):
            pass

        def get_file_info(self, path):
            assert path == "media-bucket/clips/example.mp4"
            return FakeFileInfo()

        def open_input_file(self, path):
            assert path == "media-bucket/clips/example.mp4"
            return LazyRemoteFile()

    monkeypatch.setattr(pa_fs, "S3FileSystem", FakeS3FileSystem)

    with video_reader._materialize_video_path(
        "s3://media-bucket/clips/example.mp4",
        max_remote_video_bytes=object_size,
        remote_temp_dir=str(tmp_path),
    ) as local_path:
        assert Path(local_path).stat().st_size == object_size
        assert Path(local_path).parent == tmp_path

    assert read_sizes == [chunk_bytes, chunk_bytes, chunk_bytes, chunk_bytes]
    assert list(tmp_path.iterdir()) == []


def test_remote_video_materialization_rejects_oversized_stream_and_cleans_up(monkeypatch, tmp_path):
    import pyarrow.fs as pa_fs

    import duckdb.datasource.video_reader as video_reader

    limit = 1024

    class GrowingRemoteFile:
        def __init__(self):
            self.read_count = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, size=-1):
            assert size == video_reader._REMOTE_VIDEO_READ_CHUNK_BYTES
            self.read_count += 1
            return b"x" * limit if self.read_count <= 2 else b""

    class UnknownSizeFileInfo:
        size = -1

    class FakeS3FileSystem:
        def __init__(self, **_kwargs):
            pass

        def get_file_info(self, _path):
            return UnknownSizeFileInfo()

        def open_input_file(self, _path):
            return GrowingRemoteFile()

    monkeypatch.setattr(pa_fs, "S3FileSystem", FakeS3FileSystem)

    with pytest.raises(video_reader.RemoteVideoTooLargeError, match="exceeded limit 1024"):
        with video_reader._materialize_video_path(
            "s3://media-bucket/clips/example.mp4",
            max_remote_video_bytes=limit,
            remote_temp_dir=str(tmp_path),
        ):
            pytest.fail("oversized remote video should not be yielded")

    assert list(tmp_path.iterdir()) == []


def test_remote_video_materialization_allows_unknown_size_metadata(monkeypatch, tmp_path):
    import duckdb.datasource.video_reader as video_reader

    class UnknownSizeFileInfo:
        size = None

    class FakeS3FileSystem:
        def get_file_info(self, _path):
            return UnknownSizeFileInfo()

        def open_input_file(self, _path):
            return io.BytesIO(b"video-bytes")

    monkeypatch.setattr(video_reader, "_s3_filesystem", FakeS3FileSystem)

    with video_reader._materialize_video_path(
        "s3://media-bucket/clips/example.mp4",
        max_remote_video_bytes=1024,
        remote_temp_dir=str(tmp_path),
    ) as local_path:
        assert Path(local_path).read_bytes() == b"video-bytes"

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure_point", ["metadata", "open", "read"])
def test_remote_video_source_io_errors_are_video_read_errors(monkeypatch, tmp_path, failure_point):
    import duckdb.datasource.video_reader as video_reader

    failure = OSError(f"planned {failure_point} failure")
    temporary_files = []

    class FailingRemoteFile:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            raise failure

    class FailingS3FileSystem:
        def get_file_info(self, _path):
            if failure_point == "metadata":
                raise failure
            return type("FileInfo", (), {"size": 1})()

        def open_input_file(self, _path):
            if failure_point == "open":
                raise failure
            return FailingRemoteFile()

    real_named_temporary_file = video_reader.tempfile.NamedTemporaryFile

    def recording_named_temporary_file(*args, **kwargs):
        temporary = real_named_temporary_file(*args, **kwargs)
        temporary_files.append(temporary)
        return temporary

    monkeypatch.setattr(video_reader, "_s3_filesystem", FailingS3FileSystem)
    monkeypatch.setattr(video_reader.tempfile, "NamedTemporaryFile", recording_named_temporary_file)

    with pytest.raises(video_reader.VideoReadError, match=f"planned {failure_point} failure") as raised:
        with video_reader._materialize_video_path(
            "s3://media-bucket/clips/example.mp4",
            max_remote_video_bytes=1024,
            remote_temp_dir=str(tmp_path),
        ):
            pytest.fail("a failed remote read must not yield a decoder path")

    assert raised.value.__cause__ is failure
    assert raised.value.video_path == "s3://media-bucket/clips/example.mp4"
    assert len(temporary_files) == (0 if failure_point == "metadata" else 1)
    assert all(temporary.closed for temporary in temporary_files)
    assert list(tmp_path.iterdir()) == []


def test_remote_video_tempfile_errors_are_not_classified_as_bad_input(
    recording_s3_filesystem,
    tmp_path,
):
    import duckdb.datasource.video_reader as video_reader

    missing_temp_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError) as raised:
        with video_reader._materialize_video_path(
            "s3://media-bucket/clips/example.mp4",
            max_remote_video_bytes=1024,
            remote_temp_dir=str(missing_temp_dir),
        ):
            pytest.fail("temporary-file creation failure must not yield a decoder path")

    assert not isinstance(raised.value, video_reader.VideoReadError)


def test_video_source_identity_distinguishes_same_basename_sources():
    import duckdb.datasource.video_reader as video_reader

    path_a, source_id_a = video_reader._video_source_identity("s3://bucket-a/train/clip.mp4")
    path_b, source_id_b = video_reader._video_source_identity("s3://bucket-b/eval/clip.mp4")

    assert path_a == "s3://bucket-a/train/clip.mp4"
    assert path_b == "s3://bucket-b/eval/clip.mp4"
    assert len(source_id_a) == 64
    assert len(source_id_b) == 64
    assert source_id_a != source_id_b


def test_video_source_identity_preserves_s3_object_key_semantics():
    import duckdb.datasource.video_reader as video_reader

    path, _ = video_reader._video_source_identity("s3://MEDIA-BUCKET/a//../clip.mp4")

    assert path == "s3://media-bucket/a//../clip.mp4"


def test_video_source_identity_resolves_symlink_before_parent_component(tmp_path):
    import duckdb.datasource.video_reader as video_reader

    decoded_root = tmp_path / "decoded"
    decoded_dir = decoded_root / "nested"
    decoded_dir.mkdir(parents=True)
    decoded_video = decoded_root / "clip.mp4"
    decoded_video.touch()
    direct_video = tmp_path / "clip.mp4"
    direct_video.touch()
    link = tmp_path / "link"
    link.symlink_to(decoded_dir, target_is_directory=True)

    canonical_path, source_id = video_reader._video_source_identity(str(link / ".." / "clip.mp4"))
    direct_path, direct_source_id = video_reader._video_source_identity(str(direct_video))

    assert canonical_path == str(decoded_video.resolve())
    assert direct_path == str(direct_video.resolve())
    assert source_id != direct_source_id


def test_video_decode_opens_the_resolved_local_source(monkeypatch, tmp_path):
    import duckdb.datasource.video_reader as video_reader

    decoded_root = tmp_path / "decoded"
    decoded_dir = decoded_root / "nested"
    decoded_dir.mkdir(parents=True)
    decoded_video = decoded_root / "clip.mp4"
    decoded_video.touch()
    link = tmp_path / "link"
    link.symlink_to(decoded_dir, target_is_directory=True)
    input_path = str(link / ".." / "clip.mp4")
    opened_paths = []

    def fake_open_decoder(path, **_kwargs):
        opened_paths.append(path)
        return []

    monkeypatch.setattr(video_reader, "_open_decord_reader", fake_open_decoder)

    assert (
        list(
            _decode_video_batches(
                input_path,
                height=2,
                width=2,
                max_partition_bytes=1024,
            )
        )
        == []
    )
    assert opened_paths == [str(decoded_video.resolve())]


def test_video_read_errors_are_pickle_safe():
    import pickle

    import duckdb.datasource.video_reader as video_reader

    for error_type in (
        video_reader.VideoReadError,
        video_reader.RemoteVideoTooLargeError,
        video_reader.SourceFrameTooLargeError,
    ):
        error = error_type("s3://bucket/clip.mp4", "test failure")
        restored = pickle.loads(pickle.dumps(error))

        assert type(restored) is error_type
        assert restored.video_path == error.video_path
        assert restored.message == error.message
        assert str(restored) == str(error)


@pytest.mark.parametrize(
    "credential_env",
    [
        {"AWS_PROFILE": "media-reader"},
        {
            "AWS_ROLE_ARN": "arn:aws:iam::123456789012:role/media-reader",
            "AWS_ROLE_SESSION_NAME": "vane-video-test",
            "AWS_WEB_IDENTITY_TOKEN_FILE": "/var/run/secrets/aws/token",
        },
    ],
    ids=["profile", "web-identity-role"],
)
def test_video_s3_reader_leaves_profile_and_role_credentials_to_sdk(
    monkeypatch,
    recording_s3_filesystem,
    credential_env,
):
    _clear_s3_environment(monkeypatch)
    for name, value in credential_env.items():
        monkeypatch.setenv(name, value)

    _s3_filesystem()

    assert recording_s3_filesystem["kwargs"] == {}


def test_video_s3_reader_leaves_static_session_credentials_to_sdk(monkeypatch, recording_s3_filesystem):
    _clear_s3_environment(monkeypatch)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "temporary-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "temporary-secret-key")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "temporary-session-token")

    _s3_filesystem()

    assert recording_s3_filesystem["kwargs"] == {}


@pytest.mark.parametrize(
    "partial_credentials",
    [
        {"AWS_ACCESS_KEY_ID": "access-key-without-secret"},
        {"AWS_SECRET_ACCESS_KEY": "secret-key-without-access"},
    ],
    ids=["access-key-only", "secret-key-only"],
)
def test_video_s3_reader_leaves_partial_static_credentials_to_sdk(
    monkeypatch,
    recording_s3_filesystem,
    partial_credentials,
):
    _clear_s3_environment(monkeypatch)
    for name, value in partial_credentials.items():
        monkeypatch.setenv(name, value)

    _s3_filesystem()

    assert recording_s3_filesystem["kwargs"] == {}


@pytest.mark.parametrize("region_env", ["AWS_REGION", "AWS_DEFAULT_REGION"])
def test_video_s3_reader_passes_custom_https_endpoint_and_region(
    monkeypatch,
    recording_s3_filesystem,
    region_env,
):
    _clear_s3_environment(monkeypatch)
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://objects.example.test:9443/prefix")
    monkeypatch.setenv(region_env, "eu-west-1")

    _s3_filesystem()

    assert recording_s3_filesystem["kwargs"] == {
        "endpoint_override": "objects.example.test:9443",
        "region": "eu-west-1",
        "scheme": "https",
    }


@pytest.mark.parametrize(
    ("endpoint_url", "expected_kwargs"),
    [
        (
            "objects.example.test:9443/prefix",
            {"endpoint_override": "objects.example.test:9443"},
        ),
        (
            "//objects.example.test:9443/prefix",
            {"endpoint_override": "objects.example.test:9443"},
        ),
        (
            "http://127.0.0.1:9000/prefix",
            {"endpoint_override": "127.0.0.1:9000", "scheme": "http"},
        ),
    ],
)
def test_video_s3_reader_normalizes_endpoint_override(
    monkeypatch,
    recording_s3_filesystem,
    endpoint_url,
    expected_kwargs,
):
    _clear_s3_environment(monkeypatch)
    monkeypatch.setenv("AWS_ENDPOINT_URL", endpoint_url)

    _s3_filesystem()

    assert recording_s3_filesystem["kwargs"] == expected_kwargs


def test_video_s3_reader_does_not_enable_anonymous_mode_when_explicitly_disabled(
    monkeypatch,
    recording_s3_filesystem,
):
    _clear_s3_environment(monkeypatch)
    monkeypatch.setenv("S3FS_ANON", "false")

    _s3_filesystem()

    assert recording_s3_filesystem["kwargs"] == {}


def test_video_s3_reader_uses_anonymous_mode_only_when_explicit(monkeypatch, recording_s3_filesystem):
    _clear_s3_environment(monkeypatch)
    monkeypatch.setenv("S3FS_ANON", "true")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ignored-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ignored-secret-key")

    _s3_filesystem()

    assert recording_s3_filesystem["kwargs"] == {"anonymous": True}


@pytest.mark.parametrize(("configured", "expected"), [(None, 1), ("1", 1), ("4", 4)])
def test_max_concurrent_decodes_accepts_positive_integers(configured, expected):
    env = os.environ.copy()
    if configured is None:
        env.pop("VANE_MAX_CONCURRENT_DECODES", None)
    else:
        env["VANE_MAX_CONCURRENT_DECODES"] = configured
    script = f"""
import duckdb.datasource.video_reader as video_reader

assert video_reader._MAX_CONCURRENT_DECODES == {expected}
for _ in range({expected}):
    assert video_reader._decode_semaphore.acquire(blocking=False)
assert not video_reader._decode_semaphore.acquire(blocking=False)
"""

    subprocess.run([sys.executable, "-c", script], check=True, env=env, timeout=10)


@pytest.mark.parametrize("configured", ["0", "-1", "invalid", "", "   "])
def test_max_concurrent_decodes_rejects_invalid_values(configured):
    env = os.environ.copy()
    env["VANE_MAX_CONCURRENT_DECODES"] = configured

    result = subprocess.run(
        [sys.executable, "-c", "import duckdb.datasource.video_reader"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )

    assert result.returncode != 0
    assert "VANE_MAX_CONCURRENT_DECODES must be an integer >= 1" in result.stderr


def test_video_frame_source_uses_one_ordered_task_for_frame_limit():
    source = VideoFrameSource(["a.avi", "b.avi"], height=8, width=8, frame_limit=3)

    tasks = list(source.get_tasks())

    assert len(tasks) == 1
    assert isinstance(tasks[0], LimitedVideoFrameTask)
    assert tasks[0].paths == ["a.avi", "b.avi"]
    assert tasks[0].max_frames == 3


def test_video_frame_source_keeps_parallel_per_file_tasks_without_frame_limit():
    source = VideoFrameSource(["a.avi", "b.avi"], height=8, width=8)

    tasks = list(source.get_tasks())

    assert len(tasks) == 2
    assert all(isinstance(task, VideoFrameTask) for task in tasks)


def test_video_frame_source_manifest_groups_paths_like_ray_read_tasks():
    source = VideoFrameSource(
        [f"clip-{index}.avi" for index in range(11)],
        height=8,
        width=8,
        read_task_count=4,
    )

    sql = _video_frame_source_manifest_sql(source)

    assert [len(group) for group in _split_video_path_groups(source.paths, 4)] == [3, 3, 3, 2]
    assert sql.count("list_value(") == 4
    assert "video_paths::VARCHAR[]" in sql
    assert "max_remote_video_bytes::BIGINT" in sql
    assert "max_source_frame_bytes::BIGINT" in sql
    assert "max_source_path_bytes::BIGINT" in sql
    assert "on_error::VARCHAR" in sql


@pytest.mark.parametrize("on_error", ["", "ignore", "warn"])
def test_video_frame_source_rejects_unknown_error_mode(on_error):
    with pytest.raises(ValueError, match="on_error must be one of: raise, skip"):
        VideoFrameSource(["a.avi"], on_error=on_error)


def test_video_frame_source_rejects_decoder_frame_above_cap():
    with pytest.raises(
        ValueError,
        match="bounded decoder frame size 384 exceeds max_source_frame_bytes 383",
    ):
        VideoFrameSource(
            ["a.avi"],
            height=4,
            width=4,
            max_source_frame_bytes=383,
        )


def test_video_source_uses_ray_soft_block_row_boundary():
    assert _video_source_udf_output_batch_size(640, 640, 128 * 1024**2) == 109


def test_video_source_batch_size_accounts_for_provenance_columns():
    import duckdb.datasource.video_reader as video_reader

    target_bytes = 1024
    max_source_path_bytes = 100
    batch_size = _video_source_udf_output_batch_size(
        1,
        1,
        target_bytes,
        max_source_path_bytes=max_source_path_bytes,
    )

    assert batch_size == 6
    assert (
        video_reader._video_output_batch_bytes(
            batch_size - 1,
            height=1,
            width=1,
            max_source_path_bytes=max_source_path_bytes,
        )
        <= target_bytes
    )
    assert (
        video_reader._video_output_batch_bytes(
            batch_size,
            height=1,
            width=1,
            max_source_path_bytes=max_source_path_bytes,
        )
        > target_bytes
    )
    below_target = _flush_frame_batch(
        "s" * video_reader._SOURCE_ID_BYTES,
        "p" * max_source_path_bytes,
        np.zeros((batch_size - 1, 1, 1, 3), dtype=np.uint8),
        np.arange(batch_size - 1, dtype=np.int64),
        batch_size - 1,
    )
    crossing_target = _flush_frame_batch(
        "s" * video_reader._SOURCE_ID_BYTES,
        "p" * max_source_path_bytes,
        np.zeros((batch_size, 1, 1, 3), dtype=np.uint8),
        np.arange(batch_size, dtype=np.int64),
        batch_size,
    )
    assert below_target.nbytes <= target_bytes
    assert crossing_target.nbytes > target_bytes


def test_video_source_batch_size_respects_arrow_string_offset_limit():
    import duckdb.datasource.video_reader as video_reader

    max_source_path_bytes = 1024
    expected_limit = video_reader._ARROW_STRING_DATA_MAX_BYTES // max_source_path_bytes

    assert (
        _video_source_udf_output_batch_size(
            1,
            1,
            10**15,
            max_source_path_bytes=max_source_path_bytes,
        )
        == expected_limit
    )
    with pytest.raises(ValueError, match="exceeding Arrow's .* UTF-8 offset limit"):
        video_reader._constant_string_array(
            "x" * max_source_path_bytes,
            expected_limit + 1,
        )


def test_video_source_transport_does_not_resplit_ray_soft_block(monkeypatch):
    import duckdb.datasource.video_reader as video_reader

    monkeypatch.setenv("VANE_VIDEO_SOURCE_UDF_BACKEND", "ray_task")
    target_bytes = 128 * 1024**2
    kwargs = video_reader._video_source_udf_kwargs(
        height=640,
        width=640,
        max_partition_bytes=target_bytes,
    )

    assert kwargs["output_batch_size"] == 109
    assert kwargs["output_target_max_bytes"] == 2 * target_bytes
    assert kwargs["preserve_compute_batch_boundaries"] is True


def test_video_source_coalesces_file_tails_within_one_read_task():
    frames = np.zeros((3, 2, 2, 3), dtype=np.uint8)

    def batches():
        for name in ("a.avi", "b.avi"):
            yield pa.record_batch(
                {
                    "video_path": [name] * 3,
                    "frame_index": [0, 1, 2],
                    "frame": pa.FixedShapeTensorArray.from_numpy_ndarray(frames),
                }
            )

    output = list(_coalesce_video_frame_batches(batches(), target_rows=5))

    assert [table.num_rows for table in output] == [5, 1]
    assert output[0].column("video_path").to_pylist() == ["a.avi"] * 3 + ["b.avi"] * 2


def test_video_decode_batches_do_not_mutate_emitted_arrow_buffers(monkeypatch):
    import duckdb.datasource.video_reader as video_reader

    class FakeFrame:
        def __init__(self, value):
            self._value = value

        def asnumpy(self):
            return np.full((2, 32, 3), self._value, dtype=np.uint8)

    monkeypatch.setattr(
        video_reader,
        "_open_decord_reader",
        lambda _path, **_kwargs: [FakeFrame(i) for i in range(5)],
    )
    monkeypatch.setattr(video_reader, "_VIDEO_RESIZE_THREADS", 1)
    source_path, _ = video_reader._video_source_identity("clip.avi")
    max_partition_bytes = video_reader._video_output_batch_bytes(
        1,
        height=2,
        width=2,
        max_source_path_bytes=len(source_path.encode("utf-8")),
    )

    batches = list(
        _decode_video_batches(
            "clip.avi",
            height=2,
            width=2,
            # One complete output row reaches the soft target, so the crossing
            # row makes each full batch contain two rows.
            max_partition_bytes=max_partition_bytes,
        )
    )

    values = [batch.column("frame").to_numpy_ndarray()[:, 0, 0, 0].tolist() for batch in batches]
    assert values == [[0, 1], [2, 3], [4]]
    expected_path, expected_source_id = video_reader._video_source_identity("clip.avi")
    assert batches[0].column("video_path").to_pylist() == [expected_path, expected_path]
    assert batches[0].column("source_id").to_pylist() == [expected_source_id, expected_source_id]


def test_video_decoder_uses_one_thread_for_bounded_native_buffering(monkeypatch):
    import duckdb.datasource.video_reader as video_reader

    calls = []

    class FakeVideoReader:
        def __init__(self, path, *, width, height, num_threads):
            calls.append((path, width, height, num_threads))

    class FakeDecordModule:
        VideoReader = FakeVideoReader

    monkeypatch.setattr(video_reader, "_import_video_dependency", lambda *_args: FakeDecordModule)

    reader = video_reader._open_decord_reader("clip.avi", width=320, height=180)

    assert isinstance(reader, FakeVideoReader)
    assert calls == [("clip.avi", 320, 180, 1)]


@pytest.mark.parametrize("error_type", [OSError, RuntimeError])
def test_video_decoder_open_errors_use_public_source_identity(monkeypatch, error_type):
    import duckdb.datasource.video_reader as video_reader

    failure = error_type("planned decoder open failure")

    class FailingVideoReader:
        def __init__(self, *_args, **_kwargs):
            raise failure

    class FakeDecordModule:
        VideoReader = FailingVideoReader

    monkeypatch.setattr(video_reader, "_import_video_dependency", lambda *_args: FakeDecordModule)

    with pytest.raises(video_reader.VideoReadError, match="planned decoder open failure") as raised:
        video_reader._open_decord_reader(
            "/tmp/materialized.mp4",
            width=320,
            height=180,
            source_path="s3://media-bucket/clips/example.mp4",
        )

    assert raised.value.__cause__ is failure
    assert raised.value.video_path == "s3://media-bucket/clips/example.mp4"


def test_video_decoder_iteration_wraps_direct_decord_errors(monkeypatch):
    import duckdb.datasource.video_reader as video_reader

    class FakeDecordError(Exception):
        pass

    class FakeDecordLimitReachedError(Exception):
        pass

    class FakeDecordBase:
        DECORDError = FakeDecordError
        DECORDLimitReachedError = FakeDecordLimitReachedError

    failure = FakeDecordError("planned frame decode failure")

    class FailingReader:
        def __iter__(self):
            return self

        def __next__(self):
            raise failure

    real_import_module = video_reader.importlib.import_module

    def fake_import_module(module_name, *args, **kwargs):
        if module_name == "decord._ffi.base":
            return FakeDecordBase
        return real_import_module(module_name, *args, **kwargs)

    monkeypatch.setattr(video_reader.importlib, "import_module", fake_import_module)
    assert video_reader._is_decord_error(FakeDecordLimitReachedError("recovery limit"))

    with pytest.raises(video_reader.VideoReadError, match="planned frame decode failure") as raised:
        list(
            video_reader._iter_decord_frames(
                FailingReader(),
                video_path="clip.avi",
                max_frames=None,
            )
        )

    assert raised.value.__cause__ is failure
    assert raised.value.video_path == "clip.avi"


def test_video_frame_conversion_does_not_wrap_generic_runtime_errors(monkeypatch):
    import duckdb.datasource.video_reader as video_reader

    failure = RuntimeError("planned frame conversion invariant failure")

    class FailingFrame:
        def asnumpy(self):
            raise failure

    monkeypatch.setattr(video_reader, "_is_decord_error", lambda _exc: False)

    with pytest.raises(RuntimeError, match="planned frame conversion invariant failure") as raised:
        video_reader._decord_frame_asnumpy(FailingFrame(), video_path="clip.avi")

    assert raised.value is failure


def test_video_decode_aligns_narrow_decord_output_before_exact_resize(monkeypatch):
    import duckdb.datasource.video_reader as video_reader

    class FakeFrame:
        shape = (2, 32, 3)

        def asnumpy(self):
            return np.zeros(self.shape, dtype=np.uint8)

    calls = []

    def fake_open_decoder(path, **kwargs):
        calls.append((path, kwargs))
        return [FakeFrame()]

    monkeypatch.setattr(video_reader, "_open_decord_reader", fake_open_decoder)

    batches = list(
        _decode_video_batches(
            "clip.avi",
            height=2,
            width=2,
            max_partition_bytes=1024,
            max_source_frame_bytes=2 * 32 * 3,
        )
    )

    source_path, _ = video_reader._video_source_identity("clip.avi")
    assert calls == [(source_path, {"width": 32, "height": 2, "source_path": source_path})]
    assert batches[0].column("frame").type.shape == [2, 2, 3]


def test_video_decode_does_not_advance_reader_past_frame_limit(monkeypatch):
    import duckdb.datasource.video_reader as video_reader

    class FakeFrame:
        shape = (2, 32, 3)

        def asnumpy(self):
            return np.zeros(self.shape, dtype=np.uint8)

    class BoundaryReader:
        def __init__(self):
            self.next_calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.next_calls += 1
            if self.next_calls == 1:
                return FakeFrame()
            pytest.fail("decoder advanced beyond max_frames")

    reader = BoundaryReader()
    monkeypatch.setattr(video_reader, "_open_decord_reader", lambda _path, **_kwargs: reader)

    batches = list(
        video_reader._decode_video_batches(
            "clip.avi",
            height=2,
            width=2,
            max_partition_bytes=1024,
            max_frames=1,
        )
    )

    assert sum(batch.num_rows for batch in batches) == 1
    assert reader.next_calls == 1


def test_remote_video_decode_uses_temporary_path_and_preserves_remote_identity(
    monkeypatch,
    recording_s3_filesystem,
    tmp_path,
):
    import duckdb.datasource.video_reader as video_reader

    class FakeFrame:
        def asnumpy(self):
            return np.zeros((2, 32, 3), dtype=np.uint8)

    decoder_paths = []

    def fake_open_decoder(path, **kwargs):
        assert kwargs == {
            "width": 32,
            "height": 2,
            "source_path": "s3://media-bucket/clips/example.mp4",
        }
        decoder_path = Path(path)
        assert decoder_path.is_file()
        decoder_paths.append(decoder_path)
        return [FakeFrame()]

    monkeypatch.setattr(video_reader, "_open_decord_reader", fake_open_decoder)

    batches = list(
        _decode_video_batches(
            "s3://media-bucket/clips/example.mp4",
            height=2,
            width=2,
            max_partition_bytes=1024,
            max_remote_video_bytes=1024,
            remote_temp_dir=str(tmp_path),
        )
    )

    source_path, source_id = video_reader._video_source_identity("s3://media-bucket/clips/example.mp4")
    assert len(decoder_paths) == 1
    assert not decoder_paths[0].exists()
    assert batches[0].column("video_path").to_pylist() == [source_path]
    assert batches[0].column("source_id").to_pylist() == [source_id]
    assert recording_s3_filesystem["path"] == "media-bucket/clips/example.mp4"


def test_remote_video_decode_cleans_temporary_path_when_consumer_stops_early(
    monkeypatch,
    recording_s3_filesystem,
    tmp_path,
):
    import duckdb.datasource.video_reader as video_reader

    class FakeFrame:
        def asnumpy(self):
            return np.zeros((2, 32, 3), dtype=np.uint8)

    decoder_paths = []

    def fake_open_decoder(path, **kwargs):
        assert kwargs == {
            "width": 32,
            "height": 2,
            "source_path": "s3://media-bucket/clips/example.mp4",
        }
        decoder_paths.append(Path(path))
        return [FakeFrame(), FakeFrame(), FakeFrame()]

    monkeypatch.setattr(video_reader, "_open_decord_reader", fake_open_decoder)
    source_path, _ = video_reader._video_source_identity("s3://media-bucket/clips/example.mp4")
    max_partition_bytes = video_reader._video_output_batch_bytes(
        1,
        height=2,
        width=2,
        max_source_path_bytes=len(source_path.encode("utf-8")),
    )
    batches = _decode_video_batches(
        "s3://media-bucket/clips/example.mp4",
        height=2,
        width=2,
        max_partition_bytes=max_partition_bytes,
        max_remote_video_bytes=1024,
        remote_temp_dir=str(tmp_path),
    )

    first = next(batches)
    assert first.num_rows == 2
    assert decoder_paths[0].is_file()

    batches.close()

    assert not decoder_paths[0].exists()


def test_video_decode_keeps_raw_resize_window_bounded(monkeypatch):
    import duckdb.datasource.video_reader as video_reader

    class FakeFrame:
        def __init__(self, value):
            self._value = value

        def asnumpy(self):
            return np.full((4, 32, 3), self._value, dtype=np.uint8)

    resize_window_sizes = []

    def fake_resize(frames, *, width, height, executor=None):
        del executor
        resize_window_sizes.append(len(frames))
        return [np.full((height, width, 3), frame[0, 0, 0], dtype=np.uint8) for frame in frames]

    monkeypatch.setattr(
        video_reader,
        "_open_decord_reader",
        lambda _path, **_kwargs: [FakeFrame(i) for i in range(10)],
    )
    monkeypatch.setattr(video_reader, "_resize_frame_batch", fake_resize)
    monkeypatch.setattr(video_reader, "_VIDEO_RESIZE_THREADS", 3)
    source_path, _ = video_reader._video_source_identity("clip.avi")
    max_partition_bytes = video_reader._video_output_batch_bytes(
        8,
        height=2,
        width=2,
        max_source_path_bytes=len(source_path.encode("utf-8")),
    )

    batches = list(
        _decode_video_batches(
            "clip.avi",
            height=2,
            width=2,
            max_partition_bytes=max_partition_bytes,
        )
    )

    assert [batch.num_rows for batch in batches] == [9, 1]
    assert resize_window_sizes
    assert max(resize_window_sizes) <= 3


def test_video_decode_rejects_frame_bound_before_opening_decoder(monkeypatch):
    import duckdb.datasource.video_reader as video_reader

    def fail_open(*_args, **_kwargs):
        pytest.fail("decoder must not open when its requested frame exceeds the cap")

    monkeypatch.setattr(video_reader, "_open_decord_reader", fail_open)

    with pytest.raises(
        video_reader.SourceFrameTooLargeError,
        match="bounded decoder frame size 384 exceeds limit 383",
    ):
        list(
            _decode_video_batches(
                "clip.avi",
                height=4,
                width=4,
                max_partition_bytes=1024,
                max_source_frame_bytes=383,
            )
        )


def test_datasource_schema_supports_fixed_shape_tensor_entries():
    schema = _schema_to_arrow(
        {
            "frame_index": "BIGINT",
            "frame": {"kind": "tensor", "dtype": "UINT8", "shape": [4, 5, 3]},
        }
    )

    assert schema.field("frame_index").type == pa.int64()
    assert schema.field("frame").type == pa.fixed_shape_tensor(pa.uint8(), (4, 5, 3))


def test_video_frame_source_schema_declares_typed_frame_not_blob():
    source = VideoFrameSource(["a.avi"], height=4, width=5)

    assert source.schema == {
        "source_id": "VARCHAR",
        "video_path": "VARCHAR",
        "frame_index": "BIGINT",
        "frame": {"kind": "tensor", "dtype": "UINT8", "shape": [4, 5, 3]},
    }


def test_read_datasource_uses_datasource_udf_relation_hook():
    import duckdb
    from duckdb.datasource import DataSource, read_datasource

    class HookSource(DataSource):
        @property
        def schema(self):
            return {"value": "INTEGER"}

        def get_tasks(self):
            raise AssertionError("native datasource scan should not run")

        def to_udf_relation(self, con):
            return con.sql("select 42::INTEGER as value")

    con = duckdb.connect()

    assert read_datasource(HookSource(), con=con).fetchall() == [(42,)]


def test_video_frame_source_read_datasource_builds_hidden_udf_plan(monkeypatch):
    import duckdb
    from duckdb.datasource import read_datasource

    monkeypatch.setenv("VANE_VIDEO_SOURCE_UDF_BACKEND", "ray_task")
    con = duckdb.connect()

    plan = read_datasource(VideoFrameSource(["a.avi"], height=8, width=9), con=con).explain()
    compact_plan = "".join(ch for ch in plan if ch.isalnum() or ch == "_")

    assert "STREAMING_UDF" in plan
    assert "_video_frame_source_map_batches" in compact_plan
    assert "execution_backend" in plan
    assert "ray_task" in plan
    assert "udf_queue_depth" not in compact_plan
    assert "udf_max_outstanding_batches" not in compact_plan
    assert "udf_max_ready_rows" not in compact_plan


def test_video_source_udf_identity_is_assigned_by_physical_graph(monkeypatch):
    import duckdb.datasource.video_reader as video_reader

    monkeypatch.setenv("VANE_VIDEO_SOURCE_UDF_BACKEND", "ray_task")
    monkeypatch.setenv("VANE_VIDEO_SOURCE_UDF_VIDEOS_PER_TASK", "1")
    kwargs = video_reader._video_source_udf_kwargs()

    assert kwargs["execution_backend"] == "ray_task"
    assert kwargs["memory_bytes"] == 512 * 1024**2
    assert kwargs["cpus"] == 1.0
    assert "queue_depth" not in kwargs
    assert "query_id" not in kwargs
    assert "fragment_id" not in kwargs
    assert "operator_id" not in kwargs
    assert "max_outstanding_batches" not in kwargs


def test_video_source_udf_cpu_default_accounts_for_resize_pool(monkeypatch):
    import duckdb.datasource.video_reader as video_reader

    monkeypatch.delenv("VANE_VIDEO_SOURCE_UDF_CPUS", raising=False)
    monkeypatch.setattr(video_reader, "_VIDEO_RESIZE_THREADS", 3)

    assert video_reader._video_source_udf_kwargs()["cpus"] == 3.0


def test_video_source_udf_cpu_allocation_is_overridable(monkeypatch):
    import duckdb.datasource.video_reader as video_reader

    monkeypatch.setenv("VANE_VIDEO_SOURCE_UDF_CPUS", "2.5")
    monkeypatch.setattr(video_reader, "_VIDEO_RESIZE_THREADS", 4)

    assert video_reader._video_source_udf_kwargs()["cpus"] == 2.5


def test_video_source_udf_cpu_allocation_must_be_positive(monkeypatch):
    import duckdb.datasource.video_reader as video_reader

    monkeypatch.setenv("VANE_VIDEO_SOURCE_UDF_CPUS", "0")

    with pytest.raises(ValueError, match="VANE_VIDEO_SOURCE_UDF_CPUS must be positive"):
        video_reader._video_source_udf_kwargs()


def test_video_source_udf_memory_is_stage_specific(monkeypatch):
    import duckdb.datasource.video_reader as video_reader

    monkeypatch.setenv("VANE_VIDEO_SOURCE_UDF_BACKEND", "ray_task")
    monkeypatch.setenv("VANE_VIDEO_SOURCE_UDF_MEMORY_BYTES", "268435456")
    monkeypatch.setenv("VANE_UDF_TASK_HEAP_BYTES", "1073741824")
    monkeypatch.delenv("VANE_VIDEO_SOURCE_UDF_OUTPUT_BATCH_SIZE", raising=False)

    expected_peak = video_reader._video_source_peak_memory_bytes(
        height=640,
        width=480,
        max_partition_bytes=10 * 1024**2,
        max_source_frame_bytes=128 * 1024**2,
    )
    assert video_reader._video_source_udf_kwargs()["memory_bytes"] == expected_peak

    monkeypatch.setenv("VANE_VIDEO_SOURCE_UDF_BACKEND", "subprocess_task")
    assert "memory_bytes" not in video_reader._video_source_udf_kwargs()


def test_video_source_udf_memory_covers_large_output_partition(monkeypatch):
    import duckdb.datasource.video_reader as video_reader

    monkeypatch.setenv("VANE_VIDEO_SOURCE_UDF_BACKEND", "ray_task")
    monkeypatch.delenv("VANE_VIDEO_SOURCE_UDF_MEMORY_BYTES", raising=False)
    monkeypatch.delenv("VANE_VIDEO_SOURCE_UDF_OUTPUT_BATCH_SIZE", raising=False)
    max_partition_bytes = 128 * 1024**2
    expected_peak = video_reader._video_source_peak_memory_bytes(
        height=640,
        width=640,
        max_partition_bytes=max_partition_bytes,
        max_source_frame_bytes=128 * 1024**2,
    )

    kwargs = video_reader._video_source_udf_kwargs(
        height=640,
        width=640,
        max_partition_bytes=max_partition_bytes,
    )

    assert expected_peak > 512 * 1024**2
    assert kwargs["memory_bytes"] == expected_peak


def test_video_source_peak_memory_accounts_for_provenance_and_decord_copy_overlap(monkeypatch):
    import duckdb.datasource.video_reader as video_reader

    monkeypatch.setattr(video_reader, "_VIDEO_RESIZE_THREADS", 2)
    height = 2
    width = 3
    frame_bytes = height * width * 3
    max_partition_bytes = 500
    max_source_frame_bytes = 1000
    max_source_path_bytes = 10
    decode_batch_size = video_reader._video_source_udf_output_batch_size(
        height,
        width,
        max_partition_bytes,
        max_source_path_bytes=max_source_path_bytes,
    )
    transport_batch_size = 7

    peak_bytes = video_reader._video_source_peak_memory_bytes(
        height=height,
        width=width,
        max_partition_bytes=max_partition_bytes,
        max_source_frame_bytes=max_source_frame_bytes,
        max_source_path_bytes=max_source_path_bytes,
        output_batch_size=transport_batch_size,
    )
    output_bytes = video_reader._video_output_batch_bytes(
        transport_batch_size,
        height=height,
        width=width,
        max_source_path_bytes=max_source_path_bytes,
    )

    assert decode_batch_size == 5
    assert peak_bytes == (
        video_reader._REMOTE_VIDEO_READ_CHUNK_BYTES
        + video_reader._VIDEO_DECODER_MEMORY_HEADROOM_BYTES
        + 2 * output_bytes
        + 2 * (max_source_frame_bytes + frame_bytes)
        + max_source_frame_bytes
    )


def test_video_source_udf_memory_must_be_positive(monkeypatch):
    import duckdb.datasource.video_reader as video_reader

    monkeypatch.setenv("VANE_VIDEO_SOURCE_UDF_BACKEND", "ray_task")
    monkeypatch.setenv("VANE_VIDEO_SOURCE_UDF_MEMORY_BYTES", "0")

    with pytest.raises(ValueError, match="VANE_VIDEO_SOURCE_UDF_MEMORY_BYTES must be positive"):
        video_reader._video_source_udf_kwargs()


def test_video_frame_source_map_batches_reads_manifest(monkeypatch):
    import duckdb.datasource.video_reader as video_reader

    calls = []
    frames = np.arange(2 * 8 * 9 * 3, dtype=np.uint8).reshape(2, 8, 9, 3)

    def fake_decode(video_path, *, height, width, max_partition_bytes, max_frames=None, **_kwargs):
        calls.append((video_path, height, width, max_partition_bytes, max_frames))
        yield pa.record_batch(
            {
                "video_path": [video_path, video_path],
                "frame_index": [0, 1],
                "frame": pa.FixedShapeTensorArray.from_numpy_ndarray(frames),
            }
        )

    monkeypatch.setattr(video_reader, "_wait_for_memory", lambda: None)
    monkeypatch.setattr(video_reader, "_decode_video_batches", fake_decode)
    manifest = pa.table(
        {
            "video_path": ["a.avi", "b.avi"],
            "height": [8, 8],
            "width": [9, 9],
            "max_partition_bytes": [1024, 1024],
            "frame_limit": pa.array([None, None], type=pa.int64()),
            "max_remote_video_bytes": [4096, 4096],
            "max_source_frame_bytes": [2048, 2048],
            "max_source_path_bytes": [
                video_reader._video_source_path_bytes("a.avi"),
                video_reader._video_source_path_bytes("b.avi"),
            ],
            "on_error": ["raise", "raise"],
            "remote_temp_dir": pa.array([None, None], type=pa.string()),
        }
    )

    tables = list(_video_frame_source_map_batches(manifest))

    assert calls == [
        ("a.avi", 8, 9, 1024, None),
        ("b.avi", 8, 9, 1024, None),
    ]
    assert [table.select(["video_path", "frame_index"]).to_pydict() for table in tables] == [
        {"video_path": ["a.avi", "a.avi"], "frame_index": [0, 1]},
        {"video_path": ["b.avi", "b.avi"], "frame_index": [0, 1]},
    ]
    for table in tables:
        np.testing.assert_array_equal(table.column("frame").combine_chunks().to_numpy_ndarray(), frames)


def test_video_frame_source_map_batches_honors_global_frame_limit(monkeypatch):
    import duckdb.datasource.video_reader as video_reader

    calls = []

    def fake_decode(video_path, *, height, width, max_partition_bytes, max_frames=None, **_kwargs):
        calls.append((video_path, max_frames))
        row_count = min(2, max_frames if max_frames is not None else 2)
        if row_count > 0:
            frames = np.zeros((row_count, 8, 9, 3), dtype=np.uint8)
            yield pa.record_batch(
                {
                    "video_path": [video_path] * row_count,
                    "frame_index": list(range(row_count)),
                    "frame": pa.FixedShapeTensorArray.from_numpy_ndarray(frames),
                }
            )

    monkeypatch.setattr(video_reader, "_wait_for_memory", lambda: None)
    monkeypatch.setattr(video_reader, "_decode_video_batches", fake_decode)
    manifest = pa.table(
        {
            "video_paths": pa.array([["a.avi", "b.avi"]], type=pa.list_(pa.string())),
            "height": [8],
            "width": [9],
            "max_partition_bytes": [1024],
            "frame_limit": pa.array([3], type=pa.int64()),
            "max_remote_video_bytes": [4096],
            "max_source_frame_bytes": [2048],
            "max_source_path_bytes": [
                max(
                    video_reader._video_source_path_bytes("a.avi"),
                    video_reader._video_source_path_bytes("b.avi"),
                )
            ],
            "on_error": ["raise"],
            "remote_temp_dir": pa.array([None], type=pa.string()),
        }
    )

    tables = list(_video_frame_source_map_batches(manifest))

    assert calls == [("a.avi", 3), ("b.avi", 1)]
    assert sum(table.num_rows for table in tables) == 3


def test_video_frame_source_map_batches_skips_bad_file_and_continues(
    monkeypatch,
    caplog,
):
    import duckdb.datasource.video_reader as video_reader

    frames = np.zeros((2, 2, 2, 3), dtype=np.uint8)

    class FakeDecordError(Exception):
        pass

    def fake_decode(video_path, **_kwargs):
        source_path, source_id = video_reader._video_source_identity(video_path)
        row_count = 1 if video_path == "bad.avi" else 2
        yield pa.record_batch(
            {
                "source_id": [source_id] * row_count,
                "video_path": [source_path] * row_count,
                "frame_index": list(range(row_count)),
                "frame": pa.FixedShapeTensorArray.from_numpy_ndarray(frames[:row_count]),
            }
        )
        if video_path == "bad.avi":
            failure = FakeDecordError("corrupt video")
            raise video_reader.VideoReadError(
                source_path,
                f"{type(failure).__name__}: {failure}",
            ) from failure

    monkeypatch.setattr(video_reader, "_wait_for_memory", lambda: None)
    monkeypatch.setattr(video_reader, "_decode_video_batches", fake_decode)
    manifest = pa.table(
        {
            "video_paths": pa.array([["bad.avi", "good.avi"]], type=pa.list_(pa.string())),
            "height": [2],
            "width": [2],
            "max_partition_bytes": [1024],
            "frame_limit": pa.array([None], type=pa.int64()),
            "max_remote_video_bytes": [4096],
            "max_source_frame_bytes": [2048],
            "max_source_path_bytes": [
                max(
                    video_reader._video_source_path_bytes("bad.avi"),
                    video_reader._video_source_path_bytes("good.avi"),
                )
            ],
            "on_error": ["skip"],
            "remote_temp_dir": pa.array([None], type=pa.string()),
        }
    )

    with caplog.at_level(logging.WARNING, logger=video_reader.__name__):
        tables = list(_video_frame_source_map_batches(manifest))

    bad_path, _ = video_reader._video_source_identity("bad.avi")
    good_path, _ = video_reader._video_source_identity("good.avi")
    assert [table.column("video_path").to_pylist() for table in tables] == [[bad_path, good_path, good_path]]
    assert f"Skipping unreadable video source={bad_path!r}" in caplog.text


def test_video_read_error_mode_raise_preserves_decoder_boundary_failure(monkeypatch):
    import duckdb.datasource.video_reader as video_reader

    failure = OSError("corrupt video")

    class FailingVideoReader:
        def __init__(self, *_args, **_kwargs):
            raise failure

    class FakeDecordModule:
        VideoReader = FailingVideoReader

    monkeypatch.setattr(video_reader, "_import_video_dependency", lambda *_args: FakeDecordModule)

    with pytest.raises(video_reader.VideoReadError, match="corrupt video") as raised:
        list(
            video_reader._decode_video_with_policy(
                "bad.avi",
                height=2,
                width=2,
                max_partition_bytes=1024,
                max_frames=None,
                max_remote_video_bytes=4096,
                max_source_frame_bytes=2048,
                remote_temp_dir=None,
                on_error="raise",
            )
        )

    assert raised.value.__cause__ is failure
    expected_path, _ = video_reader._video_source_identity("bad.avi")
    assert raised.value.video_path == expected_path


@pytest.mark.parametrize("error_type", [OSError, RuntimeError, ValueError])
def test_video_skip_mode_propagates_non_video_read_errors(monkeypatch, caplog, error_type):
    import duckdb.datasource.video_reader as video_reader

    failure = error_type("planned internal failure")

    def fake_decode(*_args, **_kwargs):
        raise failure
        yield

    monkeypatch.setattr(video_reader, "_decode_video_batches", fake_decode)

    with (
        caplog.at_level(logging.WARNING, logger=video_reader.__name__),
        pytest.raises(error_type, match="planned internal failure") as raised,
    ):
        list(
            video_reader._decode_video_with_policy(
                "input.mp4",
                height=2,
                width=2,
                max_partition_bytes=1024,
                max_frames=None,
                max_remote_video_bytes=4096,
                max_source_frame_bytes=2048,
                remote_temp_dir=None,
                on_error="skip",
            )
        )

    assert raised.value is failure
    assert "Skipping unreadable video" not in caplog.text


def test_video_skip_mode_propagates_resize_invariant(monkeypatch):
    import duckdb.datasource.video_reader as video_reader

    class FakeFrame:
        shape = (2, 32, 3)

        def asnumpy(self):
            return np.zeros(self.shape, dtype=np.uint8)

    monkeypatch.setattr(video_reader, "_open_decord_reader", lambda *_args, **_kwargs: [FakeFrame()])
    monkeypatch.setattr(video_reader, "_resize_frame_batch", lambda *_args, **_kwargs: [])

    with pytest.raises(RuntimeError, match="video resize returned a different number of frames"):
        list(
            video_reader._decode_video_with_policy(
                "input.mp4",
                height=2,
                width=2,
                max_partition_bytes=1024,
                max_frames=None,
                max_remote_video_bytes=4096,
                max_source_frame_bytes=2048,
                remote_temp_dir=None,
                on_error="skip",
            )
        )


def test_resize_frame_batch_preserves_order_and_uses_configured_threads(monkeypatch):
    import duckdb.datasource.video_reader as video_reader

    monkeypatch.setattr(video_reader, "_VIDEO_RESIZE_THREADS", 2)
    frame_a = np.zeros((2, 3, 3), dtype=np.uint8)
    frame_b = np.full((2, 3, 3), 255, dtype=np.uint8)

    resized = _resize_frame_batch([frame_a, frame_b], width=5, height=4)

    assert len(resized) == 2
    assert resized[0].shape == (4, 5, 3)
    assert resized[1].shape == (4, 5, 3)
    assert int(resized[0].mean()) == 0
    assert int(resized[1].mean()) == 255


def test_flush_frame_batch_uses_fixed_shape_tensor_for_frames():
    resized = np.arange(2 * 2 * 3 * 3, dtype=np.uint8).reshape(2, 2, 3, 3)

    batch = _flush_frame_batch("source-1", "clips/clip.avi", resized, [5, 6], 2)
    frame = batch.column("frame")

    assert batch.column("source_id").to_pylist() == ["source-1", "source-1"]
    assert batch.column("video_path").to_pylist() == ["clips/clip.avi", "clips/clip.avi"]
    assert batch.column("frame_index").to_pylist() == [5, 6]
    assert frame.type == pa.fixed_shape_tensor(pa.uint8(), (2, 3, 3))
    np.testing.assert_array_equal(frame.to_numpy_ndarray(), resized)


def test_flush_frame_batch_compacts_partial_output_buffer():
    import gc
    import weakref

    resized = np.zeros((100, 2, 3, 3), dtype=np.uint8)
    full_buffer_ref = weakref.ref(resized)
    indices = np.arange(100, dtype=np.int64)
    full_indices_ref = weakref.ref(indices)

    batch = _flush_frame_batch("source-1", "clips/clip.avi", resized, indices, 1)
    del resized
    del indices
    gc.collect()

    assert batch.num_rows == 1
    assert full_buffer_ref() is None
    assert full_indices_ref() is None

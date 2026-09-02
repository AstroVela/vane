# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import gc
import importlib
import io
import struct
import subprocess
import sys
import textwrap
import threading
import weakref
import zlib
from dataclasses import FrozenInstanceError
from fractions import Fraction
from types import SimpleNamespace

import av
import numpy as np
import pytest
from PIL import Image

import vane
from vane import _video_file


def _encoded_video(
    container_format: str = "mp4",
    *,
    codec_name: str | None = None,
    muxer_options: dict[str, str] | None = None,
    width: int = 16,
    height: int = 12,
    frame_count: int = 4,
    frame_rate: int = 24,
    gop_size: int | None = None,
    max_b_frames: int = 0,
) -> bytes:
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format=container_format, options=muxer_options) as container:
        codec = codec_name or ("libvpx" if container_format == "ogg" else "mpeg4")
        stream = container.add_stream(codec, rate=frame_rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.codec_context.max_b_frames = max_b_frames
        if gop_size is not None:
            stream.codec_context.gop_size = gop_size
        for index in range(frame_count):
            horizontal = np.broadcast_to(np.arange(width, dtype=np.uint8), (height, width))
            pixels = np.stack(
                (
                    horizontal + index,
                    np.full((height, width), index * 5, dtype=np.uint8),
                    np.flip(horizontal, axis=1) + index,
                ),
                axis=2,
            )
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def _encoded_image(image_format: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), color=(10, 20, 30)).save(buffer, format=image_format)
    return buffer.getvalue()


def _video_with_unknown_codec() -> bytes:
    payload = _encoded_video("matroska")
    known_codec = b"V_MPEG4/ISO/ASP"
    unknown_codec = b"V_FAKE/NO/CODEC"
    assert len(known_codec) == len(unknown_codec)
    assert known_codec in payload
    return payload.replace(known_codec, unknown_codec, 1)


def _oversized_png() -> bytes:
    payload = bytearray(_encoded_image("PNG"))
    assert payload[12:16] == b"IHDR"
    struct.pack_into(">II", payload, 16, 65_535, 32_768)
    struct.pack_into(">I", payload, 29, zlib.crc32(payload[12:29]) & 0xFFFFFFFF)
    return bytes(payload)


class _FakeDecodedVideoFrame:
    def __init__(
        self,
        *,
        pts: object = 0,
        width: int = 10,
        height: int = 10,
    ) -> None:
        self.width = width
        self.height = height
        self.pts = pts
        self.dts = None
        self.duration = 1
        self.time_base = Fraction(1, 10)
        self.key_frame = False


class _ColorRestoreFailureFrame:
    def __init__(self, failures):
        self._color_trc = 3
        self._color_primaries = 3
        self.failures = failures
        self.restore_attempts = []

    @property
    def color_trc(self):
        return self._color_trc

    @color_trc.setter
    def color_trc(self, value):
        if value != 2:
            self.restore_attempts.append("color_trc")
            error = self.failures.get("color_trc")
            if error is not None:
                raise error
        self._color_trc = value

    @property
    def color_primaries(self):
        return self._color_primaries

    @color_primaries.setter
    def color_primaries(self, value):
        if value != 2:
            self.restore_attempts.append("color_primaries")
            error = self.failures.get("color_primaries")
            if error is not None:
                raise error
        self._color_primaries = value


def _fake_decoder_video():
    return SimpleNamespace(time_base=Fraction(1, 10))


class _PacketSequenceContainer:
    """Model a container cursor across Vane's one-packet demux iterators."""

    def __init__(self, packets):
        self.packets = list(packets)

    def demux(self, video):
        del video
        if not self.packets:
            return iter(())
        return iter([self.packets.pop(0)])


def test_video_metadata_sql_and_python_value(duckdb_cursor, tmp_path):
    path = tmp_path / "video.mp4"
    path.write_bytes(_encoded_video())
    value = vane.VideoFile(str(path), "video/mp4")

    result_type, metadata, null_metadata = duckdb_cursor.execute(
        """
        SELECT
            typeof(video_metadata($1)),
            video_metadata($1),
            video_metadata(NULL::VIDEOFILE)
        """,
        [value],
    ).fetchone()

    assert result_type == (
        "STRUCT(width UINTEGER, height UINTEGER, fps DOUBLE, duration DOUBLE, frame_count BIGINT, "
        "time_base STRUCT(numerator BIGINT, denominator BIGINT))"
    )
    assert metadata["width"] == 16
    assert metadata["height"] == 12
    assert metadata["fps"] == pytest.approx(24)
    assert metadata["duration"] == pytest.approx(4 / 24)
    assert metadata["frame_count"] == 4
    assert metadata["time_base"]["numerator"] > 0
    assert metadata["time_base"]["denominator"] > 0
    assert null_metadata is None

    value_metadata = value.metadata(connection=duckdb_cursor)
    assert value_metadata.width == 16
    assert value_metadata.height == 12
    assert value_metadata.fps == pytest.approx(24)
    assert value_metadata.duration == pytest.approx(4 / 24)
    assert value_metadata.frame_count == 4
    assert value_metadata.time_base == Fraction(
        metadata["time_base"]["numerator"],
        metadata["time_base"]["denominator"],
    )


def test_video_metadata_facades(duckdb_cursor, tmp_path):
    path = tmp_path / "video.mp4"
    path.write_bytes(_encoded_video(width=20, height=14, frame_count=3, frame_rate=30))
    value = vane.VideoFile(str(path), "video/mp4")

    function_result = duckdb_cursor.sql("SELECT 1").select(vane.video_metadata(value, max_bytes=4096)).fetchone()[0]
    method_result = duckdb_cursor.sql("SELECT 1").select(vane.video_file(value).video_metadata()).fetchone()[0]

    assert function_result == method_result
    assert function_result["width"] == 20
    assert function_result["height"] == 14
    assert function_result["fps"] == pytest.approx(30)


def test_video_metadata_honors_logical_range(duckdb_cursor, tmp_path):
    payload = _encoded_video(width=18, height=10)
    prefix = b"not-a-video-prefix"
    suffix = b"not-a-video-suffix"
    path = tmp_path / "ranged.bin"
    path.write_bytes(prefix + payload + suffix)
    value = vane.VideoFile(str(path), "video/mp4", len(prefix), len(payload))

    sql_metadata = duckdb_cursor.execute("SELECT video_metadata($1)", [value]).fetchone()[0]
    value_metadata = value.metadata(connection=duckdb_cursor)

    assert (sql_metadata["width"], sql_metadata["height"]) == (18, 10)
    assert (value_metadata.width, value_metadata.height) == (18, 10)


def test_video_frames_stream_detached_rgb_images_with_exact_provenance(duckdb_cursor, tmp_path):
    path = tmp_path / "frames.mp4"
    path.write_bytes(_encoded_video(frame_count=8, frame_rate=4, gop_size=3))
    value = vane.VideoFile(str(path), "video/mp4")

    frames = list(value.frames(connection=duckdb_cursor))

    assert len(frames) == 8
    assert [frame.frame_index for frame in frames] == list(range(8))
    assert all(isinstance(frame, vane.VideoFrameData) for frame in frames)
    assert all(frame.data.mode == "RGB" and frame.data.size == (16, 12) for frame in frames)
    assert all(isinstance(frame.frame_time_base, Fraction) for frame in frames)
    assert all(
        frame.frame_time == pytest.approx(float(frame.frame_pts * frame.frame_time_base))
        for frame in frames
        if frame.frame_pts is not None and frame.frame_time_base is not None
    )
    assert np.asarray(frames[-1].data).shape == (12, 16, 3)


def test_video_frame_data_is_frozen_and_slotted():
    image = Image.new("RGB", (1, 1))
    frame = vane.VideoFrameData(0, 0.0, Fraction(1), 0, None, 1, True, image)

    try:
        assert not hasattr(frame, "__dict__")
        with pytest.raises(FrozenInstanceError):
            setattr(frame, "frame_pts", 1)
    finally:
        image.close()


def test_video_frames_honor_logical_range_and_resize(duckdb_cursor, tmp_path):
    payload = _encoded_video(width=18, height=10, frame_count=5, frame_rate=5)
    prefix = b"not-a-video-prefix"
    suffix = b"not-a-video-suffix"
    path = tmp_path / "ranged-frames.bin"
    path.write_bytes(prefix + payload + suffix)
    value = vane.VideoFile(str(path), "video/mp4", len(prefix), len(payload))

    frames = list(value.frames(width=9, height=6, buffer_size=64, connection=duckdb_cursor))

    assert len(frames) == 5
    assert all(frame.data.mode == "RGB" and frame.data.size == (9, 6) for frame in frames)


def test_video_frames_filter_time_keyframes_and_sample_exactly(duckdb_cursor, tmp_path):
    path = tmp_path / "selection.mp4"
    path.write_bytes(_encoded_video(frame_count=12, frame_rate=4, gop_size=3, max_b_frames=2))
    value = vane.VideoFile(str(path), "video/mp4")

    all_frames = list(value.frames(connection=duckdb_cursor))
    selected = list(value.frames(0.75, 2.0, connection=duckdb_cursor))
    key_frames = list(value.frames(is_key_frame=True, connection=duckdb_cursor))
    non_key_frames = list(value.frames(is_key_frame=False, connection=duckdb_cursor))
    sampled = list(value.frames(sample_interval_seconds=0.5, connection=duckdb_cursor))
    combined = list(
        value.frames(
            0.75,
            2.0,
            is_key_frame=True,
            sample_interval_seconds=0.5,
            connection=duckdb_cursor,
        )
    )

    def exact_time(frame):
        assert frame.frame_pts is not None
        assert frame.frame_time_base is not None
        return frame.frame_pts * frame.frame_time_base

    def expected_pts(*, start=Fraction(0), end=None, key_frame=None, interval=None):
        result = []
        next_sample = start if interval is not None else None
        for frame in all_frames:
            frame_time = exact_time(frame)
            if frame_time < start:
                continue
            if end is not None and frame_time > end:
                break
            if key_frame is not None and frame.is_key_frame is not key_frame:
                continue
            if interval is not None:
                assert next_sample is not None
                if frame_time < next_sample:
                    continue
                next_sample += ((frame_time - next_sample) // interval + 1) * interval
            result.append(frame.frame_pts)
        return result

    assert [frame.frame_pts for frame in selected] == expected_pts(start=Fraction(3, 4), end=Fraction(2))
    expected_indices = {
        frame.frame_pts: frame.frame_index
        for frame in all_frames
        if frame.frame_pts in expected_pts(start=Fraction(3, 4), end=Fraction(2))
    }
    assert [frame.frame_index for frame in selected] == [expected_indices[frame.frame_pts] for frame in selected]
    assert [frame.frame_pts for frame in key_frames] == expected_pts(key_frame=True)
    assert [frame.frame_pts for frame in non_key_frames] == expected_pts(key_frame=False)
    assert [frame.frame_pts for frame in sampled] == expected_pts(interval=Fraction(1, 2))
    assert [frame.frame_pts for frame in combined] == expected_pts(
        start=Fraction(3, 4),
        end=Fraction(2),
        key_frame=True,
        interval=Fraction(1, 2),
    )


def test_video_frames_preserve_presentation_order_with_b_frames(duckdb_cursor, tmp_path):
    path = tmp_path / "b-frames.mp4"
    path.write_bytes(_encoded_video(frame_count=12, frame_rate=4, gop_size=6, max_b_frames=2))

    frames = list(vane.VideoFile(str(path), "video/mp4").frames(connection=duckdb_cursor))

    assert len(frames) == 12
    assert [frame.frame_time for frame in frames] == pytest.approx([index / 4 for index in range(12)])
    assert any(frame.frame_dts != frame.frame_pts for frame in frames)


def test_video_frames_time_window_with_b_frames_matches_full_decode(duckdb_cursor, tmp_path):
    path = tmp_path / "b-frame-window.mp4"
    path.write_bytes(_encoded_video(frame_count=12, frame_rate=4, gop_size=6, max_b_frames=2))
    value = vane.VideoFile(str(path), "video/mp4")

    all_frames = list(value.frames(connection=duckdb_cursor))
    selected = list(value.frames(start_time=0.25, end_time=0.5, connection=duckdb_cursor))
    expected = [
        frame
        for frame in all_frames
        if frame.frame_pts is not None
        and frame.frame_time_base is not None
        and Fraction(1, 4) <= frame.frame_pts * frame.frame_time_base <= Fraction(1, 2)
    ]

    assert [frame.frame_pts for frame in selected] == [frame.frame_pts for frame in expected]
    assert [frame.frame_time for frame in selected] == pytest.approx([0.25, 0.5])


def test_video_frames_time_window_with_distinct_pts_dts_origins_matches_full_decode(duckdb_cursor, tmp_path):
    payload = _encoded_video(
        "mpegts",
        codec_name="mpeg2video",
        frame_count=12,
        frame_rate=8,
        gop_size=3,
        max_b_frames=1,
    )
    with av.open(io.BytesIO(payload), mode="r") as container:
        video = container.streams.video[0]
        first_packet = next(packet for packet in container.demux(video) if packet.size)
        assert first_packet.pts is not None
        assert first_packet.dts is not None
        assert first_packet.pts != first_packet.dts

    path = tmp_path / "distinct-pts-dts-origins.ts"
    path.write_bytes(payload)
    value = vane.VideoFile(str(path), "video/mp2t")

    all_frames = list(value.frames(connection=duckdb_cursor))
    selected = list(value.frames(start_time=0.125, end_time=0.375, connection=duckdb_cursor))
    assert all_frames[0].frame_pts is not None
    assert all_frames[0].frame_time_base is not None
    stream_origin = all_frames[0].frame_pts * all_frames[0].frame_time_base
    expected = [
        frame
        for frame in all_frames
        if frame.frame_pts is not None
        and frame.frame_time_base is not None
        and Fraction(1, 8) <= frame.frame_pts * frame.frame_time_base - stream_origin <= Fraction(3, 8)
    ]

    assert [frame.frame_pts for frame in selected] == [frame.frame_pts for frame in expected]
    assert [frame.frame_time for frame in selected] == pytest.approx([0.125, 0.25, 0.375])


def test_video_frames_time_window_scans_across_timestamp_discontinuities(duckdb_cursor, tmp_path):
    segment = _encoded_video(
        "mpegts",
        codec_name="mpeg2video",
        muxer_options={"mpegts_flags": "resend_headers+initial_discontinuity"},
        frame_count=8,
        frame_rate=8,
        gop_size=3,
        max_b_frames=1,
    )
    path = tmp_path / "timestamp-discontinuity.ts"
    path.write_bytes(segment + segment)
    value = vane.VideoFile(str(path), "video/mp2t")

    all_frames = list(value.frames(connection=duckdb_cursor))
    selected = list(value.frames(start_time=0.125, end_time=0.25, connection=duckdb_cursor))
    sampled = list(
        value.frames(
            start_time=0.125,
            end_time=0.25,
            sample_interval_seconds=0.125,
            connection=duckdb_cursor,
        )
    )
    expected = [frame for frame in all_frames if frame.frame_time is not None and 0.125 <= frame.frame_time <= 0.25]

    assert len(all_frames) == 16
    assert [frame.frame_index for frame in expected] == [1, 2, 9, 10]
    assert [frame.frame_index for frame in selected] == [frame.frame_index for frame in expected]
    assert [frame.frame_index for frame in sampled] == [frame.frame_index for frame in expected]


def test_video_visible_pixel_limit_is_independent_of_coded_alignment(duckdb_cursor, tmp_path):
    path = tmp_path / "aligned.mp4"
    path.write_bytes(_encoded_video(width=18, height=14, frame_count=1))

    frames = list(vane.VideoFile(str(path), "video/mp4").frames(max_pixels=18 * 14, connection=duckdb_cursor))

    assert len(frames) == 1
    assert frames[0].data.size == (18, 14)


def test_video_keyframes_reuse_streaming_selection(duckdb_cursor, tmp_path):
    path = tmp_path / "keyframes.mp4"
    path.write_bytes(_encoded_video(frame_count=12, frame_rate=4, gop_size=3))
    value = vane.VideoFile(str(path), "video/mp4")

    all_frames = list(value.frames(connection=duckdb_cursor))
    frame_records = list(value.frames(0.5, 2.25, is_key_frame=True, connection=duckdb_cursor))
    images = list(value.keyframes(0.5, 2.25, connection=duckdb_cursor))
    sampled_records = list(
        value.frames(
            width=8,
            height=6,
            is_key_frame=True,
            sample_interval_seconds=1,
            connection=duckdb_cursor,
        )
    )
    sampled_images = list(
        value.keyframes(
            width=8,
            height=6,
            sample_interval_seconds=1,
            connection=duckdb_cursor,
        )
    )
    expected = [
        frame
        for frame in all_frames
        if frame.is_key_frame
        and frame.frame_time_base is not None
        and frame.frame_pts is not None
        and Fraction(1, 2) <= frame.frame_pts * frame.frame_time_base <= Fraction(9, 4)
    ]

    assert [frame.frame_pts for frame in frame_records] == [frame.frame_pts for frame in expected]
    assert len(images) == len(expected)
    assert expected
    for image, record, baseline in zip(images, frame_records, expected, strict=True):
        assert image.mode == "RGB"
        np.testing.assert_array_equal(np.asarray(image), np.asarray(record.data))
        np.testing.assert_array_equal(np.asarray(image), np.asarray(baseline.data))
    assert len(sampled_images) == len(sampled_records)
    assert sampled_images
    for image, record in zip(sampled_images, sampled_records, strict=True):
        assert image.size == (8, 6)
        np.testing.assert_array_equal(np.asarray(image), np.asarray(record.data))


def test_video_frames_start_after_duration_is_empty(duckdb_cursor, tmp_path):
    path = tmp_path / "short.mp4"
    path.write_bytes(_encoded_video(frame_count=4, frame_rate=4, gop_size=2))
    value = vane.VideoFile(str(path), "video/mp4")

    assert list(value.frames(start_time=10, connection=duckdb_cursor)) == []
    assert list(value.keyframes(start_time=10, connection=duckdb_cursor)) == []


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"start_time": True}, TypeError, "start_time must be int or float"),
        ({"start_time": -1}, ValueError, "start_time must be non-negative"),
        ({"end_time": float("inf")}, ValueError, "end_time must be finite"),
        ({"start_time": 2, "end_time": 1}, ValueError, "end_time must be greater"),
        ({"width": 8}, ValueError, "width and height must be provided together"),
        ({"width": True, "height": 8}, TypeError, "width must be int"),
        ({"is_key_frame": 1}, TypeError, "is_key_frame must be bool or None"),
        ({"sample_interval_seconds": 0}, ValueError, "must be greater than zero"),
        ({"buffer_size": 0}, ValueError, "buffer_size must be greater than zero"),
        ({"buffer_size": 1 << 31}, OverflowError, "C int accepted by PyAV"),
        ({"max_input_bytes": 0}, ValueError, "max_input_bytes must be greater than zero"),
        ({"max_frames": 0}, ValueError, "max_frames must be greater than zero"),
        ({"max_pixels": 0}, ValueError, "max_pixels must be greater than zero"),
        ({"max_pixels": 32 * 1024 * 1024 + 1}, ValueError, "max_pixels must be at most"),
    ],
)
def test_video_frames_validate_arguments_without_opening_file(kwargs, error_type, message):
    value = vane.VideoFile("memory://not-opened")

    with pytest.raises(error_type, match=message):
        value.frames(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"start_time": 2, "end_time": 1}, "end_time must be greater"),
        ({"width": 8}, "width and height must be provided together"),
        ({"sample_interval_seconds": 0}, "must be greater than zero"),
    ],
)
def test_video_keyframes_validate_arguments_without_opening_file(kwargs, message):
    value = vane.VideoFile("memory://not-opened")

    with pytest.raises(ValueError, match=message):
        value.keyframes(**kwargs)


def test_video_frames_enforce_input_frame_and_pixel_limits(duckdb_cursor, tmp_path):
    payload = _encoded_video(frame_count=4, frame_rate=4)
    path = tmp_path / "bounded.mp4"
    path.write_bytes(payload)
    value = vane.VideoFile(str(path), "video/mp4")

    with pytest.raises(vane.VideoFileLimitError, match="max_input_bytes"):
        list(value.frames(max_input_bytes=len(payload) - 1, connection=duckdb_cursor))
    with pytest.raises(vane.VideoFileLimitError, match="max_frames=2"):
        list(value.frames(max_frames=2, connection=duckdb_cursor))
    with pytest.raises(vane.VideoFileLimitError, match="max_pixels=100"):
        list(value.frames(max_pixels=100, connection=duckdb_cursor))


def test_video_frames_apply_limits_to_actual_decoder_context(duckdb_cursor, tmp_path, monkeypatch):
    path = tmp_path / "decoder-options.mp4"
    path.write_bytes(_encoded_video(frame_count=1))
    observed = []
    configure_decoder = _video_file._configure_video_decoder

    def record_decoder_options(video):
        configure_decoder(video)
        observed.append((video.codec_context.thread_count, dict(video.codec_context.options)))

    monkeypatch.setattr(_video_file, "_configure_video_decoder", record_decoder_options)
    frames = list(vane.VideoFile(str(path), "video/mp4").frames(max_pixels=1_000_000, connection=duckdb_cursor))

    assert observed == [(1, {"max_pixels": str(64 * 1024 * 1024), "threads": "1"})]
    for frame in frames:
        frame.data.close()


def test_video_frame_conversion_neutralizes_and_restores_color_metadata():
    options = _video_file._VideoFrameOptions(
        start_time=Fraction(0),
        end_time=None,
        width=None,
        height=None,
        is_key_frame=None,
        sample_interval_seconds=None,
        buffer_size=1024,
        max_input_bytes=1024,
        max_frames=1,
        max_pixels=100,
    )
    observed = []
    output_frame = SimpleNamespace(to_image=lambda: Image.new("RGB", (10, 10)))

    def reformat(source, **kwargs):
        del kwargs
        assert source is frame
        observed.append((frame.color_trc, frame.color_primaries))
        return output_frame

    frame = SimpleNamespace(
        width=10,
        height=10,
        color_trc=3,
        color_primaries=3,
    )
    info = SimpleNamespace(width=10, height=10)
    reformatter = SimpleNamespace(reformat=reformat)
    fake_av = SimpleNamespace(error=SimpleNamespace(FFmpegError=RuntimeError))

    image = _video_file._frame_to_image(frame, info, options, fake_av, Image, reformatter, lambda: None)

    assert observed == [(2, 2)]
    assert (frame.color_trc, frame.color_primaries) == (3, 3)
    image.close()


def test_video_color_metadata_restore_preserves_control_flow_after_all_fields():
    class RestoreControlFlow(BaseException):
        pass

    control_flow = RestoreControlFlow("stop restoration")
    frame = _ColorRestoreFailureFrame(
        {
            "color_primaries": ValueError("invalid restored primaries"),
            "color_trc": control_flow,
        }
    )

    with pytest.raises(RestoreControlFlow, match="stop restoration") as raised:
        with _video_file._neutralized_color_conversion_metadata(frame):
            pass

    assert raised.value is control_flow
    assert frame.restore_attempts == ["color_primaries", "color_trc"]


@pytest.mark.parametrize("restore_error", [MemoryError("out of memory"), OSError("restore I/O failed")])
def test_video_color_metadata_restore_preserves_system_errors(restore_error):
    frame = _ColorRestoreFailureFrame({"color_primaries": restore_error})

    with pytest.raises(type(restore_error)) as raised:
        with _video_file._neutralized_color_conversion_metadata(frame):
            pass

    assert raised.value is restore_error
    assert frame.restore_attempts == ["color_primaries", "color_trc"]


def test_video_color_metadata_restore_classifies_invalid_values():
    restore_error = ValueError("invalid restored primaries")
    frame = _ColorRestoreFailureFrame({"color_primaries": restore_error})

    with pytest.raises(vane.VideoFileFormatError, match="color metadata could not be restored") as raised:
        with _video_file._neutralized_color_conversion_metadata(frame):
            pass

    assert raised.value.__cause__ is restore_error
    assert frame.restore_attempts == ["color_primaries", "color_trc"]


def test_video_frame_conversion_checks_interrupt_after_atomic_reformat():
    options = _video_file._VideoFrameOptions(
        start_time=Fraction(0),
        end_time=None,
        width=None,
        height=None,
        is_key_frame=None,
        sample_interval_seconds=None,
        buffer_size=1024,
        max_input_bytes=1024,
        max_frames=1,
        max_pixels=100,
    )
    interrupt_error = RuntimeError("query interrupted")
    checks = 0

    def check_interrupted():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise interrupt_error

    frame = SimpleNamespace(color_trc=3, color_primaries=3)
    output_frame = SimpleNamespace(to_image=lambda: pytest.fail("Pillow conversion must not start"))
    reformatter = SimpleNamespace(reformat=lambda source, **kwargs: output_frame)

    class FakeFFmpegError(Exception):
        pass

    fake_av = SimpleNamespace(error=SimpleNamespace(FFmpegError=FakeFFmpegError))

    with pytest.raises(RuntimeError, match="query interrupted") as raised:
        _video_file._frame_to_image(
            frame,
            SimpleNamespace(width=10, height=10),
            options,
            fake_av,
            Image,
            reformatter,
            check_interrupted,
        )
    assert raised.value is interrupt_error


def test_video_frame_conversion_closes_image_when_interrupted_after_creation():
    options = _video_file._VideoFrameOptions(
        start_time=Fraction(0),
        end_time=None,
        width=None,
        height=None,
        is_key_frame=None,
        sample_interval_seconds=None,
        buffer_size=1024,
        max_input_bytes=1024,
        max_frames=1,
        max_pixels=100,
    )
    interrupt_error = RuntimeError("query interrupted")
    checks = 0
    close_calls = 0
    image_refs = []

    class FakeImage:
        mode = "RGB"
        size = (10, 10)

        def __init__(self):
            image_refs.append(weakref.ref(self))

        def load(self):
            pytest.fail("image loading must not start")

        def close(self):
            nonlocal close_calls
            close_calls += 1
            raise RuntimeError("image close failed")

    def check_interrupted():
        nonlocal checks
        checks += 1
        if checks == 4:
            raise interrupt_error

    frame = SimpleNamespace(color_trc=3, color_primaries=3)
    output_frame = SimpleNamespace(to_image=FakeImage)
    reformatter = SimpleNamespace(reformat=lambda source, **kwargs: output_frame)

    class FakeFFmpegError(Exception):
        pass

    fake_av = SimpleNamespace(error=SimpleNamespace(FFmpegError=FakeFFmpegError))

    with pytest.raises(RuntimeError, match="query interrupted") as raised:
        _video_file._frame_to_image(
            frame,
            SimpleNamespace(width=10, height=10),
            options,
            fake_av,
            SimpleNamespace(Image=FakeImage),
            reformatter,
            check_interrupted,
        )

    gc.collect()
    assert raised.value is interrupt_error
    assert close_calls == 1
    assert len(image_refs) == 1
    assert image_refs[0]() is None


@pytest.mark.parametrize(
    ("stage", "error"),
    [
        ("reformat", av.error.BugError(1, "internal decoder bug")),
        ("to_image", av.error.PyAVCallbackError(1, "callback failed")),
        ("reformat", av.error.UnknownError(1, "unknown decoder failure")),
        ("to_image", av.error.OverflowError(34, "decoder range failure")),
    ],
)
def test_video_frame_conversion_preserves_pyav_internal_errors(stage, error):
    options = _video_file._VideoFrameOptions(
        start_time=Fraction(0),
        end_time=None,
        width=None,
        height=None,
        is_key_frame=None,
        sample_interval_seconds=None,
        buffer_size=1024,
        max_input_bytes=1024,
        max_frames=1,
        max_pixels=100,
    )

    def fail():
        raise error

    frame = SimpleNamespace(color_trc=3, color_primaries=3)
    output_frame = SimpleNamespace(to_image=fail)

    def reformat(source, **kwargs):
        del source, kwargs
        if stage == "reformat":
            fail()
        return output_frame

    with pytest.raises(type(error)) as raised:
        _video_file._frame_to_image(
            frame,
            SimpleNamespace(width=10, height=10),
            options,
            av,
            Image,
            SimpleNamespace(reformat=reformat),
            lambda: None,
        )
    assert raised.value is error


def test_video_frame_conversion_classifies_explicit_media_errors():
    options = _video_file._VideoFrameOptions(
        start_time=Fraction(0),
        end_time=None,
        width=None,
        height=None,
        is_key_frame=None,
        sample_interval_seconds=None,
        buffer_size=1024,
        max_input_bytes=1024,
        max_frames=1,
        max_pixels=100,
    )
    error = av.error.InvalidDataError(1, "invalid frame pixels")

    def fail_reformat(source, **kwargs):
        del source, kwargs
        raise error

    frame = SimpleNamespace(color_trc=3, color_primaries=3)
    with pytest.raises(vane.VideoFileFormatError, match="RGB pixels") as raised:
        _video_file._frame_to_image(
            frame,
            SimpleNamespace(width=10, height=10),
            options,
            av,
            Image,
            SimpleNamespace(reformat=fail_reformat),
            lambda: None,
        )
    assert raised.value.__cause__ is error


def test_video_frame_conversion_traceback_releases_native_frames():
    options = _video_file._VideoFrameOptions(
        start_time=Fraction(0),
        end_time=None,
        width=None,
        height=None,
        is_key_frame=None,
        sample_interval_seconds=None,
        buffer_size=1024,
        max_input_bytes=1024,
        max_frames=1,
        max_pixels=100,
    )

    class OutputFrame:
        def to_image(self):
            return Image.new("RGB", (1, 1))

    class SourceFrame(_FakeDecodedVideoFrame):
        color_trc = 3
        color_primaries = 3

        def __init__(self, output):
            super().__init__()
            self.output = output

    output = OutputFrame()
    source = SourceFrame(output)
    source_ref = weakref.ref(source)
    output_ref = weakref.ref(output)
    info = SimpleNamespace(width=10, height=10)
    reformatter = SimpleNamespace(reformat=lambda frame, **kwargs: frame.output)
    fake_av = SimpleNamespace(error=SimpleNamespace(FFmpegError=RuntimeError))

    with pytest.raises(RuntimeError, match="invalid RGB Pillow frame") as raised:
        _video_file._frame_to_image(source, info, options, fake_av, Image, reformatter, lambda: None)
    source.output = None
    source = None
    output = None
    gc.collect()

    assert raised.value is not None
    assert source_ref() is None
    assert output_ref() is None


def test_video_frame_provenance_rebases_nonzero_stream_origin():
    options = _video_file._VideoFrameOptions(
        start_time=Fraction(0),
        end_time=None,
        width=None,
        height=None,
        is_key_frame=None,
        sample_interval_seconds=None,
        buffer_size=1024,
        max_input_bytes=1024,
        max_frames=1,
        max_pixels=100,
    )
    frame = _FakeDecodedVideoFrame(pts=105)
    video = SimpleNamespace(start_time=100, time_base=Fraction(1, 10))
    stream_origin = _video_file._stream_time_origin(video, video.time_base)
    info = _video_file._decoded_frame_info(
        frame,
        video,
        options,
        frame_index=0,
        stream_time_origin=stream_origin,
    )

    assert info.frame_pts == 105
    assert info.exact_time == Fraction(1, 2)
    assert info.frame_time == 0.5


def test_video_frame_preserves_zero_duration_provenance():
    options = _video_file._VideoFrameOptions(
        start_time=Fraction(0),
        end_time=None,
        width=None,
        height=None,
        is_key_frame=None,
        sample_interval_seconds=None,
        buffer_size=1024,
        max_input_bytes=1024,
        max_frames=1,
        max_pixels=100,
    )
    frame = _FakeDecodedVideoFrame()
    frame.duration = 0

    info = _video_file._decoded_frame_info(
        frame,
        _fake_decoder_video(),
        options,
        frame_index=0,
        stream_time_origin=Fraction(0),
    )

    assert info.frame_duration == 0


def test_video_time_selection_rejects_dts_only_frame():
    options = _video_file._VideoFrameOptions(
        start_time=Fraction(1, 10),
        end_time=None,
        width=None,
        height=None,
        is_key_frame=None,
        sample_interval_seconds=None,
        buffer_size=1024,
        max_input_bytes=1024,
        max_frames=1,
        max_pixels=100,
    )
    frame = _FakeDecodedVideoFrame(pts=None)
    frame.dts = 105
    video = SimpleNamespace(time_base=Fraction(1, 10))

    info = _video_file._decoded_frame_info(
        frame,
        video,
        options,
        frame_index=0,
        stream_time_origin=Fraction(10),
    )

    assert info.frame_pts is None
    assert info.frame_dts == 105
    assert info.frame_time is None
    with pytest.raises(vane.VideoFileFormatError, match="presentation timestamp"):
        _video_file._require_frame_time_for_selection(info, options)


def test_video_packet_decode_checks_interrupt_after_dependency_failure():
    decode_error = RuntimeError("decoder failed")
    interrupt_error = RuntimeError("query interrupted")
    checks = 0

    def check_interrupted():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise interrupt_error

    def fail_decode():
        raise decode_error

    packet = SimpleNamespace(decode=fail_decode)
    reader = SimpleNamespace(
        check_interrupted=check_interrupted,
        raise_if_error=lambda: None,
    )
    no_error = SimpleNamespace(raise_if_error=lambda: None)

    with pytest.raises(RuntimeError, match="query interrupted") as raised:
        _video_file._decode_packet_frames(packet, reader, no_error)
    assert raised.value is interrupt_error


def test_video_packet_decode_preserves_direct_control_flow_exception():
    class DecodeControlFlow(BaseException):
        pass

    control_flow = DecodeControlFlow("stop decode")
    later_interrupt = RuntimeError("later query interrupt")
    decode_failed = False
    checks = 0

    def fail_decode():
        nonlocal decode_failed
        decode_failed = True
        raise control_flow

    def check_interrupted():
        nonlocal checks
        checks += 1
        if decode_failed:
            raise later_interrupt

    reader = SimpleNamespace(check_interrupted=check_interrupted, raise_if_error=lambda: None)
    no_error = SimpleNamespace(raise_if_error=lambda: None)

    with pytest.raises(DecodeControlFlow, match="stop decode") as raised:
        _video_file._decode_packet_frames(SimpleNamespace(decode=fail_decode), reader, no_error)

    assert raised.value is control_flow
    assert checks == 1


@pytest.mark.parametrize(
    ("callback", "arguments", "sentinel"),
    [
        ("read", (1,), b""),
        ("readinto", (bytearray(1),), 0),
        ("seek", (0,), -1),
        ("tell", (), -1),
        ("readable", (), False),
        ("seekable", (), False),
    ],
)
def test_video_reader_callbacks_stash_composite_operation_interrupt(callback, arguments, sentinel):
    interrupt_error = RuntimeError("query interrupted between callbacks")
    interrupted = False
    callback_calls = []

    class Reader:
        def _check_interrupted(self):
            if interrupted:
                raise interrupt_error

        def read(self, size=-1):
            callback_calls.append(("read", size))
            return b"x"

        def _read_and_check_interrupted(self, size=-1):
            return self.read(size)

        def readinto(self, buffer):
            callback_calls.append(("readinto", buffer))
            return 1

        def _readinto_and_check_interrupted(self, buffer):
            return self.readinto(buffer)

        def seek(self, offset, whence=io.SEEK_SET):
            callback_calls.append(("seek", offset, whence))
            return offset

        def tell(self):
            callback_calls.append(("tell",))
            return 0

        def readable(self):
            callback_calls.append(("readable",))
            return True

        def seekable(self):
            callback_calls.append(("seekable",))
            return True

    proxy = _video_file._VideoReaderProxy(Reader())
    getattr(proxy, callback)(*arguments)
    interrupted = True

    assert getattr(proxy, callback)(*arguments) == sentinel
    with pytest.raises(RuntimeError, match="interrupted between callbacks") as raised:
        proxy.raise_if_error()

    assert raised.value is interrupt_error
    assert len(callback_calls) == 1


def test_video_reader_read_atomically_observes_interrupt_after_callback_check(duckdb_cursor, tmp_path):
    path = tmp_path / "interrupt-between-check-and-read.bin"
    path.write_bytes(b"video bytes")
    reader = vane.File(str(path)).open(buffer_size=4, connection=duckdb_cursor)
    original_check = reader._check_interrupted
    check_completed = threading.Event()
    interrupt_completed = threading.Event()
    check_calls = 0

    def synchronized_check():
        nonlocal check_calls
        original_check()
        check_calls += 1
        if check_calls == 1:
            check_completed.set()
            if not interrupt_completed.wait(timeout=5):
                raise AssertionError("interrupt did not complete")

    reader._check_interrupted = synchronized_check

    def interrupt_after_check():
        if check_completed.wait(timeout=5):
            duckdb_cursor.interrupt()
        interrupt_completed.set()

    interrupt_thread = threading.Thread(target=interrupt_after_check)
    interrupt_thread.start()
    try:
        proxy = _video_file._VideoReaderProxy(reader)
        assert proxy.read(4) == b""
        with pytest.raises(vane.InterruptException):
            proxy.raise_if_error()
        # The retained operation generation is checked inside native Read,
        # before the connector can advance the logical reader position.
        assert reader.tell() == 0
    finally:
        interrupt_completed.set()
        interrupt_thread.join(timeout=5)
        reader.close()
    assert not interrupt_thread.is_alive()


@pytest.mark.parametrize("callback", ["readable", "seekable"])
def test_video_reader_capability_callbacks_check_interrupt_after_call(callback):
    interrupt_error = RuntimeError("query interrupted during capability callback")
    interrupted = False
    callback_calls = 0

    class Reader:
        def _check_interrupted(self):
            if interrupted:
                raise interrupt_error

        def readable(self):
            nonlocal interrupted, callback_calls
            callback_calls += 1
            interrupted = True
            return True

        def seekable(self):
            nonlocal interrupted, callback_calls
            callback_calls += 1
            interrupted = True
            return True

    proxy = _video_file._VideoReaderProxy(Reader())

    assert getattr(proxy, callback)() is False
    with pytest.raises(RuntimeError, match="query interrupted during capability callback") as raised:
        proxy.raise_if_error()

    assert raised.value is interrupt_error
    assert callback_calls == 1


def test_video_reader_callback_stashes_control_flow_exception():
    class CallbackControlFlow(BaseException):
        pass

    control_flow = CallbackControlFlow("stop callback")

    class Reader:
        def _check_interrupted(self):
            raise control_flow

    proxy = _video_file._VideoReaderProxy(Reader())

    assert proxy.read(1) == b""
    with pytest.raises(CallbackControlFlow, match="stop callback") as raised:
        proxy.raise_if_error()
    assert raised.value is control_flow


def test_video_reader_stashed_control_flow_precedes_later_interrupt():
    class CallbackControlFlow(BaseException):
        pass

    control_flow = CallbackControlFlow("stop callback")
    later_interrupt = RuntimeError("later query interrupt")
    checks = 0

    class Reader:
        def _check_interrupted(self):
            nonlocal checks
            checks += 1
            if checks == 1:
                raise control_flow
            raise later_interrupt

    proxy = _video_file._VideoReaderProxy(Reader())
    no_nested_error = SimpleNamespace(raise_if_error=lambda: None)

    assert proxy.read(1) == b""
    with pytest.raises(CallbackControlFlow, match="stop callback") as raised:
        _video_file._check_video_io(proxy, no_nested_error)

    assert raised.value is control_flow
    assert raised.value.__context__ is later_interrupt
    assert checks == 2


@pytest.mark.parametrize("failure_at", ["demux", "next"])
def test_video_demux_checks_interrupt_after_dependency_failure(failure_at):
    dependency_error = RuntimeError(f"{failure_at} failed")
    interrupt_error = RuntimeError("query interrupted")
    dependency_failed = False

    def fail_dependency():
        nonlocal dependency_failed
        dependency_failed = True
        raise dependency_error

    def check_interrupted():
        if dependency_failed:
            raise interrupt_error

    if failure_at == "demux":

        def demux(video):
            del video
            return fail_dependency()

    else:

        def failed_packets():
            fail_dependency()
            yield

        def demux(video):
            del video
            return failed_packets()

    options = _video_file._VideoFrameOptions(
        start_time=Fraction(0),
        end_time=None,
        width=None,
        height=None,
        is_key_frame=None,
        sample_interval_seconds=None,
        buffer_size=1024,
        max_input_bytes=1024,
        max_frames=1,
        max_pixels=100,
    )
    reader = SimpleNamespace(
        check_interrupted=check_interrupted,
        raise_if_error=lambda: None,
    )
    no_error = SimpleNamespace(raise_if_error=lambda: None)
    batches = _video_file._iter_decoded_packet_batches(
        SimpleNamespace(demux=demux),
        _fake_decoder_video(),
        options,
        reader,
        no_error,
    )

    with pytest.raises(RuntimeError, match="query interrupted") as raised:
        next(batches)
    assert raised.value is interrupt_error


@pytest.mark.parametrize("failure_at", ["demux", "next"])
def test_video_demux_preserves_direct_control_flow_exception(failure_at):
    class DemuxControlFlow(BaseException):
        pass

    control_flow = DemuxControlFlow(f"stop {failure_at}")
    later_interrupt = RuntimeError("later query interrupt")
    dependency_failed = False
    checks = 0

    def fail_dependency():
        nonlocal dependency_failed
        dependency_failed = True
        raise control_flow

    def check_interrupted():
        nonlocal checks
        checks += 1
        if dependency_failed:
            raise later_interrupt

    if failure_at == "demux":

        def demux(video):
            del video
            return fail_dependency()

    else:

        def failed_packets():
            fail_dependency()
            yield

        def demux(video):
            del video
            return failed_packets()

    reader = SimpleNamespace(check_interrupted=check_interrupted, raise_if_error=lambda: None)
    no_error = SimpleNamespace(raise_if_error=lambda: None)

    with pytest.raises(DemuxControlFlow, match=f"stop {failure_at}") as raised:
        _video_file._next_demuxed_packet(SimpleNamespace(demux=demux), object(), reader, no_error)

    assert raised.value is control_flow
    assert checks == (1 if failure_at == "demux" else 2)


def test_video_post_demux_interrupt_traceback_releases_packet_and_native_owners():
    packet_was_yielded = False

    class CodecContext:
        pass

    class Container:
        def __init__(self):
            self.packet = None

        def demux(self, selected_video):
            nonlocal packet_was_yielded
            del selected_video
            current_packet = self.packet
            self.packet = None

            def packets():
                nonlocal current_packet, packet_was_yielded
                try:
                    packet_was_yielded = True
                    yield current_packet
                finally:
                    current_packet = None

            return packets()

    class Video:
        def __init__(self, container, codec_context):
            self.container = container
            self.codec_context = codec_context

    class Packet:
        size = 1
        buffer_ptr = 1

        def __init__(self, stream):
            self.stream = stream

    container = Container()
    codec_context = CodecContext()
    video = Video(container, codec_context)
    packet = Packet(video)
    container.packet = packet
    container_ref = weakref.ref(container)
    codec_ref = weakref.ref(codec_context)
    video_ref = weakref.ref(video)
    packet_ref = weakref.ref(packet)
    packet = None

    def check_interrupted():
        if packet_was_yielded:
            raise RuntimeError("query interrupted after demux")

    reader = SimpleNamespace(
        check_interrupted=check_interrupted,
        raise_if_error=lambda: None,
    )
    no_error = SimpleNamespace(raise_if_error=lambda: None)

    with pytest.raises(RuntimeError, match="interrupted after demux") as raised:
        _video_file._next_demuxed_packet(container, video, reader, no_error)
    container = None
    codec_context = None
    video = None
    gc.collect()

    assert raised.value is not None
    assert packet_ref() is None
    assert video_ref() is None
    assert codec_ref() is None
    assert container_ref() is None


def test_video_flush_classification_traceback_releases_packet_and_native_owners():
    class CodecContext:
        pass

    class Container:
        def __init__(self):
            self.packet = None

        def demux(self, selected_video):
            del selected_video
            current_packet = self.packet
            self.packet = None

            def packets():
                nonlocal current_packet
                try:
                    yield current_packet
                finally:
                    current_packet = None

            return packets()

    class Video:
        def __init__(self, container, codec_context):
            self.container = container
            self.codec_context = codec_context

    class Packet:
        size = object()
        buffer_ptr = 1

        def __init__(self, stream):
            self.stream = stream

    container = Container()
    codec_context = CodecContext()
    video = Video(container, codec_context)
    packet = Packet(video)
    container.packet = packet
    container_ref = weakref.ref(container)
    codec_ref = weakref.ref(codec_context)
    video_ref = weakref.ref(video)
    packet_ref = weakref.ref(packet)
    packet = None
    no_error = SimpleNamespace(
        check_interrupted=lambda: None,
        raise_if_error=lambda: None,
    )

    with pytest.raises(vane.VideoFileFormatError, match="invalid packet buffer metadata") as raised:
        _video_file._next_demuxed_packet(container, video, no_error, no_error)
    container = None
    codec_context = None
    video = None
    gc.collect()

    assert raised.value is not None
    assert packet_ref() is None
    assert video_ref() is None
    assert codec_ref() is None
    assert container_ref() is None


def test_video_demux_iterator_releases_its_packet_owner_before_batch_yield():
    class Container:
        def __init__(self):
            self.packet = None
            self.demux_closed = False

        def demux(self, selected_video):
            del selected_video
            current_packet = self.packet
            self.packet = None

            def packets():
                nonlocal current_packet
                try:
                    yield current_packet
                finally:
                    self.demux_closed = True
                    current_packet = None

            return packets()

    class Video:
        time_base = Fraction(1, 10)

        def __init__(self, container):
            self.container = container

    class Packet:
        size = 1
        buffer_ptr = 1

        def __init__(self, stream):
            self.stream = stream

        def decode(self):
            assert self.stream.container.demux_closed
            return [_FakeDecodedVideoFrame()]

    options = _video_file._VideoFrameOptions(
        start_time=Fraction(0),
        end_time=None,
        width=None,
        height=None,
        is_key_frame=None,
        sample_interval_seconds=None,
        buffer_size=1024,
        max_input_bytes=1024,
        max_frames=1,
        max_pixels=100,
    )
    container = Container()
    video = Video(container)
    packet = Packet(video)
    packet_ref = weakref.ref(packet)
    container.packet = packet
    packet = None
    no_error = SimpleNamespace(
        check_interrupted=lambda: None,
        raise_if_error=lambda: None,
    )
    batches = _video_file._iter_decoded_packet_batches(
        container,
        video,
        options,
        no_error,
        no_error,
    )

    batch = next(batches)
    gc.collect()

    assert container.demux_closed
    assert packet_ref() is None
    batch.release()
    batches.close()


def test_video_demux_flush_packet_ends_one_packet_iterators():
    options = _video_file._VideoFrameOptions(
        start_time=Fraction(0),
        end_time=None,
        width=None,
        height=None,
        is_key_frame=None,
        sample_interval_seconds=None,
        buffer_size=1024,
        max_input_bytes=1024,
        max_frames=2,
        max_pixels=100,
    )
    regular = SimpleNamespace(
        size=1,
        buffer_ptr=1,
        decode=lambda: [_FakeDecodedVideoFrame(pts=0)],
    )
    flush = SimpleNamespace(
        size=0,
        buffer_ptr=0,
        has_sidedata=lambda _data_type: False,
        decode=lambda: [_FakeDecodedVideoFrame(pts=1)],
    )

    class Container(_PacketSequenceContainer):
        def demux(self, video):
            if not self.packets:
                pytest.fail("flush packet must end demux without another iterator")
            return super().demux(video)

    container = Container([regular, flush])
    no_error = SimpleNamespace(
        check_interrupted=lambda: None,
        raise_if_error=lambda: None,
    )

    batches = list(
        _video_file._iter_decoded_packet_batches(
            container,
            _fake_decoder_video(),
            options,
            no_error,
            no_error,
        )
    )

    assert [info.frame_pts for batch in batches for info in batch.infos] == [0, 1]


def test_video_demux_does_not_mistake_side_data_only_packet_for_flush():
    options = _video_file._VideoFrameOptions(
        start_time=Fraction(0),
        end_time=None,
        width=None,
        height=None,
        is_key_frame=None,
        sample_interval_seconds=None,
        buffer_size=1024,
        max_input_bytes=1024,
        max_frames=2,
        max_pixels=100,
    )
    decoded_packets = []
    checked_side_data_types = []

    def decode_side_data():
        decoded_packets.append("side-data")
        return []

    def has_side_data(data_type):
        checked_side_data_types.append(data_type)
        return data_type == "display_matrix"

    def copy_side_data():
        pytest.fail("flush detection must not copy packet side data")

    side_data_only = SimpleNamespace(
        size=0,
        buffer_ptr=0,
        has_sidedata=has_side_data,
        iter_sidedata=copy_side_data,
        decode=decode_side_data,
    )
    regular = SimpleNamespace(
        size=1,
        buffer_ptr=1,
        decode=lambda: [_FakeDecodedVideoFrame(pts=0)],
    )
    flush = SimpleNamespace(
        size=0,
        buffer_ptr=0,
        has_sidedata=lambda _data_type: False,
        iter_sidedata=copy_side_data,
        decode=lambda: [_FakeDecodedVideoFrame(pts=1)],
    )
    container = _PacketSequenceContainer([side_data_only, regular, flush])
    no_error = SimpleNamespace(
        check_interrupted=lambda: None,
        raise_if_error=lambda: None,
    )

    batches = list(
        _video_file._iter_decoded_packet_batches(
            container,
            _fake_decoder_video(),
            options,
            no_error,
            no_error,
        )
    )

    assert decoded_packets == ["side-data"]
    assert checked_side_data_types[-1] == "display_matrix"
    assert [info.frame_pts for batch in batches for info in batch.infos] == [0, 1]
    assert container.packets == []


def test_video_packet_decode_bounds_complete_returned_batch():
    options = _video_file._VideoFrameOptions(
        start_time=Fraction(0),
        end_time=None,
        width=None,
        height=None,
        is_key_frame=None,
        sample_interval_seconds=None,
        buffer_size=1024,
        max_input_bytes=1024,
        max_frames=1,
        max_pixels=100,
    )
    frames = [
        _FakeDecodedVideoFrame(),
        _FakeDecodedVideoFrame(),
    ]
    packet = SimpleNamespace(decode=lambda: frames)
    container = _PacketSequenceContainer([packet])
    no_error = SimpleNamespace(
        check_interrupted=lambda: None,
        raise_if_error=lambda: None,
    )

    with pytest.raises(vane.VideoFileLimitError, match=r"max_frames=1"):
        list(
            _video_file._iter_decoded_packet_batches(
                container,
                _fake_decoder_video(),
                options,
                no_error,
                no_error,
            )
        )


@pytest.mark.parametrize(
    ("frames", "error_type"),
    [
        (
            [_FakeDecodedVideoFrame(), _FakeDecodedVideoFrame(pts=object())],
            vane.VideoFileFormatError,
        ),
        (
            [
                _FakeDecodedVideoFrame(width=1, height=1),
                _FakeDecodedVideoFrame(width=11, height=10),
            ],
            vane.VideoFileLimitError,
        ),
    ],
    ids=["provenance", "visible-pixels"],
)
def test_video_packet_decode_prevalidates_complete_batch(frames, error_type):
    options = _video_file._VideoFrameOptions(
        start_time=Fraction(0),
        end_time=None,
        width=None,
        height=None,
        is_key_frame=None,
        sample_interval_seconds=None,
        buffer_size=1024,
        max_input_bytes=1024,
        max_frames=2,
        max_pixels=100,
    )
    packet = SimpleNamespace(decode=lambda: frames)
    container = _PacketSequenceContainer([packet])
    no_error = SimpleNamespace(
        check_interrupted=lambda: None,
        raise_if_error=lambda: None,
    )
    batches = _video_file._iter_decoded_packet_batches(
        container,
        _fake_decoder_video(),
        options,
        no_error,
        no_error,
    )

    with pytest.raises(error_type):
        next(batches)
    assert frames == []


def test_video_packet_decode_prevalidates_missing_pts_for_time_selection():
    options = _video_file._VideoFrameOptions(
        start_time=Fraction(0),
        end_time=Fraction(1),
        width=None,
        height=None,
        is_key_frame=None,
        sample_interval_seconds=None,
        buffer_size=1024,
        max_input_bytes=1024,
        max_frames=2,
        max_pixels=100,
    )
    frames = [
        _FakeDecodedVideoFrame(pts=0),
        _FakeDecodedVideoFrame(pts=None),
    ]
    packet = SimpleNamespace(decode=lambda: frames)
    container = _PacketSequenceContainer([packet])
    no_error = SimpleNamespace(
        check_interrupted=lambda: None,
        raise_if_error=lambda: None,
    )
    batches = _video_file._iter_decoded_packet_batches(
        container,
        _fake_decoder_video(),
        options,
        no_error,
        no_error,
    )

    with pytest.raises(vane.VideoFileFormatError, match="presentation timestamp"):
        next(batches)
    assert frames == []


def test_video_packet_conversion_is_atomic(monkeypatch):
    options = _video_file._VideoFrameOptions(
        start_time=Fraction(0),
        end_time=None,
        width=None,
        height=None,
        is_key_frame=None,
        sample_interval_seconds=None,
        buffer_size=1024,
        max_input_bytes=1024,
        max_frames=2,
        max_pixels=100,
    )
    frames = [_FakeDecodedVideoFrame(pts=0), _FakeDecodedVideoFrame(pts=1)]
    video = _fake_decoder_video()
    infos = tuple(
        _video_file._decoded_frame_info(frame, video, options, frame_index=index, stream_time_origin=Fraction(0))
        for index, frame in enumerate(frames)
    )
    batch = _video_file._DecodedPacketBatch(frames=frames, infos=infos)
    conversion_calls = 0
    close_calls = 0
    first_image_ref = None

    class TrackingImage:
        def close(self):
            nonlocal close_calls
            close_calls += 1

    def convert_frame(*args, **kwargs):
        nonlocal conversion_calls, first_image_ref
        del args, kwargs
        conversion_calls += 1
        if conversion_calls == 2:
            raise vane.VideoFileFormatError("second frame conversion failed")
        image = TrackingImage()
        first_image_ref = weakref.ref(image)
        return image

    monkeypatch.setattr(_video_file, "_frame_to_image", convert_frame)
    no_error = SimpleNamespace(
        check_interrupted=lambda: None,
        raise_if_error=lambda: None,
    )

    with pytest.raises(vane.VideoFileFormatError, match="second frame conversion failed") as raised:
        _video_file._prepare_video_packet_batch(
            batch,
            options,
            SimpleNamespace(),
            SimpleNamespace(),
            object(),
            no_error,
            no_error,
            next_sample_time=None,
            last_frame_time=None,
        )
    gc.collect()

    assert conversion_calls == 2
    assert close_calls == 1
    assert frames == []
    assert first_image_ref is not None
    assert first_image_ref() is None
    assert raised.value is not None


def test_video_packet_decode_traceback_releases_container_and_codec_owners():
    options = _video_file._VideoFrameOptions(
        start_time=Fraction(0),
        end_time=None,
        width=None,
        height=None,
        is_key_frame=None,
        sample_interval_seconds=None,
        buffer_size=1024,
        max_input_bytes=1024,
        max_frames=1,
        max_pixels=100,
    )

    class CodecContext:
        height = 10
        coded_height = 10

    class Video:
        time_base = Fraction(1, 10)

        def __init__(self, codec_context):
            self.codec_context = codec_context

    decoded_frame = _FakeDecodedVideoFrame(pts=object())
    returned_frames = [decoded_frame]
    packet = SimpleNamespace(decode=lambda: returned_frames)
    codec_context = CodecContext()
    video = Video(codec_context)
    container = _PacketSequenceContainer([packet])
    codec_ref = weakref.ref(codec_context)
    video_ref = weakref.ref(video)
    container_ref = weakref.ref(container)
    no_error = SimpleNamespace(
        check_interrupted=lambda: None,
        raise_if_error=lambda: None,
    )
    batches = _video_file._iter_decoded_packet_batches(
        container,
        video,
        options,
        no_error,
        no_error,
    )

    with pytest.raises(vane.VideoFileFormatError) as raised:
        next(batches)
    decoded_frame = None
    returned_frames = None
    packet = None
    codec_context = None
    video = None
    container = None
    batches = None
    gc.collect()

    assert raised.value is not None
    assert codec_ref() is None
    assert video_ref() is None
    assert container_ref() is None


def test_video_packet_batch_releases_each_taken_frame():
    options = _video_file._VideoFrameOptions(
        start_time=Fraction(0),
        end_time=None,
        width=None,
        height=None,
        is_key_frame=None,
        sample_interval_seconds=None,
        buffer_size=1024,
        max_input_bytes=1024,
        max_frames=2,
        max_pixels=100,
    )
    first = _FakeDecodedVideoFrame()
    second = _FakeDecodedVideoFrame()
    frames = [first, second]
    packet = SimpleNamespace(decode=lambda: frames)
    container = _PacketSequenceContainer([packet])
    no_error = SimpleNamespace(
        check_interrupted=lambda: None,
        raise_if_error=lambda: None,
    )
    decoded = _video_file._iter_decoded_packet_batches(
        container,
        _fake_decoder_video(),
        options,
        no_error,
        no_error,
    )

    batch = next(decoded)
    frame = batch.take_frame(0)

    assert frame is first
    assert frames == [None, second]
    assert batch.infos[0].frame_index == 0
    decoded.close()


def test_video_packet_decode_releases_previous_batch_before_next_decode():
    options = _video_file._VideoFrameOptions(
        start_time=Fraction(0),
        end_time=None,
        width=None,
        height=None,
        is_key_frame=None,
        sample_interval_seconds=None,
        buffer_size=1024,
        max_input_bytes=1024,
        max_frames=2,
        max_pixels=100,
    )

    class OneShotPacket:
        def __init__(self, frame):
            self.frame = frame

        def decode(self):
            frame = self.frame
            self.frame = None
            return [frame]

    first = _FakeDecodedVideoFrame()
    first_ref = weakref.ref(first)
    first_packet = OneShotPacket(first)
    second = _FakeDecodedVideoFrame()

    def decode_second():
        assert first_ref() is None
        return [second]

    second_packet = SimpleNamespace(decode=decode_second)
    container = _PacketSequenceContainer([first_packet, second_packet])
    no_error = SimpleNamespace(
        check_interrupted=lambda: None,
        raise_if_error=lambda: None,
    )
    batches = _video_file._iter_decoded_packet_batches(
        container,
        _fake_decoder_video(),
        options,
        no_error,
        no_error,
    )

    del first
    first_batch = next(batches)
    source_frame = first_batch.take_frame(0)
    del source_frame
    first_batch.release()
    del first_batch
    second_batch = next(batches)

    assert second_batch.take_frame(0) is second
    batches.close()


def test_video_frames_classify_media_errors_but_propagate_file_io(duckdb_cursor, tmp_path):
    corrupt = tmp_path / "corrupt-frames.mp4"
    corrupt.write_bytes(b"not a video file")

    with pytest.raises(vane.VideoFileFormatError, match="decodable video"):
        list(vane.VideoFile(str(corrupt), "video/mp4").frames(connection=duckdb_cursor))
    with pytest.raises(vane.IOException):
        list(vane.VideoFile(str(tmp_path / "missing.mp4"), "video/mp4").frames(connection=duckdb_cursor))


@pytest.mark.parametrize(
    "error",
    [
        av.error.MemoryError(12, "Cannot allocate memory", "decoder"),
        av.error.PermissionError(13, "Permission denied", "decoder"),
        av.error.BugError(1, "internal decoder bug", "decoder"),
        av.error.PyAVCallbackError(1, "callback failed", "decoder"),
        av.error.ExternalError(1, "external library failed", "decoder"),
        av.error.HTTPClientError(400, "remote request failed", "decoder"),
        av.error.FFmpegError(1, "unclassified decoder failure", "decoder"),
        av.error.UndefinedError(1, "fallback decoder failure", "decoder"),
        av.error.UnknownError(1, "unknown decoder failure", "decoder"),
        av.error.ArgumentError(22, "invalid decoder argument", "decoder"),
        av.error.BufferTooSmallError(1, "decoder buffer too small", "decoder"),
        av.error.OverflowError(34, "decoder range failure", "decoder"),
    ],
    ids=[
        "memory",
        "permission",
        "bug",
        "callback",
        "external",
        "http",
        "base",
        "fallback",
        "unknown",
        "argument",
        "buffer",
        "range",
    ],
)
def test_video_frames_preserve_pyav_system_and_internal_errors(error):
    no_stored_error = SimpleNamespace(
        check_interrupted=lambda: None,
        raise_if_error=lambda: None,
    )

    with pytest.raises(type(error)) as raised:
        _video_file._classify_video_decode_error(
            error,
            av_module=av,
            reader=no_stored_error,
            nested_io=no_stored_error,
        )
    assert raised.value is error


@pytest.mark.parametrize(
    "error",
    [
        av.error.DecoderNotFoundError(1, "decoder not found", "decoder"),
        av.error.DemuxerNotFoundError(1, "demuxer not found", "decoder"),
        av.error.EOFError(1, "unexpected end of file", "decoder"),
        av.error.InvalidDataError(1, "invalid encoded data", "decoder"),
    ],
    ids=["decoder", "demuxer", "eof", "invalid-data"],
)
def test_video_frames_classify_explicit_pyav_media_errors(error):
    no_stored_error = SimpleNamespace(
        check_interrupted=lambda: None,
        raise_if_error=lambda: None,
    )

    with pytest.raises(vane.VideoFileFormatError, match="decodable video") as raised:
        _video_file._classify_video_decode_error(
            error,
            av_module=av,
            reader=no_stored_error,
            nested_io=no_stored_error,
        )
    assert raised.value.__cause__ is error


def test_video_frames_give_pending_interrupt_precedence_during_classification():
    interrupt_error = RuntimeError("query interrupted")

    def raise_interrupt():
        raise interrupt_error

    reader = SimpleNamespace(
        check_interrupted=raise_interrupt,
        raise_if_error=lambda: None,
    )
    no_stored_error = SimpleNamespace(raise_if_error=lambda: None)

    with pytest.raises(RuntimeError, match="query interrupted") as raised:
        _video_file._classify_video_decode_error(
            vane.VideoFileFormatError("media failed"),
            av_module=av,
            reader=reader,
            nested_io=no_stored_error,
        )
    assert raised.value is interrupt_error


def test_video_frames_propagate_reader_failures(duckdb_cursor, tmp_path, monkeypatch):
    path = tmp_path / "reader-failure.mp4"
    path.write_bytes(_encoded_video())
    value = vane.VideoFile(str(path), "video/mp4")

    def fail_read(self, size=-1):
        del self, size
        raise OSError("connector frame read failed")

    monkeypatch.setattr(vane.VaneFileReader, "_read_and_check_interrupted", fail_read)
    with pytest.raises(OSError, match="connector frame read failed"):
        list(value.frames(connection=duckdb_cursor))


def test_video_frames_observe_connection_interrupt_between_buffered_frames(duckdb_cursor, tmp_path):
    path = tmp_path / "interrupt.mp4"
    path.write_bytes(_encoded_video(frame_count=12, frame_rate=4))
    frames = vane.VideoFile(str(path), "video/mp4").frames(
        sample_interval_seconds=100,
        connection=duckdb_cursor,
    )

    first = next(frames)
    duckdb_cursor.interrupt()

    with pytest.raises(vane.InterruptException):
        next(frames)
    first.data.close()


def test_video_frames_observe_connection_interrupt_before_successful_eof(duckdb_cursor, tmp_path):
    path = tmp_path / "interrupt-at-eof.mp4"
    path.write_bytes(_encoded_video(frame_count=1))
    frames = vane.VideoFile(str(path), "video/mp4").frames(connection=duckdb_cursor)

    first = next(frames)
    duckdb_cursor.interrupt()

    with pytest.raises(vane.InterruptException):
        next(frames)
    first.data.close()


def test_video_frames_observe_interrupt_during_container_close(monkeypatch):
    interrupt_error = RuntimeError("query interrupted during container close")

    class Reader:
        interrupted = False

        def close(self):
            pass

        def size(self):
            return 1

        def _check_interrupted(self):
            if self.interrupted:
                raise interrupt_error

    reader = Reader()

    class Value:
        content_type = None

        def open(self, **kwargs):
            del kwargs
            return reader

    class Container:
        def close(self):
            reader.interrupted = True

    class FakeFFmpegError(Exception):
        pass

    class FakeExitError(FakeFFmpegError):
        pass

    fake_av = SimpleNamespace(
        open=lambda *args, **kwargs: Container(),
        error=SimpleNamespace(ExitError=FakeExitError, FFmpegError=FakeFFmpegError),
        video=SimpleNamespace(reformatter=SimpleNamespace(VideoReformatter=object)),
    )
    monkeypatch.setattr(
        _video_file,
        "_metadata_from_container",
        lambda *args, **kwargs: SimpleNamespace(time_base=Fraction(1)),
    )
    monkeypatch.setattr(_video_file, "_select_video_stream", lambda *args: object())
    monkeypatch.setattr(_video_file, "_stream_time_origin", lambda *args: Fraction(0))
    monkeypatch.setattr(_video_file, "_configure_video_decoder", lambda *args: None)
    monkeypatch.setattr(_video_file, "_iter_decoded_packet_batches", lambda *args, **kwargs: (x for x in ()))
    options = _video_file._VideoFrameOptions(
        start_time=Fraction(0),
        end_time=None,
        width=None,
        height=None,
        is_key_frame=None,
        sample_interval_seconds=None,
        buffer_size=1024,
        max_input_bytes=1024,
        max_frames=1,
        max_pixels=100,
    )

    with pytest.raises(RuntimeError, match="interrupted during container close") as raised:
        list(_video_file._iter_video_frames(Value(), options, fake_av, SimpleNamespace(), None))
    assert raised.value is interrupt_error


def test_video_reader_close_failure_traceback_releases_native_owners(monkeypatch):
    close_error = RuntimeError("reader close failed")
    owner_refs = {}

    class Reader:
        def close(self):
            pass

        def _close_and_check_interrupted(self):
            raise close_error

        def size(self):
            return 1

        def _check_interrupted(self):
            pass

    class Value:
        content_type = None

        def open(self, **kwargs):
            del kwargs
            return Reader()

    class Container:
        def close(self):
            pass

    class Video:
        pass

    class Reformatter:
        def __init__(self):
            owner_refs["reformatter"] = weakref.ref(self)

    def open_container(*args, **kwargs):
        del args, kwargs
        container = Container()
        owner_refs["container"] = weakref.ref(container)
        return container

    def select_video(*args):
        del args
        video = Video()
        owner_refs["video"] = weakref.ref(video)
        return video

    class FakeFFmpegError(Exception):
        pass

    class FakeExitError(FakeFFmpegError):
        pass

    fake_av = SimpleNamespace(
        open=open_container,
        error=SimpleNamespace(ExitError=FakeExitError, FFmpegError=FakeFFmpegError),
        video=SimpleNamespace(reformatter=SimpleNamespace(VideoReformatter=Reformatter)),
    )
    monkeypatch.setattr(
        _video_file,
        "_metadata_from_container",
        lambda *args, **kwargs: SimpleNamespace(time_base=Fraction(1)),
    )
    monkeypatch.setattr(_video_file, "_select_video_stream", select_video)
    monkeypatch.setattr(_video_file, "_stream_time_origin", lambda *args: Fraction(0))
    monkeypatch.setattr(_video_file, "_configure_video_decoder", lambda *args: None)
    monkeypatch.setattr(_video_file, "_iter_decoded_packet_batches", lambda *args, **kwargs: (x for x in ()))
    options = _video_file._VideoFrameOptions(
        start_time=Fraction(0),
        end_time=None,
        width=None,
        height=None,
        is_key_frame=None,
        sample_interval_seconds=None,
        buffer_size=1024,
        max_input_bytes=1024,
        max_frames=1,
        max_pixels=100,
    )

    with pytest.raises(RuntimeError, match="reader close failed") as raised:
        list(_video_file._iter_video_frames(Value(), options, fake_av, SimpleNamespace(), None))
    gc.collect()

    assert raised.value is close_error
    assert set(owner_refs) == {"container", "video", "reformatter"}
    assert all(owner_ref() is None for owner_ref in owner_refs.values())


def test_video_reader_close_failure_does_not_replace_active_decode_error(monkeypatch):
    primary_error = vane.VideoFileFormatError("primary decode failure")
    close_error = RuntimeError("competing reader close failure")

    class NativeReader:
        _closed = False

        def _size(self):
            return 1

        def _check_interrupted(self):
            pass

        def _close(self):
            self._closed = True
            raise close_error

    reader = vane.VaneFileReader(NativeReader())

    class Value:
        content_type = None

        def open(self, **kwargs):
            del kwargs
            return reader

    class Container:
        def close(self):
            pass

    class FakeFFmpegError(Exception):
        pass

    class FakeExitError(FakeFFmpegError):
        pass

    fake_av = SimpleNamespace(
        open=lambda *args, **kwargs: Container(),
        error=SimpleNamespace(ExitError=FakeExitError, FFmpegError=FakeFFmpegError),
        video=SimpleNamespace(reformatter=SimpleNamespace(VideoReformatter=object)),
    )

    def fail_decode(*args, **kwargs):
        del args, kwargs
        if False:
            yield None
        raise primary_error

    monkeypatch.setattr(
        _video_file,
        "_metadata_from_container",
        lambda *args, **kwargs: SimpleNamespace(time_base=Fraction(1)),
    )
    monkeypatch.setattr(_video_file, "_select_video_stream", lambda *args: object())
    monkeypatch.setattr(_video_file, "_stream_time_origin", lambda *args: Fraction(0))
    monkeypatch.setattr(_video_file, "_configure_video_decoder", lambda *args: None)
    monkeypatch.setattr(_video_file, "_iter_decoded_packet_batches", fail_decode)
    options = _video_file._VideoFrameOptions(
        start_time=Fraction(0),
        end_time=None,
        width=None,
        height=None,
        is_key_frame=None,
        sample_interval_seconds=None,
        buffer_size=1024,
        max_input_bytes=1024,
        max_frames=1,
        max_pixels=100,
    )

    with pytest.raises(vane.VideoFileFormatError, match="primary decode failure") as raised:
        list(_video_file._iter_video_frames(Value(), options, fake_av, SimpleNamespace(), None))

    assert raised.value is primary_error
    assert reader.closed


def test_video_frames_give_interrupt_precedence_after_input_size_check(duckdb_cursor, tmp_path, monkeypatch):
    path = tmp_path / "interrupt-after-size.mp4"
    path.write_bytes(_encoded_video(frame_count=1))
    original_size = vane.VaneFileReader.size

    def interrupt_after_size(reader):
        size = original_size(reader)
        duckdb_cursor.interrupt()
        return size

    monkeypatch.setattr(vane.VaneFileReader, "size", interrupt_after_size)

    with pytest.raises(vane.InterruptException):
        list(vane.VideoFile(str(path), "video/mp4").frames(max_input_bytes=1, connection=duckdb_cursor))


def test_video_frames_close_undelivered_image_when_interrupted_before_yield(
    duckdb_cursor,
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "interrupt-before-yield.mp4"
    path.write_bytes(_encoded_video(frame_count=1))
    converted = False
    close_calls = 0
    image_refs = []

    class TrackingImage:
        def __init__(self):
            image_refs.append(weakref.ref(self))

        def close(self):
            nonlocal close_calls
            close_calls += 1
            raise RuntimeError("image close failed")

    original_check = _video_file._check_video_io

    def convert_frame(*args, **kwargs):
        nonlocal converted
        del args, kwargs
        converted = True
        return TrackingImage()

    def interrupt_after_conversion(reader, nested_io):
        if converted:
            raise RuntimeError("query interrupted before yield")
        original_check(reader, nested_io)

    monkeypatch.setattr(_video_file, "_frame_to_image", convert_frame)
    monkeypatch.setattr(_video_file, "_check_video_io", interrupt_after_conversion)

    frames = vane.VideoFile(str(path), "video/mp4").frames(connection=duckdb_cursor)
    with pytest.raises(RuntimeError, match="interrupted before yield") as raised:
        next(frames)
    gc.collect()

    assert raised.value is not None
    assert close_calls == 1
    assert len(image_refs) == 1
    assert image_refs[0]() is None


def test_video_frames_release_internal_image_ownership_before_next_decode(
    duckdb_cursor,
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "released-output.mp4"
    path.write_bytes(_encoded_video(frame_count=2))

    class TrackingImage:
        def close(self):
            pass

    image_refs: list[weakref.ReferenceType[TrackingImage]] = []

    def convert_frame(*args, **kwargs):
        del args, kwargs
        if image_refs:
            gc.collect()
            assert image_refs[-1]() is None
        image = TrackingImage()
        image_refs.append(weakref.ref(image))
        return image

    monkeypatch.setattr(_video_file, "_frame_to_image", convert_frame)
    frames = vane.VideoFile(str(path), "video/mp4").frames(connection=duckdb_cursor)
    first = next(frames)
    del first

    second = next(frames)

    assert len(image_refs) == 2
    del second
    frames.close()


def test_video_frames_throw_preserves_consumer_error_and_releases_handed_off_image(
    duckdb_cursor,
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "throw-output.mp4"
    path.write_bytes(_encoded_video(frame_count=2))
    image_refs = []
    close_error = RuntimeError("competing reader close failure")

    original_close = vane.VaneFileReader.close

    def fail_reader_close(reader):
        original_close(reader)
        raise close_error

    monkeypatch.setattr(vane.VaneFileReader, "close", fail_reader_close)

    class TrackingImage:
        def close(self):
            pass

    def convert_frame(*args, **kwargs):
        del args, kwargs
        image = TrackingImage()
        image_refs.append(weakref.ref(image))
        return image

    monkeypatch.setattr(_video_file, "_frame_to_image", convert_frame)
    frames = vane.VideoFile(str(path), "video/mp4").frames(connection=duckdb_cursor)
    first = next(frames)
    del first
    consumer_error = av.error.InvalidDataError(1094995529, "consumer stopped", "consumer")

    with pytest.raises(av.error.InvalidDataError) as raised:
        frames.throw(consumer_error)
    gc.collect()

    assert raised.value is consumer_error
    assert len(image_refs) == 1
    assert image_refs[0]() is None


def test_video_keyframes_release_wrapper_image_before_requesting_next():
    first_image: Image.Image | None = Image.new("RGB", (1, 1))
    first_image_ref = weakref.ref(first_image)

    def frame_records():
        nonlocal first_image
        assert first_image is not None
        first_record = vane.VideoFrameData(None, None, None, None, None, None, True, first_image)
        first_image = None
        yield first_record
        first_record = None
        gc.collect()
        assert first_image_ref() is None
        yield vane.VideoFrameData(None, None, None, None, None, None, True, Image.new("RGB", (1, 1)))

    images = _video_file._iter_keyframe_images(frame_records())
    first = next(images)
    del first

    second = next(images)

    second.close()
    images.close()


@pytest.mark.parametrize("method_name", ["frames", "keyframes"])
def test_video_iterators_close_reader_when_closed_early(duckdb_cursor, tmp_path, monkeypatch, method_name):
    path = tmp_path / "early-close.mp4"
    path.write_bytes(_encoded_video(frame_count=12, frame_rate=4))
    closed_readers = []
    original_close = vane.VaneFileReader.close

    def track_close(self):
        closed_readers.append(self)
        return original_close(self)

    monkeypatch.setattr(vane.VaneFileReader, "close", track_close)
    frames = getattr(vane.VideoFile(str(path), "video/mp4"), method_name)(connection=duckdb_cursor)

    first = next(frames)
    frames.close()

    assert len({id(reader) for reader in closed_readers}) == 1
    assert all(reader.closed for reader in closed_readers)
    image = first.data if isinstance(first, vane.VideoFrameData) else first
    image.close()


def test_video_frames_optional_dependencies_are_lazy(monkeypatch):
    original_import = importlib.import_module

    def fail_pillow(name, package=None):
        if name == "PIL.Image":
            raise ImportError("missing Pillow")
        return original_import(name, package)

    monkeypatch.setattr(_video_file.importlib, "import_module", fail_pillow)

    with pytest.raises(ImportError, match=r"Pillow.*vane-ai\[video\]"):
        vane.VideoFile("memory://not-opened").frames()


def test_video_metadata_preserves_unknown_frame_count(duckdb_cursor, tmp_path):
    path = tmp_path / "video.mkv"
    path.write_bytes(_encoded_video("matroska", frame_count=5))
    value = vane.VideoFile(str(path), "video/x-matroska")

    metadata = value.metadata(connection=duckdb_cursor)
    sql_metadata = duckdb_cursor.execute("SELECT video_metadata($1)", [value]).fetchone()[0]

    # FFmpeg reports zero when Matroska does not carry an exact count. Vane
    # keeps that unknown instead of estimating duration * fps.
    assert metadata.frame_count is None
    assert sql_metadata["frame_count"] is None
    assert metadata.duration == pytest.approx(5 / 24, rel=0.01)


def test_video_metadata_treats_zero_parser_sentinels_as_unknown():
    video = SimpleNamespace(
        type="video",
        disposition=av.stream.Disposition(0),
        codec_context=object(),
        width=16,
        height=12,
        time_base=Fraction(1, 1000),
        average_rate=Fraction(0, 1),
        guessed_rate=Fraction(30, 1),
        duration=0,
        frames=0,
    )
    container = SimpleNamespace(
        streams=[video],
        format=SimpleNamespace(name="matroska,webm"),
        duration=2_000_000,
    )
    av_module = SimpleNamespace(time_base=1_000_000, stream=av.stream)

    metadata = _video_file._metadata_from_container(container, "video/webm", av_module)

    assert metadata.fps == 30
    assert metadata.duration == 2
    assert metadata.frame_count is None

    video.guessed_rate = Fraction(0, 1)
    container.duration = 0
    unknown_metadata = _video_file._metadata_from_container(container, "video/webm", av_module)
    assert unknown_metadata.fps is None
    assert unknown_metadata.duration is None


def test_video_metadata_rejects_missing_container_format():
    video = SimpleNamespace(
        type="video",
        disposition=av.stream.Disposition(0),
        codec_context=object(),
        width=16,
        height=12,
        time_base=Fraction(1, 1000),
        average_rate=None,
        guessed_rate=None,
        duration=None,
        frames=None,
    )
    container = SimpleNamespace(streams=[video], format=None, duration=None)

    with pytest.raises(vane.VideoFileFormatError, match="did not report a container format"):
        _video_file._metadata_from_container(
            container,
            None,
            SimpleNamespace(time_base=1_000_000, stream=av.stream),
        )


def test_video_metadata_skips_attached_picture_streams():
    cover = SimpleNamespace(
        type="video",
        disposition=av.stream.Disposition.attached_pic,
    )
    video = SimpleNamespace(
        type="video",
        disposition=av.stream.Disposition(0),
        codec_context=object(),
        width=16,
        height=12,
        time_base=Fraction(1, 1000),
        average_rate=Fraction(24, 1),
        guessed_rate=None,
        duration=1000,
        frames=24,
    )
    container = SimpleNamespace(
        streams=[cover, video],
        format=SimpleNamespace(name="matroska"),
        duration=1_000_000,
    )

    metadata = _video_file._metadata_from_container(container, "video/x-matroska", av)

    assert (metadata.width, metadata.height) == (16, 12)
    container.streams = [cover]
    with pytest.raises(vane.VideoFileFormatError, match="does not contain a video stream"):
        _video_file._metadata_from_container(container, None, av)


def test_video_metadata_does_not_infer_visible_limit_from_missing_dimensions():
    video = SimpleNamespace(
        type="video",
        disposition=av.stream.Disposition(0),
        codec_context=object(),
        width=0,
        height=0,
    )
    container = SimpleNamespace(
        streams=[video],
        format=SimpleNamespace(name="mov,mp4,m4a,3gp,3g2,mj2"),
        duration=None,
    )

    with pytest.raises(vane.VideoFileFormatError, match="dimensions must be positive"):
        _video_file._metadata_from_container(container, "video/mp4", av, max_pixels=100)


def test_video_metadata_configures_bounded_probe(monkeypatch):
    class FakeFFmpegError(Exception):
        pass

    class FakeExitError(FakeFFmpegError):
        pass

    class ProbeStopped(RuntimeError):
        pass

    open_options = {}

    def inspect_open_options(stream, **kwargs):
        del stream
        open_options.update(kwargs)
        raise ProbeStopped

    fake_av = SimpleNamespace(
        open=inspect_open_options,
        error=SimpleNamespace(ExitError=FakeExitError, FFmpegError=FakeFFmpegError),
        time_base=1_000_000,
    )
    monkeypatch.setattr(_video_file, "_load_av", lambda: fake_av)

    with pytest.raises(ProbeStopped):
        _video_file._probe_video_metadata(
            lambda offset, size: b"x" * size,
            logical_size=4096,
            content_type=None,
            max_bytes=4096,
        )

    assert open_options["metadata_encoding"] == "utf-8"
    assert open_options["metadata_errors"] == "replace"
    assert open_options["timeout"] == (5.0, 5.0)
    assert open_options["options"] == {
        "max_pixels": str(64 * 1024 * 1024),
        "max_samples": str(1024 * 1024),
        "skip_frame": "all",
        "threads": "1",
    }
    assert open_options["stream_options"] == [open_options["options"]]
    assert open_options["stream_options"][0] is not open_options["options"]
    assert open_options["container_options"] == {
        "analyzeduration": "5000000",
        "formatprobesize": "4096",
        "fpsprobesize": "32",
        "indexmem": str(256 * 1024),
        "max_probe_packets": "256",
        "max_streams": "64",
        "probesize": "4096",
        "skip_estimate_duration_from_pts": "1",
    }


def test_video_metadata_classifies_probe_timeout(monkeypatch):
    class FakeFFmpegError(Exception):
        pass

    class FakeExitError(FakeFFmpegError):
        pass

    def time_out(stream, **kwargs):
        del stream, kwargs
        raise FakeExitError("Immediate exit requested")

    fake_av = SimpleNamespace(
        open=time_out,
        error=SimpleNamespace(ExitError=FakeExitError, FFmpegError=FakeFFmpegError),
        time_base=1_000_000,
    )
    monkeypatch.setattr(_video_file, "_load_av", lambda: fake_av)

    with pytest.raises(vane.VideoFileLimitError, match="exceeded its 5-second timeout"):
        _video_file._probe_video_metadata(
            lambda offset, size: b"x" * size,
            logical_size=2,
            content_type=None,
            max_bytes=2,
        )


def test_video_metadata_rejects_container_without_safe_stream_options(monkeypatch):
    class FakeFFmpegError(Exception):
        pass

    class FakeExitError(FakeFFmpegError):
        pass

    def reject_stream_options(stream, **kwargs):
        del stream, kwargs
        raise ValueError(_video_file._PYAV_UNSAFE_STREAM_OPTIONS_ERROR + " (e.g. MPEG)")

    fake_av = SimpleNamespace(
        open=reject_stream_options,
        error=SimpleNamespace(ExitError=FakeExitError, FFmpegError=FakeFFmpegError),
        time_base=1_000_000,
    )
    monkeypatch.setattr(_video_file, "_load_av", lambda: fake_av)

    with pytest.raises(vane.VideoFileFormatError, match="cannot be inspected safely"):
        _video_file._probe_video_metadata(
            lambda offset, size: b"x" * size,
            logical_size=2,
            content_type=None,
            max_bytes=2,
        )


def test_video_metadata_budget_is_enforced(duckdb_cursor, tmp_path):
    path = tmp_path / "video.mp4"
    path.write_bytes(_encoded_video())
    value = vane.VideoFile(str(path), "video/mp4")

    with pytest.raises(vane.VideoFileLimitError, match="max_bytes=16"):
        value.metadata(max_bytes=16, connection=duckdb_cursor)
    with pytest.raises(vane.InvalidInputException, match="max_bytes=16"):
        duckdb_cursor.execute("SELECT video_metadata($1, 16)", [value]).fetchone()


@pytest.mark.parametrize(
    "failure",
    [
        MemoryError("allocation failed"),
        av.error.MemoryError(12, "Cannot allocate memory", "parser"),
        av.error.PermissionError(13, "Permission denied", "parser"),
        RuntimeError("parser invariant failed"),
        _video_file.VideoFileFormatError("classified media failure"),
    ],
    ids=["memory", "pyav-memory", "pyav-permission", "internal", "classified-media"],
)
def test_video_metadata_probe_preserves_non_parser_errors_after_budget_exhaustion(monkeypatch, failure):
    class FakeFFmpegError(Exception):
        pass

    class FakeExitError(FakeFFmpegError):
        pass

    def failing_open(stream, **kwargs):
        del kwargs
        assert stream.read(1) == b"x"
        assert stream.read(1) == b""
        raise failure

    fake_av = SimpleNamespace(
        open=failing_open,
        error=SimpleNamespace(ExitError=FakeExitError, FFmpegError=FakeFFmpegError),
        time_base=1_000_000,
    )
    monkeypatch.setattr(_video_file, "_load_av", lambda: fake_av)

    with pytest.raises(type(failure)) as raised:
        _video_file._probe_video_metadata(
            lambda offset, size: b"x" * size,
            logical_size=2,
            content_type=None,
            max_bytes=1,
        )
    assert raised.value is failure


@pytest.mark.parametrize("content_type", ["audio/mp4", "image/png", "video/x-msvideo"])
def test_video_metadata_rejects_contradictory_mime(duckdb_cursor, tmp_path, content_type):
    path = tmp_path / "video.mp4"
    path.write_bytes(_encoded_video())
    value = vane.VideoFile(str(path), content_type)

    with pytest.raises(vane.VideoFileFormatError, match="content_type"):
        value.metadata(connection=duckdb_cursor)
    with pytest.raises(vane.InvalidInputException, match="content_type"):
        duckdb_cursor.execute("SELECT video_metadata($1)", [value]).fetchone()


@pytest.mark.parametrize(
    ("container_format", "content_type"),
    [
        ("mp4", "video/quicktime"),
        ("mp4", "video/x-m4v"),
        ("mp4", 'video/mp4; codecs="mp4v.20.9"'),
        ("mp4", "application/octet-stream"),
        ("mp4", "video/*"),
        ("matroska", "video/mkv"),
        ("ogg", "application/ogg"),
        ("ogg", "video/ogg"),
    ],
)
def test_video_metadata_accepts_compatible_and_generic_mimes(
    duckdb_cursor,
    tmp_path,
    container_format,
    content_type,
):
    path = tmp_path / f"video.{container_format}"
    path.write_bytes(_encoded_video(container_format))
    value = vane.VideoFile(str(path), content_type)

    assert value.metadata(connection=duckdb_cursor).width == 16
    assert duckdb_cursor.execute("SELECT (video_metadata($1)).width", [value]).fetchone()[0] == 16


@pytest.mark.parametrize("image_format", ["PNG", "JPEG", "GIF"])
@pytest.mark.parametrize("content_type", [None, "video/*", "video/mp4"])
def test_video_metadata_rejects_standalone_images(
    duckdb_cursor,
    tmp_path,
    image_format,
    content_type,
):
    path = tmp_path / f"image.{image_format.lower()}"
    path.write_bytes(_encoded_image(image_format))
    value = vane.VideoFile(str(path), content_type)

    with pytest.raises(vane.VideoFileFormatError, match="unsupported video container format"):
        value.metadata(connection=duckdb_cursor)
    with pytest.raises(vane.InvalidInputException, match="unsupported video container format"):
        duckdb_cursor.execute("SELECT video_metadata($1)", [value]).fetchone()


def test_video_metadata_rejects_stream_without_decoder(duckdb_cursor, tmp_path):
    path = tmp_path / "unsupported-codec.mkv"
    path.write_bytes(_video_with_unknown_codec())
    value = vane.VideoFile(str(path), "video/x-matroska")

    with pytest.raises(vane.VideoFileFormatError, match="does not have an available decoder"):
        value.metadata(connection=duckdb_cursor)
    with pytest.raises(vane.InvalidInputException, match="does not have an available decoder"):
        duckdb_cursor.execute("SELECT video_metadata($1)", [value]).fetchone()


def test_video_metadata_rejects_oversized_dimensions_before_extraction():
    video = SimpleNamespace(
        type="video",
        disposition=av.stream.Disposition(0),
        codec_context=object(),
        width=65_535,
        height=32_768,
    )
    container = SimpleNamespace(
        streams=[video],
        format=SimpleNamespace(name="matroska"),
        duration=None,
    )

    with pytest.raises(vane.VideoFileLimitError, match="metadata pixel limit"):
        _video_file._metadata_from_container(container, "video/x-matroska", av)


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS regression is Linux-specific")
def test_video_metadata_bounds_untrusted_probe_memory_and_time(tmp_path):
    pytest.importorskip("resource")
    path = tmp_path / "oversized.png"
    path.write_bytes(_oversized_png())
    script = textwrap.dedent(
        """
        import resource
        import sys

        import av
        import vane

        del av
        current_pages = int(open("/proc/self/statm").read().split()[0])
        address_space_limit = current_pages * resource.getpagesize() + 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (address_space_limit, address_space_limit))

        try:
            vane.VideoFile(sys.argv[1], "video/*").metadata()
        except vane.VideoFileError:
            pass
        else:
            raise AssertionError("oversized standalone image was accepted as VIDEOFILE")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_video_metadata_requires_a_video_stream(duckdb_cursor, tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00"
        b"\x80>\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    value = vane.VideoFile(str(path), "video/*")

    with pytest.raises(vane.VideoFileFormatError, match="does not contain a video stream"):
        value.metadata(connection=duckdb_cursor)
    with pytest.raises(vane.InvalidInputException, match="does not contain a video stream"):
        duckdb_cursor.execute("SELECT video_metadata($1)", [value]).fetchone()


def test_video_file_classifies_invalid_media_but_propagates_io(duckdb_cursor, tmp_path):
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not a video file")
    corrupt_value = vane.VideoFile(str(corrupt), "video/mp4")

    with pytest.raises(vane.VideoFileFormatError, match="supported encoded video"):
        corrupt_value.metadata(connection=duckdb_cursor)
    with pytest.raises(vane.InvalidInputException, match="supported encoded video"):
        duckdb_cursor.execute("SELECT video_metadata($1)", [corrupt_value]).fetchone()

    missing = vane.VideoFile(str(tmp_path / "missing.mp4"), "video/mp4")
    with pytest.raises(vane.IOException):
        missing.metadata(connection=duckdb_cursor)
    with pytest.raises(vane.IOException):
        duckdb_cursor.execute("SELECT video_metadata($1)", [missing]).fetchone()


def test_video_metadata_propagates_reader_failures_from_virtual_io(duckdb_cursor, tmp_path, monkeypatch):
    path = tmp_path / "video.mp4"
    path.write_bytes(_encoded_video())
    value = vane.VideoFile(str(path), "video/mp4")

    def fail_read(self, size=-1):
        raise OSError("connector read failed")

    monkeypatch.setattr(vane.VaneFileReader, "read", fail_read)
    with pytest.raises(OSError, match="connector read failed"):
        value.metadata(connection=duckdb_cursor)


def test_video_metadata_requires_videofile(duckdb_cursor):
    with pytest.raises(vane.BinderException, match="requires VIDEOFILE, not FILE"):
        duckdb_cursor.sql("SELECT video_metadata(file('memory://generic', NULL, NULL, NULL, NULL))")
    with pytest.raises(vane.BinderException, match="requires VIDEOFILE, not AUDIOFILE"):
        duckdb_cursor.sql("SELECT video_metadata(audio_file('memory://audio'))")


@pytest.mark.parametrize(
    ("max_bytes", "error_type", "message"),
    [
        (True, TypeError, "max_bytes must be int"),
        (0, ValueError, "greater than zero"),
        (64 * 1024 * 1024 + 1, ValueError, "at most"),
    ],
)
def test_video_file_python_argument_validation(max_bytes, error_type, message):
    value = vane.VideoFile("memory://not-opened")

    with pytest.raises(error_type, match=message):
        value.metadata(max_bytes=max_bytes)


def test_video_file_optional_dependency_is_lazy(monkeypatch):
    original_import = importlib.import_module

    def fail_av(name, package=None):
        if name == "av":
            raise ImportError("missing av")
        return original_import(name, package)

    monkeypatch.setattr(_video_file.importlib, "import_module", fail_av)

    value = vane.VideoFile("memory://not-opened")
    with pytest.raises(ImportError, match=r"vane-ai\[video\]"):
        value.metadata()


def test_video_file_unusable_native_dependency_is_actionable(duckdb_cursor, tmp_path, monkeypatch):
    path = tmp_path / "video.mp4"
    path.write_bytes(_encoded_video())
    value = vane.VideoFile(str(path), "video/mp4")
    original_import = importlib.import_module

    def fail_ffmpeg(name, package=None):
        if name == "av":
            raise OSError("FFmpeg libraries cannot be loaded")
        return original_import(name, package)

    monkeypatch.setattr(_video_file.importlib, "import_module", fail_ffmpeg)

    with pytest.raises(ImportError, match=r"bundled FFmpeg libraries.*vane-ai\[video\]"):
        value.metadata(connection=duckdb_cursor)
    with pytest.raises(vane.InvalidInputException, match=r"bundled FFmpeg libraries.*vane-ai\[video\]"):
        duckdb_cursor.execute("SELECT video_metadata($1)", [value]).fetchone()


def test_video_metadata_sql_preflights_dependency_before_opening_file(duckdb_cursor, tmp_path, monkeypatch):
    missing = vane.VideoFile(str(tmp_path / "missing.mp4"), "video/mp4")

    def fail_av():
        raise ImportError("install vane-ai[video]")

    monkeypatch.setattr(_video_file, "_load_av", fail_av)

    with pytest.raises(vane.InvalidInputException, match=r"install vane-ai\[video\]"):
        duckdb_cursor.execute("SELECT video_metadata($1)", [missing]).fetchone()


def test_video_metadata_sql_maps_non_exception_control_flow_to_interrupt(duckdb_cursor, tmp_path, monkeypatch):
    class StopVideoMetadata(BaseException):
        pass

    path = tmp_path / "video.bin"
    path.write_bytes(b"video")
    value = vane.VideoFile(str(path))

    def stop_metadata(*args, **kwargs):
        raise StopVideoMetadata

    monkeypatch.setattr(_video_file, "_probe_video_metadata", stop_metadata)

    with pytest.raises(vane.InterruptException):
        duckdb_cursor.execute("SELECT video_metadata($1)", [value]).fetchone()


def test_video_metadata_sql_prioritizes_connection_interrupt_over_probe_error(duckdb_cursor, tmp_path, monkeypatch):
    path = tmp_path / "video.bin"
    path.write_bytes(b"video")
    value = vane.VideoFile(str(path))

    def interrupt_then_fail(*args, **kwargs):
        duckdb_cursor.interrupt()
        raise _video_file.VideoFileFormatError("competing format failure")

    monkeypatch.setattr(_video_file, "_probe_video_metadata", interrupt_then_fail)

    with pytest.raises(vane.InterruptException):
        duckdb_cursor.execute("SELECT video_metadata($1)", [value]).fetchone()


def test_video_metadata_sql_maps_python_memory_error_to_out_of_memory(duckdb_cursor, tmp_path, monkeypatch):
    path = tmp_path / "video.bin"
    path.write_bytes(b"video")
    value = vane.VideoFile(str(path))

    def exhaust_memory(*args, **kwargs):
        raise MemoryError("allocation failed")

    monkeypatch.setattr(_video_file, "_probe_video_metadata", exhaust_memory)

    with pytest.raises(vane.OutOfMemoryException, match="ran out of memory"):
        duckdb_cursor.execute("SELECT video_metadata($1)", [value]).fetchone()


def test_video_metadata_view_reuses_cached_ranges():
    payload = bytes(range(64))
    requests = []

    def read_at(offset, size):
        requests.append((offset, size))
        return payload[offset : offset + size]

    stream = _video_file._VideoMetadataView(read_at, logical_size=len(payload), max_bytes=64)
    assert stream.read(8) == payload[:8]
    stream.seek(2)
    assert stream.read(4) == payload[2:6]
    stream.seek(-4, io.SEEK_END)
    assert stream.read() == payload[-4:]
    assert sum(size for _, size in requests) <= 64
    assert len(requests) == 2


def test_video_metadata_view_serves_one_large_request_with_one_source_read():
    payload = b"x" * (7 * 1024 * 1024)
    requests = []

    def read_at(offset, size):
        requests.append((offset, size))
        return payload[offset : offset + size]

    stream = _video_file._VideoMetadataView(read_at, logical_size=len(payload), max_bytes=len(payload))

    assert stream.read(len(payload)) == payload
    assert requests == [(0, len(payload))]


def test_video_metadata_view_preserves_cache_failures(monkeypatch):
    stream = _video_file._VideoMetadataView(lambda offset, size: b"x" * size, logical_size=4, max_bytes=4)

    def fail_cache(offset, data):
        raise RuntimeError("metadata cache insertion failed")

    monkeypatch.setattr(stream, "_cache_bytes", fail_cache)

    assert stream.read(1) == b""
    with pytest.raises(RuntimeError, match="metadata cache insertion failed"):
        stream.raise_if_error()


def test_video_metadata_view_bounds_reverse_adjacent_fetches():
    payload = bytes(range(256)) * 256
    requests = []

    def read_at(offset, size):
        requests.append((offset, size))
        return payload[offset : offset + size]

    stream = _video_file._VideoMetadataView(read_at, logical_size=len(payload), max_bytes=len(payload))
    for offset in range(len(payload) - 1, -1, -1):
        stream.seek(offset)
        assert stream.read(1) == payload[offset : offset + 1]

    assert len(requests) <= len(payload) // stream._fetch_size + 1
    assert not stream.fetch_limit_exhausted


def test_video_metadata_probe_preserves_control_flow_over_stored_reader_error(monkeypatch):
    class FakeFFmpegError(Exception):
        pass

    class FakeExitError(FakeFFmpegError):
        pass

    def interrupting_open(stream, **kwargs):
        del kwargs
        assert stream.read(1) == b""
        raise KeyboardInterrupt

    fake_av = SimpleNamespace(
        open=interrupting_open,
        error=SimpleNamespace(ExitError=FakeExitError, FFmpegError=FakeFFmpegError),
        time_base=1_000_000,
    )
    monkeypatch.setattr(_video_file, "_load_av", lambda: fake_av)

    def fail_read(offset, size):
        del offset, size
        raise OSError("connector read failed")

    with pytest.raises(KeyboardInterrupt):
        _video_file._probe_video_metadata(fail_read, logical_size=2, content_type=None, max_bytes=2)


def test_video_container_cleanup_does_not_replace_primary_error(monkeypatch):
    class FakeContainer:
        def close(self):
            raise RuntimeError("competing close failure")

    def interrupt_metadata(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(_video_file, "_metadata_from_container", interrupt_metadata)

    with pytest.raises(KeyboardInterrupt):
        with _video_file._close_container(FakeContainer()) as container:
            _video_file._metadata_from_container(container, None, None)


def test_video_nested_io_is_rejected():
    blocker = _video_file._NestedIOBlocker()
    nested_url = "https://example.test/segment.ts?token=secret"

    with pytest.raises(
        vane.VideoFileFormatError,
        match="does not permit nested external resources",
    ) as captured:
        blocker(nested_url, 0, {})
    assert nested_url not in str(captured.value)

    with pytest.raises(
        vane.VideoFileFormatError,
        match="does not permit nested external resources",
    ):
        blocker.raise_if_error()

# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import gc
import io
import math
import time
from dataclasses import FrozenInstanceError
from fractions import Fraction

import av
import numpy as np
import pytest

import vane
from tests.fast.test_video_file import _encoded_video
from vane import _video_clip as clipping


def make_av_video(
    path,
    *,
    origin=0,
    audio_offset=0,
    audio=True,
    rate=30,
    seconds=2,
    b_frames=0,
    sample_rate=16000,
    stereo=False,
    codec="mpeg4",
    color=None,
    sample_aspect_ratio=None,
):
    """Local generated fixture: increasing picture values and a timed tone."""
    with av.open(str(path), mode="w", format="mp4") as output:
        video = output.add_stream(codec, rate=rate)
        video.width, video.height, video.pix_fmt = 32, 24, "yuv420p"
        video.codec_context.max_b_frames = b_frames
        if sample_aspect_ratio is not None:
            video.codec_context.sample_aspect_ratio = sample_aspect_ratio
        if color is not None:
            (
                video.codec_context.colorspace,
                video.codec_context.color_range,
                video.codec_context.color_primaries,
                video.codec_context.color_trc,
            ) = color
        sound = output.add_stream("aac", rate=sample_rate) if audio else None
        if sound is not None:
            sound.layout = "stereo" if stereo else "mono"
        for index in range(rate * seconds):
            frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), index * 3, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = origin * rate + index, Fraction(1, rate)
            for packet in video.encode(frame):
                output.mux(packet)
        for packet in video.encode():
            output.mux(packet)
        if sound is not None:
            for offset in range(0, seconds * sample_rate, 1024):
                samples = min(1024, seconds * sample_rate - offset)
                timeline = np.arange(offset, offset + samples) / sample_rate + audio_offset
                signal = np.where((timeline >= 0.7) & (timeline < 0.9), 0.5 * np.sin(2 * np.pi * 500 * timeline), 0)
                signal = np.stack([signal, -signal] if stereo else [signal]).astype(np.float32)
                frame = av.AudioFrame.from_ndarray(signal, format="fltp", layout=sound.layout.name)
                frame.sample_rate, frame.time_base = sample_rate, Fraction(1, sample_rate)
                frame.pts = int((origin + audio_offset) * sample_rate) + offset
                for packet in sound.encode(frame):
                    output.mux(packet)
            for packet in sound.encode():
                output.mux(packet)
    return vane.VideoFile(str(path), "video/mp4")


@pytest.fixture
def source(tmp_path):
    return make_av_video(tmp_path / "source.mp4")


def decode_video(data):
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(stream))
        return [
            (float(frame.pts * frame.time_base), frame.to_ndarray(format="rgb24").mean()) for frame in frames
        ], float(stream.duration * stream.time_base)


@pytest.mark.parametrize("start,end", [(0, 2), (0.5, 1), (1 / 3, 2 / 3), (5 / 6, 1), (0.015, 0.016)])
def test_clip_half_open_intervals_preserve_displayed_pictures_and_mapping(source, start, end):
    clip = source.clip(start, end, include_audio=False)
    assert isinstance(clip, vane.VideoClip)
    assert clip.content_type == "video/mp4"
    assert clip.start_time == start
    assert clip.end_time == pytest.approx(end, abs=1 / 60000)
    assert clip.duration == pytest.approx(end - start, abs=1 / 60000)
    assert not clip.has_audio
    frames, duration = decode_video(clip.data)
    assert frames[0][0] == 0
    assert duration == pytest.approx(clip.duration, abs=1 / 60000)
    with av.open(source.url) as container:
        pictures = [
            (float(frame.pts * frame.time_base), frame.to_ndarray(format="rgb24").mean())
            for frame in container.decode(video=0)
        ]
    expected = [
        picture
        for index, picture in enumerate(pictures)
        if picture[0] < end and (index + 1 == len(pictures) or pictures[index + 1][0] > start)
    ]
    assert len(frames) == clip.frame_count == len(expected)
    for (timestamp, intensity), (source_time, source_intensity) in zip(frames, expected, strict=True):
        assert timestamp + clip.start_time == pytest.approx(max(start, source_time), abs=1 / 60000)
        assert intensity == pytest.approx(source_intensity, abs=4)
    with pytest.raises(FrozenInstanceError):
        clip.duration = 0


@pytest.mark.parametrize("origin,audio_offset", [(0, 0), (5, 0), (5, -0.2), (5, 0.6)])
@pytest.mark.parametrize("sample_rate", [16000, 44100, 48000])
def test_clip_preserves_audio_video_timing(tmp_path, origin, audio_offset, sample_rate):
    source = make_av_video(
        tmp_path / "av.mp4", origin=origin, audio_offset=audio_offset, b_frames=2, sample_rate=sample_rate
    )
    clip = source.clip(0.5, 1.1)
    assert clip.has_audio
    with av.open(io.BytesIO(clip.data)) as container:
        assert len(container.streams.video) == len(container.streams.audio) == 1
        assert container.streams.video[0].codec_context.name == "mpeg4"
        assert container.streams.audio[0].codec_context.name == "aac"
        active = []
        for frame in container.decode(audio=0):
            samples = frame.to_ndarray().reshape(-1)
            active.extend(
                float(frame.pts * frame.time_base) + index / frame.sample_rate
                for index in np.flatnonzero(np.abs(samples) > 0.15)
            )
    assert min(active) == pytest.approx(0.2, abs=0.025)
    assert max(active) == pytest.approx(0.4, abs=0.025)
    frames, duration = decode_video(clip.data)
    assert frames[0][0] == 0
    assert duration == pytest.approx(0.6)


@pytest.mark.parametrize("stereo", [False, True])
@pytest.mark.parametrize("facade", ["python", "sql"])
def test_clip_accepts_7350_hz_aac(tmp_path, duckdb_cursor, stereo, facade):
    source = make_av_video(tmp_path / "7350-hz.mp4", sample_rate=7350, stereo=stereo)
    if facade == "python":
        clip = source.clip(0.5, 1.1)
        assert clip.has_audio
        data = clip.data
    else:
        clip = duckdb_cursor.execute("SELECT video_clip($1, 0.5, 1.1)", [source]).fetchone()[0]
        assert clip["has_audio"]
        data = clip["data"]
    with av.open(io.BytesIO(data)) as container:
        video, audio = container.streams.video[0], container.streams.audio[0]
        assert audio.codec_context.name == "aac"
        assert audio.codec_context.sample_rate == 7350
        assert audio.codec_context.layout.name == ("stereo" if stereo else "mono")
        assert video.start_time == audio.start_time == 0
        assert video.duration * video.time_base == audio.duration * audio.time_base == Fraction(3, 5)
        samples = np.concatenate([frame.to_ndarray() for frame in container.decode(audio)], axis=1)
        assert samples.shape[0] == (2 if stereo else 1)
        assert np.max(np.abs(samples)) > 0.1


@pytest.mark.parametrize("sample_rate", [7350, 16000, 44100, 48000, 96000])
@pytest.mark.parametrize("start,end", [(0.7, 0.8234567), (0, 1 / 6), (0.71, 0.7105)])
def test_audio_endpoint_uses_sample_precision(tmp_path, sample_rate, start, end):
    source = make_av_video(tmp_path / "fractional.mp4", sample_rate=sample_rate)
    clip = source.clip(start, end)
    with av.open(io.BytesIO(clip.data)) as container:
        video, audio = container.streams.video[0], container.streams.audio[0]
        duration = video.duration * video.time_base
        expected_audio_end = Fraction(math.floor(duration * sample_rate), sample_rate)
        assert video.start_time == audio.start_time == 0
        assert float(duration) == clip.duration
        assert audio.duration * audio.time_base == expected_audio_end
        packets = [packet for packet in container.demux(audio) if packet.size]
        assert (packets[-1].pts + packets[-1].duration) * audio.time_base == expected_audio_end
    assert 0 <= duration - expected_audio_end < Fraction(1, sample_rate)


@pytest.mark.parametrize(
    "start,end",
    [(0.70403125, 0.76805), (0.76803125, 0.83205), (0.76796875, 0.85)],
    ids=["empty-end-768", "empty-end-832", "empty-start"],
)
@pytest.mark.parametrize("facade", ["python", "sql"])
def test_empty_boundary_audio_frame_respects_decode_limit(source, duckdb_cursor, start, end, facade):
    # Starts lie halfway between 16 kHz samples. The first two windows fill
    # their output from the previous block; the final intersecting AAC block
    # contributes no samples. The third window has an empty block at its start
    # and must continue decoding to retain the tone instead of padding silence.
    with av.open(source.url) as container:
        decoded = [(frame.pts * frame.time_base, frame.samples) for frame in container.decode(audio=0)]
    assert all(samples == 1024 for _, samples in decoded)
    sample_budget = sum(samples for timestamp, samples in decoded if timestamp < Fraction(str(end)))
    assert sample_budget < sum(samples for _, samples in decoded)
    if facade == "python":
        data = source.clip(start, end, max_decoded_samples=sample_budget).data
    else:
        data = duckdb_cursor.execute(
            "SELECT video_clip($1, $2, $3, max_decoded_samples => $4)", [source, start, end, sample_budget]
        ).fetchone()[0]["data"]
    with av.open(io.BytesIO(data)) as container:
        video, audio = container.streams.video[0], container.streams.audio[0]
        assert video.start_time == audio.start_time == 0
        duration = video.duration * video.time_base
        assert float(duration) == pytest.approx(end - start, abs=1 / 60000)
        assert audio.duration * audio.time_base == Fraction(math.floor(duration * 16000), 16000)
        samples = np.concatenate([frame.to_ndarray() for frame in container.decode(audio)], axis=1)
        assert np.max(np.abs(samples)) > 0.1


@pytest.mark.parametrize("block_size", [441, 2205])
@pytest.mark.parametrize("start", [0.005, 0.015, 0.025])
@pytest.mark.parametrize("facade", ["python", "sql"])
def test_continuous_odd_audio_blocks_at_half_sample_start(
    tmp_path, monkeypatch, duckdb_cursor, block_size, start, facade
):
    path = tmp_path / "continuous.mov"
    signal = (np.arange(44100) % 16384 - 8192).astype(np.int16)
    with av.open(str(path), "w") as output:
        video = output.add_stream("mpeg4", rate=30)
        video.width, video.height, video.pix_fmt = 32, 24, "yuv420p"
        audio = output.add_stream("pcm_s16le", rate=44100)
        audio.layout = "mono"
        for index in range(30):
            frame = av.VideoFrame.from_ndarray(np.zeros((24, 32, 3), dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 30)
            output.mux(video.encode(frame))
        output.mux(video.encode())
        for offset in range(0, 44100, block_size):
            frame = av.AudioFrame.from_ndarray(
                signal[np.newaxis, offset : offset + block_size], format="s16", layout="mono"
            )
            frame.pts, frame.time_base, frame.sample_rate = offset, Fraction(1, 44100), 44100
            output.mux(audio.encode(frame))
        output.mux(audio.encode())
    with av.open(str(path)) as container:
        frames = list(container.decode(audio=0))
        assert all(a.pts + a.samples == b.pts for a, b in zip(frames, frames[1:]))
        assert any(frame.samples % 2 for frame in frames[:-1])
    submissions, samples = [], []
    original_mux = clipping._ClipEncoder.mux

    def observe(self, stream, frame=None):
        if stream is self.audio and frame is not None:
            submissions.append((frame.pts, frame.samples))
            samples.append(frame.to_ndarray().reshape(-1).copy())
        original_mux(self, stream, frame)

    monkeypatch.setattr(clipping._ClipEncoder, "mux", observe)
    source = vane.VideoFile(str(path))
    if facade == "python":
        data = source.clip(start, 0.9).data
    else:
        data = duckdb_cursor.execute("SELECT video_clip($1, $2, 0.9)", [source, start]).fetchone()[0]["data"]
    assert all(b[0] == a[0] + a[1] for a, b in zip(submissions, submissions[1:]))
    actual = np.concatenate(samples)
    count = math.floor((Fraction(9, 10) - Fraction(str(start))) * 44100)
    first = math.ceil(Fraction(str(start)) * 44100)
    assert len(actual) == count
    # The first retained input sample is half a sample after the clip start.
    # Rounding that tie later pads once; no internal samples may be lost/added.
    assert actual[0] == 0
    np.testing.assert_array_equal(actual[1:], signal[first : first + count - 1].astype(np.float32) / 32768)
    with av.open(io.BytesIO(data)) as container:
        audio = container.streams.audio[0]
        assert audio.start_time == 0
        assert audio.duration * audio.time_base == Fraction(count, 44100)


def remux_audio_timestamps(source, path, *, shift=0):
    with av.open(source.url) as container, av.open(str(path), "w") as output:
        streams = {stream.index: output.add_stream_from_template(stream) for stream in container.streams}
        for packet in container.demux():
            if packet.dts is None:
                continue
            if packet.stream.type == "audio" and packet.pts >= 8192:
                packet.pts += shift
                packet.dts += shift
            packet.stream = streams[packet.stream.index]
            output.mux(packet)
    return vane.VideoFile(str(path))


@pytest.mark.parametrize("sample_rate", [16000, 48000])
@pytest.mark.parametrize("facade", ["python", "sql"])
def test_clip_preserves_one_sample_audio_gap(tmp_path, monkeypatch, duckdb_cursor, sample_rate, facade):
    source = make_av_video(tmp_path / "original.mp4", sample_rate=sample_rate)
    source = remux_audio_timestamps(source, tmp_path / "gap.mp4", shift=1)
    with av.open(source.url) as container:
        frames = list(container.decode(audio=0))
        assert all(frame.time_base == Fraction(1, sample_rate) for frame in frames)
        gaps = [(a.pts + a.samples, b.pts) for a, b in zip(frames, frames[1:]) if a.pts + a.samples != b.pts]
        assert gaps == [(8192, 8193)]
        expected = [
            (frame.pts, min(frame.samples, sample_rate - frame.pts)) for frame in frames if frame.pts < sample_rate
        ]
        expected.append((8192, 1))
        expected.sort()
    submissions, silence = [], []
    original_mux = clipping._ClipEncoder.mux

    def observe(self, stream, frame=None):
        if stream is self.audio and frame is not None:
            submissions.append((frame.pts, frame.samples))
            if frame.pts == 8192 and frame.samples == 1:
                silence.append(bool(np.all(frame.to_ndarray() == 0)))
        original_mux(self, stream, frame)

    monkeypatch.setattr(clipping._ClipEncoder, "mux", observe)
    if facade == "python":
        data = source.clip(0, 1).data
    else:
        data = duckdb_cursor.execute("SELECT video_clip($1, 0, 1)", [source]).fetchone()[0]["data"]
    assert submissions == expected
    assert silence == [True]
    with av.open(io.BytesIO(data)) as container:
        audio = container.streams.audio[0]
        assert audio.duration * audio.time_base == 1


def test_clip_rejects_one_sample_audio_overlap(tmp_path):
    source = make_av_video(tmp_path / "original.mp4")
    source = remux_audio_timestamps(source, tmp_path / "overlap.mp4", shift=-1)
    with av.open(source.url) as container:
        frames = list(container.decode(audio=0))
        overlaps = [(a.pts + a.samples, b.pts) for a, b in zip(frames, frames[1:]) if a.pts + a.samples > b.pts]
        assert overlaps == [(8192, 8191)]
    with pytest.raises(vane.VideoFileFormatError, match="overlapping audio timestamps") as caught:
        source.clip(0, 1)
    assert not native_traceback_owners(caught.value)


def test_clip_accepts_sub_tick_audio_timestamp_rounding(tmp_path, monkeypatch):
    source = make_av_video(tmp_path / "original.mp4", sample_rate=44100)
    source = remux_audio_timestamps(source, tmp_path / "coarse-clock.mkv")
    with av.open(source.url) as container:
        frames = list(container.decode(audio=0))
        assert all(frame.time_base == Fraction(1, 1000) for frame in frames)
        assert any(
            b.pts * b.time_base != a.pts * a.time_base + Fraction(a.samples, 44100) for a, b in zip(frames, frames[1:])
        )
    submissions = []
    original_mux = clipping._ClipEncoder.mux

    def observe(self, stream, frame=None):
        if stream is self.audio and frame is not None:
            submissions.append((frame.pts, frame.samples))
        original_mux(self, stream, frame)

    monkeypatch.setattr(clipping._ClipEncoder, "mux", observe)
    assert source.clip(0, 1).has_audio
    # Only boundary frames may be shortened or padded. Coarse source timestamps
    # must not create silence between otherwise continuous decoded AAC frames.
    assert len(submissions) > 10
    assert all(samples == 1024 for _, samples in submissions[2:-1])
    assert all(b[0] == a[0] + a[1] for a, b in zip(submissions, submissions[1:]))


@pytest.mark.parametrize("colorspace,primaries,trc", [(1, 1, 1), (6, 6, 6)])
@pytest.mark.parametrize("color_range", [1, 2])
def test_clip_preserves_color_interpretation(tmp_path, colorspace, primaries, trc, color_range):
    path = tmp_path / "colors.mp4"
    colors = np.array([[210, 35, 25], [30, 190, 80], [25, 45, 210], [200, 160, 35]], dtype=np.uint8)
    pixels = np.repeat(np.repeat(colors[np.newaxis, :, :], 48, axis=0), 16, axis=1)
    with av.open(str(path), "w") as container:
        video = container.add_stream("mpeg4", rate=10)
        video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
        video.codec_context.colorspace = colorspace
        video.codec_context.color_range = color_range
        video.codec_context.color_primaries = primaries
        video.codec_context.color_trc = trc
        for index in range(10):
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24").reformat(
                format="yuv420p",
                dst_colorspace="ITU709" if colorspace == 1 else "ITU601",
                dst_color_range="MPEG" if color_range == 1 else "JPEG",
            )
            frame.color_primaries, frame.color_trc = primaries, trc
            frame.pts, frame.time_base = index, Fraction(1, 10)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        original = next(container.decode(video=0)).to_ndarray(format="rgb24", dst_color_range="JPEG")
    clip = vane.VideoFile(str(path)).clip(0.1, 0.8, include_audio=False)
    with av.open(io.BytesIO(clip.data)) as container:
        codec = container.streams.video[0].codec_context
        assert (codec.colorspace, codec.color_range, codec.color_primaries, codec.color_trc) == (1, 1, primaries, trc)
        for frame in container.decode(video=0):
            actual = frame.to_ndarray(format="rgb24", dst_color_range="JPEG")
            assert np.abs(actual.astype(float) - original).mean() < 3


@pytest.mark.parametrize("audio,rate", [(False, 10), (True, 10), (True, 2)])
@pytest.mark.parametrize("facade", ["python", "sql"])
@pytest.mark.parametrize("declared_sar", [None, Fraction(3)])
def test_clip_initializes_video_properties_from_bitstream(tmp_path, duckdb_cursor, audio, rate, facade, declared_sar):
    path = tmp_path / "bitstream-properties.mp4"
    source = make_av_video(
        path, audio=audio, rate=rate, codec="libx264", b_frames=2, color=(1, 1, 1, 1), sample_aspect_ratio=Fraction(2)
    )
    # The H.264 SPS carries SAR 2:1 and BT.709, discovered when decoding frames.
    # An explicit container SAR must take precedence over the bitstream's value.
    payload = path.read_bytes()
    assert payload.count(b"colrnclx") == 1
    assert payload.count(b"pasp") == 1
    payload = payload.replace(b"colrnclx", b"freenclx")
    if declared_sar is None:
        payload = payload.replace(b"pasp", b"free")
    else:
        offset = payload.index(b"pasp") + 4
        payload = (
            payload[:offset]
            + declared_sar.numerator.to_bytes(4, "big")
            + declared_sar.denominator.to_bytes(4, "big")
            + payload[offset + 8 :]
        )
    path.write_bytes(payload)
    decoder_options, probe_options = clipping._video_probe_options(path.stat().st_size)
    with av.open(
        str(path), options=decoder_options, container_options=probe_options, stream_options=[decoder_options.copy()]
    ) as container:
        video = container.streams.video[0]
        assert (video.codec_context.color_primaries, video.codec_context.color_trc) == (2, 2)
        assert video.sample_aspect_ratio == declared_sar
        clipping._configure_video_decoder(video)
        frames = list(container.decode(video))
        assert frames and all((frame.color_primaries, frame.color_trc) == (1, 1) for frame in frames)
        assert video.codec_context.sample_aspect_ratio == 2
    if facade == "python":
        clip = source.clip(0.1, 0.6)
        data, has_audio = clip.data, clip.has_audio
    else:
        clip = duckdb_cursor.execute("SELECT video_clip($1, 0.1, 0.6)", [source]).fetchone()[0]
        data, has_audio = clip["data"], clip["has_audio"]
    assert has_audio is audio
    with av.open(io.BytesIO(data)) as container:
        video = container.streams.video[0]
        assert (video.codec_context.color_primaries, video.codec_context.color_trc) == (1, 1)
        assert video.sample_aspect_ratio == (declared_sar or 2)
        assert video.display_aspect_ratio == Fraction(4, 3) * (declared_sar or 2)
        assert video.duration * video.time_base == Fraction(1, 2)
        frames = list(container.decode(video))
        assert len(frames) == (5 if rate == 10 else 2)
        assert all((frame.color_primaries, frame.color_trc) == (1, 1) for frame in frames)


def test_clip_rejects_changing_sample_aspect_ratio(source, monkeypatch):
    original_video_frame = clipping._ClipEncoder.video_frame

    def change_aspect_ratio(self, frame, sample_aspect_ratio):
        if frame.pts * frame.time_base >= Fraction(1, 2):
            sample_aspect_ratio = Fraction(2)
        original_video_frame(self, frame, sample_aspect_ratio)

    monkeypatch.setattr(clipping._ClipEncoder, "video_frame", change_aspect_ratio)
    with pytest.raises(vane.VideoFileFormatError, match="changing sample aspect ratios") as caught:
        source.clip(0.1, 0.8)
    assert not native_traceback_owners(caught.value)


@pytest.mark.parametrize("facade", ["python", "sql"])
def test_clip_rejects_ambiguous_sar_in_reordered_frames(tmp_path, duckdb_cursor, facade):
    path = tmp_path / "changing-sar.mp4"
    with av.open(str(path), "w") as output:
        video = output.add_stream("libx264", rate=10)
        video.width, video.height, video.pix_fmt = 32, 24, "yuv420p"
        # Each segment has its own SPS and closed GOP. Pictures before 1 second
        # are square-pixel; only the second segment has a 2:1 sample aspect ratio.
        for segment in (0, 1):
            codec = av.CodecContext.create("libx264", "w")
            codec.width, codec.height, codec.pix_fmt = 32, 24, "yuv420p"
            codec.time_base, codec.framerate = Fraction(1, 10), Fraction(10)
            codec.max_b_frames = 2
            codec.sample_aspect_ratio = Fraction(segment + 1)
            codec.options = {"x264-params": "repeat-headers=1"}
            for index in range(10):
                frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), segment * 80, dtype=np.uint8), format="rgb24")
                frame.pts, frame.time_base = index + segment * 10, Fraction(1, 10)
                for packet in codec.encode(frame):
                    packet.stream = video
                    output.mux(packet)
            for packet in codec.encode():
                packet.stream = video
                output.mux(packet)
    path.write_bytes(path.read_bytes().replace(b"pasp", b"free"))
    decoder_options, probe_options = clipping._video_probe_options(path.stat().st_size)
    with av.open(
        str(path), options=decoder_options, container_options=probe_options, stream_options=[decoder_options.copy()]
    ) as container:
        video = container.streams.video[0]
        assert video.sample_aspect_ratio is None
        clipping._configure_video_decoder(video)
        observed = [
            (
                frame.pts * frame.time_base,
                video.codec_context.sample_aspect_ratio,
                frame.to_ndarray(format="rgb24").mean(),
            )
            for frame in container.decode(video)
        ]
        assert video.codec_context.has_b_frames
        # Decoder SAR already describes the second segment while it returns an
        # earlier picture. Inferring a per-frame ratio from that state is unsafe.
        assert any(timestamp == Fraction(4, 5) and sar == 2 and pixels < 4 for timestamp, sar, pixels in observed)
    source = vane.VideoFile(str(path))
    if facade == "python":
        data = source.clip(0.1, 0.6, include_audio=False).data
        with pytest.raises(vane.VideoFileFormatError, match="changing decoder sample aspect ratios") as caught:
            source.clip(0.8, 0.9, include_audio=False)
    else:
        data = duckdb_cursor.execute("SELECT video_clip($1, 0.1, 0.6)", [source]).fetchone()[0]["data"]
        with pytest.raises(vane.InvalidInputException, match="changing decoder sample aspect ratios") as caught:
            duckdb_cursor.execute("SELECT video_clip($1, 0.8, 0.9)", [source]).fetchone()
    assert not native_traceback_owners(caught.value)
    with av.open(io.BytesIO(data)) as container:
        assert container.streams.video[0].sample_aspect_ratio == 1


@pytest.mark.parametrize("attribute", ["color_primaries", "color_trc"])
def test_clip_still_rejects_changing_color(source, monkeypatch, attribute):
    original_video_frame = clipping._ClipEncoder.video_frame

    def change_color(self, frame, sample_aspect_ratio):
        if frame.pts * frame.time_base >= Fraction(1, 2):
            setattr(frame, attribute, 1)
        original_video_frame(self, frame, sample_aspect_ratio)

    monkeypatch.setattr(clipping._ClipEncoder, "video_frame", change_color)
    with pytest.raises(vane.VideoFileFormatError, match="changing color primaries or transfer") as caught:
        source.clip(0.1, 0.8)
    assert not native_traceback_owners(caught.value)


@pytest.mark.parametrize("facade", ["python", "sql"])
def test_clip_replaces_invalid_utf8_metadata(tmp_path, duckdb_cursor, facade):
    path = tmp_path / "latin1.avi"
    with av.open(str(path), "w", format="avi", metadata_encoding="latin-1") as container:
        container.metadata["title"] = "caf\u00e9"
        video = container.add_stream("mpeg4", rate=10)
        video.width, video.height, video.pix_fmt = 32, 24, "yuv420p"
        for index in range(10):
            frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), index * 20, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 10)
            container.mux(video.encode(frame))
        container.mux(video.encode())
    assert b"caf\xe9" in path.read_bytes()
    source = vane.VideoFile(str(path), "video/avi")
    if facade == "python":
        data = source.clip(0.1, 0.6).data
    else:
        data = duckdb_cursor.execute("SELECT video_clip($1, 0.1, 0.6)", [source]).fetchone()[0]["data"]
    with av.open(io.BytesIO(data)) as container:
        assert "title" not in container.metadata
        frames = list(container.decode(video=0))
        assert len(frames) == 5
        assert [frame.pts * frame.time_base for frame in frames] == [Fraction(index, 10) for index in range(5)]


def test_silent_source_and_explicit_audio_omission(tmp_path, source):
    silent = make_av_video(tmp_path / "silent.mp4", audio=False)
    assert not silent.clip(0, 1).has_audio
    assert not source.clip(0, 1, include_audio=False).has_audio


def test_clip_preserves_stereo_channels(tmp_path):
    source = make_av_video(tmp_path / "stereo.mp4", stereo=True)
    clip = source.clip(0.71, 0.81)
    with av.open(io.BytesIO(clip.data)) as container:
        assert container.streams.audio[0].codec_context.layout.name == "stereo"
        samples = np.concatenate([frame.to_ndarray() for frame in container.decode(audio=0)], axis=1)
    assert np.mean(np.abs(samples[0])) > 0.1
    assert np.mean(np.abs(samples[0] + samples[1])) < 0.01


def test_clip_preserves_variable_frame_intervals_and_display_aspect(tmp_path):
    path = tmp_path / "vfr.mp4"
    with av.open(str(path), "w") as output:
        video = output.add_stream("mpeg4", rate=10)
        video.width, video.height, video.pix_fmt = 32, 24, "yuv420p"
        video.codec_context.sample_aspect_ratio = Fraction(2, 1)
        for pts in (0, 1, 3, 4, 8, 9):
            frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), pts * 20, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 10)
            for packet in video.encode(frame):
                output.mux(packet)
        for packet in video.encode():
            output.mux(packet)
    clip = vane.VideoFile(str(path)).clip(0.15, 0.85)
    frames, duration = decode_video(clip.data)
    assert [frame[0] for frame in frames] == pytest.approx([0, 0.15, 0.25, 0.65])
    assert duration == pytest.approx(0.7)
    with av.open(io.BytesIO(clip.data)) as container:
        assert container.streams.video[0].sample_aspect_ratio == 2


def test_clip_before_delayed_audio_track_keeps_a_silent_timeline(tmp_path):
    source = make_av_video(tmp_path / "delayed.mp4", audio_offset=1)
    clip = source.clip(0, 0.5)
    assert clip.has_audio
    with av.open(io.BytesIO(clip.data)) as container:
        assert all(np.max(np.abs(frame.to_ndarray())) < 0.001 for frame in container.decode(audio=0))


def test_clip_expression_does_not_load_codecs_or_open_media(monkeypatch):
    def unavailable():
        raise ImportError("codec unavailable")

    monkeypatch.setattr(clipping, "_load_av", unavailable)
    value = vane.VideoFile("/not/a/real/video.mp4")
    assert isinstance(vane.video_clip(value, 0, 1), vane.Expression)
    with pytest.raises(ImportError, match="codec unavailable"):
        value.clip(0, 1)


def test_clip_honors_file_byte_window_and_connection_policy(source, tmp_path):
    payload = open(source.url, "rb").read()
    path = tmp_path / "envelope.bin"
    path.write_bytes(b"prefix" + payload + b"suffix")
    value = vane.VideoFile(str(path), "video/mp4", 6, len(payload))
    assert value.clip(0, 0.5).frame_count == 15
    with vane.connect(config={"enable_external_access": False}) as connection:
        with pytest.raises(vane.PermissionException):
            value.clip(0, 0.5, connection=connection)


@pytest.mark.parametrize(
    "options,error",
    [
        ({"start_time": -1}, ValueError),
        ({"end_time": float("nan")}, ValueError),
        ({"start_time": 1, "end_time": 1}, ValueError),
        ({"start_time": True}, TypeError),
        ({"include_audio": 1}, TypeError),
        ({"max_output_bytes": 0}, ValueError),
        ({"max_output_bytes": 257 * 1024**2}, ValueError),
        ({"max_duration": 0}, ValueError),
        ({"timeout_seconds": 0}, ValueError),
        ({"timeout_seconds": float("inf")}, ValueError),
        ({"max_duration": 0.1}, vane.VideoFileLimitError),
    ],
)
def test_invalid_options_fail_before_io(options, error):
    arguments = {"start_time": 0, "end_time": 1, **options}
    value = vane.VideoFile("/not/a/real/video.mp4")
    with pytest.raises(error):
        value.clip(**arguments)
    with pytest.raises(error):
        vane.video_clip(value, **arguments)


@pytest.mark.parametrize(
    "options,limit",
    [
        ({"max_input_bytes": 1}, "max_input_bytes"),
        ({"max_decoded_frames": 1}, "max_decoded_frames"),
        ({"max_decoded_samples": 1}, "max_decoded_samples"),
        ({"max_pixels": 1}, "max_pixels"),
        ({"max_output_bytes": 128}, "max_output_bytes"),
    ],
)
def test_clip_enforces_resource_limits(source, options, limit):
    with pytest.raises(vane.VideoFileLimitError, match=limit):
        source.clip(0, 1, **options)


def native_traceback_owners(error):
    """Inspect retained real-error frames without relying on allocator RSS."""
    native_types = (
        av.frame.Frame,
        av.buffer.Buffer,
        av.container.Container,
        av.stream.Stream,
        av.CodecContext,
        av.AudioResampler,
    )
    owners, pending, seen = [], [error], set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, native_types):
            owners.append(value)
        elif isinstance(value, BaseException):
            pending.extend([value.__cause__, value.__context__])
            tb = value.__traceback__
            while tb is not None:
                if tb.tb_frame.f_globals.get("__name__") == "vane._video_clip":
                    pending.extend(tb.tb_frame.f_locals.values())
                tb = tb.tb_next
        elif isinstance(value, clipping._ClipEncoder):
            pending.extend(vars(value).values())
        elif isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
    return owners


@pytest.mark.parametrize(
    "options",
    [
        {"max_decoded_frames": 2},
        {"max_decoded_samples": 1},
        {"max_output_bytes": 128},
        {"start_time": 1.99, "end_time": 2.3},
        {"end_time": 1 / 60000},
    ],
)
def test_retained_clip_errors_release_native_owners(source, options):
    errors = []
    for _ in range(3):
        try:
            source.clip(**{"start_time": 0, "end_time": 1, **options})
        except vane.VideoFileError as error:
            errors.append(error)
    assert len(errors) == 3
    gc.collect()
    for error in errors:
        assert error.__traceback__ is not None
        owners = native_traceback_owners(error)
        assert not owners, [type(owner).__name__ for owner in owners]


@pytest.mark.parametrize("limit", ["max_output_bytes", "max_decoded_frames"])
def test_early_audio_stays_bounded_and_is_released_on_error(tmp_path, monkeypatch, limit):
    source = make_av_video(tmp_path / "early-audio.mp4", rate=2)
    original_mux = clipping._ClipEncoder.mux
    early_audio = []

    def observe(self, stream, frame=None):
        if stream is self.audio and self.frame_count == 0:
            early_audio.append(True)
        original_mux(self, stream, frame)

    monkeypatch.setattr(clipping._ClipEncoder, "mux", observe)
    with pytest.raises(vane.VideoFileLimitError, match=limit) as caught:
        source.clip(0.1, 0.6, **{limit: 1})
    assert early_audio
    assert not native_traceback_owners(caught.value)


@pytest.mark.parametrize("failure", [MemoryError("allocation"), KeyboardInterrupt(), OSError("I/O failure")])
def test_clip_preserves_system_failures_and_closes_reader(source, monkeypatch, failure):
    readers = []
    original_open = vane.VideoFile.open

    def opening(self, **options):
        reader = original_open(self, **options)
        readers.append(reader)
        return reader

    def mux(*args):
        raise failure

    monkeypatch.setattr(vane.VideoFile, "open", opening)
    monkeypatch.setattr(clipping._ClipEncoder, "mux", mux)
    with pytest.raises(type(failure)) as caught:
        source.clip(0, 1)
    assert len(readers) == 1 and readers[0].closed
    assert not native_traceback_owners(caught.value)


@pytest.mark.parametrize("start,end", [(3, 4), (1.9, 2.2)])
def test_clip_rejects_out_of_source_intervals(source, start, end):
    with pytest.raises(vane.VideoFileFormatError):
        source.clip(start, end)


def test_corrupt_video_and_missing_file_stay_errors(tmp_path):
    with pytest.raises((OSError, vane.IOException)):
        vane.VideoFile(str(tmp_path / "missing.mp4")).clip(0, 1)
    invalid = tmp_path / "invalid.mp4"
    invalid.write_bytes(b"not a video")
    with pytest.raises(vane.VideoFileFormatError):
        vane.VideoFile(str(invalid)).clip(0, 1)


def test_video_clip_facades_and_sql_null_rows(source, duckdb_cursor):
    sql = str(vane.ConstantExpression(source))
    expression = vane.video_clip(vane.col("file"), 0.5, 1, include_audio=False)
    method = vane.col("file").video_clip(0.5, 1, include_audio=False)
    assert str(expression) == str(method)
    result = duckdb_cursor.sql(f"SELECT {sql} AS file").select(expression).fetchone()[0]
    sql_result = duckdb_cursor.sql(f"SELECT video_clip({sql}, 0.5, 1, include_audio => false)").fetchone()[0]
    assert result == sql_result
    assert decode_video(result["data"])[1] == pytest.approx(0.5)
    rows = duckdb_cursor.sql(
        f"SELECT video_clip(CASE WHEN i = 1 THEN NULL ELSE {sql} END, 0, 0.5) FROM range(3) t(i)"
    ).fetchall()
    assert rows[1] == (None,)
    assert rows[0][0]["has_audio"] and rows[2][0]["has_audio"]
    assert duckdb_cursor.sql(f"SELECT video_clip({sql}, NULL, 1)").fetchone() == (None,)


@pytest.mark.parametrize("value", ["'some.mp4'", "file('some.mp4', NULL, NULL, NULL, NULL)", "NULL::AUDIOFILE"])
def test_sql_requires_video_type(duckdb_cursor, value):
    with pytest.raises(vane.BinderException, match="requires VIDEOFILE"):
        duckdb_cursor.sql(f"SELECT video_clip({value}, 0, 1)")


@pytest.mark.parametrize("option", ["max_output_bytes => -1", "max_duration => 0", "timeout_seconds => 0"])
def test_sql_option_errors(duckdb_cursor, source, option):
    with pytest.raises(vane.InvalidInputException):
        duckdb_cursor.sql(f"SELECT video_clip({vane.ConstantExpression(source)}, 0, 1, {option})").fetchall()


def test_sql_missing_native_backend_fails_explicitly(source):
    with vane.connect(config={"video_backend": "native"}) as connection:
        with pytest.raises((vane.InvalidInputException, vane.BinderException), match="native"):
            connection.sql(f"SELECT video_clip({vane.ConstantExpression(source)}, 0, 1)").fetchall()


def test_sql_chunk_output_budget_stops_before_encoding_another_row(source, monkeypatch, duckdb_cursor):
    # Stand in for large encoded clips while exercising the actual SQL vector
    # output budget, without spending CPU on hundreds of megabytes of video.
    data = bytes(64 * 1024**2)
    calls = []

    def encoded(file, options, context):
        calls.append(options["max_output_bytes"])
        return (data, "video/mp4", 0.0, 1.0, 1.0, 30, False)

    monkeypatch.setattr(clipping, "_scalar_video_clip", encoded)
    with pytest.raises(vane.OutOfRangeException, match="output budget per chunk"):
        duckdb_cursor.sql(f"SELECT video_clip({vane.ConstantExpression(source)}, 0, 1) FROM range(5)").fetchall()
    assert calls == [64 * 1024**2] * 4


def test_sql_deadline_covers_helper_dispatch(source, monkeypatch, duckdb_cursor):
    original = clipping._scalar_video_clip

    def delayed(*args):
        time.sleep(0.08)
        return original(*args)

    monkeypatch.setattr(clipping, "_scalar_video_clip", delayed)
    with pytest.raises(vane.OutOfRangeException, match="timeout_seconds"):
        duckdb_cursor.sql(
            f"SELECT video_clip({vane.ConstantExpression(source)}, 0, 0.5, timeout_seconds => 0.05)"
        ).fetchall()


@pytest.mark.parametrize("fail", [False, True])
def test_sql_revokes_clip_execution_context_after_success_and_failure(source, monkeypatch, duckdb_cursor, fail):
    retained = []
    original = clipping._scalar_video_clip

    def observe(file, options, context):
        retained.append(context)
        if fail:
            raise vane.VideoFileFormatError("invalid fixture")
        return original(file, options, context)

    monkeypatch.setattr(clipping, "_scalar_video_clip", observe)
    query = f"SELECT video_clip({vane.ConstantExpression(source)}, 0, 0.5)"
    if fail:
        with pytest.raises(vane.InvalidInputException, match="invalid fixture"):
            duckdb_cursor.sql(query).fetchall()
    else:
        assert duckdb_cursor.sql(query).fetchone()[0]["has_audio"]
    assert len(retained) == 1
    with pytest.raises(vane.InvalidInputException, match="no longer active"):
        retained[0]._check_interrupted()


@pytest.mark.parametrize("stage", ["open", "encode", "close"])
def test_timeout_includes_open_encoding_and_reader_cleanup(source, monkeypatch, stage):
    clock = [0.0]
    monkeypatch.setattr(clipping.time, "monotonic", lambda: clock[0])
    original_open = vane.VideoFile.open
    original_mux = clipping._ClipEncoder.mux

    class Reader:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def _close_and_check_interrupted(self):
            self.inner._close_and_check_interrupted()
            if stage == "close":
                clock[0] = 2.0

    def opening(self, **options):
        value = Reader(original_open(self, **options))
        if stage == "open":
            clock[0] = 2.0
        return value

    def mux(self, *args):
        if stage == "encode":
            clock[0] = 2.0
        return original_mux(self, *args)

    monkeypatch.setattr(vane.VideoFile, "open", opening)
    monkeypatch.setattr(clipping._ClipEncoder, "mux", mux)
    with pytest.raises(vane.VideoFileLimitError, match="timeout_seconds") as caught:
        source.clip(0, 1, timeout_seconds=1)
    assert not native_traceback_owners(caught.value)


def test_clip_output_bounds_trailer_and_sparse_writes():
    with clipping._ClipOutput(16, lambda: None) as output:
        output.write(b"12345678")
        output.seek(14)
        with pytest.raises(vane.VideoFileLimitError):
            output.write(b"abc")
        assert len(output.getvalue()) == 8


def test_frame_boundary_at_48fps(tmp_path):
    path = tmp_path / "48fps.mp4"
    path.write_bytes(_encoded_video(frame_count=48, frame_rate=48))
    clip = vane.VideoFile(str(path)).clip(5 / 6, 1, include_audio=False)
    assert clip.frame_count == 8
    frames, duration = decode_video(clip.data)
    assert len(frames) == 8
    assert duration == pytest.approx(1 / 6, abs=1 / 60000)


@pytest.mark.parametrize("start,end,count", [(0, 1 / 6 + 1e-9, 5), (0.5 - 1e-9, 0.75, 8)])
def test_partial_boundary_pictures_below_output_precision_do_not_fail(source, start, end, count):
    clip = source.clip(start, end, include_audio=False)
    frames, duration = decode_video(clip.data)
    assert len(frames) == clip.frame_count == count
    assert duration == pytest.approx(end - start, abs=1 / 60000)

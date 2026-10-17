# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError
from fractions import Fraction
from types import SimpleNamespace

import av
import numpy as np
import pytest

import vane
from vane import _media_probe, _video_file


def _container(path, *, video=True, audio=1, subtitles=True):
    with av.open(str(path), "w", format="matroska") as output:
        visual = output.add_stream("mpeg4", rate=10) if video else None
        if visual is not None:
            visual.width, visual.height, visual.pix_fmt = 16, 12, "yuv420p"
        sounds = [output.add_stream("pcm_s16le", rate=16000) for _ in range(audio)]
        for index, stream in enumerate(sounds):
            stream.layout = "mono"
            stream.metadata["language"] = "eng" if index == 0 else "fra"
            stream.disposition = av.stream.Disposition.default if index == 0 else av.stream.Disposition(0)
        subtitle = output.add_mux_stream("subrip") if subtitles else None
        if subtitle is not None:
            subtitle.metadata["language"] = "zho"
            subtitle.disposition = av.stream.Disposition.forced
            subtitle.time_base = Fraction(1, 1000)
        if visual is not None:
            for index in range(2):
                frame = av.VideoFrame.from_ndarray(np.zeros((12, 16, 3), dtype=np.uint8), format="rgb24")
                frame.pts = index
                for packet in visual.encode(frame):
                    output.mux(packet)
            for packet in visual.encode():
                output.mux(packet)
        for stream in sounds:
            frame = av.AudioFrame.from_ndarray(np.zeros((1, 3200), dtype=np.int16), format="s16", layout="mono")
            frame.sample_rate, frame.pts = 16000, 0
            for packet in stream.encode(frame):
                output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        if subtitle is not None:
            packet = av.Packet("字幕".encode())
            packet.stream = subtitle
            packet.pts = packet.dts = 0
            packet.duration = 200
            packet.time_base = Fraction(1, 1000)
            output.mux(packet)


def test_probe_real_video_audio_and_subtitle_tracks(duckdb_cursor, tmp_path):
    path = tmp_path / "media.mkv"
    _container(path, audio=2)
    result = vane.probe_media(vane.VideoFile(str(path), "video/x-matroska"), connection=duckdb_cursor)
    assert "matroska" in result.formats
    assert [stream.index for stream in result.streams] == [0, 1, 2, 3]
    assert [stream.kind for stream in result.streams] == ["video", "audio", "audio", "subtitle"]
    visual, english, french, subtitle = result.streams
    assert (visual.width, visual.height) == (16, 12)
    assert (english.sample_rate, english.channels, english.language, english.default) == (16000, 1, "eng", True)
    assert (french.language, french.default) == ("fra", False)
    assert (subtitle.codec, subtitle.language, subtitle.forced) == ("srt", "zho", True)
    assert not any(stream.attached_picture for stream in result.streams)
    with pytest.raises(FrozenInstanceError):
        english.index = 10


@pytest.mark.parametrize("video,audio,kinds", [(True, 0, ["video"]), (False, 1, ["audio"])])
def test_probe_reports_absent_tracks_without_requiring_video(duckdb_cursor, tmp_path, video, audio, kinds):
    path = tmp_path / "media.mkv"
    _container(path, video=video, audio=audio, subtitles=False)
    result = vane.probe_media(vane.File(str(path)), connection=duckdb_cursor)
    assert [stream.kind for stream in result.streams] == kinds


def test_probe_honors_logical_file_range_and_budget(duckdb_cursor, tmp_path):
    path = tmp_path / "media.mkv"
    _container(path)
    data = path.read_bytes()
    path.write_bytes(b"not a media header" + data + b"not a media trailer")
    value = vane.File(str(path), position=18, size=len(data))
    result = vane.probe_media(value, connection=duckdb_cursor, buffer_size=128)
    assert len(result.streams) == 3
    with pytest.raises(vane.VideoFileLimitError):
        vane.probe_media(value, max_bytes=16, connection=duckdb_cursor)


def test_probe_unknowns_and_attached_artwork():
    stream = SimpleNamespace(
        index=2,
        type="video",
        codec_context=None,
        time_base=None,
        start_time=None,
        duration=None,
        disposition=av.stream.Disposition.attached_pic,
        metadata={"language": "und"},
    )
    container = SimpleNamespace(format=SimpleNamespace(name="matroska,webm"), streams=[stream], duration=None)
    result = _media_probe._inspect_media(container, av)
    assert result.duration is None
    assert result.streams[0].attached_picture
    assert result.streams[0].language is None
    assert result.streams[0].codec is None
    assert result.streams[0].width is None
    assert result.streams[0].time_base is None
    assert result.streams[0].duration is None


def test_probe_timing_sentinels_and_invalid_values():
    assert _media_probe._seconds(0, Fraction(1, 1000), duration=True) is None
    assert _media_probe._seconds(0, Fraction(1, 1000)) == 0
    assert _media_probe._seconds(-1000, Fraction(1, 1000)) == -1
    for value in (-1, float("nan"), float("inf")):
        with pytest.raises(vane.VideoFileFormatError, match="timing metadata"):
            _media_probe._seconds(value, Fraction(1, 1000), duration=True)


@pytest.mark.parametrize(
    "options,error",
    [
        ({"max_bytes": 0}, ValueError),
        ({"max_bytes": True}, TypeError),
        ({"buffer_size": 0}, ValueError),
        ({"buffer_size": 2**31}, OverflowError),
    ],
)
def test_probe_validates_options_before_loading_codecs(monkeypatch, options, error):
    def unexpected():
        pytest.fail("invalid options must not load PyAV")

    monkeypatch.setattr(_media_probe, "_load_av", unexpected)
    with pytest.raises(error):
        vane.probe_media(vane.File("memory://unused"), **options)


def test_probe_requires_file_value():
    with pytest.raises(TypeError, match="File value"):
        vane.probe_media("video.mp4")


def test_probe_preserves_io_failure(duckdb_cursor, tmp_path, monkeypatch):
    path = tmp_path / "media.mkv"
    _container(path)

    def failure(self, size=-1):
        raise OSError("authored read failure")

    monkeypatch.setattr(vane.VaneFileReader, "read", failure)
    with pytest.raises(OSError, match="authored read failure"):
        vane.probe_media(vane.File(str(path)), connection=duckdb_cursor)


def test_probe_does_not_return_partial_inventory_after_inspector_failure(monkeypatch):
    class Rejected(RuntimeError):
        pass

    def reject(container, av_module):
        raise Rejected("incomplete inventory")

    closed = []
    container = SimpleNamespace(close=lambda: closed.append(True))
    fake = SimpleNamespace(open=lambda *args, **kwargs: container, error=av.error)
    monkeypatch.setattr(_video_file, "_load_av", lambda: fake)
    with pytest.raises(Rejected):
        _video_file._probe_container(lambda offset, size: b"x" * size, 1024, 1024, 128, reject)
    assert closed == [True]

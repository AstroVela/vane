# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Real container timestamps and normal missing tracks, without model weights."""

from fractions import Fraction
from io import BytesIO

import numpy as np
import pytest

from vane.ai.video import ClipWindow, TimedContent, VaneVideoSource


def make_media(path, *, audio=True, subtitles=True, audio_start=0.5, origin=0, container_format="matroska"):
    av = pytest.importorskip("av")
    with av.open(str(path), "w", format=container_format) as output:
        video = output.add_stream("mpeg4", rate=4)
        video.width, video.height, video.pix_fmt = 16, 16, "yuv420p"
        sound = output.add_stream("pcm_s16le", rate=8000) if audio else None
        if sound:
            sound.layout = "mono"
        packets = []
        if subtitles:
            with av.open(BytesIO(b"1\n00:00:00,500 --> 00:00:01,500\ncrossing caption\n"), format="srt") as cues:
                captions = output.add_stream_from_template(cues.streams[0])
                for packet in cues.demux():
                    if packet.pts is not None:
                        offset = int(origin / packet.time_base)
                        packet.pts += offset
                        packet.dts += offset
                        packet.stream = captions
                        packets.append(packet)
        for index in range(8):
            frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), index * 20, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = origin * 4 + index, Fraction(1, 4)
            packets.extend(video.encode(frame))
        packets.extend(video.encode(None))
        if sound:
            frame = av.AudioFrame.from_ndarray(np.full((1, 8000), 2000, dtype=np.int16), format="s16", layout="mono")
            frame.sample_rate, frame.time_base, frame.pts = 8000, Fraction(1, 8000), int((origin + audio_start) * 8000)
            packets.extend(sound.encode(frame))
            packets.extend(sound.encode(None))
        for packet in sorted(packets, key=lambda p: p.dts * p.time_base):
            output.mux(packet)
    return str(path)


@pytest.mark.parametrize("sample_rate", [8000, 16000, 48000])
@pytest.mark.parametrize("origin", [0, 10])
def test_container_probe_and_windowed_visual_audio_subtitles(tmp_path, sample_rate, origin):
    uri = make_media(tmp_path / "timed.mkv", origin=origin)
    source = VaneVideoSource(frames_per_clip=2, sample_rate=sample_rate)
    media = source.probe(uri)
    assert media.duration == pytest.approx(2)
    assert media.origin == pytest.approx(origin)
    assert media.available == frozenset({"visual", "audio", "subtitles", "transcription", "ocr"})
    for start in (0, 1):
        window = ClipWindow(start, start + 1)
        frames = source.extract(uri, "visual", window, media)
        assert len(frames) == 1
        assert len(frames[0].value.frames) == 2
        assert all(start <= t < start + 1 for t in frames[0].value.frame_times)
        audio = source.extract(uri, "audio", window, media)[0]
        assert audio.start == start and audio.end == start + 1
        assert audio.value.samples.shape == (sample_rate, 1)
        # Delayed audio must not be shifted to the start of its clip.
        if start == 0:
            assert np.all(audio.value.samples[: sample_rate // 2] == 0)
            assert np.all(audio.value.samples[sample_rate // 2 :] > 0)
        else:
            assert np.all(audio.value.samples[: sample_rate // 2] > 0)
            assert np.all(audio.value.samples[sample_rate // 2 :] == 0)
        captions = source.extract(uri, "subtitles", window, media)
        assert captions == [TimedContent(max(start, 0.5), min(start + 1, 1.5), "crossing caption")]


def test_mp4_duration_and_frames_rebase_to_common_origin(tmp_path):
    uri = make_media(tmp_path / "shifted.mp4", audio=False, subtitles=False, origin=10, container_format="mp4")
    source = VaneVideoSource(frames_per_clip=2)
    media = source.probe(uri)
    assert media.origin == pytest.approx(10)
    assert media.duration == pytest.approx(2)
    frames = source.extract(uri, "visual", ClipWindow(1, 2), media)
    assert frames[0].value.frame_times == (1, 1.5)


def test_container_missing_tracks_and_speech_extractor_contract(tmp_path):
    uri = make_media(tmp_path / "silent.mkv", audio=False, subtitles=False)
    source = VaneVideoSource()
    media = source.probe(uri)
    assert media.available == frozenset({"visual", "ocr"})
    assert "transcription" not in source.supported_modalities
    with pytest.raises(ValueError, match="extractor"):
        source.extract(uri, "transcription", ClipWindow(0, 1), media)


def test_external_subtitles_are_discovered_and_keep_absolute_timestamps(tmp_path):
    uri = make_media(tmp_path / "external.mkv", audio=False, subtitles=False)
    source = VaneVideoSource(extractors={"subtitles": lambda *_: [TimedContent(0.5, 1.5, "external")]})
    media = source.probe(uri)
    assert "subtitles" in media.available
    assert source.extract(uri, "subtitles", ClipWindow(1, 2), media) == [TimedContent(0.5, 1.5, "external")]


def test_corrupt_media_is_not_normal_absence(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "bad.mp4"
    path.write_bytes(b"broken container")
    with pytest.raises(av.error.InvalidDataError):
        VaneVideoSource().probe(str(path))


def test_decode_limits_are_errors(tmp_path):
    uri = make_media(tmp_path / "bounded.mkv")
    source = VaneVideoSource(frames_per_clip=2, sample_rate=8000, max_clip_bytes=100)
    media = source.probe(uri)
    for modality in ("visual", "audio"):
        with pytest.raises(ValueError, match="max_clip_bytes"):
            source.extract(uri, modality, ClipWindow(0, 1), media)

# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in CUDA speech/silence controls with a provisioned local checkpoint."""

import os
from pathlib import Path

import numpy as np
import pytest

from vane.ai import AudioClip
from vane.ai.media import MediaWindow, TimedContent
from vane.ai.media_models import FasterWhisperASR

pytestmark = [pytest.mark.external_service, pytest.mark.gpu]


@pytest.fixture(scope="module")
def whisper():
    if os.environ.get("VANE_RUN_FASTER_WHISPER_TESTS") != "1":
        pytest.skip("Enable the provisioned Faster-Whisper GPU controls explicitly")
    model = Path(os.environ["VANE_FASTER_WHISPER_MODEL"])
    assert model.is_absolute() and model.is_dir()
    adapter = FasterWhisperASR(str(model), device="cuda", compute_type="int8_float16")
    try:
        yield adapter
    finally:
        adapter.close()


@pytest.fixture(scope="module")
def speech(whisper):
    import soundfile as sf

    path = Path(os.environ["VANE_FASTER_WHISPER_SPEECH_WAV"])
    assert path.is_absolute() and path.is_file()
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    assert rate == 16000 and samples.shape[1] == 1 and 0 < len(samples) <= 3 * rate and samples.any()
    expected = os.environ["VANE_FASTER_WHISPER_EXPECTED_TEXT"].strip().casefold()
    assert expected
    return samples, rate, expected


@pytest.mark.parametrize("seconds", [0.032, 1.0, 4.0, 4.05, 4.8])
def test_default_vad_returns_no_speech_for_exact_silence(whisper, seconds):
    samples = np.zeros((round(seconds * 16000), 1), np.float32)
    window = MediaWindow(10, 10 + seconds)
    assert whisper(window, [TimedContent(window.start, window.end, AudioClip(samples, 16000))]) == []
    assert not samples.any()


def test_vad_preserves_short_speech_and_restores_padded_timeline(whisper, speech):
    samples, rate, expected = speech
    before = samples.copy()
    duration = len(samples) / rate
    window = MediaWindow(10, 10 + duration)
    result = whisper(window, [TimedContent(window.start, window.end, AudioClip(samples, rate))])
    assert expected in " ".join(row.value for row in result).casefold()
    assert all(window.start <= row.start < row.end <= window.end for row in result)
    assert np.array_equal(samples, before)

    # A long leading gap must not disappear from the returned asset times.
    leading, trailing = 3, 3
    padded = np.concatenate(
        [np.zeros((leading * rate, 1), np.float32), samples, np.zeros((trailing * rate, 1), np.float32)]
    )
    padded_before = padded.copy()
    window = MediaWindow(20, 20 + len(padded) / rate)
    result = whisper(window, [TimedContent(window.start, window.end, AudioClip(padded, rate))])
    assert expected in " ".join(row.value for row in result).casefold()
    assert all(20 + leading - 0.5 <= row.start < row.end <= 20 + leading + duration + 0.5 for row in result)
    assert np.array_equal(padded, padded_before)

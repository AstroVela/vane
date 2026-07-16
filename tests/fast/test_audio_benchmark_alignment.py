from __future__ import annotations

import pytest


def test_audio_resample_decode_failure_is_not_silently_skipped(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    pytest.importorskip("transformers")
    pa = pytest.importorskip("pyarrow")

    from multimodal_inference_benchmarks.audio_transcription import vane_main

    def fail_decode(_payload: bytes):
        raise ValueError("bad audio")

    monkeypatch.setattr(vane_main, "_decode_audio_bytes", fail_decode)

    with pytest.raises(ValueError, match="bad audio"):
        vane_main._stream_resample(pa.table({"id": [1], "audio_bytes": [b"bad"]}))

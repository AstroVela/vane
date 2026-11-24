# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Local SDK and cloud protocol contracts without weights or paid requests."""

import json
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from vane.ai import media_models as models
from vane.ai._audio_embedding import AudioClip
from vane.ai._video_embedding import VideoClip
from vane.ai.media import MediaWindow, TimedContent


def audio(rate=16000):
    return [TimedContent(10, 11, AudioClip(samples=np.zeros((rate, 1), np.float32), sample_rate=rate))]


def video():
    rgb = np.zeros((32, 32, 3), np.uint8)
    rgb[:, :, 0] = 255
    return [TimedContent(10, 11, VideoClip((rgb, rgb), (10.1, 10.5), (0, 1)))]


@pytest.fixture
def pillow():
    pytest.importorskip("PIL.Image", reason="Install the video extra to test image encoding")


def test_asr_rebases_times_resamples_and_caches_worker_model(monkeypatch, tmp_path):
    pytest.importorskip("av", reason="Install the video extra to test audio resampling")
    calls = []
    for name in ("model.bin", "config.json", "tokenizer.json"):
        (tmp_path / name).write_bytes(b"fixture")

    class Whisper:
        def __init__(self, path, **kwargs):
            calls.append(kwargs)

        def transcribe(self, samples, **kwargs):
            assert samples.shape == (16000,)
            assert kwargs["temperature"] == 0.0
            assert kwargs["vad_filter"] is True
            assert kwargs["vad_parameters"] == {"speech_pad_ms": 800}
            return iter([SimpleNamespace(start=0.2, end=0.6, text="你好")]), None

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Whisper))
    adapter = models.FasterWhisperASR(str(tmp_path))
    assert adapter._runtime is None
    assert adapter(MediaWindow(10, 11), audio(48000)) == [TimedContent(10.2, 10.6, "你好")]
    adapter(MediaWindow(10, 11), audio())
    assert calls == [{"device": "cpu", "compute_type": "int8", "local_files_only": True}]
    adapter.close()
    assert adapter._runtime is None


@pytest.mark.parametrize("vad_filter, padding", [(True, 0), (True, 1200), (False, 400)])
def test_asr_speech_detection_is_explicit_and_silence_can_be_empty(monkeypatch, tmp_path, vad_filter, padding):
    for name in ("model.bin", "config.json", "tokenizer.json"):
        (tmp_path / name).write_bytes(b"fixture")
    calls = []

    class Whisper:
        def __init__(self, path, **kwargs):
            calls.append("load")

        def transcribe(self, samples, **kwargs):
            assert kwargs["vad_filter"] is vad_filter
            assert kwargs["vad_parameters"] == {"speech_pad_ms": padding}
            if kwargs["vad_filter"] and not samples.any():
                return iter(()), None
            return iter([SimpleNamespace(start=0.1, end=0.8, text="speech")]), None

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Whisper))
    adapter = models.FasterWhisperASR(str(tmp_path), vad_filter=vad_filter, vad_speech_pad_ms=padding)
    content = audio()
    expected = [] if vad_filter else [TimedContent(10.1, 10.8, "speech")]
    assert adapter(MediaWindow(10, 11), content) == expected
    assert not content[0].value.samples.any()
    spoken = [TimedContent(20, 21, AudioClip(samples=np.full((16000, 1), 0.2, np.float32), sample_rate=16000))]
    assert adapter(MediaWindow(20, 21), spoken) == [TimedContent(20.1, 20.8, "speech")]
    assert calls == ["load"]


@pytest.mark.parametrize("invalid", [None, 0, 1, "false", np.bool_(True)])
def test_asr_rejects_nonboolean_vad_before_loading_sdk(tmp_path, invalid):
    with pytest.raises(ValueError, match="vad_filter must be a bool"):
        models.FasterWhisperASR(str(tmp_path), vad_filter=invalid)


@pytest.mark.parametrize("invalid", [None, -1, 1.0, True, "800"])
def test_asr_rejects_invalid_speech_padding_before_loading_sdk(tmp_path, invalid):
    with pytest.raises(ValueError, match="vad_speech_pad_ms must be a nonnegative int"):
        models.FasterWhisperASR(str(tmp_path), vad_speech_pad_ms=invalid)


@pytest.mark.parametrize("error", [MemoryError("VAD resource failure"), RuntimeError("VAD execution failure")])
def test_asr_vad_errors_remain_errors(monkeypatch, tmp_path, error):
    for name in ("model.bin", "config.json", "tokenizer.json"):
        (tmp_path / name).write_bytes(b"fixture")

    class Whisper:
        def __init__(self, path, **kwargs):
            pass

        def transcribe(self, samples, **kwargs):
            assert kwargs["vad_filter"] is True
            raise error

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Whisper))
    with pytest.raises(type(error), match="VAD"):
        models.FasterWhisperASR(str(tmp_path))(MediaWindow(10, 11), audio())


def test_rapidocr_uses_local_models_and_rgb_conversion(monkeypatch, tmp_path):
    paths = [tmp_path / name for name in ("det.onnx", "rec.onnx", "keys.txt")]
    for path in paths:
        path.write_bytes(b"fixture")
    calls = []

    class OCR:
        def __init__(self, *args):
            calls.append(args)

        def __call__(self, pixels):
            assert pixels[0, 0].tolist() == [0, 0, 255]
            return ("字幕", "")

    monkeypatch.setattr(models, "_RapidRuntime", OCR)
    adapter = models.RapidOCRExtractor(*map(str, paths))
    assert adapter(MediaWindow(10, 11), video()) == [TimedContent(10.1, 10.1, "字幕"), TimedContent(10.5, 10.5, "字幕")]
    assert len(calls) == 1
    assert calls[0] == tuple(map(str, paths))


def test_deepgram_uploads_pcm_and_preserves_word_times(monkeypatch):
    calls = []
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-secret")

    def post(url, data, headers, timeout):
        calls.append((url, data, headers))
        return {
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": "Hello.",
                                "words": [{"start": 0.1, "end": 0.8, "word": "hello", "punctuated_word": "Hello."}],
                            }
                        ]
                    }
                ]
            }
        }

    monkeypatch.setattr(models, "_post", post)
    adapter = models.DeepgramASR()
    assert "test-secret" not in repr(adapter) and b"test-secret" not in pickle.dumps(adapter)
    assert adapter(MediaWindow(10, 11), audio()) == [TimedContent(10.1, 10.8, "Hello.")]
    assert len(calls) == 1 and len(calls[0][1]) == 32000
    assert "sample_rate=16000" in calls[0][0]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"results": {"channels": []}},
        {"results": {"channels": [{"alternatives": [{"words": [], "transcript": "missing times"}]}]}},
    ],
)
def test_invalid_asr_is_an_error_not_absence(monkeypatch, payload):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test")
    monkeypatch.setattr(models, "_post", lambda *args: payload)
    with pytest.raises(ValueError):
        models.DeepgramASR()(MediaWindow(10, 11), audio())


@pytest.mark.parametrize("start,end", [(0, 5), (-1, 0), (0.8, 0.1), (0, float("nan")), (True, 0.1)])
def test_asr_rejects_invalid_timeline(start, end):
    with pytest.raises(ValueError):
        models._segment(audio()[0], start, end, "text")


def test_google_ocr_empty_success_and_provider_error_differ(monkeypatch, pillow):
    monkeypatch.setenv("GOOGLE_VISION_API_KEY", "test")
    responses = iter([{"responses": [{}]}, {"responses": [{"textAnnotations": [{"description": "visible"}]}]}])
    monkeypatch.setattr(models, "_post", lambda *args: next(responses))
    adapter = models.GoogleVisionOCR()
    assert adapter(MediaWindow(10, 11), video()) == [TimedContent(10.5, 10.5, "visible")]
    monkeypatch.setattr(models, "_post", lambda *args: {"responses": [{"error": {"code": 503}}]})
    with pytest.raises(ValueError, match="failed"):
        adapter(MediaWindow(10, 11), video())


def test_cohere_sends_one_joint_input_with_timestamps_and_no_truncation(monkeypatch, pillow):
    monkeypatch.setenv("COHERE_API_KEY", "test")
    calls = []

    def post(url, data, headers, timeout):
        calls.append(json.loads(data))
        return {"embeddings": {"float": [[0.0] * 256]}}

    monkeypatch.setattr(models, "_post", post)
    adapter = models.CohereJointEmbedder(dimensions=256)
    assert adapter.input_modalities == {"video", "text"}
    selected = {"visual": video(), "transcription": [TimedContent(10.2, 10.6, "speech")]}
    assert (
        len(
            adapter.embed_joint(
                selected["visual"][0].value, json.dumps({"text": [{"text": "hello", "start_sec": 10.2}]})
            )
        )
        == 256
    )
    assert len(calls) == 1 and len(calls[0]["inputs"]) == 1
    assert calls[0]["truncate"] == "NONE"
    content = calls[0]["inputs"][0]["content"]
    assert sum(part["type"] == "image_url" for part in content) == 2
    assert "10.2" in content[0]["text"] and content[1]["text"] == "frame_time_sec=10.1"


def test_cohere_rejects_oversize_before_network(monkeypatch, pillow):
    monkeypatch.setattr(models, "_post", lambda *args: pytest.fail("must not send"))
    with pytest.raises(ValueError, match="max_request_bytes"):
        models.CohereJointEmbedder(max_request_bytes=1).embed_joint(video()[0].value, "text")


def test_qwen_joint_adapter_keeps_absolute_times_and_closes_runtime(tmp_path, monkeypatch, pillow):
    import vane.ai.providers.qwen3_vl as qwen

    calls = []

    class Encoder:
        def __init__(self, path):
            calls.append(path)

        def embed_joint(self, clip, text):
            assert clip.frame_times == (10.1, 10.5)
            assert clip.frames[0].shape == (224, 224, 3)
            assert json.loads(text)["text"][0]["text"] == "hello"
            return np.ones(2048, np.float32)

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(qwen, "Qwen3VLEmbedder", Encoder)
    model_path = str(tmp_path / "qwen")
    adapter = models.Qwen3VLJointEmbedder(model_path)
    selected = {"visual": video(), "ocr": [TimedContent(10.1, 10.1, "hello")]}
    adapter.embed_joint(selected["visual"][0].value, json.dumps({"text": [{"text": "hello", "start_sec": 10.2}]}))
    adapter.embed_joint(selected["visual"][0].value, json.dumps({"text": [{"text": "hello", "start_sec": 10.2}]}))
    adapter.close()
    assert calls == [model_path, "closed"]


def test_http_errors_are_bounded_and_not_retried(monkeypatch):
    from urllib.error import HTTPError

    calls = []

    def open_request(*args, **kwargs):
        calls.append(1)
        raise HTTPError("https://example.com/private", 503, "secret", {}, io.BytesIO(b"private"))

    import io

    monkeypatch.setattr(models, "build_opener", lambda *args: SimpleNamespace(open=open_request))
    with pytest.raises(RuntimeError, match="status 503") as caught:
        models._post("https://example.com", b"{}", {}, 1)
    assert "private" not in str(caught.value) and "secret" not in str(caught.value)
    assert calls == [1]


def test_cloud_credentials_are_resolved_on_workers(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    adapter = models.DeepgramASR()
    with pytest.raises(ValueError, match="credential"):
        adapter(MediaWindow(10, 11), audio())


def test_rapidocr_sdk_uses_only_detection_recognition_and_propagates_errors(monkeypatch, tmp_path):
    pytest.importorskip("rapidocr")
    from rapidocr import ch_ppocr_det, ch_ppocr_rec

    initialized = []

    class Detector:
        def __init__(self, config):
            assert config.model_path == str(tmp_path / "det.onnx")
            assert config.engine_cfg.use_cuda is False
            initialized.append("det")

        def __call__(self, pixels):
            return SimpleNamespace(boxes=np.array([[[0, 0], [24, 0], [24, 24], [0, 24]]], np.float32))

    class Recognizer:
        def __init__(self, config):
            assert config.rec_keys_path == str(tmp_path / "keys.txt")
            initialized.append("rec")

        def __call__(self, request):
            assert len(request.img) == 1
            return SimpleNamespace(txts=("recognized",), scores=(0.9,))

    monkeypatch.setattr(ch_ppocr_det, "TextDetector", Detector)
    monkeypatch.setattr(ch_ppocr_rec, "TextRecognizer", Recognizer)
    runtime = models._RapidRuntime(str(tmp_path / "det.onnx"), str(tmp_path / "rec.onnx"), str(tmp_path / "keys.txt"))
    assert runtime(np.zeros((32, 32, 3), np.uint8)) == ("recognized",)
    assert initialized == ["det", "rec"]
    runtime.recognizer = lambda request: SimpleNamespace(txts=None, scores=None)
    with pytest.raises(ValueError, match="incomplete"):
        runtime(np.zeros((32, 32, 3), np.uint8))
    runtime.detector = lambda pixels: SimpleNamespace(boxes=None)
    assert runtime(np.zeros((32, 32, 3), np.uint8)) == ()


def test_incomplete_local_whisper_cannot_download_tokenizer(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=lambda *a, **k: pytest.fail("must reject before SDK construction")),
    )
    with pytest.raises(ValueError, match="tokenizer.json"):
        models.FasterWhisperASR(str(tmp_path))(MediaWindow(10, 11), audio())


def test_cohere_query_uses_the_paired_retrieval_role(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "fixture")
    calls = []

    def post(url, data, headers, timeout):
        calls.append(json.loads(data))
        return {"embeddings": {"float": [[1.0] * 256]}}

    monkeypatch.setattr(models, "_post", post)
    model = models.CohereJointEmbedder(dimensions=256)
    assert len(model.embed_query("search")) == 256
    assert calls[0]["input_type"] == "search_query"

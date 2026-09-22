# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import json
import pickle
from dataclasses import replace

import pyarrow as pa
import pytest

import vane
from vane.ai import Transcription, TranscriptionInput, TranscriptionSegment, transcribe
from vane.ai._transcription import _prepare_options, _TranscribeBatch
from vane.ai._transcription_types import validate_transcription
from vane.ai.protocols import TranscriberDescriptor
from vane.ai.provider import load_provider
from vane.ai.typing import UDFOptions
from vane.execution._async_runtime import AsyncRuntime


def transcript():
    return Transcription(
        " A bicycle. A car.",
        "english",
        3.0,
        (
            TranscriptionSegment(0.25, 1.5, " A bicycle."),
            TranscriptionSegment(1.5, 2.8, " A car."),
        ),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("duration", float("inf")),
        ("duration", float("nan")),
        ("duration", True),
        ("duration", 0),
        ("text", None),
        ("text", "a" * 100_001),
        ("language", ""),
        ("language", 3),
        ("segments", None),
        ("segments", ()),
        ("segments", (None,)),
    ],
)
def test_invalid_transcripts_are_rejected(field, value):
    with pytest.raises(ValueError):
        validate_transcription(replace(transcript(), **{field: value}))


@pytest.mark.parametrize(
    "segment",
    [
        TranscriptionSegment(-1, 1, " A bicycle."),
        TranscriptionSegment(1, 1, " A bicycle."),
        TranscriptionSegment(0, 4, " A bicycle."),
        TranscriptionSegment(True, 1, " A bicycle."),
        TranscriptionSegment(0, float("nan"), " A bicycle."),
        TranscriptionSegment(0, 1, ""),
        TranscriptionSegment(0, 1, None),
        TranscriptionSegment(2, 2.5, " A bicycle."),
    ],
)
def test_invalid_segment_values_are_rejected(segment):
    with pytest.raises(ValueError):
        validate_transcription(replace(transcript(), segments=(segment, transcript().segments[1])))


def test_unicode_silence_and_overlapping_segments_are_preserved():
    value = Transcription(
        "你好。 世界。",
        "chinese",
        4.0,
        (
            TranscriptionSegment(0.25, 2.5, "你好。"),
            TranscriptionSegment(2, 3.5, " 世界。"),
        ),
    )
    assert validate_transcription(value) is value
    assert validate_transcription(replace(value, text="你好。世界。"))
    silent = Transcription("", None, 3.0, ())
    assert validate_transcription(silent) is silent
    with pytest.raises(ValueError):
        validate_transcription(replace(value, segments=value.segments[::-1]))


@pytest.mark.parametrize(
    "options",
    [
        {"batch_size": 0},
        {"batch_size": True},
        {"max_concurrency_per_actor": 0},
        {"max_retries": -1},
        {"execution_backend": "invalid"},
        {"execution_backend": "ray_task", "actor_number": 2},
        {"unknown": True},
    ],
)
def test_execution_controls_are_closed_and_validated(options):
    with pytest.raises((ValueError, TypeError)):
        _prepare_options(options, "raise")


@pytest.mark.parametrize(
    "options", [{"language": "english"}, {"timeout": float("nan")}, {"prompt": 4}, {"temperature": 0}]
)
def test_provider_rejects_unsupported_request_options(options):
    with pytest.raises((ValueError, TypeError)):
        load_provider("openai", api_key="test").get_transcriber("whisper-1", options=options)


def test_provider_capability_validation_and_client_snapshot(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "application-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9999/v1")
    provider = load_provider("openai")
    descriptor = provider.get_transcriber("whisper-1")
    assert descriptor.get_provider() == "openai" and descriptor.get_model() == "whisper-1"
    assert "audio/wav" in descriptor.supported_media_mime_types()
    assert "application-test-key" not in repr(descriptor)
    clone = pickle.loads(pickle.dumps(descriptor))
    assert clone.get_options() == descriptor.get_options()
    with pytest.raises(ValueError, match="segment timestamps"):
        provider.get_transcriber("gpt-4o-transcribe")
    with pytest.raises(NotImplementedError, match="transcribe"):
        load_provider("google").get_transcriber()
    with pytest.raises(TypeError, match="Expression"):
        transcribe("audio.wav")


class StubDescriptor(TranscriberDescriptor):
    def get_provider(self):
        return "test"

    def get_model(self):
        return "test-asr"

    def get_options(self):
        return {}

    def supported_media_mime_types(self):
        return frozenset({"audio/wav"})

    def instantiate(self):
        return StubTranscriber()


class StubTranscriber:
    async def transcribe(self, audio):
        if audio.data == b"fail":
            raise ValueError("opaque-credential-private-recording")
        if audio.data == b"invalid":
            return replace(transcript(), duration=-1)
        return transcript()

    async def aclose(self):
        pass


def packed(data):
    return {"message_0": {"data": data, "content_type": "audio/wav", "error": None}}


def test_null_batch_and_serialization_never_initialize_clients():
    wrapper = _TranscribeBatch(StubDescriptor(), UDFOptions())
    table = pa.table(
        {
            "audio": pa.array(
                [None, {"message_0": None}],
                type=pa.struct(
                    [
                        (
                            "message_0",
                            pa.struct([("data", pa.binary()), ("content_type", pa.string()), ("error", pa.string())]),
                        )
                    ]
                ),
            )
        }
    )
    assert wrapper(table)["transcription"].to_pylist() == [None, None]
    assert wrapper._client is None
    assert pickle.loads(pickle.dumps(wrapper))._run_async is None
    assert b"private" not in repr(TranscriptionInput(b"private", "audio/wav")).encode()


@pytest.mark.parametrize("on_error", ["raise", "ignore"])
def test_row_errors_nulls_and_results_preserve_order(on_error, caplog):
    runtime = AsyncRuntime()
    wrapper = _TranscribeBatch(
        StubDescriptor(), UDFOptions(on_error=on_error, max_retries=0, max_concurrency_per_actor=2)
    )
    wrapper.bind_async_runtime(runtime.run)
    table = pa.table({"audio": [packed(b"ok"), None, packed(b"fail"), packed(b"invalid")]})
    try:
        if on_error == "raise":
            with pytest.raises(RuntimeError, match="Transcribe execution") as error:
                wrapper(table)
            assert error.value.__context__ is None
        else:
            values = wrapper(table)["transcription"].to_pylist()
            assert json.loads(values[0])["segments"][1]["end"] == 2.8 and values[1:] == [None, None, None]
            assert "substituted NULL" in caplog.text and "opaque-credential" not in caplog.text
    finally:
        wrapper.close()
        runtime.close()


def test_native_null_transport_remains_typed_without_loading_models(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as conn:
        source = conn.from_arrow(pa.table({"audio": pa.array([None, None], type=pa.string())}))
        result = transcribe(source, vane.col("audio"), execution_backend="subprocess_task", max_retries=0)
        assert result.select("transcription").fetchall() == [(None,), (None,)]
        assert "segments" in str(result.types[-1])

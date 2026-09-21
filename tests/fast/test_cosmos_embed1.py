# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Model invocation contract using small in-memory SDK substitutes."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._video_embedding import VideoClip
from vane.ai.providers._cosmos_embed1 import COSMOS_MODEL, CosmosTextEmbedderDescriptor, CosmosVideoEmbedderDescriptor


@pytest.fixture
def sdk(monkeypatch):
    state = SimpleNamespace(loads=[], calls=[], inference=False, cuda=True, offline=True, max_len=128, dim=256)

    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values)

        def to(self, device, *, dtype):
            assert device == "cuda" and dtype in {"float32", "float16"}
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values.astype(np.float32)

    class Inputs(dict):
        def to(self, device, *, dtype):
            assert state.inference
            return self

    class Tokenizer:
        def encode(self, text, **options):
            assert options == {"add_special_tokens": True, "truncation": False}
            return text.split() + ["CLS", "SEP"]

    class Processor:
        num_video_frames = 8
        resolution = 224
        tokenizer = Tokenizer()

        @property
        def max_txt_len(self):
            return state.max_len

        @classmethod
        def from_pretrained(cls, name, **options):
            state.loads.append(("processor", name, options))
            return cls()

        def __call__(self, *, videos=None, text=None):
            if text is not None:
                return Inputs(text=text)
            assert videos.dtype == np.uint8
            assert videos.shape[:3] == (1, 8, 3)
            state.calls.append(videos.copy())
            # Small stand-in for resizing each clip to the model resolution.
            return Inputs(videos=Tensor(videos[:, :, :, :1, :1]))

    class Model:
        @classmethod
        def from_pretrained(cls, name, **options):
            state.loads.append(("model", name, options))
            return cls()

        @property
        def config(self):
            return SimpleNamespace(
                embed_dim=state.dim, num_video_frames=8, resolution=224, transformer_engine=False, use_fp8=False
            )

        def to(self, device, *, dtype):
            assert device == "cuda"
            state.dtype = dtype
            return self

        def eval(self):
            state.evaluating = True
            return self

        def get_video_embeddings(self, *, videos):
            assert state.inference and state.evaluating
            values = videos.values
            assert values.shape[1:] == (8, 3, 1, 1)
            return SimpleNamespace(visual_proj=Tensor(np.repeat(values[:, -1, 0, 0, 0, None], 256, axis=1)))

        def get_text_embeddings(self, *, text):
            assert state.inference and state.evaluating
            return SimpleNamespace(text_proj=Tensor(np.ones((len(text), 256))))

    @contextmanager
    def inference_mode():
        state.inference = True
        try:
            yield
        finally:
            state.inference = False

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: state.cuda),
            float32="float32",
            float16="float16",
            inference_mode=inference_mode,
            cat=lambda values, dim: Tensor(np.concatenate([value.values for value in values], axis=dim)),
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=Model, AutoProcessor=Processor))
    monkeypatch.setitem(sys.modules, "transformers.utils", SimpleNamespace(is_offline_mode=lambda: state.offline))
    return state


def descriptor(kind=CosmosVideoEmbedderDescriptor, **overrides):
    return kind(
        COSMOS_MODEL,
        options={
            "trust_remote_code": True,
            "revision": "a" * 40,
            "dtype": "float16",
            "device": "cuda",
            "local_files_only": True,
            "cache_folder": "/model-cache",
            **overrides,
        },
    )


def make_clip(height, base):
    return VideoClip(
        tuple(np.full((height, 4, 3), base + i, dtype=np.uint8) for i in range(8)),
        tuple(i / 2 for i in range(8)),
        tuple(range(8)),
    )


@pytest.mark.parametrize("precision", ["float16", "float32"])
def test_model_revision_precision_and_ordered_frames_reach_worker(sdk, precision):
    embedder = descriptor(dtype=precision).instantiate()
    expected_loading = dict(revision="a" * 40, trust_remote_code=True, local_files_only=True, cache_dir="/model-cache")
    assert sdk.loads == [
        ("processor", COSMOS_MODEL, expected_loading),
        ("model", COSMOS_MODEL, {**expected_loading, "torch_dtype": precision}),
    ]
    clips = [make_clip(2, 0), make_clip(5, 10)]
    outputs = embedder.embed_video(clips)
    assert len(outputs) == 2 and all(vector.shape == (256,) and vector.dtype == np.float32 for vector in outputs)
    assert outputs[0][0] == 7 and outputs[1][0] == 17
    np.testing.assert_array_equal(sdk.calls[0][0, :, 0, 0, 0], np.arange(8))
    embedder.embed_video(clips)
    assert len(sdk.loads) == 2
    assert sdk.dtype == precision


def test_long_text_cannot_be_silently_truncated(sdk):
    embedder = descriptor(CosmosTextEmbedderDescriptor).instantiate()
    assert len(embedder.embed_text(["word " * 126])) == 1
    with pytest.raises(EmbeddingConfigurationError, match="128 tokens"):
        embedder.embed_text(["word " * 127])


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("cuda", False, "requires CUDA"),
        ("offline", False, "HF_HUB_OFFLINE"),
        ("max_len", 64, "processor"),
        ("dim", 128, "model"),
    ],
)
def test_worker_rejects_unsupported_runtime_before_inference(sdk, field, value, match):
    setattr(sdk, field, value)
    with pytest.raises(EmbeddingConfigurationError, match=match):
        descriptor().instantiate()
    assert not sdk.calls


def test_unequal_frame_sizes_are_not_padded_or_resampled(sdk):
    embedder = descriptor().instantiate()
    value = make_clip(2, 0)
    value = VideoClip((*value.frames[:-1], np.zeros((3, 4, 3), dtype=np.uint8)), value.frame_times, value.frame_indices)
    with pytest.raises(EmbeddingConfigurationError, match="same shape"):
        embedder.embed_video([value])

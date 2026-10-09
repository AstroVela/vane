# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._video_embedding import VideoClip
from vane.ai.functions import _prepare_embed_call
from vane.ai.providers.transformers import TransformersProvider

MODEL = "Qwen/Qwen3-VL-Embedding-2B"
OPTIONS = {"device": "cpu", "dtype": "float32"}


@pytest.mark.parametrize("kind", ["text", "image", "video"])
def test_qwen_descriptors_are_lazy_paired_and_serializable(kind):
    desc, dims, udf, *_ = _prepare_embed_call(
        "transformers", MODEL, 128, "ignore", OPTIONS, relation=True, input_kind=kind
    )
    restored = pickle.loads(pickle.dumps(desc))
    assert dims == restored.get_dimensions() == 128 and udf.num_gpus == 0
    assert restored.get_model() == MODEL
    if kind == "video":
        assert restored.supports_image_queries()
        assert restored.get_input_spec().max_frames == 64
    if kind == "text":
        assert not restored.supports_chunking()


@pytest.mark.parametrize(
    "options",
    [
        {"device": "auto"},
        {"dtype": "bfloat16"},
        {"max_frames": 0},
        {"max_frames": 257},
        {"max_length": True},
        {"instruction": ""},
        {"max_pixels": 0},
        {"trust_remote_code": True},
    ],
)
def test_invalid_qwen_model_configuration_fails_before_loading(options):
    with pytest.raises((TypeError, ValueError)):
        TransformersProvider().get_video_embedder(MODEL, options={**OPTIONS, **options})


@pytest.mark.parametrize("dimensions", [0, 16, True, 2049])
def test_qwen_rejects_invalid_mrl_dimensions(dimensions):
    with pytest.raises(EmbeddingConfigurationError, match="dimensions"):
        TransformersProvider().get_video_embedder(MODEL, dimensions=dimensions, options=OPTIONS)


def test_instruct_is_not_an_embedding_model():
    with pytest.raises(EmbeddingConfigurationError, match="Instruct"):
        TransformersProvider().get_video_embedder("Qwen/Qwen3-VL-2B-Instruct", options=OPTIONS)


@pytest.fixture
def sdk(monkeypatch):
    torch = pytest.importorskip("torch")
    state = SimpleNamespace(loads=[], calls=[], inferences=0, token_count=4, missing=[])

    class Inputs(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class Processor:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            state.loads.append(("processor", model, kwargs))
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            state.messages = messages
            assert kwargs == {"add_generation_prompt": True, "tokenize": False}
            return "rendered"

        def __call__(self, **kwargs):
            assert kwargs["truncation"] is False
            state.calls.append(kwargs)
            mask = torch.ones((1, state.token_count), dtype=torch.long)
            mask[0, -1] = 0
            return Inputs(input_ids=torch.zeros_like(mask), attention_mask=mask)

    class Model:
        config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=2048))

        @classmethod
        def from_pretrained(cls, model, **kwargs):
            state.loads.append(("model", model, kwargs))
            return cls(), {"missing_keys": state.missing, "mismatched_keys": []}

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, input_ids, attention_mask, use_cache):
            assert not use_cache and not torch.is_grad_enabled()
            state.inferences += 1
            return SimpleNamespace(last_hidden_state=torch.arange(input_ids.shape[1] * 2048).reshape(1, -1, 2048))

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=Model, AutoProcessor=Processor))
    monkeypatch.setitem(sys.modules, "transformers.video_utils", SimpleNamespace(VideoMetadata=SimpleNamespace))
    return state


def test_qwen_pooling_and_input_contracts(sdk):
    desc = TransformersProvider().get_video_embedder(
        MODEL, dimensions=128, options={**OPTIONS, "local_files_only": True, "max_pixels": 8192}
    )
    runtime = desc.instantiate()
    frames = tuple(np.full((8, 8, 3), i, dtype=np.uint8) for i in (10, 20, 30))
    clip = VideoClip(frames, (0.125, 0.25, 1.375), (1, 2, 11))
    text = runtime.embed_text(["find the event"])[0]
    image = runtime.embed_image([frames[0]])[0]
    video = runtime.embed_video([clip])[0]
    # Last nonpadding token, followed by MRL slicing; never the padding token.
    for vector in (text, image, video):
        np.testing.assert_array_equal(vector, np.arange(4096, 4224))
    kwargs = sdk.calls[-1]
    metadata = kwargs["videos_kwargs"]["video_metadata"][0]
    assert [i / metadata.fps for i in metadata.frames_indices] == list(clip.frame_times)
    np.testing.assert_array_equal(kwargs["videos"][0], np.stack(frames))
    assert kwargs["videos_kwargs"]["do_sample_frames"] is False
    assert kwargs["videos_kwargs"]["size"] == {"shortest_edge": 4096, "longest_edge": 8192}
    assert all(call[2]["trust_remote_code"] is False and call[2]["local_files_only"] for call in sdk.loads)


def test_qwen_overlength_and_mismatched_checkpoint_fail_explicitly(sdk):
    desc = TransformersProvider().get_text_embedder(MODEL, options={**OPTIONS, "max_length": 2})
    with pytest.raises(EmbeddingConfigurationError, match="max_length"):
        desc.instantiate().embed_text(["too long"])
    assert sdk.inferences == 0
    sdk.missing = ["model.weight"]
    with pytest.raises(EmbeddingConfigurationError, match="checkpoint"):
        desc.instantiate()


def test_real_qwen_processor_preserves_presampled_video_timestamps():
    """Exercise HF's actual kwargs routing without downloading a model or tokenizer."""
    transformers = pytest.importorskip("transformers", minversion="4.57.1")
    pytest.importorskip("torchvision")
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    from vane.ai.providers._qwen3_vl_embedding import QwenEmbedder

    class RecordingTokenizer(transformers.Qwen2TokenizerFast):
        def __call__(self, text, **kwargs):
            self.recorded_text = text
            return super().__call__(text, **kwargs)

    special = ["<unk>", "<pad>", "<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]
    tokenizer = RecordingTokenizer(
        tokenizer_object=Tokenizer(WordLevel({value: index for index, value in enumerate(special)}, unk_token="<unk>")),
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=special[2:],
    )
    processor = Qwen3VLProcessor(
        image_processor=transformers.Qwen2VLImageProcessor(),
        tokenizer=tokenizer,
        video_processor=Qwen3VLVideoProcessor(),
        chat_template="{{ '<|vision_start|><|video_pad|><|vision_end|>' }}",
    )
    runtime = QwenEmbedder.__new__(QwenEmbedder)
    runtime.descriptor = TransformersProvider().get_video_embedder(MODEL, dimensions=128, options=OPTIONS)
    runtime.options = runtime.descriptor.options
    runtime.device, runtime.torch, runtime.processor = "cpu", torch, processor
    runtime.model = lambda **inputs: SimpleNamespace(
        last_hidden_state=torch.zeros((1, inputs["input_ids"].shape[1], 2048))
    )
    frames = tuple(np.full((64, 64, 3), value, dtype=np.uint8) for value in (10, 20, 30))
    vector = runtime.embed_video([VideoClip(frames, (0.2, 0.6, 1.4), (2, 6, 14))])[0]
    assert vector.shape == (128,)
    # Qwen merges each pair of frames and renders the mean time at one decimal.
    # Losing our metadata instead silently uses a 24 fps clock starting at zero.
    assert "<0.4 seconds>" in tokenizer.recorded_text[0]
    assert "<1.4 seconds>" in tokenizer.recorded_text[0]

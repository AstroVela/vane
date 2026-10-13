# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint encoding and loading contracts, without downloaded model assets."""

from __future__ import annotations

import asyncio
import pickle
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._video_embedding import VideoClip
from vane.ai.provider import ProviderImportError, load_provider
from vane.ai.providers.qwen3_vl import CONTRACT, INSTRUCTION, Qwen3VLEmbedder, pool_last, video_payload


def test_registry_descriptors_need_no_model_sdk_or_manual_registration():
    # A fresh interpreter catches accidental eager imports hidden by test order.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins, pickle
original = builtins.__import__
def checked(name, *args, **kwargs):
    assert name.split('.')[0] not in {'torch', 'transformers', 'safetensors', 'torchvision'}
    return original(name, *args, **kwargs)
builtins.__import__ = checked
from vane.ai.provider import load_provider
provider = load_provider('qwen3_vl')
for modality in ('text', 'image', 'video'):
    descriptor = getattr(provider, 'get_' + modality + '_embedder')('/models/qwen')
    restored = pickle.loads(pickle.dumps(descriptor))
    assert restored.get_dimensions() == 2048
    assert restored.get_udf_options().num_gpus == 1
    assert restored.get_options()['add_special_tokens'] is True
""",
        ],
        check=True,
    )
    descriptor = load_provider("qwen3_vl").get_video_embedder("/models/qwen")
    assert descriptor.get_input_spec().max_frames == 64
    assert descriptor.supports_image_queries()
    assert not load_provider("qwen3_vl").get_text_embedder("/models/qwen").supports_chunking()
    options = descriptor.get_options()
    options["instruction"] = "changed"
    assert descriptor.get_options()["instruction"] == INSTRUCTION


@pytest.mark.parametrize(
    "options",
    [
        {"model": None},
        {"model": "remote/model"},
        {"dimensions": 1024},
        {"options": {"instruction": "different"}},
        {"dimensions": True},
    ],
)
def test_descriptor_rejects_changes_to_trained_encoding(options):
    with pytest.raises(EmbeddingConfigurationError):
        load_provider("qwen3_vl").get_video_embedder(**{"model": "/models/qwen", **options})


def test_missing_dependency_error_preserves_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ProviderImportError) as caught:
        Qwen3VLEmbedder("/models/qwen")
    assert caught.value.extra == "qwen3_vl"
    assert pickle.loads(pickle.dumps(caught.value)).extra == "qwen3_vl"


def clip(count=2, size=32, times=None, indices=None):
    return VideoClip(
        tuple(np.full((size, size, 3), i, dtype=np.uint8) for i in range(count)),
        times if times is not None else tuple(12.345678 + i * 0.5 for i in range(count)),
        indices if indices is not None else tuple(100 + i * 5 for i in range(count)),
    )


def test_video_preserves_frames_and_absolute_microseconds():
    source = clip()
    pixels, metadata = video_payload(source)
    np.testing.assert_array_equal(pixels, np.stack(source.frames))
    assert metadata == {"fps": 1_000_000, "frames_indices": [12345678, 12845678], "total_num_frames": 12845679}


@pytest.mark.parametrize(
    "source",
    [
        clip(count=0),
        clip(count=3),
        clip(count=66),
        clip(size=0),
        clip(size=31),
        clip(size=288),
        clip(count=64, size=256),
        clip(times=(0.0, 0.0000001)),
        clip(times=(0.0, float("nan"))),
        clip(times=(0.0, 1e300)),
        clip(indices=(1, 1)),
        clip(indices=(-1, 1)),
    ],
)
def test_video_contract_is_bounded_and_never_resamples(source):
    with pytest.raises(EmbeddingConfigurationError):
        video_payload(source)


def test_last_pooling_retains_eos_under_left_and_right_padding():
    torch = pytest.importorskip("torch")
    hidden = torch.arange(24).reshape(2, 4, 3).to(torch.bfloat16)
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])
    result = pool_last(hidden, mask)
    assert result.dtype == torch.float32
    assert torch.equal(result, torch.stack((hidden[0, 1], hidden[1, 3])).float())
    with pytest.raises(EmbeddingConfigurationError, match="nonempty"):
        pool_last(hidden, torch.zeros_like(mask))


def test_default_special_token_postprocessing_and_query_budget():
    pytest.importorskip("torch")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({"[UNK]": 0, "[PAD]": 1, "word": 2, "[EOS]": 3}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="$A [EOS]", special_tokens=[("[EOS]", 3)]
    )
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", eos_token="[EOS]")
    encoder = Qwen3VLEmbedder.__new__(Qwen3VLEmbedder)

    def template(conversations, **options):
        assert options == {"tokenize": False, "add_generation_prompt": True}
        assert all(c[0] == {"role": "system", "content": INSTRUCTION} for c in conversations)
        return [c[1]["content"] for c in conversations]

    encoder.processor = SimpleNamespace(tokenizer=fast, apply_chat_template=template)
    inputs = encoder._text_inputs(["word", "word word"])
    assert inputs["input_ids"].tolist() == [[2, 3, 1], [2, 2, 3]]
    assert inputs["attention_mask"].tolist() == [[1, 1, 0], [1, 1, 1]]
    assert encoder._text_inputs(["word " * 127])["input_ids"].shape[1] == 128
    with pytest.raises(EmbeddingConfigurationError, match="truncation"):
        encoder._text_inputs(["word " * 128])


@pytest.fixture
def real_processor():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    # An authored byte tokenizer preserves timestamps and every boundary token.
    # Exercise the real SDK without fetching checkpoint assets or model weights.
    special = [
        "[UNK]",
        "[PAD]",
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|video_pad|>",
        "<|image_pad|>",
    ]
    vocabulary = special + sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.BPE({token: i for i, token in enumerate(vocabulary)}, merges=[], unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = tokenizers.decoders.ByteLevel()
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="$A <|endoftext|>", special_tokens=[("<|endoftext|>", 2)]
    )
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=special[0],
        pad_token=special[1],
        eos_token=special[2],
        additional_special_tokens=special[3:],
        model_input_names=["input_ids", "attention_mask"],
    )
    template = """
{%- for message in messages -%}
    {{- '<|im_start|>' + message.role + '\n' -}}
    {%- if message.content is string -%}
        {{- message.content -}}
    {%- else -%}
        {%- for item in message.content -%}
            {%- if item.type == 'video' -%}
                {{- '<|vision_start|><|video_pad|><|vision_end|>' -}}
            {%- elif item.type == 'image' -%}
                {{- '<|vision_start|><|image_pad|><|vision_end|>' -}}
            {%- else -%}
                {{- item.text -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
    {{- '<|im_end|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}
"""
    return transformers.Qwen3VLProcessor(
        tokenizer=fast,
        image_processor=transformers.Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2),
        video_processor=transformers.Qwen3VLVideoProcessor(patch_size=16, temporal_patch_size=2, merge_size=2),
        chat_template=template,
    )


@pytest.mark.parametrize("counts", [(2,), (4,), (64,), (2, 4)])
@pytest.mark.parametrize("wrapped", [True, False])
@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning:transformers.processing_utils"
)
def test_real_video_processor_matches_reference_temporal_tokens(real_processor, counts, wrapped):
    torch = pytest.importorskip("torch")
    if not wrapped:
        real_processor.chat_template = real_processor.chat_template.replace(
            "<|vision_start|><|video_pad|><|vision_end|>", "<|video_pad|>"
        )
    encoder = Qwen3VLEmbedder.__new__(Qwen3VLEmbedder)
    encoder.processor = real_processor
    encoder._infer = lambda inputs: inputs
    sources = [clip(count=count, size=32 * (i + 1)) for i, count in enumerate(counts)]
    inputs = encoder.embed_video(sources)
    expected_prompts = []
    for source in sources:
        # Reference contract: a timestamp followed by one vision-boundary pair
        # per two-frame patch. There is no outer pair enclosing the whole video.
        patches = []
        spatial_tokens = (source.frames[0].shape[0] // 32) ** 2
        for i in range(0, len(source.frames), 2):
            timestamp = (source.frame_times[i] + source.frame_times[i + 1]) / 2
            patches.append(
                f"<{timestamp:.1f} seconds><|vision_start|>" + "<|video_pad|>" * spatial_tokens + "<|vision_end|>"
            )
        expected_prompts.append(
            f"<|im_start|>system\n{INSTRUCTION}<|im_end|>\n<|im_start|>user\n"
            + "".join(patches)
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
    expected = real_processor.tokenizer(expected_prompts, add_special_tokens=True, padding=True, return_tensors="pt")
    assert torch.equal(inputs["input_ids"], expected["input_ids"])
    assert torch.equal(inputs["attention_mask"], expected["attention_mask"])
    for i, count in enumerate(counts):
        ids = inputs["input_ids"][i][inputs["attention_mask"][i].bool()].tolist()
        assert ids.count(real_processor.vision_start_token_id) == count // 2
        assert ids.count(real_processor.vision_end_token_id) == count // 2
        assert ids[-1] == real_processor.tokenizer.eos_token_id
        assert inputs["video_grid_thw"][i].tolist() == [count // 2, 2 * (i + 1), 2 * (i + 1)]


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning:transformers.processing_utils"
)
def test_real_image_processor_keeps_its_single_vision_boundary_pair(real_processor):
    encoder = Qwen3VLEmbedder.__new__(Qwen3VLEmbedder)
    encoder.processor = real_processor
    encoder._infer = lambda inputs: inputs
    inputs = encoder.embed_image([np.zeros((64, 64, 3), dtype=np.uint8)])
    prompt = (
        f"<|im_start|>system\n{INSTRUCTION}<|im_end|>\n<|im_start|>user\n<|vision_start|>"
        + "<|image_pad|>" * 4
        + "<|vision_end|><|im_end|>\n<|im_start|>assistant\n"
    )
    assert inputs["input_ids"][0].tolist() == real_processor.tokenizer(prompt, add_special_tokens=True)["input_ids"]


def test_real_processor_text_keeps_literal_video_tokens(real_processor):
    encoder = Qwen3VLEmbedder.__new__(Qwen3VLEmbedder)
    encoder.processor = real_processor
    query = "Explain <|vision_start|><|video_pad|><|vision_end|>."
    inputs = encoder._text_inputs([query])
    prompt = (
        f"<|im_start|>system\n{INSTRUCTION}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    )
    assert inputs["input_ids"][0].tolist() == real_processor.tokenizer(prompt, add_special_tokens=True)["input_ids"]


@pytest.fixture
def sdk(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    # Saved tensors, not the stale config, determine the trained vocabulary.
    safetensors.save_file({"language_model.embed_tokens.weight": torch.ones(6, 2048)}, tmp_path / "model.safetensors")
    config = SimpleNamespace(
        model_type="qwen3_vl",
        text_config=SimpleNamespace(hidden_size=2048, vocab_size=4),
        vision_config=SimpleNamespace(patch_size=16, spatial_merge_size=2, temporal_patch_size=2),
    )
    state = SimpleNamespace(
        config=config, loads=[], calls=[], loading={}, hidden=None, vocab={"pad": 0, "eos": 5}, cuda=True
    )

    class Tokenizer:
        def get_vocab(self):
            return state.vocab

        def __call__(self, prompts, **options):
            assert options == {"add_special_tokens": True, "truncation": False, "padding": True, "return_tensors": "pt"}
            return {
                "input_ids": torch.tensor([[2, 5, 0]] * len(prompts)),
                "attention_mask": torch.tensor([[1, 1, 0]] * len(prompts)),
            }

    class Qwen3VLProcessor:
        tokenizer = Tokenizer()
        video_processor = SimpleNamespace(patch_size=16, merge_size=2, temporal_patch_size=2)
        video_token = "<|video_pad|>"
        vision_start_token = "<|vision_start|>"
        vision_end_token = "<|vision_end|>"

        def apply_chat_template(self, conversations, **options):
            state.calls.append((conversations, options))
            return ["prompt"] * len(conversations)

        def __call__(self, **options):
            state.calls.append(options)
            inputs = self.tokenizer(
                options["text"], add_special_tokens=True, truncation=False, padding=True, return_tensors="pt"
            )
            inputs["pixel_values_videos" if "videos" in options else "pixel_values"] = torch.ones(1)
            return inputs

    processor = Qwen3VLProcessor()
    state.processor = processor

    def load_config(path, **options):
        state.loads.append(("config", options))
        return config

    def load_processor(path, **options):
        state.loads.append(("processor", options))
        return processor

    class Model:
        @classmethod
        def from_pretrained(cls, path, **options):
            state.loads.append(("model", options))
            return cls(), state.loading

        def to(self, device):
            assert device == "cuda"
            return self

        def eval(self):
            return self

        def requires_grad_(self, value):
            assert value is False
            return self

        def __call__(self, **inputs):
            assert not torch.is_grad_enabled()
            assert inputs["use_cache"] is False and inputs["return_dict"] is True
            for key in ("pixel_values", "pixel_values_videos"):
                if key in inputs:
                    assert inputs[key].dtype == torch.bfloat16
            if state.hidden is None:
                hidden = torch.ones(*inputs["input_ids"].shape, 2048, dtype=torch.bfloat16)
                hidden[:, 1, 0] = 4  # Last real token; the padding row is deliberately different.
            else:
                hidden = state.hidden
            return SimpleNamespace(last_hidden_state=hidden)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=SimpleNamespace(from_pretrained=load_config),
            AutoProcessor=SimpleNamespace(from_pretrained=load_processor),
            Qwen3VLModel=Model,
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: state.cuda)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: state.cuda)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: state.calls.append("closed"))
    original_to = torch.Tensor.to

    def cpu_to(tensor, *args, **kwargs):
        return original_to(tensor, *(("cpu",) if args == ("cuda",) else args), **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    state.path = str(tmp_path)
    return state


def test_strict_local_loading_and_normalized_shared_modalities(sdk):
    encoder = Qwen3VLEmbedder(sdk.path)
    assert sdk.config.text_config.vocab_size == 6
    assert all(
        options["local_files_only"] is True and options["trust_remote_code"] is False for _, options in sdk.loads
    )
    model_options = sdk.loads[-1][1]
    assert model_options["use_safetensors"] is True and model_options["attn_implementation"] == "sdpa"
    vectors = [
        encoder.embed_text(["query"])[0],
        encoder.embed_video([clip()])[0],
        encoder.embed_image([np.zeros((32, 32, 3), dtype=np.uint8)])[0],
    ]
    for vector in vectors:
        assert vector.shape == (2048,) and vector.dtype == np.float32
        assert np.linalg.norm(vector) == pytest.approx(1.0)
        assert vector[0] == pytest.approx(4 * vector[1])
    video_options, image_options = [c for c in sdk.calls if isinstance(c, dict)]
    for options in (video_options, image_options):
        assert options["add_special_tokens"] is True and options["do_resize"] is False
        assert options["truncation"] is False
    assert video_options["do_sample_frames"] is False
    assert video_options["video_metadata"][0]["frames_indices"] == [12345678, 12845678]
    assert encoder.embed_text([]) == encoder.embed_video([]) == encoder.embed_image([]) == []
    asyncio.run(encoder.aclose())
    assert encoder.model is None and sdk.calls[-1] == "closed"


@pytest.mark.parametrize(
    "loading",
    [
        {"missing_keys": ["language_model.layers.0.weight"]},
        {"mismatched_keys": ["embedding"]},
        {"error_msgs": ["invalid"]},
        {"unexpected_keys": ["foreign_model.weight"]},
    ],
)
def test_incomplete_or_incompatible_checkpoint_never_runs(sdk, loading):
    sdk.loading = loading
    with pytest.raises(EmbeddingConfigurationError, match="exact trained"):
        Qwen3VLEmbedder(sdk.path)
    assert not sdk.calls


def test_unused_lm_head_is_allowed_but_untrained_token_rows_are_not(sdk):
    sdk.loading = {"unexpected_keys": ["lm_head.weight"]}
    encoder = Qwen3VLEmbedder(sdk.path)
    encoder.close()
    sdk.vocab["new"] = 6
    with pytest.raises(EmbeddingConfigurationError, match="untrained"):
        Qwen3VLEmbedder(sdk.path)


def test_cuda_and_checkpoint_shape_are_checked_before_loading_model(sdk):
    sdk.cuda = False
    with pytest.raises(EmbeddingConfigurationError, match="CUDA"):
        Qwen3VLEmbedder(sdk.path)
    assert sdk.loads == []
    sdk.cuda = True
    sdk.config.text_config.hidden_size = 1024
    with pytest.raises(EmbeddingConfigurationError, match="2048"):
        Qwen3VLEmbedder(sdk.path)
    assert all(kind != "model" for kind, _ in sdk.loads)


def test_token_and_image_budgets_precede_inference(sdk):
    torch = pytest.importorskip("torch")
    encoder = Qwen3VLEmbedder(sdk.path)
    for size in (0, 31, 288):
        with pytest.raises(EmbeddingConfigurationError):
            encoder.embed_image([np.zeros((size, size, 3), dtype=np.uint8)])
    with pytest.raises(EmbeddingConfigurationError, match="token budget"):
        encoder._infer({"attention_mask": torch.ones(1, CONTRACT["max_input_tokens"] + 1)})
    assert sdk.calls == []


@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf")])
def test_invalid_model_outputs_never_become_substitute_vectors(sdk, value):
    torch = pytest.importorskip("torch")
    encoder = Qwen3VLEmbedder(sdk.path)
    sdk.hidden = torch.full((1, 3, 2048), value)
    with pytest.raises(EmbeddingConfigurationError):
        encoder.embed_text(["query"])

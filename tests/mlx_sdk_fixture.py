# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""SDK boundary doubles; no model implementation or optional SDK imports."""

from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def media_sdk(monkeypatch, tmp_path):
    state = SimpleNamespace(
        loads=0,
        processor_loads=0,
        processor_calls=[],
        model_calls=[],
        clears=0,
        synchronizes=0,
        failures={},
        change=lambda value: value,
        output=None,
    )

    def fail(phase):
        if phase in state.failures:
            raise state.failures[phase]

    class Values:
        def __init__(self, value):
            self.value, self.evaluated = value, False

        def astype(self, dtype):
            assert dtype == np.float32
            return self

        def __array__(self, dtype=None, copy=None):
            assert self.evaluated, "output copied before evaluation"
            return np.array(self.value, dtype=dtype, copy=True)

    state.weights = {
        key: SimpleNamespace(dtype="bf16")
        for key in (
            "language_model.embed_tokens.weight",
            "language_model.embedding_projection.weight",
            "vision_tower.encoder.weight",
            "embed_vision.embedding_projection.weight",
            "audio_tower.encoder.weight",
            "embed_audio.embedding_projection.weight",
        )
    }

    class Model:
        audio_tower = embed_audio = vision_tower = embed_vision = object()

        def parameters(self):
            return state.weights

        def eval(self):
            state.eval_mode = True

        def __call__(self, **inputs):
            state.model_calls.append(inputs)
            assert "num_frames_per_video" not in inputs
            fail("forward")
            value = np.eye(len(inputs["input_ids"]), 768, dtype=np.float32)
            return SimpleNamespace(text_embeds=Values(state.output if state.output is not None else value))

    extractor = dict(
        sampling_rate=16000,
        feature_size=128,
        frame_length=320,
        hop_length=160,
        fft_length=512,
        dither=0,
        padding_side="right",
        preemphasis=0,
        input_scale_factor=1,
    )
    image = dict(
        max_soft_tokens=280,
        patch_size=16,
        pooling_kernel_size=3,
        do_convert_rgb=True,
        do_resize=True,
        do_rescale=True,
        do_normalize=False,
        rescale_factor=1 / 255,
    )
    video = image | {"max_soft_tokens": 140, "add_timestamps": False}

    class Processor:
        audio_token_id = 258881
        video_token_id = 258884
        feature_extractor = SimpleNamespace(**extractor)
        video_processor = SimpleNamespace(**video)

        def __call__(self, **kwargs):
            state.processor_calls.append(kwargs)
            fail("processor")
            assert kwargs["text_kwargs"] == {"padding": True, "truncation": False}
            assert kwargs["return_tensors"] == "np"
            if "audio" in kwargs:
                assert kwargs["audio_kwargs"] == dict(
                    sampling_rate=16000,
                    padding="longest",
                    truncation=False,
                    pad_to_multiple_of=128,
                    return_attention_mask=True,
                )
                audio = kwargs["audio"]
                assert all(
                    isinstance(row, list)
                    and len(row) == 1
                    and row[0].ndim == 1
                    and row[0].dtype == np.float32
                    and row[0].flags.c_contiguous
                    for row in audio
                )
                # Authored boundary examples, deliberately independent from the
                # production STFT/mask formula. No feature extraction is mocked.
                masks = {161: [1], 321: [1, 1], 641: [1, 1, 1, 1]}
                valid = [masks[len(row[0])] for row in audio]
                t = max(len(row) for row in valid)
                mask = np.array([row + [0] * (t - len(row)) for row in valid], dtype=bool)
                ids = np.array([[2, self.audio_token_id, 1] for _ in audio])
                encoded = dict(
                    input_ids=ids,
                    attention_mask=np.ones_like(ids),
                    input_features=np.zeros((len(audio), t, 128), dtype=np.float32),
                    input_features_mask=mask,
                )
            else:
                assert kwargs["videos_kwargs"] == dict(
                    input_data_format="channels_last",
                    do_sample_frames=False,
                    add_timestamps=False,
                    return_metadata=False,
                )
                videos = kwargs["videos"]
                assert all(row.ndim == 4 and row.shape[-1] == 3 and row.dtype == np.uint8 for row in videos)
                counts = [len(row) for row in videos]
                width = max(counts) + 2
                ids = np.array(
                    [[2] + [self.video_token_id] * count + [1] + [0] * (width - count - 2) for count in counts]
                )
                mask = np.array([[1] * (count + 2) + [0] * (width - count - 2) for count in counts])
                grid = np.array([(x, y) for y in range(3) for x in range(3)] + [(-1, -1)])
                encoded = dict(
                    input_ids=ids,
                    attention_mask=mask,
                    pixel_values_videos=np.zeros((sum(counts), 10, 768), dtype=np.float32),
                    video_position_ids=np.tile(grid, (sum(counts), 1, 1)),
                    num_frames_per_video=np.array(counts),
                )
            return state.change(encoded)

    state.model, state.processor = Model(), Processor()
    mx = ModuleType("mlx.core")
    mx.gpu, mx.float32, mx.bfloat16, mx.floating = "gpu", np.float32, "bf16", np.floating
    mx.metal = SimpleNamespace(is_available=lambda: True)
    mx.stream = lambda device: nullcontext()
    mx.array = np.asarray
    mx.issubdtype = lambda dtype, category: dtype in ("bf16", np.float32, np.float16)

    def evaluate(value):
        fail("evaluation")
        value.evaluated = True

    def clear():
        state.clears += 1
        fail("clear")

    def sync(device):
        assert device == "gpu"
        state.synchronizes += 1
        fail("synchronize")

    mx.eval, mx.clear_cache, mx.synchronize = evaluate, clear, sync
    hub = ModuleType("huggingface_hub")

    def snapshot_download(**kwargs):
        assert kwargs == dict(
            repo_id="mlx-community/embeddinggemma-2-bf16",
            revision="1a4ffddb7905d3f63486748deabe091a01fb6201",
            local_files_only=True,
        )
        return str(tmp_path)

    hub.snapshot_download = snapshot_download
    loader = ModuleType("mlx_vlm.embedding_loader")

    def load(path, **kwargs):
        assert path == tmp_path and kwargs == {"lazy": False, "strict": True}
        state.loads += 1
        fail("load")
        return state.model

    def load_processor(path, **kwargs):
        assert path == str(tmp_path) and kwargs == {"local_files_only": True, "trust_remote_code": False}
        state.processor_loads += 1
        fail("processor_load")
        return state.processor

    loader.load_embedding_model = load
    utils = ModuleType("mlx.utils")
    utils.tree_flatten = lambda value: list(value.items())
    processor_module = ModuleType("mlx_vlm.models.embedding_gemma2")
    processor_module.EmbeddingGemma2Processor = SimpleNamespace(from_pretrained=load_processor)
    for name, value in {
        "mlx": ModuleType("mlx"),
        "mlx.core": mx,
        "mlx.utils": utils,
        "mlx_vlm": ModuleType("mlx_vlm"),
        "mlx_vlm.embedding_loader": loader,
        "mlx_vlm.models": ModuleType("mlx_vlm.models"),
        "mlx_vlm.models.embedding_gemma2": processor_module,
        "huggingface_hub": hub,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr("vane.ai.providers.mlx.platform.system", lambda: "Darwin")
    monkeypatch.setattr("vane.ai.providers.mlx.platform.machine", lambda: "arm64")
    (tmp_path / "config.json").write_text(
        json.dumps(
            dict(
                model_type="embedding_gemma2",
                text_config={"embedding_dim": 768},
                dtype="bfloat16",
                audio_token_id=258881,
                video_token_id=258884,
                audio_config=dict(
                    model_type="gemma4_audio",
                    dtype="bfloat16",
                    hidden_size=1024,
                    num_hidden_layers=12,
                    output_proj_dims=1536,
                    subsampling_conv_channels=[128, 32],
                ),
                vision_config=dict(
                    model_type="gemma4_vision",
                    dtype="bfloat16",
                    patch_size=16,
                    pooling_kernel_size=3,
                    default_output_length=280,
                ),
            )
        )
    )
    (tmp_path / "processor_config.json").write_text(
        json.dumps(dict(feature_extractor=extractor, image_processor=image, video_processor=video))
    )
    state.path, state.mx = tmp_path, mx
    return state

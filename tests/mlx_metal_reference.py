# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent fixed-SDK references and real actor observations for Mac acceptance."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def prepared_metal():
    if os.environ.get("VANE_RUN_MLX_TESTS") != "1":
        pytest.skip("set VANE_RUN_MLX_TESTS=1 with Metal and the fixed checkpoint")
    assert platform.system() == "Darwin" and platform.machine() == "arm64"
    import mlx.core as mx
    from huggingface_hub import snapshot_download

    assert mx.metal.is_available(), "explicit MLX acceptance requires Metal"
    path = Path(
        snapshot_download(
            "mlx-community/embeddinggemma-2-bf16",
            revision="1a4ffddb7905d3f63486748deabe091a01fb6201",
            local_files_only=True,
        )
    )
    assert (path / "model.safetensors").is_file(), "fixed checkpoint must be complete"


@contextmanager
def sdk_model():
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_vlm.embedding_loader import load_embedding_model
    from mlx_vlm.models.embedding_gemma2 import EmbeddingGemma2Processor

    snapshot = Path(
        snapshot_download(
            "mlx-community/embeddinggemma-2-bf16",
            revision="1a4ffddb7905d3f63486748deabe091a01fb6201",
            local_files_only=True,
        )
    )
    sdk = SimpleNamespace(model=None, processor=None)
    try:
        with mx.stream(mx.gpu):
            sdk.model = load_embedding_model(snapshot, lazy=False, strict=True)
            sdk.model.eval()
            sdk.processor = EmbeddingGemma2Processor.from_pretrained(
                str(snapshot), local_files_only=True, trust_remote_code=False
            )
        yield sdk
    finally:
        mx.synchronize(mx.gpu)
        sdk.model = sdk.processor = None
        gc.collect()
        mx.clear_cache()


def vectors(values, count):
    values = np.asarray(values)
    assert values.shape == (count, 768) and values.dtype == np.float32
    assert np.isfinite(values).all()
    np.testing.assert_allclose(np.linalg.norm(values.astype(np.float64), axis=1), 1, rtol=0, atol=1e-4)
    return values


def infer(model, encoded):
    import mlx.core as mx

    inputs = {k: mx.array(np.asarray(v)) for k, v in encoded.items() if k != "num_frames_per_video"}
    with mx.stream(mx.gpu):
        output = model(**inputs).text_embeds.astype(mx.float32)
        mx.eval(output)
        raw = np.array(output, dtype=np.float32, copy=True)
    norms = np.linalg.norm(raw.astype(np.float64), axis=1, keepdims=True)
    assert np.isfinite(raw).all() and np.all(norms > 0) and np.all(np.abs(norms - 1) <= 1e-2)
    return vectors((raw / norms).astype(np.float32), len(raw))


def audio_inputs(processor, rows):
    return processor(
        audio=[[np.array(row, dtype=np.float32, copy=True)] for row in rows],
        text_kwargs={"padding": True, "truncation": False},
        audio_kwargs={
            "sampling_rate": 16000,
            "padding": "longest",
            "truncation": False,
            "pad_to_multiple_of": 128,
            "return_attention_mask": True,
        },
        return_tensors="np",
    )


def video_inputs(processor, rows):
    # An explicit CHW reference is independent from the provider's HWC path.
    return processor(
        videos=[row.transpose(0, 3, 1, 2) for row in rows],
        text_kwargs={"padding": True, "truncation": False},
        videos_kwargs={
            "input_data_format": "channels_first",
            "do_sample_frames": False,
            "add_timestamps": False,
            "return_metadata": False,
        },
        return_tensors="np",
    )


def text_vectors(model, processor, texts):
    encoded = processor.tokenizer(
        ["task: search result | query: " + t for t in texts], padding=True, truncation=False, return_tensors="np"
    )
    return infer(model, {k: encoded[k] for k in ("input_ids", "attention_mask")})


def parity(actual, expected, record_property=None, label="parity"):
    actual, expected = actual.astype(np.float64), expected.astype(np.float64)
    cosine = np.sum(actual * expected, axis=1) / (np.linalg.norm(actual, axis=1) * np.linalg.norm(expected, axis=1))
    metrics = {
        "min": float(cosine.min()),
        "median": float(np.median(cosine)),
        "max": float(cosine.max()),
        "max_absolute_difference": float(np.abs(actual - expected).max()),
    }
    if record_property:
        record_property(label, json.dumps(metrics))
    assert np.all(cosine >= 0.999), metrics
    return metrics


@pytest.fixture
def local_actor(monkeypatch):
    import psutil
    import ray

    import vane

    assert not ray.is_initialized()

    def forbidden(*args, **kwargs):
        raise AssertionError("MLX acceptance must not initialize Ray")

    monkeypatch.setattr(ray, "init", forbidden)
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    parent = psutil.Process()
    previous = {p.pid for p in parent.children(recursive=True)}
    try:
        yield
    finally:
        vane.teardown_runner()
        owned = [
            p
            for p in parent.children(recursive=True)
            if p.pid not in previous and "udf_subprocess_worker" in " ".join(p.cmdline())
        ]
        _, alive = psutil.wait_procs(owned, timeout=10)
        assert not alive, "MLX actor survived teardown"
        assert not ray.is_initialized()


def traced_provider(kind, trace):
    from vane.ai.providers import mlx as module

    descriptor_class = getattr(module, "MLX" + kind.title() + "EmbedderDescriptor")
    encoder_class = getattr(module, "MLX" + kind.title() + "Embedder")
    trace = str(trace)

    def event(name, **values):
        import mlx.core as mx
        import psutil

        with Path(trace).open("a") as log:
            log.write(
                json.dumps(
                    dict(
                        event=name,
                        pid=os.getpid(),
                        machine=platform.machine(),
                        metal=mx.metal.is_available(),
                        rss=psutil.Process().memory_info().rss,
                        metal_peak=mx.get_peak_memory(),
                        metal_active=mx.get_active_memory(),
                        **values,
                    )
                )
                + "\n"
            )

    class TracedEncoder(encoder_class):
        def _load_processor(self, *args):
            super()._load_processor(*args)
            event("processor_loaded")

        def embed_audio(self, clips):
            event(
                "input",
                samples=[len(c.samples) for c in clips],
                rates=[c.sample_rate for c in clips],
                input_sha256=[hashlib.sha256(c.samples.tobytes()).hexdigest() for c in clips],
            )
            return super().embed_audio(clips)

        def embed_video(self, clips):
            event(
                "input",
                indices=[list(c.frame_indices) for c in clips],
                times=[list(c.frame_times) for c in clips],
                frame_sha256=[[hashlib.sha256(f.tobytes()).hexdigest() for f in c.frames] for c in clips],
            )
            return super().embed_video(clips)

        def _infer(self, inputs, count):
            event("forward", rows=count, model_keys=sorted(inputs))
            return super()._infer(inputs, count)

        def close(self):
            closed = self._closed
            super().close()
            if not closed:
                event("closed")

    class TracedDescriptor(descriptor_class):
        def instantiate(self):
            actual = TracedEncoder(self.get_options())
            event("loaded", options=self.get_options())
            return actual

    class TracedProvider(module.MLXProvider):
        pass

    def factory(self, model=None, dimensions=None, *, options=None):
        return TracedDescriptor(model, dimensions, options=options)

    setattr(TracedProvider, "get_" + kind + "_embedder", factory)
    return TracedProvider()

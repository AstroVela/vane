# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Authored ONNX/processor fixtures, with no downloaded models or SDK import."""

import hashlib
import json
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

from vane.ai.providers._onnx_artifacts import SIDECARS, bundle_files, bundle_sha256

CONFIG = {
    "model_type": "embedding_gemma2",
    "text_config": {"embedding_dim": 768, "hidden_size": 512},
    "image_token_id": 3,
    "video_token_id": 4,
    "audio_token_id": 5,
}
TOKENS = {"<|image|>": 3, "<|video|>": 4, "<|audio|>": 5}


def write_bundle(root):
    root.mkdir(parents=True)
    for name in bundle_files("fp32"):
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            json.dumps(CONFIG if name == "config.json" else {}) if name in SIDECARS else "authored ONNX fixture"
        )
    return {"model_path": str(root), "model_sha256": bundle_sha256(root, "fp32"), "max_length": 128}


def install_sdk(monkeypatch):
    sdk = SimpleNamespace(
        events=[],
        available=["CPUExecutionProvider", "CUDAExecutionProvider", "CoreMLExecutionProvider"],
        failure=None,
        fail_tower=None,
        response=None,
        entered=None,
        release=None,
        active=0,
        returned_provider=None,
    )
    ort = ModuleType("onnxruntime")
    ort.get_available_providers = lambda: list(sdk.available)

    class Settings:
        def __init__(self):
            self.entries = {}

        def add_session_config_entry(self, key, value):
            self.entries[key] = value

    ort.SessionOptions = Settings

    class Session:
        def __init__(self, path, *, sess_options, providers, enable_fallback):
            self.tower = Path(path).stem
            self.selected = providers[0][0]
            self.live = False
            sdk.events.append(
                ("load", self.tower, threading.get_ident(), self.selected, enable_fallback, sess_options.entries.copy())
            )
            if sdk.failure is not None and (sdk.fail_tower is None or self.tower == sdk.fail_tower):
                raise sdk.failure
            self.live = True
            sdk.active += 1

        def get_providers(self):
            return [sdk.returned_provider or self.selected]

        def disable_fallback(self):
            sdk.events.append(("disable_fallback", self.tower))

        def get_inputs(self):
            if self.tower == "model":
                inputs = {
                    "input_ids": "int64",
                    "attention_mask": "int64",
                    "image_features": "float",
                    "video_features": "float",
                    "audio_features": "float",
                }
            elif self.tower == "vision_encoder":
                inputs = {"pixel_values": "float", "pixel_position_ids": "int64"}
            else:
                inputs = {"input_features": "float", "input_features_mask": "bool"}
            return [SimpleNamespace(name=key, type=f"tensor({value})") for key, value in inputs.items()]

        def get_outputs(self):
            return [
                SimpleNamespace(
                    name={
                        "model": "sentence_embedding",
                        "vision_encoder": "image_features",
                        "audio_encoder": "audio_features",
                    }[self.tower]
                )
            ]

        def run(self, output_names, feed):
            sdk.events.append(
                ("run", self.tower, threading.get_ident(), {name: value.copy() for name, value in feed.items()})
            )
            if self.tower == "model":
                if sdk.entered is not None:
                    sdk.entered.set()
                    assert sdk.release.wait(10)
                if sdk.response is not None:
                    return [sdk.response]
                digest = hashlib.sha256(b"".join(feed[name].tobytes() for name in sorted(feed))).digest()
                vector = np.zeros((1, 768), dtype=np.float32)
                vector[0, :32] = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) + 1
                vector /= np.linalg.norm(vector)
                return [vector]
            key = "pixel_values" if self.tower == "vision_encoder" else "input_features"
            result = np.zeros((len(feed[key]), 512), dtype=np.float32)
            result[:, 0] = np.mean(feed[key], axis=tuple(range(1, feed[key].ndim)))
            return [result]

        def __del__(self):
            if getattr(self, "live", False):
                self.live = False
                sdk.active -= 1
                sdk.events.append(("closed", self.tower, threading.get_ident()))

    ort.InferenceSession = Session
    graph = ModuleType("onnx")
    graph.load = lambda *args, **kwargs: SimpleNamespace(
        DESCRIPTOR=SimpleNamespace(full_name="onnx.ModelProto"), ListFields=lambda: []
    )
    processor_module = ModuleType("transformers")

    class Processor:
        tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda value: TOKENS[value])

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            sdk.events.append(("processor", threading.get_ident(), kwargs))
            assert kwargs == {"local_files_only": True, "trust_remote_code": False}
            return cls()

        def __call__(self, **kwargs):
            sdk.events.append(("process", threading.get_ident(), kwargs))
            prompt = kwargs["text"][0]
            values = {}
            counts = {}
            for kind, key, position in (
                ("image", "images", "image_position_ids"),
                ("video", "videos", "video_position_ids"),
            ):
                media = kwargs.get(key, [])
                media = (
                    media[0]
                    if kind == "image" and media
                    else [frame for video in media for frame in video]
                    if kind == "video"
                    else media
                )
                counts[kind] = len(media)
                if media:
                    values["pixel_values" if kind == "image" else "pixel_values_videos"] = np.stack(
                        [np.full((1, 768), np.mean(image), np.float32) for image in media]
                    )
                    values[position] = np.zeros((len(media), 1, 2), dtype=np.int64)
            audio = kwargs.get("audio", [])
            counts["audio"] = len(audio)
            if audio:
                values["input_features"] = np.stack([np.full((1, 128), np.mean(clip), np.float32) for clip in audio])
                values["input_features_mask"] = np.ones((len(audio), 1), dtype=bool)
            ids = [2]
            while prompt:
                matched = next((token for token in TOKENS if prompt.startswith(token)), None)
                if matched:
                    kind = {3: "image", 4: "video", 5: "audio"}[TOKENS[matched]]
                    count = counts[kind] if kind == "video" else 1
                    ids.extend([TOKENS[matched]] * count)
                    prompt = prompt[len(matched) :]
                else:
                    ids.append(ord(prompt[0]) + 100)
                    prompt = prompt[1:]
            values.update(
                input_ids=np.asarray([ids], dtype=np.int64), attention_mask=np.ones((1, len(ids)), dtype=np.int64)
            )
            return values

    processor_module.AutoProcessor = Processor
    for name, module in (("onnxruntime", ort), ("onnx", graph), ("transformers", processor_module)):
        monkeypatch.setitem(__import__("sys").modules, name, module)
    return sdk

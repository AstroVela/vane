# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Authored cross-language inputs for real ONNX model qualification."""

import argparse
import json
from pathlib import Path

import numpy as np

DOCUMENTS = ["Mars is the Red Planet.", "The Pacific Ocean is the largest ocean.", "大熊猫主要吃竹子。"]
QUERIES = ["Which planet is red?", "What is the largest ocean?", "大熊猫吃什么？"]


def fixture():
    image = np.zeros((32, 40, 3), dtype=np.uint8)
    image[4:28, 8:32, 0] = 255
    second = image[:, ::-1, ::-1].copy()
    audio = (0.25 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32)
    return {
        "documents": DOCUMENTS,
        "queries": QUERIES,
        "image": image.tolist(),
        "audio": audio.tolist(),
        "frames": [image.tolist(), second.tolist()],
    }


if __name__ == "__main__":
    from importlib.metadata import version

    from transformers import AutoProcessor

    parser = argparse.ArgumentParser(
        description="Prepare official Python processor tensors for independent Node inference"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = fixture()
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    image = np.asarray(data["image"], dtype=np.uint8)
    audio = np.asarray(data["audio"], dtype=np.float32)
    video = np.asarray(data["frames"], dtype=np.uint8)
    cases = {
        "documents": dict(text=["title: none | text: " + value for value in DOCUMENTS]),
        "queries": dict(text=["task: search result | query: " + value for value in QUERIES]),
        "image": dict(text=["<|image|>"], images=[[image]], images_kwargs={"device": "cpu"}),
        "audio": dict(text=["<|audio|>"], audio=[audio], sampling_rate=16000),
        "video": dict(
            text=["<|video|>"],
            videos=[video],
            videos_kwargs={"device": "cpu", "do_sample_frames": False, "add_timestamps": False},
        ),
        "mixed": dict(
            text=["title: none | text: red shape<|image|><|audio|><|video|>"],
            images=[[image]],
            audio=[audio],
            videos=[video],
            sampling_rate=16000,
            images_kwargs={"device": "cpu"},
            videos_kwargs={"device": "cpu", "do_sample_frames": False, "add_timestamps": False},
        ),
    }
    result = {
        "processor_versions": {name: version(name) for name in ("transformers", "torch", "torchvision", "Pillow")},
        "cases": {},
    }
    dtypes = {np.dtype("float32"): "float32", np.dtype("int64"): "int64", np.dtype("bool"): "bool"}
    for name, options in cases.items():
        values = processor(**options, return_tensors="np", padding=True, truncation=False)
        result["cases"][name] = {}
        for key, value in values.items():
            array = np.ascontiguousarray(value)
            if array.dtype not in dtypes:
                continue
            filename = f"{name}-{key}.bin"
            array.tofile(args.output / filename)
            result["cases"][name][key] = {"type": dtypes[array.dtype], "dims": list(array.shape), "file": filename}
    (args.output / "inputs.json").write_text(json.dumps(result, indent=2))

# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Authored cross-language inputs for real ONNX model qualification."""

import json
import sys
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
    Path(sys.argv[1]).write_text(json.dumps(fixture()))

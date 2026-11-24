# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Fingerprint every graph, external weight file and processor sidecar.

Run ``python -m vane.ai.providers._onnx_artifacts /absolute/model/path --dtype fp32``
before configuring the provider. No model or SDK is loaded by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._onnx_options import ONNX_DTYPES

SIDECARS = (
    "config.json",
    "config_sentence_transformers.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def bundle_files(dtype: str) -> tuple[str, ...]:
    suffix = ONNX_DTYPES[dtype]
    return SIDECARS + tuple(
        f"onnx/{tower}{suffix}.onnx{data}"
        for tower in ("model", "vision_encoder", "audio_encoder")
        for data in ("", "_data")
    )


def bundle_sha256(root: str | Path, dtype: str) -> str:
    root = Path(root)
    if dtype not in ONNX_DTYPES or not root.is_absolute() or not root.is_dir():
        raise EmbeddingConfigurationError("ONNX requires an absolute model directory and a declared dtype")
    # Optional tokenizer/remote-code sidecars must never alter an unhashed processor.
    for path in root.iterdir():
        if (
            path.is_file()
            and path.name not in SIDECARS
            and (
                path.suffix in {".json", ".py", ".jinja", ".model", ".txt"}
                and path.name not in {"README.txt", "LICENSE.txt"}
            )
        ):
            raise EmbeddingConfigurationError("Unexpected ONNX processor or tokenizer sidecar")
    manifest: dict[str, str] = {}
    resolved = root.resolve()
    for name in bundle_files(dtype):
        path = root / name
        if not path.is_file() or not path.resolve().is_relative_to(resolved) or path.is_symlink():
            raise EmbeddingConfigurationError(f"Missing or unsafe ONNX bundle file: {name}")
        if path.suffix in {".json", ".onnx"} and path.stat().st_size > 64 * 1024**2:
            raise EmbeddingConfigurationError("ONNX graph/processor metadata exceeds 64 MiB")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024**2), b""):
                digest.update(chunk)
        manifest[name] = digest.hexdigest()
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_external_data(model: Any, expected: str) -> None:
    """Validate tensor attributes as well as initializers before ORT opens data."""
    if model.DESCRIPTOR.full_name == "onnx.TensorProto" and model.external_data:
        values = {entry.key: entry.value for entry in model.external_data}
        if (
            len(values) != len(model.external_data)
            or values.keys() - {"location", "offset", "length", "checksum"}
            or values.get("location") != expected
        ):
            raise EmbeddingConfigurationError("ONNX external weights must reference their fingerprinted sibling file")
    for field, value in model.ListFields():
        if field.message_type is not None:
            children = value if field.is_repeated else (value,)
            for child in children:
                validate_external_data(child, expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--dtype", choices=ONNX_DTYPES, default="fp32")
    args = parser.parse_args()
    print(bundle_sha256(args.model_path, args.dtype))

# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL embeddings with the encoding used by paired SQR checkpoints.

Install vane-ai[qwen3_vl,video] on the driver and Ray workers. Supply an absolute
local Qwen3-VL checkpoint directory containing its tokenizer, processor and
safetensors weights. Model code is not downloaded or executed from the checkpoint.
The executing worker needs one CUDA GPU with BF16 support. Vane chooses its
default UDF backend; with the default Ray runner, the model lives in a Ray actor.

Example:
    python examples/qwen3_vl_sqr.py --model /models/Qwen3-VL-Embedding-2B \
        --query "A person opens a refrigerator."

The built-in provider name is qwen3_vl. It implements text, image and decoded
video embedding through the regular vane.ai APIs, with one fixed contract:

* 2048 dimensions, BF16 model execution with SDPA, last nonpadding token pooling
  and normalized float32 vectors;
* system instruction "Represent the user's input.", the checkpoint's chat
  template and default special-token processing, including any appended EOS;
* no encoding overrides, automatic truncation, resizing or frame resampling;
* text at most 128 tokens including the template and special tokens, multimodal
  inputs at most 8192 tokens;
* RGB UInt8 images and video frames with positive dimensions divisible by 32,
  at most 65536 pixels per frame;
* videos with an even 2..64 ordered frames, at most 8 MiB decoded pixels per
  clip, and distinct increasing nonnegative microsecond timestamps. Input video
  rows use the existing embed_video LIST of {frame_index, frame_time, data IMAGE}.

Model dimensions, tokenizer vocabulary, saved embedding rows, vision patch
configuration and weight-loading results are checked before inference. The
unused language-model head is allowed; missing trained model weights are not.
Planning/descriptor serialization does not import the model SDK or load weights.

For Vane Context, select this provider in the normal models JSON, without custom
Python registration:

    "embedding_profiles": {
        "video": {
            "provider": "qwen3_vl",
            "model": "/models/Qwen3-VL-Embedding-2B",
            "input_kind": "video", "modalities": ["video"],
            "dimensions": 2048, "execution": {"batch_size": 2}, "options": {}
        }
    }

Pair the SQR backend with this profile and the same absolute query_model_path,
query_instruction and BF16 dtype. Keep document_options/query_options empty.
Use caller sampling that fits the budgets (for example 32 frames at 256x256).
Reingest into a fresh store when changing encoding providers. Matching a model
name and vector width alone does not make older vLLM/SGLang vectors compatible:
their special-token contract differs. Validate reference query vectors for the
selected checkpoint before claiming retrieval quality. The fixed contract here
does not imply compatibility with every Qwen3-VL/SQR checkpoint or instruction.
"""

from __future__ import annotations

import argparse

import vane


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    try:
        with vane.connect() as connection:
            inputs = connection.sql("SELECT ? AS query", params=[args.query])
            result = inputs.select(
                vane.ai.embed(vane.col("query"), provider="qwen3_vl", model=args.model, dimensions=2048)
            ).fetchall()
            print(result)
    finally:
        vane.teardown_runner()


if __name__ == "__main__":
    main()

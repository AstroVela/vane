# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit multimodal model deployments, through Vane's regular AI APIs.

HTTP prompt: install vane-ai[openai,image]; deploy a video-capable model with
vLLM or SGLang, then run this example with --base-url and --model. Optional
authentication comes from VLLM_API_KEY or SGLANG_API_KEY, captured before worker
dispatch. No OpenAI environment setting is inherited by these deployments.

Local embedding: install vane-ai[qwen] and use --embedding with
Qwen/Qwen3-VL-Embedding-2B (or -8B). Select CPU/float32 or CUDA/float16 explicitly.
Model loading happens on the executing worker. Images, text and clips share
the model's vector space. Instruct models cannot be used as embedding models.

Decoded clip embedding uses vane.ai.embed_video on an ordered LIST of
{frame_index, frame_time, data IMAGE} records. The caller controls sampling.
Both Transformers Qwen and HTTP vLLM preserve all supplied frames and express
presentation timestamps at microsecond precision. Qwen has variable frame
counts, defaults to 64 frames / 64 MiB decoded bytes, and rejects token overflow.
Its max_pixels limits image pixels or total clip pixels during preprocessing.

The vLLM HTTP embedding deployment must support /v1/embeddings messages and
video/jpeg sequences with media_io_kwargs (as in Qwen3-VL-Embedding). SGLang
uses its multimodal input dictionaries for text/image embedding and its serving
chat template; decoded video embedding is rejected because this wire format
does not carry the supplied timestamps. Neither provider starts a local serving
engine in transport='http' mode. Native text inference remains a separate mode.

Declare media_mime_types for the actual prompt model. An HTTP endpoint alone
does not establish model capabilities. Unsupported inputs and endpoint errors
fail explicitly, without changing models, sampling, or transport.
"""

from __future__ import annotations

import argparse

import vane
from vane.ai import embed, prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["vllm", "sglang"], default="vllm")
    parser.add_argument("--base-url", help="Explicit deployment URL, including /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", help="MP4 path accessible to executing workers")
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    with vane.connect() as connection:
        inputs = connection.sql("SELECT 'Find the person opening the door' AS query")
        if args.embedding:
            expression = embed(
                vane.col("query"),
                provider="transformers",
                model=args.model,
                device=args.device,
                dtype="float32" if args.device == "cpu" else "float16",
                normalize=True,
            )
        else:
            if not args.video or not args.base_url:
                parser.error("HTTP video prompt requires --video and --base-url")
            expression = prompt(
                [vane.col("query"), vane.file(args.video, "video/mp4")],
                provider=args.provider,
                model=args.model,
                transport="http",
                base_url=args.base_url,
                media_mime_types=["video/mp4"],
                max_tokens=256,
            )
        print(inputs.select(expression.alias("result")).fetchall())


if __name__ == "__main__":
    main()

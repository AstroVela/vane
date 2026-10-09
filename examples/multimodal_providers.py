# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Native Python engines through Vane's regular AI APIs.

Install vane-ai[vllm,image] or vane-ai[sglang,image] on execution workers.
No serving endpoint is used. Select a model ID/path explicitly; for example,
Qwen/Qwen3-VL-Embedding-2B for embedding or a video-capable Instruct model for
prompting. These are examples, not default models. Declare embedding dimensions
and compatible engine parameters, such as dtype and model context length.

Vane's actor owns the native model engine and releases it when execution ends.
The engine's tensor/pipeline parallelism must fit gpus_per_actor. Use zero only
for an explicitly configured engine/device that can execute on CPU.

Embedding options accept engine_args, pooling_args (vLLM), processor_kwargs,
chat_template and chat_template_kwargs. For prompts, configure engine_args and
generate_args (including native sampling_params). Top-level max_tokens and
sampling_params token limits cannot both specify the same setting. Declare the
image/video media_mime_types actually supported by the selected model.
For a checkpoint containing only weights/configuration, set engine_args.tokenizer
(vLLM) or engine_args.tokenizer_path (SGLang) to its matching processing source.
This source supplies the chat template and tokenizer; image/video inputs also
require the matching processor configuration there. vLLM's tokenizer_revision
selects the processing revision independently of the model revision.

Decoded clip embedding uses vane.ai.embed_video on an ordered LIST of
{frame_index, frame_time, data IMAGE}. Sampling belongs to the caller; both
native adapters retain frame timestamps at microsecond precision. SGLang uses
its native processor_output interface and submits the resulting token IDs
directly, including for text. Chat templates own special tokens, so SGLang
embedding processor_kwargs.add_special_tokens must remain False. Models
must accept the configured processor inputs. Incompatible inputs or SDK/model
parameters fail; no model, transport or modality is substituted.
Embedding dimensions describe the expected output shape. Set
supports_overriding_dimensions=True only to request a model-supported
Matryoshka dimension override; fixed-width models must leave it disabled.
"""

from __future__ import annotations

import argparse
import json

import vane
from vane.ai import embed, prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["vllm", "sglang"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", help="Video path accessible to executing workers")
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--engine-args", type=json.loads, default={})
    parser.add_argument("--gpus-per-actor", type=int, default=1)
    args = parser.parse_args()
    options = {"engine_args": args.engine_args, "gpus_per_actor": args.gpus_per_actor}
    with vane.connect() as connection:
        inputs = connection.sql("SELECT 'Find the person opening the door' AS query")
        if args.embedding:
            if args.dimensions is None:
                parser.error("--embedding requires --dimensions matching the selected model")
            expression = embed(
                vane.col("query"),
                provider=args.provider,
                model=args.model,
                dimensions=args.dimensions,
                normalize=True,
                **options,
            )
        else:
            if not args.video:
                parser.error("video prompting requires --video")
            expression = prompt(
                [vane.col("query"), vane.file(args.video, "video/mp4")],
                provider=args.provider,
                model=args.model,
                media_mime_types=["video/mp4"],
                max_tokens=256,
                **options,
            )
        print(inputs.select(expression.alias("result")).fetchall())


if __name__ == "__main__":
    main()

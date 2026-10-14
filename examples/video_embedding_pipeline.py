# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Create visual clip embeddings with a local Cosmos profile and a task ID.

Install vane-ai[video,cosmos]. Supply a reviewed model commit with --revision.
Model weights are loaded only after task submission and successful probing.
"""

import argparse

from vane.ai import VideoEmbeddingPipeline, VideoEmbeddingProfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", choices=["cuda", "cuda:0"], default="cuda")
    args = parser.parse_args()
    profile = VideoEmbeddingProfile.from_provider(
        f"cosmos-{args.revision}",
        modality="visual",
        provider="transformers",
        model="nvidia/Cosmos-Embed1-224p",
        revision=args.revision,
        device=args.device,
        dtype="float16",
        trust_remote_code=True,
    )
    with VideoEmbeddingPipeline(profiles={"visual": profile}) as pipeline:
        task = pipeline.create(
            {
                "input": {"type": "video", "uri": args.video},
                "modalities": ["visual"],
                "selection": {"policy": "all", "unit": "clip"},
                "segmentation": {"clip_seconds": 10},
                "output": "separate",
                "return_timeline": True,
                "async": True,
            }
        )
        print(f"Task ID: {task.id}", flush=True)
        for vector in task.result()["data"]:
            print(
                f"{vector['start_sec']:g}-{vector['end_sec']:g}s "
                f"{vector['modality']} {len(vector['embedding'])} dimensions {vector['profile']}"
            )


if __name__ == "__main__":
    main()

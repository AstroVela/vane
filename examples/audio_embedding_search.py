# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Embed a local WAV (at most ten seconds) and rank English sound descriptions.

Install vane-ai[clap,audio]. Run with --audio recording.wav; add --device cuda
for the worker's allocated GPU. No speech recognizer or hosted API is used.
Long recordings must be windowed explicitly before embedding.
"""

import argparse

import vane
from vane.ai import embed, embed_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--revision", required=True, help="Hugging Face model commit to reproduce the run")
    args = parser.parse_args()
    options = dict(
        provider="transformers",
        model="laion/clap-htsat-unfused",
        device=args.device,
        revision=args.revision,
        normalize=True,
    )
    with vane.connect() as conn:
        clips = conn.sql(
            "SELECT struct_pack(sample_rate := 48000, data := resample(audio_file(?), 48000)) AS audio",
            params=[args.audio],
        )
        vector = embed_audio(clips, vane.col("audio"), **options).select("embedding").fetchone()[0]
        queries = conn.sql("SELECT unnest(['A dog barking', 'A ringing alarm', 'Music playing', 'Heavy rain']) AS text")
        embedded = embed(queries, vane.col("text"), **options)
        for text, values in embedded.select("text", "embedding").fetchall():
            print(f"{sum(a * b for a, b in zip(vector, values, strict=True)):.4f}\t{text}")


if __name__ == "__main__":
    main()

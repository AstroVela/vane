# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Search local images with CLIP: pip install 'vane-ai[transformers,image]'.

python examples/image_embedding_search.py ./photos 'a dog in the snow'
The first run downloads the model; --revision pins both encoders to one commit.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import vane
from vane.ai import embed, embed_image

MODEL = "sentence-transformers/clip-ViT-B-32"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("query")
    parser.add_argument("--revision", help="Hugging Face commit SHA for both encoders")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    paths = sorted(
        str(path.resolve())
        for path in args.directory.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"} and path.is_file()
    )
    if not paths:
        parser.error("directory contains no supported image files")

    # Workers should import the installed package when launched from a checkout.
    os.environ["PYTHONSAFEPATH"] = "1"
    vane.set_runner_local()
    with vane.connect(config={"image_backend": "python"}) as conn:
        source = conn.sql("SELECT unnest(?::VARCHAR[]) AS path", params=[paths])
        source.create_view("image_paths")
        decoded = conn.sql(
            "SELECT path, decode_image_file(image_file(path), mode => 'RGB', on_error => 'null') AS image FROM image_paths"
        )
        vectors = decoded.select(
            vane.col("path"),
            embed_image(vane.col("image"), model=MODEL, revision=args.revision, normalize=True, batch_size=16).alias(
                "embedding"
            ),
        )
        # Materialize once so retrieval does not encode every image again.
        conn.register("image_vectors", vectors.to_arrow_table())
        query = conn.sql("SELECT ?::VARCHAR AS text", params=[args.query])
        query_vector = query.select(
            embed(vane.col("text"), provider="transformers", model=MODEL, revision=args.revision, normalize=True).alias(
                "embedding"
            )
        ).to_arrow_table()
        conn.register("query_vector", query_vector)
        rows = conn.sql(
            "SELECT path, array_cosine_similarity(i.embedding, q.embedding) AS score FROM image_vectors i CROSS JOIN query_vector q WHERE i.embedding IS NOT NULL ORDER BY score DESC LIMIT ?",
            params=[args.top_k],
        ).fetchall()
        for path, score in rows:
            print(f"{score:.4f}\t{path}")


if __name__ == "__main__":
    main()

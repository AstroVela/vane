# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Local semantic search with an explicitly provisioned LiteRT-LM checkpoint."""

import argparse

import numpy as np

import vane
from vane.ai import TextInput, embed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    options = vars(args)
    documents = ["Mars is the Red Planet.", "The Pacific Ocean is the largest ocean.", "大熊猫主要吃竹子。"]
    vane.set_runner_local()
    try:
        with vane.connect() as connection:
            source = connection.sql("SELECT unnest(?::VARCHAR[]) AS text", params=[documents])
            vectors = (
                embed(
                    source,
                    TextInput(vane.col("text")),
                    provider="litert",
                    dimensions=256,
                    batch_size=8,
                    max_retries=0,
                    **options,
                )
                .to_arrow_table()["embedding"]
                .to_pylist()
            )
            query = connection.sql("SELECT 'Which planet is red?' AS text")
            query_vector = (
                embed(
                    query,
                    TextInput(vane.col("text")),
                    provider="litert",
                    dimensions=256,
                    input_type="query",
                    max_retries=0,
                    **options,
                )
                .to_arrow_table()["embedding"][0]
                .as_py()
            )
            scores = np.asarray(vectors) @ np.asarray(query_vector)
            for index in np.argsort(-scores):
                print(f"{scores[index]:.4f}\t{documents[index]}")
    finally:
        vane.teardown_runner()


if __name__ == "__main__":
    main()

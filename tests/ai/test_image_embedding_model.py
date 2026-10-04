# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real CLIP validation: VANE_TEST_CLIP=1 (downloads model weights)."""

from __future__ import annotations

import os

import numpy as np
import pytest

import vane
from vane.ai import embed, embed_image
from vane.execution.udf_file_contract import FileUDFContract


@pytest.mark.external_service
@pytest.mark.skipif(os.environ.get("VANE_TEST_CLIP") != "1", reason="set VANE_TEST_CLIP=1 to load real CLIP weights")
def test_clip_text_and_images_share_a_vector_space():
    pytest.importorskip("sentence_transformers")
    import pyarrow as pa

    model = "sentence-transformers/clip-ViT-B-32"
    dtype = vane.image_type("RGB")
    colors = [np.full((32, 32, 3), rgb, dtype=np.uint8) for rgb in ([255, 0, 0], [0, 0, 255])]
    images = FileUDFContract("fixture", (), (dtype,)).scalar_outputs_to_array(colors)
    with vane.connect() as conn:
        source = conn.from_arrow(pa.table({"id": [0, 1], "image": images}))
        vectors = (
            source.select(vane.col("id"), embed_image(vane.col("image"), model=model, normalize=True).alias("vector"))
            .order("id")
            .fetchall()
        )
        query = conn.sql("SELECT * FROM (VALUES (0, 'a solid red image'), (1, 'a solid blue image')) t(id, text)")
        text = (
            query.select(
                vane.col("id"),
                embed(vane.col("text"), provider="transformers", model=model, normalize=True).alias("vector"),
            )
            .order("id")
            .fetchall()
        )
    matrix = np.asarray([row[1] for row in vectors])
    queries = np.asarray([row[1] for row in text])
    assert matrix.shape == queries.shape == (2, 512)
    assert np.isfinite(matrix).all() and np.isfinite(queries).all()
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal((queries @ matrix.T).argmax(axis=1), [0, 1])

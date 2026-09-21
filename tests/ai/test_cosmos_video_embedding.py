# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real Cosmos video/text retrieval through Vane's default Ray runner.

No downloads happen inside this test and no model/media artifacts are
distributed with Vane.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

import vane
from vane.ai import embed, embed_video

REVISION = "787e0b996f5260a71ad474a283c90539a2e12986"
VIDEO_SHA256 = "75ab52aa5868d866b9974b922bdf292b501d2e26a13396b5638a3c58058aae8a"
CAPTIONS = [
    "a train moving through a snowy landscape",
    "an empty room with a chair by a window",
    "a dog playing in a garden",
    "a man wearing red spandex throwing a javelin",
    "an athlete stretching beside a track",
    "a crowd watching a basketball game",
]


@pytest.fixture
def cosmos_assets():
    cache = os.environ.get("VANE_TEST_COSMOS_CACHE")
    video = os.environ.get("VANE_TEST_COSMOS_VIDEO")
    if not cache or not video:
        pytest.skip("set VANE_TEST_COSMOS_CACHE and VANE_TEST_COSMOS_VIDEO to opt into cached real-model validation")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        pytest.fail("start the test process with HF_HUB_OFFLINE=1")
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("av")
    if not torch.cuda.is_available():
        pytest.skip("Cosmos integration requires CUDA")
    assert Path(cache).is_dir()
    assert hashlib.sha256(Path(video).read_bytes()).hexdigest() == VIDEO_SHA256
    return cache, video


@pytest.mark.gpu
@pytest.mark.real_ray
@pytest.mark.parametrize("precision", ["float16", "float32"])
def test_video_file_to_joint_vector_retrieval(cosmos_assets, ray_local, monkeypatch, precision):
    cache, video = cosmos_assets
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    options = dict(
        provider="transformers",
        model="nvidia/Cosmos-Embed1-224p",
        revision=REVISION,
        trust_remote_code=True,
        device="cuda",
        dtype=precision,
        cache_folder=cache,
        local_files_only=True,
    )
    with vane.connect(config={"video_backend": "python"}) as conn:
        # Sampling is an explicit upstream step. Keep the original indices,
        # timestamps and VIDEOFILE provenance rather than rebuilding images.
        clips = conn.sql(
            "WITH decoded AS (SELECT video_frames(video_file($1), max_output_frames => 128) AS all_frames) "
            "SELECT list_select(all_frames, list_transform(range(8), i -> "
            "1 + floor(i * (len(all_frames) - 1) / 7.0)::BIGINT)) AS frames FROM decoded",
            params=[video],
        )
        result = clips.select(embed_video(vane.col("frames"), **options).alias("embedding"))
        assert "ray_actor" in result.explain()
        video_vectors = np.asarray([row[0] for row in result.fetchall()], dtype=np.float32)
        conn.register("captions", pa.table({"id": range(len(CAPTIONS)), "text": CAPTIONS}))
        result = embed(conn.table("captions"), vane.col("text"), **options)
        text_vectors = np.asarray(
            [row[1] for row in result.select("id", "embedding").order("id").fetchall()], dtype=np.float32
        )
    assert video_vectors.shape == (1, 256) and text_vectors.shape == (6, 256)
    assert np.isfinite(video_vectors).all() and np.isfinite(text_vectors).all()
    np.testing.assert_allclose(np.linalg.norm(video_vectors, axis=1), 1.0, atol=2e-3)
    np.testing.assert_allclose(np.linalg.norm(text_vectors, axis=1), 1.0, atol=2e-3)
    assert int(np.argmax(video_vectors @ text_vectors.T)) == 3

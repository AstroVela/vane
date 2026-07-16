from __future__ import annotations

import pytest


def test_image_decode_failure_raises_instead_of_zero_tensor(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("PIL")
    pa = pytest.importorskip("pyarrow")

    from multimodal_inference_benchmarks.image_classification import vane_main

    def fail_download(_path: str) -> bytes:
        raise ValueError("cannot read image")

    monkeypatch.setattr(vane_main, "_download_image_bytes", fail_download)

    with pytest.raises(RuntimeError, match="Failed to decode/transform image"):
        vane_main._decode_and_transform(pa.table({"image_url": ["broken.jpg"]}))

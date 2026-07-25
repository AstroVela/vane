# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import sys

import numpy as np
import pytest

from duckdb.datasource import video_reader


def test_missing_decord_error_names_video_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "decord", None)
    with pytest.raises(ImportError, match=r"vane-ai\[video\]") as exc_info:
        video_reader._open_decord_reader("nonexistent.mp4")
    assert "decord" in str(exc_info.value)
    assert "Linux x86-64" in str(exc_info.value)


def test_missing_pillow_error_names_video_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "PIL.Image", None)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ImportError, match=r"vane-ai\[video\]") as exc_info:
        video_reader._resize_rgb_frame(frame, width=1, height=1)
    assert "pillow" in str(exc_info.value)


def test_missing_psutil_error_names_video_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    with pytest.raises(ImportError, match=r"vane-ai\[video\]"):
        video_reader._wait_for_memory()


def test_provider_ndarray_without_pillow_names_image_extra(monkeypatch):
    pytest.importorskip("openai")
    from vane.ai.providers.openai import OpenAIPrompter

    monkeypatch.setitem(sys.modules, "PIL", None)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ImportError, match=r"vane-ai\[image\]"):
        OpenAIPrompter._process_ndarray(None, frame)

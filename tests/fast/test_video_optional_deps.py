# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest

from vane.datasource import video_reader


def test_video_source_reports_psutil_extra_when_memory_admission_dependency_is_missing(monkeypatch):
    real_import = importlib.import_module

    def fail_psutil(name, package=None):
        if name == "psutil":
            raise ImportError("missing psutil")
        return real_import(name, package)

    monkeypatch.setattr(video_reader.importlib, "import_module", fail_psutil)

    with pytest.raises(ImportError, match=r"psutil.*vane-ai\[video\]"):
        video_reader._wait_for_memory(lambda: None)

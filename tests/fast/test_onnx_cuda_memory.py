# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""CUDA budget probes restore caller ownership before ORT sessions are created."""

import ctypes
from types import SimpleNamespace

import pytest

from vane.ai.providers import _onnx_cuda


def driver_fixture(monkeypatch, *, free=10 * 1024**3, total=12 * 1024**3, failure=None):
    calls = []

    class Function:
        def __init__(self, name):
            self.name = name

        def __call__(self, *args):
            calls.append(self.name)
            if self.name == failure:
                return 2
            if self.name == "cuDeviceGet":
                assert args[1] == 0
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_int))[0] = 7
            elif self.name == "cuDevicePrimaryCtxRetain":
                assert args[1].value == 7
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = 0x1234
            elif self.name == "cuCtxPushCurrent_v2":
                assert args[0].value == 0x1234
            elif self.name == "cuCtxPopCurrent_v2":
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = 0x1234
            elif self.name == "cuDevicePrimaryCtxRelease_v2":
                assert args[0].value == 7
            elif self.name == "cuMemGetInfo_v2":
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_size_t))[0] = free
                ctypes.cast(args[1], ctypes.POINTER(ctypes.c_size_t))[0] = total
            return 0

    names = (
        "cuInit",
        "cuDeviceGet",
        "cuDevicePrimaryCtxRetain",
        "cuCtxPushCurrent_v2",
        "cuMemGetInfo_v2",
        "cuCtxPopCurrent_v2",
        "cuDevicePrimaryCtxRelease_v2",
    )
    driver = SimpleNamespace(**{name: Function(name) for name in names})
    monkeypatch.setattr(_onnx_cuda, "_load_driver", lambda: driver)
    return calls


@pytest.mark.parametrize(
    ("free", "total", "expected"),
    [(10 * 1024**3, 12 * 1024**3, 7 * 1024**3), (800 * 1024**2, 1024**3, 288 * 1024**2)],
)
def test_budget_leaves_device_reserve_and_releases_its_context(monkeypatch, free, total, expected):
    calls = driver_fixture(monkeypatch, free=free, total=total)
    assert _onnx_cuda.cuda_arena_limit() == expected
    assert calls[-2:] == ["cuCtxPopCurrent_v2", "cuDevicePrimaryCtxRelease_v2"]
    assert len(calls) == 7


@pytest.mark.parametrize(
    ("failure", "cleanup"),
    [
        ("cuInit", []),
        ("cuDeviceGet", []),
        ("cuDevicePrimaryCtxRetain", []),
        ("cuCtxPushCurrent_v2", ["cuDevicePrimaryCtxRelease_v2"]),
        ("cuMemGetInfo_v2", ["cuCtxPopCurrent_v2", "cuDevicePrimaryCtxRelease_v2"]),
        ("cuCtxPopCurrent_v2", ["cuDevicePrimaryCtxRelease_v2"]),
        ("cuDevicePrimaryCtxRelease_v2", []),
    ],
)
def test_query_failure_is_not_retried_and_releases_acquired_ownership(monkeypatch, failure, cleanup):
    calls = driver_fixture(monkeypatch, failure=failure)
    with pytest.raises(RuntimeError, match=failure):
        _onnx_cuda.cuda_arena_limit()
    failed_at = calls.index(failure)
    assert calls[failed_at + 1 :] == cleanup
    assert calls.count(failure) == 1


@pytest.mark.parametrize(("free", "total"), [(0, 12 * 1024**3), (1024, 1024**3)])
def test_exhausted_budget_fails_before_any_model_work(monkeypatch, free, total):
    calls = driver_fixture(monkeypatch, free=free, total=total)
    with pytest.raises(MemoryError, match="no headroom"):
        _onnx_cuda.cuda_arena_limit()
    assert calls[-2:] == ["cuCtxPopCurrent_v2", "cuDevicePrimaryCtxRelease_v2"]


@pytest.mark.parametrize(("free", "total"), [(1, 0), (1025, 1024)])
def test_invalid_device_report_does_not_keep_a_context(monkeypatch, free, total):
    calls = driver_fixture(monkeypatch, free=free, total=total)
    with pytest.raises(RuntimeError, match="invalid device capacity"):
        _onnx_cuda.cuda_arena_limit()
    assert calls[-2:] == ["cuCtxPopCurrent_v2", "cuDevicePrimaryCtxRelease_v2"]

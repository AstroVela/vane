# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from duckdb.execution import udf_subprocess
from duckdb.execution.udf_subprocess import _SingleSubprocessExecutor


class _FakeSocket:
    """Tracks the timeout like a real socket: None = blocking, 0.0 = non-blocking."""

    def __init__(self, initial: float | None):
        self.timeout = initial
        self.settimeout_calls: list[float | None] = []

    def gettimeout(self) -> float | None:
        return self.timeout

    def settimeout(self, value: float | None) -> None:
        self.timeout = value
        self.settimeout_calls.append(value)


def _executor_with(sock: _FakeSocket) -> _SingleSubprocessExecutor:
    executor = object.__new__(_SingleSubprocessExecutor)
    executor._require_socket = lambda: sock  # type: ignore[method-assign]
    executor._broken_error = None

    def _mark_broken(error: str, *, actor_lost: bool = False) -> None:
        executor._broken_error = error

    executor._mark_broken = _mark_broken  # type: ignore[method-assign]
    return executor


# None = blocking mode (the leak in the issue), 5.0 = existing finite timeout,
# 0.0 = non-blocking mode.
@pytest.mark.parametrize("initial", [None, 5.0, 0.0], ids=["blocking", "finite", "non-blocking"])
def test_recv_expected_restores_timeout_after_success(monkeypatch, initial):
    sock = _FakeSocket(initial)
    executor = _executor_with(sock)
    monkeypatch.setattr(udf_subprocess, "_recv_message", lambda _sock: (0x1, b"ok"))

    msg_type, payload = executor._recv_expected((0x1,), timeout_s=2.0)

    assert (msg_type, payload) == (0x1, b"ok")
    assert sock.timeout == initial
    # The temporary timeout was actually applied before being restored.
    assert sock.settimeout_calls[0] == 2.0
    assert sock.settimeout_calls[-1] == initial


@pytest.mark.parametrize("initial", [None, 5.0, 0.0], ids=["blocking", "finite", "non-blocking"])
def test_recv_expected_restores_timeout_after_failure(monkeypatch, initial):
    sock = _FakeSocket(initial)
    executor = _executor_with(sock)

    def _boom(_sock):
        raise OSError("connection reset")

    monkeypatch.setattr(udf_subprocess, "_recv_message", _boom)

    with pytest.raises(RuntimeError):
        executor._recv_expected((0x1,), timeout_s=2.0)

    assert sock.timeout == initial
    assert sock.settimeout_calls[-1] == initial


def test_recv_expected_without_timeout_never_touches_socket_timeout(monkeypatch):
    sock = _FakeSocket(None)
    executor = _executor_with(sock)
    monkeypatch.setattr(udf_subprocess, "_recv_message", lambda _sock: (0x1, b""))

    executor._recv_expected((0x1,))

    assert sock.settimeout_calls == []

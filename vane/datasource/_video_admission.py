# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Process-local bounded decoder admission without blocking native workers."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable

from vane.datasource._iterator import _DataSourceWait


class _DecodePermit(_DataSourceWait):
    def __init__(self, owner: _DecodeAdmission):
        self._owner = owner
        self._state = "pending"
        self._error: BaseException | None = None
        self._wakeup: Callable[[], None] | None = None

    def subscribe(self, wakeup: Callable[[], None]) -> bool:
        with self._owner._condition:
            if self._state != "pending":
                return True
            self._wakeup = wakeup
            return False

    def check_admitted(self) -> None:
        with self._owner._condition:
            if self._error is not None:
                raise self._error
            if self._state != "admitted":
                raise RuntimeError("Decoder permit is not admitted")

    def close(self) -> None:
        with self._owner._condition:
            if self._state == "closed":
                return
            if self._state == "admitted":
                self._owner._active -= 1
            self._owner._pending.pop(self, None)
            self._state = "closed"
            self._wakeup = None
            self._error = None
            self._owner._condition.notify_all()


class _DecodeAdmission:
    """FIFO permits; one short-lived coordinator polls host memory off-engine.

    The coordinator only admits requests. It neither decodes nor buffers frames,
    and exits when its queue becomes empty. Open decoders retain permits across
    downstream backpressure, so the configured resource bound stays intact.
    """

    def __init__(self, capacity: int, memory_ready: Callable[[], bool], interval: float):
        if capacity < 1 or interval <= 0:
            raise ValueError("Decoder capacity and memory polling interval must be positive")
        self._capacity = capacity
        self._memory_ready = memory_ready
        self._interval = interval
        self._condition = threading.Condition()
        self._pending: OrderedDict[_DecodePermit, None] = OrderedDict()
        self._active = 0
        self._thread: threading.Thread | None = None

    def request(self) -> _DecodePermit:
        permit = _DecodePermit(self)
        with self._condition:
            self._pending[permit] = None
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="vane-video-admission", daemon=True)
                try:
                    self._thread.start()
                except BaseException:
                    self._thread = None
                    self._pending.pop(permit)
                    raise
            self._condition.notify_all()
        return permit

    def _run(self) -> None:
        while True:
            with self._condition:
                if not self._pending:
                    self._thread = None
                    return
                if self._active >= self._capacity:
                    self._condition.wait()
                    continue
            error = None
            try:
                ready = self._memory_ready()
            except BaseException as exc:
                error = exc
                ready = True
            with self._condition:
                if not self._pending:
                    continue
                if not ready:
                    self._condition.wait(timeout=self._interval)
                    continue
                permit, _ = self._pending.popitem(last=False)
                if error is None:
                    permit._state = "admitted"
                    self._active += 1
                else:
                    permit._state = "failed"
                    permit._error = error
                wakeup, permit._wakeup = permit._wakeup, None
            if wakeup is not None:
                wakeup()


class _MemoryAdmission:
    """Preserve the reader's high/low-watermark hysteresis."""

    def __init__(self, sample: Callable[[], object], minimum_available: int, high: float, low: float):
        self._sample = sample
        self._minimum_available = minimum_available
        self._high = high
        self._low = low
        self._recovering = False

    def __call__(self) -> bool:
        memory = self._sample()
        enough_available = self._minimum_available > 0 and memory.available >= self._minimum_available  # type: ignore[attr-defined]
        watermark = self._low if self._recovering else self._high
        ready = enough_available or memory.percent < watermark  # type: ignore[attr-defined]
        self._recovering = not ready
        return bool(ready)

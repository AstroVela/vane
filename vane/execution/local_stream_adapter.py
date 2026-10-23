# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""A cancellable local producer window matching Ray's generator window."""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import AbstractContextManager

from vane.execution.udf_lifecycle import ExecutionCancellationScope
from vane.execution.udf_stream_backpressure import STREAM_BUFFER_BLOCKS


class LocalStreamAdapter:
    """Own unread-block permits for one invocation, not a reusable worker.

    The worker requests a permit before advancing its output iterator. A
    native-capacity-aware read returns it; releasing an Arrow view does not.
    Cancellation wakes this wait independently of DATA delivery.
    """

    def __init__(self, max_blocks: int = STREAM_BUFFER_BLOCKS) -> None:
        if type(max_blocks) is not int or max_blocks <= 0:
            raise ValueError("stream output block capacity must be a positive integer")
        self._max_blocks = max_blocks
        self._condition = threading.Condition()
        self._outstanding = 0
        self._closed = False

    def _wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def reserve(
        self,
        scope: ExecutionCancellationScope,
        wait_context: Callable[[], AbstractContextManager[None]],
    ) -> None:
        unregister = scope.register_cancel_wakeup(self._wake)
        try:
            while True:
                with self._condition:
                    scope.raise_if_cancelled("local output stream")
                    if self._closed:
                        raise RuntimeError("local output stream is closed")
                    if self._outstanding < self._max_blocks:
                        self._outstanding += 1
                        return
                # CPU suspension/resumption can call admission wakeups. Do
                # not hold the stream lock while entering or leaving it.
                with wait_context():
                    with self._condition:
                        self._condition.wait_for(
                            lambda: self._closed or scope.is_set() or self._outstanding < self._max_blocks
                        )
        finally:
            unregister()

    def consumed(self) -> None:
        with self._condition:
            if self._outstanding <= 0:
                raise RuntimeError("local output stream returned an unowned permit")
            self._outstanding -= 1
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

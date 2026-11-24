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

    def __init__(self) -> None:
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
                    if self._outstanding < STREAM_BUFFER_BLOCKS:
                        self._outstanding += 1
                        return
                # CPU suspension/resumption can call admission wakeups. Do
                # not hold the stream lock while entering or leaving it.
                with wait_context():
                    with self._condition:
                        self._condition.wait_for(
                            lambda: self._closed or scope.is_set() or self._outstanding < STREAM_BUFFER_BLOCKS
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

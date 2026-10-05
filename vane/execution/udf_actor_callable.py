# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Framework-owned serial computation for synchronous callable-class actors."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pyarrow as pa

from vane.execution._udf_validation import ensure_synchronous_udf_result


class ActorCallableRuntime:
    """One lazy worker per actor, shared by Ray and subprocess execution.

    Construction, lifecycle hooks and batch conversion stay on the owner.
    Only the synchronous callable invocation moves to the worker. Waiting for
    each result keeps existing output backpressure in control of subsequent work.
    """

    def __init__(self) -> None:
        self._owner = threading.get_ident()
        self._pool: ThreadPoolExecutor | None = None
        self._closed = False

    def check_owner(self) -> None:
        if threading.get_ident() != self._owner:
            raise RuntimeError("UDF must be called and closed by its owning actor thread")

    def __call__(self, udf: Any, table: pa.Table) -> Any:
        self.check_owner()
        if self._closed:
            raise RuntimeError("UDF actor worker is closed")
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vane-actor-udf")
        future = self._pool.submit(udf, table)
        try:
            return ensure_synchronous_udf_result(future.result())
        finally:
            # Do not retain a Future (including its exception graph) while the
            # caller suspends output consumption. Returned Arrow owns its buffers.
            del future

    def close(self) -> None:
        self.check_owner()
        self._closed = True
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

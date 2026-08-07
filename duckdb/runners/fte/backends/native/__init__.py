# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from duckdb.runners.fte.backends.native.backend import (
    NativeFteWorkerManagerBackend as NativeFteWorkerManagerBackend,
)
from duckdb.runners.fte.backends.native.backend import (
    NativeTaskResultHandle as NativeTaskResultHandle,
)
from duckdb.runners.fte.backends.native.backend import (
    NativeWorkerHandle as NativeWorkerHandle,
)

__all__ = [
    "NativeFteWorkerManagerBackend",
    "NativeTaskResultHandle",
    "NativeWorkerHandle",
]

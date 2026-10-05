# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""A branchable filesystem implemented in C++ with SQLite storage."""

from vane_fs._native import (
    BranchInfo,
    CapacityError,
    Change,
    CollectionResult,
    ConflictError,
    Error,
    FileStat,
    MergePreview,
    RecoveryResult,
    Session,
    StalePreviewError,
    Workspace,
)
from vane_fs.vane import register_workspace, snapshot_url, unregister_workspace

__all__ = [
    "BranchInfo",
    "CapacityError",
    "Change",
    "CollectionResult",
    "ConflictError",
    "Error",
    "FileStat",
    "MergePreview",
    "RecoveryResult",
    "Session",
    "StalePreviewError",
    "Workspace",
    "register_workspace",
    "snapshot_url",
    "unregister_workspace",
]

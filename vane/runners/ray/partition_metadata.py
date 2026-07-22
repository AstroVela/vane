# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Ray-specific materialized result types.

This module exists to avoid a circular import between ray/runner.py
(which imports RayQueryDriverClient) and ray/driver.py (which needs PartitionMetadata).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ray

from vane.runners.common import MaterializedResult, PartitionMetadata
from vane.runners.ray.safe_get import resolve_object_refs_blocking

if TYPE_CHECKING:
    import pyarrow as pa


class RayMaterializedResult(MaterializedResult):
    """Wraps a Ray ObjectRef with partition metadata."""

    def __init__(
        self,
        partition: ray.ObjectRef[Any],
        metadatas: PartitionMetadataAccessor | None = None,
        metadata_idx: int | None = None,
        release_owner: Any | None = None,
        release_plan_id: str | None = None,
        release_token: str | None = None,
    ):
        self._partition = partition
        self._metadatas = metadatas
        self._metadata_idx = metadata_idx
        self._release_owner = release_owner
        self._release_plan_id = release_plan_id
        self._release_token = release_token
        self._released = False

    def partition(self) -> pa.Table:
        """Resolve the Ray ObjectRef and return the underlying pa.Table."""
        try:
            if isinstance(self._partition, ray.ObjectRef):
                return resolve_object_refs_blocking(self._partition)
            return self._partition
        finally:
            self.close()

    def partition_ref(self) -> ray.ObjectRef:
        """Return the raw Ray ObjectRef without resolving."""
        return self._partition

    def metadata(self) -> PartitionMetadata:
        if self._metadatas is not None and self._metadata_idx is not None:
            return self._metadatas.get_index(self._metadata_idx)
        return PartitionMetadata(num_rows=0, size_bytes=None)

    def cancel(self) -> None:
        try:
            if isinstance(self._partition, ray.ObjectRef):
                return ray.cancel(self._partition)
            return None
        finally:
            self.close()

    def close(self) -> None:
        if self._released:
            return
        self._released = True
        if self._release_owner is None:
            return
        if self._release_plan_id is None:
            raise TypeError("release_plan_id is required when release_owner is set")
        if self._release_token is None:
            raise TypeError("release_token is required when release_owner is set")
        release_method = self._release_owner.release_result_partition_ref
        remote = release_method.remote
        if not callable(remote):
            raise TypeError("release_owner.release_result_partition_ref.remote must be callable")
        remote(self._release_plan_id, self._release_token)


class PartitionMetadataAccessor:
    """Wrapper class around Remote[List[PartitionMetadata]] to memoize lookups."""

    def __init__(self, ref: ray.ObjectRef) -> None:
        self._ref = ref
        self._metadatas: None | list[PartitionMetadata] = None

    def _get_metadatas(self) -> list[PartitionMetadata]:
        if self._metadatas is None:
            self._metadatas = resolve_object_refs_blocking(self._ref)
        return self._metadatas

    def get_index(self, key: int) -> PartitionMetadata:
        return self._get_metadatas()[key]

    @classmethod
    def from_metadata_list(cls, meta: list[PartitionMetadata]) -> PartitionMetadataAccessor:
        ref = ray.put(meta)
        accessor = cls(ref)
        accessor._metadatas = meta
        return accessor

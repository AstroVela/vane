# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _check_non_negative(name: str, value: int) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True, order=True)
class FteTaskId:
    query_id: str
    fragment_execution_id: int
    partition_id: int

    def __post_init__(self) -> None:
        query_id = str(self.query_id).strip()
        if not query_id:
            raise ValueError("query_id must be non-empty")
        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(
            self, "fragment_execution_id", _check_non_negative("fragment_execution_id", self.fragment_execution_id)
        )
        object.__setattr__(
            self,
            "partition_id",
            _check_non_negative("partition_id", self.partition_id),
        )

    def __str__(self) -> str:
        return f"{self.query_id}.{self.fragment_execution_id}.{self.partition_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "fragment_execution_id": self.fragment_execution_id,
            "partition_id": self.partition_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FteTaskId:
        return cls(
            query_id=str(payload["query_id"]),
            fragment_execution_id=int(payload["fragment_execution_id"]),
            partition_id=int(payload["partition_id"]),
        )

    @classmethod
    def parse(cls, value: str) -> FteTaskId:
        try:
            query_id, fragment_execution_id, partition_id = str(value).rsplit(".", 2)
        except ValueError as exc:
            raise ValueError(f"Invalid FTE task id: {value!r}") from exc
        return cls(query_id, int(fragment_execution_id), int(partition_id))

    @classmethod
    def coerce(cls, value: Any) -> FteTaskId:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls.parse(value)
        if isinstance(value, Mapping):
            if "task_id" in value and isinstance(value["task_id"], Mapping):
                return cls.from_dict(value["task_id"])
            return cls.from_dict(value)
        raise TypeError(f"Cannot coerce {type(value).__name__} to FteTaskId")


@dataclass(frozen=True, order=True)
class FteTaskAttemptId:
    task_id: FteTaskId
    attempt_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", FteTaskId.coerce(self.task_id))
        object.__setattr__(
            self,
            "attempt_id",
            _check_non_negative("attempt_id", self.attempt_id),
        )

    @property
    def query_id(self) -> str:
        return self.task_id.query_id

    @property
    def fragment_execution_id(self) -> int:
        return self.task_id.fragment_execution_id

    @property
    def partition_id(self) -> int:
        return self.task_id.partition_id

    def __str__(self) -> str:
        return f"{self.task_id}.{self.attempt_id}"

    def to_dict(self) -> dict[str, Any]:
        payload = self.task_id.to_dict()
        payload["attempt_id"] = self.attempt_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FteTaskAttemptId:
        if "task_id" in payload and isinstance(payload["task_id"], Mapping):
            task_id = FteTaskId.from_dict(payload["task_id"])
        else:
            task_id = FteTaskId.from_dict(payload)
        return cls(task_id, int(payload["attempt_id"]))

    @classmethod
    def parse(cls, value: str) -> FteTaskAttemptId:
        try:
            query_id, fragment_execution_id, partition_id, attempt_id = str(value).rsplit(".", 3)
        except ValueError as exc:
            raise ValueError(f"Invalid FTE task attempt id: {value!r}") from exc
        return cls(FteTaskId(query_id, int(fragment_execution_id), int(partition_id)), int(attempt_id))

    @classmethod
    def coerce(cls, value: Any) -> FteTaskAttemptId:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls.parse(value)
        if isinstance(value, Mapping):
            return cls.from_dict(value)
        raise TypeError(f"Cannot coerce {type(value).__name__} to FteTaskAttemptId")


def validate_fte_status_identity(
    status: Mapping[str, Any],
    expected: FteTaskAttemptId | str | Mapping[str, Any],
) -> FteTaskAttemptId:
    """Require every identity carried by a worker status to match the request."""
    expected_attempt = FteTaskAttemptId.coerce(expected)
    raw_identities = [
        (field_name, status[field_name])
        for field_name in ("task_id", "task_id_string")
        if status.get(field_name) is not None
    ]
    if not raw_identities:
        raise ValueError(f"FTE status identity missing for expected attempt {expected_attempt}")
    for field_name, raw_identity in raw_identities:
        try:
            actual_attempt = FteTaskAttemptId.coerce(raw_identity)
        except Exception as exc:
            raise ValueError(f"FTE status identity is malformed in {field_name}: {raw_identity!r}") from exc
        if actual_attempt != expected_attempt:
            raise RuntimeError(
                f"FTE status identity mismatch: expected={expected_attempt} actual={actual_attempt} field={field_name}"
            )
    return expected_attempt


@dataclass(frozen=True)
class FteSplit:
    source_node_id: str
    sequence_id: int
    kind: str
    data: Any = None
    source_partition_id: int = 0
    size_bytes: int | None = None
    addresses: tuple[str, ...] = ()
    remotely_accessible: bool = True
    catalog: str | None = None

    def __post_init__(self) -> None:
        source_node_id = str(self.source_node_id).strip()
        if not source_node_id:
            raise ValueError("source_node_id must be non-empty")
        kind = str(self.kind).strip()
        if not kind:
            raise ValueError("split kind must be non-empty")
        object.__setattr__(self, "source_node_id", source_node_id)
        object.__setattr__(self, "sequence_id", _check_non_negative("sequence_id", self.sequence_id))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self,
            "source_partition_id",
            _check_non_negative("source_partition_id", self.source_partition_id),
        )
        if self.size_bytes is not None:
            object.__setattr__(self, "size_bytes", _check_non_negative("size_bytes", self.size_bytes))
        object.__setattr__(self, "addresses", tuple(str(address) for address in self.addresses))
        object.__setattr__(self, "remotely_accessible", bool(self.remotely_accessible))
        if self.catalog is not None:
            object.__setattr__(self, "catalog", str(self.catalog))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "source_node_id": self.source_node_id,
            "sequence_id": self.sequence_id,
            "kind": self.kind,
        }
        if self.data is not None:
            payload["data"] = self.data
        payload["source_partition_id"] = self.source_partition_id
        if self.size_bytes is not None:
            payload["size_bytes"] = self.size_bytes
        if self.addresses:
            payload["addresses"] = list(self.addresses)
        if not self.remotely_accessible:
            payload["remotely_accessible"] = self.remotely_accessible
        if self.catalog is not None:
            payload["catalog"] = self.catalog
        return payload

    @classmethod
    def from_dict(cls, source_node_id: str, payload: Mapping[str, Any]) -> FteSplit:
        return cls(
            source_node_id=str(payload.get("source_node_id", source_node_id)),
            sequence_id=int(payload["sequence_id"]),
            kind=str(payload["kind"]),
            data=payload.get("data"),
            source_partition_id=int(payload.get("source_partition_id", 0)),
            size_bytes=payload.get("size_bytes"),
            addresses=tuple(payload.get("addresses") or ()),
            remotely_accessible=bool(payload.get("remotely_accessible", True)),
            catalog=payload.get("catalog"),
        )

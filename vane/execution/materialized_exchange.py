# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Immutable materialized-exchange identities. Sealing does not select a winner."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from vane.execution.plan import _decode, _encode, _fields, _items, _payload
from vane.execution.resource_demand import _capacity

MATERIALIZED_PROTOCOL = 1
MATERIALIZED_CODEC = "vane.materialized-arrow:1"


def _label(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 256 or "\0" in value:
        raise ValueError(f"invalid {name}")


def _hex(value: str, length: int, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(f"[0-9a-f]{{{length}}}", value) is None:
        raise ValueError(f"invalid {name}")


def _object_key(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32}/[0-9a-f]{32}/[0-9a-f]{32}\.mat", value) is None:
        raise ValueError("invalid materialized object key")


def canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class PartitionSpec:
    exchange_id: str
    partition: int
    schema: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _label(self.exchange_id, "exchange_id")
        _capacity(self.partition, "partition", minimum=0)
        object.__setattr__(self, "schema", _payload(self.schema, "schema"))
        if len(self.schema) > 65536:
            raise ValueError("materialized schema exceeds metadata limit")

    @property
    def identity(self) -> tuple[str, int]:
        return self.exchange_id, self.partition

    def to_dict(self) -> dict[str, Any]:
        return {"exchange_id": self.exchange_id, "partition": self.partition, "schema": _encode(self.schema)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PartitionSpec:
        _fields(value, {"exchange_id", "partition", "schema"}, cls.__name__)
        return cls(value["exchange_id"], value["partition"], _decode(value["schema"], "schema"))


@dataclass(frozen=True)
class ObjectMeta:
    bytes: int
    rows: int
    frames: int
    sha256: str

    def __post_init__(self) -> None:
        _capacity(self.bytes, "object bytes", minimum=40)
        _capacity(self.rows, "rows", minimum=0)
        _capacity(self.frames, "frames", minimum=0)
        _hex(self.sha256, 64, "object checksum")
        if self.bytes > 1 << 40 or self.frames > self.rows or (self.frames == 0) != (self.rows == 0):
            raise ValueError("invalid materialized object counts")

    def to_dict(self) -> dict[str, Any]:
        return {"bytes": self.bytes, "rows": self.rows, "frames": self.frames, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ObjectMeta:
        _fields(value, {"bytes", "rows", "frames", "sha256"}, cls.__name__)
        return cls(**value)


@dataclass(frozen=True)
class OutputObject:
    output: PartitionSpec
    key: str
    metadata: ObjectMeta

    def __post_init__(self) -> None:
        if not isinstance(self.output, PartitionSpec) or not isinstance(self.metadata, ObjectMeta):
            raise ValueError("invalid materialized output")
        _object_key(self.key)

    def to_dict(self) -> dict[str, Any]:
        return {"output": self.output.to_dict(), "key": self.key, "metadata": self.metadata.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OutputObject:
        _fields(value, {"output", "key", "metadata"}, cls.__name__)
        return cls(PartitionSpec.from_dict(value["output"]), value["key"], ObjectMeta.from_dict(value["metadata"]))


@dataclass(frozen=True)
class AttemptToken:
    query_id: str
    stage_id: str
    task_id: str
    attempt: int
    fence: str
    worker_epoch: str
    input_id: str

    def __post_init__(self) -> None:
        for name in ("query_id", "stage_id", "task_id", "worker_epoch"):
            _label(getattr(self, name), name)
        _capacity(self.attempt, "attempt")
        _hex(self.fence, 32, "attempt fence")
        _hex(self.input_id, 64, "immutable input identity")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AttemptToken:
        _fields(value, set(cls.__dataclass_fields__), cls.__name__)
        return cls(**value)


@dataclass(frozen=True)
class AttemptManifest:
    engine_identity: str
    token: AttemptToken
    objects: tuple[OutputObject, ...]

    def __post_init__(self) -> None:
        _label(self.engine_identity, "engine_identity")
        if not isinstance(self.token, AttemptToken):
            raise ValueError("invalid attempt token")
        objects = _items(self.objects, "objects")
        if not 1 <= len(objects) <= 4096 or any(not isinstance(item, OutputObject) for item in objects):
            raise ValueError("attempt must seal between 1 and 4096 partition objects")
        if len({o.output.identity for o in objects}) != len(objects) or len({o.key for o in objects}) != len(objects):
            raise ValueError("duplicate materialized output partition or object")
        object.__setattr__(self, "objects", tuple(sorted(objects, key=lambda o: o.output.identity)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": MATERIALIZED_PROTOCOL,
            "codec": MATERIALIZED_CODEC,
            "engine_identity": self.engine_identity,
            "token": self.token.to_dict(),
            "objects": [o.to_dict() for o in self.objects],
        }

    @property
    def identity(self) -> str:
        return fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AttemptManifest:
        _fields(value, {"protocol", "codec", "engine_identity", "token", "objects"}, cls.__name__)
        if (
            type(value["protocol"]) is not int
            or value["protocol"] != MATERIALIZED_PROTOCOL
            or value["codec"] != MATERIALIZED_CODEC
        ):
            raise ValueError("unsupported materialized manifest protocol or codec")
        return cls(
            value["engine_identity"],
            AttemptToken.from_dict(value["token"]),
            tuple(OutputObject.from_dict(o) for o in _items(value["objects"], "objects")),
        )


@dataclass(frozen=True)
class StageManifest:
    query_id: str
    stage_id: str
    engine_identity: str
    attempts: tuple[AttemptManifest, ...]

    def __post_init__(self) -> None:
        for name in ("query_id", "stage_id", "engine_identity"):
            _label(getattr(self, name), name)
        attempts = _items(self.attempts, "attempts")
        if not 1 <= len(attempts) <= 4096 or any(not isinstance(a, AttemptManifest) for a in attempts):
            raise ValueError("stage must contain between 1 and 4096 committed tasks")
        if len({a.token.task_id for a in attempts}) != len(attempts):
            raise ValueError("duplicate task in stage")
        if any(
            (a.token.query_id, a.token.stage_id, a.engine_identity)
            != (self.query_id, self.stage_id, self.engine_identity)
            for a in attempts
        ):
            raise ValueError("stage manifest identity mismatch")
        object.__setattr__(self, "attempts", tuple(sorted(attempts, key=lambda a: a.token.task_id)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": MATERIALIZED_PROTOCOL,
            "query_id": self.query_id,
            "stage_id": self.stage_id,
            "engine_identity": self.engine_identity,
            "attempts": [a.to_dict() for a in self.attempts],
        }

    @property
    def identity(self) -> str:
        return fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StageManifest:
        _fields(value, {"protocol", "query_id", "stage_id", "engine_identity", "attempts"}, cls.__name__)
        if type(value["protocol"]) is not int or value["protocol"] != MATERIALIZED_PROTOCOL:
            raise ValueError("unsupported stage manifest protocol")
        return cls(
            value["query_id"],
            value["stage_id"],
            value["engine_identity"],
            tuple(AttemptManifest.from_dict(a) for a in _items(value["attempts"], "attempts")),
        )


@dataclass(frozen=True)
class MaterializedTask:
    task_id: str
    stage_id: str
    input_id: str
    outputs: tuple[PartitionSpec, ...]

    def __post_init__(self) -> None:
        _label(self.task_id, "task_id")
        _label(self.stage_id, "stage_id")
        _hex(self.input_id, 64, "immutable input identity")
        outputs = _items(self.outputs, "outputs")
        if not 1 <= len(outputs) <= 4096 or any(not isinstance(o, PartitionSpec) for o in outputs):
            raise ValueError("task must declare between 1 and 4096 output partitions")
        if len({o.identity for o in outputs}) != len(outputs):
            raise ValueError("duplicate task output partition")
        object.__setattr__(self, "outputs", tuple(sorted(outputs, key=lambda o: o.identity)))

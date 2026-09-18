# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit, credential-free configuration for one table and its destinations."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(encode(value).encode("utf-8")).hexdigest()


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be a nonempty string without NUL bytes")
    return value


def _keys(value: Any, allowed: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.keys() - allowed:
        raise ValueError(f"Invalid {label} fields; allowed fields: {sorted(allowed)}")
    return value


@dataclass(frozen=True)
class Target:
    name: str
    kind: str
    command: tuple[str, ...]
    cwd: str
    destination: str


@dataclass(frozen=True)
class Config:
    name: str
    catalog: str
    table: tuple[str, ...]
    branch: str
    id_column: str
    content_column: str
    title_column: str | None
    uri_column: str | None
    state_dir: Path
    targets: tuple[Target, ...]
    poll_seconds: float = 30
    timeout_seconds: float = 600
    max_document_bytes: int = 65536

    @property
    def identity(self) -> str:
        value = asdict(self)
        for key in ("state_dir", "poll_seconds", "timeout_seconds"):
            value.pop(key)
        return digest(value)

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(c for c in (self.id_column, self.content_column, self.title_column, self.uri_column) if c)
        )

    @classmethod
    def load(cls, path: Path) -> Config:
        path = path.resolve()
        raw = _keys(json.loads(path.read_text()), set(cls.__dataclass_fields__), "configuration")
        for key in ("name", "catalog", "branch", "id_column", "content_column"):
            raw[key] = _text(raw.get(key, "main" if key == "branch" else None), key)
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", raw["name"]):
            raise ValueError("name must be a lowercase identifier, at most 64 characters")
        for key in ("title_column", "uri_column"):
            raw[key] = _text(raw[key], key) if raw.get(key) is not None else None
        table = raw.get("table")
        if not isinstance(table, list) or not table:
            raise ValueError("table must be a nonempty array of namespace and table components")
        raw["table"] = tuple(_text(c, "table component") for c in table)
        raw["state_dir"] = (path.parent / _text(raw.get("state_dir"), "state_dir")).absolute()
        targets = raw.get("targets")
        if not isinstance(targets, list) or not targets:
            raise ValueError("targets must be a nonempty array")
        parsed = []
        for value in targets:
            value = _keys(value, {"name", "kind", "command", "cwd", "destination"}, "target")
            for key in ("name", "kind", "cwd", "destination"):
                _text(value.get(key), f"target {key}")
            if value["kind"] not in {"openwiki", "gbrain"}:
                raise ValueError("target kind must be openwiki or gbrain")
            command = value.get("command", [value["kind"]])
            if not isinstance(command, list) or not command:
                raise ValueError("target command must be a nonempty argv array")
            value["command"] = tuple(_text(c, "command argument") for c in command)
            value["cwd"] = str((path.parent / value["cwd"]).resolve(strict=True))
            if not Path(value["cwd"]).is_dir():
                raise ValueError("target cwd must be a directory")
            if value["kind"] == "openwiki" and not re.fullmatch(
                r"custom-mcp-[A-Za-z0-9][A-Za-z0-9._-]{0,108}", value["destination"]
            ):
                raise ValueError("OpenWiki destination must name one custom-mcp-<name> source instance")
            if value["destination"].startswith("-") or value["destination"] == "__all__":
                raise ValueError("destination must name one source")
            parsed.append(Target(**value))
        if len({t.name for t in parsed}) != len(parsed):
            raise ValueError("target names must be unique")
        raw["targets"] = tuple(parsed)
        for key, default in (("poll_seconds", 30), ("timeout_seconds", 600)):
            val = raw.get(key, default)
            if type(val) not in (int, float) or not math.isfinite(val) or val <= 0:
                raise ValueError(f"{key} must be finite and positive")
            raw[key] = float(val)
        size = raw.get("max_document_bytes", 65536)
        if type(size) is not int or not 1 <= size <= 65536:
            raise ValueError("max_document_bytes must be between 1 and 65536")
        return cls(**raw)

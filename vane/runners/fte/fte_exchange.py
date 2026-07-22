# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vane.runners.fte.fte_types import _check_non_negative

if TYPE_CHECKING:
    import os


@dataclass(frozen=True)
class ExchangeSinkHandle:
    query_id: str
    exchange_id: str
    partition_id: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "exchange_id": self.exchange_id,
            "partition_id": self.partition_id,
        }


@dataclass(frozen=True)
class ExchangeSinkInstanceHandle:
    sink_handle: ExchangeSinkHandle
    attempt_id: int
    attempt_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "sink_handle": self.sink_handle.to_dict(),
            "attempt_id": self.attempt_id,
        }
        if self.attempt_path is not None:
            payload["attempt_path"] = self.attempt_path
        return payload


@dataclass(frozen=True)
class ExchangeSourceHandle:
    sink_handle: ExchangeSinkHandle
    attempt_id: int
    attempt_path: str | None = None
    files: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "sink_handle": self.sink_handle.to_dict(),
            "attempt_id": self.attempt_id,
        }
        if self.attempt_path is not None:
            payload["attempt_path"] = self.attempt_path
        if self.files:
            payload["files"] = list(self.files)
        return payload


def _sink_instance_payload(exchange_sink_instance: Any) -> dict[str, Any]:
    if exchange_sink_instance is None:
        return {}
    if hasattr(exchange_sink_instance, "to_dict"):
        return dict(exchange_sink_instance.to_dict())
    if isinstance(exchange_sink_instance, Mapping):
        return dict(exchange_sink_instance)
    return {}


def derive_exchange_sink_instance_for_attempt(
    exchange_sink_instance: Any,
    attempt_id: int,
    task_partition_id: int | None = None,
) -> Any:
    payload = _sink_instance_payload(exchange_sink_instance)
    if not payload:
        return exchange_sink_instance
    attempt_id = _check_non_negative("attempt_id", attempt_id)
    derived = dict(payload)
    derived["attempt_id"] = attempt_id
    preserve_plan_sink_partition = bool(derived.get("preserve_plan_exchange_sink_instance"))
    if task_partition_id is not None:
        task_partition_id = _check_non_negative("task_partition_id", task_partition_id)
        if not preserve_plan_sink_partition:
            derived["task_partition_id"] = task_partition_id
            derived["partition_id"] = task_partition_id
            sink_handle = dict(derived.get("sink_handle") or {})
            sink_handle["task_partition_id"] = task_partition_id
            sink_handle["partition_id"] = task_partition_id
            derived["sink_handle"] = sink_handle

    location = derived.get("output_location") or derived.get("attempt_path")
    if location is not None:
        location_text = str(location)
        if task_partition_id is not None and not preserve_plan_sink_partition:
            replaced = re.sub(
                r"(__sink_)\d+(__attempt_)\d+$",
                rf"\g<1>{task_partition_id}\g<2>{attempt_id}",
                location_text,
            )
        else:
            replaced = re.sub(r"(__attempt_)\d+$", rf"\g<1>{attempt_id}", location_text)
        if replaced != location_text:
            derived["output_location"] = replaced
            if "attempt_path" in derived:
                derived["attempt_path"] = replaced
        elif "attempt_path" in derived and "output_location" not in derived:
            derived["output_location"] = location_text

    return derived


def _partition_key_from_arrow_path(path: Path, root: Path) -> str:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    for component in reversed(parts):
        if not component.startswith("partition_"):
            continue
        token = component[len("partition_") :].split("_", 1)[0]
        if token.isdigit():
            return token
    return "unknown"


def collect_spooling_output_stats(exchange_sink_instance: Any) -> dict[str, Any] | None:
    payload = _sink_instance_payload(exchange_sink_instance)
    if not payload:
        return None
    location = payload.get("attempt_path") or payload.get("output_location")
    if not location:
        return None

    explicit_attempt_path = "attempt_path" in payload
    attempt_path = Path(str(location))
    if not explicit_attempt_path and not attempt_path.is_absolute() and not attempt_path.exists():
        return None

    stats: dict[str, Any] = {
        "attempt_path": str(attempt_path),
        "exists": attempt_path.exists(),
        "committed": (attempt_path / SpoolingExchangeManager.COMMITTED_MARKER).exists()
        if attempt_path.exists()
        else False,
        "aborted": (attempt_path / SpoolingExchangeManager.ABORTED_MARKER).exists() if attempt_path.exists() else False,
        "file_count": 0,
        "total_bytes": 0,
        "files": [],
        "partitions": {},
    }
    if not attempt_path.exists():
        return stats if explicit_attempt_path else None

    manifest_path = attempt_path / SpoolingExchangeManager.MANIFEST_FILE
    manifest_files: list[Any] = []
    if manifest_path.exists():
        stats["manifest_path"] = str(manifest_path)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8") or "{}")
            if isinstance(manifest, Mapping):
                manifest_files = list(manifest.get("files") or [])
        except Exception as exc:
            stats["manifest_error"] = f"{type(exc).__name__}: {exc}"

    candidate_paths: list[Path] = []
    if manifest_files:
        for entry in manifest_files:
            path = Path(str(entry))
            candidate_paths.append(path if path.is_absolute() else attempt_path / path)
    else:
        candidate_paths = [
            path for path in attempt_path.rglob("*.arrow") if path.is_file() and path.name != "schema.arrow"
        ]

    file_entries: list[dict[str, Any]] = []
    partitions: dict[str, dict[str, int]] = {}
    total_bytes = 0
    for path in sorted(set(candidate_paths), key=lambda p: str(p)):
        if not path.exists() or not path.is_file():
            continue
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = 0
        partition_key = _partition_key_from_arrow_path(path, attempt_path)
        total_bytes += size_bytes
        file_entries.append(
            {
                "path": str(path),
                "name": path.name,
                "partition_id": partition_key,
                "size_bytes": size_bytes,
            }
        )
        partition = partitions.setdefault(
            partition_key,
            {"file_count": 0, "total_bytes": 0},
        )
        partition["file_count"] += 1
        partition["total_bytes"] += size_bytes

    stats["file_count"] = len(file_entries)
    stats["total_bytes"] = total_bytes
    stats["files"] = file_entries
    stats["partitions"] = partitions
    return stats


class FteExchangeSourceOutputSelector:
    """Selected-attempt state for an exchange source."""

    def __init__(self) -> None:
        self._selected_attempts: dict[int, int] = {}
        self._finished_attempts: dict[int, set[int]] = {}
        self._aborted_attempts: dict[int, set[int]] = {}
        self._final = False
        self._final_required_partition_ids: frozenset[int] | None = None

    def record_finished(self, partition_id: int, attempt_id: int) -> bool:
        partition_id = _check_non_negative("partition_id", partition_id)
        attempt_id = _check_non_negative("attempt_id", attempt_id)
        if self._final:
            if self._final_required_partition_ids is None or partition_id not in self._final_required_partition_ids:
                raise RuntimeError("cannot record a finished attempt after selector is final")
            self._finished_attempts.setdefault(partition_id, set()).add(attempt_id)
            return False
        self._finished_attempts.setdefault(partition_id, set()).add(attempt_id)
        if partition_id in self._selected_attempts:
            return False
        self._selected_attempts[partition_id] = attempt_id
        return True

    def record_aborted(self, partition_id: int, attempt_id: int) -> None:
        if self._final:
            return
        partition_id = _check_non_negative("partition_id", partition_id)
        attempt_id = _check_non_negative("attempt_id", attempt_id)
        self._aborted_attempts.setdefault(partition_id, set()).add(attempt_id)

    def selected_attempt(self, partition_id: int) -> int | None:
        return self._selected_attempts.get(_check_non_negative("partition_id", partition_id))

    def is_final(self) -> bool:
        return self._final

    def try_mark_final(self, required_partition_ids: set[int] | list[int] | tuple[int, ...]) -> bool:
        required = {_check_non_negative("partition_id", partition_id) for partition_id in required_partition_ids}
        if not required:
            if self._selected_attempts:
                raise RuntimeError("cannot mark selector final with unrequired selected partitions")
            return False
        final_required = frozenset(required)
        if self._final:
            if self._final_required_partition_ids != final_required:
                raise RuntimeError("cannot mark selector final with different required partitions")
            return False
        if set(self._selected_attempts) - required:
            raise RuntimeError("cannot mark selector final with unrequired selected partitions")
        if not required.issubset(self._selected_attempts):
            return False
        self._final = True
        self._final_required_partition_ids = final_required
        return True


class FteExchangeTracker:
    """Coordinator-side selected-attempt tracker for tests and FTE wiring."""

    def __init__(self, query_id: str, exchange_id: str) -> None:
        self.query_id = str(query_id)
        self.exchange_id = str(exchange_id)
        self._sinks: dict[int, ExchangeSinkHandle] = {}
        self._finished_attempts: dict[int, set[int]] = {}
        self._aborted_attempts: dict[int, set[int]] = {}
        self._selected_attempts: dict[int, int] = {}
        self.output_selector = FteExchangeSourceOutputSelector()

    def add_sink(self, partition_id: int) -> ExchangeSinkHandle:
        partition_id = _check_non_negative("partition_id", partition_id)
        handle = self._sinks.get(partition_id)
        if handle is None:
            if self.output_selector.is_final():
                raise RuntimeError("cannot add exchange sink after selector is final")
            handle = ExchangeSinkHandle(self.query_id, self.exchange_id, partition_id)
            self._sinks[partition_id] = handle
        return handle

    def instantiate_sink(
        self,
        sink_handle: ExchangeSinkHandle,
        attempt_id: int,
    ) -> ExchangeSinkInstanceHandle:
        self._require_sink(sink_handle)
        return ExchangeSinkInstanceHandle(
            sink_handle,
            _check_non_negative("attempt_id", attempt_id),
        )

    def sink_finished(self, sink_handle: ExchangeSinkHandle, attempt_id: int) -> None:
        self._require_sink(sink_handle)
        attempt_id = _check_non_negative("attempt_id", attempt_id)
        partition_id = sink_handle.partition_id
        selected = self.output_selector.record_finished(partition_id, attempt_id)
        self._finished_attempts.setdefault(partition_id, set()).add(attempt_id)
        if selected:
            self._selected_attempts[partition_id] = attempt_id

    def sink_aborted(self, sink_handle: ExchangeSinkHandle, attempt_id: int) -> None:
        self._require_sink(sink_handle)
        self._aborted_attempts.setdefault(sink_handle.partition_id, set()).add(
            _check_non_negative("attempt_id", attempt_id)
        )
        self.output_selector.record_aborted(sink_handle.partition_id, attempt_id)

    def selected_attempt(self, sink_handle: ExchangeSinkHandle) -> int | None:
        self._require_sink(sink_handle)
        return self._selected_attempts.get(sink_handle.partition_id)

    def finalize(self) -> bool:
        return self.output_selector.try_mark_final(set(self._sinks))

    def is_final(self) -> bool:
        return self.output_selector.is_final()

    def get_source_handles(self) -> list[ExchangeSourceHandle]:
        return [
            ExchangeSourceHandle(self._sinks[partition_id], attempt_id)
            for partition_id, attempt_id in sorted(self._selected_attempts.items())
        ]

    def _require_sink(self, sink_handle: ExchangeSinkHandle) -> None:
        existing = self._sinks.get(sink_handle.partition_id)
        if existing != sink_handle:
            raise KeyError(f"unknown exchange sink: {sink_handle}")


class SpoolingExchangeManager(FteExchangeTracker):
    MANIFEST_FILE = "manifest.json"
    COMMITTED_MARKER = "committed"
    ABORTED_MARKER = "aborted"

    def __init__(self, base_dir: str | os.PathLike[str], query_id: str, exchange_id: str) -> None:
        super().__init__(query_id, exchange_id)
        self.base_dir = Path(base_dir)
        self.query_root = self.base_dir / self.query_id
        self.exchange_root = self.query_root / self.exchange_id
        self.exchange_root.mkdir(parents=True, exist_ok=True)

    def add_sink(self, partition_id: int) -> ExchangeSinkHandle:
        handle = super().add_sink(partition_id)
        self._sink_dir(handle).mkdir(parents=True, exist_ok=True)
        return handle

    def instantiate_sink(
        self,
        sink_handle: ExchangeSinkHandle,
        attempt_id: int,
    ) -> ExchangeSinkInstanceHandle:
        self._require_sink(sink_handle)
        attempt_id = _check_non_negative("attempt_id", attempt_id)
        attempt_dir = self._attempt_dir(sink_handle, attempt_id)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        return ExchangeSinkInstanceHandle(
            sink_handle,
            attempt_id,
            attempt_path=str(attempt_dir),
        )

    def record_output_file(
        self,
        instance_handle: ExchangeSinkInstanceHandle,
        output_partition_id: int,
        file_id: str | int,
        data: bytes = b"",
    ) -> str:
        self._require_sink(instance_handle.sink_handle)
        output_partition_id = _check_non_negative("output_partition_id", output_partition_id)
        file_id_text = str(file_id)
        if "/" in file_id_text or "\\" in file_id_text:
            raise ValueError("file_id must not contain path separators")
        attempt_dir = self._attempt_dir(instance_handle.sink_handle, instance_handle.attempt_id)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        path = attempt_dir / f"partition_{output_partition_id}_{file_id_text}.arrow"
        path.write_bytes(data)
        return str(path)

    def finish_attempt(self, instance_handle: ExchangeSinkInstanceHandle) -> None:
        self._require_sink(instance_handle.sink_handle)
        attempt_dir = self._attempt_dir(instance_handle.sink_handle, instance_handle.attempt_id)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(path.name for path in attempt_dir.glob("partition_*_*.arrow"))
        manifest = {
            "sink": instance_handle.sink_handle.to_dict(),
            "attempt_id": instance_handle.attempt_id,
            "files": files,
        }
        (attempt_dir / self.MANIFEST_FILE).write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        (attempt_dir / self.COMMITTED_MARKER).write_text("", encoding="utf-8")

    def sink_finished(self, sink_handle: ExchangeSinkHandle, attempt_id: int) -> None:
        super().sink_finished(sink_handle, attempt_id)
        attempt_dir = self._attempt_dir(sink_handle, attempt_id)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        if not (attempt_dir / self.MANIFEST_FILE).exists():
            self.finish_attempt(ExchangeSinkInstanceHandle(sink_handle, attempt_id, str(attempt_dir)))
        (attempt_dir / self.COMMITTED_MARKER).touch()

    def sink_aborted(self, sink_handle: ExchangeSinkHandle, attempt_id: int) -> None:
        super().sink_aborted(sink_handle, attempt_id)
        attempt_dir = self._attempt_dir(sink_handle, attempt_id)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / self.ABORTED_MARKER).write_text("", encoding="utf-8")

    def get_source_handles(self) -> list[ExchangeSourceHandle]:
        handles: list[ExchangeSourceHandle] = []
        for partition_id, attempt_id in sorted(self._selected_attempts.items()):
            sink_handle = self._sinks[partition_id]
            attempt_dir = self._attempt_dir(sink_handle, attempt_id)
            manifest = self._read_manifest(attempt_dir)
            files = tuple(str(attempt_dir / name) for name in manifest.get("files", []))
            handles.append(
                ExchangeSourceHandle(
                    sink_handle=sink_handle,
                    attempt_id=attempt_id,
                    attempt_path=str(attempt_dir),
                    files=files,
                )
            )
        return handles

    def cleanup_unselected_attempts(self) -> int:
        removed = 0
        for partition_id, sink_handle in list(self._sinks.items()):
            selected_attempt = self._selected_attempts.get(partition_id)
            sink_dir = self._sink_dir(sink_handle)
            if not sink_dir.exists():
                continue
            for attempt_dir in sink_dir.glob("attempt_*"):
                if not attempt_dir.is_dir():
                    continue
                try:
                    attempt_id = int(attempt_dir.name.split("_", 1)[1])
                except (IndexError, ValueError):
                    continue
                if selected_attempt is not None and attempt_id == selected_attempt:
                    continue
                shutil.rmtree(attempt_dir)
                removed += 1
        return removed

    def close(self) -> None:
        self.cleanup_unselected_attempts()

    def destroy_query(self) -> None:
        if self.query_root.exists():
            shutil.rmtree(self.query_root)

    def _sink_dir(self, sink_handle: ExchangeSinkHandle) -> Path:
        return self.exchange_root / f"sink_{sink_handle.partition_id}"

    def _attempt_dir(self, sink_handle: ExchangeSinkHandle, attempt_id: int) -> Path:
        return self._sink_dir(sink_handle) / f"attempt_{_check_non_negative('attempt_id', attempt_id)}"

    def _read_manifest(self, attempt_dir: Path) -> dict[str, Any]:
        manifest_path = attempt_dir / self.MANIFEST_FILE
        if not manifest_path.exists():
            return {"files": sorted(path.name for path in attempt_dir.glob("partition_*_*.arrow"))}
        raw = manifest_path.read_text(encoding="utf-8")
        payload = json.loads(raw) if raw else {}
        if not isinstance(payload, dict):
            raise ValueError(f"invalid spooling manifest: {manifest_path}")
        return payload

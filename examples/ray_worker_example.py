from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SimpleResultPart:
    object_ref: dict[str, Any]

    def get_num_rows(self) -> int:
        return 1

    def get_size_bytes(self) -> int:
        return len(repr(self.object_ref).encode())

    def get_object_ref(self) -> dict[str, Any]:
        return self.object_ref


class SimpleTaskHandle:
    def __init__(self, result_parts: list[SimpleResultPart]) -> None:
        self._result_parts = result_parts

    async def get_result(self) -> tuple[str, list[SimpleResultPart], bytes]:
        return ("Success", self._result_parts, b"")


class SimpleWorker:
    def submit_task(self, task: dict[str, Any]) -> SimpleTaskHandle:
        return SimpleTaskHandle([SimpleResultPart(task)])


__all__ = ["SimpleResultPart", "SimpleTaskHandle", "SimpleWorker"]

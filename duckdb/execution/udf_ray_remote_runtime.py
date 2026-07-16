from __future__ import annotations

from typing import Any

from duckdb.runners.ray.safe_get import resolve_object_refs_blocking


class RemoteUDFRuntimeMixin:
    def _resolve_object_ref(self, ref: Any) -> Any:
        return resolve_object_refs_blocking(ref)


__all__ = [name for name in globals() if not name.startswith("__")]

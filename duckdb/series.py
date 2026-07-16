"""Minimal Series shim used by RecordBatch and MicroPartition helpers."""

from __future__ import annotations

from typing import Any


class Series:
    def __init__(self, values: list[Any]):
        self._values = list(values)

    @classmethod
    def _from_pyseries(cls, pyseries: Any) -> Series:
        # In real code pyseries is an extension type; here we wrap iterables
        try:
            if hasattr(pyseries, "to_pylist"):
                values = pyseries.to_pylist()
            elif hasattr(pyseries, "to_list"):
                values = pyseries.to_list()
            elif hasattr(pyseries, "to_arrow"):
                values = pyseries.to_arrow().to_pylist()
            else:
                values = list(pyseries)
            s = cls(list(values))
        except Exception:
            s = cls([pyseries])
        # Preserve the original low-level series object for places in the
        # codebase that expect `. _series` to exist (e.g., RecordBatch).
        s._series = pyseries
        return s

    def __len__(self) -> int:
        return len(self._values)

    def to_list(self) -> list[Any]:
        return list(self._values)

    def to_pylist(self) -> list[Any]:
        # Keep compatibility with pyarrow/duckdb series APIs.
        return self.to_list()

    def name(self) -> str:
        # Attempt to extract name from underlying series if available
        try:
            n = getattr(self._series, "name", None)
            if callable(n):
                return n()
            if n is not None:
                return n
            return ""
        except Exception:
            return ""

    def to_arrow(self):
        # Convert to a pyarrow.Array where possible
        try:
            import pyarrow as pa

            if hasattr(self, "_series") and hasattr(self._series, "to_arrow"):
                arr = self._series.to_arrow()
                # Normalize ChunkedArray -> Array for pyarrow.from_pydict expectations
                if hasattr(arr, "combine_chunks"):
                    return arr.combine_chunks()
                return arr
            # fall back to converting python list values
            return pa.array(self._values)
        except Exception:
            return self._values


def item_to_series(*args) -> Series:
    # Accept either (item) or (name, item) to match caller patterns.
    if len(args) == 1:
        item = args[0]
    else:
        item = args[1]
    if isinstance(item, Series):
        return item
    # Wrap arbitrary Python objects into a Series, preserving the original
    # low-level object on the returned Series via _from_pyseries so callers
    # that expect a `_series` attribute (e.g., RecordBatch) continue to work.
    return Series._from_pyseries(item)


__all__ = ["Series", "item_to_series"]

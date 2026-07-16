from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from os import PathLike


def read_committed_copy_direct_write_parquet(
    base_path: str | PathLike[str],
    run_id: str,
    *,
    conn: Any | None = None,
    **read_parquet_kwargs: Any,
) -> Any:
    """Read a direct-write COPY result through its committed manifest.

    This is the manifest-aware reader boundary for Vane direct-write COPY:
    uncommitted runs and files not selected by the committed manifest remain
    invisible even if they physically exist under the run prefix.
    """
    import duckdb

    if conn is None:
        conn = duckdb.connect()

    result = duckdb.ray_cxx.read_committed_copy_direct_write_result(str(base_path), run_id)
    files = [entry["final_path"] for entry in result["files"]]
    if not files:
        raise ValueError(
            "committed direct-write COPY result contains no parquet files; cannot infer a schema for read_parquet"
        )
    return conn.read_parquet(files, **read_parquet_kwargs)

# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
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
    import vane

    if conn is None:
        conn = vane.connect()

    result = vane.ray_cxx.read_committed_copy_direct_write_result(str(base_path), run_id, conn)
    files = [entry["final_path"] for entry in result["files"]]
    if not files:
        raise ValueError(
            "committed direct-write COPY result contains no parquet files; cannot infer a schema for read_parquet"
        )
    return conn.read_parquet(files, **read_parquet_kwargs)


def inspect_copy_direct_write_run(
    base_path: str | PathLike[str],
    run_id: str,
    *,
    conn: Any | None = None,
) -> dict[str, Any]:
    """Inspect a direct-write COPY run without changing its storage state.

    ``UNCOMMITTED`` is an observation, not proof that a prior marker write
    failed. A successful shared-storage
    ``force_abort_copy_direct_write_run`` can return ``safe_to_retry=True``.
    """
    import vane

    result = vane.ray_cxx.inspect_copy_direct_write_run(str(base_path), run_id, conn)
    if not isinstance(result, Mapping):
        raise TypeError("inspect_copy_direct_write_run() must return a mapping")
    return dict(result)


def force_abort_copy_direct_write_run(
    base_path: str | PathLike[str],
    run_id: str,
    *,
    conn: Any | None = None,
) -> dict[str, Any]:
    """Discard one direct-write COPY run after an operator chooses data loss.

    The native operation is available only when worker output is on shared
    remote storage. It removes the committed marker first, deletes only output
    owned by ``run_id``, removes its metadata, and verifies the marker remains
    absent. Node-local output requires cleanup on every worker and is rejected
    without changing its marker or lifecycle record. A raised exception means
    the run is not safe to retry. This is operator-driven recovery, not an
    exactly-once fence: call it only after the originating COPY has terminated
    and do not race another writer.
    """
    import vane

    result = vane.ray_cxx.force_abort_copy_direct_write_run(str(base_path), run_id, conn)
    if not isinstance(result, Mapping):
        raise TypeError("force_abort_copy_direct_write_run() must return a mapping")
    return dict(result)

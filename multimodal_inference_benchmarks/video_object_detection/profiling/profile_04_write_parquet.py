from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

try:
    from .profile_common import add_synthetic_output_options
    from .profile_common import build_common_parser
    from .profile_common import directory_size_bytes
    from .profile_common import make_synthetic_output_table
    from .profile_common import parquet_file_count
    from .profile_common import timed_block
    from .profile_common import write_json_result
except ImportError:
    from profile_common import add_synthetic_output_options
    from profile_common import build_common_parser
    from profile_common import directory_size_bytes
    from profile_common import make_synthetic_output_table
    from profile_common import parquet_file_count
    from profile_common import timed_block
    from profile_common import write_json_result


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _profile_ray_write(table, output_path: Path) -> dict[str, Any]:
    import ray
    import ray.data

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    _clean_dir(output_path)
    ds = ray.data.from_arrow(table)
    with timed_block() as wall:
        ds.write_parquet(str(output_path))
    return {
        "wall_s": wall.elapsed_s,
        "rows_per_s": table.num_rows / wall.elapsed_s if wall.elapsed_s > 0 else 0.0,
        "bytes": directory_size_bytes(output_path),
        "parquet_files": parquet_file_count(output_path),
        "output_path": str(output_path),
    }


def _profile_vane_write(table, output_path: Path) -> dict[str, Any]:
    import vane

    _clean_dir(output_path)
    con = vane.connect(config={"local_exchange_streaming": "true"})
    try:
        con.execute("SET preserve_insertion_order=false")
        rel = con.from_arrow(table)
        with timed_block() as wall:
            rel.write_parquet(str(output_path), per_thread_output=True)
    finally:
        con.close()
    return {
        "wall_s": wall.elapsed_s,
        "rows_per_s": table.num_rows / wall.elapsed_s if wall.elapsed_s > 0 else 0.0,
        "bytes": directory_size_bytes(output_path),
        "parquet_files": parquet_file_count(output_path),
        "output_path": str(output_path),
    }


def main() -> None:
    parser = build_common_parser("Profile terminal Parquet write in isolation.")
    add_synthetic_output_options(parser)
    args = parser.parse_args()

    table_start = time.perf_counter()
    table = make_synthetic_output_table(rows=args.rows, object_bytes=args.object_bytes)
    table_build_s = time.perf_counter() - table_start
    output_root = args.output_dir.expanduser().resolve()
    payload = {
        "stage": "write_parquet",
        "rows": table.num_rows,
        "object_bytes": args.object_bytes,
        "table_build_s": table_build_s,
        "ray_data": _profile_ray_write(table, output_root / "ray_data_write"),
        "vane": _profile_vane_write(table, output_root / "vane_write"),
    }
    write_json_result(payload, output_json=args.output_json, quiet=args.quiet)


if __name__ == "__main__":
    main()

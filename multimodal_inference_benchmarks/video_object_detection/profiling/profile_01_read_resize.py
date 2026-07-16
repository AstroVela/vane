from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

try:
    from .profile_common import add_input_options
    from .profile_common import build_common_parser
    from .profile_common import ray_data_paths
    from .profile_common import resolve_video_files_for_args
    from .profile_common import summarize_values
    from .profile_common import timed_block
    from .profile_common import vane_frame_shape
    from .profile_common import write_json_result
except ImportError:
    from profile_common import add_input_options
    from profile_common import build_common_parser
    from profile_common import ray_data_paths
    from profile_common import resolve_video_files_for_args
    from profile_common import summarize_values
    from profile_common import timed_block
    from profile_common import vane_frame_shape
    from profile_common import write_json_result


_READER_TIMING_PREFIX = "[vane_video][reader_timing]"
_READER_TIMING_FIELD_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^ ]+)")
_READER_TIMING_STAGE_KEYS = ("open_s", "decode_s", "resize_s", "flush_s", "total_s")


class RayResizeFrameProfiler:
    def __init__(self, image_height: int, image_width: int):
        self.image_height = image_height
        self.image_width = image_width

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        import numpy as np
        from PIL import Image

        total_start = time.perf_counter()
        stage_start = time.perf_counter()
        pil_image = Image.fromarray(row["frame"])
        image_fromarray_s = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        resized_pil = pil_image.resize((self.image_height, self.image_width))
        resize_s = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        resized_frame = np.array(resized_pil)
        numpy_array_s = time.perf_counter() - stage_start

        row["frame"] = resized_frame
        row["frame_bytes"] = resized_frame.nbytes
        row["image_fromarray_s"] = image_fromarray_s
        row["resize_s"] = resize_s
        row["numpy_array_s"] = numpy_array_s
        row["total_s"] = time.perf_counter() - total_start
        return row


def _parse_vane_reader_timing_line(line: str) -> dict[str, float | int] | None:
    if _READER_TIMING_PREFIX not in line:
        return None
    fields = dict(_READER_TIMING_FIELD_RE.findall(line))
    if "rows" not in fields:
        return None
    parsed: dict[str, float | int] = {"rows": int(fields["rows"])}
    for key in _READER_TIMING_STAGE_KEYS:
        parsed[key] = float(fields.get(key, "0"))
    return parsed


def _summarize_vane_reader_timing_lines(lines) -> dict[str, Any]:
    batches = [parsed for line in lines if (parsed := _parse_vane_reader_timing_line(line)) is not None]
    rows = sum(int(batch["rows"]) for batch in batches)
    return {
        "frames": rows,
        "batches": len(batches),
        "open_s": summarize_values(float(batch["open_s"]) for batch in batches),
        "decode_s": summarize_values(float(batch["decode_s"]) for batch in batches),
        "resize_s": summarize_values(float(batch["resize_s"]) for batch in batches),
        "flush_s": summarize_values(float(batch["flush_s"]) for batch in batches),
        "total_s": summarize_values(float(batch["total_s"]) for batch in batches),
    }


def _set_env_temporarily(updates: dict[str, str]):
    class _TemporaryEnv:
        def __enter__(self):
            self._previous = {key: os.environ.get(key) for key in updates}
            os.environ.update(updates)

        def __exit__(self, exc_type, exc, tb):
            for key, value in self._previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    return _TemporaryEnv()


def _profile_vane(video_files: list[str], args) -> dict[str, Any]:
    import vane
    from vane.datasource import read_datasource
    from vane.datasource.video_reader import VideoFrameSource

    frame_height, frame_width = vane_frame_shape(args.height, args.width)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_log = output_dir / "vane_fte_reader_timing.log"
    timing_log.unlink(missing_ok=True)
    run_id = f"read_resize_profile_{os.getpid()}_{time.time_ns()}"
    env_updates = {
        "VANE_RUNNER": "ray",
        "VANE_PROGRESS": "0",
        "RAY_DEDUP_LOGS": "0",
        "VIDEO_BENCHMARK_RUN_ID": run_id,
        "VANE_VIDEO_READER_TIMING_LOG": "1",
        "VANE_VIDEO_READER_TIMING_SAMPLE_RATE": "1",
        "VANE_VIDEO_READER_TIMING_LOG_PATH": str(timing_log),
        "VANE_VIDEO_READER_TIMING_STDOUT": "0",
    }
    with _set_env_temporarily(env_updates):
        con = vane.connect(config={"local_exchange_streaming": "true"})
        try:
            con.execute("SET preserve_insertion_order=false")
            rel = read_datasource(
                VideoFrameSource(
                    video_files,
                    height=frame_height,
                    width=frame_width,
                    max_partition_bytes=args.max_partition_bytes,
                    frame_limit=args.frames if args.frames > 0 else None,
                ),
                con=con,
            )
            with timed_block() as wall:
                (frames,) = rel.query(
                    "frames",
                    "select count(*) as frames from frames",
                ).fetchall()[0]
        finally:
            con.close()

    timing = _summarize_vane_reader_timing_lines(
        timing_log.read_text(encoding="utf-8").splitlines() if timing_log.exists() else []
    )
    frames = int(frames or 0)
    return {
        "mode": "fte_datasource_scan",
        **timing,
        "frames": frames,
        "frame_bytes": frames * frame_height * frame_width * 3,
        "wall_s": wall.elapsed_s,
        "rows_per_s": frames / wall.elapsed_s if wall.elapsed_s > 0 else 0.0,
        "timing_log": str(timing_log),
    }


def _ray_data_read_video_kwargs(requested_frames: int) -> dict[str, int]:
    if requested_frames > 0:
        return {"override_num_blocks": 1}
    return {}


def _profile_ray_data(video_files: list[str], args) -> dict[str, Any]:
    import ray
    import ray.data

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    read_paths = ray_data_paths(video_files)
    ds = ray.data.read_videos(read_paths, **_ray_data_read_video_kwargs(args.frames))
    if args.frames > 0:
        ds = ds.limit(args.frames)
    ds = ds.map(RayResizeFrameProfiler(args.height, args.width))

    frames = 0
    frame_bytes = 0
    image_fromarray_s: list[float] = []
    resize_s: list[float] = []
    numpy_array_s: list[float] = []
    total_s: list[float] = []
    with timed_block() as wall:
        for batch in ds.iter_batches(batch_format="pandas"):
            frames += len(batch)
            frame_bytes += sum(int(frame.nbytes) for frame in batch["frame"])
            image_fromarray_s.extend(float(value) for value in batch["image_fromarray_s"])
            resize_s.extend(float(value) for value in batch["resize_s"])
            numpy_array_s.extend(float(value) for value in batch["numpy_array_s"])
            total_s.extend(float(value) for value in batch["total_s"])

    return {
        "mode": "ray_data_read_videos_resize_frame_payload",
        "frames": frames,
        "frame_bytes": frame_bytes,
        "wall_s": wall.elapsed_s,
        "rows_per_s": frames / wall.elapsed_s if wall.elapsed_s > 0 else 0.0,
        "image_fromarray_s": summarize_values(image_fromarray_s),
        "resize_s": summarize_values(resize_s),
        "numpy_array_s": summarize_values(numpy_array_s),
        "total_s": summarize_values(total_s),
    }


def main() -> None:
    parser = build_common_parser("Profile video read and resize stages in isolation.")
    add_input_options(parser)
    args = parser.parse_args()

    video_files = resolve_video_files_for_args(args)
    payload = {
        "stage": "read_resize",
        "requested_frames": args.frames,
        "input_files": len(video_files),
        "image_height": args.height,
        "image_width": args.width,
        "vane": _profile_vane(video_files, args),
        "ray_data": _profile_ray_data(video_files, args),
    }
    write_json_result(payload, output_json=args.output_json, quiet=args.quiet)


if __name__ == "__main__":
    main()

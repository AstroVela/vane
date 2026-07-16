from __future__ import annotations

import io
import os
import time
from typing import Any

try:
    from .profile_common import add_synthetic_output_options
    from .profile_common import build_common_parser
    from .profile_common import chunks
    from .profile_common import frame_table_from_frames
    from .profile_common import load_resized_frames_vane
    from .profile_common import load_vane_main
    from .profile_common import make_synthetic_features
    from .profile_common import resolve_video_files_for_args
    from .profile_common import summarize_values
    from .profile_common import timed_block
    from .profile_common import vane_frame_shape
    from .profile_common import write_json_result
except ImportError:
    from profile_common import add_synthetic_output_options
    from profile_common import build_common_parser
    from profile_common import chunks
    from profile_common import frame_table_from_frames
    from profile_common import load_resized_frames_vane
    from profile_common import load_vane_main
    from profile_common import make_synthetic_features
    from profile_common import resolve_video_files_for_args
    from profile_common import summarize_values
    from profile_common import timed_block
    from profile_common import vane_frame_shape
    from profile_common import write_json_result


def _profile_ray_crop(frame_indices, frames, features_list) -> dict[str, Any]:
    from PIL import Image

    rows: list[dict[str, float | int]] = []
    with timed_block() as wall:
        for frame_index, frame, features in zip(frame_indices, frames, features_list, strict=False):
            for feature in features:
                total_start = time.perf_counter()
                bbox = feature["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                stage_start = time.perf_counter()
                pil_image = Image.fromarray(frame)
                cropped_pil = pil_image.crop((x1, y1, x2, y2))
                crop_s = time.perf_counter() - stage_start
                stage_start = time.perf_counter()
                buf = io.BytesIO()
                cropped_pil.save(buf, format="PNG", compress_level=2)
                cropped_pil_png = buf.getvalue()
                png_s = time.perf_counter() - stage_start
                rows.append(
                    {
                        "frame_index": int(frame_index),
                        "crop_s": crop_s,
                        "png_s": png_s,
                        "total_s": time.perf_counter() - total_start,
                        "png_bytes": len(cropped_pil_png),
                        "bbox_area": (x2 - x1) * (y2 - y1),
                    }
                )

    return {
        "output_rows": len(rows),
        "wall_s": wall.elapsed_s,
        "rows_per_s": len(rows) / wall.elapsed_s if wall.elapsed_s > 0 else 0.0,
        "crop_s": summarize_values(row["crop_s"] for row in rows),
        "png_s": summarize_values(row["png_s"] for row in rows),
        "total_s": summarize_values(row["total_s"] for row in rows),
        "png_bytes": summarize_values(row["png_bytes"] for row in rows),
        "bbox_area": summarize_values(row["bbox_area"] for row in rows),
    }


def _profile_vane_flat_map(frame_indices, frames, features_list, vane_main) -> dict[str, Any]:
    rows: list[dict[str, float | int]] = []
    with timed_block() as wall:
        for frame_index, frame, features in zip(frame_indices, frames, features_list, strict=False):
            total_start = time.perf_counter()
            out_rows = list(
                vane_main._crop_flat_map(
                    {
                        "frame_index": int(frame_index),
                        "frame": frame,
                        "features": features,
                    }
                )
            )
            rows.append(
                {
                    "output_rows": len(out_rows),
                    "total_s": time.perf_counter() - total_start,
                    "png_bytes": sum(len(row["object"]) for row in out_rows),
                }
            )

    output_rows = sum(int(row["output_rows"]) for row in rows)
    return {
        "mode": "flat_map",
        "input_rows": len(frames),
        "output_rows": output_rows,
        "wall_s": wall.elapsed_s,
        "rows_per_s": output_rows / wall.elapsed_s if wall.elapsed_s > 0 else 0.0,
        "total_s": summarize_values(row["total_s"] for row in rows),
        "png_bytes": summarize_values(row["png_bytes"] for row in rows),
    }


def _profile_vane_map_batches(frame_indices, frames, features_list, vane_main, *, batch_size: int) -> dict[str, Any]:
    import pyarrow as pa

    timings: list[float] = []
    output_rows = 0
    png_bytes: list[int] = []
    start = 0
    with timed_block() as wall:
        for frame_batch in chunks(frames, batch_size):
            end = start + len(frame_batch)
            table = frame_table_from_frames(frame_indices[start:end], frame_batch)
            table = pa.table(
                {
                    "frame_index": table.column("frame_index"),
                    "frame": table.column("frame"),
                    "features": vane_main._features_array(features_list[start:end]),
                }
            )
            chunk_start = time.perf_counter()
            out_tables = list(vane_main._crop_generator(table))
            timings.append(time.perf_counter() - chunk_start)
            output_rows += sum(out_table.num_rows for out_table in out_tables)
            for out_table in out_tables:
                png_bytes.extend(len(value) for value in out_table.column("object").to_pylist())
            start = end
    return {
        "mode": "map_batches",
        "input_rows": len(frames),
        "output_rows": output_rows,
        "wall_s": wall.elapsed_s,
        "rows_per_s": output_rows / wall.elapsed_s if wall.elapsed_s > 0 else 0.0,
        "total_s": summarize_values(timings),
        "png_bytes": summarize_values(png_bytes),
    }


def main() -> None:
    parser = build_common_parser("Profile crop and PNG encoding in isolation.")
    add_synthetic_output_options(parser)
    parser.add_argument("--vane-crop-mode", choices=["flat_map", "map_batches"], default=os.getenv("VIDEO_CROP_MODE", "flat_map"))
    args = parser.parse_args()

    video_files = resolve_video_files_for_args(args)
    frame_indices, frames = load_resized_frames_vane(
        video_files,
        frame_limit=args.frames,
        image_height=args.height,
        image_width=args.width,
    )
    frame_height, frame_width = vane_frame_shape(args.height, args.width)
    features_list = make_synthetic_features(
        frame_count=len(frames),
        width=frame_width,
        height=frame_height,
        objects_per_frame=args.objects_per_frame,
    )
    vane_main = load_vane_main(image_height=args.height, image_width=args.width)

    if args.vane_crop_mode == "map_batches":
        vane_result = _profile_vane_map_batches(
            frame_indices,
            frames,
            features_list,
            vane_main,
            batch_size=args.batch_size,
        )
    else:
        vane_result = _profile_vane_flat_map(frame_indices, frames, features_list, vane_main)

    payload = {
        "stage": "crop_png",
        "requested_frames": args.frames,
        "frames": len(frames),
        "objects_per_frame": args.objects_per_frame,
        "image_height": args.height,
        "image_width": args.width,
        "ray_data": _profile_ray_crop(frame_indices, frames, features_list),
        "vane": vane_result,
    }
    write_json_result(payload, output_json=args.output_json, quiet=args.quiet)


if __name__ == "__main__":
    main()

from __future__ import annotations

import time
from typing import Any

try:
    from .profile_common import add_yolo_options
    from .profile_common import build_common_parser
    from .profile_common import chunks
    from .profile_common import frame_table_from_frames
    from .profile_common import load_resized_frames_vane
    from .profile_common import load_vane_main
    from .profile_common import resolve_video_files_for_args
    from .profile_common import summarize_values
    from .profile_common import timed_block
    from .profile_common import write_json_result
except ImportError:
    from profile_common import add_yolo_options
    from profile_common import build_common_parser
    from profile_common import chunks
    from profile_common import frame_table_from_frames
    from profile_common import load_resized_frames_vane
    from profile_common import load_vane_main
    from profile_common import resolve_video_files_for_args
    from profile_common import summarize_values
    from profile_common import timed_block
    from profile_common import write_json_result


def _to_features(res) -> list[dict[str, object]]:
    return [
        {
            "label": label,
            "confidence": confidence.item(),
            "bbox": bbox.tolist(),
        }
        for label, confidence, bbox in zip(res.names, res.boxes.conf, res.boxes.xyxy, strict=False)
    ]


def _profile_ray_batches(frames, *, batch_size: int, model, device) -> dict[str, Any]:
    from multimodal_inference_benchmarks.video_object_detection.video_tensor import frames_to_tensor_batch
    import torch

    timings: list[dict[str, float | int]] = []
    with timed_block() as wall:
        for frame_batch in chunks(frames, batch_size):
            total_start = time.perf_counter()
            stage_start = time.perf_counter()
            stack = frames_to_tensor_batch(frame_batch, device=device)
            tensor_s = time.perf_counter() - stage_start
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            stage_start = time.perf_counter()
            results = model(stack, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_s = time.perf_counter() - stage_start
            stage_start = time.perf_counter()
            features = [_to_features(res) for res in results]
            feature_s = time.perf_counter() - stage_start
            timings.append(
                {
                    "rows": len(frame_batch),
                    "output_rows": sum(len(frame_features) for frame_features in features),
                    "tensor_s": tensor_s,
                    "model_s": model_s,
                    "feature_s": feature_s,
                    "total_s": time.perf_counter() - total_start,
                }
            )

    rows = sum(int(item["rows"]) for item in timings)
    return {
        "frames": rows,
        "batches": len(timings),
        "wall_s": wall.elapsed_s,
        "rows_per_s": rows / wall.elapsed_s if wall.elapsed_s > 0 else 0.0,
        "output_rows": sum(int(item["output_rows"]) for item in timings),
        "tensor_s": summarize_values(item["tensor_s"] for item in timings),
        "model_s": summarize_values(item["model_s"] for item in timings),
        "feature_s": summarize_values(item["feature_s"] for item in timings),
        "total_s": summarize_values(item["total_s"] for item in timings),
    }


def _profile_vane_batches(frame_indices, frames, *, batch_size: int, model, device, vane_main) -> dict[str, Any]:
    import torch

    timings: list[dict[str, float | int]] = []
    with timed_block() as wall:
        for offset, frame_batch in enumerate(chunks(frames, batch_size)):
            indices = frame_indices[offset * batch_size : offset * batch_size + len(frame_batch)]
            total_start = time.perf_counter()
            stage_start = time.perf_counter()
            table = frame_table_from_frames(indices, frame_batch)
            frame_col = table.column("frame")
            arrow_input_s = time.perf_counter() - stage_start
            stage_start = time.perf_counter()
            stack = vane_main._frame_column_to_tensor_batch(frame_col, device)
            tensor_s = time.perf_counter() - stage_start
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            stage_start = time.perf_counter()
            results = model(stack, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_s = time.perf_counter() - stage_start
            stage_start = time.perf_counter()
            features = [_to_features(res) for res in results]
            feature_s = time.perf_counter() - stage_start
            stage_start = time.perf_counter()
            vane_main._features_array(features)
            arrow_output_s = time.perf_counter() - stage_start
            timings.append(
                {
                    "rows": len(frame_batch),
                    "output_rows": sum(len(frame_features) for frame_features in features),
                    "arrow_input_s": arrow_input_s,
                    "tensor_s": tensor_s,
                    "model_s": model_s,
                    "feature_s": feature_s,
                    "arrow_output_s": arrow_output_s,
                    "total_s": time.perf_counter() - total_start,
                }
            )

    rows = sum(int(item["rows"]) for item in timings)
    return {
        "frames": rows,
        "batches": len(timings),
        "wall_s": wall.elapsed_s,
        "rows_per_s": rows / wall.elapsed_s if wall.elapsed_s > 0 else 0.0,
        "output_rows": sum(int(item["output_rows"]) for item in timings),
        "arrow_input_s": summarize_values(item["arrow_input_s"] for item in timings),
        "tensor_s": summarize_values(item["tensor_s"] for item in timings),
        "model_s": summarize_values(item["model_s"] for item in timings),
        "feature_s": summarize_values(item["feature_s"] for item in timings),
        "arrow_output_s": summarize_values(item["arrow_output_s"] for item in timings),
        "total_s": summarize_values(item["total_s"] for item in timings),
    }


def main() -> None:
    parser = build_common_parser("Profile tensor conversion, YOLO, and feature materialization.")
    add_yolo_options(parser)
    args = parser.parse_args()

    video_files = resolve_video_files_for_args(args)
    frame_indices, frames = load_resized_frames_vane(
        video_files,
        frame_limit=args.frames,
        image_height=args.height,
        image_width=args.width,
    )

    import torch
    from ultralytics import YOLO

    vane_main = load_vane_main(image_height=args.height, image_width=args.width)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(args.yolo_model)
    if torch.cuda.is_available():
        model.to(device)

    if args.warmup_batches > 0 and frames:
        warmup = list(chunks(frames, args.batch_size))[: args.warmup_batches]
        for frame_batch in warmup:
            from multimodal_inference_benchmarks.video_object_detection.video_tensor import frames_to_tensor_batch

            stack = frames_to_tensor_batch(frame_batch, device=device)
            model(stack, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    payload = {
        "stage": "tensor_yolo",
        "requested_frames": args.frames,
        "frames": len(frames),
        "input_files": len(video_files),
        "image_height": args.height,
        "image_width": args.width,
        "batch_size": args.batch_size,
        "device": str(device),
        "ray_data": _profile_ray_batches(frames, batch_size=args.batch_size, model=model, device=device),
        "vane": _profile_vane_batches(
            frame_indices,
            frames,
            batch_size=args.batch_size,
            model=model,
            device=device,
            vane_main=vane_main,
        ),
    }
    write_json_result(payload, output_json=args.output_json, quiet=args.quiet)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROFILE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = PROFILE_DIR.parent
REPO_ROOT = VIDEO_DIR.parents[1]
DEFAULT_INPUT_PATH = str(
    Path(os.getenv("VANE_BENCHMARK_DATA_ROOT", "~/.cache/vane/benchmarks")).expanduser() / "hollywood2" / "AVIClips"
)
DEFAULT_OUTPUT_DIR = "/tmp/vane-video-stage-profiles"
DEFAULT_MAX_PARTITION_BYTES = 10 * 1024 * 1024


def ensure_project_paths() -> None:
    for path in (str(REPO_ROOT), str(VIDEO_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _read_int_env(name: str, default: int, minimum: int | None = None) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        result = default
    else:
        result = int(value)
    if minimum is not None:
        result = max(minimum, result)
    return result


def build_common_parser(description: str, *, default_frames: int = 4096) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input-path", default=os.getenv("INPUT_PATH", DEFAULT_INPUT_PATH))
    parser.add_argument("--input-manifest", default=os.getenv("INPUT_MANIFEST", "").strip() or None)
    parser.add_argument("--frames", type=int, default=_read_int_env("INPUT_LIMIT", default_frames, minimum=1))
    parser.add_argument(
        "--height",
        type=int,
        default=_read_int_env("IMAGE_HEIGHT", 640, minimum=1),
        help="Ray Data IMAGE_HEIGHT value; PIL receives resize((height, width)).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=_read_int_env("IMAGE_WIDTH", 640, minimum=1),
        help="Ray Data IMAGE_WIDTH value; the resized numpy frame shape is (width, height, 3).",
    )
    parser.add_argument("--batch-size", type=int, default=_read_int_env("BATCH_SIZE", 16, minimum=1))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(os.getenv("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)))
    parser.add_argument("--quiet", action="store_true")
    return parser


def add_input_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-partition-bytes",
        type=int,
        default=_read_int_env("VANE_VIDEO_MAX_PARTITION_BYTES", DEFAULT_MAX_PARTITION_BYTES, minimum=1),
    )


def add_yolo_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--yolo-model", default=os.getenv("YOLO_MODEL", "yolo11n.pt"))
    parser.add_argument("--warmup-batches", type=int, default=1)


def add_synthetic_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rows", type=int, default=_read_int_env("VIDEO_PROFILE_ROWS", 4096, minimum=1))
    parser.add_argument(
        "--object-bytes",
        type=int,
        default=_read_int_env("VIDEO_PROFILE_OBJECT_BYTES", 8192, minimum=1),
    )
    parser.add_argument(
        "--objects-per-frame",
        type=int,
        default=_read_int_env("VIDEO_PROFILE_OBJECTS_PER_FRAME", 3, minimum=1),
    )


def vane_frame_shape(image_height: int, image_width: int) -> tuple[int, int]:
    # Ray Data currently calls PIL as resize((IMAGE_HEIGHT, IMAGE_WIDTH)).
    # PIL interprets that tuple as (width, height), so Vane uses the swapped
    # frame shape to keep the produced numpy arrays aligned.
    return image_width, image_height


def resolve_video_files_for_args(args: argparse.Namespace) -> list[str]:
    ensure_project_paths()
    from video_inputs import path_is_s3_like, resolve_video_files

    filesystem = None
    if args.input_manifest is None and path_is_s3_like(args.input_path):
        import pyarrow.fs as pa_fs

        endpoint_url = os.getenv("AWS_ENDPOINT_URL", "")
        endpoint = endpoint_url.split("://", 1)[-1] if endpoint_url else None
        scheme = endpoint_url.split("://", 1)[0] if "://" in endpoint_url else "http"
        filesystem = pa_fs.S3FileSystem(
            access_key=os.getenv("AWS_ACCESS_KEY_ID", ""),
            secret_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            endpoint_override=endpoint,
            scheme=scheme,
            region=os.getenv("AWS_REGION", "us-east-1"),
        )
    return resolve_video_files(args.input_path, input_manifest=args.input_manifest, filesystem=filesystem)


def ray_data_paths(video_files: Sequence[str]) -> list[str]:
    ensure_project_paths()
    from video_inputs import ray_data_read_paths

    return ray_data_read_paths(video_files)


def load_vane_main(*, image_height: int, image_width: int):
    ensure_project_paths()
    os.environ["IMAGE_HEIGHT"] = str(image_height)
    os.environ["IMAGE_WIDTH"] = str(image_width)
    module_name = f"video_vane_main_profile_{image_height}_{image_width}"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, VIDEO_DIR / "vane_main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load vane_main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chunks(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def summarize_values(values: Iterable[float]) -> dict[str, float | int]:
    vals = sorted(float(value) for value in values)
    if not vals:
        return {"count": 0, "sum": 0.0, "avg": 0.0, "min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    def clean(value: float) -> float:
        return round(float(value), 9)

    def percentile_nearest_rank(pct: float) -> float:
        index = max(0, min(len(vals) - 1, math.ceil((pct / 100.0) * len(vals)) - 1))
        return vals[index]

    total = sum(vals)
    return {
        "count": len(vals),
        "sum": clean(total),
        "avg": clean(total / len(vals)),
        "min": clean(vals[0]),
        "p50": clean(percentile_nearest_rank(50.0)),
        "p95": clean(percentile_nearest_rank(95.0)),
        "max": clean(vals[-1]),
    }


@dataclass
class TimedBlock:
    elapsed_s: float = 0.0


@contextmanager
def timed_block() -> Iterator[TimedBlock]:
    result = TimedBlock()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_s = time.perf_counter() - start


def write_json_result(payload: dict[str, Any], *, output_json: Path | None, quiet: bool = False) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output_json is not None:
        output_json = output_json.expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text + "\n", encoding="utf-8")
    if not quiet:
        print(text)


def make_synthetic_features(
    *,
    frame_count: int,
    width: int,
    height: int,
    objects_per_frame: int,
) -> list[list[dict[str, object]]]:
    box_width = max(1.0, width / (objects_per_frame * 2.0))
    box_height = max(1.0, height / (objects_per_frame * 2.0))
    result: list[list[dict[str, object]]] = []
    for frame_idx in range(frame_count):
        frame_features: list[dict[str, object]] = []
        for object_idx in range(objects_per_frame):
            x1 = float((frame_idx * 17 + object_idx * box_width) % max(1.0, width - box_width))
            y1 = float((frame_idx * 11 + object_idx * box_height) % max(1.0, height - box_height))
            frame_features.append(
                {
                    "label": object_idx,
                    "confidence": 0.5 + object_idx * 0.01,
                    "bbox": [x1, y1, x1 + box_width, y1 + box_height],
                }
            )
        result.append(frame_features)
    return result


def load_resized_frames_vane(
    video_files: Sequence[str],
    *,
    frame_limit: int,
    image_height: int,
    image_width: int,
) -> tuple[list[int], list[Any]]:
    ensure_project_paths()
    from duckdb.datasource.video_reader import _open_decord_reader, _resize_frame_batch

    frame_height, frame_width = vane_frame_shape(image_height, image_width)
    frame_indices: list[int] = []
    frames: list[Any] = []
    remaining = frame_limit
    for video_path in video_files:
        if remaining <= 0:
            break
        reader = _open_decord_reader(video_path)
        raw_frames = []
        raw_indices = []
        for frame_idx, frame in enumerate(reader):
            if remaining <= 0:
                break
            raw_frames.append(frame.asnumpy())
            raw_indices.append(frame_idx)
            remaining -= 1
        resized_frames = _resize_frame_batch(raw_frames, width=frame_width, height=frame_height)
        frames.extend(resized_frames)
        frame_indices.extend(raw_indices)
    return frame_indices, frames


def frame_table_from_frames(frame_indices: Sequence[int], frames: Sequence[Any]):
    ensure_project_paths()
    import numpy as np
    import pyarrow as pa

    from duckdb.datasource.video_reader import _build_frame_array, _int64_array

    if not frames:
        empty_frames = np.empty((0, 1, 1, 3), dtype=np.uint8)
        return pa.table(
            {
                "frame_index": pa.array([], type=pa.int64()),
                "frame": pa.FixedShapeTensorArray.from_numpy_ndarray(empty_frames),
            }
        )
    batch = np.stack(list(frames), axis=0)
    return pa.table(
        {
            "frame_index": _int64_array(list(frame_indices)),
            "frame": _build_frame_array(batch),
        }
    )


def make_synthetic_output_table(*, rows: int, object_bytes: int):
    import numpy as np
    import pyarrow as pa

    labels = pa.array([idx % 80 for idx in range(rows)], type=pa.int64())
    confidences = pa.array([0.5 + ((idx % 50) / 100.0) for idx in range(rows)], type=pa.float64())
    bbox_values = []
    bbox_offsets = [0]
    for idx in range(rows):
        x1 = float(idx % 640)
        y1 = float((idx * 3) % 640)
        bbox_values.extend([x1, y1, x1 + 32.0, y1 + 32.0])
        bbox_offsets.append(len(bbox_values))
    bbox = pa.ListArray.from_arrays(pa.array(bbox_offsets, type=pa.int32()), pa.array(bbox_values, type=pa.float64()))
    features = pa.StructArray.from_arrays([labels, confidences, bbox], names=["label", "confidence", "bbox"])

    rng = np.random.default_rng(0)
    blobs = rng.integers(0, 256, size=(rows, object_bytes), dtype=np.uint8)
    objects = pa.array([memoryview(blobs[idx]).tobytes() for idx in range(rows)], type=pa.binary())
    return pa.table(
        {
            "frame_index": pa.array(range(rows), type=pa.int64()),
            "features": features,
            "object": objects,
        }
    )


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def parquet_file_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for file in path.rglob("*.parquet") if file.is_file())

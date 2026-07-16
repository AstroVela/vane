import io
import os
import sys
import time
import uuid

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if PROJECT_ROOT not in os.getenv("PYTHONPATH", "").split(os.pathsep):
    os.environ["PYTHONPATH"] = os.pathsep.join([PROJECT_ROOT, os.getenv("PYTHONPATH", "")]).strip(os.pathsep)

import numpy as np
import pyarrow.fs as pafs
import ray
import torch
from PIL import Image
from ultralytics import YOLO
from video_inputs import has_s3_video_files, path_is_s3_like, ray_data_read_paths, resolve_video_files
from video_kernels import (
    crop_bbox_to_png,
    frames_to_torch_tensor,
    resize_rgb_frame,
    video_gpu_transport_config_from_env,
    yolo_result_to_features,
)

NUM_GPU_NODES = int(os.getenv("NUM_GPU_NODES", "2"))
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11n.pt")
_DATA_ROOT = os.path.expanduser(os.getenv("VANE_BENCHMARK_DATA_ROOT", "~/.cache/vane/benchmarks"))
_DEFAULT_LOCAL_VIDEO_DIR = os.path.join(_DATA_ROOT, "hollywood2", "AVIClips")


def _read_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def _read_int_env(name: str, default: int, minimum: int | None = None) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        result = default
    else:
        result = int(value)
    if minimum is not None:
        result = max(minimum, result)
    return result


def _read_optional_text_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        text = value.strip()
        if text:
            return text
    return None


RAY_DATA_TIMING_LOG = _read_bool_env(
    "VIDEO_RAY_DATA_TIMING_LOG",
    _read_bool_env("VIDEO_UDF_TIMING_LOG", False),
)
RAY_DATA_TIMING_SAMPLE_RATE = _read_int_env("VIDEO_RAY_DATA_TIMING_SAMPLE_RATE", 1, minimum=1)
RAY_DATA_TIMING_LOG_PATH = _read_optional_text_env(
    ("VIDEO_RAY_DATA_TIMING_LOG_PATH", "VIDEO_UDF_TIMING_LOG_PATH", "UDF_TIMING_LOG_PATH")
)
BENCHMARK_RUN_ID = os.getenv("VIDEO_BENCHMARK_RUN_ID", "-").strip() or "-"
_RESIZE_TIMING_CALLS = 0
_CROP_TIMING_CALLS = 0


def _format_seconds_and_ms(name: str, seconds: float) -> str:
    value = max(0.0, float(seconds))
    return f"{name}_s={value:.6f} {name}_ms={value * 1000.0:.3f}"


def _format_stage_timing_fields(*, total_s: float, rows_per_s: float, **stage_seconds: float) -> str:
    fields = [
        _format_seconds_and_ms(name[:-2] if name.endswith("_s") else name, seconds)
        for name, seconds in stage_seconds.items()
    ]
    fields.append(_format_seconds_and_ms("total", total_s))
    fields.append(f"rows_per_s={float(rows_per_s):.2f}")
    return " ".join(fields)


def _emit_ray_data_timing_line(line: str) -> None:
    print(line, flush=True)
    if not RAY_DATA_TIMING_LOG_PATH:
        return
    try:
        log_path = os.path.abspath(os.path.expanduser(RAY_DATA_TIMING_LOG_PATH))
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
    except Exception as exc:
        print(
            "[ray_data_video][stage_timing_log_error] "
            f"run_id={BENCHMARK_RUN_ID} pid={os.getpid()} path={RAY_DATA_TIMING_LOG_PATH!r} "
            f"error={type(exc).__name__}: {exc}",
            flush=True,
        )


def _emit_ray_data_stage_timing(
    *,
    stage: str,
    call: int,
    rows: int,
    output_rows: int | None = None,
    total_s: float,
    **stage_seconds: float,
) -> None:
    if not RAY_DATA_TIMING_LOG:
        return
    bbox_area = stage_seconds.pop("bbox_area", None)
    png_bytes = stage_seconds.pop("png_bytes", None)
    rows_per_s = rows / total_s if total_s > 0 else 0.0
    timing_fields = _format_stage_timing_fields(total_s=total_s, rows_per_s=rows_per_s, **stage_seconds)
    output = f" output_rows={output_rows}" if output_rows is not None else ""
    metrics = ""
    if bbox_area is not None:
        metrics += f" bbox_area={int(bbox_area)}"
    if png_bytes is not None:
        metrics += f" png_bytes={int(png_bytes)}"
    _emit_ray_data_timing_line(
        "[ray_data_video][stage_timing] "
        f"run_id={BENCHMARK_RUN_ID} pid={os.getpid()} stage={stage} call={call} rows={rows}{output} "
        f"{timing_fields}{metrics}"
    )


# S3/MinIO configuration
S3_ENDPOINT = os.getenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9000")
S3_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "")
S3_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
S3_REGION = os.getenv("AWS_REGION", "us-east-1")

_DEFAULT_S3_VIDEO_DIR = "datasets/multimodal_inference_benchmarks/hollywood2/AVIClips"

INPUT_PATH = os.getenv("INPUT_PATH", _DEFAULT_S3_VIDEO_DIR)
INPUT_MANIFEST = os.getenv("INPUT_MANIFEST", "").strip() or None
_DEFAULT_S3_OUTPUT = (
    f"datasets/multimodal_inference_benchmarks/video_object_detection_output/raydata_{uuid.uuid4().hex}"
)
_DEFAULT_LOCAL_OUTPUT = f"/tmp/raydata-video-write-benchmark/{uuid.uuid4().hex}"
OUTPUT_PATH = os.getenv("OUTPUT_PATH", _DEFAULT_LOCAL_OUTPUT)
WRITE_S3 = os.getenv("WRITE_S3", "false").lower() in ("true", "1", "yes")
IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "640"))
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "640"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
VIDEO_GPU_TRANSPORT = video_gpu_transport_config_from_env()
INPUT_LIMIT = int(os.getenv("INPUT_LIMIT", "0"))

s3_fs = pafs.S3FileSystem(
    access_key=S3_ACCESS_KEY,
    secret_key=S3_SECRET_KEY,
    endpoint_override=S3_ENDPOINT,
    scheme="http",
    region=S3_REGION,
)

source_s3 = path_is_s3_like(INPUT_PATH) and INPUT_MANIFEST is None
video_files = resolve_video_files(
    INPUT_PATH,
    input_manifest=INPUT_MANIFEST,
    filesystem=s3_fs if source_s3 else None,
)
READ_S3 = has_s3_video_files(video_files)

ray_data_context = ray.data.DataContext.get_current()
ray_data_context.target_max_block_size = VIDEO_GPU_TRANSPORT.target_max_block_bytes
ray.init()

print(f"[ray_data] input_path={INPUT_PATH} (s3={READ_S3})")
print(f"[ray_data] input_manifest={INPUT_MANIFEST or '<generated>'} files={len(video_files)}")
print(f"[ray_data] output_path={OUTPUT_PATH} (s3={WRITE_S3})")
print(
    f"[ray_data] model={YOLO_MODEL} batch_size={BATCH_SIZE} image={IMAGE_WIDTH}x{IMAGE_HEIGHT} "
    f"num_gpu_nodes={NUM_GPU_NODES} tensor_mode=reference "
    f"target_block_bytes={VIDEO_GPU_TRANSPORT.target_max_block_bytes} "
    "actor_scheduling=native"
)
print(f"[ray_data] input_limit={INPUT_LIMIT}")

t_start = time.time()


class ExtractImageFeatures:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(YOLO_MODEL)
        if torch.cuda.is_available():
            self.model.to(self.device)
        self._timing_calls = 0

    def to_features(self, res):
        return yolo_result_to_features(res)

    def __call__(self, batch):
        self._timing_calls += 1
        call_number = self._timing_calls
        should_time = RAY_DATA_TIMING_LOG and (call_number % RAY_DATA_TIMING_SAMPLE_RATE == 0)
        if should_time:
            total_start = time.perf_counter()
            frames = batch["frame"]
            if len(frames) == 0:
                batch["features"] = []
                _emit_ray_data_stage_timing(
                    stage="yolo",
                    call=call_number,
                    rows=0,
                    output_rows=0,
                    total_s=time.perf_counter() - total_start,
                    tensor_s=0.0,
                    model_s=0.0,
                    feature_s=0.0,
                )
                return batch

            stage_start = time.perf_counter()
            stack = frames_to_torch_tensor(frames, None)
            tensor_s = time.perf_counter() - stage_start

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            stage_start = time.perf_counter()
            results = self.model(stack, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_s = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            features = [self.to_features(res) for res in results]
            feature_s = time.perf_counter() - stage_start
            batch["features"] = features
            total_s = time.perf_counter() - total_start
            _emit_ray_data_stage_timing(
                stage="yolo",
                call=call_number,
                rows=len(frames),
                output_rows=sum(len(frame_features) for frame_features in features),
                total_s=total_s,
                tensor_s=tensor_s,
                model_s=model_s,
                feature_s=feature_s,
            )
            return batch

        frames = batch["frame"]
        if len(frames) == 0:
            batch["features"] = []
            return batch
        stack = frames_to_torch_tensor(frames, None)
        results = self.model(stack, verbose=True)
        features = [self.to_features(res) for res in results]
        batch["features"] = features
        return batch


def resize_frame(row):
    global _RESIZE_TIMING_CALLS

    _RESIZE_TIMING_CALLS += 1
    call_number = _RESIZE_TIMING_CALLS
    should_time = RAY_DATA_TIMING_LOG and (call_number % RAY_DATA_TIMING_SAMPLE_RATE == 0)
    if should_time:
        total_start = time.perf_counter()
        frame = row["frame"]

        stage_start = time.perf_counter()
        pil_image = Image.fromarray(frame)
        image_fromarray_s = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        resized_pil = pil_image.resize((IMAGE_HEIGHT, IMAGE_WIDTH))
        resize_s = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        resized_frame = np.array(resized_pil)
        numpy_array_s = time.perf_counter() - stage_start

        row["frame"] = resized_frame
        _emit_ray_data_stage_timing(
            stage="resize_frame",
            call=call_number,
            rows=1,
            output_rows=1,
            total_s=time.perf_counter() - total_start,
            image_fromarray_s=image_fromarray_s,
            resize_s=resize_s,
            numpy_array_s=numpy_array_s,
        )
        return row

    frame = row["frame"]
    row["frame"] = resize_rgb_frame(frame, width=IMAGE_HEIGHT, height=IMAGE_WIDTH)
    return row


def explode_features(row):
    features_list = row["features"]
    for feature in features_list:
        row["features"] = feature
        yield row


def crop_image(row):
    global _CROP_TIMING_CALLS

    _CROP_TIMING_CALLS += 1
    call_number = _CROP_TIMING_CALLS
    should_time = RAY_DATA_TIMING_LOG and (call_number % RAY_DATA_TIMING_SAMPLE_RATE == 0)
    if should_time:
        total_start = time.perf_counter()
        frame = row["frame"]
        bbox = row["features"]["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        bbox_area = (x2 - x1) * (y2 - y1)

        stage_start = time.perf_counter()
        pil_image = Image.fromarray(frame)
        cropped_pil = pil_image.crop((x1, y1, x2, y2))
        crop_s = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        buf = io.BytesIO()
        cropped_pil.save(buf, format="PNG", compress_level=2)
        cropped_pil_png = buf.getvalue()
        png_bytes = len(cropped_pil_png)
        crop_encode_s = time.perf_counter() - stage_start

        row["object"] = cropped_pil_png
        _emit_ray_data_stage_timing(
            stage="crop_image",
            call=call_number,
            rows=1,
            output_rows=1,
            total_s=time.perf_counter() - total_start,
            crop_s=crop_s,
            crop_encode_s=crop_encode_s,
            bbox_area=bbox_area,
            png_bytes=png_bytes,
        )
        return row

    frame = row["frame"]
    bbox = row["features"]["bbox"]
    row["object"] = crop_bbox_to_png(frame, bbox)
    return row


read_paths = ray_data_read_paths(video_files)
read_video_kwargs = {"override_num_blocks": 1} if INPUT_LIMIT > 0 else {}
if READ_S3:
    ds = ray.data.read_videos(read_paths, filesystem=s3_fs, **read_video_kwargs)
else:
    ds = ray.data.read_videos(read_paths, **read_video_kwargs)
if INPUT_LIMIT > 0:
    ds = ds.limit(INPUT_LIMIT)
ds = ds.map(resize_frame)
ds = ds.map_batches(
    ExtractImageFeatures,
    batch_size=BATCH_SIZE,
    num_gpus=1.0,
    concurrency=NUM_GPU_NODES,
)
ds = ds.flat_map(explode_features)
ds = ds.map(crop_image)
ds = ds.drop_columns(["frame"])
stage_start = time.perf_counter()
if WRITE_S3:
    ds.write_parquet(OUTPUT_PATH, filesystem=s3_fs)
else:
    ds.write_parquet(OUTPUT_PATH)
_emit_ray_data_stage_timing(
    stage="write_parquet",
    call=1,
    rows=0,
    total_s=time.perf_counter() - stage_start,
)

elapsed = time.time() - t_start
print(f"Runtime: {elapsed}")

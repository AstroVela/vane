from __future__ import annotations

import os
from collections.abc import Sequence

RAY_DATA_DEFAULT_READ_OP_MIN_NUM_BLOCKS = 200
RAY_DATA_DEFAULT_TARGET_MIN_BLOCK_SIZE_BYTES = 1024 * 1024

VIDEO_FILE_EXTENSIONS = (
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
    ".3gp",
    ".mpeg",
    ".mpg",
    ".ts",
    ".ogv",
    ".rm",
    ".rmvb",
    ".vob",
    ".asf",
    ".f4v",
    ".m2ts",
    ".mts",
    ".divx",
    ".xvid",
    ".mxf",
)


def path_is_local(path: str) -> bool:
    return path.startswith("file://") or os.path.isabs(path) or os.path.exists(path)


def path_is_s3_like(path: str) -> bool:
    return path.startswith("s3://") or not path_is_local(path)


def has_s3_video_files(video_files: Sequence[str]) -> bool:
    s3_flags = [path_is_s3_like(path) for path in video_files]
    if any(s3_flags) and not all(s3_flags):
        raise ValueError("video input files must be all S3 or all local")
    return any(s3_flags)


def normalize_s3_path(path: str) -> str:
    if path.startswith("s3://"):
        return path
    return f"s3://{path.lstrip('/')}"


def read_video_manifest(manifest_path: str) -> list[str]:
    files: list[str] = []
    with open(manifest_path, encoding="utf-8") as handle:
        for raw_line in handle:
            path = raw_line.strip()
            if not path or path.startswith("#"):
                continue
            files.append(normalize_s3_path(path) if path_is_s3_like(path) else path)
    if not files:
        raise ValueError(f"input manifest is empty: {manifest_path}")
    return files


def list_local_video_files(path: str) -> list[str]:
    local_path = path[len("file://") :] if path.startswith("file://") else path
    if os.path.isdir(local_path):
        files: list[str] = []
        for root, _dirs, names in os.walk(local_path):
            for name in names:
                full_path = os.path.join(root, name)
                if full_path.lower().endswith(VIDEO_FILE_EXTENSIONS):
                    files.append(full_path)
        files.sort()
        return files
    if os.path.isfile(local_path) and local_path.lower().endswith(VIDEO_FILE_EXTENSIONS):
        return [local_path]
    return []


def list_s3_video_files(path: str, filesystem) -> list[str]:
    if filesystem is None:
        raise ValueError(f"S3 input requires a filesystem: {path}")

    import pyarrow.fs as pa_fs

    s3_path = normalize_s3_path(path)
    selector = pa_fs.FileSelector(s3_path[len("s3://") :].rstrip("/"), recursive=True)
    infos = filesystem.get_file_info(selector)
    files = [
        f"s3://{info.path}"
        for info in infos
        if info.type == pa_fs.FileType.File and info.path.lower().endswith(VIDEO_FILE_EXTENSIONS)
    ]
    files.sort()
    return files


def resolve_video_files(input_path: str, *, input_manifest: str | None = None, filesystem=None) -> list[str]:
    if input_manifest:
        return read_video_manifest(input_manifest)

    if path_is_s3_like(input_path):
        files = list_s3_video_files(input_path, filesystem)
    else:
        files = list_local_video_files(input_path)
    if not files:
        raise RuntimeError(f"No video files found under {input_path!r}")
    return files


def ray_data_read_paths(video_files: Sequence[str]) -> list[str]:
    return [path[len("s3://") :] if path.startswith("s3://") else path for path in video_files]


def estimate_video_input_size_bytes(video_files: Sequence[str], *, filesystem=None) -> int:
    """Return the same compressed-file size estimate used by Ray's file datasource."""
    if not video_files:
        return 0

    if has_s3_video_files(video_files):
        if filesystem is None:
            raise ValueError("S3 video size estimation requires a filesystem")
        infos = filesystem.get_file_info(ray_data_read_paths(video_files))
        missing = [video_files[index] for index, info in enumerate(infos) if int(info.size) < 0]
        if missing:
            raise RuntimeError(f"could not determine video file size for {missing[0]!r}")
        return sum(int(info.size) for info in infos)

    total_size = 0
    for path in video_files:
        local_path = path[len("file://") :] if path.startswith("file://") else path
        total_size += os.path.getsize(local_path)
    return total_size


def ray_data_read_task_count(
    video_files: Sequence[str],
    *,
    input_size_bytes: int,
    available_cpus: int,
    target_max_block_size: int,
    target_min_block_size: int = RAY_DATA_DEFAULT_TARGET_MIN_BLOCK_SIZE_BYTES,
    read_op_min_num_blocks: int = RAY_DATA_DEFAULT_READ_OP_MIN_NUM_BLOCKS,
    input_limit: int = 0,
) -> int:
    """Mirror Ray Data's default file-read parallelism for boundary alignment.

    Ray Data first derives a requested parallelism from compressed input size,
    block-size limits, and cluster CPUs. ``FileBasedDatasource`` then caps the
    number of read tasks at the number of files. ``read_videos(...,
    override_num_blocks=1)`` is used by this benchmark when ``input_limit`` is
    active, so that path is explicitly one task.
    """
    file_count = len(video_files)
    if file_count == 0:
        return 0
    if int(input_limit) > 0:
        return 1

    size_bytes = max(0, int(input_size_bytes))
    cpu_count = max(1, int(available_cpus))
    max_block_bytes = int(target_max_block_size)
    min_block_bytes = int(target_min_block_size)
    min_blocks = int(read_op_min_num_blocks)
    if max_block_bytes <= 0 or min_block_bytes <= 0 or min_blocks <= 0:
        raise ValueError("Ray Data read-task sizing inputs must be positive")

    min_safe_parallelism = max(1, size_bytes // max_block_bytes)
    max_reasonable_parallelism = max(1, size_bytes // min_block_bytes)
    detected_parallelism = max(
        min(min_blocks, max_reasonable_parallelism),
        min_safe_parallelism,
        cpu_count * 2,
    )
    return min(file_count, detected_parallelism)

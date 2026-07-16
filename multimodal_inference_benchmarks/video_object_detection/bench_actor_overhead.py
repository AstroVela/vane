#!/usr/bin/env python3
"""Actor overhead benchmark: measures the cost of Ray actor IPC vs local compute.

Tests:
  1. LOCAL   - YOLO inference directly in the main process (no actor)
  2. ACTOR   - Same inference via a Ray actor (measures serialization + scheduling)
  3. PIPELINE_VANE - Full Vane pipeline overhead estimate
  4. PIPELINE_RAY  - Full Ray Data pipeline overhead estimate

For each test we measure:
  - Pure GPU compute time (inside __call__)
  - Wall-clock round-trip time (submit → result)
  - Serialization size (Arrow table bytes)

Usage:
  source .venv-system/bin/activate
  python multimodal_inference_benchmarks/video_object_detection/bench_actor_overhead.py
"""

import json
import os
import time

import numpy as np

os.environ.setdefault("YOLO_MODEL", "yolo11n.pt")

import pyarrow as pa
import ray
import torch
from ultralytics import YOLO

YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11n.pt")
IMAGE_H = 640
IMAGE_W = 640
FRAME_SIZE = IMAGE_H * IMAGE_W * 3
BATCH_SIZES = [1, 4, 8, 16, 32]
WARMUP_BATCHES = 3
MEASURE_BATCHES = 10


def make_fake_frames(n: int) -> pa.Table:
    """Create a table with n random 640x640 RGB frames as fixed-shape tensors."""
    rng = np.random.default_rng(42)
    frames = rng.integers(0, 256, (n, IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
    return pa.table(
        {
            "video_path": [f"fake_{i}.avi" for i in range(n)],
            "frame_index": list(range(n)),
            "frame": pa.FixedShapeTensorArray.from_numpy_ndarray(frames),
        }
    )


def arrow_table_bytes(t: pa.Table) -> int:
    """Approximate serialized size of an Arrow table."""
    buf = pa.ipc.serialize_pandas(t.to_pandas())
    return len(buf)


# ─── Local compute (no actor) ───────────────────────────────────────


class LocalYOLO:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL)
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.compute_time = 0.0

    def run(self, table: pa.Table) -> pa.Table:
        frame_col = table.column("frame")
        frames = frame_col.combine_chunks().to_numpy_ndarray()
        tensors = []
        for frame in frames:
            arr = np.array(frame, copy=True)
            t = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)
            tensors.append(t)
        stack = torch.stack(tensors, dim=0)

        t0 = time.perf_counter()
        results = self.model(stack)
        torch.cuda.synchronize()
        self.compute_time += time.perf_counter() - t0

        features = []
        for res in results:
            frame_features = []
            if res.boxes is not None and len(res.boxes) > 0:
                for label_idx, conf, bbox in zip(res.boxes.cls, res.boxes.conf, res.boxes.xyxy, strict=False):
                    frame_features.append(
                        {
                            "label": res.names.get(int(label_idx.item()), "?"),
                            "confidence": round(conf.item(), 4),
                            "bbox": [round(x, 1) for x in bbox.tolist()],
                        }
                    )
            features.append(json.dumps(frame_features))

        return pa.table(
            {
                "video_path": table.column("video_path").to_pylist(),
                "frame_index": table.column("frame_index").to_pylist(),
                "frame": table.column("frame"),
                "features_json": features,
            }
        )


# ─── Ray actor ──────────────────────────────────────────────────────


@ray.remote(num_gpus=1)
class YOLOActor:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL)
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.compute_time = 0.0

    def run(self, table: pa.Table) -> pa.Table:
        frame_col = table.column("frame")
        frames = frame_col.combine_chunks().to_numpy_ndarray()
        tensors = []
        for frame in frames:
            arr = np.array(frame, copy=True)
            t = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)
            tensors.append(t)
        stack = torch.stack(tensors, dim=0)

        t0 = time.perf_counter()
        results = self.model(stack)
        torch.cuda.synchronize()
        self.compute_time += time.perf_counter() - t0

        features = []
        for res in results:
            frame_features = []
            if res.boxes is not None and len(res.boxes) > 0:
                for label_idx, conf, bbox in zip(res.boxes.cls, res.boxes.conf, res.boxes.xyxy, strict=False):
                    frame_features.append(
                        {
                            "label": res.names.get(int(label_idx.item()), "?"),
                            "confidence": round(conf.item(), 4),
                            "bbox": [round(x, 1) for x in bbox.tolist()],
                        }
                    )
            features.append(json.dumps(frame_features))

        return pa.table(
            {
                "video_path": table.column("video_path").to_pylist(),
                "frame_index": table.column("frame_index").to_pylist(),
                "frame": table.column("frame"),
                "features_json": features,
            }
        )

    def get_compute_time(self):
        return self.compute_time

    def reset_compute_time(self):
        self.compute_time = 0.0


def bench_local(batch_sizes):
    print("\n" + "=" * 60)
    print("TEST 1: LOCAL (no actor, direct GPU call)")
    print("=" * 60)

    detector = LocalYOLO()

    # Warmup
    warmup_table = make_fake_frames(32)
    for _ in range(WARMUP_BATCHES):
        detector.run(warmup_table)
    detector.compute_time = 0.0

    results = {}
    for bs in batch_sizes:
        table = make_fake_frames(bs)
        ser_bytes = arrow_table_bytes(table)

        wall_times = []
        detector.compute_time = 0.0
        for _ in range(MEASURE_BATCHES):
            t0 = time.perf_counter()
            detector.run(table)
            wall_times.append(time.perf_counter() - t0)

        avg_wall = np.mean(wall_times)
        avg_compute = detector.compute_time / MEASURE_BATCHES
        avg_overhead = avg_wall - avg_compute

        results[bs] = {
            "wall_ms": avg_wall * 1000,
            "compute_ms": avg_compute * 1000,
            "overhead_ms": avg_overhead * 1000,
            "ser_KB": ser_bytes / 1024,
        }
        print(
            f"  batch={bs:3d}  wall={avg_wall * 1000:7.1f}ms  compute={avg_compute * 1000:7.1f}ms  "
            f"overhead={avg_overhead * 1000:7.1f}ms  data={ser_bytes / 1024:.0f}KB"
        )

    return results


def bench_actor(batch_sizes):
    print("\n" + "=" * 60)
    print("TEST 2: RAY ACTOR (same node, measures IPC overhead)")
    print("=" * 60)

    actor = YOLOActor.remote()

    # Warmup
    warmup_table = make_fake_frames(32)
    for _ in range(WARMUP_BATCHES):
        ray.get(actor.run.remote(warmup_table))
    ray.get(actor.reset_compute_time.remote())

    results = {}
    for bs in batch_sizes:
        table = make_fake_frames(bs)
        ser_bytes = arrow_table_bytes(table)

        ray.get(actor.reset_compute_time.remote())
        wall_times = []
        for _ in range(MEASURE_BATCHES):
            t0 = time.perf_counter()
            ray.get(actor.run.remote(table))
            wall_times.append(time.perf_counter() - t0)

        avg_wall = np.mean(wall_times)
        avg_compute = ray.get(actor.get_compute_time.remote()) / MEASURE_BATCHES
        avg_overhead = avg_wall - avg_compute

        results[bs] = {
            "wall_ms": avg_wall * 1000,
            "compute_ms": avg_compute * 1000,
            "overhead_ms": avg_overhead * 1000,
            "overhead_pct": (avg_overhead / avg_wall * 100) if avg_wall > 0 else 0,
            "ser_KB": ser_bytes / 1024,
        }
        print(
            f"  batch={bs:3d}  wall={avg_wall * 1000:7.1f}ms  compute={avg_compute * 1000:7.1f}ms  "
            f"overhead={avg_overhead * 1000:7.1f}ms ({results[bs]['overhead_pct']:.1f}%)  data={ser_bytes / 1024:.0f}KB"
        )

    ray.kill(actor)
    return results


def bench_actor_async(batch_sizes):
    """Async actor: submit multiple batches in flight (like Vane's actor_count)."""
    print("\n" + "=" * 60)
    print("TEST 3: RAY ACTOR ASYNC (4 in-flight, measures pipeline overlap)")
    print("=" * 60)

    MAX_INFLIGHT = 4
    actor = YOLOActor.remote()

    # Warmup
    warmup_table = make_fake_frames(32)
    for _ in range(WARMUP_BATCHES):
        ray.get(actor.run.remote(warmup_table))
    ray.get(actor.reset_compute_time.remote())

    results = {}
    for bs in batch_sizes:
        table = make_fake_frames(bs)
        ser_bytes = arrow_table_bytes(table)
        total_batches = MEASURE_BATCHES * 2  # more batches for async

        ray.get(actor.reset_compute_time.remote())

        t0 = time.perf_counter()
        pending = []
        completed = 0
        submitted = 0
        while completed < total_batches:
            while len(pending) < MAX_INFLIGHT and submitted < total_batches:
                pending.append(actor.run.remote(table))
                submitted += 1
            if pending:
                done, pending = ray.wait(pending, num_returns=1)
                ray.get(done[0])
                completed += 1

        wall_total = time.perf_counter() - t0
        avg_wall = wall_total / total_batches
        avg_compute = ray.get(actor.get_compute_time.remote()) / total_batches
        gpu_util = (avg_compute / avg_wall * 100) if avg_wall > 0 else 0

        results[bs] = {
            "wall_ms": avg_wall * 1000,
            "compute_ms": avg_compute * 1000,
            "throughput_fps": bs / avg_wall,
            "gpu_util_pct": gpu_util,
            "ser_KB": ser_bytes / 1024,
        }
        print(
            f"  batch={bs:3d}  wall/batch={avg_wall * 1000:7.1f}ms  compute={avg_compute * 1000:7.1f}ms  "
            f"gpu_util={gpu_util:.1f}%  throughput={results[bs]['throughput_fps']:.1f}fps  data={ser_bytes / 1024:.0f}KB"
        )

    ray.kill(actor)
    return results


def bench_serialization(batch_sizes):
    """Pure serialization cost (no GPU, just cloudpickle round-trip)."""
    print("\n" + "=" * 60)
    print("TEST 4: SERIALIZATION ONLY (Arrow table cloudpickle round-trip)")
    print("=" * 60)

    import pickle

    import cloudpickle

    results = {}
    for bs in batch_sizes:
        table = make_fake_frames(bs)

        # Measure cloudpickle serialization (what Ray actually uses)
        ser_times = []
        deser_times = []
        for _ in range(MEASURE_BATCHES):
            t0 = time.perf_counter()
            data = cloudpickle.dumps(table)
            ser_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            _ = pickle.loads(data)
            deser_times.append(time.perf_counter() - t0)

        avg_ser = np.mean(ser_times) * 1000
        avg_deser = np.mean(deser_times) * 1000
        data_size = len(data)

        # Also measure Arrow IPC serialization
        ipc_ser_times = []
        ipc_deser_times = []
        for _ in range(MEASURE_BATCHES):
            sink = pa.BufferOutputStream()
            writer = pa.ipc.new_stream(sink, table.schema)
            t0 = time.perf_counter()
            writer.write_table(table)
            writer.close()
            ipc_ser_times.append(time.perf_counter() - t0)
            buf = sink.getvalue()

            t0 = time.perf_counter()
            reader = pa.ipc.open_stream(buf)
            _ = reader.read_all()
            ipc_deser_times.append(time.perf_counter() - t0)

        avg_ipc_ser = np.mean(ipc_ser_times) * 1000
        avg_ipc_deser = np.mean(ipc_deser_times) * 1000
        ipc_size = len(buf)

        results[bs] = {
            "cloudpickle_ser_ms": avg_ser,
            "cloudpickle_deser_ms": avg_deser,
            "cloudpickle_size_KB": data_size / 1024,
            "arrow_ipc_ser_ms": avg_ipc_ser,
            "arrow_ipc_deser_ms": avg_ipc_deser,
            "arrow_ipc_size_KB": ipc_size / 1024,
        }
        print(
            f"  batch={bs:3d}  cloudpickle: ser={avg_ser:6.1f}ms deser={avg_deser:6.1f}ms size={data_size / 1024:.0f}KB"
            f"  |  arrow_ipc: ser={avg_ipc_ser:6.1f}ms deser={avg_ipc_deser:6.1f}ms size={ipc_size / 1024:.0f}KB"
        )

    return results


def summary(local_res, actor_res, async_res, ser_res):
    print("\n" + "=" * 60)
    print("SUMMARY: Actor Overhead Analysis")
    print("=" * 60)

    print(
        f"\n{'batch':>5} | {'Local':>10} | {'Actor':>10} | {'IPC overhead':>12} | {'overhead%':>9} | {'Async GPU%':>10} | {'Ser+Deser':>10}"
    )
    print("-" * 80)
    for bs in BATCH_SIZES:
        local_wall = local_res[bs]["wall_ms"]
        actor_wall = actor_res[bs]["wall_ms"]
        actor_overhead = actor_res[bs]["overhead_ms"]
        actor_pct = actor_res[bs]["overhead_pct"]
        async_gpu = async_res[bs]["gpu_util_pct"]
        ser_total = ser_res[bs]["cloudpickle_ser_ms"] + ser_res[bs]["cloudpickle_deser_ms"]

        print(
            f"  {bs:3d}  | {local_wall:8.1f}ms | {actor_wall:8.1f}ms | {actor_overhead:10.1f}ms | {actor_pct:7.1f}% | {async_gpu:8.1f}% | {ser_total:8.1f}ms"
        )

    print("\nKey:")
    print("  Local       = direct GPU call, no actor/IPC")
    print("  Actor       = sync ray.get(actor.run.remote()), 1 in-flight")
    print("  IPC overhead= Actor wall - Actor compute (serialization + scheduling + network)")
    print("  overhead%   = IPC overhead / Actor wall")
    print("  Async GPU%  = GPU utilization with 4 in-flight (higher = better overlap)")
    print("  Ser+Deser   = cloudpickle serialization + deserialization time")


if __name__ == "__main__":
    ray.init()
    print(f"Ray cluster: {ray.cluster_resources()}")
    print(f"Model: {YOLO_MODEL}, Image: {IMAGE_H}x{IMAGE_W}, Frame size: {FRAME_SIZE / 1024:.0f}KB")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Warmup: {WARMUP_BATCHES}, Measure: {MEASURE_BATCHES} batches each")

    ser_res = bench_serialization(BATCH_SIZES)
    local_res = bench_local(BATCH_SIZES)
    actor_res = bench_actor(BATCH_SIZES)
    async_res = bench_actor_async(BATCH_SIZES)
    summary(local_res, actor_res, async_res, ser_res)

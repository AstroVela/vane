# Video batch profiling

Vane already exposes query/operator profiling through
`connection.enable_profiling()` and
`connection.get_profiling_information(format="json")`;
`vane.query_graph.ProfilingInfo` can render that information as HTML.
Operator timings do not separate the Python preprocessing and model stages
inside an inference UDF. The optional instrumentation here fills that gap for
the Vane and Ray Data video benchmarks using the same timing definitions.

## Run

Use the installed benchmark requirements and cached model described in the
parent README. From this directory, run the pipelines sequentially so they do
not compete for the GPU:

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)
export INPUT_PATH=/data/multimodal_inference_benchmarks/hollywood2/AVIClips
export BATCH_SIZE=32
export NUM_GPU_NODES=1
unset RAY_ADDRESS

VIDEO_PROFILE_DIR="/tmp/video_profile_$RUN_ID/vane" \
  VANE_RUNNER=ray OUTPUT_PATH="/tmp/vane_video_$RUN_ID" python vane_main.py
VIDEO_PROFILE_DIR="/tmp/video_profile_$RUN_ID/ray_data" \
  OUTPUT_PATH="/tmp/ray_video_$RUN_ID" python ray_data_main.py

python summarize_profile.py "/tmp/video_profile_$RUN_ID/vane" --skip-batches 1
python summarize_profile.py "/tmp/video_profile_$RUN_ID/ray_data" --skip-batches 1
```

For a smoke test set `INPUT_PATH` to a small directory of videos. For a bounded
diagnostic window, the existing completed JSONL records remain useful even if
the pipeline is interrupted. The summary ignores a truncated final line with
a warning; other malformed records are errors. An interrupted run does **not**
measure completed Parquet throughput. Stop any remaining actors before starting
another GPU run.

`VIDEO_PROFILE_DIR` is read when each actor is constructed. Unset or empty means
disabled: no clocks are sampled and no files are created by this profiler.
Use a fresh directory for each run; otherwise the summary combines runs.
Each actor writes its own UUID-named file with engine, host, PID, batch index,
input row count and wall-clock timestamp. This workflow targets the local
single-machine benchmark; on a cluster the directory is local to each worker
unless it is shared. Provision it consistently and collect all worker files.

## Timing definitions

All durations are milliseconds. Outer phases use `perf_counter_ns()` for wall
time and `thread_time_ns()` for CPU time on the calling thread. Thread CPU time
excludes other PyTorch threads, Ray workers and GPU work; it is not whole-process
CPU consumption. Outer phases partition the observed call body, with small
clock/bookkeeping overhead included.

| Metric | What it measures |
| --- | --- |
| `phase.array` | Vane: frame indices, Arrow/IMAGE to contiguous NumPy conversion. Ray Data: access to the already converted frame column; Ray's conversion before the UDF is outside this timer. |
| `phase.cpu_tensor` | Existing PIL/torchvision conversion, normalization and stacking into a CPU tensor. |
| `phase.model` | Complete existing YOLO call, including setup, warmup and framework overhead. |
| `model.preprocess` | YOLO's synchronized preprocessing interval: CPU-to-GPU transfer and dtype conversion for this tensor-input workload. This is **not** pure DMA time. |
| `model.inference` | YOLO's synchronized inference interval, including host launch/wait overhead; not a sum of CUDA kernel durations. |
| `model.postprocess` | YOLO's synchronized postprocessing, including NMS and construction of result images/objects. |
| `phase.features` | Extraction of labels, confidences and boxes into Python values, including any device-to-host synchronization in `.item()`/`.tolist()`. |
| `phase.pack` | Construction of the returned Arrow table or assignment of the returned Ray Data column. |
| `actor_gap_ms` | Time between the preceding record write finishing and the next call beginning on the same actor. First call: null. |
| `previous_write_ms` | Previous call's final metric collection and JSONL write overhead, excluded from the gap. First call: null. |

The three `model.*` metrics are nested **inside** `phase.model`; do not add them
to outer phases. They reuse Ultralytics 8.3.200's `Results.speed`: per-image
values are summed once to recover batch durations. No predictor monkeypatch,
tensor movement, extra GPU events or CUDA synchronization is added. Missing,
invalid or mismatched model timings are reported as null, not zero. Model
initialization before the first call is not recorded; first-call setup/warmup
is included in `phase.model` and excluded by the summary's default first-batch
filter. Increase `--skip-batches` if warmup lasts longer on your workload.

The gap can contain upstream decode/resize starvation, framework conversions,
serialization/transport, scheduling, downstream backpressure, and return-time
cleanup. It is **not** a direct measurement of scheduler latency. Video decoding,
cropping, PNG encoding and Parquet writing occur outside the measured inference
UDF. Use query/operator profiles, Ray logs and CPU/I/O/GPU sampling to locate
those costs. Low GPU utilization alone cannot distinguish them.

The summary reports sample count, total, mean, median and nearest-rank p95 for
each metric, plus completed batch and inferred frame counts. Totals across
actors can overlap in time. It deliberately does not convert them into
end-to-end FPS or compare them with full-run runtimes. Correlate raw
`wall_start_ns` values with GPU utilization samples over the same time window.

Profiling adds clocks, bookkeeping and one file open/write/close per batch.
Compare enabled/disabled runs with the same input, batch size, model, thread
settings and GPU count. Keep headline throughput runs unprofiled; neither a
microbenchmark overhead estimate nor an actor gap proves a framework bottleneck.

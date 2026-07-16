# Multimodal inference benchmarks

These programs compare equivalent Vane, Ray Data, and Daft pipelines. They are engineering benchmarks, not independently audited product claims.

## Data and output

Set paths explicitly for a reproducible run. Local fallback paths are below `~/.cache/vane/benchmarks`, configurable with `VANE_BENCHMARK_DATA_ROOT`.

Common variables:

```bash
export NUM_GPU_NODES=1
export INPUT_PATH=/path/to/input
export OUTPUT_PATH=/tmp/vane-benchmark-output
```

Dataset-specific variables include `LOCAL_IMAGE_ROOT`, `LOCAL_PDF_ROOT`, `YOLO_MODEL_PATH`, and `INPUT_MANIFEST`. Each implementation reads the same environment variables where possible.

For S3-compatible storage, set the endpoint and credentials yourself:

```bash
export AWS_ENDPOINT_URL=http://127.0.0.1:9000
export AWS_ACCESS_KEY_ID='development-access-key'
export AWS_SECRET_ACCESS_KEY='development-secret-key'
export AWS_REGION=us-east-1
```

No benchmark contains working default credentials. Use short-lived, scoped credentials and never include them in a result bundle.

## Model cache and warm-up

Hugging Face and framework loaders reuse their normal persistent caches. On a cluster, mount the same cache path on every node or pre-stage each node:

```bash
export HF_HOME="$HOME/.cache/huggingface"
export TORCH_HOME="$HOME/.cache/torch"
```

Download and load every model before starting a timed measurement. After the cache is populated, set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to prove that a run is loading cached artifacts rather than downloading them. Pin model revisions in published results.

## Batch-size sweep

Use the implementation's current default `BATCH_SIZE` as the baseline, then test `2x`, `4x`, and so on. Stop only after the larger size produces no meaningful throughput improvement, increases latency beyond the workload target, or causes memory pressure/failures.

For each size:

1. Run an untimed warm-up that loads models and initializes Ray actors.
2. Run at least three measured trials with identical input ordering and limits.
3. Record median wall time, rows/s, GPU utilization and memory, CPU utilization, peak object-store memory, failures, and output row count/hash.
4. Treat the smallest size within normal run-to-run variance of the best throughput as the sweet spot.

Do not compare a warm cache against a cold cache or include a failed/OOM size as a performance result. CPU operators can also batch; distinguish their row batch from the GPU model batch when tuning.

## Reproducibility record

Every published result should include:

- Vane and `external/duckdb` commits;
- Ray, Daft, Python, CUDA, driver, PyTorch, and model versions;
- immutable model revision and cache/warm-up policy;
- dataset identity, count, preprocessing, and input limit;
- node, CPU, RAM, GPU, disk, and network details;
- all non-default environment variables;
- per-stage batch sizes and concurrency;
- raw trial timings and output validation.

The FTE readiness and chaos helpers in this directory are intended for correctness and recovery gates, not just throughput measurement.

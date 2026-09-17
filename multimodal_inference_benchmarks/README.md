# Multimodal inference benchmarks

This directory compares equivalent Vane, Ray Data, and Daft pipelines on one local GPU machine.

This guide covers audio transcription, document embedding, image classification, and video object detection. `large_image_embedding` is currently excluded.

The benchmark entrypoints read local files only. The separate download scripts use anonymous access to copy public S3 data to the local machine.

Workloads follow Anyscale's [multimodal benchmark methodology](https://www.anyscale.com/blog/ray-data-daft-benchmarking-multimodal-ai-workloads).
These runs use downloaded data on one local GPU machine; they do not reproduce
the article's distributed storage setup or performance numbers.

## Prerequisites

Start from the repository root, activate the project environment, and enter this directory:

```bash
source .venv/bin/activate
cd multimodal_inference_benchmarks
```

The Vane entrypoints use the installed `vane-ai` distribution from this
environment. Install or reinstall the project wheel before running them; the
benchmark scripts do not import Vane directly from the source checkout because
the native extension is provided by the installed wheel.

Install the benchmark dependencies:

```bash
for benchmark in \
  audio_transcription \
  document_embedding \
  image_classification \
  video_object_detection; do
  python -m pip install -r "$benchmark/requirements.in"
done
```

Install `s5cmd` separately and check `s5cmd version`. Install the model download CLI:

```bash
python -m pip install huggingface_hub==0.36.2 hf_transfer==0.1.9
```

Optional mirrors can be configured through `PIP_INDEX_URL` and `HF_ENDPOINT`.
Unset `HF_ENDPOINT` to retry against the official service if a mirror fails.

## Download the models first

Populate the model caches before measuring:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1

hf download openai/whisper-tiny
hf download sentence-transformers/all-MiniLM-L6-v2
```

Download the image and video model weights:

```bash
python - <<'PY'
from torchvision.models import ResNet18_Weights, resnet18
from ultralytics import YOLO

resnet18(weights=ResNet18_Weights.DEFAULT)
YOLO("yolo11n.pt")
PY
```

The benchmark runs reuse these local caches and do not include model downloading in their measured runtime.

## Download S3 data to the local machine

The examples below write data to the paths used by the benchmark defaults:

```bash
export BENCHMARK_DATA_ROOT=/data/multimodal_inference_benchmarks
mkdir -p "$BENCHMARK_DATA_ROOT"
```

Use `--limit` for a small download. Remove `--limit` when preparing the complete benchmark dataset.

### Audio

One Common Voice Parquet shard is approximately 0.5 GB.

```bash
python audio_transcription/download_common_voice_parquet.py \
  --out-dir "$BENCHMARK_DATA_ROOT/common_voice_17/parquet" \
  --batch-file /tmp/download_common_voice.s5cmd \
  --limit 1 \
  --run
```

### Document

```bash
python document_embedding/download_pdfs_from_metadata.py \
  --metadata "$BENCHMARK_DATA_ROOT/digitalcorpora/metadata" \
  --out-dir "$BENCHMARK_DATA_ROOT/digitalcorpora/pdf_dump" \
  --batch-file /tmp/download_pdfs.s5cmd \
  --limit 100 \
  --run
```

### Image

```bash
python image_classification/download_imagenet_from_metadata.py \
  --metadata "$BENCHMARK_DATA_ROOT/imagenet/metadata_file.parquet" \
  --out-dir "$BENCHMARK_DATA_ROOT/imagenet" \
  --batch-file /tmp/download_imagenet.s5cmd \
  --limit 100 \
  --run
```

### Video

```bash
python video_object_detection/download_hollywood2_videos.py \
  --out-dir "$BENCHMARK_DATA_ROOT/hollywood2/AVIClips" \
  --batch-file /tmp/download_hollywood2.s5cmd \
  --limit 10 \
  --run
```

A limited audio shard or video directory can be run directly by all three systems. For document and image benchmarks, remove `--limit` for a full three-system comparison because their metadata describes the complete dataset.

## Run locally on one machine

Choose a workload and use its settings below. Paths are relative to
`$BENCHMARK_DATA_ROOT`; document and image inputs use metadata plus local files.

| Directory | `INPUT_PATH` suffix | `BATCH_SIZE` | Additional setting |
| --- | --- | ---: | --- |
| `audio_transcription` | `common_voice_17/parquet` | 128 | None |
| `document_embedding` | `digitalcorpora/metadata` | 10 | `LOCAL_PDF_ROOT=$BENCHMARK_DATA_ROOT/digitalcorpora/pdf_dump` |
| `image_classification` | `imagenet/metadata_file.parquet` | 100 | `LOCAL_IMAGE_ROOT=$BENCHMARK_DATA_ROOT/imagenet/train` |
| `video_object_detection` | `hollywood2/AVIClips` | 32 | Row-group settings below |

Run all three engines with the same inputs and a unique output directory.
For example, from this directory:

```bash
(
  unset RAY_ADDRESS INPUT_LIMIT
  export NUM_GPU_NODES=1
  cd audio_transcription
  RUN_ID=$(date +%Y%m%d_%H%M%S)
  export INPUT_PATH="$BENCHMARK_DATA_ROOT/common_voice_17/parquet"
  export BATCH_SIZE=128
  VANE_RUNNER=ray OUTPUT_PATH="/tmp/vane_audio_$RUN_ID" python vane_main.py
  OUTPUT_PATH="/tmp/ray_data_audio_$RUN_ID" python ray_data_main.py
  OUTPUT_PATH="/tmp/daft_audio_$RUN_ID" python daft_main.py
)
```

For another workload, change the directory, input path, batch size and output
prefix, and export its additional settings before running the same entrypoints.
`NUM_GPU_NODES=1` uses one GPU actor; `VANE_RUNNER=ray` selects Vane's local Ray runner.

### Video object detection

Install the matching `native_media` provider wheel, which bundles its runtime,
on the coordinator and every Ray node before running Vane. The entrypoint explicitly loads that
provider and selects `image_backend='native'`; a missing provider is an error.
Video decoding uses the default Python video backend. Frames stay
`IMAGE('RGB', 640, 640)` through the detector's Arrow batches. After detection,
SQL expands the features and runs native `crop` and `encode_image` (PNG).
YOLO's floating-point `(left, top, right, bottom)` coordinates are truncated
toward zero, then converted to `(x, y, width, height)`. Crop dimensions must be
positive; out-of-bounds pixels are zero-filled.

PNG output preserves the crop's RGB pixels. Encoded bytes and sizes may differ
across systems: Vane's native encoder uses zlib's default compression and PNG's
None filter, while the Ray reference uses Pillow with compression level 2.
Record encoder settings when comparing runtimes or output sizes.

Set these additional options for video:

```bash
export PARQUET_ROW_GROUP_SIZE=122880
export PARQUET_ROW_GROUP_SIZE_BYTES=256MB
```

The Parquet row-group settings apply to the Vane entrypoint. The byte limit
makes DuckDB flush each writer's buffered row group once its estimated size
reaches the threshold, independently of its row count. Vane disables
insertion-order preservation for this write because DuckDB requires it when
`ROW_GROUP_SIZE_BYTES` is set; output row order is not part of this
benchmark's result contract.

## Batch-size sweep

Start with the batch size shown above and repeatedly double it. Keep the input data, model cache, GPU count, and all other settings unchanged. Stop when doubling no longer improves throughput or causes unacceptable GPU memory pressure. Run each setting at least three times and compare the median runtime.

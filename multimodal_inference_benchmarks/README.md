# Multimodal inference benchmarks

This directory compares equivalent Vane, Ray Data, and Daft pipelines on one local GPU machine.

This guide covers audio transcription, document embedding, image classification, and video object detection. `large_image_embedding` is currently excluded.

The benchmark entrypoints read local files only. The separate download scripts use anonymous access to copy public S3 data to the local machine.

## Benchmark background

The workload definitions and baseline methodology primarily follow Anyscale's [Benchmarking Multimodal AI Workloads on Ray Data](https://www.anyscale.com/blog/ray-data-daft-benchmarking-multimodal-ai-workloads), which compares Ray Data and Daft on image, document, audio, and video pipelines.

This project currently has a limited infrastructure budget and does not have access to a sufficiently high-throughput S3 environment. Instead of attempting to reproduce the article's distributed storage and cluster setup, this repository first downloads the public S3 datasets to local storage and then compares Vane, Ray Data, and Daft on one machine using local paths. Results from this setup describe only the local single-machine environment and should not be interpreted as a direct reproduction of the performance numbers in the original article.

## Prerequisites

Start from the repository root, activate the project environment, and enter this directory:

```bash
source .venv-system/bin/activate
cd multimodal_inference_benchmarks
```

Install the dependencies for each benchmark. The Tsinghua PyPI mirror is used here to speed up installation:

```bash
for benchmark in \
  audio_transcription \
  document_embedding \
  image_classification \
  video_object_detection; do
  python -m pip install \
    -r "$benchmark/requirements.in" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
done
```

Install the Hugging Face download tools:

```bash
python -m pip install \
  huggingface_hub==0.36.2 \
  hf_transfer==0.1.9 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

`huggingface_hub` already provides the `hf` CLI, so a separate `huggingface-cli` package is not required.

The dataset download scripts require `s5cmd`. Install it separately and ensure this command succeeds:

```bash
s5cmd version
```

## Download the models first

Use the Hugging Face mirror and accelerated transfer when downloading the Hugging Face models:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

hf download openai/whisper-tiny
hf download sentence-transformers/all-MiniLM-L6-v2
```

If the mirror returns a metadata or redirect error for a model, run `unset HF_ENDPOINT` and retry the same `hf download` command against the official endpoint.

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

Run commands from the corresponding benchmark directory. `NUM_GPU_NODES=1` uses one GPU actor. `VANE_RUNNER=ray` explicitly selects Vane's local Ray runner.

Before running:

```bash
unset RAY_ADDRESS
unset INPUT_LIMIT
export NUM_GPU_NODES=1
```

### Audio transcription

```bash
(
  cd audio_transcription
  RUN_ID=$(date +%Y%m%d_%H%M%S)
  export INPUT_PATH=/data/multimodal_inference_benchmarks/common_voice_17/parquet
  export BATCH_SIZE=128

  VANE_RUNNER=ray OUTPUT_PATH="/tmp/vane_audio_$RUN_ID" python vane_main.py
  OUTPUT_PATH="/tmp/ray_data_audio_$RUN_ID" python ray_data_main.py
  OUTPUT_PATH="/tmp/daft_audio_$RUN_ID" python daft_main.py
)
```

### Document embedding

```bash
(
  cd document_embedding
  RUN_ID=$(date +%Y%m%d_%H%M%S)
  export INPUT_PATH=/data/multimodal_inference_benchmarks/digitalcorpora/metadata
  export LOCAL_PDF_ROOT=/data/multimodal_inference_benchmarks/digitalcorpora/pdf_dump
  export BATCH_SIZE=10

  VANE_RUNNER=ray OUTPUT_PATH="/tmp/vane_document_$RUN_ID" python vane_main.py
  OUTPUT_PATH="/tmp/ray_data_document_$RUN_ID" python ray_data_main.py
  OUTPUT_PATH="/tmp/daft_document_$RUN_ID" python daft_main.py
)
```

### Image classification

```bash
(
  cd image_classification
  RUN_ID=$(date +%Y%m%d_%H%M%S)
  export INPUT_PATH=/data/multimodal_inference_benchmarks/imagenet/metadata_file.parquet
  export LOCAL_IMAGE_ROOT=/data/multimodal_inference_benchmarks/imagenet/train
  export BATCH_SIZE=100

  VANE_RUNNER=ray OUTPUT_PATH="/tmp/vane_image_$RUN_ID" python vane_main.py
  OUTPUT_PATH="/tmp/ray_data_image_$RUN_ID" python ray_data_main.py
  OUTPUT_PATH="/tmp/daft_image_$RUN_ID" python daft_main.py
)
```

### Video object detection

```bash
(
  cd video_object_detection
  RUN_ID=$(date +%Y%m%d_%H%M%S)
  export INPUT_PATH=/data/multimodal_inference_benchmarks/hollywood2/AVIClips
  export BATCH_SIZE=32
  export PARQUET_ROW_GROUP_SIZE=122880
  export PARQUET_ROW_GROUP_SIZE_BYTES=256MB

  VANE_RUNNER=ray OUTPUT_PATH="/tmp/vane_video_$RUN_ID" python vane_main.py
  OUTPUT_PATH="/tmp/ray_data_video_$RUN_ID" python ray_data_main.py
  OUTPUT_PATH="/tmp/daft_video_$RUN_ID" python daft_main.py
)
```

The Parquet row-group settings apply to the Vane entrypoint. The byte limit
makes DuckDB flush each writer's buffered row group once its estimated size
reaches the threshold, independently of its row count. Vane disables
insertion-order preservation for this write because DuckDB requires it when
`ROW_GROUP_SIZE_BYTES` is set; output row order is not part of this
benchmark's result contract.

## Batch-size sweep

Start with the batch size shown above and repeatedly double it. Keep the input data, model cache, GPU count, and all other settings unchanged. Stop when doubling no longer improves throughput or causes unacceptable GPU memory pressure. Run each setting at least three times and compare the median runtime.

#!/usr/bin/env bash
# Audio Transcription Benchmark - Optimised Runner
# Usage: ./run_benchmark.sh [output_suffix]
#
# Prerequisites:
#   - Ray cluster running (ray start --head / ray start --address=...)
#   - vane package installed (pip install . --no-build-isolation)
#   - torchaudio installed (pip install torchaudio==2.11.0+cu126 --index-url https://download.pytorch.org/whl/cu126)
#   - Whisper model cached locally (HF_HUB_OFFLINE=1)
#
# Key tuning levers:
#   CPU_UDF_MAX_CONCURRENCY  – CPU preprocessing actors (default: 32)
#   BATCH_SIZE               – rows per GPU inference batch (default: 64)
#   ROW_GROUP_SPLIT          – enable parquet row-group splitting (default: 10)
#
# Benchmark history (114K rows, 2 nodes, 2 GPUs):
#   2 actors / 10 tasks  / bs=64  → 1482.7s
#   32 actors / 10 tasks / bs=64  → 131.4s   (11.3×)
#   32 actors / 50 tasks / bs=64  → 119.9s   (12.4×)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Activate venv ──────────────────────────────────────────────────
if [[ -d "$REPO_ROOT/.venv-system" ]]; then
  source "$REPO_ROOT/.venv-system/bin/activate"
fi

# ── Output path ────────────────────────────────────────────────────
SUFFIX="${1:-$(date +%Y%m%d_%H%M%S)}"
export OUTPUT_PATH="${OUTPUT_PATH:-s3://datasets/multimodal_inference_benchmarks/audio_transcription_output/${SUFFIX}}"

# ── Scan & task partitioning ───────────────────────────────────────
export VANE_RAY_SCAN_TASK_SIZE_GROUPING="${VANE_RAY_SCAN_TASK_SIZE_GROUPING:-1}"

# ── Batch size ─────────────────────────────────────────────────────
export BATCH_SIZE="${BATCH_SIZE:-64}"

# ── HuggingFace offline (proxy-free workers) ───────────────────────
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ── Ray logging ────────────────────────────────────────────────────
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Audio Transcription Benchmark                             ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  %-58s ║\n" "OUTPUT_PATH=$OUTPUT_PATH"
printf "║  %-58s ║\n" "BATCH_SIZE=$BATCH_SIZE"
printf "║  %-58s ║\n" "SIZE_GROUPING=$VANE_RAY_SCAN_TASK_SIZE_GROUPING"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

python "$SCRIPT_DIR/vane_main.py" 2>&1 | tee "/tmp/audio_bench_${SUFFIX}.log"

echo ""
echo "Log saved to /tmp/audio_bench_${SUFFIX}.log"

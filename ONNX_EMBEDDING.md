# EmbeddingGemma 2 with ONNX Runtime

Use `provider="onnx"` with the separately provisioned
[`onnx-community/embeddinggemma-2-ONNX`](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX)
export. Text, images, 16 kHz mono audio, decoded video frames and ordered mixed
parts share a bundle-bound embedding space. Output widths are 128, 256, 512 and
768. The text graph returns the projected sentence embedding; encoder features
are intermediate inputs, never final retrieval vectors. Truncated vectors are
renormalized.

## Dependencies and artifacts

Install `vane-ai[onnx]` for CPU/CoreML or the alternative
`vane-ai[onnx_cuda]` for CUDA. Install one ONNX Runtime distribution per
environment. ONNX Runtime executes all three model graphs. The official
Transformers processor uses CPU Torch/torchvision for image/video preprocessing;
it does not load a PyTorch model or checkpoint. Model/tokenizer files must
already exist locally; remote code and model downloads are disabled at runtime.

Provision the six JSON sidecars (`config.json`,
`config_sentence_transformers.json`, `preprocessor_config.json`,
`processor_config.json`, `tokenizer.json`, `tokenizer_config.json`) plus these
graphs and each graph's `.onnx_data` sibling:

| dtype | Graph basenames under `onnx/` |
| --- | --- |
| `fp32` | `model`, `vision_encoder`, `audio_encoder` |
| `fp16` | The same basenames with `_fp16` |
| `q8` | The same basenames with `_quantized` |
| `q4` | The same basenames with `_q4` |
| `q4f16` | The same basenames with `_q4f16` |

Compute `model_sha256` from the exact provisioned bundle:

```bash
python -m vane.ai.providers._onnx_artifacts /models/embeddinggemma2-onnx --dtype fp32
```

The digest covers every graph, external weight file and processor/tokenizer
sidecar, in a canonical filename-to-SHA-256 mapping. It is checked before any
session starts. External tensors must reference their fingerprinted sibling
weight file. Quantized exports have distinct identities; do not reuse an
existing store merely because its vectors have the same width.

## Native execution

```python
import vane
from vane.ai import TextInput, embed

query = embed(
    TextInput(vane.lit("Where do giant pandas live?")),
    provider="onnx",
    model_path="/models/embeddinggemma2-onnx",
    model_sha256=bundle_sha256,
    dimensions=768,
    backend="cpu",
    dtype="fp32",
    input_type="query",
    max_length=2048,
)

with vane.connect() as connection:
    vector = connection.sql("SELECT 1").select(query.alias("embedding")).fetchone()[0]
```

`input_type="document"` is the default. Text queries receive
`task: search result | query: `; documents receive `title: none | text: `.
In mixed inputs the first text part receives the document prefix. Media-only
inputs receive no text prefix. Use the unified `embed` API with typed
`TextInput`, `ImageInput`, `AudioInput`, `VideoInput` or `MultimodalInput` for
relations. Native relation calls default to one subprocess actor. CUDA reserves
a Vane worker GPU; CPU and CoreML reserve none.

The explicit `backend` selects `cpu`, `cuda` or `coreml`.
CUDA selects logical device 0 in the worker's assigned visible devices;
use `CUDA_VISIBLE_DEVICES` to select a physical GPU for an external service.
An unavailable backend or unsupported graph fails. Session initialization and execution fallback are
disabled; non-CPU sessions also disable CPU execution-provider fallback.
ONNX Runtime supports multiple operating systems, but availability and graph/
quantization support depend on the selected execution provider. A platform
claim requires a real run there. The generic Vane subprocess launcher currently
requires POSIX; direct descriptors and the loopback service are separate paths.

## Shared service

Install `vane-ai[onnx,embedding_service]` (or
`vane-ai[onnx_cuda,embedding_service]`) in the service environment. Decoded-input
clients need only `vane-ai[embedding_service]`, without the model SDK. Select
`VANE_RUNNER=local` for native local query execution.

Create a private rendezvous directory and adapt
[`examples/onnx_service.json`](examples/onnx_service.json):

```bash
python -m vane.ai.embedding_service --config /private/onnx/config.json
```

Clients select `runtime="service"` and provide `service_path`, `service_id`,
`model_sha256` and the same `max_length`. Clients neither load the SDK nor
reserve worker GPUs. One service-owned thread creates the processor and three
sessions, runs inference, and releases them. Per-request query/document roles
share that engine. Bundle, model, dimensions and encoding mismatches fail before
inference. The service uses the existing authenticated PyZMQ/SharedMemory
transport and holds input ownership until committed work finishes. Cancellation
and shutdown join owned calls; failed cleanup remains retryable.

## Input limits and validation

Inputs require decoded uint8 HWC RGB images/frames and finite mono PCM arrays.
Video uses every supplied frame, in order, up to 32 frames with equal shape;
frame selection belongs to the caller and the model's visual-only video path
does not add timestamps. Audio is bounded to 180 seconds at 16 kHz. Each row
has at most 16 parts, each batch at most 256 rows/64 MiB, and rows obey
`max_input_bytes` (16 MiB by default). `max_length` is 1–8192 tokens including
media placeholders. Overflow fails without truncation. Character chunk
averaging, encoded media and uncertain-request replay are unsupported.

The [development workflow](DEVELOPMENT.md) runs the affected provider/service
contracts first. Real model checks are in `tests/ai/test_onnx_embedding.py`;
record the bundle digest, export revision, processor/runtime versions and
execution provider. CPU fixtures do not qualify a real model, GPU or platform.

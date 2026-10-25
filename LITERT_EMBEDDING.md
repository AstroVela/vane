# Local embeddings with LiteRT-LM

Install `vane-ai[litert]` to use `provider="litert"` with the typed
`vane.ai.embed` API. The adapter targets `litert-lm-api==0.18.0` and loads an
explicit local `.litertlm` file in one persistent subprocess actor. Select
`backend="cpu"` or `backend="gpu"`; initialization failures propagate directly.
Use the `local` runner for native execution. Native execution requires one
subprocess actor. This provider does not request CUDA resources, download
models, choose a backend automatically or substitute another model.

## Share one resident engine

`runtime="service"` connects CPU-only clients to a separately started LiteRT
engine. Ingestion, text queries and media clients share that engine. Install
`vane-ai[litert,embedding_service]` in the service environment; clients only need
`vane-ai[embedding_service]` for decoded inputs, without the LiteRT SDK, Pillow
or model file. Decoding files still requires the usual media extras.
Use [the service configuration](examples/litert_service.json) and
[the authenticated transport contract](EMBEDDING_DESIGN.md#explicit-service-configuration).
Set an absolute checkpoint path and its SHA-256 in the service configuration,
create a user-private rendezvous directory, then run:

```bash
mkdir -m 700 /tmp/vane-litert
python -m vane.ai.embedding_service --config examples/litert_service.json
```

On Windows replace the paths with absolute paths in a directory with a private
user ACL. The service transport and direct asynchronous clients use portable
PyZMQ/SharedMemory. The existing generic Vane subprocess UDF launcher still
requires POSIX; this adapter does not qualify local Windows query execution.
Existing Ray task/actor execution can use CPU-only service clients without
reserving a GPU for each client. Model SDK requirements remain platform specific.

Client calls pass the file hash and fixed encoding settings, without a model
path, backend, thread count or engine cache directory:

```python
service = dict(
    provider="litert", dimensions=256, runtime="service",
    service_path="/tmp/vane-litert/engine.json", service_id="embeddinggemma2-v1",
    model_sha256="e7a8a2204b91e0f96e92960e84a09a89212e1633dcb7575a9bf3378b4df77f4c",
    max_length=2048, vision_tokens_per_image=70,
)
documents = embed(TextInput(vane.col("text")), input_type="document", **service)
queries = embed(TextInput(vane.col("query")), input_type="query", **service)
```

The service checks model, dimensions, file SHA-256, `max_length` and
`vision_tokens_per_image` before reserving input memory. Clients must match the
service's encoding settings; a smaller client `max_input_bytes` only restricts
that client's input. The query/document role belongs to each request. Media
and composed inputs use document mode; query mode accepts one text part.
Native and service descriptors retain the same hash-bound output space.

One dedicated thread initializes, invokes and closes the SDK engine. Requests
serialize on that thread while the network loop remains responsive. Committed
requests drain through cancellation and service shutdown before buffers,
credits and the engine are released. A client never starts a local engine or
replays a request after uncertain completion.
If shutdown fails, the service retains the runtime and its SDK thread. A later
`aclose()` retries cleanup on that same thread; concurrent callers share the
active attempt. Cleanup must succeed before the service can restart.

| Model identifier | Native inputs |
| --- | --- |
| `litert-community/embeddinggemma-2-text-270m-litert-lm` | Text |
| `litert-community/embeddinggemma-2-text-vision-440m-litert-lm` | Text, image |
| `litert-community/embeddinggemma-2-740m-litert-lm` (default) | Text, image, audio |

The official [LiteRT-LM embedding API](https://developers.google.com/edge/litert-lm/embedding_models)
accepts ordered text, image and audio contents. Version 0.18's Python API has no
video input type. Native/service execution rejects video input and does not synthesize a video vector
by averaging images. `MultimodalInput` sends each row's ordered, decoded parts in
one native embedding request. Paths and encoded BLOB input are not media inputs;
decode media through Vane before embedding.

## Provision a model

Download a reviewed CPU/GPU `.litertlm` artifact from the model's repository.
Device-specific NPU artifacts are outside this adapter's contract. Retain the
model revision, filename and SHA-256 with the deployment. For the full model,
the reviewed CPU/GPU artifact is:

- Repository revision: `24d962e906c7d332c6428e71c9676855024569e2`
- File: `embeddinggemma-2-740m.litertlm`
- SHA-256: `e7a8a2204b91e0f96e92960e84a09a89212e1633dcb7575a9bf3378b4df77f4c`

Pass an absolute `model_path` and `model_sha256`. Planning performs no model I/O;
the worker verifies the file hash, native output dimensions, modalities and input
signature before constructing an engine. The vector-space descriptor includes
the model identifier and file hash. No tokenizer, processor or model weights ship
with Vane. Review the model's [Apache-2.0 license and model card](https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm).

```python
import vane
from vane.ai import TextInput, embed

vane.set_runner_local()
try:
    with vane.connect() as connection:
        rows = connection.sql("SELECT 'Mars is the Red Planet.' AS text")
        result = embed(
            rows,
            TextInput(vane.col("text")),
            provider="litert",
            model_path="/models/embeddinggemma-2-740m.litertlm",
            model_sha256="e7a8a2204b91e0f96e92960e84a09a89212e1633dcb7575a9bf3378b4df77f4c",
            dimensions=256,
            input_type="document",
            max_retries=0,
        )
        print(result.to_arrow_table())
finally:
    vane.teardown_runner()
```

Use `input_type="query"` for paired text queries. The checkpoint's document
prefix is `title: none | text: ` and query prefix is
`task: search result | query: `. Media options omit `input_type`; a composed
input prefixes its first text part as a document and preserves subsequent part
order. Images use lossless PNG transport to the SDK; mono 16 kHz floating PCM
uses PCM16 WAV transport, with clipping/rounding at the PCM16 endpoints.

Output dimensions are 128, 256, 512 or 768. The SDK truncates and L2-normalizes
the vector; Vane validates its shape, finite values and norm. Pair query and
document encoders at the same dimensions and file hash.

`max_length` selects 128, 256, 512, 1024, 2048 (default) or 8192 tokens.
The SDK uses `InputOverflowStrategy.ERROR`; it never truncates or averages
overlength input. `vision_tokens_per_image` selects 70 (default) or 140.
`num_threads` defaults to 2 for CPU; `cache_dir` may select an absolute directory.
`max_input_bytes` defaults to 16 MiB per row (maximum 64 MiB), and includes text
prefixes and encoded media. A batch supports at most 256 rows and 64 MiB of
decoded or encoded input. Images are RGB uint8 HWC with at most 16 Mi pixels;
audio is finite mono float32/float64 PCM in [-1, 1], at 16 kHz, up to 180 seconds.
The selected token budget can impose a lower effective limit on composed inputs.

Run [the search example](examples/litert_embedding_search.py) outside the source
checkout against an installed wheel. See [DEVELOPMENT.md](DEVELOPMENT.md) for
installation and the CPU/real-model validation commands.

## Official HTTP server

`runtime="http"` connects to the official `litert-lm==0.18.0` CLI server's
`/v1/embeddings` endpoint. Install `vane-ai[embedding_http]` in clients, with the
usual decoding extras; they require neither the LiteRT SDK nor local weights.
Install the CLI separately in the model service environment. Provision/import
the reviewed model under its declared model ID, select its supported backends
explicitly in the CLI configuration, then run `litert-lm serve`.
See [the official HTTP contract](https://developers.google.com/edge/litert-lm/cli/openai_server).

```python
remote = dict(
    provider="litert", runtime="http", base_url="http://127.0.0.1:9379/v1",
    model="litert-community/embeddinggemma-2-740m-litert-lm", dimensions=768,
    model_sha256="e7a8a2204b91e0f96e92960e84a09a89212e1633dcb7575a9bf3378b4df77f4c",
    request_batch_size=8, max_concurrency_per_actor=1, timeout=300,
)
vectors = embed(
    MultimodalInput([VideoInput(vane.col("frames")), AudioInput(vane.col("audio"))]),
    **remote,
)
queries = embed(TextInput(vane.col("query")), input_type="query", **remote)
```

Use the usual `vane.ai` typed-input imports. The endpoint must serve the configured
model artifact. The file hash declares the paired vector space; unlike the local
worker or shared-memory service, the official HTTP API cannot attest that hash.
Optional authentication uses `load_provider("litert", api_key=...)` or an
application-side `LITERT_API_KEY` snapshot. HTTP key validation does not affect
native/service initialization. The official CLI itself does not add authentication.

The HTTP adapter accepts decoded text, RGB images, mono 16 kHz floating PCM and
ordered clips. A clip expands to ordered lossless PNG image parts, alongside
PCM16 WAV audio and optional text in one input row. The server computes one native
joint vector for that row. This is an image-sequence encoding: LiteRT 0.18 has no
native video/timestamp input, and timestamps are retained for validation/provenance
rather than inserted into the model prompt. It does not average frame vectors.
Text is prefixed once using the same document/query prefixes as native execution.
Unary text preserves empty/whitespace semantics; absent/blank joint parts are omitted.
Encoded BLOBs require upstream decoding for this HTTP adapter.

The 270M model accepts text; the 440M model additionally accepts images and
image-sequence clips; the 740M model also accepts audio. Clips contain 1–64 frames,
indices increase and timestamps may repeat but cannot decrease. Model token limits
can impose a substantially smaller effective limit. Media retain their input order.

`request_batch_size` (default 8, maximum 256) bounds independent rows in one HTTP
request; it does not declare a GPU execution batch. The official server retains
one engine and its SDK owns execution scheduling. Client concurrency defaults to
one. Decoded and encoded media/prefixes must fit `max_input_bytes` (default 16 MiB,
maximum 64 MiB); Base64/JSON must also fit `max_request_bytes` (default 32 MiB,
maximum 256 MiB). Every row is validated before client creation or dispatch,
including under `on_error="ignore"`. Resource allocation failures remain fatal.
Responses must preserve row indices and return finite unit vectors of the
requested dimensions. The official CLI reports zero token usage; this adapter
does not infer token throughput from those counters.

The official HTTP endpoint does not expose `max_length`, `vision_tokens_per_image`,
backend/thread/cache selection or overflow policy. These are rejected as HTTP client
options. The server's default overflow policy is an error. Encoding settings belong
to the serving deployment, not the client's HTTP payload.

### Configure the embedded input-length default

The reviewed file's `EmbeddingMetadata.max_input_length` is 1024. This is a
Protobuf field inside the binary `.litertlm` model package, alongside its weights,
graphs and tokenizer; it is not a separate JSON setting. The package already
contains real 128/256/512/1024/2048/8192-token encoder signatures. The embedded
default restricts signature selection when no explicit engine override is supplied;
1024 is not a GPU hardware or exported-graph limit.

The native Python API accepts `EmbeddingEngine(..., max_input_length=8192)`.
The official 0.18 HTTP engine factory does not forward that setting, and adding
`max_length` to Vane's HTTP options cannot configure the server.
For the exact reviewed artifact, the offline
[deployment-copy example](examples/litert_http_model_metadata.py) changes only that
embedded default to 8192. Run it in the separately provisioned
`litert-lm==0.18.0` environment:

```bash
export LITERT_LM_DIR="$PWD/litert-8192"
python examples/litert_http_model_metadata.py \
  /models/embeddinggemma-2-740m.litertlm \
  "$LITERT_LM_DIR/models/litert-community--embeddinggemma-2-740m-litert-lm/model.litertlm"
```

The destination is the deployment copy that the official CLI reads under its
declared model ID. The example refuses other source hashes and existing targets,
verifies the resulting whole-file hash and publishes the new file only after
verification. The two identities are:

| Artifact | SHA-256 |
| --- | --- |
| Original reviewed file | `e7a8a2204b91e0f96e92960e84a09a89212e1633dcb7575a9bf3378b4df77f4c` |
| 8192-default deployment copy | `6d738925f2686fa545e50dee7e26e0c4ee8af43b69e63833783ab463e92f63ba` |

Exactly one metadata byte changes; every other section, including the weights,
graphs and tokenizer, remains byte-identical. No model re-export, official source
change or custom inference HTTP server is required. Set Vane's `model_sha256`
to the deployment copy's hash explicitly, pair query/document encoders with that
deployment and use a new index. HTTP still does not attest the declared hash.

### Select encoder backends on the server

In a CLI config file such as `litert-http-server.json`, the tested experimental
configuration for the 740M model on an RTX 4080 SUPER is:

```json
{
  "default": {
    "backend": "gpu",
    "vision_backend": "gpu",
    "audio_backend": "gpu",
    "cache": "disk"
  }
}
```

```bash
litert-lm serve --config litert-http-server.json --host 127.0.0.1 --port 9379
```

All three main encoders ran on WebGPU, including the 8192-token encoder signature.
Small adapters and media decoding/preprocessing still used CPU. This corrects the
earlier description that vision/audio were CPU-only. A backend loading successfully
does not establish numerical equivalence or retrieval quality; the experimental
GPU results below do not qualify it as a default.

The official 0.18
[`ComputeEmbeddingBatch` implementation](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.18.0/runtime/core/embedding_engine_impl.cc)
processes the independent rows sequentially. Eight rows in one HTTP request
therefore reduce transport calls but do not constitute one eight-row GPU tensor
batch. Increasing HTTP concurrency does not establish concurrent SDK execution.

### Recorded 100-video HTTP qualification

An independently admitted single-video ingest and fresh 100-video batch completed
on the local runner with official CLI/API/builder 0.18.0 and the metadata-only
deployment copy. Both LiteRT and a fresh vLLM control produced 100 ready videos,
zero failed/unknown jobs and 274 complete visual/raw-sound clips. Each 100-video
case ran 20 actual flat/grouped vector queries and repeated them after reopening
storage, with unchanged job reports and result-ID order.

The fixed input corpus totals 3177.863 seconds. Both arms used windows of at most
15 seconds, 1 FPS, 224-pixel Context previews, 12 local CPUs/SQL threads, two
preprocessing workers, one embedding actor, invocation/HTTP batches of eight,
observed HTTP concurrency one and a 256 MiB stage budget. A private benchmark
observer applied the same 8 GiB memory-admission cap; it is not a public Vane or
Context production setting. ASR/OCR/subtitles were not used.

| Measurement | LiteRT, three encoders on GPU | vLLM CUDA control |
| --- | ---: | ---: |
| Batch submission to ready | 519.008 s | 174.911 s |
| Model HTTP active union | 414.297 s | 122.852 s |
| Full case including startup/search/reopen | 613.318 s | 268.811 s |
| Fixed 10-query Recall@1 / Recall@10 | 0.6 / 0.9 | 0.9 / 1.0 |

All 3226 LiteRT PNG images and the timeline-aligned PCM16 input samples matched
retained evidence; every response vector matched its own stored vector. The vLLM
arm's 274 MP4/WAV input fingerprints matched the frozen control and its minimum
same-runtime reference cosine exceeded 0.9999. These are functional/input-ownership
checks, not cross-provider equivalence.

The LiteRT GPU probe's cosine against the CPU vision/audio reference was
0.9956053, below this experiment's original 0.9999 gate. Same-GPU repetition and
single-versus-two-row batch cosine was 1.0. Retrieval also failed the retained
Recall@1 >= 0.9 / Recall@10 = 1.0 criterion, so the GPU candidate is not promoted
to a default. Ten fixed queries are a small sample, not a full-corpus quality score.
The two arms use different quantized/BF16 formats, PNG-sequence/MP4 representations,
sampling/pixels and processors; the timings do not isolate runtime algorithms.
HTTP-active time includes preprocessing, transfer, queueing and inference, not
pure GPU compute. CPU vision/audio 100-video throughput was not measured.

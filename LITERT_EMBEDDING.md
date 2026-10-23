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
video input type. Vane rejects video input and does not synthesize a video vector
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

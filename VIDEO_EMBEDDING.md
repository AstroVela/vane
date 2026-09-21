# Video and text embeddings

`vane.ai.embed_video` maps one ordered clip to one fixed-size `FLOAT[d]`
vector. `vane.ai.embed` encodes text with the matching model so text queries
can retrieve video clips in the same vector space. Both use Vane's existing
UDF execution, including the default Ray runner.

## Input contract

Each input row contains a `LIST<STRUCT>` with these required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `frame_index` | `BIGINT` | Nonnegative presentation index, strictly increasing within the clip |
| `frame_time` | `DOUBLE` | Nonnegative finite timestamp in seconds, nondecreasing within the clip |
| `data` | `IMAGE` | Decoded UInt8 RGB pixels |

`video_frames()` already returns this schema, with additional provenance.
Alternatively, aggregate streaming frame rows with an explicit `ORDER BY`
inside `list(...)`. Keep clip ID, source, start/end times and sampled frame
records alongside the vector for retrieval evidence. Vane does not store an
index or infer clip boundaries as part of embedding.

NULL clips produce NULL vectors without loading a model. Empty clips, NULL
frames, out-of-order frames and invalid image types raise errors, including
with `on_error='ignore'`. Decoded input is bounded by the provider's
`VideoInputSpec`; the initial Cosmos adapter allows at most 64 MiB per clip.
This limit is measured on the Arrow pixel buffers before copying them.

The API does not decode paths, select frames, repeat short clips, average
image vectors or generate captions implicitly. Temporal sampling belongs
upstream of the embedding call. Arbitrary providers can implement
`Provider.get_video_embedder()` and the `VideoEmbedderDescriptor` /
`VideoEmbedder` contracts in `vane.ai.protocols`; each descriptor declares its
dimensions, frame limits and GPU resources without loading the model.
`VideoClip` and `VideoInputSpec` are exported from `vane.ai`.

## Cosmos-Embed1-224p

The initial adapter is available through `provider='transformers'`, using
the official model's video and text encoders. It supports exactly eight
frames of the same shape per clip and produces 256-dimensional vectors.
Different clips may have different resolutions; the official processor
resizes each to 224 × 224. Text is limited to 128 tokens including special
tokens. Longer queries and text chunk averaging are rejected explicitly.

Install `vane-ai[cosmos]`, with a CUDA-enabled PyTorch installation, on every
worker. Video decoding through the Python backend additionally requires
`vane-ai[video]`. The Cosmos extra uses Transformers 4.51.3, the version
validated with the model's custom code. Model assets and custom code are
downloaded by Hugging Face on first execution unless cached.

Set `trust_remote_code=True` and an explicit full commit SHA after reviewing
that model code. Set `device='cuda'` and `dtype='float16'` or `'float32'`;
unsupported hardware or precision raises an error. Each model actor requests
one GPU, and `cuda` refers to the GPU visible inside that worker. The default
video batch size is one clip. Increase `batch_size` only within the worker's
memory capacity. Model instances are reused across batches within an actor.

```python
import vane
from vane.ai import embed, embed_video

options = dict(
    provider="transformers",
    model="nvidia/Cosmos-Embed1-224p",
    revision="787e0b996f5260a71ad474a283c90539a2e12986",
    trust_remote_code=True,
    device="cuda",
    dtype="float16",
)

# clips: one row per clip, with an explicitly selected eight-frame list.
vectors = clips.select(
    vane.col("clip_id"),
    embed_video(vane.col("frames"), **options).alias("embedding"),
)
queries = query_rows.select(
    vane.col("query_id"),
    embed(vane.col("text"), **options).alias("embedding"),
)
```

Use identical model, revision and precision for clips and queries. Relation
forms `embed_video(clips, vane.col('frames'), ...)` and
`clips.embed_video(vane.col('frames'), ...)` append an `embedding` column and
retain the input columns. Both accept `output_column=...`.

SQL uses the same contract:

```sql
SELECT clip_id, ai_embed_video(
    frames,
    provider => 'transformers',
    model => 'nvidia/Cosmos-Embed1-224p',
    options => {
        revision: '787e0b996f5260a71ad474a283c90539a2e12986',
        trust_remote_code: true, device: 'cuda', dtype: 'float16'
    }
) AS embedding
FROM clips;
```

Other video models, including other Cosmos resolutions, are currently
unsupported. Existing text and image embedding providers retain their own
model capabilities. Selecting OpenAI or Google for `embed_video` raises a
capability error.

## Offline workers and real-model validation

For an existing Ray cluster, provision dependencies, caches and source video
access on each worker before running the query. `cache_folder` refers to a
worker-accessible Hugging Face Hub cache directory. It is not an upload or
cache distribution mechanism.

The model's QFormer also loads `bert-base-uncased/config.json` from the default
Hugging Face cache without forwarding `cache_folder` or `local_files_only`.
Prepare that auxiliary config on every worker. For `local_files_only=True`,
set `HF_HUB_OFFLINE=1` **before starting worker processes**; the adapter rejects
that option when Transformers is not in offline mode. This prevents the
upstream constructor from making an unexpected network request.

The opt-in integration test decodes a real video with `video_frames`, explicitly
selects eight evenly spaced frame indices, computes video and text embeddings
on the default Ray runner, and checks the correct caption ranks first. It
runs both FP16 and FP32. Prepare these assets separately:

- The model snapshot at the revision above in the Hub cache, including code,
  tokenizer and weights, plus the auxiliary BERT config in the default cache.
- The [javelin video used in NVIDIA's model card](https://upload.wikimedia.org/wikipedia/commons/3/3d/Branko_Paukovic%2C_javelin_throw.webm),
  SHA256 `75ab52aa5868d866b9974b922bdf292b501d2e26a13396b5638a3c58058aae8a`.

```bash
HF_HUB_OFFLINE=1 \
VANE_TEST_COSMOS_CACHE=/shared/huggingface/hub \
VANE_TEST_COSMOS_VIDEO=/shared/fixtures/javelin.webm \
scripts/run_installed_pytest.sh tests/ai/test_cosmos_video_embedding.py -v
```

The ordinary release gate uses deterministic CPU providers and SDK substitutes
for API validation, NULL handling, serialization, model invocation and Ray
transport. It requires no weights, CUDA, credentials or network. Model code,
weights and test media are not bundled with Vane. See [THIRD_PARTY.md](THIRD_PARTY.md)
for their separate terms and the [official model card](https://huggingface.co/nvidia/Cosmos-Embed1-224p)
for architecture and model limitations.

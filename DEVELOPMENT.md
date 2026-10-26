# Development

Vane contains Python, pybind11, and a modified DuckDB C++ engine. A native build also links Arrow Flight, gRPC, and selected DuckDB extensions.

## Prerequisites

- Linux x86-64 for the complete build and test path
- macOS arm64 and Windows x86-64 for the native build and distributed-test paths used by CI
- Python 3.10 through 3.14; Python 3.12 is recommended and is the primary development version
- Git with `git subtree` support
- A C++20 compiler and CMake 3.29+; Ninja and ccache on Linux/macOS, or Visual Studio 2022 on Windows
- vcpkg at the baseline pinned in `vcpkg.json`

The DuckDB engine fork is included directly under `external/duckdb`; a normal
clone contains all source needed for the build.

Bootstrap native dependencies from the repository root:

```bash
bash scripts/bootstrap_vcpkg.sh
```

The helper checks out the exact baseline from `vcpkg.json`, installs into
`vcpkg_installed`, and verifies the committed native-dependency license bundle.
It selects the host platform's release-only target and host triplets by default,
including `x64-linux-release` on Linux x86-64, `arm64-osx-release` on Apple
Silicon, and `x64-windows-static-release` on Windows x86-64. Set
`VCPKG_TARGET_TRIPLET=x64-linux` when both release and debug target dependency
builds are needed; `VCPKG_HOST_TRIPLET` independently overrides the host tools
triplet. CMake selects only the requested or platform-default triplet, without
searching other installed triplets. Set `VCPKG_INSTALLED_DIR` to select an
alternative dependency installation for bootstrap, CMake, and license tools.
Relative installation paths are resolved from the repository root.
When intentionally changing native dependencies, regenerate the bundle
with `python scripts/sync_vcpkg_licenses.py` and review its diff. Successful port
builds are cached before their temporary build and package trees are removed,
keeping bootstrap within hosted-runner disk limits.

Run `python -I scripts/check_copyleft.py` after source or dependency changes.
The bootstrap also checks installed GPL-family notices against the reviewed
inventory. Follow [COPYLEFT.md](COPYLEFT.md) before updating license hashes,
selecting a different license alternative, or adding codec features.
When checking an optional native dependency tree, pass `--share-dir <share>`
and repeat `--feature <vcpkg-feature>` for every selected feature so missing
required transitive notices are rejected as well.

## Incremental package build

The [unified embedding design](EMBEDDING_DESIGN.md) tracks the typed input
contract, provider/protocol boundaries, Context integration and delivery gates.
Native output changes use `tests/fast/test_embedding_outputs.py` together with
related text/media, SQL, HTTP and resource-boundary contracts. Run the real
subprocess cases on `local-fast` with Ray startup forbidden. Named outputs are
fixture-qualified; this does not qualify a checkpoint or add native multi-output
support to an existing serving protocol. Python-only output typing requires a
noneditable package reinstall, without a native rebuild.
For the persistent native service, see its
[configuration and ownership contract](EMBEDDING_DESIGN.md#explicit-service-configuration).
Run `tests/fast/test_embedding_service.py`,
`tests/fast/test_embedding_service_transport.py`,
`tests/fast/test_native_multimodal.py` and the affected unified/HTTP embedding
tests through the installed launcher.
The service tests use real loopback PyZMQ sockets, Python SharedMemory and
subprocesses with CPU SDK fixtures. Install the `embedding_service` extra for
these tests; the development `test` group also includes it. The service transport
targets Windows, macOS and Linux; the public subprocess backend tests still
require POSIX because the generic UDF launcher passes file descriptors.
Real processor tests require the optional Transformers
test dependencies; they do not qualify a GPU checkpoint or throughput.

Create and activate a virtual environment, then reuse a persistent native build directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

export SKBUILD_BUILD_DIR="$PWD/build/python-release"
export SKBUILD_CMAKE_BUILD_TYPE=Release
python -m pip install . --no-build-isolation -v
```

Do not use `pip install -e`. An editable install can cause Ray workers to invoke the build backend during import and delay actor startup.

Python-only changes do not require a native rebuild, but reinstall the
non-editable package so the test environment receives them. Changes below
`src/vane_py/` or `external/duckdb/src/` require an incremental native build.

For Apple Silicon EmbeddingGemma 2 text/image/audio/video inference, install the
optional `mlx` extra. Run `tests/fast/test_mlx_provider.py`,
`test_mlx_image_provider.py`, `test_mlx_audio_provider.py` and
`test_mlx_video_provider.py` through the installed launcher before the base
release gate; these contracts need no model SDK or Metal.
The real checkpoint tests in `tests/ai/test_mlx_text_embedding.py` require
`VANE_RUN_MLX_TESTS=1`, a complete fixed-revision HF cache, and
`-m 'metal and external_service'`. They exercise BF16 reference parity,
padding, retrieval, subprocess actor reuse and teardown. Use the complete
`mlx-community/embeddinggemma-2-bf16` snapshot at revision
`1a4ffddb7905d3f63486748deabe091a01fb6201`, with mlx 0.32.4, mlx-vlm 0.7.8
and transformers 5.18.0. Download it explicitly with Hugging Face Hub before
setting `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; workers share `HF_HOME`.
Inference requires native arm64 macOS and Metal, `dtype="bfloat16"`, one
`subprocess_actor`, and 768 output dimensions. The document prefix is
`title: none | text: `; the query prefix is `task: search result | query: `.
The default 2048-token limit includes prefixes and special tokens; overlength
inputs fail before inference. Other models, precisions and backends are rejected. Run manual subprocess examples outside the checkout so worker
imports resolve to the installed package as described below.
MLX and SQR pin different Transformers versions; use separate environments.

Ordered joint inputs use `embed(MultimodalInput([...]), provider="mlx")` or the
provider's `get_multimodal_embedder()` / `embed_multimodal()` contracts. One row
contains 1–16 ordered text/image/video/audio parts, including repeated media,
with a combined 64 MiB input budget and one 768-dimensional float32 unit vector.
It is a document encoding: no `input_type` option. A nonblank text part adds
`title: none | text: ` once; other text and media remain at their original
positions without inserted separators. Media-only inputs have no prefix.
NULL parts and blank text are absent; an empty row fails. Text cannot contain
the checkpoint's image/video/audio placeholders or begin/end markers, including
markers formed across text parts. Existing unary text semantics are unchanged.

Joint inference shares one BF16 model, one processor and the existing Metal
forward/output/close path. Images are nested by input row; clips and independent
float32 waveforms retain flattened part order. Exact processor keys, expanded
tokens, complete ordered patch grids, frame counts, original waveform masks and
per-part media structures are checked before forward. The 2048-token limit
includes all expanded media, prefixes, special tokens and padding width.
Joint uses only the fixed revision above. The four unary descriptors retain their
existing output identities and paired query roles. Start with batch size one,
one actor and concurrency one; no global model sharing is implied.

Run `test_mlx_multimodal_provider.py` with the four existing fast files. Run
`tests/ai/test_mlx_multimodal_embedding.py` in its own installed-launcher process,
with `VANE_RUN_MLX_TESTS=1 -m 'metal and external_service'`, and the five Metal
files sequentially. The new file compares independent CHW SDK references,
heterogeneous batches, repeats/order, public relation/expression/SQL actors,
actual actor OOM injection, cancellation and teardown. Set
`VANE_MLX_JOINT_FIXTURES` to a retained independently SDK-qualified corpus
containing `manifest.json` (`frozen_before_implementation`, three `joint` entries
with source/subtitle hashes and `input_file`) and `joint-sdk-reference.npz`
(`vectors`, paired text `queries`, `scores`). Each input NPZ retains decoded
`frames`, `frame_times`, `frame_indices`, mono `audio`, and the exact joint `text`.
Explicit opt-in fails when required assets are missing. This fixture establishes
execution and parity for its declared material; report natural-recording quality
qualification separately. Context uses its existing video/audio/subtitle pipeline;
it does not add image joint request fields. MLX HTTP and cross-request batching
remain outside this adapter. Explicit service execution is described below.

For local LiteRT-LM inference, see [the provider contract](LITERT_EMBEDDING.md).
For the official LiteRT HTTP adapter, complete two clean reviews before the noneditable
package build. Run `tests/fast/test_litert_http.py`, the LiteRT native/service contracts
and affected HTTP/request/media tests through the installed launcher. Context
qualifies actual ingestion/search separately; fixture HTTP contracts do not qualify
checkpoint throughput or device capacity. HTTP clients need no LiteRT SDK.

For the persistent LiteRT service, run `tests/fast/test_litert_service.py`
alongside the service transport tests. It exercises real PyZMQ and shared
memory with an authored CPU SDK fixture, including fixed encoding identities,
per-request query/document roles, one SDK-owning thread and cancellation drain.
Complete two consecutive clean reviews before building/installing the wheel.
The opt-in `tests/ai/test_litert_embedding_service.py` uses the real checkpoint,
public APIs and separate service/client processes. It requires the existing
LiteRT model environment variables and the `embedding_service` extra. GPU cases
also require `VANE_RUN_LITERT_GPU_TESTS=1` and `-m 'external_service and gpu'`;
missing prerequisites after opt-in fail without CPU fallback. Run the Context
service acceptance separately with Ray startup forbidden. No native source is
changed by this adapter; platform simulation does not qualify a real Mac/Windows host.
Install the optional `litert` extra, which pins `litert-lm-api==0.18.0`.
Run `scripts/run_installed_pytest.sh tests/fast/test_litert_provider.py` before
the base release gate. These CPU contracts require no model SDK or weights.
For real CPU SDK parity, multilingual retrieval, image/audio/composed inputs,
public subprocess execution and teardown, explicitly provision the full
CPU/GPU artifact documented in that contract and set `VANE_RUN_LITERT_TESTS=1`,
`VANE_TEST_LITERT_MODEL_PATH` and `VANE_TEST_LITERT_MODEL_SHA256`. Then run
`scripts/run_installed_pytest.sh tests/ai/test_litert_embedding.py -m external_service`.
Missing assets or SDKs fail after opt-in; no download or model/backend fallback
occurs. CPU tests do not qualify Metal, another accelerator, or representative
multimodal retrieval quality. This is Python-only and needs no native rebuild.

Image encoding uses the same public `embed(ImageInput(...), provider="mlx")`
entry point and paired vector space. One decoded RGB `uint8` HWC array is one
row, with at most 16 Mi pixels. Noncontiguous and read-only arrays are accepted;
encoded bytes and paths first require the existing ImageFile decoder and the
`image` extra (`vane-ai[mlx,image]`). Image options omit `input_type`; paired
text queries retain `input_type="query"`. Workers load the SDK's explicit
`EmbeddingGemma2Processor` from the fixed local snapshot, with nested singleton
image batches, NumPy/Pillow preprocessing, no remote code and no truncation.
The same `max_length` bounds expanded visual tokens before model forward.

Run `tests/ai/test_mlx_image_embedding.py` separately with the text test's
`VANE_RUN_MLX_TESTS=1` and `-m 'metal and external_service'` settings. Explicit
opt-in requires the SDKs, full cache and actual Metal; missing prerequisites fail.
Tests compare the real SDK and provider at cosine >= 0.999, mixed-size batches,
input reordering, public expression/Relation/SQL output and subprocess model/
processor reuse and exit. The CPU contracts include malformed processor outputs,
input budgets, safe diagnostics and fatal allocation failures under both error
policies. Ordinary CPU installations need none of the optional model SDKs.

Audio uses `embed(AudioInput(...), provider="mlx")` with decoded mono
float32/float64 PCM shaped `(samples, 1)`, sample rate 16000, 161–480000 samples,
at most 4 MiB per clip and finite values in `[-1, 1]`. Decode/resample files
upstream with the existing AudioFile API and `vane-ai[mlx,audio]`.
The SDK receives independent nested waveforms with both text and audio
truncation disabled. The provider validates the complete padded mel length,
every valid/padding mask bit and twice-subsampled audio token count before
forward. Valid silence is encoded; unsupported short tails and long clips fail.

Video uses `embed(VideoInput(...), provider="mlx")` with 1–12 already sampled
RGB `uint8` HWC frames, equal H/W within each clip, at most 16 Mi pixels per
frame and 64 MiB per clip. Frame indices are nonnegative strictly increasing
integers; finite nonnegative timestamps may repeat but cannot decrease.
The processor receives one four-dimensional array per clip with explicit HWC
layout, `do_sample_frames=False` and timestamps disabled. Complete frame counts,
patch grids/padding and expanded tokens are checked before inference. Encoded
media requires upstream decoding; install `vane-ai[mlx,image,audio,video]` for
the corresponding file APIs. Image queries use the paired video vector space.

Run `tests/ai/test_mlx_audio_embedding.py` and `test_mlx_video_embedding.py` in
separate installed-launcher processes using the text/image opt-in settings.
Set `VANE_MLX_AV_FIXTURES` to the externally retained, independently SDK-qualified
audio/video corpus directory with its frozen `manifest.json`. Missing SDKs,
Metal, checkpoint or required corpus after opt-in fail explicitly. The four
modality suites run sequentially, covering SDK parity at cosine >= 0.999,
boundaries, public expression/Relation/SQL APIs and owned worker cleanup.
Do not install the CLAP extra alongside MLX: their Transformers requirements
conflict. Context sound/visual selection and clip policy remain upstream work.
Each descriptor owns its actor's full checkpoint and processor. Four modality
profiles can therefore create several model copies; they do not share one
global model or a Metal memory reservation. Start with small batches and run
the four qualification suites sequentially, joining all workers between them.

Video clipping regression checks include fractional starts ending exactly at a
stream's rational endpoint. Duration quantization must not manufacture an
overrun, and an actual extension by an output clock tick must still fail. After
two clean reviews, build/install the Python wheel noneditably and run
`tests/fast/test_video_clip.py`; no native rebuild or unrelated suite is needed
for a Python-only endpoint comparison change.

For the optional `qwen3_vl` provider, run
`scripts/run_installed_pytest.sh tests/fast/test_qwen3_vl.py` before the base gate.
Its CPU fixtures check lazy registration, tokenizer postprocessing, trained
weight validation and media budgets without downloaded weights. Real processor
tests also check exact video tokens, temporal boundaries, timestamps, mixed
batches and unchanged image/text processing with an authored tokenizer/template.
Tensor/SDK cases require the `qwen3_vl` extra. Separately validate a real local
checkpoint on a BF16 CUDA worker through the default Ray runner: compare query
and video inputs/vectors with that checkpoint's reference encodings, then
exercise video ingestion, search and restart with the paired Context profile.
Query parity alone does not validate video encoding. Record model/dependency
identities and sampling settings; CPU fixtures do not establish model quality.
See the [provider example](examples/qwen3_vl_sqr.py) for its fixed contract.

## Faster-Whisper speech extraction

`vane.ai.media_models.FasterWhisperASR` enables Faster-Whisper's local Silero
VAD by default. No detected speech returns an empty content list; extraction
errors still raise. The SDK restores detected speech to the original audio
positions before Vane rebases segments to the asset timeline. Original sound
samples and other embedding inputs are not filtered by this ASR option.
`vad_speech_pad_ms=800` retains context before and after detected speech so a
quiet word onset is less likely to be cut off; configure a nonnegative integer
number of milliseconds explicitly when tuning for a recording.
Use `vad_filter=False` to explicitly transcribe without VAD. Context passes this
option through `transcription.options`; it needs no separate pipeline.

Run the affected `tests/fast/test_ai_media_models.py` through
`scripts/run_installed_pytest.sh`. Complete two clean reviews before installing
the Python-only wheel noneditably; no native rebuild is needed. For the real
CUDA controls, provision a local Faster-Whisper checkpoint and a short, mono
16 kHz PCM WAV with known spoken text, then set
`VANE_RUN_FASTER_WHISPER_TESTS=1`, `VANE_FASTER_WHISPER_MODEL`,
`VANE_FASTER_WHISPER_SPEECH_WAV`, and `VANE_FASTER_WHISPER_EXPECTED_TEXT`.
Run `scripts/run_installed_pytest.sh tests/ai/test_faster_whisper_asr.py -m 'gpu and external_service'`
with CUDA dependencies and devices available. These controls check exact
silence, short speech, padding/time rebasing and unchanged sound samples;
they do not certify general ASR accuracy. Only related checks are required;
no full/release suite, Ray tests or serving-engine restart is needed.

## Building a loadable extension artifact

For the Gravitino Catalog and Fileset connector, see
[Gravitino integration](GRAVITINO.md). It uses the same optional artifact and
provider-wheel workflow below.

For `native_media`, first prepare its separate SDK and shared libraries using
[the media build guide](NATIVE_MEDIA_EXTENSIONS.md#build-and-package).
Run `tests/fast/test_ray_native_runtime_replacement.py` separately from
shared-cluster tests with the signed provider fixture; it owns two-node clusters.

`VANE_LOADABLE_EXTENSIONS` builds selected DuckDB extensions as self-contained
`.duckdb_extension` artifacts without linking them into `vane._native`. The
default is empty, so base Vane builds and wheels do not contain staged optional
extensions. DuckDB's pinned source configuration is preserved for external
extensions such as `httpfs`. For example, build and exercise both the in-tree
`tpch` artifact and the externally sourced `httpfs` artifact:

```bash
export SKBUILD_BUILD_DIR="$PWD/build/python-release"
export SKBUILD_CMAKE_BUILD_TYPE=Release
uv pip install . --no-build-isolation \
  '-Ccmake.define.VANE_LOADABLE_EXTENSIONS=tpch;httpfs'
cmake --build "$SKBUILD_BUILD_DIR" --target vane_loadable_extensions
VANE_TEST_LOADABLE_EXTENSION_PATH=\
"$SKBUILD_BUILD_DIR/vane_extensions/tpch.duckdb_extension" \
VANE_TEST_LOADABLE_HTTPFS_EXTENSION_PATH=\
"$SKBUILD_BUILD_DIR/vane_extensions/httpfs.duckdb_extension" \
  scripts/run_installed_pytest.sh tests/fast/test_loadable_extension_artifacts.py
```

Loadable artifacts require `EXTENSION_STATIC_BUILD=ON`. This keeps each
artifact self-contained and preserves Vane's private `_native` symbol boundary.
The staging directory is configurable with
`VANE_LOADABLE_EXTENSION_OUTPUT_DIRECTORY`.

The Linux native unit suite also builds a signed, self-contained
`loadable_extension_batch_demo` fixture. Its COPY rewrite recursively plans a
UDF child from inside a private engine copy. After `scripts/run_native_tests.sh`,
run its installed-Python regression with
`VANE_TEST_LOADABLE_BATCH_DEMO_PATH="$PWD/build/native-cxx20/test/extension/loadable_extension_batch_demo.duckdb_extension" scripts/run_installed_pytest.sh tests/fast/test_loadable_extension_artifacts.py -k private_extension`.
This covers COPY rewrites and connections created by the private engine, both
local runner names, task and actor execution, empty input, multiple batches,
nulls and fixed-size vectors while forbidding Ray startup. It also checks host
vLLM/SGLang factory dispatch without loading models and restores DataSource bind
data through the private engine's deserializer and scan callbacks.

Executor factories and DataSource callbacks belong to `DatabaseInstance`.
Python installs them once before exposing a connection, including cached databases,
and all connections to that database share them. Native embedders must register
their callbacks on the database they execute against. Do not add library-local
runtime registries: self-contained extensions have their own copy of that static
storage. External block readers travel with their descriptors and do not need a
consumer-side registry lookup.
The Python bridge also installs a database-owned connection callback for UDF
resource preparation and teardown. Connections created inside an extension use
that host callback, including actor pool and transport-capacity preparation.
Schema-only native preparation does not create actor resources. Native
Prepare/Execute calls prepare fresh query-owned resources when rebinding for
execution, including repeated execution of the same prepared statement.
Rebuild optional providers against the changed native SDK; a matching Python
package version alone does not establish C++ compatibility.

For the independently selectable image, audio, and video operators, see
[NATIVE_MEDIA_EXTENSIONS.md](NATIVE_MEDIA_EXTENSIONS.md).

## Building an optional extension wheel

Keep optional artifacts out of the base `vane-ai` wheel. After installing Vane
and staging a release-approved extension, package it with:

```bash
python -I scripts/build_extension_wheel.py \
  --artifact "$SKBUILD_BUILD_DIR/vane_extensions/<extension>.duckdb_extension" \
  --extension-name <extension> \
  --release-number 1 \
  --platform-tag manylinux_2_28_x86_64 \
  --trust-identity astrovela/vane \
  --license-expression "Apache-2.0 AND MIT" \
  --license-file LICENSE \
  --license-file NOTICE \
  --license-file LICENSES/DuckDB-MIT.txt \
  --output-directory dist/extensions
```

Replace the example platform and license expression with those of the actual
artifact, and supply all required notices. The builder checks binary platform
requirements, package metadata, source identity, signatures and dependency
closure. It pins the matching Vane version and emits a content-addressed provider.
Pass the complete dependency closure in load order with repeated
`--dependency-wheel` arguments and explicitly allowlist each unique signer with
`--dependency-trust-identity`.

Verify the exact base and provider wheels in a clean environment on the target
platform's minimum supported runtime:

```bash
python -I scripts/verify_extension_wheel.py \
  --base-wheel dist/vane_ai-*.whl \
  --extension-wheel dist/extensions/vane_extension_<extension>-*.whl \
  --extension-name <extension> \
  --trust-identity astrovela/vane
```

Supply the same dependency closure and signer allowlist to verification.
`tpch` is an in-tree test artifact and must not be published as an extension
wheel. Test fixtures may use `--test-only`; they must never be published.

For specialized workflows and their contracts, see:

- [Local model lifetime](LOCAL_MODEL_RUNTIME.md): explicit registration, query
  borrowing, and runtime shutdown contracts.
- [Native media packaging](NATIVE_MEDIA_EXTENSIONS.md#build-and-package):
  dynamic runtime/source SDK inputs and static release materials.
- [Media publication](NATIVE_MEDIA_RELEASE.md) and
  [library replacement](NATIVE_MEDIA_REPLACEMENT.md): delivery and acceptance.
- [Distributed extensions](DISTRIBUTED_EXTENSIONS.md#build-and-loading):
  provider discovery, exact artifact identity and Ray worker preparation.
- [Release process](RELEASE.md): production and TestPyPI signing policies.
- [Copyleft inventory](COPYLEFT.md): reviewed dependencies and source delivery.

The public CI signing key is enabled only by
`VANE_ENABLE_TEST_EXTENSION_SIGNING_KEY`; never enable it for release artifacts.
No-tag TestPyPI candidates instead use
`VANE_ENABLE_TESTPYPI_EXTENSION_SIGNING_KEY`. Production signing belongs only
in protected release-signing jobs; promote the same signed bytes after
qualification. Never bypass signature verification to load an incompatible
provider.

The validators in `vane_packaging/` define archive bounds and ELF, Mach-O and PE
checks. Keep policy changes with their regression tests. The manylinux policy
is an unmodified auditwheel snapshot: update its pinned version, digest and
license together in `vane_packaging/manylinux_policy.py` and `_vendor/auditwheel/`,
then run the policy parity and Linux extension-wheel tests. Do not edit the
snapshot to admit an otherwise incompatible artifact.

## Native C++ tests

Vane, DuckDB, and the non-Arrow distributed engine build as C++11, while the
Arrow Flight exchange, its direct tests, and the diagnostics boundary tests use
C++20. The diagnostics tests link C++20 callers against C++11 engine definitions.
This keeps Arrow's requirement isolated from the engine and its consumers. The
script refreshes the CMake configuration (`cmake --fresh`) to avoid configuration
drift and reuses compiled objects whose inputs are unchanged:

```bash
scripts/run_native_tests.sh "[distributed]"
```

The optional `native_media` extension requires C++17 for media reader construction
and exact video time arithmetic. Both its static and loadable targets keep this
requirement private, so it does not change the engine's language standard.
The native suite also links C++17 references to logical type constants against
their single exported definitions in the C++11 engine.

Run a named engine test or the complete unit suite with the same build:

```bash
scripts/run_native_tests.sh "test name" -s
scripts/run_native_tests.sh
```

The build uses two parallel compile jobs by default to stay within standard CI
runner memory. Override that limit with `VANE_NATIVE_BUILD_JOBS` when the local
machine has more capacity.

The launcher uses Ninja's single Release configuration by default. Windows CI
disables Git's automatic CRLF conversion before checkout so source-license
hashes and DuckDB SourceID use the committed bytes. Windows source checkouts
must likewise use `core.autocrlf=false` before files are checked out.
The Windows job follows DuckDB's native MSVC path with
`VANE_NATIVE_CMAKE_GENERATOR="Visual Studio 17 2022"` and
`VANE_NATIVE_CMAKE_GENERATOR_PLATFORM=x64`; the launcher restricts the
multi-config build to Release and runs the corresponding test executable. It
uses the pinned vcpkg toolchain in classic mode so Windows package wrappers
resolve release-only static library names without installing dependencies a
second time.

Statically linked DuckDB extensions participate in Ray execution through the
explicit scan callback and write provider contracts described in
[DISTRIBUTED_EXTENSIONS.md](DISTRIBUTED_EXTENSIONS.md). Add engine-level
protocol tests and extension-specific normal and fault-tolerant tests when
implementing either contract.

## Python tests

See [Python source execution](DATASOURCE_EXECUTION.md) for the native readiness,
decoder admission, and cancellation contracts and their focused regression checks.

The required release gate covers the supported base installation and does not
need model downloads, cloud credentials, GPUs, or external services:

```bash
scripts/run_release_tests.sh
```

The gate includes native local-runtime backpressure acceptance: small-budget
progress, concurrent model reuse, retained results, cancellation/cleanup retry,
and final UDF statistics. See [coverage and failure diagnostics](LOCAL_MODEL_RUNTIME.md#backpressure-acceptance-gate).
Real-Ray release tests run in a separate process; the full local parameter
matrices remain in the fast-test shards.

The release launcher accepts `non-ray`, `shared-ray`, or `owner-ray` to run one
phase. With no argument (or `all`), it runs all three phases in order. CI runs
them as separate steps with independent deadlines so earlier phases cannot
consume the time available to later phases.

Vane's native extension is private to the installed `vane` package. Test
launchers therefore run outside the checkout and put the installed
site-packages directory before repository support modules. This prevents the
source package from shadowing `vane._native` and ensures tests exercise the
same layout shipped in the wheel. Run an affected test through the same
installed-package wrapper:

```bash
scripts/run_installed_pytest.sh tests/fast/test_udf_process.py
```

The inherited compatibility suites are broader and require the development
dependency group. Run them when changing the corresponding integration:

```bash
scripts/run_fast_tests.sh
scripts/run_installed_pytest.sh tests/slow
scripts/run_ai_tests.sh
```

The AI launcher runs non-Ray core tests, non-Ray expression tests and Ray tests
in three fresh processes. Run one group with `scripts/run_ai_tests.sh core`,
`expressions` or `ray`. Together they cover the same CPU-only AI suite; external
services and GPU tests require explicit opt-in through the installed pytest
launcher. CI keeps a 600-second deadline per group, reports the first failure
immediately, and uploads JUnit results instead of allowing one long non-Ray run
to exhaust that deadline before printing its failures.

The fast-test launcher runs non-Ray tests, shared-cluster Ray tests, and
test-owned Ray clusters in separate pytest processes. This keeps the real Ray
runtime out of the long-lived non-Ray pytest process. Fast and release Ray test
clusters let Ray size the object store from the node's available memory by
default. `VANE_TEST_RAY_OBJECT_STORE_BYTES` pins the capacity for a specialized
test; it does not configure production clusters. Tests that call `ray.init()`
directly must be marked `real_ray` and `ray_cluster_owner`.

CI further splits the non-Ray phase across CPU-only jobs. The jobs install the
built wheel, use CPU-only PyTorch, and set hard pytest-process and job deadlines
so the suite fits a standard 4-vCPU, 16-GiB GitHub-hosted runner. Tests marked
`gpu` are excluded there because standard runners do not provide CUDA hardware;
run the default launcher on a GPU host to include them.

Tests that require an externally provisioned service are excluded by default.
Run them explicitly when the required service and credentials are available:

```bash
scripts/run_installed_pytest.sh -m external_service tests/fast
```

Other optional tests may require network access, model weights, GPUs, credentials, or a local Ray setup. Tests must
skip with a clear reason when an optional environment is absent; they must not silently use a maintainer's local
endpoint or credentials.

Cosmos image queries reuse the video model's visual encoder with one frame.
`tests/fast/test_cosmos_embed1.py` checks preprocessing, precision, RGB validation,
NULL batches and planning without model downloads. The image/video embedding
contract tests cover the common Python and SQL paths. The opt-in
`tests/ai/test_cosmos_video_embedding.py` exercises paired image/video/text vectors
on default Ray using cached weights and reviewed media. Set `HF_HUB_OFFLINE=1`,
`VANE_TEST_COSMOS_CACHE` and `VANE_TEST_COSMOS_VIDEO` for that GPU check.
Single-frame support follows the upstream processor's explicit
[`num_video_frames` override](https://huggingface.co/nvidia/Cosmos-Embed1-224p/blob/787e0b996f5260a71ad474a283c90539a2e12986/preprocessing_embed1.py).
This execution check does not establish image-to-video retrieval quality on a
representative corpus; keep that acceptance separate from the CPU release gate.

## Formatting and static checks

```bash
python -m pip install pre-commit
pre-commit install
scripts/format root --changed
pre-commit run --from-ref origin/main --to-ref HEAD
```

Run `pre-commit install` once per clone.

Add `--check` to verify formatting without modifying files. Use `workspace`
when both Vane-owned files and the DuckDB subtree have changed:

```bash
scripts/format workspace --changed --check
```

To check changes relative to a committed ref, including in CI, use:

```bash
scripts/format workspace --from-ref origin/main --check
```

The root formatter deliberately excludes `external/duckdb`. Format DuckDB subtree changes with:

```bash
scripts/format duckdb --changed
```

## Updating the DuckDB subtree

The official engine baseline is imported from `duckdb/duckdb` as a squashed
subtree snapshot. Pull a reviewed upstream revision using the same mode:

```bash
git subtree pull --prefix=external/duckdb --squash \
  https://github.com/duckdb/duckdb.git main
```

The subtree metadata records the exact official DuckDB revision in
`git-subtree-split`. Vane-specific engine changes live as subsequent commits
under `external/duckdb`; review and resolve them when updating the official
baseline. When replaying a change formerly maintained in another repository,
preserve its author and date and record the original commit and upstream parent
as commit trailers. To inspect both engine identities without writing the
checkout, run:

```bash
python scripts/sync_duckdb_source_id.py --print
python scripts/resolve_duckdb_fork_version.py --print-version
```

`SourceID` identifies the contents of `external/duckdb`, including non-ignored
untracked files and file modes. The fork version identifies the last Vane
commit that changed that subtree, prefixed by `DUCKDB_UPSTREAM_VERSION` and
suffixed with `-dirty` for uncommitted subtree changes. Both commands are read-only.
Incremental builds refresh these identities automatically, including mode-only
changes; generated headers live in the build directory.

Source distributions carry generated `DUCKDB_SOURCE_ID` and
`DUCKDB_FORK_REVISION` manifests. Do not commit them. Git-exported trees can
derive a Git-compatible SourceID from their files, but an archive without Git
history still requires the fork revision manifest. Update `SOURCE_PROVENANCE.md`
and `DUCKDB_UPSTREAM_VERSION` only for baseline, version or provenance changes.

A custom `DUCKDB_SOURCE_PATH` requires explicit full `VANE_DUCKDB_SOURCE_ID`
and `VANE_DUCKDB_FORK_REVISION` values, plus `VANE_DUCKDB_UPSTREAM_VERSION`
in `vX.Y.Z` form. Configuration fails if any is absent.

The original upstream history remains in `duckdb/duckdb`. Vane's path history
begins at the squashed snapshot and includes every later Vane engine commit. To
inspect or export that history with DuckDB-rooted paths, split it to a temporary
branch:

```bash
git subtree split --prefix=external/duckdb --ignore-joins -b duckdb-history
git log --stat duckdb-history
```

`--ignore-joins` produces a self-contained compact history containing the
official snapshot and Vane's subsequent commits. To reconnect the split branch
to DuckDB's complete upstream history instead, fetch `duckdb/duckdb` first and
omit `--ignore-joins`; Git uses the recorded `git-subtree-split` revision as the
join point.

## Debugging Ray workers

Query teardown accepts the RPC target's terminal `ActorDiedError` as proof that
the worker process cannot issue more side effects. It still joins local registry
operations before dropping that query's control tails and registrations.
Completed control failures are collected before those tails are released and
reported after storage cleanup, even if no earlier control barrier observed them.
A timeout, unavailable actor, or task exception caused by a different actor's
death does not provide that proof and retains teardown ownership. The focused
cases in `test_ray_fragment_submission.py` cover each teardown phase, unrelated
queries and local operations that have not joined. Context's subtitle-source
worker-exit integration also checks zero publication and query-owned scratch
cleanup through a real Ray worker death.

Set `DUCKDB_DISTRIBUTED_DEBUG=1`. Native debug output uses `DistributedDebugStream()` and appears in Ray worker error logs, normally below `/tmp/ray/session_latest/logs/worker-*.err`. Plain C `stdout` output is not reliably captured by Ray workers.

Ray UDF pools return pending handles during executor construction; task admission
waits for the first initialized actor without blocking native startup. The default
initialization limit is 300 seconds, including time waiting for Ray capacity.
After admission, both table and ref-bundle submission resolve the leased actor's
initialization result even if its local delivery or readiness notification is
delayed. This wait honors query interruption and query/ObjectRef deadlines; an
empty local ready set alone must not reject an admitted task.
Set `VANE_RAY_ACTOR_INIT_TIMEOUT_S` to a finite positive number on the client
before opening its Ray runtime to change this limit. Shorter query/ObjectRef
deadlines still apply. Resource maintenance reports the expired pool's requested
CPU/GPU resources and terminates its actors; it never switches execution backends.

A client that stops heartbeating is reclaimed after
`VANE_RAY_CLIENT_LEASE_TIMEOUT_S` (60 seconds by default). Reclamation fences
its queries and terminates their UDF actors before joining native operations,
so an abandoned GPU actor cannot prevent session teardown. Failed termination
retains ownership and is retried; connections and resource leases remain owned
until normal teardown completes.

## Release artifacts

Build and validate an sdist before opening a release pull request:

```bash
python -m build --sdist
python scripts/check_release_artifacts.py dist/*.tar.gz
```

See [RELEASE.md](RELEASE.md) for the complete process.

AI error diagnostics use `vane.ai.summarize_error(error)` to produce a bounded,
public-safe root-cause summary, including cleanup failures. Known engine option and
GPU memory errors retain actionable details; arbitrary upstream text is
not copied because it can contain credentials or input data. Numeric status and
errno values remain available. Provider wrappers preserve these diagnostics over
Ray/native transport, capturing cause/cleanup chains before detaching upstream
exceptions. Flattened diagnostics keep each exception's own status, including
HTTP status stored on an SDK response. A resolved actor initializer failure now fails its pool and
releases owned actors immediately, even if other actors are pending for resources.
Worker integrations should raise `vane.ai.DiagnosticError(operation, error)`
outside the original exception handler. It captures a sanitized summary and
retains its type through pickle and structured Ray error metadata. Raw SDK
messages are never interpreted as diagnostic envelopes, even if they contain
`upstream error:` or forged diagnostic headers. Actor activation preserves the
typed cause chain before Ray serializes it, and executor creation captures it
before the native dispatcher converts the exception to text. The Ray driver
captures native diagnostics at result-stream, COPY and DataSink call boundaries.
Terminal INSERT tasks can add a native `INTERNAL` spawn envelope around the FTE
failure. Decoding requires the exact task prefix and the nested typed UDF
diagnostic; unrelated internal errors and forged provider text remain opaque.
COPY outcome exceptions retain sanitized cause/cleanup snapshots in their pickle
state while preserving the committed/unknown outcome type and retry restrictions.
Terminal query cleanup failures are captured from their exception objects into
the transport summary; arbitrary warning strings and exception notes are not parsed.
A Ray wrapper never authorizes decoding a plain `RuntimeError` body. Native
decoding is restricted to native exceptions and those native query producers;
descriptions and model identifiers cannot replace the encoded summary. Known
safe provider failures such as `vllm returned no outputs` retain their explanation
through wrapping and transport using a closed vocabulary.
Remote vLLM/SGLang executors retain typed actor diagnostics, including failed
submission acknowledgements and cleanup, before returning errors to native SQL.
Rewrapping merges worker and query cleanup into one bounded cleanup section,
keeping errno attached to the correct failure. Length limits preserve complete
diagnostic items in both sections so another transport can still parse cleanup;
items exceeding a small caller budget omit details, and type names that cannot
fit use `Exception`. Cleanup also reserves space in the parser's item limit.
Remote wait deadlines, missing wait references and incomplete actor results
retain their closed technical explanations; actor/result counts are validated
as bounded integers.
Validate changes with the AI diagnostic tests, actor-pool lifecycle tests and
isolated Ray regressions for initialization, fetching results and writing Parquet.
The dependent Context HTTP regressions exercise actual `vane.cls.batch`
constructor failures through a complete query and HTTP error response.

For the explicit SGLang `/encode` HTTP format, complete two clean reviews before
building/installing the Python-only wheel noneditably. Run the affected
`test_ai_embedding_sglang_encode.py` and `test_ai_embedding_http_runtime.py`
through the installed launcher. Cover Base64 video/audio, explicit media tokens,
text-only queries, mixed absent inputs, batching/order and serialized byte limits
before client creation, including ignore mode. Real-model acceptance uses the
unchanged SGLang server with audio/video/subtitles and a matching Context catalog;
CPU HTTP fixtures do not establish model quality. Only related tests are required;
no native rebuild or Ray suite is needed for these Python-only HTTP changes.

For vLLM HTTP tokenization options, review validation before client creation,
explicit true/false versus omission, descriptor serialization and every split
request, including text-only queries under the multimodal format. Complete two
clean reviews before the noneditable Python wheel install, then run the affected
`test_ai_embedding_http_runtime.py` and `test_ai_embedding_sglang_encode.py`
through the installed launcher with Ray startup forbidden. The dependent Context
preset/local gate checks model-owned tokenization for ingestion and search.
Compare real token counts and vectors to an independent checkpoint processor;
protocol fixtures alone do not qualify numerical behavior. Only related checks
are required; no native rebuild, full/release suite or Ray tests.

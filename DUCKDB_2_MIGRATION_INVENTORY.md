# DuckDB 2.0 capability migration inventory

P1 is complete as of 2026-10-26: the official Python client is imported at
`external/duckdb-python`, its tree matches upstream, and its engine pin matches
the existing pure 2.0 engine. This inventory identifies work to restore Vane;
**every capability below is still pending on the 2.0 branch**. Retained source
files and tests do not establish runtime compatibility.

The [design and roadmap](DUCKDB_2_MIGRATION_DESIGN.md) owns stage acceptance.
[Source provenance](SOURCE_PROVENANCE.md#duckdb-python-client-baseline) records
the complete import identities. Build and test commands belong in
[Development](DEVELOPMENT.md).

## Reference rules

All reference implementation paths in the capability tables refer to Vane
`main` at `06cfbd5cc39f8b2c19531676f2bdcfff64e1a6e4`, synchronized on
2026-10-26. This supersedes the original import starting point
`95acdea830eb1fff407f0e09f4b97ee7b0a02e83`. In particular, the 1.5.5 custom
`external/duckdb` paths may no longer exist in the current checkout. Inspect
them without restoring files over the new baseline:

```bash
git show 06cfbd5cc39f8b2c19531676f2bdcfff64e1a6e4:src/vane_py/bound_plan.cpp
git ls-tree -r --name-only 06cfbd5cc39f8b2c19531676f2bdcfff64e1a6e4 -- external/duckdb/src/execution/distributed
```

Tests linked below are existing behavioral starting points, not an exhaustive
acceptance suite. Preserve their contracts when adapting them to the new
client. A test against an old installed Vane package does not qualify a row.
Features committed after this pinned main revision and uncommitted work in
other worktrees require separate entries with their source revision and tests
before migration.

The baseline correction brings in 45 main commits and preserves their Vane-owned
files and tests. Both official 2.0 subtrees remain unchanged. The reference now
includes native local execution and its shared runtime, callback and result
ownership fixes, Gravitino Fileset batch writes, VaneFS durability and same-host
Ray access, typed multimodal embeddings, and resident model/service backends.
The former public local FTE entrypoint is already removed on main; restoring
it is not a migration requirement. See [the current local contract](LOCAL_MODEL_RUNTIME.md).

Disposition terms used below:

- **Adapt**: preserve the Vane contract using the 2.0 engine/client interfaces.
- **Port**: reintroduce Vane-owned behavior after its dependencies are ready.
- **Replace inherited implementation**: use the official client implementation
  as the starting point; preserve or explicitly resolve Vane API differences.

No public capability is approved for removal by this inventory. An intentional
API change needs a recorded decision, replacement path, and regression coverage.
When a row is complete, add its implementation commit, installed artifact
identity, checks and outcomes, and any remaining limitations to the completion
record. Split rows into smaller changes where needed; do not restore directories
wholesale just because they appear in a row.

## Imported client layout and integration decisions

The import is the official `duckdb/duckdb-python` v2.0.0 commit
`83c719961098611e769dc4bab0bda2bf149a2d1e`, tree
`b2e1819b1bd881106f5b3af31a62bc61ff8110d3`. These paths are relative to
`external/duckdb-python`:

| Upstream area | Observed role | Vane integration consequence |
| --- | --- | --- |
| `duckdb/__init__.py`, `_relation.py`, `_expression.py`, `_seam.py`, `_sources/` | Public package, SQL-text relations and expressions, connection methods, and Python data sources | Adapt public imports and query construction to `vane`; do not retain a second inherited connection implementation |
| `src/duckdb_python.cpp`, `src/pyconnection.cpp`, `src/pyconnection/`, `src/pyresult.cpp`, `src/engine/` | Nanobind module entrypoint, connections, streaming results, and engine adapters | Establish one private `vane._native` module, then add Vane hooks at explicit boundaries |
| `src/arrow/`, `src/numpy/`, `src/pandas/`, `src/native/`, `src/python_udf.cpp` | Conversion, ownership, replacement scans, and scalar UDF bindings | Reuse upstream mechanisms where they satisfy Vane contracts; qualify FILE/Tensor and UDF changes separately |
| `_duckdb-stubs/`, `duckdb/sqltypes/`, `duckdb/py.typed` | Hand-maintained native type stubs and Python typing surface | Map to Vane's native stubs and public names; retain Vane-only signatures when their implementations return |
| `adbc_driver_duckdb/`, `duckdb/experimental/spark/`, `duckdb/query_graph/` | Additional Python APIs and package data | Inventory Vane counterparts, package paths, and notices; prevent imports of the official module by accident |
| `CMakeLists.txt`, `cmake/duckdb_loader.cmake`, `src/CMakeLists.txt` | C++17 build, nanobind targets, configurable engine path, static extension linking | Rework root target selection and naming; do not compile old pybind11 binding objects alongside the new client |
| `pyproject.toml`, `duckdb_packaging/` | Python >=3.11, nanobind >=3, NumPy >=2 build headers, scikit-build-core wrapper and build identities | Adapt Vane's packaging and support matrix; upstream wheel metadata and version resolution cannot be used unchanged |
| `tests/`, `benchmarks/` | Upstream client regression and workload inputs | Keep the snapshot intact; select and adapt relevant tests outside it to exercise the installed Vane artifact |
| `.gitmodules`, `external/duckdb` | Engine gitlink at `d54cbbe3df17e8e5ce68312f483b4155836112aa` | Retain the exact metadata without initializing a second engine checkout |

The source basis for these decisions is the imported
[CMake configuration](external/duckdb-python/CMakeLists.txt),
[engine loader](external/duckdb-python/cmake/duckdb_loader.cmake),
[package configuration](external/duckdb-python/pyproject.toml), and
[build identity resolver](external/duckdb-python/duckdb_packaging/build_info.py).
The [official release notes](https://github.com/duckdb/duckdb-python/releases/tag/v2.0.0)
provide the public migration context.

P2 must set `DUCKDB_SOURCE_PATH` to the repository's absolute
`external/duckdb` path and configure that engine only once. The client's
default points to its nested gitlink. Its packaging backend also reads
submodule and repository history independently of CMake, so setting this path
alone is insufficient. Integrate Vane's package version, fork revision, and
SourceID rules without deriving identities from the empty nested directory or
the parent repository as if it were the official client repository.

Upstream builds `_duckdb`, installs the `duckdb` namespace, and selects stable
ABI support on eligible CPython versions. Those choices are inputs for P2, not
the desired Vane artifact. The Vane package must retain its private native
module and coexist with an independently installed official `duckdb`. Prove
native type and symbol isolation under nanobind; old pybind11 flags do not
establish it. Keep non-editable builds as required by this repository.

The upstream default extension set is `core_functions`, `json`, `parquet`, and
`icu`. Start P2 with a reviewed minimal set, then restore Vane extensions at
their listed stages. The pure import does not supply the former FILE,
distributed, Gravitino, or native media extensions.

## P2: minimal native installation

| ID / contract | Reference implementation anchors | Disposition and dependencies | Representative evidence |
| --- | --- | --- | --- |
| B01 Build and package identity | `CMakeLists.txt`, `cmake/duckdb_loader.cmake`, `pyproject.toml`, `vane/__init__.py`, `src/vane_py/vane_python.cpp` | Replace inherited implementation; integrate one engine/client, C++17 consumers, Python support metadata, private module and non-editable wheel | [Package metadata](tests/fast/test_package_metadata.py); fresh wheel install and native SQL smoke |
| B02 Basic connection, relation and value APIs | `src/vane_py/pyconnection.cpp`, `pyrelation.cpp`, `pyexpression.cpp`, `merge_relation.cpp`, `vane/_native/` | Adapt after B01; inventory method/signature differences, parameters, transactions, registered data, MERGE, and typing before replacing old bindings | [Relations](tests/fast/test_relation.py), [MERGE](tests/fast/test_merge_relation.py), `tests/fast/relational_api/`, `tests/fast/arrow/`, `tests/fast/numpy/`, `tests/fast/pandas/` |
| B03 Native results, ownership and interruption | `src/vane_py/pyresult.cpp`, `pyresult_source.cpp`, `python_input_callback.cpp`, `vane/_query_interrupt.py` | Adapt after B01/B02; qualify partial fetch, close, connection reuse, callback reentry guards, Arrow release callbacks and interrupt propagation; configured local runtime delivery follows in N08 | [Dependency lifetime](tests/fast/test_relation_dependency_leak.py), [input callbacks](tests/fast/test_python_callback_entry.py), [binding callbacks](tests/fast/test_python_binding_callbacks.py), [parameter callbacks](tests/fast/test_python_parameter_callbacks.py), [teardown](tests/fast/test_vane_teardown.py); adapt upstream lifetime tests |
| B04 Process coexistence and ADBC | `CMakeLists.txt`, `src/vane_py/pybind11/`, `vane/adbc/` | Adapt after B01; keep both import orders safe, reject foreign native objects, route ADBC to Vane's own entrypoints | [Coexistence validator](scripts/verify_duckdb_coexistence.py), `tests/fast/adbc/` |
| B05 Engine and artifact identities | `scripts/sync_duckdb_source_id.py`, `scripts/resolve_duckdb_fork_version.py`, root CMake, `vane/_native_runtime.py` | Adapt after B01; restore build hooks and runtime identity reporting, including sdist builds and custom engine paths; worker enforcement follows in P4/P6 | [SourceID](tests/fast/test_sync_duckdb_source_id.py), [fork revision](tests/fast/test_resolve_duckdb_fork_version.py), [native runtime](tests/fast/test_native_runtime.py) |
| B06 Source tools, notices, artifacts and CI | `scripts/format`, `.pre-commit-config.yaml`, `scripts/check_source_license_headers.py`, `scripts/check_release_artifacts.py`, `vane_packaging/copyleft_policy.py`, `.github/workflows/` | Adapt alongside B01; classify both vendor trees, review changed notices and hashes, select sdist contents, and add integration-branch checks | [Copyleft policy](tests/fast/test_copyleft_policy.py); formatter, license and artifact validators on the actual new source/wheel |

P2 API qualification starts with native behavior. Methods depending on later
rows must remain explicitly unavailable; an explicit Ray request must not
silently run through native DuckDB. `local` and its `local-fast` alias select
native execution, but basic native SQL does not qualify the configured local
runtime or managed results in N08. Before retiring old inherited
binding files, record the public API comparison, including methods absent or
changed in 2.0, and where each Vane-only method is restored.

## P3: native Vane values, I/O and UDFs

| ID / contract | Reference implementation anchors | Disposition and dependencies | Representative evidence |
| --- | --- | --- | --- |
| N01 FILE values and filesystem access | `external/duckdb/extension/file/`, `vane/_file.py`, `src/vane_py/pyfilesystem.cpp`, `vane/filesystem.py` | Port FILE semantics and adapt filesystem bindings after B02/B03; retain ownership, paths, reader behavior and NULL handling | [FILE](tests/fast/test_file.py), [readers](tests/fast/test_file_reader.py), [filesystems](tests/fast/test_filesystem.py), [API contract](FILE_PYTHON_API.md) |
| N02 Tensor representation and conversion | `external/duckdb/src/function/tensor_functions.cpp`, `vane/_tensor.py`, `src/vane_py/numpy/`, `src/vane_py/pandas/` | Adapt after B02/B03; preserve variable shape, dtype, nested/null values and buffer ownership with 2.0 vectors | [Tensor](tests/fast/test_variable_tensor.py), [Tensor UDF output](tests/fast/test_udf_pandas_tensor_output.py), [type contract](VARIABLE_TENSOR.md) |
| N03 Python sources and readiness | `external/duckdb/src/function/table/datasource_scan.cpp`, `src/vane_py/datasource_function.cpp`, `vane/datasource/` | Port after B03/N01/N02; align bounded buffering, wakeups and cancellation with 2.0 execution rather than duplicating blocking protocols | [Source readiness](tests/fast/test_datasource_readiness.py), [execution contract](DATASOURCE_EXECUTION.md) |
| N04 Scalar, batch and expression UDF contracts | `src/vane_py/python_udf.cpp`, `python_udf_utils.cpp`, `udf_executor.cpp`, `vane/execution/udf.py`, `vane/_expression_udf.py`, `external/duckdb/src/function/scalar/udf.cpp` | Adapt after N01/N02/N03; preserve schema, row count, NULL, batch formats, process execution and exception behavior | [Process UDFs](tests/fast/test_udf_process.py), [sync contract](tests/fast/test_udf_sync_contract.py), [schemas](tests/fast/test_udf_output_schema.py), [batch formats](tests/fast/test_udf_batch_format.py), [expressions](tests/fast/test_expression_udf_contracts.py) |
| N05 Async UDF admission and lifecycle | `vane/execution/_async_dispatch.py`, `_async_runtime.py`, `udf_task_admission.py`, `udf_actor_callable.py`, `src/vane_py/python_udf_actor_resources.cpp`, `external/duckdb/src/include/duckdb/execution/execution_runtime.hpp` | Port after N03/N04; preserve limits, blocked-task wakeups, early finish, shared database runtime across private execution engines, and cleanup; Ray execution is qualified later | [Async executor](tests/fast/test_udf_async_runtime_executor.py), [admission](tests/fast/test_udf_task_admission.py), [early finish](tests/fast/test_udf_early_finish.py), [executor lifetime](tests/fast/test_udf_executor_lifecycle.py), [private extension runtime](tests/fast/test_loadable_extension_artifacts.py) |
| N06 Binding and runner selection | `src/vane_py/bound_plan.cpp`, `query_parameters.cpp`, `vane-runners/vane_runners.cpp`, `vane/runners/runner.py` | Adapt after B02/B03/N04; preserve bound parameters; normalize local/local-fast to native execution with no local FTE runner; keep Ray explicitly unavailable until P4/P5 | [Bound plans](tests/fast/test_bound_plan_runner.py), [client catalog](tests/fast/test_runner_client_catalog.py), [local runtime baseline](tests/fast/test_local_runtime_baseline.py) |
| N07 Native DataSink contract | `vane/datasink/`, `external/duckdb/src/execution/distributed/data_sink.cpp`, `src/vane_py/bound_plan.cpp` | Port after N03/N06; qualify native sink behavior separately from distributed attempt publication in P6; retain the configured local runtime's write restrictions | [DataSink](tests/fast/test_datasink.py); select native-path cases first |
| N08 Shared native local runtime and managed results | `src/vane_py/pyconnection.cpp`, `native_resource_graph.cpp`, `vane/execution/local_query.py`, `local_query_coordinator.py`, `local_result_delivery.py`, `local_output_read.py` | Port after B03/N03/N05/N06; preserve session/cursor ownership, request/task/data admission, resident model pools, resource graphs, bounded input/output leases, cancellation, and managed result lifetime | [Local queries](tests/fast/test_local_query_runtime.py), [resource graphs](tests/fast/test_local_resource_graph.py), [results](tests/fast/test_local_query_results.py), [streaming](tests/fast/test_local_query_streaming.py), [output admission](tests/fast/test_local_output_read_admission.py), [serving acceptance](LOCAL_SERVING_ACCEPTANCE.md) |

## P4: distributed reads

| ID / contract | Reference implementation anchors | Disposition and dependencies | Representative evidence |
| --- | --- | --- | --- |
| D01 Plan translation and transport | `src/vane_py/ray/distributed_plan_bindings.cpp`, `logical_plan_bindings.cpp`, `external/duckdb/src/execution/physical_operator.cpp`, `external/duckdb/src/execution/distributed/plan/`, `external/duckdb/src/execution/distributed/pipeline_node/` | Adapt after N06; define a versioned plan/fragment contract and supported operator set; reject unsupported operations before effects | [Plan conversion](tests/fast/test_ray_relation_to_distributed_plan.py), [aggregates](tests/fast/test_distributed_aggregates.py), [ordered limits](tests/fast/test_ray_ordered_limit.py); joins and other translator cases |
| D02 Worker session and UDF replay | `src/vane_py/ray/worker.cpp`, `src/vane_py/ray/worker_manager.cpp`, `vane/runners/ray/`, `src/vane_py/python_udf.cpp` | Adapt after D01/N04/N05; carry session configuration, registrations, bound inputs, secrets and providers through explicit replay rules | [Connection snapshots](tests/fast/test_ray_worker_connection_snapshot.py), [UDF replay](tests/fast/test_ray_udf_plan_replay.py) |
| D03 Fragment execution and exchange | `src/vane_py/ray/native_fragment_executor.cpp`, `vane/runners/ray/fragment_worker_exchange.py`, `external/duckdb/src/execution/distributed/exchange/`, `vane/runners/fte/` | Port after D01/D02/B05; use the shared FTE primitives, matching worker artifacts, bounded Flight exchange and split assignment | [Distributed sources](tests/fast/test_datasource_distributed_scan.py), [fragment submission](tests/fast/test_ray_fragment_submission.py), [FTE success paths](tests/fast/test_ray_fte.py) |
| D04 Distributed result consumption | `src/vane_py/ray/result_stream_bindings.cpp`, `query_lifecycle_coordinator.cpp`, `vane/runners/ray/fragment_worker_results.py` | Adapt after B03/D03; preserve ordering where promised, duplicates, empty/nested results, partial consumption and close propagation | [Result contracts](tests/fast/test_ray_result_contract.py), [consumers](tests/fast/test_distributed_result_consumers.py), [FILE on Ray](tests/fast/test_file_ray.py) |

P4 qualifies successful Ray execution on the shared fragment architecture.
Retry correctness requires P5. Native local execution and its configured
runtime are qualified in P2/P3, independently of the Ray scheduler.

## P5: Ray recovery, resources and diagnostics

| ID / contract | Reference implementation anchors | Disposition and dependencies | Representative evidence |
| --- | --- | --- | --- |
| F01 Attempt and failure semantics | `vane/runners/fte/fte_attempts.py`, `fte_failures.py`, `fte_events.py`, `fte_state.py`, `fte_scheduler.py`, `vane/runners/ray/fte_fragment_scheduler.py` | Port after D03/D04; qualify retries, stale messages, lost workers and failures while blocked without missing or duplicated results | [Fault injection](tests/fast/test_ray_fte_fault_injection.py), [event scheduling](tests/fast/test_ray_fte_event_scheduler.py), [status observation](tests/fast/test_ray_fte_status_observation.py) |
| F02 Admission, cancellation and resource ownership | `vane/runners/ray/query_resource_runtime.py`, `query_resource_graph_builder.py`, `worker_pool.py`, `vane/execution/query_resource_policy.py`, `cluster_resource_policy.py`, `udf_ray.py` | Port after F01/N05; bound query/cluster resources and release workers, actors, references and pending work on every terminal path | [UDF resources](tests/fast/test_ray_udf_resource_lifecycle.py), [query resources](tests/fast/test_query_resource_manager.py), [cluster resources](tests/fast/test_cluster_resource_coordinator.py), [teardown](tests/fast/test_vane_teardown.py) |
| F03 Ray progress and diagnostics | `vane/runners/fte/fte_execution.py`, `vane/runners/ray/fte_fragment_scheduler.py`, `vane/runners/progress.py` | Adapt after F01/F02; observe progress without mutating scheduler ownership; qualify interruption and terminal snapshots; this supersedes the obsolete local-FTE inventory row | [Progress](tests/fast/test_ray_fte_progress.py), [status observation](tests/fast/test_ray_fte_status_observation.py), [progress environment](tests/fast/test_ray_progress_env.py) |

## P6: writes, extensions and catalogs

| ID / contract | Reference implementation anchors | Disposition and dependencies | Representative evidence |
| --- | --- | --- | --- |
| W01 COPY and DataSink publication | `vane/datasink/`, `vane/runners/ray/committed_copy.py`, `external/duckdb/src/include/duckdb/execution/distributed/copy_finalize.hpp`, `external/duckdb/src/execution/distributed/pipeline_node/data_sink_finish.cpp` | Port after N07/F01/F02; preserve attempt-scoped output, coordinator finalization, ordered writes, abort and retry contracts | [DataSink](tests/fast/test_datasink.py), [ordered COPY](tests/fast/test_ray_ordered_copy.py) |
| W02 Extension scan/write providers | `src/vane_py/dynamic_extension.cpp`, `external/duckdb/src/main/distributed_extension_manager.cpp`, `external/duckdb/src/execution/distributed/extension_write.cpp`, `vane/extensions.py` | Adapt after D02/W01; verify provider identity before deserialization and execution, with coordinator-owned publication | [Provider resolution](tests/fast/test_dynamic_extension_resolver.py), [Ray replay](tests/fast/test_ray_dynamic_extension_replay.py), [extension contracts](DISTRIBUTED_EXTENSIONS.md) |
| W03 Extension and worker artifact delivery | `vane/_native_runtime.py`, `vane_packaging/extension_wheel.py`, `extension_materials.py`, root CMake | Adapt after B05/W02; rebuild for the modified engine SourceID, enforce manifest/material checks, and deliver matching artifacts to workers | [Loadable artifacts](tests/fast/test_loadable_extension_artifacts.py), [extension wheels](tests/fast/test_extension_wheel.py), [runtime wheels](tests/fast/test_native_runtime_extension_wheels.py) |
| W04 Gravitino and catalog operations | `external/duckdb/extension/gravitino/`, `src/vane_py/pyrelation.cpp`, `vane/catalog.py` | Port after W01/W02/W03; retain provider pinning, Fileset batch writes and native publication, literal local parent paths, metadata and transaction boundaries | [Gravitino](tests/fast/test_gravitino.py), [Fileset writes](tests/fast/test_gravitino_write.py), [live service](tests/fast/test_gravitino_live.py), [catalog contract](GRAVITINO.md), [write design](GRAVITINO_DESIGN.md) |
| W05 HTTPFS and VaneFS integration | `cmake/duckdb_loader.cmake`, `src/vane_py/vane_fs.cpp`, `vane_fs/` | Adapt after N01/W02/W03; qualify the pinned HTTPFS provider, VaneFS I/O/fsync durability, and same-host Ray adapter within the documented storage/access model | [Filesystem tests](tests/fast/test_filesystem.py), [VaneFS on Ray](tests/fast/test_vane_fs_ray.py), `vane_fs/tests/test_durability.cpp`, [VaneFS design](vane_fs/DESIGN.md); provider-specific I/O and artifact checks |

## P7 and P8: application parity and release

| ID / contract | Reference implementation anchors | Disposition and dependencies | Representative evidence |
| --- | --- | --- | --- |
| M01 Native image/audio/video | `external/duckdb/extension/image/`, `audio/`, `video/`, `media_common/`, `native_media/`; `src/vane_py/image_file_functions.cpp`, `audio_file_functions.cpp`, `video_file_functions.cpp` | Port in P7 after N01/N02/W03; restore optional codecs, value conversions, matching artifacts and lifetime rules | [Native media](tests/fast/test_native_media_extensions.py), [Ray media](tests/fast/test_ray_native_media_extensions.py), [multimodal values](tests/fast/test_native_multimodal.py), [extension design](NATIVE_MEDIA_EXTENSIONS.md) |
| M02 Video frames and admission | `src/vane_py/read_video_frames.cpp`, `video_clip_functions.cpp`, `video_frame_functions.cpp`, `vane/datasource/video_reader.py`, `vane/_read_video_frames.py` | Adapt in P7 after M01/N03/N05/D03/F02; preserve frame/time semantics, bounded decoder admission and cancellation | [Decoder admission](tests/fast/test_video_decoder_admission.py), [frame API](VIDEO_FRAME_API.md) and its linked regressions |
| M03 AI expressions and inference providers | `vane/ai/`, `src/vane_py/ai_sql_functions.cpp`, `vllm_executor.cpp`, `external/duckdb/src/function/scalar/vllm.cpp`, `vane/_expression_udf.py` | Port in P7 after N04/N05/N08/M01/M02 as required by each provider; add F02 for Ray; preserve typed multimodal inputs/outputs, row selection, SQL embeddings, SGLang/vLLM HTTP semantics, batching and optional dependencies | [AI contracts](tests/fast/test_ai_release_contracts.py), [embedding outputs](tests/fast/test_embedding_outputs.py), [HTTP runtime](tests/fast/test_ai_embedding_http_runtime.py), [multimodal embeddings](tests/fast/test_ai_multimodal_embedding.py), [embedding design](EMBEDDING_DESIGN.md); separate GPU/service checks |
| M04 Resident embedding engines and service transport | `vane/ai/embedding_service.py`, `vane/ai/providers/_embedding_service_backend.py`, `_mlx_service.py`, `_litert_service.py`, `vane/ai/_embedding_zmq.py`, `_embedding_shared_memory.py` | Port in P7 after N08/M03; preserve persistent native MLX/LiteRT-LM engines, typed requests, PyZMQ/shared-memory ownership, process reuse, timeouts, cancellation and teardown; keep provider environments explicit | [Service](tests/fast/test_embedding_service.py), [transport](tests/fast/test_embedding_service_transport.py), [resource boundaries](tests/fast/test_embedding_resource_boundaries.py), [MLX runtime](tests/fast/test_mlx_service_runtime.py), [LiteRT runtime](tests/fast/test_litert_service.py) |
| R01 Remaining compatibility surface | `vane/experimental/spark/`, `vane/query_graph/`, Python API/stub files and inherited test suites | Adapt by P8 after B02/M03; resolve every supported method and intentional upstream SQL/storage behavior change, including rollback implications | `tests/fast/spark/`, `tests/fast/relational_api/`, `tests/slow/`; explicit API/storage compatibility record |
| R02 Release, platform and workload qualification | `.github/workflows/`, `scripts/run_release_tests.sh`, `scripts/run_fast_tests.sh`, `vane_packaging/`, `RELEASE.md` | Adapt in P8 after all accepted rows; validate wheels, sdists, extensions, notices, supported platforms and representative resource/performance budgets | [Release contract](RELEASE.md), [artifact validator](scripts/check_release_artifacts.py), complete applicable test gates and recorded workload comparisons |

## P1 verification and remaining work

The pure client import commit is
`362db617be3ed09a9fcba9aaf23f37372a56f899`; its squash snapshot is
`5f5292f695286827847d95ac88d162402e21a76a`. Verification established:

- The imported client tree equals `b2e1819b1bd881106f5b3af31a62bc61ff8110d3`.
  All 630 materialized files were checked against upstream blob identities and
  file modes, separately from the engine gitlink. Squash metadata records the
  exact official client commit.
- The engine tree remains `c671df12df7e48e80306cb6a42d4761417c01695`; the
  client's gitlink selects its official commit
  `d54cbbe3df17e8e5ce68312f483b4155836112aa`. The nested engine is not checked out.
- A root `.gitmodules` mapping lets Git resolve the nested gitlink during
  checkout and cleanup without modifying the client snapshot. Default
  submodule updates and recursive fetches are disabled; a disposable clone
  verifies metadata traversal and that recursive initialization leaves the
  nested engine empty.
- The upstream MIT license and Spark Apache-2.0 notice are retained. No
  generated engine identity manifest is added to the repository.
- Root build configuration and old bindings are unchanged. No client/Vane
  build or runtime tests are claimed for P1.

B06 includes known work before artifact qualification: the root formatter and
source-license checker do not yet classify `external/duckdb-python` as vendor
source; the engine header checker still uses the old 1.5.0 reference; the
copyleft inventory requires review of removed libpg_query files and changed
Mbed TLS, Zstandard, and `THIRD_PARTY.md` content. Adapt source and artifact
inventories to the new client layout as well. Do not stamp, reformat or update reviewed license hashes
across either pure import to make those checks pass.

For P1 metadata-only edits after the source import commit, the root formatter's
`--changed` selection does not include the imported client. Avoid `--all` or
comparisons spanning that import until vendor classification is adapted.

The subsequent main synchronization preserves these P1 source identities and
import parents. It updates the reference code, tests, and this inventory to
`06cfbd5cc39f8b2c19531676f2bdcfff64e1a6e4`; it does not restore a 2.0 build
or qualify any capability row.

## Capability completion record

No capability row has been qualified yet. The next implementation step is P2,
starting with B01 and the public API comparison required by B02.

# Distributed DuckDB extension architecture

## Scope

This document defines the explicit contracts that let a DuckDB extension
participate in Vane's Ray distributed execution. The engine owns scheduling,
transport, and the coordinator transaction boundary; each extension continues
to own its bind state, scan semantics, artifact production, and catalog mutation
logic.

An extension participates by attaching a concrete distributed scan contract to
a normal table function or by registering a concrete distributed write hook.
The loader derives one internal manifest from those implementations. Extension
authors do not separately register generic declarations. Vane does not infer
extension behavior, adapt older contracts, or execute unsupported extension
operators locally.

The derived manifest is additional metadata. It does not replace DuckDB's
`ExtensionLoader` registrations. The same extension therefore keeps its
original scalar, aggregate, table, type, cast, COPY, PRAGMA, collation, secret,
filesystem, replacement-scan, storage, parser, planner, and optimizer behavior
when it runs through DuckDB's native runner.

## Design rules

- Extensions used by Ray are pinned and reviewed. They are either statically
  linked into the Vane release artifact or supplied by an exact preinstalled
  dynamic provider whose descriptor and artifact are verified during worker
  preparation.
- The coordinator resolves catalog state and selects immutable work before
  worker execution starts.
- Workers receive portable split data and restore extension state through the
  extension's normal DuckDB `serialize`/`deserialize` callbacks.
- The scheduler treats extension split payloads as opaque.
- Workers may produce immutable files, but only the coordinator may mutate an
  extension catalog transaction.
- Invalid or incomplete distributed contracts fail before distributed side
  effects begin.
- Missing, duplicate, or version-mismatched distributed capabilities are hard
  errors; there is no compatibility mode or local fallback.

## Registration model

An extension declares its distributed protocol from the same `Load` method
that performs its ordinary DuckDB registrations:

```cpp
void IcebergExtension::Load(ExtensionLoader &loader) {
    auto scan_functions = GetIcebergScanFunction(loader);
    TableFunctionDistributedScanCallbacks scan_callbacks;
    scan_callbacks.protocol_version = 1;
    scan_callbacks.split_codec = {"iceberg.scan-split", 1};
    scan_callbacks.plan_splits = IcebergPlanScanSplits;
    scan_callbacks.create_worker_bind = IcebergCreateWorkerBind;
    scan_callbacks.apply_splits = IcebergApplyScanSplits;
    scan_functions.SetDistributedScanCallbacks(std::move(scan_callbacks));
    // This remains the ordinary DuckDB catalog registration. The loader
    // derives and stages the table-function capability from the callbacks.
    loader.RegisterFunction(std::move(scan_functions));

    DistributedWriteOperatorExtension write;
    write.name = "iceberg_write_fragments";
    write.protocol_version = 1;
    write.mode = DistributedWriteMode::CALLBACK;
    write.fragment_codec = {"iceberg.commit-fragment", 1};
    write.callbacks = IcebergWriteCallbacks();
    DistributedWriteOperatorExtension::Register(loader, std::move(write));
}
```

Each table-function overload with callbacks owns a capability and protocol
version, identified by its canonical catalog signature (for example,
`range(BIGINT, BIGINT)`). Different overloads of one function may evolve
independently, and distributed and native-only overloads may coexist. Write
contracts remain identified by operator name. These signatures, names, and
versions are stable wire identities, not display labels. The current registry
accepts only implemented hook kinds: distributed table scans and distributed
write operators. It has no public placeholder API for hypothetical aggregate,
COPY, storage, or context protocols.

Ordinary DuckDB registrations remain available on every prepared worker
DatabaseInstance after the exact static or provider-backed dynamic extension is
loaded:

- ordinary worker-safe scalar functions, types, casts, and collations use their
  normal DuckDB registrations after the exact extension is loaded;
- table sources that require distributed partitioning add portable split
  enumeration, explicit worker-bind construction, and worker split application;
- custom write roots add either the fixed file adapter or a worker fragment
  sink plus coordinator commit behavior;
- replacement scans and parser, planner, and optimizer hooks normally run on
  the coordinator; any custom physical operator they produce still needs an
  implemented distributed hook.

`ExtensionLoader` automatically stages a table-function capability when the
normally registered `TableFunction` carries distributed scan callbacks. The
extension name, canonical overload signature, and capability kind are derived
rather than repeated by the extension author. A write hook similarly receives
its extension identity from the loader. The loader stages every concrete
overload contract and publishes the manifest and write implementations
atomically only when loading finalizes successfully. Duplicate overload
identities, zero protocol versions, incomplete callbacks, and non-canonical
names are rejected.

`DistributedWriteOperatorExtension` is a hook type, not another loadable
`Extension`: it has no `Load`, `Name`, or `Version` lifecycle. The one top-level
extension owns it and registers it from that extension's existing `Load`
method, matching DuckDB's `OperatorExtension`, `ParserExtension`,
`PlannerExtension`, and `OptimizerExtension` layering.

## Build and loading

The connection snapshot records the content-derived DuckDB `SourceID`, every
loaded static extension name and exact extension version, an ordered dynamic
extension descriptor manifest, and a sorted list of canonical distributed
contract identities. Each dynamic descriptor carries the artifact SHA-256,
platform and source identity, trust identity, and direct dependencies. Vane
records it only after `DynamicExtensionResolver.load()` verifies the local
artifact and DuckDB accepts the cached bytes. A non-static extension loaded by
another route makes snapshot capture fail closed.

Coordinator progress-topology inspection clones each fragment with an isolated
cursor from the resolver-owned planning DatabaseInstance. It reuses the
already-verified extension state and does not require a provider entry point on
the driver after an explicit artifact load. This coordinator path is
verification-only and does not apply worker security settings to the user's
DatabaseInstance.

Fragment registration prepares an isolated worker DatabaseInstance before any
task can deserialize the plan. Immediately before native admission, the worker
refreshes the exact database identity and prepares any cache miss first, such
as one caused by S3 credential rotation. Preparation validates the `SourceID`,
loads the declared static extensions, resolves only the exact preinstalled
`vane.dynamic_extension_providers` entry points, verifies the complete
dependency graph before loading, invokes the existing resolver in manifest
order, and compares registered distributed contracts. The manifest is part of the worker
database/cache identity. The worker acquires a cursor from that prepared entry
before admission; the cursor lease prevents credential rotation from retiring
the DatabaseInstance during handoff. Execution issues no further dynamic-extension
load and verifies the exact recorded descriptors, native loaded names, static
identities, and contracts.

No coordinator artifact path or binary payload is transported. Workers do not
scan an implicit directory or use a repository, network download, autoinstall,
autoload, unsigned loading, compatibility behavior, or fallback artifact.
Worker bootstrap sanitization ignores coordinator extension/home directories
and extension repository settings, so the resolver cache remains node-local.
Missing or ambiguous providers, altered bytes, trust failures, platform or
source mismatches, invalid dependency order, and worker disagreement fail
deterministically during preparation.

The snapshot schema is strict. Legacy name-only extension lists, absent
contract data, extra worker contracts, and any protocol mismatch are rejected.
This validation occurs before task admission and plan deserialization.

The snapshot also carries declarations of attached catalogs needed to plan a
transported logical plan. Transported logical plans use an isolated planning
DatabaseInstance, and attached catalogs are recreated only there. The
declarations are removed when the coordinator produces a physical worker plan.
Workers consume the immutable bind state serialized by the table function and
never receive attachment SQL or repeat coordinator metadata binding.

Connection snapshots do not transport DuckDB `CREATE SECRET` objects. A Ray
worker disables host persistent-secret loading before a newly opened snapshot
database first uses its secret manager, so host secrets are never an implicit
credential fallback. An extension operation that requires a DuckDB secret on a
worker is unsupported unless it also accepts credentials through Vane's explicit
session or connection-setting path. Persistent source databases are opened
read-only, and exact snapshot databases are retained for the worker lifetime.
The worker database identity includes the canonical set of replayed settings,
so snapshots with different global-only setting values never mutate the same
DatabaseInstance. Read-only worker instances let the same source file support
isolated exact extension and setting identities without sharing a mutable
catalog; extension catalog commits remain coordinator-only.

Connection snapshots can contain explicit S3 connection settings and session
configuration. These are trusted coordinator-to-worker payloads and must remain
inside the authenticated Ray cluster; they are not an untrusted interchange
format or a persistence format. Executable attachment declarations travel only
as far as the isolated coordinator planning connection.

Storage-facing scalar functions from the static `file` extension use this same
contract. FILE values retain only their canonical five fields; connection and
credential state travels in the separate query/session snapshot. The official
Vane build loads the statically linked `file` and `httpfs` extensions, and the
existing snapshot records and verifies those exact extension identities before
Worker execution. Workers never install or autoload them. Local FILE URLs
identify Worker-visible paths and therefore require a shared or identically
mounted filesystem when a query can run on more than one node.

## Distributed scans

### Worker bind serialization

An extension scan must implement both of DuckDB's normal table-function
`serialize` and `deserialize` callbacks. Its distributed `create_worker_bind`
callback constructs a new bind by explicitly selecting only state required by
worker execution. It must not retain coordinator split collections, catalog or
client-context pointers, or other mutable process-local state. The returned
bind is serialized normally when the worker physical plan is transported.
`FunctionData::Copy()` keeps its ordinary DuckDB meaning and is not used to
construct this distributed worker bind.

```cpp
unique_ptr<FunctionData>
IcebergCreateWorkerBind(const TableFunctionDistributedScanInput &input) {
    const auto &source = input.bind_data.Cast<IcebergScanBindData>();
    auto worker = make_uniq<IcebergScanBindData>();
    worker->snapshot_id = source.snapshot_id;
    worker->schema = source.schema;
    worker->projected_columns = source.projected_columns;
    worker->filters = source.filters;
    worker->scan_options = source.scan_options;
    return worker;
}
```

After loading the required extension through its declared static or verified
dynamic path, each worker resolves the normal DuckDB table function from its
catalog and calls its `deserialize` callback. The worker never invokes the
original bind callback and never repeats catalog or metadata planning. Missing
or incomplete bind serde is a hard error.

### Extension-owned splits

`TableFunctionDistributedScanCallbacks` is attached directly to the registered
DuckDB `TableFunction`. It provides the table-function capability protocol
version, a split codec identity, and three operations:

1. `plan_splits` expands coordinator bind state into independently assignable
   opaque split envelopes.
2. `create_worker_bind` constructs the minimal split-free worker bind.
3. `apply_splits` decodes and installs the splits assigned to one worker task
   after worker bind
   deserialization.

The ordinary `loader.RegisterFunction(...)` call binds the complete
`DistributedExtensionCapabilityReference` from the loader's extension identity
and the table function overload's canonical catalog signature. No second
table-function registration is required, native-only overloads remain
unaffected, and native DuckDB continues to execute the original bind/init/scan
callbacks.

Each envelope contains a stable `split_id`, one opaque payload, and optional
cardinality/byte estimates. The split codec describes the complete extension
payload. An adapter that needs paths, delete vectors, credentials, partition
metadata, or other fat split state serializes all of it inside that payload.
Vane serializes the outer envelope with DuckDB's binary serializer but never
interprets the payload. A `split_id` is an exact opaque identity and is never
trimmed or otherwise normalized. File-backed and non-file extensions use the
same envelope; `OpenFileInfo` is reserved for the engine-owned MultiFile path.

`plan_splits` receives `TableFunctionDistributedScanPlanningInput`, whose
`target_split_count` is a granularity hint derived from available worker slots
and `VANE_RAY_SCAN_SPLIT_MIN_COUNT`. It is not a task count. An extension may
return more or fewer splits, but every returned split must remain independently
executable and have an ID unique within that logical scan source.

The planning input also borrows the coordinator `FileSystem` for the duration
of the callback. File-backed implementations may use it to inspect the
already-selected files and produce size estimates or safe byte boundaries.
They must not retain the filesystem or any connection-level resource in worker
bind data.

Extensions may return per-split byte and cardinality estimates for balancing.
Vane never opens or parses an extension payload to infer those estimates, even
when the payload encodes a URI; split interpretation remains extension-owned.
The byte estimate describes logical scan input, not serialized envelope size.

Source planning emits one engine `ScanSplit` for each extension envelope and
wraps it in a singleton `ScanSplitBatch` only to reuse the typed transport. The
FTE scheduler is the only layer that may merge multiple singleton batches into
the batch assigned to one source in one task attempt. A batch is never a second
planning or scheduling unit. Batches carry the exact extension capability and
codec version, and only compatible batches with distinct split IDs may be
merged. This gives each split an independent scheduling, retry, and
observability identity while still allowing one task to consume multiple
splits.

Every worker extension scan must receive an explicit assignment before executor
initialization. When planning yields no splits, the coordinator emits one
explicit empty split. That marker is the only representation of an empty scan,
whether assigned statically or delivered through an FTE queue. Extension state
is never replaced with an engine-owned file list.

Engine-owned MultiFile scans use the same strict assignment boundary. Their
worker bind copy contains an empty `SimpleMultiFileList`; static batches or
an FTE queue replace it before executor initialization. Planning emits one
`ScanSplit` per `OpenFileInfo`, including a stable ordinal identity for repeated
occurrences of the same path. A zero-file MultiFile scan is represented by one
explicit empty file split. The coordinator's original file list is never
retained as a worker fallback.

Vane's built-in CSV reader uses the extension-owned contract because a single
seekable CSV file can be split below file granularity. A multi-file scan emits
one split per bound file. A single uncompressed UTF-8 file may emit explicit
byte ranges, with overlap used only to finish the record that starts inside a
range. The worker bind serializes the bound schema, projection-facing column
metadata, reader options, union-by-name per-file state, and complete
`OpenFileInfo` options. It is detached until an assignment is applied, and an
explicit empty assignment remains a zero-row scan. Compressed, non-seekable,
non-UTF-8, `skip_rows`, and leading-row header detection cannot be byte-range
split, so a single replayable input in one of those modes produces one explicit
whole-file split even when the target granularity hint is larger. Pipe inputs
are rejected at every granularity because they cannot provide retryable tasks.
Rejects-table scans are not distributed. CSV itself has no snapshot identifier,
so referenced files must remain immutable for the lifetime of the query and its
retries; a same-size replacement cannot be detected by this protocol.

## Distributed writes

### Ordinary COPY-format file writes

Python relations can route any registered DuckDB COPY format through the
distributed file-artifact path with
`relation.write_file(path, format="format_name")`. This constructs a normal
`CopyStatement`; it does not register an extension write operator or add
format-specific behavior to Vane. The `write_csv` and `write_parquet`
convenience methods construct the same format-neutral relation while retaining
their format-specific option surfaces. The named `CopyFunction` must provide:

- owned `FunctionData::Copy()` state;
- matching `serialize` and `deserialize` callbacks;
- `copy_to_get_written_statistics` with exact row and file byte counts.

The coordinator binds the source and COPY format once. Each selected source
task receives a cloned physical COPY sink and a unique run-scoped worker output
path. Retries create a different immutable output path, and finalization uses
only the selected attempts when it publishes the manifest. An empty source is
still scheduled as an explicit empty input and must produce a valid zero-row
artifact when the format's `write_empty_file` behavior is enabled.

The output filesystem remains part of the format contract. Every worker and the
coordinator must resolve the output namespace consistently; Vane does not turn
a local-only writer into an object-store writer or copy node-local artifacts
between hosts.

### Extension-owned write roots

`ExtensionWriteTaskProvider` is the coordinator-side contract exposed by the
extension's ordinary physical root. That root is never serialized or run on a
worker. It returns only a `DistributedExtensionWritePlan`: the extension name,
write-operator name, plus opaque dynamic worker bind bytes for callback mode.
File-artifact plans must leave the worker bind empty. Vane resolves the mode,
capability protocol, fragment codec, and worker callbacks from the
database-local immutable `DistributedWriteOperatorExtension`. A physical plan
cannot override that registered contract.

In callback mode, the worker bind may carry an extension-created write handle
and artifact namespace, analogous to a connector insert handle. The physical
provider retains the coordinator half of that state. This extension-owned
handle, not Vane's query ID, is what finalization and pre-finalize abort use to
identify the write.

`PlanRunner` validates the capability, codec, provider, and worker callback
contract before translation, task selection, or artifact creation. There is no
mode inference: the physical shape must exactly match the declared mode.

Python relation INSERT, UPDATE, DELETE, and CTAS mutations follow the selected
backend strictly. This includes `insert_into`, row-value `insert`, `update`,
`delete`, and `create`/`to_table`. An unset, empty, or explicit
`VANE_RUNNER=ray` dispatches the mutation to Ray and requires the target to
translate to a registered distributed extension write; an ordinary DuckDB
table target therefore reports an unsupported distributed operator instead of
executing locally. `VANE_RUNNER=local-fast` selects native DuckDB execution as
a separate backend. Other runner values are not mutation backends, and neither
backend falls back to the other.

### File-artifact mode

`FILE_ARTIFACT` is the fixed adapter for extension operators whose single child
is DuckDB COPY configured to return `WRITTEN_FILE_STATISTICS`. Workers execute
the COPY child and write immutable output paths scoped to the current execution
and COPY sink. Vane selects successful attempts and converts every
selected file DTO into the common opaque result envelope. The provider receives
those envelopes and can decode them with `DecodeDistributedFileWriteResults`
to obtain the exact paths, row counts, sizes, footer sizes, column statistics,
and partition keys. An empty selected-result set decodes to an empty file set;
the provider does not need Vane's query ID.

Coordinator staging and rename-based publication are rejected because the
catalog must reference the exact immutable paths produced by workers. DuckDB's
ordinary distributed COPY finalizer records only whether the selected worker
artifacts were published successfully. That marker is not an extension catalog
commit and cannot prove that extension finalization ran. An ambiguous
publication result therefore terminates the extension write with an unknown
outcome; Vane does not invoke extension finalization, and it retains the
artifacts.

### Callback mode

`CALLBACK` supports extensions whose worker output is a commit fragment rather
than DuckDB's file-statistics row. `DistributedWriteOperatorExtension::Register`
registers one complete hook containing its mode, protocol, codec, and five
mandatory worker callbacks:
`initialize_global`, `initialize_local`, `sink`, `combine`, and `finalize`.
Like DuckDB's parser, planner, optimizer, storage, and operator extension hooks,
this is deliberately separate from catalog-function registration. The
distributed translator replaces the coordinator-only extension root with a
generic streaming sink on each worker input task. The sink passes the
extension's opaque `worker_bind_data` and an explicit
`DistributedWriteTaskContext` containing the current execution query ID and
runtime FTE task-attempt ID to every callback. These values correlate work
within one execution; neither is a durable commit or idempotency key.

`finalize` returns zero or more `DistributedWriteFragment` values. A fragment
has a stable ID, opaque payload, row and byte counts, and zero or more
`DistributedWriteArtifact` values. An artifact has its own stable ID, optional
URI, codec identity, and opaque payload. Binary NUL bytes are valid in all
opaque fields. Vane wraps this data in a DuckDB-binary
`DistributedWriteTaskResult`, emits one BLOB row per selected worker task, and
validates the exact query ID, capability, fragment codec, unique
task-attempt IDs, globally unique fragment IDs, and count overflows before
calling the provider. Every result envelope repeats both the query and
task-attempt identities supplied to its worker callback.

The outer envelope is the engine DTO; extension payloads are not required to be
JSON. Iceberg, Delta, DuckLake, and Lance adapters may place their native JSON,
Avro, protobuf, or other binary commit metadata inside the opaque payload while
keeping Vane independent of that schema.

Callback mode has no engine-owned publication marker because Vane cannot
interpret the extension's catalog state. The extension's own catalog
transaction and ACID rules are authoritative. Worker attempts may be retried or
speculated before coordinator finalization, so they must create immutable,
non-overwriting artifacts. Vane selects successful attempts and invokes
`FinalizeDistributedWrite` at most once for that execution. The extension write
hook layer does not retry, reconcile, or deduplicate that coordinator call.

### Coordinator finalization

Both modes use the same coordinator sequence:

1. Validate the complete provider and worker protocol before physical source
   translation, task enumeration, or side effects.
2. Execute worker sinks and select successful task attempts.
3. Validate and aggregate their opaque result envelopes.
4. Call `FinalizeDistributedWrite` once with exactly the selected envelopes
   inside Vane's active coordinator transaction.
5. Require the provider's affected-row count to match the envelope total.
6. Commit through DuckDB's transaction manager.

`AbortDistributedWrite` is mandatory. Once the current attempt may have
created worker or file output, Vane invokes abort once for a known failure only
while coordinator finalization has not started. The provider uses its own
coordinator state and extension-planned artifact namespace to locate output from
this execution.
Validation, translation, and setup failures before output is possible do not
invoke abort.

Once `FinalizeDistributedWrite` starts, Vane never invokes abort, deletes
selected artifacts, or retries finalization. A provider exception, an affected
row-count mismatch, or an exception while committing the surrounding DuckDB
transaction is terminal with an unknown outcome. The extension catalog may
already reference the selected artifacts, so Vane retains them and leaves
catalog inspection or repair to the extension or operator.

Extension writes require DuckDB auto-commit mode. Vane owns that transaction
boundary; an explicit caller-managed transaction is rejected before workers
start. This check also applies when `Runner.run_write` is called directly.

A worker or provider failure known to precede catalog commit rolls back the
transaction. Once current-attempt output may exist, that failure also invokes
explicit cleanup. Failures before that side-effect boundary return without
aborting durable state. If the catalog commit call itself fails, Vane retains
the artifacts and reports an unknown outcome: a remote catalog may have
committed before its response was lost.

## Extension author contract

For distributed table-function scan callbacks:

- declare one non-zero capability protocol version on the callbacks and let the
  ordinary `RegisterFunction` call derive the complete capability identity;
- emit deterministic split IDs and payloads without process-local pointers;
- implement complete DuckDB bind `serialize`/`deserialize` callbacks and make
  the deserializer accept a split-free worker bind;
- construct the worker bind by explicitly copying only required portable state;
  do not retain coordinator splits, catalog/context pointers, or mutable caches;
- make subset application idempotent and valid before global scan state starts;
- keep coordinator split enumeration free of logical query side effects;
- preserve schema, projection, filters, partitions, and delete semantics after
  bind deserialization and split application;
- provide split byte or cardinality metadata when available for balancing.

For every distributed write provider:

- return a `DistributedExtensionWritePlan` containing the exact registered
  extension/operator key and portable dynamic worker bind bytes;
- register the mode, protocol, codec, and callbacks once in the database-local
  immutable `DistributedWriteOperatorExtension`; do not duplicate them in the
  physical provider;
- keep `ValidateDistributedWrite` read-only and limit it to catalog and output
  precondition checks before workers start;
- accept only the selected task-result envelopes during finalization;
- use the extension catalog's own transaction and ACID mechanism in
  `FinalizeDistributedWrite` and do not depend on Vane retrying that call;
- register or commit exactly the selected files or opaque fragments from that
  one coordinator invocation;
- return the exact affected row count;
- implement `AbortDistributedWrite` for known failures before finalization
  starts, including an empty selected-result set;
- retain immutable artifacts when a remote catalog commit response is
  ambiguous; never interpret a commit exception as proof of rollback;
- never require query or task-attempt IDs to be durable idempotency keys.

For callback-mode worker code:

- register all five callbacks as one `DistributedWriteOperatorExtension` hook;
- treat `worker_bind_data`, fragment payloads, and artifact payloads as portable
  bytes with no process-local pointers;
- use the query and task-attempt IDs only to correlate the current execution;
- generate immutable, non-overwriting artifact identities for speculative or
  retried worker attempts;
- keep artifact creation worker-local and catalog mutation coordinator-only;
- report exact row and byte counts and stable fragment and artifact IDs;
- encode every fragment according to the registered opaque codec.

Both scan callbacks and write providers require normal and fault-tolerant
distributed tests before an extension is enabled in release builds.

## Failure rules

- Unknown static extension names, non-static names without an exact tracked
  dynamic descriptor, and invalid dynamic descriptor manifests are rejected.
- DuckDB source, static extension version, and distributed contract mismatches
  are rejected before planning or scheduling.
- A callback/provider whose capability identity was not registered is rejected
  before split enumeration or write validation.
- Worker rebind is not supported for distributed table functions.
- A scan function with distributed callbacks must expose both DuckDB
  serialization callbacks.
- A worker extension scan without an explicit split assignment is rejected,
  including a sealed but batch-free FTE queue; an explicit empty split
  remains a valid empty scan.
- A source node cannot receive both a static split batch and an FTE split queue.
- Extension-owned splits are never merged with engine-discovered file splits; the
  extension remains responsible for validating the opaque payloads it emits and
  receives.
- An explicit empty split produces an empty scan.
- An extension physical operator without `ExtensionWriteTaskProvider` is
  rejected before its child can run.
- A file-artifact write must have exactly one COPY sink; a callback write must
  have exactly one registered callback sink.
- Extension writes must use direct worker artifacts and explicit coordinator
  finalization.
- Extension writes reject explicit DuckDB transactions and ambiguous catalog
  commits never trigger destructive artifact cleanup.

## Framework test matrix

The engine-level suite covers scan callback discovery, opaque fat-split payload
serialization, explicit worker bind serde, split subset application,
empty assignments, schema validation, generic COPY-format relation binding,
file and callback write translation, write callback execution, binary fragment
and artifact envelopes, runtime query/task-attempt identity, pre-write
validation, selected-result propagation, row-count validation, transaction
ordering, pre-finalize abort behavior, and terminal unknown outcomes after
finalization starts.

Each concrete extension adds its own adapter tests for catalog pinning, split
semantics, object storage, commit behavior, and failure boundaries.

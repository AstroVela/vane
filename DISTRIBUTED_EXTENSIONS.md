# Distributed DuckDB extension architecture

## Scope

This document defines the explicit contracts that let a statically linked
DuckDB extension participate in Vane's Ray distributed execution. The engine
owns scheduling, transport, and the coordinator transaction boundary; each
extension continues to own its bind state, scan semantics, artifact production,
and catalog mutation logic.

An extension participates by registering an explicit distributed manifest and
implementing the corresponding callback or provider contract. Vane does not
infer extension behavior, adapt older contracts, or execute unsupported
extension operators locally.

The manifest is additional metadata. It does not replace DuckDB's
`ExtensionLoader` registrations. The same extension therefore keeps its
original scalar, aggregate, table, type, cast, COPY, PRAGMA, collation, secret,
filesystem, replacement-scan, storage, parser, planner, and optimizer behavior
when it runs through DuckDB's native runner.

## Design rules

- Extensions used by Ray are pinned, reviewed, and statically linked into the
  Vane release artifact.
- The coordinator resolves catalog state and selects immutable work before
  worker execution starts.
- Workers receive portable task data and restore extension state through the
  extension's normal DuckDB `serialize`/`deserialize` callbacks.
- The scheduler treats extension task payloads as opaque.
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
    loader.RegisterDistributedExtension(1);

    auto scan_functions = GetIcebergScanFunction(loader);
    TableFunctionDistributedScanCallbacks scan_callbacks;
    scan_callbacks.protocol_version = 1;
    scan_callbacks.task_codec = "iceberg.scan-task";
    scan_callbacks.task_codec_version = 1;
    scan_callbacks.plan = IcebergPlanScanTasks;
    scan_callbacks.prepare_bind = IcebergPrepareWorkerBind;
    scan_callbacks.apply_tasks = IcebergApplyScanTasks;
    for (auto &function : scan_functions.functions) {
        function.SetDistributedScanCallbacks(scan_callbacks);
    }
    // This remains the ordinary DuckDB catalog registration. The loader
    // derives and stages the table-function capability from the callbacks.
    loader.RegisterFunction(std::move(scan_functions));

    // A FILE_ARTIFACT provider declares only the operator capability.
    loader.RegisterDistributedOperator("iceberg_write_files", 1);
    // A CALLBACK provider also registers its complete worker sink contract.
    DistributedWriteOperatorExtension::Register(
        loader, {"iceberg_write_fragments", 1, IcebergWriteCallbacks()});
}
```

The extension-level version identifies the manifest schema. Each capability
also has its own protocol version so scan and write contracts can evolve
independently. Names and versions are stable wire identities, not display
labels.

A capability declaration is discovery and validation metadata; it does not by
itself make an unsupported operator distributable. The extension must still
implement the corresponding execution contract. The current framework enforces
table-function scan callbacks and coordinator-owned write providers, while the
remaining groups provide stable identities for worker-safe registrations and
subsequent contracts.

The available capability groups are table function, aggregate function, COPY
function, operator, storage, and context. These groups describe where the
extension adds distributed behavior rather than duplicating every DuckDB
registration:

- ordinary worker-safe scalar functions, types, casts, and collations use their
  normal DuckDB registrations after the exact extension is loaded;
- table sources add portable task enumeration, detached bind serialization, and
  worker task application;
- distributed aggregates add portable partial-state and merge behavior;
- COPY, storage, and custom write operators add worker artifact and
  coordinator commit behavior;
- context capabilities transport required settings, secrets, or filesystem
  state;
- replacement scans and parser, planner, and optimizer hooks normally run on
  the coordinator; any custom physical operator they produce still needs an
  explicit distributed operator capability.

`ExtensionLoader` automatically stages a table-function capability when the
normally registered `TableFunction` carries distributed scan callbacks. The
extension name, extension protocol, function name, and capability kind are
derived rather than repeated by the extension author. It stages the complete
manifest while the extension loads and publishes it only when loading finalizes
successfully. Duplicate extension declarations, duplicate capability
identities, zero protocol versions, mixed distributed/native-only overloads,
and non-canonical names are rejected.

## Build and loading

The connection snapshot records the content-derived DuckDB `SourceID`, every
loaded static extension name and exact extension version, and the complete
sorted distributed manifest. A worker first validates its `SourceID`, invokes
DuckDB's generated static loader, compares the loaded extension identities, and
then compares the registered manifests before deserializing a plan. Dynamically
installed extension binaries are not accepted by distributed execution.

The snapshot schema is strict. Legacy name-only extension lists, absent
manifest data, extra worker manifests, and any protocol mismatch are rejected.
This validation occurs before task scheduling.

The snapshot also carries serializable DuckDB secrets and the declarations of
attached catalogs needed to plan a transported logical plan. Secrets are
recreated as temporary worker-local secrets; non-serializable secrets are
rejected. Attached catalogs are recreated only on an isolated planning
connection. Worker physical plans consume the immutable bind state serialized by
the table function and never repeat coordinator metadata binding.

The secret list is authoritative for its DatabaseInstance. Transported plans
with secrets use an isolated driver planning database. A Ray worker disables
host persistent-secret loading before a newly opened snapshot database first
uses its secret manager, opens persistent source databases read-only, retains
exact snapshot databases for the worker lifetime, and leases one exact
snapshot-plus-secret digest through execution. Read-only worker instances let
the same source file support isolated exact extension identities without
sharing a mutable catalog; extension catalog commits remain coordinator-only.
Tasks with the same digest may execute concurrently; a different digest waits
until every current lease is released before replacing temporary secrets. If a
reused database had already loaded persistent secrets, every such secret must
be byte-identical to the source snapshot. Any extra or different persistent
secret is rejected rather than inherited, overwritten, or used as a fallback.

Connection snapshots contain credential material and executable attachment
declarations. They are trusted coordinator-to-worker payloads and must remain
inside the authenticated Ray cluster; they are not an untrusted interchange
format or a persistence format.

## Distributed scans

### Worker bind serialization

An extension scan must implement both of DuckDB's normal table-function
`serialize` and `deserialize` callbacks. The coordinator copies the bind object,
invokes the distributed `prepare_bind` callback to remove coordinator-only task
collections, and serializes the remaining worker bind state through the normal
DuckDB binary DTO path.

After loading the required static extension, each worker resolves the normal
DuckDB table function from its catalog and calls its `deserialize` callback.
The worker never invokes the original bind callback and never repeats catalog or
metadata planning. Missing or incomplete bind serde is a hard error.

### Extension-owned tasks

`TableFunctionDistributedScanCallbacks` is attached directly to the registered
DuckDB `TableFunction`. It provides the table-function capability protocol
version, a task codec identity, and three operations:

1. `plan` expands coordinator bind state into elementary opaque task envelopes.
2. `prepare_bind` removes coordinator-only tasks from a copied worker bind.
3. `apply_tasks` decodes and installs an assigned subset after worker bind
   deserialization.

The ordinary `loader.RegisterFunction(...)` call binds the complete
`DistributedExtensionCapabilityReference` from the loader's extension identity
and the table function's catalog name. No second table-function registration is
required, and native DuckDB continues to execute the original bind/init/scan
callbacks.

Each envelope contains a stable task ID, opaque payload bytes, optional binary
artifacts with their own codec identities, and optional cardinality/byte
estimates. Vane serializes the envelope with DuckDB's binary serializer but does
not interpret its payload or artifacts. File-backed and non-file extensions use
the same envelope; `OpenFileInfo` is reserved for the engine-owned MultiFile
path.

Extensions may return per-task byte and cardinality estimates for balancing.
Vane never opens or parses an extension payload to infer those estimates, even
when the payload encodes a URI; task interpretation remains extension-owned.
The byte estimate describes logical scan input, not serialized envelope size.

Vane groups these tasks into existing `ScanTaskDescriptor` objects and
transports them through normal and fault-tolerant execution. Descriptors carry
the exact extension capability and codec version, and only matching descriptors
may be merged or applied. Every worker extension scan must receive an explicit
assignment before executor initialization. When planning yields no elementary
tasks, the coordinator emits one explicit descriptor containing zero opaque
envelopes. That descriptor is the only representation of an empty scan, whether
assigned statically or delivered through an FTE queue. Extension state is never
replaced with an engine-owned file list.

Engine-owned MultiFile scans use the same strict assignment boundary. Their
worker bind copy contains an empty `SimpleMultiFileList`; static descriptors or
an FTE queue replace it before executor initialization. A zero-file MultiFile
scan is represented by one explicit empty file descriptor. The coordinator's
original file list is never retained as a worker fallback.

## Distributed writes

`ExtensionWriteTaskProvider` is the coordinator-side contract exposed by the
extension's ordinary physical root. That root is never serialized or run on a
worker. Its immutable `DistributedExtensionWriteInfo` selects one of two strict
worker protocols and carries the exact operator capability, fragment codec, and
opaque worker bind bytes.

`PlanRunner` validates the capability, codec, provider, and worker callback
contract before translation, task selection, or artifact creation. There is no
mode inference: the physical shape must exactly match the declared mode.

### File-artifact mode

`FILE_ARTIFACT` is the fixed adapter for extension operators whose single child
is DuckDB COPY configured to return `WRITTEN_FILE_STATISTICS`. Workers execute
the COPY child and write directly to a stable namespace derived from the query
identity and COPY sink ID. Vane selects successful attempts and converts every
selected file DTO into the common opaque result envelope. The provider receives
those envelopes and can decode them with `DecodeDistributedFileWriteResults`,
passing the same operation context, to obtain the exact paths, row counts,
sizes, footer sizes, column statistics, and partition keys.

Coordinator staging and rename-based publication are rejected because the
catalog must reference the exact immutable paths produced by workers. Before
provider finalization, Vane persists a `catalog_commit_pending` lifecycle fence
so age-based cleanup cannot delete files that a remote catalog may already
reference. After the owned catalog transaction commits, Vane publishes the
committed file marker. A retry with the same query identity returns that marker
without invoking the provider again.

### Callback mode

`CALLBACK` supports extensions whose worker output is a commit fragment rather
than DuckDB's file-statistics row. `DistributedWriteOperatorExtension::Register`
registers a write hook containing five mandatory worker callbacks:
`initialize_global`, `initialize_local`, `sink`, `combine`, and `finalize`.
Like DuckDB's parser, planner, optimizer, storage, and operator extension hooks,
this is deliberately separate from catalog-function registration. The
distributed translator replaces the coordinator-only extension root with a
generic streaming sink on each worker input task. The sink passes the
extension's opaque `worker_bind_data` and an explicit
`DistributedWriteTaskContext` containing the stable operation ID and runtime
FTE task-attempt ID to every callback. Extensions never parse Vane's internal
task-attempt naming scheme to recover the operation namespace.

`finalize` returns zero or more `DistributedWriteFragment` values. A fragment
has a stable ID, opaque payload, row and byte counts, and zero or more
`DistributedWriteArtifact` values. An artifact has its own stable ID, optional
URI, codec identity, and opaque payload. Binary NUL bytes are valid in all
opaque fields. Vane wraps this data in a DuckDB-binary
`DistributedWriteTaskResult`, emits one BLOB row per selected worker task, and
validates the exact operation ID, capability, fragment codec, unique
task-attempt IDs, globally unique fragment IDs, and count overflows before
calling the provider. Every result envelope repeats both the operation and
task-attempt identities supplied to its worker callback.

The outer envelope is the engine DTO; extension payloads are not required to be
JSON. Iceberg, Delta, DuckLake, and Lance adapters may place their native JSON,
Avro, protobuf, or other binary commit metadata inside the opaque payload while
keeping Vane independent of that schema.

Callback mode has no engine-owned publication marker because Vane cannot
interpret the extension's catalog state. A retry therefore presents the same
stable operation ID to the provider. `ValidateDistributedWrite` must reconcile
any durable state from an earlier attempt, and `FinalizeDistributedWrite` must
be idempotent for that operation, including when an earlier catalog commit
succeeded but its response was lost.

### Coordinator finalization

Both modes use the same coordinator sequence:

1. Validate the complete provider and worker protocol, passing the stable Vane
   operation/query ID, before side effects.
2. Execute worker sinks and select successful task attempts.
3. Validate and aggregate their opaque result envelopes.
4. Call `FinalizeDistributedWrite` with exactly the selected envelopes inside
   Vane's active coordinator transaction.
5. Require the provider's affected-row count to match the envelope total.
6. Commit through DuckDB's transaction manager.
7. For `FILE_ARTIFACT`, publish the committed file marker after catalog commit.

`AbortDistributedWrite` is mandatory. Vane supplies the stable operation ID
and invokes it once for known pre-commit failures, including cases where no
result envelope was returned, so the provider must be able to locate
operation-owned artifacts from coordinator state and deterministic identities.
A catalog commit exception is an unknown
outcome: Vane retains all opaque or file artifacts and does not call abort.

Extension writes require DuckDB auto-commit mode. Vane owns that transaction
boundary; an explicit caller-managed transaction is rejected before workers
start because Vane could not safely publish a marker before the caller's later
commit. File-artifact retries inspect the stable namespace before validation or
worker scheduling. A valid committed marker returns the persisted result
without invoking the provider again.

A worker, validation, provider, or other failure known to precede catalog
commit rolls back the transaction and invokes explicit cleanup. If the catalog
commit call itself fails, Vane retains the artifacts and reports an unknown
outcome: a remote catalog may have committed before its response was lost.
Failure to publish the file-mode marker after a known successful catalog commit
is also reported as an unknown output-lifecycle outcome because the committed
catalog transaction cannot be rolled back.

A file-mode prepared manifest without a committed marker is therefore not
retryable by the generic engine. Vane retains its files and rejects automatic
replay because it cannot prove whether a remote catalog committed before its
response was lost. Extension-specific reconciliation may establish that
outcome separately.

## Extension author contract

For distributed table-function scan callbacks:

- declare one non-zero capability protocol version on the callbacks and let the
  ordinary `RegisterFunction` call derive the complete capability identity;
- emit deterministic task payloads without process-local pointers;
- implement complete DuckDB bind `serialize`/`deserialize` callbacks and make
  the deserializer accept detached task state;
- detach coordinator-only tasks without changing the original bind object;
- make subset application idempotent and valid before global scan state starts;
- keep coordinator task enumeration free of logical query side effects;
- preserve schema, projection, filters, partitions, and delete semantics after
  bind deserialization and task application;
- provide task byte or cardinality metadata when available for balancing.

For every distributed write provider:

- return the exact registered operator capability identity;
- return a complete immutable `DistributedExtensionWriteInfo` with a stable
  diagnostic name, explicit mode, exact fragment codec, and portable worker
  bind bytes;
- validate all catalog and output preconditions before workers start;
- use the supplied stable operation ID as the root of the complete speculative
  artifact namespace, including when no worker envelope is returned;
- accept only the selected task-result envelopes during finalization;
- reconcile the operation-owned task-attempt namespace against those selected
  envelopes and remove artifacts from unselected or retried attempts before
  registering catalog state;
- make validation and finalization idempotent for a repeated operation ID and
  reconcile an earlier ambiguous callback-mode commit before new workers run;
- register their files or opaque fragments in the active coordinator
  transaction without committing it;
- return the exact affected row count;
- implement `AbortDistributedWrite` for known pre-commit failure, including an
  empty selected-result set;
- retain immutable artifacts when a remote catalog commit response is
  ambiguous; never interpret a commit exception as proof of rollback;
- leave transaction commit and output-lifecycle publication to Vane.

For callback-mode worker code:

- register all five callbacks as one `DistributedWriteOperatorExtension` hook;
- treat `worker_bind_data`, fragment payloads, and artifact payloads as portable
  bytes with no process-local pointers;
- use the supplied operation ID as the artifact namespace and task-attempt ID
  to make speculative retries independently identifiable and deterministic;
- keep artifact creation worker-local and catalog mutation coordinator-only;
- report exact row and byte counts and stable fragment and artifact IDs;
- make the registered codec match `DistributedExtensionWriteInfo` exactly.

Both scan callbacks and write providers require normal and fault-tolerant
distributed tests before an extension is enabled in release builds.

## Failure rules

- Unknown or non-static extension names in a connection snapshot are rejected.
- DuckDB source, static extension version, and distributed manifest mismatches
  are rejected before planning or scheduling.
- A callback/provider whose capability identity was not registered is rejected
  before task enumeration or write validation.
- Worker rebind is not supported for distributed table functions.
- A scan function with distributed callbacks must expose both DuckDB
  serialization callbacks.
- A worker extension scan without an explicit task descriptor is rejected,
  including a sealed but descriptor-free FTE queue; an explicit empty envelope
  remains a valid empty scan.
- A source node cannot receive both a static descriptor and an FTE split queue.
- Extension-owned tasks are never merged with engine-discovered file tasks; the
  extension remains responsible for validating the opaque payloads it emits and
  receives.
- An empty task assignment produces an empty scan.
- An extension physical operator without `ExtensionWriteTaskProvider` is
  rejected before its child can run.
- A file-artifact write must have exactly one COPY sink; a callback write must
  have exactly one registered callback sink.
- Extension writes must use direct worker artifacts and explicit coordinator
  finalization.
- Extension writes reject explicit DuckDB transactions and ambiguous catalog
  commits never trigger destructive artifact cleanup.

## Framework test matrix

The engine-level suite covers scan callback discovery, opaque scan task and
artifact serialization, detached worker bind serde, task subset application,
empty assignments, schema validation, file and callback write translation,
write callback execution, binary fragment and artifact envelopes, runtime
operation/task-attempt identity, pre-write validation, selected-result propagation,
row-count validation, transaction ordering, abort behavior, output cleanup,
catalog-commit fencing, and final marker publication.

Each concrete extension adds its own adapter tests for catalog pinning, task
semantics, object storage, commit behavior, and failure boundaries.

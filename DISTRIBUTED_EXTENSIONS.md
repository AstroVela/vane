# Distributed DuckDB extension architecture

## Scope

This document defines the explicit contracts that let a statically linked
DuckDB extension participate in Vane's Ray distributed execution. The engine
owns scheduling, transport, and the coordinator transaction boundary; each
extension continues to own its bind state, scan semantics, artifact production,
and catalog mutation logic.

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
    auto scan_functions = GetIcebergScanFunction(loader);
    TableFunctionDistributedScanCallbacks scan_callbacks;
    scan_callbacks.protocol_version = 1;
    scan_callbacks.task_codec = {"iceberg.scan-task", 1};
    scan_callbacks.plan = IcebergPlanScanTasks;
    scan_callbacks.create_worker_bind = IcebergCreateWorkerBind;
    scan_callbacks.apply_tasks = IcebergApplyScanTasks;
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

Each concrete capability owns a protocol version so scan and write contracts
can evolve independently. Names and versions are stable wire identities, not
display labels. The current registry accepts only implemented hook kinds:
distributed table scans and distributed write operators. It has no public
placeholder API for hypothetical aggregate, COPY, storage, or context
protocols.

Ordinary DuckDB registrations remain available on every statically linked
worker:

- ordinary worker-safe scalar functions, types, casts, and collations use their
  normal DuckDB registrations after the exact extension is loaded;
- table sources that require distributed partitioning add portable task
  enumeration, explicit worker-bind construction, and worker task application;
- custom write roots add either the fixed file adapter or a worker fragment
  sink plus coordinator commit behavior;
- replacement scans and parser, planner, and optimizer hooks normally run on
  the coordinator; any custom physical operator they produce still needs an
  implemented distributed hook.

`ExtensionLoader` automatically stages a table-function capability when the
normally registered `TableFunction` carries distributed scan callbacks. The
extension name, function name, and capability kind are derived rather than
repeated by the extension author. A write hook similarly receives its extension
identity from the loader. The loader stages every concrete contract and
publishes the manifest and write implementations atomically only when loading
finalizes successfully. Duplicate identities, zero protocol versions, mixed
distributed/native-only overloads, incomplete callbacks, and non-canonical
names are rejected.

`DistributedWriteOperatorExtension` is a hook type, not another loadable
`Extension`: it has no `Load`, `Name`, or `Version` lifecycle. The one top-level
extension owns it and registers it from that extension's existing `Load`
method, matching DuckDB's `OperatorExtension`, `ParserExtension`,
`PlannerExtension`, and `OptimizerExtension` layering.

## Build and loading

The connection snapshot records the content-derived DuckDB `SourceID`, every
loaded static extension name and exact extension version, and a sorted list of
canonical distributed contract identities. A worker first validates its
`SourceID`, invokes DuckDB's generated static loader, compares the loaded
extension identities, and then compares the registered contracts before
deserializing a plan. Dynamically
installed extension binaries are not accepted by distributed execution.

The snapshot schema is strict. Legacy name-only extension lists, absent
contract data, extra worker contracts, and any protocol mismatch are rejected.
This validation occurs before task scheduling.

The snapshot also carries declarations of attached catalogs needed to plan a
transported logical plan. Transported logical plans use an isolated planning
DatabaseInstance, and attached catalogs are recreated only there. Worker
physical plans consume the immutable bind state serialized by the table
function and never repeat coordinator metadata binding.

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

Connection snapshots can contain explicit S3 connection settings, session
configuration, and executable attachment declarations. They are trusted
coordinator-to-worker payloads and must remain inside the authenticated Ray
cluster; they are not an untrusted interchange format or a persistence format.

## Distributed scans

### Worker bind serialization

An extension scan must implement both of DuckDB's normal table-function
`serialize` and `deserialize` callbacks. Its distributed `create_worker_bind`
callback constructs a new bind by explicitly selecting only state required by
worker execution. It must not retain coordinator task collections, catalog or
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

After loading the required static extension, each worker resolves the normal
DuckDB table function from its catalog and calls its `deserialize` callback.
The worker never invokes the original bind callback and never repeats catalog or
metadata planning. Missing or incomplete bind serde is a hard error.

### Extension-owned tasks

`TableFunctionDistributedScanCallbacks` is attached directly to the registered
DuckDB `TableFunction`. It provides the table-function capability protocol
version, a task codec identity, and three operations:

1. `plan` expands coordinator bind state into elementary opaque task envelopes.
2. `create_worker_bind` constructs the minimal task-free worker bind.
3. `apply_tasks` decodes and installs an assigned subset after worker bind
   deserialization.

The ordinary `loader.RegisterFunction(...)` call binds the complete
`DistributedExtensionCapabilityReference` from the loader's extension identity
and the table function's catalog name. No second table-function registration is
required, and native DuckDB continues to execute the original bind/init/scan
callbacks.

Each envelope contains a stable task ID, one opaque payload, and optional
cardinality/byte estimates. The task codec describes the complete extension
payload. An adapter that needs paths, delete vectors, credentials, partition
metadata, or other fat task state serializes all of it inside that payload.
Vane serializes the outer envelope with DuckDB's binary serializer but never
interprets the payload. File-backed and non-file extensions use the same
envelope; `OpenFileInfo` is reserved for the engine-owned MultiFile path.

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
worker. It returns only a `DistributedExtensionWritePlan`: the extension name,
write-operator name, and opaque dynamic worker bind bytes. Vane resolves the
mode, capability protocol, fragment codec, and worker callbacks from the
database-local immutable `DistributedWriteOperatorExtension`. A physical plan
cannot override that static contract.

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
committed file marker. PlanRunner's replay path for the same query identity
validates the provider and then returns that marker without invoking worker
execution or provider finalization again. The Ray driver may instead replay an
outcome already present in its terminal cache without re-entering PlanRunner.

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
interpret the extension's catalog state. Vane therefore does not implement
engine-level exactly-once commit, ask validation to determine whether an older
attempt committed, or short-circuit worker execution as already committed.
`ValidateDistributedWrite` performs read-only precondition checks. A retry may
execute the source and worker sink again with the same stable operation ID.
FTE task-attempt IDs distinguish speculative attempts within one execution but
may repeat after a complete same-operation resubmission; they are not durable
artifact names. `FinalizeDistributedWrite` must use the extension's own catalog
transaction and idempotency mechanism to commit the selected attempt or
recognize its existing commit without changing catalog state. Retry artifacts
must use extension-generated, non-overwriting identities so they cannot replace
objects referenced by the existing commit. The explicit retry surface is
`relation.insert_into(..., operation_id=<previous-id>)`, which forwards to
`Runner.run_write(..., operation_id=<previous-id>)`. Callback outcome-unknown
errors permit only a same-ID retry; the Ray driver rejects a different logical
plan with the same ID while allowing refreshed connection credentials for the
same write intent. File-artifact outcome-unknown errors remain non-retryable
through this surface because their engine-owned fence must be resolved first.
Once a callback outcome becomes unknown, a failed retry retains the same-ID
requirement.

### Coordinator finalization

Both modes use the same coordinator sequence:

1. Validate the complete provider and worker protocol, passing the stable Vane
   operation/query ID, before physical source translation, task enumeration, or
   side effects.
2. Execute worker sinks and select successful task attempts.
3. Validate and aggregate their opaque result envelopes.
4. Call `FinalizeDistributedWrite` with exactly the selected envelopes inside
   Vane's active coordinator transaction.
5. Require the provider's affected-row count to match the envelope total.
6. Commit through DuckDB's transaction manager.
7. For `FILE_ARTIFACT`, publish the committed file marker after catalog commit.

A Ray-driver terminal cache hit returns the previously recorded outcome and
does not repeat this coordinator sequence.
The driver also retains a lightweight operation-ID-to-plan-fingerprint
tombstone until that driver exits. The fingerprint survives session closure and
terminal-result eviction. Detaching an owner may discard result payloads and
ownership records, but it never makes the operation ID available to a different
write intent while another client keeps the driver alive. The fingerprint
covers the serialized logical plan, semantic connection replay state, and the
transported Python UDF implementations. Transport-only session identities and
refreshable credential values are excluded.

`AbortDistributedWrite` is mandatory. Once the current attempt may have
created worker or file output, Vane supplies the stable operation ID and invokes
abort once for known pre-commit failures, including cases where no result
envelope was returned. The provider must locate current-attempt artifacts from
coordinator state and deterministic identities, consult its catalog, and never
delete artifacts referenced by a committed attempt with the same operation ID.
Validation, translation, and setup failures before current-attempt output is
possible do not invoke abort: the same operation ID may name an earlier attempt
whose durable state must be preserved. A catalog commit exception is an unknown
outcome, so Vane retains all opaque or file artifacts and does not call abort.

Extension writes require DuckDB auto-commit mode. Vane owns that transaction
boundary; an explicit caller-managed transaction is rejected before workers
start because Vane could not safely publish a marker before the caller's later
commit. After provider validation, file-artifact retries inspect the stable
namespace before worker scheduling. A valid committed marker returns the
persisted result without invoking provider finalization again.

A worker or provider failure known to precede catalog commit rolls back the
transaction. Once current-attempt output may exist, that failure also invokes
explicit cleanup. Failures before that side-effect boundary return without
aborting durable state. If the catalog commit call itself fails, Vane retains
the artifacts and reports an unknown outcome: a remote catalog may have
committed before its response was lost.
Failure to publish the file-mode marker after a known successful catalog commit
is also reported as an unknown output-lifecycle outcome because the committed
catalog transaction cannot be rolled back.

A file-mode prepared manifest without a committed marker is therefore not
retryable by the generic engine. Vane retains its files and rejects automatic
replay because it cannot prove whether a remote catalog committed before its
response was lost. Extension-specific catalog recovery may establish that
outcome separately.

## Extension author contract

For distributed table-function scan callbacks:

- declare one non-zero capability protocol version on the callbacks and let the
  ordinary `RegisterFunction` call derive the complete capability identity;
- emit deterministic task payloads without process-local pointers;
- implement complete DuckDB bind `serialize`/`deserialize` callbacks and make
  the deserializer accept a task-free worker bind;
- construct the worker bind by explicitly copying only required portable state;
  do not retain coordinator tasks, catalog/context pointers, or mutable caches;
- make subset application idempotent and valid before global scan state starts;
- keep coordinator task enumeration free of logical query side effects;
- preserve schema, projection, filters, partitions, and delete semantics after
  bind deserialization and task application;
- provide task byte or cardinality metadata when available for balancing.

For every distributed write provider:

- return a `DistributedExtensionWritePlan` containing the exact registered
  extension/operator key and portable dynamic worker bind bytes;
- register the mode, protocol, codec, and callbacks once in the static
  `DistributedWriteOperatorExtension`; do not duplicate them in the physical
  provider;
- keep `ValidateDistributedWrite` read-only and limit it to catalog and output
  precondition checks before workers start;
- use the supplied stable operation ID as the root of the complete speculative
  artifact namespace, including when no worker envelope is returned;
- accept only the selected task-result envelopes during finalization;
- use the extension catalog's own ACID and idempotency mechanism in
  `FinalizeDistributedWrite`; Vane does not provide callback exactly-once or a
  pre-execution committed-result shortcut;
- when the same operation ID is retried, preserve any existing committed
  catalog state and its referenced artifacts, and dispose of redundant
  uncommitted artifacts without overwriting committed objects;
- register their files or opaque fragments in the active coordinator
  transaction without committing it;
- return the exact affected row count;
- implement `AbortDistributedWrite` for known pre-commit failure, including an
  empty selected-result set, while preserving catalog-referenced artifacts from
  an earlier committed attempt;
- retain immutable artifacts when a remote catalog commit response is
  ambiguous; never interpret a commit exception as proof of rollback;
- leave transaction commit and output-lifecycle publication to Vane.

For callback-mode worker code:

- register all five callbacks as one `DistributedWriteOperatorExtension` hook;
- treat `worker_bind_data`, fragment payloads, and artifact payloads as portable
  bytes with no process-local pointers;
- use the supplied operation ID as the artifact namespace and task-attempt ID
  to distinguish speculative retries within the current distributed execution;
- do not treat the task-attempt ID as globally unique across complete
  same-operation resubmissions; generate immutable, non-overwriting artifact
  identities and never replace an object that a committed catalog entry may
  reference;
- keep artifact creation worker-local and catalog mutation coordinator-only;
- report exact row and byte counts and stable fragment and artifact IDs;
- encode every fragment according to the registered opaque codec.

Both scan callbacks and write providers require normal and fault-tolerant
distributed tests before an extension is enabled in release builds.

## Failure rules

- Unknown or non-static extension names in a connection snapshot are rejected.
- DuckDB source, static extension version, and distributed contract mismatches
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

The engine-level suite covers scan callback discovery, opaque fat-task payload
serialization, explicit worker bind serde, task subset application,
empty assignments, schema validation, file and callback write translation,
write callback execution, binary fragment and artifact envelopes, runtime
operation/task-attempt identity, pre-write validation, selected-result propagation,
row-count validation, transaction ordering, abort behavior, output cleanup,
catalog-commit fencing, and final marker publication.

Each concrete extension adds its own adapter tests for catalog pinning, task
semantics, object storage, commit behavior, and failure boundaries.

# Distributed DuckDB extension architecture

## Scope

This document defines the explicit contracts that let a statically linked
DuckDB extension participate in Vane's Ray distributed execution. The engine
owns scheduling and transport; each extension continues to own its bind state,
scan semantics, file production, and catalog transaction.

An extension participates only by implementing the corresponding provider
interface. Vane does not infer extension behavior, adapt older provider types,
or execute unsupported extension operators locally.

## Design rules

- Extensions used by Ray are pinned, reviewed, and statically linked into the
  Vane release artifact.
- The coordinator resolves catalog state and selects immutable work before
  worker execution starts.
- Workers receive portable task data and rebuild extension state through the
  extension's normal bind path.
- The scheduler treats extension task payloads as opaque.
- Workers may produce immutable files, but only the coordinator may mutate an
  extension catalog transaction.
- Invalid or incomplete provider contracts fail before distributed side
  effects begin.

## Build and loading

The connection snapshot records loaded static extensions. A worker validates
every recorded name against its compiled extension registry and invokes
DuckDB's generated static loader before rebinding a plan. Dynamically installed
extension binaries are not accepted by distributed execution.

The snapshot also carries serializable DuckDB secrets and the declarations of
attached catalogs needed to plan a transported logical plan. Secrets are
recreated as temporary worker-local secrets; non-serializable secrets are
rejected. Attached catalogs are recreated only on an isolated planning
connection, while worker rebinds consume the immutable identifiers prepared by
the scan provider.

The secret list is authoritative for its DatabaseInstance. Transported plans
with secrets use an isolated driver planning database. A Ray worker disables
host persistent-secret loading before a newly opened snapshot database first
uses its secret manager, retains non-default bootstrap databases for the worker
lifetime, and leases one exact bootstrap-plus-secret digest through execution.
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

### Worker rebind

Extension bind data commonly contains contexts, file systems, catalog objects,
or caches and therefore cannot be copied between processes. A scan that
implements `ExtensionScanTaskProvider` serializes its table-function identity,
worker positional parameters, named parameters, and input table schema instead.

After loading the required static extension, each worker resolves the normal
DuckDB table function and invokes its bind callback again. The rebound state
must expose the provider and reproduce the coordinator's output schema.

### Extension-owned tasks

`ExtensionScanTaskProvider` may be implemented directly by table-function bind
data or by the custom `MultiFileList` owned by `MultiFileBindData`. It provides
three operations:

1. Expand the logical scan into portable tasks represented by `OpenFileInfo`.
2. Apply an assigned task subset to freshly rebound worker state.
3. Optionally normalize worker bind parameters before plan serialization.

A file-backed extension can use `OpenFileInfo::path` directly. Other extensions
can encode an opaque portable token in `path` and scalar metadata in
`ExtendedOpenFileInfo::options`.

Providers may return per-task byte and cardinality estimates for balancing.
Vane never opens a provider task path to infer those estimates, even when the
token happens to resemble a URI; task interpretation remains extension-owned.

Vane groups these tasks into existing `ScanTaskDescriptor` objects and
transports them through normal and fault-tolerant execution. Empty assignment
must remain an empty scan. Provider-owned state is never replaced with an
engine-owned file list.

## Distributed writes

`ExtensionWriteTaskProvider` is the coordinator-side contract for an extension
operator whose single child is DuckDB COPY configured to return
`WRITTEN_FILE_STATISTICS`. The extension operator stays on the coordinator;
workers execute only the COPY child. Submission preflight therefore validates
serialization of the worker-executable child, not the coordinator-only
extension root.

The write protocol is:

1. Validate the provider contract before any worker creates output.
2. Derive a stable direct-write namespace from the distributed query identity
   and COPY sink ID, then write immutable files into that namespace.
3. Select successful attempts, validate their metadata, persist their manifest,
   and remove unselected files.
4. Pass exactly the selected file paths, row counts, sizes, column statistics,
   and partition keys to the provider.
5. Let the provider register those files in the active coordinator catalog
   transaction and return the affected row count.
6. Commit through DuckDB's transaction manager.
7. Publish the committed file marker only after catalog commit succeeds.

Coordinator staging and rename-based output are rejected because the catalog
must reference the exact immutable paths produced by workers. The provider's
affected row count must match the total in worker file metadata.

Extension writes require DuckDB auto-commit mode. Vane owns that transaction
boundary; an explicit caller-managed transaction is rejected before workers
start because Vane could not safely publish a marker before the caller's later
commit. A retry with the same query identity inspects the stable namespace
before validation or worker scheduling. A valid committed marker returns the
persisted result without invoking the provider again.

A worker, validation, provider, or other failure known to precede catalog
commit rolls back the transaction and removes the operation's uncommitted
namespace. If the catalog commit call itself fails, Vane retains the files,
does not publish a committed marker, and reports an unknown outcome: a remote
catalog may have committed before its response was lost. Failure to publish the
final marker after a known successful catalog commit is also reported as an
unknown output-lifecycle outcome because the committed catalog transaction
cannot be rolled back.

A prepared manifest without a committed marker is therefore not retryable by
the generic engine. Vane retains its files and rejects automatic replay because
it cannot prove whether a remote catalog committed before its response was
lost. Extension-specific reconciliation may establish that outcome separately.

## Extension author contract

For a distributed scan provider:

- emit deterministic task payloads without process-local pointers;
- normalize moving references to immutable identifiers before serialization,
  deterministically and safely on an already-normalized input;
- make subset application idempotent and valid before global scan state starts;
- keep coordinator task enumeration free of logical query side effects;
- preserve schema, projection, filters, partitions, and delete semantics after
  worker rebind;
- provide task byte or cardinality metadata when available for balancing.

For a distributed write provider:

- expose a stable diagnostic name;
- validate all catalog and output preconditions before workers start;
- accept only the selected immutable worker files during finalization;
- register files in the active coordinator transaction without committing it;
- return the exact affected row count;
- retain immutable data files when a remote catalog commit response is
  ambiguous; never interpret every commit exception as proof of rollback;
- leave transaction commit and output-lifecycle publication to Vane.

Both providers require normal and fault-tolerant distributed tests before an
extension is enabled in release builds.

## Failure rules

- Unknown or non-static extension names in a connection snapshot are rejected.
- A worker rebind schema mismatch is a serialization error.
- A declared scan provider must be recreated by worker bind.
- A scan function that exposes only one of DuckDB's serialization callbacks is
  rejected unless it implements the scan provider contract.
- Provider-owned tasks are never merged with engine-discovered file tasks; the
  provider remains responsible for validating the opaque payloads it emits and
  receives.
- An empty task assignment produces an empty scan.
- An extension physical operator without `ExtensionWriteTaskProvider` is
  rejected before its child can run.
- Extension writes must use direct worker output and explicit two-phase
  finalization.
- Extension writes reject explicit DuckDB transactions and ambiguous catalog
  commits never trigger destructive output cleanup.

## Framework test matrix

The engine-level suite covers provider discovery, portable scan-task
serialization, worker rebind, task subset application, empty assignments,
schema validation, extension write translation, pre-write validation, selected
file propagation, row-count validation, transaction ordering, output cleanup,
and final marker publication.

Each concrete extension adds its own adapter tests for catalog pinning, task
semantics, object storage, commit behavior, and failure boundaries.

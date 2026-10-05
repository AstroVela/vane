# VaneFS feasibility and design

VaneFS is feasible as a branchable filesystem backed by SQLite, initially for
local agent workspaces and small mutable files. The recommended first delivery
is a filesystem API with durable snapshots and branch isolation. Vane query
adapters, a FUSE mount, and access from multiple machines are separate stages.

Status: the C++ core, bindings and read-only adapter implement stages 1–3.
Stage 4 includes a Linux FUSE adapter and a local native FILE/media connector.
Owner recovery, mount retention and benchmark tooling are implemented alongside
them. Remote worker access remains future work.
See [build instructions and current limits](README.md). The assessment uses Vane
`upstream/main` at `3095a34737d991097ed5f9737b89761736678050`, fetched on
2026-10-05. Later stages remain proposals and do not imply current support.

## Basis and scope

[Chronos sections 3–5](https://arxiv.org/html/2609.14889v2#S3) describe interval
visibility, metadata-only forks, retained fork bases, copy-on-write updates,
and three-way merging. Its ChronosFS implementation represents inodes,
directory entries, and 4 KiB blocks in relational tables and exposes FUSE.
These mechanisms support this proposal; the paper's benchmark results do not
establish VaneFS performance. The choices below adapt the idea to Vane and a
single SQLite transaction domain.

| Workload | Assessment | Proposed boundary |
| --- | --- | --- |
| Local agent workspaces, configuration, generated reports | Good initial fit | One database on local storage; serialized writes |
| CSV and Parquet queries over a stable workspace | Feasible first Vane integration | Explicitly pinned snapshot and local runner |
| Ordinary shell tools using a mounted directory | Implemented on Linux | FUSE adapter with the documented filesystem subset |
| Large video datasets and sustained bulk writes | Requires measurement and a different payload strategy | Keep large immutable inputs in existing storage initially |
| Ray workers on multiple machines | Requires a storage service or exported snapshots | Workers must not open a shared SQLite WAL file over NFS |
| Atomic changes across VaneFS, DuckDB tables, and vector indexes | Separate project stage | Requires store adapters, publication and recovery protocols |

SQLite permits concurrent readers but only one writer per database. WAL relies
on coordination between processes on the same host. A service can provide
remote access while keeping SQLite local, but does not remove the single-writer
limit. These are deployment constraints, not problems that branch intervals
solve. See [SQLite transactions](https://www.sqlite.org/lang_transaction.html)
and [WAL](https://www.sqlite.org/wal.html).

## Existing Vane integration points

| Repository evidence | Consequence for the design |
| --- | --- |
| [`vane/filesystem.py`](../vane/filesystem.py) provides an internal memory filesystem | Add a distinct VaneFS component; keep the internal object store's purpose intact |
| [`RegisterFilesystem`](../src/vane_py/pyconnection.cpp) accepts fsspec implementations and `vane_directory_semantics` | A local adapter can advertise real directories and reuse existing CSV/Parquet I/O |
| [`PythonFilesystem::OpenFile`](../src/vane_py/pyfilesystem.cpp) rejects nonblocking opens | Registration alone cannot enable the current FILE resolver |
| [`test_filesystem.py`](../tests/fast/test_filesystem.py) covers CSV/Parquet reads and writes and the FILE rejection | Reuse these contracts when adding the adapter |
| [FILE I/O and distributed boundaries](../external/duckdb/extension/file/README.md) govern media reads and worker context | Full FILE/media support needs its own connector work and worker validation |
| [Distributed extensions](../DISTRIBUTED_EXTENSIONS.md) require explicit worker preparation | Driver registration is insufficient evidence of Ray support |
| [`vcpkg.json`](../vcpkg.json) has no direct SQLite dependency; [`pyproject.toml`](../pyproject.toml) ships the `vane` package | Native SQLite and a new top-level Python package each need explicit build decisions |

SQLite should own VaneFS metadata and small block payloads. Vane/DuckDB should
query files through the filesystem interface. Attaching the internal database
with a SQLite scanner would expose implementation tables and would not provide
filesystem operations, branch enforcement, or cross-engine atomicity.

## Component boundaries

The dependency direction is application → branch/filesystem API → versioned
store → SQLite. fsspec and FUSE are adapters above the same API. The SQLite
database and its WAL/SHM files live on the host filesystem outside any VaneFS
mount, preventing recursive storage access.

The storage implementation is C++17 in `src/workspace.cpp`, with a public header
under `include/vane_fs/`. It has no dependency on DuckDB or Python. The pybind11
translation unit only converts arguments, releases the GIL, and calls the same
core. The optional fsspec module adapts immutable snapshot reads.

VaneFS is built separately as `vane-fs`. Its vcpkg manifest uses the repository's
pinned baseline and SQLite 3.53.2. SQLite is linked statically with private
symbols and its upstream notice accompanies the wheel. Python's system SQLite
library is not used for filesystem operations. The base Vane wheel is unchanged.

## Persistent model

Use one SQLite database per workspace so namespace updates and file contents
can commit together. Initial table responsibilities are:

| Table | Key or identity | Data and role |
| --- | --- | --- |
| `format` | Workspace UUID | Schema version, block size, interval encoding |
| `branches` | Non-reused branch ID | Display name, parent ID, fork-base ID, bounds, frontier, writer ID, mutation generation, lifecycle state |
| `snapshots` | Non-reused snapshot ID | Immutable read point and explicit retention flag; includes fork bases |
| `inode_versions` | Inode ID plus interval low bound | File kind, size, mode, timestamps, visibility and deletion metadata |
| `dirent_versions` | Parent inode, name, interval low bound | Target inode and visibility/deletion metadata |
| `block_versions` | Inode, block number, interval low bound | Payload reference and visibility/deletion metadata |
| `block_payloads` | Payload ID | Immutable 4 KiB bytes, checked by the schema |
| `pins` | Handle or query token | Snapshot retention while callers can still access it |
| `inode_ids` | Monotonic SQLite integer | Allocates inode identities across all branches |
| `owners` | Connection ID | OS lock location and file identity used to prove an owner has exited |
| `mounts` | Branch ID | Exclusive writable mount owner |
| `open_inodes` | Owner, branch, inode | Kernel lookup and open reference counts |
| `orphans` | Branch, inode | Unlinked nodes retained until their final reference closes |

All three version tables carry `low`, `high`, `writer`, and `deleted`.
Their logical keys must have disjoint visibility intervals. Initial lookup
indexes combine the logical key with interval bounds; directory enumeration
also needs a parent-inode index. Measure their actual plans with `EXPLAIN QUERY
PLAN`; a constant-size visibility predicate does not imply constant-time I/O.

Payloads are separate from version rows so splitting an inherited block's
interval does not duplicate its bytes. Payload deduplication is deferred;
semantic comparisons inspect bytes when payload IDs differ.
Allocate inode IDs across the entire workspace, including all branches.
An index on `block_versions(payload)` supports GC reference checks; without
it the correlated payload check scanned all block versions for each payload.
See the [measured baseline](benchmarks/BASELINE.md).

The coordinate format is fixed-width, 32-byte unsigned big-endian BLOBs,
with arithmetic performed in the storage layer. Enforce BLOB type and length
on columns and bound parameters. Equal-width BLOB order supports numeric range
comparison; SQLite INTEGER is signed and limited to 64 bits, and decimal text
without fixed-width encoding has the wrong ordering. See
[SQLite types and comparison](https://www.sqlite.org/datatype3.html).

This 256-bit choice needs allocator benchmarks; it is not a promise of unlimited
depth. Store the encoding version, retain a writable parent range at every
fork, and fail an exhausted allocation without modifying data. Do not silently
wrap coordinates or reuse deleted ranges. Widening or relabeling is a later,
explicit migration under exclusive workspace access.

## Branch and snapshot contracts

At read point `f`, visibility is `low <= f < high` and `deleted = false`.
Writes replace only the current active interval `[f, high)`, retaining fragments
outside it. Fork reserves the old frontier for the base, assigns a child range,
and advances the parent. These are the interval mechanics adopted from
[Chronos sections 3 and 4](https://arxiv.org/html/2609.14889v2#S4).

API responsibilities (the C++ API uses CamelCase names):

| Operation | Initial contract |
| --- | --- |
| `fork(source, name, terminal=False)` | Create an isolated writable child and retain its immutable base; copy no file records |
| `checkout(branch_id)` | Return a live session that resolves current branch state within each transaction |
| `snapshot(branch_id)` | Reserve the current read point, advance the writable frontier, and return an immutable snapshot ID |
| `open_snapshot(snapshot_id)` | Open a read-only session and hold a retention pin |
| `diff(source_snapshot, target_snapshot)` | Compare retained states and report added, modified and deleted paths |
| `preview_merge(child, parent)` | Produce conflicts and a generation-bound proposal |
| `merge(preview, resolutions)` | Validate and publish a resolved proposal atomically |
| `delete_branch(branch_id, recursive=False)` | Retire a branch; require an explicit recursive request for descendants |

The allocator reserves roughly one eighth of remaining capacity per ordinary
child and one point for a terminal child. A minimum-width clamp still leaves a
base point, child point, and writable parent point. Compare this policy with
the paper's hinted allocator before freezing the format or capacity targets.

A branch ID identifies mutable state. A cached frontier or mutation counter
alone is not a durable snapshot: an ordinary write can replace data at the same
frontier. The `snapshot` operation reserves a point that later writes
cannot overwrite and rotates the branch writer ID. Vane queries use this
retained snapshot for their entire execution. Future distributed adapters must
preserve that identity across retries and workers.

Keep IDs separate from branch names. Deleting and recreating a name must never
make an old session or FILE reference access a new branch. Initial URL proposal:
`vanefs://<workspace-id>/snapshots/<snapshot-id>/<path>`. Resolve workspace
location through adapter configuration; do not embed machine paths or secrets
in persistent FILE values.

The implemented native connector registers that mapping in each Vane
`ClientContext`. A stateless `vanefs_native` router handles the `vanefs://`
scheme; it resolves no database without a mapping on the invoking connection.
The optional package exports `vane_fs.snapshot_reader.v1`, a versioned capsule
of C function pointers, opaque handles and fixed-layout metadata. It owns all
SQLite state and native allocations. The base Vane adapter retains the capsule
and consumes no private engine or STL objects from the provider. This keeps
SQLite optional and preserves the private native-module symbol boundary.

Each query pins a snapshot on its first access until `QueryEnd`, including
errors and cancellation. Individual file handles independently retain their inode
snapshot, provider and stateless filesystem implementation. The provider shares
one live immutable session per snapshot across these references, releasing its
SQLite pin with the final handle. This avoids a durable write pair per file
while removing expired session-cache entries immediately.
Closing the caller's workspace or unregistering the routing filesystem cannot
invalidate an open reader. Positional reads preserve the implicit cursor;
short exact reads fail. Nonblocking virtual opens cannot encounter host FIFOs.
Reads check cancellation between bounded chunks, with SQLite lock waits still
limited by the configured busy timeout. The connector accepts explicit paths
on `local-fast`; it does not serialize SQLite connections or callbacks to Ray.

## Transactions and durability

Every mutating filesystem operation starts a `BEGIN IMMEDIATE` transaction,
then reads branch metadata and performs all affected record changes before
commit. Fork, snapshot creation, deletion, and merge use the same writer
serialization. A reader loads metadata and data within one read transaction.
This deliberately avoids caching a writable range across operations, so the
first implementation does not need a distributed epoch barrier. SQLite's
[transaction rules](https://www.sqlite.org/lang_transaction.html) provide the
underlying serialization; the application must still enforce this ordering.

The implementation configures WAL, `synchronous=FULL`, foreign-key checks and a
bounded busy timeout on each connection, and verifies that WAL was enabled.
Lock contention beyond the timeout raises `Busy`; other SQLite failures raise
`Storage`. There is no additional application retry or cancellation mechanism.
Callers may retry complete operations, never an arbitrary suffix of a failed
split. Keep SQL read transactions short; application snapshots and pins provide
longer retention without holding a SQLite transaction for an entire query.
The [synchronous documentation](https://www.sqlite.org/pragma.html#pragma_synchronous)
describes the durability tradeoff; weakening it must be an explicit option.

The native build and runtime require SQLite 3.51.3 or later. The pinned build
uses 3.53.2, independent of the Python runtime's SQLite version. See the
[official WAL-reset notice](https://www.sqlite.org/wal.html#walresetbug).

Backups must capture a consistent SQLite state through its backup facilities
or a cleanly closed database. Track WAL growth and checkpoint latency. Runtime
state belongs outside the checkout. This design does not enable arbitrary
write access to internal tables through Vane SQL.

## Filesystem behavior

The first API supports directories, regular files, `stat`, `listdir`, creation,
offset reads and writes, truncation, rename, unlink, and empty-directory removal.
Paths are rooted in the virtual namespace. Reject NULs, invalid entry names,
and attempts to walk above its root. Use case-sensitive names initially.

Use 4 KiB logical blocks for the prototype. A partial write reads and replaces
only touched blocks and commits the resulting inode size in the same
transaction. Holes read as zero. Shrinking a file must remove visible trailing
blocks and clear the unused tail of its final block so re-extension cannot
reveal old bytes. Writing data and unlinking or renaming its entry must never
leave a partially updated namespace after a crash.

Renames update directory entries atomically while preserving inode identity.
Enforce directory-cycle checks, file/directory replacement rules and
empty-directory requirements. Handle identity is a branch/snapshot plus inode,
not a path that could resolve to a different file after rename.

The API and mount implement a documented subset of POSIX. Hard links, symlinks,
advisory locks and writable mmap remain unsupported. Open-unlink lifetime,
permissions, fsync and kernel cache behavior have the following contracts.
The Linux FUSE adapter uses the low-level inode interface and durable inode
references under an exclusive branch lease. Paths can change or disappear
without retargeting open descriptors. The final reference release deletes an
orphan's live versions. Branch topology/publication operations wait until the
mount is released, so snapshots and merges cannot capture transient orphans.
Existing snapshots and other branches remain usable. The optional fsspec
adapter's handles refer to immutable, pinned snapshots.

Mutable mounts disable kernel writeback and data caching, use zero attribute
and entry timeouts, and commit every mutation with SQLite `synchronous=FULL`.
Read-only snapshot mounts may cache immutable bytes. Each mount currently uses
one dispatch loop and one SQLite connection. The supported POSIX subset and
metadata limitations are listed in [the usage guide](README.md#native-commands-and-linux-mounts).

Directory import is a future convenience API. A later lazy importer may
refer to an immutable host snapshot or immutable object version, but a pathname
plus size/mtime is not sufficient to preserve content. Source mutation, symlinks
escaping the import root, unavailable originals and truncated files need defined
errors. Large external payloads also require a separate recovery protocol;
writing a SQLite row and an external file is not one atomic transaction.

## Merge and reclamation

Start with direct child-to-parent merges of a leaf branch. Its stored fork base
is unambiguous. Compare source, target and base in one SQLite read transaction,
then resolve conflicts before taking the write lock. The preview records
both branch identities and mutation generations; publication revalidates them
under the lock and aborts a stale proposal. Successful merge seals the child.
Continuing work requires a fresh fork. General sibling merges, repeated mutable
merges and merge ancestry are deferred until their history model is specified.

For the first release, report a conservative conflict when both sides change
the same file differently. Block-level storage does not require block-level
automatic conflict resolution. In particular, disjoint block edits can still
conflict through file size, truncate, delete, rename, or application semantics.
The current diff reports paths. Text and byte-range presentation remain future
work; neither would prove that a merge is semantically safe.

Validate the complete proposed namespace before committing: every visible
entry targets a visible inode, entry names are unique per directory, directories
have no cycles, and file sizes agree with block visibility. Generic row-level
comparison does not establish these filesystem invariants. With all records
inside one database, a resolved merge commits in one SQLite transaction.

Branch deletion first retires metadata and prevents new sessions. Retained
snapshots, fork bases and live handles protect their data. Garbage collection
checks visibility from all remaining retention roots before reclaiming version
rows, then removes payloads with no remaining references. The initial collector
scans version tables; writer-based candidate acceleration is deferred. A writer ID identifies
provenance, not exclusive ownership; deleting every row written by a deleted
branch could destroy data inherited by surviving states. The root branch is
protected, and retired interval space is initially not recycled.

Format 2 adds OS-locked connection owners. Recovery holds the SQLite writer
lock, acquires an owner's separate file lock and verifies its device/inode
identity before removing its pins, mount lease and inode references. Active
owners, unavailable locks and unverifiable format-1 pins are retained. Locks
remain held until retirement commits. No timeout declares a process dead.
Mount startup invokes recovery; the API and CLI also expose it explicitly.
Persistent IDs use fresh system entropy because a process fork copies SQLite's
userspace PRNG state. SQLite connections inherited across fork are rejected.

If future publication spans multiple databases or object stores, specify
durable staging, fencing, acknowledgements, publication, retry identity and
recovery as a separate protocol. The present single-database guarantee does
not extend to an external table merely because both use the same branch name.

## Delivery and acceptance

| Stage | Deliverable | Acceptance boundary |
| --- | --- | --- |
| 1 | SQLite store, filesystem API, forks, snapshots and retention | Reopen durability, parent/child isolation, no file-row copying on fork, atomic rename and truncate |
| 2 | Diff, conservative direct-parent merge, retirement and GC | Conflicts and stale previews cannot partially publish; retained states survive deletion and GC |
| 3 | Optional fsspec adapter | Explicit local-runner CSV/Parquet reads from pinned snapshots; ordinary adapter I/O tests and directory semantics |
| 4 | Linux FUSE and local native FILE connector | Mounted inode lifetime; FILE ranges, connection isolation, query/reader retention, cancellation, image decode and video frames |
| 5 | Remote service and Ray support | Stable snapshot identity, per-worker setup, leases, retry behavior and multi-node correctness |

Stage 3 provides read-only Vane query integration. Native write APIs
target live branches, but SQL COPY publication is a separate acceptance
case; do not imply that several filesystem calls form one transaction. Expose
new code through an explicit package import before considering a top-level
`vane` API.

Correctness checks should cover randomized filesystem traces against a simple
snapshot model; nested and sibling forks; partial-block writes; sparse files;
shrink/re-extend; competing writes and forks on separate connections/processes;
stale sessions after branch deletion; failed allocations; merge rollback;
GC with retained fork bases; and restart after interruption. Kill/reopen tests
validate process-crash recovery, not every possible power-loss failure mode.

Benchmark file count and byte volume separately. Measure fork latency and row
counts, first write after fork, repeated same-block updates, directory lookup,
sequential reads, writer contention, physical version growth, WAL size and GC
cost. A 1 GiB file has 262,144 logical 4 KiB blocks, which is a reason to test
large-file costs before claiming suitability for Vane's video workloads.
Publish observed p50/p95 results rather than adopting paper speedups as targets.

Run affected tests through `scripts/run_installed_pytest.sh`, then the required
`scripts/run_release_tests.sh` gate. Use
`scripts/format root --changed`; native changes also require the documented
non-editable incremental build. Benchmark cleanup must retain measurements,
configurations, identities, counts and hashes while removing unneeded generated
data after owned processes stop, with removed paths recorded in a manifest.

The first implementation milestone is complete when an independently reopened
workspace can fork, edit and snapshot files without changing sibling or parent
contents, and injected failures leave both metadata and file bytes consistent.
Full POSIX, multimedia and distributed claims follow only after their own
acceptance stages.

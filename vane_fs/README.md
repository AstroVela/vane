# VaneFS

VaneFS is a local, branchable filesystem implemented in **C++17 with SQLite**.
The native library owns file and directory operations, 4 KiB block copy-on-write,
256-bit visibility intervals, snapshots, three-way merge and garbage collection.
Python provides pybind11 bindings and an optional read-only fsspec adapter.

This is a separately installed experimental component. It supports local
workspaces, Linux FUSE mounts and native Vane FILE, CSV/Parquet and media reads
over immutable snapshots. Remote Ray access remains a later stage in the
[architecture and roadmap](DESIGN.md).

## Build and install

The tested build is Linux x86-64. Use CMake 3.29+, Ninja, a C++17 compiler, and
the vcpkg baseline pinned in `vcpkg.json`. The native library requires SQLite
3.51.3 or later; this manifest resolves SQLite 3.53.2. It uses its own SQLite
library, independently of the version linked into Python's `sqlite3` module.

From this directory, point `VCPKG_ROOT` at a bootstrapped vcpkg checkout at
`44819aa2a6c10e56065e2b0330e7d6c89d1d2574`, then install the dependency:

```bash
"$VCPKG_ROOT/vcpkg" install \
  --x-manifest-root="$PWD" --x-install-root="$PWD/vcpkg_installed" \
  --triplet=x64-linux-release --host-triplet=x64-linux-release \
  --clean-buildtrees-after-build --clean-packages-after-build
export CMAKE_PREFIX_PATH="$PWD/vcpkg_installed/x64-linux-release"
```

Build the C++ library and native tests without Python:

```bash
cmake -S . -B build/core -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVANE_FS_BUILD_PYTHON=OFF -DVANE_FS_BUILD_TESTS=ON \
  -DVANE_FS_BUILD_TOOLS=ON
cmake --build build/core -j 2
ctest --test-dir build/core --output-on-failure
```

Install the Python binding in an activated virtual environment:

```bash
uv pip install 'scikit-build-core>=0.11.4' 'pybind11>=3.0'
export SKBUILD_BUILD_DIR="$PWD/build/python-release"
export SKBUILD_CMAKE_BUILD_TYPE=Release
uv pip install '.[fsspec,test]' --no-build-isolation
```

Use a non-editable install after changing C++ or Python. The build links SQLite
statically and includes its notice in the wheel. It does not rebuild the Vane
engine. Other platforms need the corresponding static vcpkg triplet and their
own validation before release.

## Native commands and Linux mounts

`build/core/vane-fs` provides `init`, `branches`, `fork SOURCE NAME`,
`snapshot [BRANCH]`, `recover`, and `gc`. These commands do not require Python.
For example, from this directory:

```bash
build/core/vane-fs /tmp/workspace.sqlite init
build/core/vane-fs /tmp/workspace.sqlite fork main candidate
```

To build the optional mount executable, install the platform's libfuse3
development package and `fusermount3` (Ubuntu: `libfuse3-dev` and `fuse3`), then:

```bash
cmake -S . -B build/core -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVANE_FS_BUILD_PYTHON=OFF -DVANE_FS_BUILD_TESTS=ON \
  -DVANE_FS_BUILD_TOOLS=ON -DVANE_FS_BUILD_FUSE=ON
cmake --build build/core -j 2
mkdir -p /tmp/vane-fs-live
build/core/vane-fs-mount /tmp/workspace.sqlite --branch candidate /tmp/vane-fs-live
```

The mount runs in the foreground. In another terminal, use ordinary commands
such as `ls`, `cat`, `cp`, or an editor in `/tmp/vane-fs-live`. Unmount with
`fusermount3 -u /tmp/vane-fs-live`; the process exits and releases its branch.
An empty mountpoint and a database outside that mountpoint are required.
Append `--debug` for libfuse request diagnostics.

After unmounting, retain and mount a read-only snapshot:

```bash
snapshot_id="$(build/core/vane-fs /tmp/workspace.sqlite snapshot candidate)"
mkdir -p /tmp/vane-fs-snapshot
build/core/vane-fs-mount /tmp/workspace.sqlite --snapshot "$snapshot_id" /tmp/vane-fs-snapshot
```

The C++ adapter uses libfuse's low-level inode interface. Kernel lookup and
open references retain unlinked inodes; rename, replacement and unlink cannot
retarget an existing file descriptor. The final reference release reclaims an
orphan. Appends select the current EOF inside the write transaction. Each file
write and namespace/attribute mutation commits with `synchronous=FULL` before
replying; kernel writeback is disabled, and `fsync` has no deferred application
data to flush.

The first inode reference persists a pin; intermediate opens and releases update
exact counts in a connection-private in-memory SQLite table. The final release
removes the pin and reclaims an orphan atomically. Temporary counts participate
in transaction rollback, while durable owner pins protect live handles from GC
and allow recovery after a mount crash. Reference operations retain writer-lock
serialization but avoid repeated WAL writes and syncs for an already-pinned inode.

A writable mount exclusively leases its branch. Other connections may read
it, but writes, fork, snapshot, merge and deletion involving that branch require
unmounting first. Other branches and existing snapshots remain usable. Positive
directory entries and attributes have a 60-second kernel cache timeout. All live
mutations go through this mount, so Linux invalidates the affected cached
metadata as part of each operation. Writes and O_TRUNC also send an explicit
attribute invalidation to preserve the atime/mtime alias, including for statx
queries that request only atime. Missing entries are not cached. Live mounts
still use direct I/O; read-only snapshots can retain immutable file data in the
kernel cache. Requests run through one native dispatch loop and one SQLite
connection. Allowing future out-of-band writers would require a notification
protocol before retaining these cache timeouts.

Each open directory handle retains the sorted entry list captured at open,
including across pagination and rewind. Removing, renaming or adding entries
cannot skip or duplicate unrelated entries in that stream. Open a new directory
handle to see the latest listing. Listing memory is proportional to the entries
in each open directory and is released when the handle closes or the mount exits.

This is a filesystem subset: regular files, directories, modes through `0777`,
mtime, seek/read/write, append, truncate, rename, unlink and directory removal.
All entries belong to the mounting user/group; atime and ctime report mtime.
Hard links, symlinks, ownership changes, special mode bits, xattrs, advisory
locking, special files and writable mmap are not supported. The executable
dynamically links the system libfuse3; it is not included in the Python wheel.

## Recovery and format compatibility

On POSIX systems each connection holds an OS lock in
`DATABASE.vane_fs-locks/`. `recover` takes a SQLite writer transaction and only
retires another owner after acquiring its lock and verifying the lock file's
device/inode identity. It releases that owner's snapshot pins, mount lease and
inode references, then reclaims unlinked orphans. A stopped or slow process
continues to hold its lock and is never expired by a timer.

```bash
build/core/vane-fs /tmp/workspace.sqlite recover
build/core/vane-fs /tmp/workspace.sqlite gc
```

The Python equivalent is `workspace.recover_owners()`. Mount startup performs
this recovery automatically. After a mount process is killed, unmount its
disconnected kernel mount with `fusermount3 -u` before mounting again.

Opening format 1 performs a transactional upgrade to format 2, which adds owner,
mount, open-inode and orphan records. Older binaries reject the upgraded format.
Legacy pins without owner information, missing/replaced lock files and platforms
without supported OS locks remain conservatively retained. Automatic recovery
does not infer that an unknown owner is dead. Keep the database and lock
directory in place while connections are active; close all connections and
unmount before moving or cloning a workspace. Reopen connections after `fork()`;
inherited SQLite connections are rejected.

## C++ API

Include `vane_fs/workspace.hpp` and link the `vane_fs` CMake target. The library
does not depend on DuckDB or Python.

```cpp
#include "vane_fs/workspace.hpp"

vane_fs::Workspace workspace("/tmp/workspace.sqlite");
auto main = workspace.Checkout();
main->WriteFile("/config.txt", "base");

auto child = workspace.Fork("main", "candidate");
auto candidate = workspace.Checkout(child.id);
candidate->WriteFile("/config.txt", "candidate");
// main->Read("/config.txt") still returns "base".

auto snapshot_id = workspace.Snapshot(child.id);
auto snapshot = workspace.OpenSnapshot(snapshot_id);
auto bytes = snapshot->Read("/config.txt");

auto preview = workspace.PreviewMerge(child.id, "main");
workspace.Merge(preview); // Fails atomically on conflicts or a stale preview.
```

`Session` also exposes `MakeDirectory`, `Stat`, `ListDirectory`, `Write` with an
offset, `Truncate`, `Rename`, `Unlink` and `RemoveDirectory`. Every operation is
one transaction. Independent connections and processes can share a database on
the same local filesystem; SQLite serializes writers. One connection serializes
its own calls with a mutex. Errors carry a native `ErrorCode`.

`WriteFile` replaces the complete file or creates it in an existing directory.
`Write` edits an existing file and supports sparse extension. Paths are rooted
in the virtual workspace, use case-sensitive names, and reject every `..`
component. Parent directories must exist. Modes are stored metadata; this API
does not enforce an operating-system permission model.

## Python bindings

The corresponding operations execute in C++ and release the GIL during database
work. File contents use `bytes`.

```python
from vane_fs import Workspace

with Workspace("/tmp/example.sqlite") as workspace:
    main = workspace.checkout()
    main.write_file("/data.csv", b"id,value\n1,original\n")
    child = workspace.fork("main", "candidate")
    candidate = workspace.checkout(child.id)
    candidate.write_file("/data.csv", b"id,value\n1,candidate\n")

    frozen = workspace.snapshot(child.id)
    with workspace.open_snapshot(frozen) as snapshot:
        assert snapshot.read("/data.csv") == b"id,value\n1,candidate\n"

    preview = workspace.preview_merge(child.id, "main")
    assert not preview.conflicts
    workspace.merge(preview)
    workspace.delete_branch(child.id)
    workspace.drop_snapshot(frozen)
    workspace.collect_garbage()
```

Common filesystem failures become the corresponding Python filesystem
exceptions. `ConflictError`, `StalePreviewError`, and `CapacityError` distinguish
merge and allocation failures. Recompute a stale preview before publishing.
Resolve a conflict with a path mapping such as
`workspace.merge(preview, {"/config.txt": "source"})`; the other choice is
`"target"`. Namespace inconsistencies also reject the complete merge.
This includes a selected child whose parent path now names a different inode.
Resolve the parent and affected entries together to keep a consistent tree.

## Vane queries

With a Vane build that includes the native connector, register an existing
workspace on an explicitly local connection. The optional `vane-fs` package
owns SQLite; the base Vane wheel does not link it.

```python
import os

os.environ["VANE_RUNNER"] = "local-fast"
import vane
from vane_fs import Workspace, register_workspace, snapshot_url, unregister_workspace

database = "/tmp/query.sqlite"
with Workspace(database) as workspace, vane.connect() as connection:
    workspace.checkout().write_file("/data.csv", b"id,value\n1,example\n")
    frozen = workspace.snapshot()
    identity = register_workspace(connection, database)
    url = snapshot_url(identity, frozen, "/data.csv")
    assert connection.execute("SELECT * FROM read_csv(?)", [url]).fetchall() == [(1, "example")]
    value = connection.execute("SELECT to_file(?)", [url]).fetchall()[0][0]
    with value.open(connection=connection) as reader:
        assert reader.read(2) == b"id"
    unregister_workspace(connection, identity)
    workspace.drop_snapshot(frozen)
```

Native file opens, metadata and positional reads call C++ directly through a
versioned C interface. They do not call Python/fsspec for file I/O or share
private DuckDB C++ objects between extension modules. This supports `File.open`,
`to_file`, range-limited FILE values, CSV/Parquet, `ImageFile` metadata/decoding,
and `VideoFile` metadata/frames with Vane's applicable media dependencies.

Each connection, including a cursor, must register its own mapping from
workspace ID to database. URLs contain immutable snapshot IDs and literal
UTF-8 paths, without URL decoding or embedded database filenames. Registration
opens an independent native workspace: closing the original `Workspace` does
not invalidate registered reads. Queries retain each accessed snapshot until
completion, failure or cancellation; standalone file readers retain it until
closed. A streaming result keeps its query pin until exhausted or closed;
`fetchone()` alone may leave that result active. Standalone reads do not attach
pins to unrelated active results. Unregistering prevents new opens without
invalidating open readers.
`drop_snapshot` fails while either kind of reference remains.

The native connector is read-only and currently requires `local-fast`. It
supports explicit file paths, not wildcard expansion or directory listing.
SQLite write access is required to record/release retention pins. Interrupts
are checked between 1 MiB read chunks; a SQLite lock wait is bounded by
`timeout_ms` (default 5000), not immediately interrupted. Query completion
releases pins; it does not delete explicitly retained snapshots.
If a write lock prevents pin cleanup during reader close or query completion,
the next write transaction on that database in the same process retries it.
Both `recover_owners()` and `drop_snapshot()` perform this retry, without
unregistering the workspace. A failed retry or rolled-back transaction keeps
the cleanup pending. An explicit snapshot session `close()` instead raises
the SQLite error and leaves the session available for another close attempt.

The older fsspec adapter remains available for listing, globbing and clients
that use Python file objects. Use it on a separate Vane database instance from
the native connector; its nonblocking-open limitation still prevents FILE and
media access through `connection.register_filesystem`.

```python
import os

os.environ["VANE_RUNNER"] = "local-fast"
import vane
from vane_fs import Workspace
from vane_fs.fsspec import SnapshotFileSystem

with Workspace("/tmp/query.sqlite") as workspace:
    workspace.checkout().write_file("/data.csv", b"id,value\n1,example\n")
    frozen = workspace.snapshot()
    with SnapshotFileSystem(workspace, frozen) as fs, vane.connect() as connection:
        connection.register_filesystem(fs)
        rows = connection.execute(
            "SELECT * FROM read_csv(?)", [fs.url("/data.csv")]
        ).fetchall()
        assert rows == [(1, "example")]
        connection.unregister_filesystem("vanefs")
    workspace.drop_snapshot(frozen)
```

The adapter also supports Parquet reads, seeking, metadata, listing and globbing.
Each URL identifies the workspace and immutable snapshot. Each open file pins
the snapshot separately, so closing an adapter does not invalidate existing
read handles. Explicitly closing the workspace invalidates all its sessions.

## Retention and current limits

- Forks and snapshots copy no file rows. Mutating an inherited block splits its
  visibility metadata and stores only the changed block's new bytes.
- A live branch is mutable. Create a snapshot for stable reads across multiple
  operations. Snapshot creation advances the branch's writable frontier.
- Interval space is finite and never silently recycled. Terminal branches have
  one writable point: they can be edited and merged, but cannot fork or create a
  snapshot until sealed. Exhaustion fails without changing state.
- Merge supports a leaf child into its direct parent and seals it on success.
  Same-file divergent changes require explicit resolution; mtime-only differences
  do not. Diff reports paths, not text or byte ranges.
- Explicit snapshots survive branch deletion until dropped. Fork bases remain
  until no live branch needs them. Closing a session releases its pin; closing
  a workspace releases pins owned by that connection. Recover crashed owners
  using the lock-based protocol above; unverifiable owners remain retained.
- GC scans retained states and version tables; it is not yet optimized for large
  histories. SQLite checkpointing and page reuse do not guarantee that the main
  database file immediately shrinks on disk.
- Directory import, SQL COPY writes and remote/Ray access are not
  implemented. Put the database on local storage outside managed file content.

## Verification

Run the native test command above, then, from the repository root with the
non-editable package installed:

```bash
scripts/run_installed_pytest.sh "$PWD/vane_fs/tests" --rootdir="$PWD/vane_fs"
scripts/run_release_tests.sh
```

Set `VANE_FS_MOUNT_BINARY="$PWD/vane_fs/build/core/vane-fs-mount"` to require the
real Linux mount tests; they are explicitly skipped when it is not set. With
the variable set, unavailable mount support is a test failure. The dedicated
[VaneFS CI workflow](../.github/workflows/vane-fs.yml) builds SQLite at the pinned
baseline, tests the native core, rebuilds a wheel from its source archive,
executes real mounts, checks ASan/UBSan, and verifies benchmark cleanup.
The main [CI workflow](../.github/workflows/ci.yml) additionally installs both
packages on Python 3.12 and runs native FILE/media acceptance against the built
Vane wheel. Those tests cover connection isolation, reader and query lifetimes,
streaming results, cancellation, range reads, and image/video parity.

The component root avoids pytest constructing a source-tree `vane_fs` namespace
over the installed package. The fsspec Vane integration test requires Vane and
PyArrow. Native media acceptance also requires Pillow, PyAV and NumPy; the
storage tests require only the installed component and pytest.

Coverage includes reopen, no-copy forks, snapshot retention, sparse and
cross-block writes, shrink/re-extend, rename, conflict and stale-preview rejection,
namespace validation, GC, randomized branch traces, concurrent connections and
processes, and process interruption. Native tests inject a SQLite write failure
and deterministically kill a child inside fork, merge and GC transactions to
verify rollback and recovery. Mounted tests cover open-unlink, overwritten
destinations, directory descriptors across rename, concurrent append, cache
isolation, read-only snapshots and recovery after killing the mount process.
These checks do not establish every possible power-loss
behavior or a production performance claim.

## Performance measurements

The native benchmark validates content and branch isolation while measuring
forks, first and repeated block writes, directory lookup, sequential I/O,
four-writer contention, version growth, WAL size and GC. Run from the repository
root after building the Release tools:

```bash
python vane_fs/benchmarks/run.py \
  --binary vane_fs/build/core/vane-fs-benchmark \
  --output vane_fs/build/benchmarks --repeat 3
```

The profiles separate 1,000/10,000 small files from a 64 MiB file and include
64 sibling forks plus 16 levels of nesting. `--quick --repeat 1` runs a small
validation case. Measurements include per-operation samples and p50/p95,
configuration, source and binary hashes, and physical row/page counts. The
runner checks free space, waits for every owned process to exit, records data
hashes, and removes generated databases outside measured windows. Results,
logs and `cleanup.json` remain. These are warm-cache core measurements; they do
not measure FUSE throughput or establish performance for large video workloads.

The [2026-10-05 baseline](benchmarks/BASELINE.md) records three repetitions per
profile, including the GC payload-reference index investigation and its
before/after measurements.

The [FUSE I/O optimization measurements](benchmarks/IO_OPTIMIZATION.md) compare
the prepared-statement and block-range changes against the previous core using
the same 64 MiB workloads, with unchanged FULL durability and live-mount caching.

The subsequent [metadata cache measurements](benchmarks/METADATA_OPTIMIZATION.md)
record stat, directory and Git workloads after enabling kernel metadata caching
under the exclusive mount lease, including invalidation checks and an I/O
regression comparison.

The [inode reference measurements](benchmarks/REFERENCE_OPTIMIZATION.md) record
directory and Git workloads after moving exact reference counts into
connection-private memory, while retaining durable first/last pins and FULL
file-mutation commits. They include rollback/lifetime tests and syscall counts.

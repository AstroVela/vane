# Exact-interval version update experiment

This isolated C++ candidate extends the [transaction-control cache](../random_io/README.md).
It updates a version in place only when the sole overlapping record has exactly
the current write interval's low and high bounds. All other cases retain the
existing delete, split and insert path. Production sources and the installed
package remain unchanged; this is still an experimental format-1004 component.
**This candidate has not passed performance acceptance and is not selected for
production.** It removes verified SQL work, but the primary GC-aged comparison
regresses substantially. Later diagnostics do not establish a stable cause or
erase those failed acceptance samples.

## Evidence and change

The preceding comparison found a 21.5% aged overwrite request deficit against
production BLOBs. That ratio does not repeat consistently in the new diagnostic's
plain runs, so it is not treated as a fixed external-format penalty. The new
diagnosis instead identifies shared work that can be removed without changing
transaction or persistence boundaries.

Each measured phase issues 512 aligned 4 KiB FUSE overwrites. Two requested blocks
already contain the replacement bytes, so unchanged-content reuse leaves 510
block versions to change, along with 512 inode versions. The previous code
deletes and reinserts every one of those 1,022 rows even when neither interval
boundary moves. Both BLOB and external candidates exhibit this pattern.

The existing writer transaction protects the overlap check and UPDATE. The new
statement replaces writer, deletion state and value fields, leaving the key and
interval coordinates untouched. Deleted blocks bind a NULL payload. A snapshot
or inherited version requiring interval splitting still takes the old path.
The result is the same version set as the previous delete/insert operation for
the exact-match case. SQLite's [UPDATE semantics](https://www.sqlite.org/lang_update.html)
leave unassigned columns unchanged; version tables use
[WITHOUT ROWID](https://www.sqlite.org/withoutrowid.html) primary keys.

Two balanced instrumented orders, with fresh and GC-aged workspaces, reproduce
these counts for the cached and updated candidates:

| Per 512-write phase | Cached controls | Exact-interval UPDATE |
| --- | ---: | ---: |
| Version mutation statements | 2,044 DELETE/INSERT | 1,022 UPDATE |
| Wrapped foreground statement VM steps | 224,448 | 188,676 |
| Counted C++ allocations | 35,286 | 27,620 |
| Fresh WAL write bytes | 22,046,120 | 17,596,520 |
| Aged WAL write bytes | 22,074,960 | 17,592,400 |

This removes 15.9% of the counted VM steps, 21.7% of counted allocations and
about 20% of WAL write bytes. WAL counts describe successful write syscalls,
not physical device amplification. VM counts cover wrapped `sqlite3_step` calls;
they must not be compared with the BLOB diagnostic's uncounted internal
`sqlite3_exec` steps as if both were complete totals.

The daemon runs one foreground FUSE callback loop and a separate checkpoint
worker. Counters are per thread. All measured overwrite phases observe 512
WRITE callbacks and zero GETATTR callbacks. The initial diagnosis finds little
GC-lock or admission waiting; it does not justify removing those protections.
The post-change diagnostic also retains an aged sample with about 125 ms in
pread calls versus 9 ms of thread CPU, and a separate 719 ms final fsync.
Timing probes perturb the request path, so their throughput is not used for
the adoption comparison below.

## Same-window FUSE comparison

The plain-binary comparison uses six balanced fresh-workspace orders and three
rotating GC-aged orders. Each variant starts with its own workspace. Large writes
use 1 MiB application requests; the read fixture is 64 MiB, followed by 4,096
seeded random reads and 512 seeded overwrites. Aged workspaces retain a 32 MiB
guard whose alternate blocks are removed before GC. Small-file and directory
tests issue 128 operations each. Durability is `fsync`, caches use their existing
defaults, and no owned builds or correctness tests overlap the measurements.

Medians from this window:

| Workload | Production BLOB | Cached controls | Exact-interval UPDATE |
| --- | ---: | ---: | ---: |
| Fresh 4 KiB overwrite requests, MiB/s | 16.54 | 14.20 | 19.05 |
| Fresh overwrite + final fsync, MiB/s | 14.02 | 12.49 | 14.28 |
| Fresh final fsync, ms | 23.51 | 20.58 | 35.22 |
| Fresh warm random read, MiB/s | 65.60 | 64.89 | 61.16 |
| Warm sequential read, MiB/s | 532.92 | 732.51 | 762.14 |
| Subsequent 256 MiB write + fsync, MiB/s | 93.79 | 142.08 | 142.90 |
| Aged overwrite requests, MiB/s | 14.70 | 15.61 | 8.17 |
| Aged overwrite + final fsync, MiB/s | 12.33 | 15.01 | 2.03 |
| Aged final fsync, ms | 25.21 | 5.08 | 698.63 |
| Aged random read, MiB/s | 69.11 | 64.45 | 59.99 |
| Aged 4 KiB file + fsync, files/s | 258.88 | 235.60 | 26.20 |
| Aged mkdir + directory fsync, ops/s | 1,888.24 | 1,736.08 | 1,821.45 |
| Aged rename + directory fsync, ops/s | 2,070.62 | 1,934.05 | 2,041.71 |
| Aged rmdir + directory fsync, ops/s | 2,080.88 | 2,184.59 | 2,326.16 |

Compared with cached controls, fresh request throughput improves by 34.2% and
overwrite-plus-fsync by 14.3%. The updated candidate's longer fresh final fsync
offsets part of the request gain. Phase medians describe different distributions
and must not be added to reconstruct the median whole operation.

The aged gate fails: request throughput is 47.6% below cached controls, and all
three final fsyncs take 652–799 ms. Small-file fsync throughput falls by 88.9%.
Random reads also remain below the BLOB baseline. These samples are retained;
the fresh gain is insufficient grounds for adopting the change.

Large writes remain variable. Updated 256 MiB throughput ranges from 19.79 to
144.85 MiB/s, cached from 30.03 to 147.29, and BLOB from 41.63 to 97.33. The
64 MiB medians differ sharply, but their ranges overlap: updated 25.50–178.48,
cached 13.52–157.07 and BLOB 7.83–101.96 MiB/s. This window does not establish a
stable large-write gain from UPDATE. Host CPU busy time has a 4.6% median and
14.1% maximum; iowait peaks at 15.0%. These aggregate counters include preparation
and cleanup and do not attribute waiting to another workload or a device cause.

## Investigating the aged failure

After GC, both external candidates have 679 database pages, one freelist page,
and identical payload allocation. Their block-version B-tree contents differ
slightly, but there is no large space expansion. In the earlier instrumented
slow request, the block-version SELECT accounts for about 125 ms of wall time
versus 9 ms of CPU; old external payload reads account for only about 5 ms.
That places the observed stall in SQLite page access, not in the UPDATE CPU work.

Two further diagnostic windows each run two balanced aged orders. One uses
in-process slow-syscall and checkpoint/sync timelines, plus external 10 ms thread
sampling. The other uses the original plain binaries with external sampling
only. Neither reproduces the original long overwrite fsyncs: updated final
fsyncs take about 3–9 ms in the first and 3–4 ms in the second. These separately
timed windows are not pooled with the primary comparison or treated as a pass.

The plain-binary diagnostic still captures a severe updated small-file run:
128 operations take 6.51 seconds, or 19.65 files/s. During that interval, 527
foreground samples observe syscall 74 (`fsync`) on the WAL waiting in
`jbd2_log_wait_commit`, and 26 in `folio_wait_bit_common`; the checkpoint worker
has 577 samples in a futex wait. This locates a real foreground persistence
stall even while the worker is mostly idle. It does not prove why the device or
journal is slow, that the UPDATE algorithm inherently causes the wait, or that
all earlier stalls share one cause.

The remaining question is the interaction between WAL growth/reuse, checkpoint
progress and filesystem journal work. That needs a controlled comparison before
changing checkpoint or retention policy. A separate BLOB-only application of
the SQL optimization also needs its own validation; these external-format
results do not substitute for it. The existing GC leases, publication ordering
and FULL barriers remain intact throughout this experiment.

## Validation

The new candidate passes 144 related Python tests, with one explicit skip for
the unsupported production-v1 migration. All eleven native tests pass in Release
and ASan/UBSan. The existing publication, checkpoint, fork, GC and recovery tests
remain included, including 33 deterministic staging crash/loss cases. These
software fault tests do not certify physical power-loss behavior.

The new regression fails against the preceding cached-control library and passes
against the updated one. It checks same-interval row reuse, snapshot splitting,
parent/child isolation, payload tombstones and revival, and reused directory
names receiving a new inode. An injected inode UPDATE failure after the block
UPDATE verifies rollback of bytes, payload allocation and branch generation;
retry, synchronization and reopening preserve both branches and the snapshot.
All forty prepared source files match the tested component. Only related tests
are run.

## Reproduce

```bash
.venv/bin/python vane_fs/benchmarks/staged_payload/aged_write/prepare.py /tmp/vane-version-update
cmake -S /tmp/vane-version-update/updated -B /tmp/vane-version-update/updated/build \
  -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/vane_fs/vcpkg_installed/x64-linux-release" \
  -DVANE_FS_BUILD_PYTHON=OFF -DVANE_FS_BUILD_TESTS=ON
cmake --build /tmp/vane-version-update/updated/build --parallel 2
ctest --test-dir /tmp/vane-version-update/updated/build --output-on-failure
```

The preparer inherits the preceding frozen-source checks and applies
`update.patch` to a separate copy. Component wheels use persistent build
directories and isolated non-editable installations. No schema migration,
remote filesystem adapter or online-backup protocol is added.

## Retained evidence

[results.json](results.json) retains every scalar FUSE result, request-latency
summaries, diagnostic counters, validation and host summaries. The
[artifact manifest](artifact-manifest.json) identifies complete latency vectors,
per-thread traces, logs, frozen sources, runners, binaries and cleanup manifests
retained locally. All slow samples remain included. Owned workspaces are removed
after their processes and mounts stop, outside measured work; other workloads
and the production installation are left intact.

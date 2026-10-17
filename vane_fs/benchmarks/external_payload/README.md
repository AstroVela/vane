# Hybrid external-payload experiment: 2026-10-08

This directory contains an isolated C++ implementation of a local external
payload store. Production source, the installed production binary and its
SQLite format remain unchanged. The experiment evaluates data volume,
synchronization, reads, small writes, snapshots, reclamation and recovery
together before considering adoption.

Do not adopt either candidate as the default backend. The hybrid store reduces
software write volume and improves sequential reads, but neither candidate
beats the baseline 256 MiB FUSE write median. Checkpoint coalescing reduces
database synchronization yet introduces significant aged metadata regressions.
The production format and checkpoint policy are retained.

## Results and decision

The follow-up fresh FUSE table contains MiB/s, median (complete range), from
six orders. Comparisons below are within that window.

| Workload | Baseline | External | External + coalescing |
| --- | ---: | ---: | ---: |
| 64 MiB write + fsync + close | 15.14 (10.15–98.03) | 29.71 (9.08–88.74) | 55.63 (14.01–92.78) |
| 256 MiB write + fsync + close | 93.62 (33.82–95.90) | 83.54 (29.90–84.81) | 89.80 (86.71–94.69) |
| Warm 64 MiB sequential read | 516.84 (495.53–562.37) | 723.59 (684.69–748.35) | 750.69 (715.82–762.23) |
| Warm 4 KiB random read | 67.87 (63.47–70.57) | 63.73 (61.61–65.18) | 63.48 (61.98–66.06) |
| 4 KiB random overwrite + fsync | 11.42 (9.22–14.42) | 13.11 (9.44–14.83) | 9.33 (1.69–11.10) |

External payloads improve the sequential-read median by 40.0%; adding
checkpoint coalescing raises that to 45.2%. The 256 MiB write medians are
10.8% and 4.1% below baseline, respectively. Small-write and random-read
tradeoffs remain. The severe 64 MiB stalls in all variants make that row
unsuitable for claiming a stable large-write improvement. Every sample is
included; neither the medians nor host counters explain individual stalls.

The first eight-round comparison independently records baseline/external
256 MiB write medians of 93.84/85.71 MiB/s and sequential-read medians of
526.64/806.34 MiB/s. Its aged small-file medians vary from 252.22 to 139.25
files/s, whereas the follow-up places all variants near 262–267 files/s.
These are distinct windows and are not pooled or selectively filtered.
The full initial data is in [initial-results.json](initial-results.json).

After the retained guard, alternating block deletion and GC, medians from
three fresh aged mounts per variant are:

| Workload | Baseline | External | External + coalescing |
| --- | ---: | ---: | ---: |
| Warm sequential read, MiB/s | 551.73 | 708.95 | 756.32 |
| Warm 4 KiB random read, MiB/s | 68.52 | 61.95 | 62.56 |
| 4 KiB random overwrite, MiB/s | 15.00 | 12.03 | 11.44 |
| 4 KiB file + fsync + close, files/s | 267.25 | 263.57 | 261.79 |
| mkdir, operations/s | 2093.72 | 2049.21 | 892.59 |
| rename, operations/s | 2434.94 | 2313.95 | 1471.71 |
| rmdir, operations/s | 2306.94 | 2271.70 | 1564.71 |

Coalescing loses 57.4% of aged mkdir throughput and 39.6%/32.2% of aged
rename/rmdir throughput. This test does not isolate the precise reason for
those regressions; delaying checkpoints can change which connection does
later maintenance. The smaller checkpoint count is not an adoption criterion.

### Why fewer bytes do not produce a write win

Native counters below include setup, writes, Sync, content validation and
close. Values are medians across six orders. They count SQLite VFS and
payload-file operations, not device traffic or media write amplification.

| Native 256 MiB metric | Baseline | External | External + coalescing |
| --- | ---: | ---: | ---: |
| WAL writes, MiB | 338.64 | 49.13 | 49.13 |
| Database writes, MiB | 307.93 | 36.19 | 19.52 |
| Payload-file writes, MiB | 0.00 | 256.00 | 256.00 |
| Total software writes / application bytes | 2.53× | 1.33× | 1.27× |
| Database sync calls | 7 | 168.5 | 2 |
| WAL sync calls | 25 | 255 | 8 |
| Payload sync calls | 0 | 257 | 257 |

The 257 payload syncs comprise initialization plus 256 data-publication
barriers. They remain after checkpoint coalescing. Its native write + Sync
median is 6.75 s versus 2.57 s for baseline; its full range is 2.57–8.69 s.
The median payload-sync time is 5.66 s, locating the additional waiting inside
data synchronization in this diagnostic. This is not evidence that fewer
checkpoints necessarily cause storage stalls, and overlapping worker timings
must not be added to foreground wall time. The uninstrumented FUSE write
medians are reported separately above.

A useful next design target is reducing data-publication barriers through a
recoverable staging/batch-publication protocol in fsync mode. Simply removing
the pre-COMMIT data sync would break the current recovery argument. That
requires an explicit visibility and recovery design, rather than a smaller
SQLite insert or a relaxed synchronization flag.

### Random reads, space and GC

For 4,096 native random reads, fresh software-read volume falls from 38.21
MiB to 20.04 MiB, but median time increases from 0.0995 s to 0.1137/0.1190 s.
After aging, volume is 22.87/15.94/15.98 MiB and time is
0.0922/0.1075/0.1071 s. Less transferred data does not remove the extra
payload-file/locking overhead; the counters do not isolate each part of that
cost. Full samples and phase scopes remain in the result files.

After retaining only 128 KiB of an 8 MiB external file and dropping its
snapshot, the sidecar has 136 KiB physically allocated in this fixture,
including its header and filesystem bookkeeping. Its logical length remains
about 7.76 MiB because the last live slice determines the tail. After unlink
and GC it is a 4 KiB logical file with 8 KiB allocated on this ext4 filesystem.
The SQL logical-payload count is exactly 128 KiB before final unlink and zero
afterward. No copy of the full 8 MiB extent remains pinned by a single slice.

Median GC times for dropping the small-overwrite snapshot are
10.93/11.07/11.05 ms; dropping the sparse snapshot costs
25.32/30.09/23.77 ms. This workload does not show the much larger repacking
cost of the earlier SQLite extent experiment, but physical GC still requires
exclusive native-reader coordination and the complete reference scan.

### Host variability and validation

The host is an Intel Xeon E5-2686 v4 with 36 logical CPUs, about 62.7 GiB RAM,
and ext4 on a Fanxiang S103Pro SSD. Builds use GCC 13.3.0, SQLite 3.53.2 and
libfuse 3.14.0. The follow-up 30-second baseline records 95.74% CPU idle and
0.009% I/O wait; measurement/validation/cleanup records 92.22% idle and 1.74%
I/O wait. These host-wide averages do not rule out individual I/O stalls.

Both formal runs pass all content and SQLite checks: 204 FUSE full hashes,
208,896 FUSE random slices, 2,176 small-file content checks, 77 native full
content checks including aged guards, 139,264 native random slices, 68 space
content checks and 230 SQLite quick checks. All owned measurement processes
and mounts stop, and every generated workspace is removed with a retained
cleanup manifest. No samples are removed because of host load or latency.

## Storage and commit protocol

The `external` and `coalesced` variants preserve independent 4 KiB logical
blocks and the existing version, snapshot and branch algorithms. A full batch
of 64 nonzero blocks in a vacant write range is concatenated into a 256 KiB
append to `<canonical-database-path>.payload`. Its SQLite payload rows contain
8-byte, big-endian offsets. Smaller batches and occupied-range updates retain
4 KiB inline BLOBs. This is a batch eligibility rule, not a file-size threshold:
small calls appending to a large file still use inline payloads.

Reads join visible block versions with their payload rows, combine adjacent
logical and physical external ranges into one `pread`, and copy inline data
into the same result. Missing blocks remain sparse zeroes. Invalid offsets,
missing rows, malformed payload sizes and short external reads fail explicitly.
Published external bytes are immutable.

Every external append occurs under the existing SQLite IMMEDIATE transaction.
Before its metadata COMMIT, the connection calls `fdatasync` on the dirty
payload file, including in fsync/NORMAL mode. A failed append or synchronization
prevents publication; abandoned bytes are garbage. Partial failed appends are
rounded up to a block boundary on retry. This preserves the data-before-reference
ordering even when an otherwise unsynchronized WAL commit reaches storage.

A 4 KiB sidecar header contains the workspace UUID. New headers are synchronized
before the first SQL commit, and the containing directory is synchronized too.
[Linux fsync documentation](https://man7.org/linux/man-pages/man2/fsync.2.html)
distinguishes file synchronization from persistence of the directory entry.
Existing sidecars must match the workspace UUID; missing ones are not recreated.
Symlink sidecars and nonregular files are rejected.

Each native transaction holds a shared `flock` on the connection's payload
descriptor. GC holds an exclusive lock across a FULL metadata deletion commit
and the physical sweep, preventing reclamation underneath native readers.
It punches unreferenced interior ranges, truncates the unused tail and
synchronizes the result. [Linux hole punching](https://man7.org/linux/man-pages/man2/fallocate.2.html)
reclaims whole filesystem blocks without changing the retained file length.
A physical reclamation failure can occur after the SQL deletion has committed;
retrying GC completes reclamation. Live references are validated before any
hole is punched.

Interior holes are not reused by the logical append allocator. Truncating a
dead tail permits later append reuse; the underlying filesystem may also
reuse freed physical blocks. This prototype does not compact live payloads.
Raw SQLite readers can inspect metadata but do not participate in native
payload-read protection.

## Checkpoint follow-up

The original `external` candidate retains the production checkpoint worker.
Its per-write data barriers expose a costly interaction: after total WAL size
crosses 16 MiB, each small metadata commit can request another checkpoint even
when almost all previous frames have already been copied.

The `coalesced` candidate first checks the current log and copied frame counts
with [SQLite's NOOP checkpoint](https://sqlite.org/c3ref/wal_checkpoint_v2.html).
A new checkpoint waits for 16 MiB of uncopied frames. Once a real checkpoint
starts, unfinished work and errors retain the original retry behavior. Write
admission still checks total WAL size at 64 MiB, including the fully backfilled
but reader-pinned case; it retains the same restart, timeout and error handling.
Explicit FULL barriers and external data synchronization are unchanged.

The new deterministic checkpoint test holds a reader on an already backfilled
WAL generation, adds small commits, then crosses the new-work budget. It checks
that small additions do not restart maintenance, that reader release permits
the pending checkpoint to finish without another write, and that subsequent
admission/reuse retains the original WAL bound. The original worker fails the
small-addition assertion; the coalesced worker passes.

## Measurement design

The first comparison has eight alternating fresh FUSE rounds of `baseline`
and `external`, plus four alternating rounds of native write diagnostics,
aged FUSE, fresh/aged native reads and space/GC checks. Its complete results
are retained, including the negative write result.

The follow-up compares `baseline`, `external` and `coalesced` in one measurement
window. Three cyclic orders place every variant in every position; reversing
these produces six orders for fresh FUSE and native writes. Aged FUSE, native
reads and GC use the first three orders. Baseline and original external
binaries are unchanged between experiments. Every sample uses a new workspace.

FUSE uses fsync mode, live direct I/O and disabled kernel writeback. Fresh mounts
run 64/256 MiB writes, warm 64 MiB reads, 4,096 seeded aligned 4 KiB reads, then
512 distinct 4 KiB overwrites. Sequential-write timers include final fsync and
close; overwrite timers include final fsync and exclude close. Complete hashes
and random slices are checked outside the corresponding timed workloads.

Independent aged mounts first write a 32 MiB guard in 1 MiB calls, zero every
other 4 KiB block, and collect garbage. Large initial calls exercise external
allocation and physical hole punching. The surviving guard is verified before
and after the measured work. These mounts use a new 64 MiB read/overwrite file
and measure 128 small files with write/fsync/close per file, plus 128 operations
of each mkdir/rename/rmdir with a final directory fsync per group. This is one
deterministic allocation history, not a steady-state model.

The native 256 MiB driver issues 256 pairs of 1,048,528 and 48 byte writes,
matching the observed FUSE split, then Sync, validation and close. Separate
read diagnostics use warm 64 MiB fresh and 16 MiB aged fixtures. SQLite VFS
and payload-class counters record software I/O, not physical SSD traffic.
Concurrent worker and foreground times overlap and must not be added as wall
time. Payload counters include data-file calls, not directory synchronization.

The space workload retains an 8 MiB snapshot during 128 small overwrites, then
drops it and collects garbage. A separate dense-to-sparse rewrite retains one
4 KiB slice every 256 KiB. Snapshot retention, post-GC bytes, complete content
and final unlink are checked. GC timers include FULL metadata commit and, for
external variants, physical reclamation and synchronization. Both logical
length and allocated sidecar blocks are recorded; neither implies an SQLite
database file shrink.

SQLite pages, suggested cache settings and strict-mode checkpoint defaults
match stock production. The only follow-up checkpoint-policy change is the
new-work check described above. No owned builds or correctness tests overlap
either formal comparison. Host CPU, memory, pressure and disk counters are
sampled without cache eviction or host configuration changes. All stalls are
retained. Hashing and removal of owned data follow mount/process shutdown and
occur outside measured workloads.

## Correctness and implementation limits

The baseline passes 132 component Python tests and five Release native suites.
Both external variants pass 139 applicable Python tests, including eight new
external-payload cases. The original external variant passes five native suites
in Release and ASan/UBSan; coalesced passes six in both configurations. Each
also passes a separate C++ fault/crash executable in Release and ASan/UBSan.
Only related tests are run.

Coverage includes mixed inline/external and unaligned reads, snapshots,
branches/merge, sparse reclamation, missing/truncated/wrong-workspace sidecars,
cross-format refusal, short reads/writes, partial-write and data-sync failures,
read and GC failure retries, concurrent old readers, and SIGKILL before data
sync, after data sync before SQL publication, and after durable GC deletion
before hole punching. The durable-image VFS test includes 1 MiB of external
content in the synchronized image. These are injected failures and process
crashes, not physical power-loss tests.

Copied checkpoint tests add test-only inline padding to external payload
inserts so the existing WAL pressure, admission and error tests still exercise
their original byte budgets. That padding is absent from all benchmarks and
normal prototype databases. The copied v1-migration test is explicitly skipped
once per external variant: this experimental format has no production migration.

The format uses application ID `0x56465833` and version `1003`, distinct from
production. This local Linux prototype does not provide a DuckDB-filesystem
adapter, S3 storage, migration or online backup. Database/WAL and sidecar must
be treated as a consistent set. A crash during the very first initialization
can leave a nonempty sidecar next to an empty database; initialization refuses
to overwrite or guess ownership of that pair. It requires explicit cleanup of
the owned incomplete workspace before retrying. Those lifecycle requirements
remain release gates independently of performance.

## Reproduction and retained artifacts

Run `python3 vane_fs/benchmarks/external_payload/prepare.py <new-directory>` to
create `baseline`, `external` and `coalesced` component copies. The preparer
checks its input source hashes before applying [storage.patch](storage.patch),
[fixtures.patch](fixtures.patch) and [coalescing.patch](coalescing.patch).
It copies the C++ payload implementation and applicable tests. The generated
sources were checked byte-for-byte against all tested variants.

Build each copy in Release with its own persistent CMake build directory,
the existing SQLite SDK prefix, and `VANE_FS_BUILD_TESTS`, `VANE_FS_BUILD_TOOLS`
and `VANE_FS_BUILD_FUSE` enabled. `ctest --test-dir <copy>/build
--output-on-failure` runs native suites. Python tests use separately built,
noneditable wheels in isolated environments, each selecting its own mount
binary through `VANE_FS_MOUNT_BINARY`. Set `VANE_FS_PRODUCTION_PYTHON` to the
unchanged production environment for the format-refusal test.

Compile [test_external.cpp](test_external.cpp) against the variant's static
library and SQLite with linker wraps for `pwrite`, `pread`, `fdatasync` and
`fallocate`. It takes a new owned test directory; retain its validation log and
remove its generated data after exit. The coalesced CMake target supplies the
checkpoint test's wrap for `sqlite3_wal_checkpoint_v2` and cleans its owned
temporary directory on success or failure.

[results.json](results.json) records configuration, identities, every per-run
measurement summary, validation and cleanup references. Local raw artifacts
are retained under `vane_fs/build/external-payload-20261008T134719Z/`; `followup/`
holds the independent three-variant comparison. These include exact build and
test commands, probe/measurement sources, all request latencies, sample logs,
host telemetry, binary hashes, synchronized-image/crash logs and cleanup
manifests. Large generated databases and payload files are removed after their
owned processes and mounts stop. The preparation source, patches and tests
are checked in; raw benchmark harnesses and binaries remain local artifacts.

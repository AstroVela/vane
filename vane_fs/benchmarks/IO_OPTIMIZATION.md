# FUSE I/O optimization: 2026-10-05

Reusing prepared statements, reading contiguous block ranges with one ordered
join, and removing the preliminary read for complete block writes improved
the measured sequential and random I/O paths. File data remains in 4 KiB
SQLite payloads. Transaction boundaries, `synchronous=FULL`, and live-mount
cache settings are unchanged.

## Paired FUSE measurements

The baseline is `53c452bff8b1a31521f2151821deac829b54c330`. Both binaries use
GCC 13.3.0 Release (`-O3 -DNDEBUG`), the same FUSE object and libfuse 3.14.0,
and static SQLite 3.53.2 on Linux 6.11/ext4. The optimized source and binaries
are identified by SHA-256 in the [measurement artifact](io-optimization-20261005.json).

These runs call the unchanged sequential-write, sequential-read, and random-read
functions from Drive9's `tools/bench/drive9_fuse_bench.py` at
`3334e43d47e00d1411c32defe837a87a969d1e56`. Each file is 64 MiB; sequential
requests are 1 MiB, and random reads use 4,096 seeded 4 KiB requests. Writes
include flush, fsync, and close. Reads are warmed before timing. The live
mount uses direct I/O, disables retained file data and attribute caching,
and commits every mutation synchronously.

Three repetitions use fresh databases and alternate the order of baseline
and optimized binaries. Agent-owned builds and tests finished before timing;
other builds and services were active on this shared host. All samples are
retained, including one slow write in each variant. These are local observations,
not a prediction of production throughput or a new Drive9 server comparison.

| Workload | Baseline median, MiB/s | Optimized median, MiB/s | Ratio |
| --- | ---: | ---: | ---: |
| Sequential write | 22.04 | 41.96 | 1.90× |
| Sequential read | 145.33 | 575.16 | 3.96× |
| 4 KiB random read | 27.85 | 67.86 | 2.44× |

The write ranges were 8.86–25.50 MiB/s before and 10.19–45.16 MiB/s after;
the medians should be read alongside this variation. The corresponding read
ranges were 145.15–148.31 and 551.69–668.47 MiB/s. All 18 generated file
contents passed size and SHA-256 checks outside the timed intervals.

## SQL diagnosis

A separate C++ diagnostic wraps SQLite calls and counts work on the inode API.
It writes 64 MiB as 64 pairs of 1,048,528 and 48 bytes, reproducing the request
split observed in the earlier FUSE trace, then reads warmed 1 MiB ranges.
These wrapped binaries add measurement overhead and are separate from the
FUSE throughput measurements above.

| Per 64 MiB operation | Baseline | Optimized |
| --- | ---: | ---: |
| Write prepares | 83,456 | 7 |
| Read prepares, after warmup | 32,960 | 0 |
| Write SQLite steps | 83,904 | 67,584 |
| Read SQLite steps | 49,408 | 16,704 |
| Write commits | 128 | 128 |
| Read commits | 64 | 64 |

Counts were identical across all three diagnostic repetitions. The read still
binds the current view and executes SQL; no file contents or branch frontiers
are cached. Both variants recorded zero SQLite full-scan steps in these phases.
Median diagnostic write time fell from 2.411 to 1.444 seconds, and warmed read
time from 0.419 to 0.059 seconds. Median COMMIT time in the optimized write
remained 1.048 seconds, making synchronous persistence a substantial remaining
write cost.

## Validation and retained evidence

- 102 component tests passed, including real FUSE mounts, snapshot isolation,
  sparse and unaligned range reads, maximum-size sparse files, corrupt payload
  references, overlapping versions, failed-statement reuse, and fork recovery.
- 196 affected FILE/filesystem tests passed; one HTTPFS case skipped because
  the extension was unavailable. The full Vane suites were not run.
- Both C++ tests passed in Release and under ASan/UBSan. Formatting and the
  copyleft inventory check passed.
- All 12 benchmark databases passed `PRAGMA quick_check`. Owning processes
  exited and mounts were removed before generated databases were hashed and
  deleted; the cleanup manifest accounts for approximately 1.79 GiB.

The JSON artifact includes every FUSE sample, SQL diagnostic totals, source and
binary identities, validation hashes, and removed paths. The unchanged upstream
workloads, comparison runner, cache-aware SQL probe, raw SQL breakdowns, logs,
baseline source and binaries remain locally under
`vane_fs/build/drive9-comparison/io-optimization-20261005/`. Its `run.py` and
`native_run.py` refuse to overwrite recorded results; use a fresh artifact
directory to repeat the experiment.

## New-block batching: 2026-10-06

The baseline for this stage is `0fc47514e438c6bf965b6af3af198281927e00a5`,
including bounded background checkpoints. Writes of at least 8 KiB first check
the entire block range for overlapping visibility intervals, including deleted
versions. Vacant ranges insert up to 64 nonzero payloads and their versions in
two SQL statements per batch. Occupied ranges retain the existing version
splitting and unchanged-content handling. Partial blocks preserve sparse zeros.
All batches and metadata remain in the original write transaction. Payload IDs
are allocated under its writer lock, with the existing automatic-rowid path as
a fallback near the integer limit. No persistent format, block size, durability
setting or checkpoint threshold changes.

Each diagnostic uses six alternating pairs, fresh databases and a 256 MiB file.
The native probe uses 1,048,528-byte plus 48-byte requests and wraps SQLite/VFS
calls. The tmpfs runs isolate SQL and block-processing cost; they do not measure
physical persistence. The ext4 runs include real storage syncs. The separate,
uninstrumented FUSE comparison uses 1 MiB application writes and includes create,
write, fsync and close in the timer. Both variants use fsync durability. Builds
and tests finished before all measurements; the host is shared and no global
caches were evicted. Source and binary hashes, every sample and cleanup records
are retained in [the batching artifact](bulk-write-20261006.json).

| 256 MiB native diagnostic | Before | After |
| --- | ---: | ---: |
| Write-phase SQLite steps | 270,336 | 10,496 |
| Write commits | 512 | 512 |
| WAL bytes written, MiB | 338.421 | 338.421 |
| tmpfs block processing, median seconds | 0.852 | 0.358 |
| tmpfs write + fsync, median seconds | 1.281 | 0.790 |
| ext4 block processing, median seconds | 0.873 | 0.434 |
| ext4 write + fsync, median seconds | 2.794 | 2.675 |

The SQL step count falls by 96.1% and tmpfs block-processing time by 58.0%.
The tmpfs write+fsync ranges are 1.238–1.302 seconds before and 0.772–0.832
after. On ext4, the corresponding ranges are 2.732–7.829 and 2.556–7.396
seconds. Admission-wait medians rise from 1.317 to 1.617 seconds as less block
processing overlaps the checkpoint worker. The payload layout and WAL traffic
are unchanged, so checkpoint I/O remains a limit.

Actual FUSE throughput medians are **71.88 before / 71.26 after MiB/s**, with
ranges of **47.74–87.51 / 17.19–94.96 MiB/s**. These samples do **not** establish
a sustained-throughput improvement. Both variants have storage stalls; the
optimized run also contains 11.63- and 14.90-second writes. Every sample is
retained. Sampled WAL allocation stays near 64 MiB. Unmount medians are
0.118/0.493 seconds, with maxima of approximately 0.970 seconds in both variants.
The verified gain is reduced SQL and block-processing work, rather than a
claim that storage stalls or end-to-end throughput have been fixed.

Validation covers 132 related component tests, all five native tests in Release
and ASan/UBSan, and 13 selected FUSE tests using the sanitized executable. New
cases exercise sparse and unaligned writes spanning multiple batches, snapshot
and branch isolation, truncation tombstones, concurrent payload allocation,
failure in either table after earlier batches have completed, rollback/retry,
and automatic rowid fallback at the integer limit. Raw SQL fault injection and
inspection for the new Python rollback test run in a subprocess because
[separate SQLite copies in one process cannot coordinate POSIX locks](https://www.sqlite.org/howtocorrupt.html#multiple_copies_of_sqlite_linked_into_the_same_application).
That test setup failure also reproduces on the baseline; same-library native
fault injection and isolated-process Python injection both pass.

All 24 native files passed full content comparison, all 12 FUSE files passed
size/SHA-256 verification, and all 36 databases passed quick checks. Mounts and
owned processes stopped before generated data was hashed and removed outside
measurement windows. The frozen probes, binaries, raw logs and host telemetry
remain under `vane_fs/build/bulk-write-20261006T082600Z/`. Only related tests ran.

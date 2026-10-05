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

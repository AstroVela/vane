# Strict-mode synchronization diagnosis

The Linux SQLite `fdatasync` build is now an explicit opt-in, with a separate
vcpkg triplet, source-distribution inclusion, and CI coverage for both sync
primitives. The default build and production 4 KiB BLOB format are unchanged.
See the [build instructions](../../README.md#optional-linux-sqlite-fdatasync-build).

Twelve instrumented production-format mounts reproduce long strict-mode
operations in both builds. In the slow candidate small-file phase, **5.838 s
of 6.394 s is inside WAL synchronization**. Creating and writing the files
trigger those barriers; the application's subsequent `fsync()` is already
cheap. This supports keeping the option opt-in and treating checkpoint stalls
and per-commit WAL stalls as separate optimization targets.

## Build and validation

Production sources are from `e794bc7166d16277eff52061b7c80774352ac215`, with
only the build/test integration changes in this report's commit. Both SDKs use
SQLite 3.53.2 and vcpkg baseline
`44819aa2a6c10e56065e2b0330e7d6c89d1d2574`. The default SDK reproduces the
existing static library byte for byte. The optional triplet applies
`HAVE_FDATASYNC=1` to SQLite only. Undefined symbols and actual WAL calls
confirm `fsync` versus `fdatasync`; both VaneFS core static libraries are
identical. The shared SDK and existing installed module are unchanged.

Each variant passes **132 related Python tests and six C++ tests** using its
own non-editable wheel built from the source archive. The new native test
checks the actual WAL syscall, six injected sync errors across both durability
modes, rollback, retries, close and reopened bytes. Its `auto` expectation
passes; an explicitly opposite expectation fails as intended. Invalid CMake
expectations are rejected, and the archive contains the new test and triplet.
The two CI matrix entries also retain sanitizer coverage; this local run does
not claim a new sanitizer run or completed remote CI.

Two setup failures are retained separately: a missing binary-cache directory,
then contention on a shared temporary vcpkg build directory. The final build
creates the cache and uses independent buildtrees/packages directories. No
other process was terminated and neither attempt is counted as a product test.

## Measurement method

The same host uses Linux `6.11.0-24-generic`, ext4 on `/dev/sda2`, and a
Fanxiang S103Pro 2 TB SATA SSD with write-back caching, `mq-deadline` and the
existing 2,000 us WBT target. Free space is approximately 104 GB. No kernel,
device, cache or scheduler settings are changed. A 20-second baseline and
one-second host monitoring precede and cover the runs. Owned builds and tests
finish before the diagnostic window.

Six rounds alternate the two build orders, for twelve fresh strict mounts.
Each runs a 64 MiB sequential write, a separate 64 MiB read fixture, 4,096
seeded 4 KiB reads, 512 seeded 4 KiB overwrites with final fsync, 128 small
file create/write/fsync/close operations, and 128 operations per metadata
phase. Content hashes, read slices, SQLite `quick_check`, format version 2 and
4,096-byte BLOB lengths are checked. Every run passes; no slow run is excluded.

The unchanged [timeline wrapper](../staged_payload/block_sync/timeline.cpp)
records sync calls, slow I/O and commits, file allocation, and device counters
immediately around each call. A nominal 2 ms sampler observes the mount's
threads; small-file application calls have individual timestamps. Records are
buffered until shutdown. The extra writeback stage is disabled. These are
diagnostic binaries, so their rates do not replace the preceding
[plain production measurements](../production_sync/README.md).

## Where strict small-file time goes

All twelve mounts record **258 WAL syncs** for 128 files: 129 inside create
and 129 inside write, with none inside the application's explicit fsync or
close. This agrees with the [FUSE implementation](../../src/fuse.cpp): strict
mutations already commit under FULL, so its fsync handler checks the inode;
the explicit workspace barrier is needed for the separate fsync durability
mode. Removing create/write barriers would change the strict contract.

Candidate repetition 3 has the following totals:

| Interval | Total time | Share of phase |
| --- | ---: | ---: |
| Complete small-file phase | 6,393.888 ms | 100% |
| 258 WAL sync calls | 5,837.533 ms | 91.3% |
| One database-file sync during checkpoint | 334.411 ms | 5.2% |
| Application explicit fsync calls | 17.241 ms | 0.27% |
| Application close calls | 12.434 ms | 0.19% |

The remaining time includes FUSE/SQL work and diagnostic overhead. Application
create and write totals include the storage sync intervals, so they must not
be added to them. WAL length stays at 21,135,632 bytes with no WAL truncation
in this phase. Of its WAL calls, 69 take at least 10 ms, totaling 5,644.550 ms;
their device intervals record 69 flushes and 5,625 ms of flush time. The
slowest call takes 150.134 ms, alongside 150 ms of device flush time, 88
written sectors and 56 `submit_bio_wait` observations for the calling thread.

These aligned observations point to storage flush completion as the dominant
wait in those WAL intervals. They do **not** identify SSD-internal behavior or
exclude other host I/O. In particular, these calls also write sectors; the
zero-sector observations in earlier external-format experiments must not be
substituted for this production evidence. Counter definitions and their
millisecond rounding are documented in the
[Linux 6.11 I/O statistics guide](https://www.kernel.org/doc/html/v6.11/admin-guide/iostats.html).

## Random-write tails have an additional checkpoint component

Every random-write phase records 516 WAL syncs and two database syncs. Three
retained slow phases show both components:

| SQLite build / repetition | Phase | WAL sync total | Database sync total | Longest database sync |
| --- | ---: | ---: | ---: | ---: |
| fdatasync / 0 | 6.545 s | 5.265 s | 0.984 s | 0.968 s |
| fsync / 1 | 7.793 s | 6.335 s | 1.107 s | 1.090 s |
| fdatasync / 5 | 7.177 s | 5.742 s | 1.138 s | 1.122 s |

The long database calls include `folio_wait_bit_common` and
`jbd2_log_wait_commit` samples; the slow small-file checkpoint also includes
`rq_qos_wait`. The control's longest WAL sync takes 300.437 ms with journal
commit waits. Thus the default build also exhibits tails, and each stall
cannot be reduced to the same mechanism. ext4's
[sync path](https://github.com/torvalds/linux/blob/v6.11/fs/ext4/fsync.c)
includes data writeback, journal synchronization and possible device flushes.
Whole-device counters and sampled waits cannot assign every request exactly.

Moving strict checkpoint work off the foreground is a bounded next experiment,
while preserving each acknowledged FULL commit. Database-file sync alone
accounts for about 14–16% of these random-write phases and 5.2% of the slow
small-file phase; that is not a prediction of the resulting speedup. The WAL
barriers still dominate. Optimizing the already cheap FUSE fsync handler or
silently weakening strict durability would not address the measured problem.

## Retained evidence

[Results](results.json) records build identities, validation and all rate
vectors. The [fsync](fsync.json) and [fdatasync](fdatasync.json) summaries keep
all twelve mounts, stage timings, wait counts and representative exact sync
intervals. The [artifact manifest](artifact-manifest.json) identifies the full
raw vectors, all timeline/wait records, host monitoring, configurations,
source archive, wheels, binaries, build/test logs and failed setup attempts.
The frozen local scripts record the exact build and diagnostic commands;
they must be copied and repointed to a new output directory before rerunning.

Owned measurement databases and test temporary directories are removed after
their processes stop. Their hashes and removed paths remain in the cleanup
manifest. No owned mounts or run processes remain. Source, shared-library and
installed-module checks confirm that diagnostic instrumentation is confined
to the isolated binaries. Only affected tests and repository checks run.

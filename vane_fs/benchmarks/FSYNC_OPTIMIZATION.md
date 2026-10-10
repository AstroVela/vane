# Optional fsync durability: 2026-10-06

VaneFS now offers `vane-fs-mount ... --durability=fsync`. The default remains
`strict`, including for the Python Workspace API. C++ callers can select
`Durability::Fsync` and use `Workspace::Sync()` and explicit `Close()`.

Ordinary operations in fsync mode commit immediately with SQLite WAL/NORMAL.
Other handles and connections see those commits without waiting for a flush.
A file or directory fsync changes a singleton barrier row under FULL, forcing
a real WAL commit that synchronizes all preceding commits. An empty transaction
would not suffice. `O_SYNC`/`O_DSYNC` writes use FULL in the original write
transaction, so a failed synchronous write can roll back atomically. A failed
barrier propagates its error and is retryable; the connection keeps FULL until
a successful synchronous transaction restores NORMAL.

Ordinary close is not a barrier in fsync mode. Explicit Workspace close and
clean mount shutdown perform a final FULL commit. Wait for the mount process
to exit successfully to observe shutdown errors. Unsynchronized writes can be
lost after OS crash or power failure; these guarantees assume storage honors
sync. See SQLite's [synchronous documentation](https://sqlite.org/pragma.html#pragma_synchronous).
Kernel writeback stays disabled, live data uses direct I/O, and file contents
remain in SQLite. The additional auxiliary table preserves format-2 compatibility.

## Same-window FUSE measurements

The frozen baseline is `08535e6e5323745e8cd79a26136f37cbeb945cac`.
Three rotating rounds compare that binary with the current strict and fsync
modes on Linux 6.11.0-24, GCC 13.3.0 Release, SQLite 3.53.2 and libfuse 3.14.0.
Each variant has its own database and mount. All retain the 4,096-page WAL
checkpoint trigger, connection-private intermediate inode counts and 60-second
metadata cache. There is no debug logging during measurements.

Sequential writes create a 64 MiB file in one-MiB application writes, including
flush, fsync and close inside the timer. Sequential reads warm a 64 MiB file.
Random reads use 4,096 fixed-seed 4 KiB reads against a warmed 64 MiB file.
Small writes create 100 separate 4 KiB files, fsyncing each before close.
Fixture preparation, hash validation, cleanup and shutdown are outside timers.
No owned builds or tests overlap the measurements; other host work continues.

| Workload | Baseline strict | Current strict | Current fsync |
| --- | ---: | ---: | ---: |
| Sequential write + fsync, MiB/s | 44.815 | 46.210 | 59.826 |
| Warm sequential read, MiB/s | 810.543 | 791.332 | 812.382 |
| Warm 4 KiB random read, MiB/s | 67.538 | 66.364 | 71.074 |
| 4 KiB file + fsync, files/s | 167.042 | 175.166 | 286.535 |

Against current strict, the optional mode improves sequential writes by 29.5%
and small-file fsync throughput by 63.6%. Sequential-write ranges separate:
42.89–46.90 MiB/s for strict versus 59.75–62.24 for fsync. Small-file ranges
are 166.76–180.80 versus 237.33–301.62 files/s. No random-read regression is
observed: the fsync range is 70.51–71.74 MiB/s. The default strict read medians
are 2.4% and 1.7% below the old baseline, with overlapping sample ranges.

Baseline round three has two large write outliers: 10.10 MiB/s sequential and
22.97 small files/s. Their cause is not established; all samples are retained.
The improvement percentages above compare modes of the current binary, avoiding
dependence on those old-binary outliers. These are three local samples, not a
production performance guarantee. Successful unmount times were 19.1 ms for
the baseline, 17.9 ms for strict and 17.8 ms for fsync.

## Storage mechanism

A separate instrumented native probe writes 64 MiB as 64 pairs of 1,048,528
and 48 bytes, representing the split requests previously observed with live
direct I/O. Its fsync-mode write phase includes the final `Workspace::Sync()`.
It runs after the FUSE comparison has stopped; instrumented timings are not
mixed with the throughput table.

| Write phase, three-run median | Strict | Fsync |
| --- | ---: | ---: |
| Native write requests | 128 | 128 |
| SQLite commits | 128 FULL | 128 NORMAL + 1 FULL barrier |
| Storage sync calls | 140 | 13 |
| External `sqlite3_step` calls | 67,584 | 67,584 |
| Elapsed seconds | 1.309 | 0.892 |
| Seconds inside storage sync | 0.644 | 0.391 |

Including the final checkpoint and close, sync counts fall from 148 to 19.
The change removes per-request durability barriers; it does not batch block
SQL or eliminate WAL writes and checkpoints. The fsync probe still spends
about 0.617 seconds inside commits, including sync time, so those overlapping
measurements must not be added together. Both block processing and checkpoint
costs remain relevant to future sequential-write work.

## Correctness and retention

All 126 related component Python tests pass, including 34 real FUSE cases.
All four native tests pass in Release and ASan/UBSan; 13 selected mount tests
also pass with the sanitized executable. New coverage checks:

- Real VFS sync calls, repeated barriers, a pinned WAL reader, and synchronization
  of another connection's NORMAL writes by a strict connection.
- Injected VFS sync failures, rolled-back synchronous writes, busy barriers,
  retry, and failed final-close propagation.
- Recovery using only file images captured at successful VFS syncs. This models
  loss of unsynchronized bytes; it is not a physical power-cut test.
- Cross-handle/connection visibility, concurrent append/truncate, file and
  directory fsync, fdatasync, synchronous writes, rename, SIGKILL recovery,
  branch/snapshot isolation, and clean unmount.

The comparison validates 27 VaneFS large-file hashes, 900 small-file payloads
and three SQLite quick checks. Raw samples, source/binary hashes, native SQL
counters and cleanup records are in [fsync-optimization-20261006.json](fsync-optimization-20261006.json).
All owned mounts/processes stopped and generated databases, service data and
temporary directories were removed after measurements. Logs, frozen binaries,
runner sources and full comparison artifacts remain under the ignored build
directories. The source checkout is not used as benchmark data.

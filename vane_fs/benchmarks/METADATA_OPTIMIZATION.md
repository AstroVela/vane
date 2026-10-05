# FUSE metadata optimization: 2026-10-05

The live mount now caches positive directory entries and attributes for 60
seconds under its exclusive branch lease. Linux invalidates metadata affected
by operations through the mount. Writes and O_TRUNC additionally invalidate
all inode attributes, preserving VaneFS's atime/mtime alias even when statx
requests only atime. Missing entries remain uncached. Data still uses direct
I/O, writeback remains disabled, and mutations still commit with SQLite FULL.

## Root cause and diagnostic evidence

The baseline, `87b38929436591b1b20cc6ff2ee27449886589c4`, already includes the
SQL and block-range optimizations. With zero metadata timeouts, each pathname
lookup repeats a durable update to `open_inodes.refs`. In a separate diagnostic,
64 warmed stat calls through a three-component mounted path caused 192 LOOKUP,
256 GETATTR, 384 pwrite64 and 192 fsync calls. With metadata caching, the same
loop caused none of these calls. Trace timings are excluded from benchmarks.

The consistency boundary is the exclusive mount lease: other workspace
connections cannot mutate or publish topology changes involving the mounted
branch. All supported mutations go through the same kernel, and snapshots are
immutable. Any future out-of-band writer must implement cache invalidation
before relaxing this boundary. Merely enabling a timeout was insufficient:
the new statx(ATIME) tests exposed stale atime after writes and truncating opens,
which the explicit attributes-only notifications correct.

## Three-run medians

The unchanged workload functions come from Drive9 commit
`3334e43d47e00d1411c32defe837a87a969d1e56`. Each variant uses three fresh
databases: all baseline runs precede optimized runs. Fixtures contain 1,000
stat targets, 20/1,000 directory entries, and a Git repository with 100 payload
files totaling approximately 4 MiB. Git uses `--no-local`, and status, diff,
find and Go build are warmed as prescribed by the upstream harness.

Both mounts use GCC 13.3.0 Release, static SQLite 3.53.2, libfuse 3.14.0 and
Python 3.12.13 on the same Linux/ext4 host. Agent-owned builds and tests finished
before measured windows; other host workloads were active. Baseline stat ranged
from 54.29 to 144.01 operations/s, optimized stat from 65,327 to 109,911. One
optimized small-directory sample took 158 ms rather than approximately 4 ms;
one baseline Go build took 15.68 seconds rather than approximately 4.5 seconds.
All samples are retained, without discarding these outliers.

| Workload | Before | After | Earlier Drive9 |
| --- | ---: | ---: | ---: |
| Warm stat, operations/s | 68.62 | 106,086.49 | 175,144.66 |
| 20-entry readdir, entries/s | 2,298.94 | 4,648.58 | 25,628.57 |
| 1,000-entry readdir, entries/s | 57,674.64 | 79,523.88 | 81,434.69 |
| Git clone, seconds | 6.987 | 2.199 | 3.099 |
| Git status, seconds | 1.066 | 0.279 | 0.176 |
| Git diff, seconds | 0.720 | 0.173 | 0.045 |
| find, seconds | 0.213 | 0.088 | 0.012 |
| Go build, seconds | 4.600 | 0.298 | 0.184 |

The Drive9 column is historical, not a new run in the same measurement window.
It used local TiDB unistore and MinIO, fsync mode, 60-second attribute/entry
caches and `--trust-process-local-events`. Cache and durability boundaries
remain different. The VaneFS before/after counts, particularly elimination of
hot-stat database writes, provide stronger evidence than cross-run comparisons.

## I/O check and remaining costs

A separate three-run comparison alternated baseline and optimized mounts using
the same 64 MiB sequential workloads and 4,096 random reads. Its medians were:

| Workload, MiB/s | Before | After |
| --- | ---: | ---: |
| Sequential write, including fsync | 41.00 | 41.30 |
| Warm sequential read | 638.83 | 682.29 |
| Warm 4 KiB random read | 59.28 | 65.76 |

These samples show no material I/O regression from attribute notifications;
they do not establish a latency bound. Sequential writes retain their FULL
commit cost. Open and release callbacks still update durable inode reference
counts, so directory reads and Git operations that open files still incur
synchronous commits. This change does not add a userspace directory cache or
change the fixed listing owned by each open directory handle.

## Validation and artifacts

- 25 affected FUSE/recovery tests and both native CTest cases passed. Five new
  metadata tests also passed with the ASan/UBSan mount; logs contain no sanitizer
  diagnostics. Only related tests were run.
- Tests cover cache hits, frozen snapshots, cross-process writes and truncation,
  chmod and permission checks, timestamp aliases, replace/unlink/recreate,
  parent-directory updates, rename/recreate of a directory, open-unlink lifetime,
  stable directory pagination, mount exclusion and crash recovery.
- Metadata fixtures passed 12,848 file checks, including diagnostic runs; Git
  fsck and expected dirty-file checks passed in all six benchmark runs. All 18
  I/O files passed size/hash validation, and all 14 databases passed SQLite
  quick checks. Every owned process and mount stopped before data cleanup.

The [machine-readable report](metadata-optimization-20261005.json) retains all
samples, build/source/binary identities, validation hashes, historical Drive9
values and cleanup manifests. Runners, binaries, traces and logs remain locally
in `vane_fs/build/drive9-comparison/metadata-optimization-20261005/` and
`metadata-io-20261005/`. Their runners refuse to overwrite existing results.

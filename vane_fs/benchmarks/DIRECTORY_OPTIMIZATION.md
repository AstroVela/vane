# Directory enumeration optimization: 2026-10-06

Opening a FUSE directory used to enumerate every child immediately, even when
the descriptor was only needed for `openat` or `chdir`. Enumeration then fetched
each child's inode with a separate query. Directory handles now capture their
listing on the first read, and a single ordered join fetches child attributes
at the same branch or snapshot view point.

After the first read, the list remains fixed across pagination and rewind.
Mutating a directory before that first read is now reflected in its listing.
Open a new handle to refresh an already-read list. A missing inode still raises
an error, and overlapping visible inode versions are rejected. The directory
pin remains active before enumeration, including after unlink and GC.

## Paired FUSE measurements

The baseline is `49545265f8569080a7d999d3ef42abbe00d56f40`. Three alternating
baseline/optimized pairs use fresh copies of a closed seed database and a new
mount per sample on Linux 6.11.0-24, GCC 13.3.0 Release, SQLite 3.53.2 and
libfuse 3.14.0. Both variants retain live direct I/O, disabled writeback cache,
FULL commits, the 4,096-page checkpoint trigger and 60-second metadata caching.
There is no FUSE debug logging during these measurements.

The flat directory contains 1,000 one-byte files. Each sample runs 500 directory
open/close pairs, 20 complete listings and 20 `find` commands. A retained
directory descriptor keeps the inode pinned in both variants. The Git fixture
contains 100 payload files totaling approximately 4 MiB plus two source files;
`find` sees 133 files including Git internals. Each mount creates a fresh clone,
then warms and repeats status, diff and find 20 times after dirtying one file.
The table gives medians of the three sample means, except clone, which runs
once per mount. Fixture preparation and validation are outside timers.

| Operation, milliseconds | Before | After | Time change |
| --- | ---: | ---: | ---: |
| Open/close directory with 1,000 entries | 3.580 | 0.092 | -97.4% |
| List all 1,000 entries | 4.174 | 2.362 | -43.4% |
| `find` in flat directory | 10.785 | 5.988 | -44.5% |
| Git clone | 1,728.287 | 1,705.802 | -1.3% |
| Git status | 33.806 | 35.022 | +3.6% |
| Git diff | 6.553 | 6.695 | +2.2% |
| `find` in Git fixture | 10.543 | 10.020 | -5.0% |

The directory results separate clearly across the three pairs. The Git ranges
overlap: status is 32.1–38.1 ms before and 27.3–38.9 ms after; diff is
6.0–10.9 ms before and 6.6–12.8 ms after. These runs establish a directory
improvement, not a consistent Git improvement. Other jobs remain active on the
shared host. All samples and unmount costs are retained; no owned builds or
tests overlap the measurement window.

## SQL mechanism

A separate instrumented native probe enumerates the same 1,000-entry directory
20 times after each mount has stopped. Linker wrappers count SQLite operations;
their timings are excluded from the FUSE table.

| Twenty enumerations | Before | After |
| --- | ---: | ---: |
| `sqlite3_step` calls | 60,180 | 20,180 |
| SQLite VM steps | 922,740 | 642,760 |
| Median elapsed time | 82.595 ms | 29.539 ms |
| Sync calls | 0 | 0 |

The query change removes 1,000 individual child-inode lookups per enumeration.
The remaining row steps stream the joined entries and resolve the directory's
own metadata. Lazy enumeration additionally removes all listing work for
directory handles that are never read.

## Write-through experiment

A separate prototype changes only the mount's `direct_io` flag. With a buffer
starting 48 bytes into a page, 64 one-MiB writes produce 128 FUSE requests in the
baseline (64 pairs of 1,048,528 and 48 bytes) and 64 full-MiB requests with cached
write-through. Both use the unchanged FULL core. Three debug-enabled runs give
median write-phase times of 1.346 and 1.197 seconds; the prototype also has a
5.041-second outlier. These are diagnostic timings, not production throughput.

Cached mode also enables shared writable mmap, as described by the
[Linux FUSE I/O documentation](https://www.kernel.org/doc/html/latest/filesystems/fuse/fuse-io.html).
The prototype accepts a shared mapping, exposes its new byte through both the
mapping and `pread`, then loses that byte when the mount is killed before any
msync/fsync. The baseline rejects the mapping with `ENODEV`. This is the
expected separate synchronization boundary of mapped writes; it does not show
a lost acknowledged `write(2)`. Writable mmap is outside the current VaneFS
subset, so changing the default cache mode needs a separate API and consistency
decision. The prototype is not included in production. This change does not
claim to improve sequential-write throughput.

## Validation and retention

- 113 related component Python tests passed, including real FUSE tests.
- All three native tests passed in Release and ASan/UBSan; ten FUSE directory
  and crash-recovery cases also passed with the sanitized mount.
- New tests check first-read capture across rename, fixed listings after
  rewind, an unread removed directory surviving GC, branch/snapshot attribute
  visibility, missing/deleted/overlapping child inodes, and query retry after
  faults. The first-read test fails on the frozen baseline as expected.
- All 6,600 measured-fixture payloads were checked. Six Git repositories passed
  `git fsck --full`; seven production-benchmark databases passed quick checks.
  All six prototype sequential-write files matched their hashes through FUSE
  and after reopening the database.
- All fourteen owned mounts/processes stopped. Cleanup records preserve paths,
  hashes and sizes for 81,203,069 bytes of production-benchmark data and record
  481,181,696 bytes of prototype data removed after their owners stopped.

The [JSON artifact](directory-optimization-20261006.json) contains every timing,
SQL counter, configuration, source/binary hash, content validation and cleanup
record. Frozen runners, binaries and logs remain in
`vane_fs/build/directory-optimization-20261006/`. Only related tests were run.

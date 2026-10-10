# Inode reference optimization: 2026-10-05

Repeated opens and releases of an already-pinned inode no longer rewrite its
durable reference count. The first reference persists a pin in `open_inodes`;
exact counts live in `temp.inode_references`, private to the SQLite connection.
The final release removes the pin and reclaims an orphan in one transaction.
Temporary counts roll back with failed operations. Recovery still uses durable
owner pins and OS locks to distinguish live mounts from crashed processes.

File-content and namespace/attribute mutations retain `synchronous=FULL`.
Reference operations still acquire the writer lock. The persistent schema,
60-second positive metadata cache, direct file I/O and disabled writeback are
unchanged. This change does not introduce a directory cache or defer file writes.

## Evidence

The baseline is `98b0fe44c6d475a3331b9de8cebbc15084a12571`, including the earlier
SQL and kernel metadata-cache optimizations. A new real-FUSE regression test
keeps file/directory handles live and repeats additional opens and releases.
The baseline changes the inspection connection's `PRAGMA data_version`; the
optimized mount leaves the main database unchanged.

A separate trace repeats 64 warmed file open/read/close cycles, each followed
by directory open/list/close. Both variants issue the same FUSE requests,
including 64 each of OPEN, RELEASE, OPENDIR and RELEASEDIR. Baseline work
produces 512 `pwrite64` and 256 `fsync` calls; optimized work produces none.
An FSYNCDIR request drains releases before ending the trace window. Trace
timings include strace overhead and are excluded from throughput comparisons.

## Three-run medians

The unchanged workload functions and fixture sizes match the
[preceding metadata comparison](METADATA_OPTIMIZATION.md): 1,000 stat targets,
20/1,000 directory entries and a Git repository with 100 payload files totaling
approximately 4 MiB. Git clone uses `--no-local`; status, diff, find and Go build
are warmed by the harness. Baseline and optimized runs alternate in three
pairs, using a fresh database for each run. Both builds use GCC 13.3.0 Release,
static SQLite 3.53.2 and libfuse 3.14.0 on the same Linux/ext4 host.

| Workload | Before | After |
| --- | ---: | ---: |
| Warm stat, operations/s | 111,207.61 | 113,178.26 |
| 20-entry readdir, entries/s | 4,578.92 | 36,156.82 |
| 1,000-entry readdir, entries/s | 81,311.32 | 113,381.15 |
| Git clone, seconds | 2.246 | 1.682 |
| Git status, seconds | 0.1384 | 0.0666 |
| Git diff, seconds | 0.0546 | 0.0519 |
| find, seconds | 0.0908 | 0.0120 |
| Go build, seconds | 0.3316 | 0.0927 |

Other jobs were active on this shared host; agent-owned builds/tests did not
overlap measured windows. All samples are retained. Optimized Git clone has
one 13.238-second outlier alongside 1.682 and 1.573 seconds, versus baseline
2.191–2.691 seconds. The cause of that outlier is not established, and the
median improvement does not establish a latency bound. Git diff varies from
0.046–0.072 seconds before and 0.020–0.056 seconds after; its small median
change is not strong evidence of improvement. Hot stat was already served by
the kernel cache. The directory, find and Go build results and eliminated
sync calls are the clearest improvements.

## I/O regression check

Three additional alternating pairs use the unchanged 64 MiB sequential
workloads and 4,096 seeded 4 KiB random reads. Writes include flush, fsync and
close; reads are warmed. Positive metadata TTL is 60 seconds for both variants.

| Workload, MiB/s | Before | After |
| --- | ---: | ---: |
| Sequential write | 42.47 | 44.58 |
| Warm sequential read | 730.66 | 800.68 |
| Warm 4 KiB random read | 64.76 | 67.75 |

These observations show no material I/O regression. The sequential-write
transaction boundaries and FULL commit cost are unchanged. First/last inode
pins also remain durable, and writer-lock contention remains possible even
when only temporary counts change.

## Validation and retention

- 108 related component Python tests passed, including real FUSE mounts,
  snapshot/native-reader integration, recovery and stable directory pagination.
- All three native tests passed in Release and ASan/UBSan. The new native case
  covers repeated references across sessions, invalid counts and overflow,
  failure rollback, last-close retry, orphan lifetime, branch isolation and
  lease release/reacquisition. Four key FUSE lifetime, append, reference and
  crash tests also passed with the ASan/UBSan mount without sanitizer diagnostics.
- Metadata fixtures passed 12,848 file checks with matching hashes between
  variants. Git fsck and the expected dirty-file check passed in all six runs.
  All 18 I/O files passed size/hash checks and all 14 databases passed SQLite
  quick checks.
- All 14 owned mount processes exited successfully and their mounts were
  removed. Cleanup manifests account for 1,559,855,104 bytes of generated
  data removed outside measured intervals. Only related tests were run.

The [machine-readable artifact](reference-optimization-20261005.json) retains
every sample, source/build identities, trace counts, validation hashes and
cleanup records. Frozen binaries, runners, raw reports and logs remain in
`vane_fs/build/drive9-comparison/reference-optimization-20261005/` and
`reference-io-20261005/`. No comparison server was rerun in this experiment.

# WAL checkpoint optimization: 2026-10-05 UTC

Each VaneFS connection now starts a passive automatic checkpoint at 4,096 WAL
pages instead of SQLite's default 1,000. With default 4 KiB pages the trigger
moves from approximately 4 MiB to 16 MiB. This reduces checkpoint sync frequency;
every file-content and namespace/attribute mutation still commits with
`synchronous=FULL` before returning. Transaction boundaries, persistent schema,
page-cache settings, live `direct_io`, disabled writeback and the 60-second
positive metadata cache are unchanged.

The trigger is not a maximum WAL size. Transactions can exceed it, and active
readers can prevent a checkpoint from completing. Larger WALs can also affect
reads and leave more work for recovery or connection shutdown. SQLite describes
these tradeoffs in its [WAL documentation](https://www.sqlite.org/wal.html).

## Paired FUSE measurements

The baseline is `9ee791a625b50a571e0939c7bc4e6b1ebc52e244`, including the earlier
SQL, metadata-cache and inode-reference optimizations. Three alternating pairs
use a fresh database per variant/repetition, GCC 13.3.0 Release, SQLite 3.53.2
and libfuse 3.14.0 on the same Linux/ext4 host. The unchanged workload functions
write/read 64 MiB and perform 4,096 seeded 4 KiB random reads. Sequential writes
include flush, fsync and close; reads use hot caches. Fixture preparation and
validation are outside timers. Other host jobs remain active; agent-owned
builds/tests did not overlap the measurements.

| Workload, MiB/s | Before | After |
| --- | ---: | ---: |
| Sequential write | 43.03 | 47.06 |
| Warm sequential read | 839.68 | 798.95 |
| Warm 4 KiB random read | 66.14 | 64.75 |

Sequential-write throughput improves by 9.4%. The read medians decline by 4.9%
and 2.1%, respectively; this is a write optimization with a measured read
tradeoff. Three samples on a shared host do not establish universal effects.
All samples are retained, including slower results.

The sampled WAL file grows from 5,450,792 bytes to 17,983,832–18,189,832 bytes.
Unmount times are 0.036/0.118/0.036 seconds before and 0.118/0.118/0.820 seconds
after. These cover a whole three-workload mount, rather than just the timed
sequential write, and include remaining shutdown work. The cause of the
0.820-second outlier is not established; the change does not promise lower
shutdown latency.

## Sync counts and closing costs

A separate diagnostic links the unchanged baseline core and varies connection
settings on disposable databases. It reproduces 1,048,528-byte plus 48-byte
writes, with two FULL commits per MiB. Linker wrappers count SQLite operations
and sync calls; those instrumented timings are separate from FUSE throughput.

| Diagnostic | Default checkpoint | 4,096-page checkpoint |
| --- | ---: | ---: |
| 64 MiB write commits | 128 | 128 |
| 64 MiB write-phase sync calls | 191 | 140 |
| 256 MiB write commits | 512 | 512 |
| 256 MiB write-phase sync calls | 767 | 569 |
| 256 MiB write + tail checkpoint + close, median seconds | 6.145 | 5.626 |

The 256 MiB diagnostic retains an 8.4% elapsed-time reduction after accounting
for a final checkpoint and close. Its three total times are
6.145/6.038/11.291 seconds before and 5.626/5.600/5.652 seconds after. The final
checkpoint itself takes 0.003–0.004 seconds before and 0.049–0.058 seconds
after, showing that some work moves to the end. The first 64 MiB experiment
contains large baseline stalls, so its median is not used to claim a speedup.
Increasing the page cache from 2,000 to 8,192 KiB had little independent benefit
in that experiment and is not part of the implementation.

## Validation and retained evidence

- 111 related component Python tests passed, including real FUSE mounts.
- All three native tests passed in Release and ASan/UBSan. Four FUSE crash tests
  also passed under ASan/UBSan without logged sanitizer diagnostics.
- New crash cases write 4 KiB, 4 MiB and 20 MiB, then kill the mount immediately
  after `os.write` returns, without an intervening fsync or close. Recovery, GC
  and a retained snapshot preserve the exact payload. These check process-crash
  recovery; they do not simulate storage hardware failure.
- All 18 FUSE I/O files passed size/hash checks. All six FUSE databases and all
  18 diagnostic databases passed SQLite quick checks. Every diagnostic read
  matched the expected content.
- All six benchmark mounts and owned processes stopped successfully. Cleanup
  manifests record 4,334,493,696 bytes of generated database data removed after
  the owners stopped and outside measured intervals.

The [machine-readable artifact](write-optimization-20261005.json) retains every
timing sample, sync count, configuration, source/binary identity, content hash
and cleanup record. Frozen runners, binaries, full SQL diagnostics and logs
remain in `vane_fs/build/drive9-comparison/write-optimization-20261005/`.

Only related tests were run. The separate
[same-window comparison](SAME_WINDOW_COMPARISON.md) remeasures the remaining
application-level costs against a local comparison deployment.

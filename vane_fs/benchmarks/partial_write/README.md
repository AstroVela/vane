# Partial-edge batching evaluation: 2026-10-10

**The candidate is not enabled.** It fixes a measurable SQL batching cliff,
but fresh large-write tails fail the complete-workload acceptance check.
Production C++ sources, the shared SQLite SDK and the installed component retain
their baseline hashes. The complete implementation and tests are preserved in
[partial-write.patch](partial-write.patch).

## Root cause and candidate

`WriteBytes` checks whether an entire replacement range is vacant. If a partial
edge block already exists, every new full block in that call uses the individual
version-update path. A 16 MiB native write in sixteen 1 MiB calls needs 320
wrapped SQLite steps at offset zero, but 16,752 steps when appended after one
byte. Both paths preserve the expected bytes.

This also occurs through FUSE. A traced 8 MiB application write with its buffer
48 bytes into a memory page becomes requests of 1,048,528 bytes, seven requests
of 1,048,576 bytes and a final 48 bytes. After the first request, those full-sized
requests begin inside an existing logical block. Another buffer offset produces
the same pattern with different first/last sizes. Request logs, buffer offsets
and full-content hashes are retained; this is a measured request pattern, not
an assumption that every application write splits this way.

The candidate preserves the whole-range fast path, then separately checks the
fully covered middle blocks when partial edges prevent its use. Both checks
include tombstones and the entire replacement version interval. Leading edges
use the existing version logic before explicit payload IDs are allocated;
pending batches flush before the trailing edge allocates an automatic ID.
Overflow fallback, immutable 4 KiB payloads, transaction boundaries, durability,
cache limits and checkpoint policy remain unchanged.

## Native diagnostic

Six alternating control/candidate pairs per offset use the original SQLite
3.53.2 SDK and strict durability. The offset-one results are:

| Metric, 16 MiB write | Control | Candidate |
| --- | ---: | ---: |
| Wrapped SQLite steps | 16,752 | 592 |
| Wrapped SQLite VM steps | 491,296 | 370,752 |
| CPU time median, seconds | 0.15324 | 0.13300 |
| Write throughput median, MiB/s | 61.18 | 65.79 |
| WAL bytes written | 21,370,472 | 21,370,472 |

Every round has the same respective step counts and WAL bytes. At offset zero,
both use 320 steps, 366,029 VM steps and 21,230,392 WAL bytes. The aligned native
timings vary substantially; timing is diagnostic only. Equal WAL byte counts do
not establish identical physical page order or checkpoint scheduling.

## Plain FUSE comparison

Six alternating pairs run on fresh workspaces and six on independent post-GC
workspaces: 24 mounts. Each measures 64 MiB files with 1 and 8 MiB application
requests, followed by 256 MiB files with the same two request sizes. Every write
rate includes creation, fsync and close. The remaining work includes warm reads,
seeded random reads/overwrites,
per-small-file fsync and directory operations with a final directory fsync.

The post-GC fixture retains a 32 MiB guard after zeroing alternate 4 KiB blocks
and collecting through an independent strict workspace. Both variants expose
4,128 free pages. Guard hashes, all read slices, final content hashes, format-2
checks and SQLite quick checks pass. Kernel caches stay enabled. A 20-second
host baseline and one-second counters are retained; no owned builds or tests
overlap measurement.

| Workload | Fresh control | Fresh candidate | Post-GC control | Post-GC candidate |
| --- | ---: | ---: | ---: | ---: |
| 64 MiB, 1 MiB requests, MiB/s | 28.41 | 57.16 | 94.31 | 95.44 |
| 64 MiB, 8 MiB requests, MiB/s | 96.23 | 105.67 | 94.89 | 102.02 |
| 256 MiB, 1 MiB requests, MiB/s | 94.33 | 94.97 | 92.97 | 94.63 |
| 256 MiB, 8 MiB requests, MiB/s | 94.63 | 62.87 | 93.66 | 97.79 |
| 4 KiB overwrite + final fsync, MiB/s | 14.66 | 14.83 | 13.83 | 10.63 |
| Small-file fsync, files/s | 258.45 | 241.01 | 256.03 | 251.05 |
| Complete lifecycle, seconds | 26.58 | 27.68 | 31.64 | 29.20 |

These are medians, not a tail guarantee. Fresh 256 MiB writes with 8 MiB
requests take about 2.6–2.8 seconds in all six controls, versus three candidate
samples of 8.57, 8.80 and 14.03 seconds. Combined throughput falls from 94.68 to
39.06 MiB/s. The corresponding maximum application-request latency rises from
334 ms to 6.55 seconds. Post-GC combined throughput for this workload also falls
from 70.70 to 58.12 MiB/s despite the better median. All slow samples are retained.
Complete lifecycle includes initialization, preparation, validation, deletion
and shutdown; its timer excludes the final artifact inspection and cleanup.

## I/O diagnostics and limits

A separate four-mount instrumented window reproduces other stalls. In one
candidate fresh 64 MiB write, background WAL `fsync` calls take 2.52 and 1.04
seconds. Its random-overwrite final WAL barrier takes 1.74 seconds. That window
does **not** reproduce the primary fresh 256 MiB regression, so those calls do
not prove the cause of every slow sample. Instrumented and plain timings are
kept separate; overlapping thread durations must not be summed as wall time.

A subsequent direct-file diagnostic uses six alternating fresh/reused pairs
on the same ext4 filesystem: 64 MiB per file, 1 MiB writes and a file `fsync`
every 16 MiB. All 48 measured barriers take approximately 37–67 ms and all bytes
match. This serial workload contains neither SQLite nor concurrent checkpoint
traffic. It neither reproduces the seconds-long stalls nor rules out storage
effects under the actual workload. The remaining investigation concerns the
interaction of WAL writes, synchronization, checkpoint I/O and admission;
reducing SQL calls alone has not solved it.

A final, separate four-mount fresh window samples the mount's threads every
20 ms and joins those observations to instrumented sync intervals. Both controls
now also show slow 256 MiB streams, demonstrating that the earlier per-variant
timing difference is not a stable causal attribution. One control spends 3.53
seconds syncing the WAL and 1.94 seconds syncing the database during that phase.
Its WAL sync has 127 samples in `folio_wait_bit_common`, 14 in `rq_qos_wait` and
four in `jbd2_log_wait_commit`; the database sync has 85 in `rq_qos_wait`.
Other slow control/candidate barriers show the same wait locations. These are
sampled observations, not exact time allocations or complete kernel stacks.

`rq_qos_wait` participates in block-layer admission throttling; both
[writeback throttling](https://raw.githubusercontent.com/torvalds/linux/v6.11/block/blk-wbt.c)
and [cgroup latency control](https://raw.githubusercontent.com/torvalds/linux/v6.11/block/blk-iolatency.c)
call it. This host reports Linux 6.11, ext4 on `/dev/sda2`, `mq-deadline` and
`wbt_lat_usec=2000`. The observations support investigating block-I/O pressure
during WAL synchronization and checkpoint copying. They do not identify which
controller, another workload or device behavior caused each wait. Reading full
kernel stacks was unavailable; no host, scheduler or device setting was changed.
[Wait observations](kernel-waits-summary.json) and [read-only host settings](block-settings.json)
retain the evidence independently of the throughput comparisons.

## Validation and retained evidence

The isolated non-editable Release candidate passes **142 related Python tests
and six C++ suites**; the control also passes its six native suites. Added cases
cover existing head/tail bytes, snapshots and child branches, GC, rollback after
payload/version/inode failures, retries and payload IDs at the integer limit.
No broader Python or release suite runs.

[Results](results.json), [summary](summary.json), raw per-variant files and the
artifact manifest retain configurations, source/binary identities, every sample,
validation hashes, host observations and cleanup records. Generated databases
and test directories are removed only after their processes stop. The earlier
[UPDATE experiments](../version_update/README.md) are also retained as rejected
candidates; their controls belong to separate measurement windows.

The [preparer](prepare.py) exports the baseline component at
`d31b0a151468b6d68912e2e406f2eb9892e8663d`, applies a candidate without fuzz and
verifies all native, binding and test source hashes. It also accepts
`--variant all-versions` or `--variant live-blocks` for the preceding experiments.

```bash
python vane_fs/benchmarks/partial_write/prepare.py vane_fs/build/partial-write-repro
```

Build each prepared component separately with the [component instructions](../../README.md).
The [native reproducer](alignment_probe.cpp) accepts a fresh database path and
an offset of `0` or `1`, writes 16 MiB in sixteen calls and validates all bytes.
Its [reproducer validation](published-probe-validation.json) records the compiler
and SQLite wrapper arguments and rechecks the 16,752/592-step offset-one result.
The archived runners record the exact non-editable build commands, alternating
orders, SDK paths and linker instrumentation. Use a new output directory for a
new measurement window and retain failed attempts separately.

# Separating WAL writeback from the durability barrier

This follows the [SQLite synchronization experiment](../wal_sync/README.md).
**The remaining slow calls include device flush completion, data writeback and
ext4 journal waits. Waiting for writeback before `fdatasync` does not remove
the tail.** This investigation adds diagnostic binaries and reproducible probes;
it does not change production behavior or adopt a new storage format.

## Method

Both FUSE binaries link the same format-1004 UPDATE component and the same
SQLite 3.53.2 library built with `HAVE_FDATASYNC=1`. The ordinary binary records
the existing calls. The split binary first calls `sync_file_range` with
`WAIT_BEFORE | WRITE | WAIT_AFTER` on WAL descriptors, then executes the original
`fdatasync`. Other files retain their original sync sequence. Neither build
changes FULL barriers, checkpoint thresholds or WAL retention.

[`sync_file_range`](https://man7.org/linux/man-pages/man2/sync_file_range.2.html)
waits for page writeback but does not persist all necessary metadata or flush
volatile device caches. It is only an extra diagnostic stage here. The final
durability call remains necessary. Concurrent checkpoint activity can dirty
the file again between stages, so the remaining call is not assumed to be
exclusively a device flush.

The C++ wrapper records syscall intervals, file length/allocation, checkpoint
and truncate events, and whole-device `/sys/block/sda/stat` snapshots around
sync stages. An external sampler observes owned threads with a nominal 2 ms interval. Counters
and timestamps are kept in memory during each mount, then saved after shutdown.
The low-frequency host monitor also records the wider measurement window.

Linux documents the final two
[device statistics](https://www.kernel.org/doc/html/v6.11/admin-guide/iostats.html)
as completed flush requests and elapsed flush milliseconds. These counters are
device-wide, rounded to milliseconds and can include other processes. Overlapping
foreground/background intervals must not be added together. Zero-length flush
activity also appears in the observed write-request counters; those counters
alone must not be interpreted as data writes. Written sectors provide a separate
check on completed data volume.

The host runs Ubuntu `6.11.0-24-generic`, ext4 on `/dev/sda2`, a SATA SSD with
write-back caching, `mq-deadline` and `wbt_lat_usec=2000`. Kernel request tracing
was unavailable: effective capabilities were empty, `perf_event_paranoid=4`,
and noninteractive sudo required a password. The denial is retained. No tracing
permissions, global caches, scheduler, device or throttling settings were changed.

## Ordinary-file control

Six balanced orders compare three variants, 128 calls per variant per order.
Each file starts with 16 MiB of fully initialized and persisted data. Ordinary
and split cases overwrite 32 KiB before each barrier; clean cases issue barriers
without intervening writes. Every final file is compared byte-for-byte and hashed.

| Individual call, pooled median | Ordinary | Split | Clean |
| --- | ---: | ---: | ---: |
| Writeback wait, ms | — | 0.169 | — |
| Final fdatasync, ms | 0.954 | 0.797 | 0.042 |

The shorter split barrier excludes its additional writeback stage. It is not
an equivalent improvement in complete-operation latency. The split case still
contains a 29.835 ms barrier with zero completed write sectors, one completed
flush and a 30 ms increase in flush time. Thus a slow final flush is reproducible
without SQLite or VaneFS. This control does not reproduce every longer FUSE tail.

## FUSE evidence

Six balanced orders produce twelve independent mounts. Each uses the preceding
32 MiB fragmented guard, GC, 64 MiB read fixture, 512 seeded 4 KiB overwrites,
128 small files with individual fsync, and directory operations. All contents,
guard hashes and SQLite `quick_check` results pass. These are instrumented
diagnostic measurements, separate from the preceding plain performance window.

| Diagnostic median | Ordinary | Split |
| --- | ---: | ---: |
| Overwrites + final fsync, MiB/s | 11.64 | 11.92 |
| Small file + fsync, files/s | 361.93 | 358.04 |

Every slow sample remains included. The ordinary small-file run at repetition 3
takes 4.524 seconds (28.29 files/s); the split run at repetition 4 takes
4.418 seconds (28.97 files/s). The split operation therefore does not meet a
tail-latency acceptance gate.

The split repetition 4 provides the clearest separation:

- Its overwrite workload's final application fsync takes 803.81 ms. The foreground
  WAL writeback stage takes 403.70 ms, then the final fdatasync takes 399.69 ms.
  The latter interval completes one device flush, adds 399 ms of flush time,
  and has 140 samples in `submit_bio_wait`. The background checkpoint worker
  also participates earlier in this interval.
- During its 128 small-file operations, foreground WAL writeback totals
  597.77 ms and final WAL barriers total 3,592.96 ms. The WAL remains at 16 MiB
  with no truncate during this phase. The worker has 1,532 futex-wait samples.
- Thirty-seven final barriers exceed 10 ms while completing zero write sectors.
  Their elapsed time totals 2,896.42 ms; device flush time increases by 2,896 ms
  across those nonoverlapping foreground intervals. Examples take 108.19 and
  86.66 ms, each with one flush and predominantly `submit_bio_wait` samples.
  This strongly supports flush completion as the dominant wait in these calls.
- Not every slow call has that explanation: one final barrier spends 152.33 ms
  with samples in `jbd2_log_wait_commit`. A 156.39 ms writeback stage samples
  `folio_wait_bit_common` while a device flush also completes.

A separate slow call during preparation in split repetition 0 spends 1,909.42 ms
in WAL writeback, including 489 `rq_qos_wait` samples. It is retained as lifecycle
evidence, not counted as small-file or overwrite throughput. This establishes
that block-layer QoS waiting also occurs; it does not establish that changing
the WBT target would improve overall behavior.

The observed separation agrees with the upstream
[ext4 sync path](https://github.com/torvalds/linux/blob/v6.11/fs/ext4/fsync.c),
which waits for file data, commits the necessary journal transaction and issues
a device flush when required. The installed Ubuntu kernel may carry additional
patches. Device counters and wait samples are not individual request tracing:
they do not identify SSD internal work, assign all interference to another
process, or establish a hardware fault.

## Decision

Keep the split call sequence in diagnostic binaries. It moves some waiting into
an earlier call and leaves both the serial fsync tail and journal/QoS waits.
Do not adopt it as a performance optimization or remove the final barrier.

For this serial small-file workload, each acknowledged fsync still needs its
durability boundary. SQL and copying optimizations cannot remove the observed
flush-completion component of that boundary. An application-controlled batch can
amortize barriers only when its durability contract permits that batching.

The next bounded software experiment is to validate `HAVE_FDATASYNC=1` against
the production BLOB format in isolation, including syscall failures and recovery,
then measure the same durability contracts. The earlier improvement in ordinary
sync cost may transfer without a format migration; these external-payload results
are insufficient to adopt it. A different physical device or privileged request
trace would be needed to separate the unresolved device/scheduler causes further.

## Reproduce and retained evidence

The raw probe needs Python and Linux, and refuses an existing output directory:

```bash
python3 vane_fs/benchmarks/staged_payload/block_sync/raw_probe.py \
  /absolute/path/on/test/device/raw-probe --stat /sys/block/sda/stat
```

Use the whole-device stat file corresponding to the filesystem under test.
The raw probe never changes device settings and removes its owned data after
verification and closing descriptors.

For FUSE, first build the preceding experiment's fdatasync C++ component and
SQLite SDK. The following preparer links both diagnostic binaries without
modifying those inputs. It also links an existing isolated `component/venv`
when present; the aged workload uses that non-editable installation for GC.

```bash
python3 vane_fs/benchmarks/staged_payload/block_sync/prepare.py \
  /absolute/path/to/new-diagnostic-directory \
  --component /absolute/path/to/fdatasync-component \
  --sqlite-prefix /absolute/path/to/fdatasync-sqlite-sdk \
  --fuse-prefix /absolute/path/to/fuse3-sdk/usr
export VANE_FS_SYNC_STAT_PATH=/sys/block/sda/stat
```

The complete frozen workload, serial runner, host monitor, sampler output,
all traces, source/binary identities, build logs and cleanup records are listed
in [artifact-manifest.json](artifact-manifest.json). The recorded runner uses
the preceding workload unchanged and checks one successful writeback stage per
WAL barrier in split mounts, none in ordinary mounts. It writes full timing
vectors locally. The preparer is also rebuilt separately after measurements;
its generated sources and binaries are compared with the measured versions.

[results.json](results.json), [raw controls](raw-results.json),
[ordinary mounts](fuse-ordinary.json) and [split mounts](fuse-split.json) retain
all scalar samples and diagnostic summaries. All eighteen raw files and twelve
mounted workspaces are removed after descriptors/processes stop, outside measured
phases. No owned builds or tests overlap measurement windows. Production sources,
the installed module and both linked static libraries retain their recorded
hashes. Validation is limited to these affected probes, content/trace checks,
reproduction builds and repository checks; the full base suite is not rerun.

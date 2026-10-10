# Strict WAL capacity reuse evaluation

**Do not adopt this candidate as the production strict policy.** It removes
repeated WAL growth during reuse, but retains cold-write and random-write
regressions and does not improve the complete workload consistently.

This experiment isolates the allocation policy from the preceding
[checkpoint batching evaluation](../batching/README.md). The baseline revision
is `f1a162f7ca98e97a3d32ca40759becb46f9e8514`. Production sources, the shared
SQLite SDK and installed component remain unchanged.

## Policy and correctness

The isolated [candidate patch](reused.patch) changes the batched worker's
`journal_size_limit` from 16 to 64 MiB. The 16 MiB unbackfilled-work batching
threshold, 64 MiB admission budget, partial-checkpoint retry state, reader
backpressure and every strict FULL commit are unchanged. No new preallocation
or idle-reclamation mechanism is introduced. The candidate changes one runtime
setting; the rest of the patch updates allocation assertions and adds coverage
for reuse and final-close reclamation.

In the tested SQLite source, `walLimitSize` only truncates a file that exceeds
the limit. It does not extend a smaller WAL. A reset can retain previously
allocated capacity, while a single atomic transaction can temporarily exceed
the limit. `journal_size_limit` is therefore a reclamation policy, not a hard
space quota. SQLite handles final-close deletion; persistent-WAL mode is not
enabled by this experiment.

Tests cover 4,096/8,192-byte pages and both VaneFS durability modes: a small
workspace does not allocate 64 MiB, an oversized transaction resets to the
retained capacity, successive writes reuse that capacity, strict writes still
synchronize, final close removes WAL/shared memory, and reopen preserves all
changed and untouched bytes without preallocating the WAL again. Existing
tests retain pinned-reader admission, mixed writers, background/foreground I/O
failures, close retry, batching, fork and process-crash recovery. The prior
16 MiB policy fails the new capacity assertion, as expected; this is a policy
difference, not a correctness defect in that policy. Process-crash and syscall
tests are not physical power-cut tests.

Both SQLite builds pass **132 related Python tests, seven Release C++ tests
and seven C++ tests under ASan/UBSan**. The preparer reproduces all 24 tested
source/include/test files byte for byte; those files match the source archive
and both built components. Actual-syscall auto-detection passes and opposite
expectations are rejected. Only related tests run.

## Measurement method

All measured mounts use **VaneFS strict durability**. The `fsync` and
`fdatasync` labels identify matching SQLite SDK builds, not the VaneFS
durability setting. Each SDK window compares the current inline production
policy, the preceding 16 MiB batched policy, and the new 64 MiB retained-capacity
policy. Six rounds rotate the three positions, then reverse the orders. Every
sample, including slow runs, is retained.

The workload remains 64/256 MiB sequential writes, a separate warmed 64 MiB
read fixture, seeded 4 KiB random reads/overwrites, 128 small files with fsync,
and 128 mkdir/rename/rmdir operations with final directory barriers. Writes
include creation, application fsync and file close. Content hashes, requested
slices, SQLite quick checks and format-2/4,096-byte BLOB checks validate every
mount. Instrumented runs identify syscall, checkpoint and file-growth
behavior; their rates are not used as plain benchmark results.

Initialization, mount readiness, the existing 250 ms settling interval,
shutdown and the complete lifecycle are timed separately. Lifecycle time
includes fixtures, content validation and deletion, ending when the mount
process exits; the post-close SQLite inspection, output hashing and removal of
owned temporary data occur afterward. These totals expose shifted work but
are not a pure application-throughput measurement. Mount readiness uses a
50 ms polling interval, so it cannot resolve small startup differences.

Logical size and allocated blocks are captured after initialization, after
mount readiness, after each large-write phase, before shutdown and after
shutdown. Per-write WAL sizes are sampled as before. File sizes are not
instantaneous peaks; retained file capacity is distinct from active WAL frames
and unbackfilled work. Every comparison window has a 20-second host baseline
and one-second monitoring. Owned builds and tests finish before measurement;
no device, kernel, cache or scheduler setting is changed.

## Diagnostic findings

The capacity change removes WAL extension from both the random-overwrite and
small-file phases in both SDKs. With the optional SDK:

| Phase / policy | Syncs following WAL growth | Syncs at unchanged size | Median sync after growth, ms | Median sync at unchanged size, ms |
| --- | ---: | ---: | ---: | ---: |
| Random overwrite, 16 MiB batched | 383 | 129 | 2.017 | 0.924 |
| Random overwrite, 64 MiB reused | 0 | 512 | — | 0.960 |
| Small file, 16 MiB batched | 256 | 0 | 2.019 | — |
| Small file, 64 MiB reused | 0 | 256 | — | 0.967 |

All 512 random-write and 256 small-file foreground WAL barriers remain in the
candidate. The default `fsync` build also avoids growth, but its fixed-size
sync medians remain about 1.96/1.98 ms. Its instrumented random-write phase
still takes 6.748 seconds, including 6.431 seconds in 512 foreground WAL syncs.
Their p95 is 76.44 ms and maximum is 175.59 ms. During the longest call the WAL
stays at 64 MiB, and the calling thread is sampled waiting in
`jbd2_log_wait_commit`. Capacity reuse does not eliminate all foreground
synchronization delays; this remaining stall cannot be explained by WAL
extension during that call.

The first 64 MiB write starts with an unallocated WAL: growth still precedes
98 foreground syncs in the candidate, versus 104 in the preceding batched
policy and 28 in the inline control. Reuse cannot remove the first allocation.
These single instrumented runs establish the changed syscall/file-growth
behavior; the repeated plain results below determine the performance decision.
Whole-device counters and thread waits cannot assign every request or identify
SSD-internal causes. Durations on different threads overlap.

## Plain results and decision

The 36 plain mounts retain six observations for every SDK/policy combination.
Medians across all six mounts:

| Workload | fsync inline | fsync 16 MiB | fsync 64 MiB | fdatasync inline | fdatasync 16 MiB | fdatasync 64 MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 MiB write + fsync, MiB/s | 16.58 | 12.61 | 35.59 | 59.07 | 55.30 | 35.03 |
| 256 MiB write + fsync, MiB/s | 50.76 | 51.79 | 56.28 | 59.37 | 54.72 | 67.54 |
| Warm sequential read, MiB/s | 532.80 | 511.80 | 524.04 | 543.39 | 544.88 | 503.80 |
| Warm 4 KiB random read, MiB/s | 73.11 | 69.60 | 68.79 | 74.94 | 68.56 | 70.36 |
| 4 KiB overwrite + final fsync, MiB/s | 1.42 | 1.45 | 1.49 | 2.50 | 1.60 | 2.30 |
| Small file + fsync, files/s | 175.69 | 157.48 | 177.98 | 293.51 | 168.66 | 298.97 |
| mkdir, operations/s | 321.70 | 299.57 | 315.47 | 538.37 | 331.27 | 457.86 |
| rename, operations/s | 336.38 | 340.72 | 339.88 | 609.45 | 644.05 | 595.22 |
| rmdir, operations/s | 200.65 | 200.29 | 201.88 | 376.67 | 383.30 | 373.51 |

The isolated capacity change improves common reuse costs: with the optional
SDK, small-file medians rise from **168.66 to 298.97 files/s** and 256 MiB write
medians from **54.72 to 67.54 MiB/s** relative to the 16 MiB batched policy.
Small-file combined throughput rises from 78.38 to 304.04 files/s; the control
retains one 21.23 files/s run while the candidate has no similarly slow sample
in these six mounts. Its pooled per-file p95 falls from 78.40 to 3.69 ms.
This is a result for the observed runs, not a guarantee that such stalls cannot
recur.

Several remaining comparisons prevent adoption:

- Optional-SDK first 64 MiB write combined throughput falls from 25.20 MiB/s
  inline to **17.25 MiB/s** with reuse. The candidate's six rates are 56.75,
  8.79, 14.44, 58.83, 8.91 and 55.61 MiB/s; its median is 35.03 versus 59.07
  inline. The candidate's 256 MiB combined rate does improve from 42.62 to
  54.27 MiB/s.
- Optional-SDK random-write combined throughput falls from 2.50 MiB/s inline
  to **0.72 MiB/s**. The candidate retains two runs near 0.30 MiB/s. Pooled
  request p95 increases from 1.54 to 29.87 ms, even though the candidate's
  median throughput exceeds that of the preceding batched policy.
- With the default SDK, the 256 MiB median improves from 50.76 to 56.28 MiB/s,
  but combined throughput falls from 41.90 to 39.54 MiB/s. Random-write
  combined throughput falls from 0.82 to 0.59 MiB/s. Both inline and reuse
  small-file controls retain one run near 18 files/s.

The default SDK's inline 64 MiB median is itself much lower than in the prior
window. Results from different windows must not be treated as interchangeable
baselines. Current controls, every raw vector, combined rates and pooled
latencies are retained in [results](results.json) and the six per-policy JSON
files. These finite shared-host measurements do not establish a universal
speedup or a stable tail-latency distribution.

## Lifecycle and space

Medians across the same six mounts, in milliseconds except total seconds:

| SDK / policy | Initialize, ms | Mount ready, ms | Shutdown, ms | Complete lifecycle, s |
| --- | ---: | ---: | ---: | ---: |
| fsync / inline | 465.35 | 231.66 | 19.44 | 24.35 |
| fsync / 16 MiB | 364.92 | 231.52 | 27.32 | 24.95 |
| fsync / 64 MiB | 264.75 | 128.69 | 35.78 | 28.26 |
| fdatasync / inline | 32.31 | 52.35 | 19.95 | 21.91 |
| fdatasync / 16 MiB | 48.14 | 51.75 | 19.33 | 23.53 |
| fdatasync / 64 MiB | 32.21 | 51.74 | 51.86 | 21.98 |

Default-SDK complete lifecycle time increases from **24.35 to 28.26 seconds**
relative to inline. With the optional SDK it is essentially unchanged,
**21.91 versus 21.98 seconds**, although it improves over the 16 MiB candidate.
Initialization variation is not evidence of preallocation: the tests and
file-size observations confirm that no 64 MiB allocation occurs at startup.
Each lifecycle is measured directly; its median is not the sum of independent
stage medians.

Before shutdown, the 16 MiB policy retains 16 MiB logical WAL size and
16 MiB + 4 KiB allocated in every plain run. The new policy retains 64 MiB
logical and 64 MiB + 4 KiB allocated: **48 MiB more per active database in this
workload**. Inline retains about 28.99 MiB. Every final close removes the WAL
and shared-memory file before the post-close inspection. Retained capacity
and active committed frames remain distinct; an oversized atomic operation
can still exceed the allocation limit until a later reset. These observations
do not promise immediate idle reclamation while a connection remains open.

The next targeted question is how to reduce **first-generation WAL growth**
and obtain earlier reuse without weakening FULL commits or pinned-reader
admission. Increasing retained capacity further would not address the
fixed-size WAL sync stalls observed here. No such additional policy is
implemented in this experiment.

## Retained evidence and cleanup

[Diagnostics](diagnostics.json) and [WAL-growth groups](wal-growth.json) retain
all six instrumented mounts. The [artifact manifest](artifact-manifest.json)
keeps full timelines, individual file-size observations, thread waits and host
monitors, exact SDK/source/binary identities, source archives, wheels,
build/test logs, the capacity-policy control and preparer output. No timing
samples are removed. Earlier frozen experimental artifacts are reverified
unchanged.

All **46 recorded benchmark/test data roots** are removed after process
shutdown, outside measured phases. All 42 measured mounts are unmounted and
stopped; no owned processes remain. No owned build or functional-test interval
overlaps a diagnostic or plain comparison window. Production source,
installed-module and shared-SDK hashes still match the recorded baseline.

## Reproduce

The preparer verifies the recorded baseline and composes the preceding
batching patch with this capacity-only delta. The output directory must not
exist:

```bash
python3 vane_fs/benchmarks/strict_checkpoint/wal_reuse/prepare.py /absolute/path/to/controls
export CMAKE_PREFIX_PATH=/absolute/path/to/sqlite-sdk
cmake -S /absolute/path/to/controls/reused \
  -B /absolute/path/to/controls/reused/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DVANE_FS_BUILD_PYTHON=OFF \
  -DVANE_FS_BUILD_TESTS=ON -DVANE_FS_TEST_SQLITE_SYNC=fsync
cmake --build /absolute/path/to/controls/reused/build -j 2
ctest --test-dir /absolute/path/to/controls/reused/build --output-on-failure
```

Use `inline` or `batched` for the controls. Repeat with the optional SDK and
`fdatasync` expectation. Add the tools/FUSE targets for mounted workloads;
Python validation uses isolated, non-editable wheels built from the source
archive. The retained harness, source and binary identities define the exact
measurement configuration.

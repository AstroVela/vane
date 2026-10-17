# Strict checkpoint batching evaluation

**Do not adopt this candidate as the production strict policy.** Batching
eliminates the repeated tiny checkpoints found in the
[direct worker-reuse evaluation](../README.md), but the plain comparison still
has substantial write regressions. The final candidate and its tests remain
isolated in [batched.patch](batched.patch). Production sources, shared SQLite
SDK and installed component are unchanged. The baseline commit is
`0e165969758224511597717b3f890c23f1f3ca2d`.

## Policy and correctness

Before starting a fresh passive checkpoint, the worker queries SQLite's current
WAL progress with `SQLITE_CHECKPOINT_NOOP`. If the difference between committed
and copied frames is below 16 MiB, it defers the work. No cached frame index is
carried across WAL generations. Unknown progress and lock contention follow the
existing checkpoint path; status errors are reported before a subsequent
mutation or barrier.

An already-started checkpoint bypasses this batching gate. Its retry must
finish after a reader releases its pin, even if the remaining work is below
16 MiB and no new commit arrives. The retry flag also survives a joined
`Stop`/`Start`, as used when an explicit close fails. The initial candidate
kept this flag local to `Run`; the new restart regression reproduced a stalled
partial checkpoint. The final patch moves that state onto the worker. The
failed candidate, regression output and its initial six diagnostic mounts are
retained separately from the final candidate.

Strict writers still use FULL for every acknowledged mutation. Fsync-mode
writers retain their existing NORMAL commits and foreground FULL barriers.
The worker's 64 MiB admission budget, 16 MiB retained allocation and retry
interval are unchanged. Strict-mode sub-budget work may remain in the durable
WAL while idle; this policy adds no idle timer and preserves commit durability.
Admission still considers total WAL size, including a fully backfilled log
that a reader prevents from being reused.

The new native test uses both FULL/NORMAL and 4,096/8,192-byte pages. It verifies
that small additions to a retained WAL do not each restart a checkpoint,
partial work resumes after worker restart and reader release without another
commit, a new WAL generation triggers work normally, and an injected status
failure is reported and remains retryable. The unbatched implementation fails
the small-addition assertion; the initial batched candidate fails the restart
assertion. Existing expanded tests cover mixed writers, pinned-reader
admission, foreground WAL failures, startup/close retries, fork and crash
recovery. Process-crash and syscall tests are not physical power-cut tests.

Both final candidates pass **132 related Python tests and seven native tests**,
plus the same seven native tests under ASan/UBSan. A separately rebuilt
preparer output also passes the batching test. Only affected tests run.

## Measurements

Two matching SQLite SDKs are compared in separate windows: the default `fsync`
build and the optional `fdatasync` build. These names denote the SQLite syscall;
**all measured mounts use VaneFS strict durability**. Each SDK window alternates
inline/batched order over six rounds, for **24 fresh FUSE mounts**. Every run
retains its hashes, slice comparisons, format-2/4,096-byte BLOB checks and SQLite
quick checks. No samples are excluded.

The workload is unchanged: 64 and 256 MiB writes include creation, fsync and
close; reads use a separately prepared, warmed 64 MiB file; random reads and
overwrites use fixed seeds; small files include create/write/fsync/close; each
directory phase performs 128 operations and a final directory fsync. A
20-second baseline and one-second host monitoring accompany each window.
All owned builds and functional tests finish outside measurement windows.
No device, kernel, scheduler or cache setting is changed.

Medians across all six mounts:

| Workload | fsync inline | fsync batched | fdatasync inline | fdatasync batched |
| --- | ---: | ---: | ---: | ---: |
| 64 MiB write + fsync, MiB/s | 52.64 | 11.93 | 59.12 | 55.59 |
| 256 MiB write + fsync, MiB/s | 49.53 | 51.78 | 59.57 | 54.11 |
| Warm sequential read, MiB/s | 515.97 | 500.65 | 519.36 | 508.64 |
| Warm 4 KiB random read, MiB/s | 66.98 | 65.48 | 67.55 | 64.87 |
| 4 KiB overwrite + final fsync, MiB/s | 1.40 | 1.43 | 2.50 | 1.61 |
| Small file + fsync, files/s | 173.26 | 161.90 | 288.16 | 166.48 |
| mkdir, operations/s | 326.79 | 306.36 | 547.65 | 337.27 |
| rename, operations/s | 338.66 | 349.37 | 610.46 | 570.63 |
| rmdir, operations/s | 200.78 | 201.65 | 372.23 | 354.60 |

The default candidate's 64 MiB rates are 11.76, 11.24, 53.12, 10.15, 12.09 and
13.96 MiB/s. Its combined rate (total bytes / total time) falls from 36.98 to
13.46 MiB/s; pooled application-write p95 increases from 88.86 to 244.26 ms.
The small 256 MiB median improvement also hides a combined-rate decline from
42.38 to 38.92 MiB/s and an application write lasting 4.173 seconds.

With the optional SDK, random-write combined throughput falls from 2.49 to
0.91 MiB/s, and small-file combined throughput from 294.28 to 149.58 files/s.
Not every aggregate declines: its 64 MiB combined rate improves from 22.03 to
29.58 MiB/s because the control contains two slow runs, versus one for the
candidate. Both default-SDK small-file variants retain a slow run near
19 files/s. [Results](results.json) and the four plain-run files preserve every
vector, combined rate and pooled latency; these finite shared-host samples do
not establish a universal speedup or a tail-latency distribution.

## What the diagnostics establish

The initial three-way comparison records inline, unbatched and initially
batched mounts for both SDKs. After fixing retry persistence, two further
mounts verify the final candidate. Instrumented rates are diagnostic only.

In each unbatched rename phase, 128 operations produce 128 foreground WAL
syncs plus 128 background database syncs. Both initial and final batched
candidates retain the **128 foreground WAL syncs and zero database syncs**.
The repeated-checkpoint mechanism is resolved for this workload.

However, WAL allocation still grows much further before reuse: the plain
64 MiB write samples peak at about **17.15 MiB inline versus 63.15 MiB batched**.
The 256 MiB phase peaks at 20.16 versus 63.78 MiB. These are sampled file sizes,
not a hard per-operation quota or guaranteed instantaneous peak.

The [WAL-size analysis](wal-growth.json) compares consecutive foreground WAL
syncs in the diagnostic traces. For the optional SDK:

| Phase / policy | Syncs following WAL growth | Syncs at unchanged size | Median sync after growth, ms | Median sync at unchanged size, ms |
| --- | ---: | ---: | ---: | ---: |
| Random overwrite, inline | 0 | 516 | — | 0.942 |
| Random overwrite, batched | 383 | 129 | 1.942 | 0.925 |
| Small file, inline | 0 | 258 | — | 0.979 |
| Small file, batched | 256 | 0 | 1.929 | — |
| Rename, inline | 0 | 128 | — | 1.003 |
| Rename, batched | 0 | 128 | — | 1.004 |

The candidate's random-write phase spends 949.34 ms in 512 foreground WAL
syncs, compared with 515.76 ms in 516 syncs for the inline diagnostic control.
The small-file figures are 536.77 ms in 256 syncs versus 282.80 ms in 258 syncs.
Reducing checkpoint calls therefore does not ensure cheaper foreground syncs.
The association with file growth gives a concrete next hypothesis: the
admission/reuse/allocation policy can consume the savings from batching.
This is not an independently controlled proof that allocation explains every
plain slow sample, nor an attribution to SSD-internal behavior.

The next experiment should **isolate WAL allocation and reuse while holding
the batch rule and every FULL commit fixed**. It must include initialization
and shutdown costs, retained space and pinned-reader admission, so moving work
outside the timed write phase is not mistaken for a throughput improvement.
That experiment is not implemented here.

## Retained evidence and cleanup

[Diagnostics](diagnostics.json) includes both the superseded attempt and final
candidate. The [artifact manifest](artifact-manifest.json) retains all eight
timelines, thread waits and host monitors, exact sources/SDKs/binaries,
build/test logs, both expected-failure regressions and preparer reproduction.
The host summaries retain load, CPU and I/O-wait counters; they do not exclude
samples or assign unrelated host activity to an individual call.

All **42 recorded owned data roots** are absent after shutdown, including all
32 measured mounts across both attempts. Cleanup occurs outside measured
phases. No owned processes or mounts remain. Source and installed-module
hashes match the production baseline, and previous experimental artifacts are
reverified unchanged.

## Reproduce

Use the recorded baseline, a matching static SQLite SDK, CMake, Ninja, a C++17
compiler and `patch`. The output directory must not exist:

```bash
python3 vane_fs/benchmarks/strict_checkpoint/batching/prepare.py /absolute/path/to/controls
export CMAKE_PREFIX_PATH=/absolute/path/to/sqlite-sdk
cmake -S /absolute/path/to/controls/batched \
  -B /absolute/path/to/controls/batched/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DVANE_FS_BUILD_PYTHON=OFF \
  -DVANE_FS_BUILD_TESTS=ON -DVANE_FS_TEST_SQLITE_SYNC=fsync
cmake --build /absolute/path/to/controls/batched/build -j 2
ctest --test-dir /absolute/path/to/controls/batched/build --output-on-failure
```

Use `inline` for the control and repeat with the fdatasync SDK and expectation
for the optional SQLite build. Add tools/FUSE targets for mounted workloads;
Python validation uses isolated, non-editable wheels. The preparer checks its
inputs and reproduces all 24 tested source/include/test files. The source
archive and both compiled components match those files byte for byte.

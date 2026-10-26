# Strict background checkpoint evaluation

**Do not adopt this candidate as the production strict policy.** Reusing the
fsync-mode worker preserves FULL commits and passes correctness checks, but it
adds checkpoint syncs to small operations and regresses several measured
workloads. The implementation and expanded tests remain in
[background.patch](background.patch), applied only by the isolated
[preparer](prepare.py). Production strict mode retains its 4,096-page automatic
checkpoint. The installed component and shared SQLite SDK are unchanged.

## Candidate and validation

The baseline is `7ca27ab099f92b719f899c97b7fa4e2a96e7786e`. The candidate
disables the automatic callback and gives strict connections the existing
worker: wake at 16 MiB of committed WAL, admission at 64 MiB and retained WAL
allocation of 16 MiB. Only fsync-mode writers select NORMAL; strict writers
remain FULL. This comparison therefore includes the scheduling, admission and
retained-allocation changes together, rather than isolating any one of them.

Both SQLite 3.53.2 SDKs from the preceding
[build-option validation](../strict_sync/README.md) are reused without changes.
Each before/after pair links the same SDK. The default SDK uses `fsync`; the
optional SDK uses `fdatasync`. Both candidates are built and non-editably
installed from a source archive in separate directories.

Each candidate passes **132 related Python tests and six C++ tests in Release**,
plus the same six native tests under ASan/UBSan. Checkpoint tests now exercise
both durability modes and mixed-mode writers: progress, pinned readers at two
page sizes, admission before mutation, background errors, startup failure,
close retries, slow storage, fork and crash recovery. A new test pauses database
copying and checks that strict writes still synchronize the WAL before returning,
that WAL sync failures roll back, and that retry reaches storage. The strict
crash case kills the process during paused copying without issuing an additional
Sync after the acknowledged write. These are syscall/VFS and process-crash
checks, not a physical power-cut test.

The new foreground-copy assertion fails against the old core, as intended.
The final preparer reproduces all 23 source/include/test files from the tested
candidate. Its separately built checkpoint test also passes. No full base suite
is run, and no sanitizer/functional test overlaps any measurement window.

## Plain measurements

The host is Linux `6.11.0-24-generic`, ext4 on a Fanxiang S103Pro SATA SSD,
using the existing write-back cache, `mq-deadline` and 2,000 us WBT setting.
Approximately 104 GB is free. Device and kernel settings are unchanged.

For each SQLite SDK, six rounds alternate inline/background order, giving
**24 fresh strict FUSE mounts**. Every mount runs 64 and 256 MiB sequential
writes including fsync and close, a separate warmed 64 MiB read fixture,
4,096 seeded 4 KiB reads, 512 seeded 4 KiB overwrites plus fsync, 128 small
file create/write/fsync/close operations, and 128 operations per metadata
phase. Preparation and content validation are outside phase timers. Each SDK
window has a 20-second baseline and one-second host monitoring. All hashes,
read slices, format-2/4,096-byte BLOB checks and SQLite quick checks pass.

Medians across all six mounts are shown below. No samples are excluded.

| Workload | fsync inline | fsync background | fdatasync inline | fdatasync background |
| --- | ---: | ---: | ---: | ---: |
| 64 MiB write + fsync, MiB/s | 52.41 | 15.61 | 60.32 | 37.15 |
| 256 MiB write + fsync, MiB/s | 36.68 | 56.64 | 57.24 | 61.36 |
| Warm sequential read, MiB/s | 529.08 | 526.31 | 535.04 | 530.73 |
| Warm 4 KiB random read, MiB/s | 69.43 | 67.92 | 68.73 | 67.59 |
| 4 KiB overwrite + final fsync, MiB/s | 1.45 | 0.28 | 2.39 | 1.86 |
| Small file + fsync, files/s | 176.22 | 178.79 | 286.50 | 298.06 |
| mkdir, operations/s | 326.44 | 220.98 | 552.56 | 260.58 |
| rename, operations/s | 336.34 | 226.96 | 577.89 | 281.99 |
| rmdir, operations/s | 201.08 | 131.94 | 364.19 | 152.87 |

The inline fsync 256 MiB result is bimodal, so its median ratio is not a stable
54% speedup. The candidate's 64 MiB writes are also bimodal: default-SDK rates
range from 10.83 to 57.46 MiB/s. Their pooled application-write p95 rises from
85.05 to 236.52 ms, and the maximum from 118.72 to 1,212.03 ms. Sampled WAL
allocation rises from about 17.15 to 63.15 MiB. Samples after application writes
may miss peaks within split FUSE requests.

Small-file medians also hide an optional-SDK slow run at 20.68 files/s. Total
work divided by total phase time falls from 286.91 to 92.44 files/s for that
pair. The corresponding default-SDK random-write rate falls from 0.84 to
0.37 MiB/s. Optional-SDK random writes instead improve on that aggregate
(0.71 to 0.99 MiB/s) while their median declines, because the control also has
slow samples. All vectors, pooled latency statistics and combined rates are in
[results.json](results.json). These finite shared-host windows do not establish
a tail-latency distribution or attribute every individual slow call.

## Why direct worker reuse regresses metadata

Four additional instrumented mounts cover each SDK/policy pair with the same
workload. Their rates are diagnostic only. The unchanged
[timeline wrapper](../staged_payload/block_sync/timeline.cpp) records actual
WAL/database syncs, checkpoint calls, allocation and whole-device counters;
a nominal 2 ms sampler records each mount thread's waits. Extra writeback
waiting remains disabled.

For 128 rename operations, both inline controls perform 128 foreground WAL
syncs and no database-file syncs in the phase. Both candidates retain those
128 foreground WAL syncs and add **128 background database-file syncs**. Each
candidate has 127 complete passive checkpoint calls inside the phase. All
return with copied frames equal to total frames, yet successive calls usually
advance only **eight frames**, about 32 KiB with the default page size. Their
total frame indices range from 8,765 to 10,027, well above the wake threshold.

The [worker](../../src/checkpoint.hpp) wakes on total committed WAL frames and
immediately services each request. That total remains large even after most
frames have been backfilled. Reusing it for the strict workload therefore
creates frequent small checkpoints, adding database syncs alongside the
unavoidable foreground FULL syncs. The trace confirms this mechanism; it does
not imply that every background checkpoint is unnecessary in every workload.
Counts include only complete intervals inside a phase, so a checkpoint crossing
a phase boundary can contribute a sync without a counted checkpoint call.

The candidate also reproduces a slow 64 MiB write with the optional SDK:
5.947 seconds total, including 5.251 seconds in foreground WAL syncs. Its
longest WAL sync takes 1.090 seconds with journal-commit waits. Moving database
copying onto another thread does not remove foreground storage waits. Background
and foreground durations overlap and must not be added as disjoint phase time;
whole-device counters cannot assign every request or identify SSD-internal
causes. The trace does not assign causes to each earlier plain sample.

The next bounded experiment should schedule batches using **WAL frames still
to backfill**, with explicit handling for WAL reset, idle work, errors and
admission pressure. It must retain every strict FULL commit and verify that
small-operation checkpoint frequency falls before repeating the performance
comparison. This report does not implement or claim a benefit for that policy.

## Reproduce and retained evidence

Use this commit's production sources, a matching static SQLite SDK, Linux,
CMake, Ninja, a C++17 compiler and `patch`. The output directory must not exist:

```bash
python3 vane_fs/benchmarks/strict_checkpoint/prepare.py /absolute/path/to/controls
export CMAKE_PREFIX_PATH=/absolute/path/to/sqlite-sdk
cmake -S /absolute/path/to/controls/background \
  -B /absolute/path/to/controls/background/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DVANE_FS_BUILD_PYTHON=OFF \
  -DVANE_FS_BUILD_TESTS=ON -DVANE_FS_TEST_SQLITE_SYNC=fsync
cmake --build /absolute/path/to/controls/background/build -j 2
ctest --test-dir /absolute/path/to/controls/background/build --output-on-failure
```

Use the `inline` directory for its matching control. Select the fdatasync SDK
and expectation in another fresh build for the optional variant. Add tools and
FUSE targets for mounted workloads; use a separate `SKBUILD_BUILD_DIR` and
non-editable installation for Python. The frozen runners record exact commands.

[Diagnostics](diagnostics.json) retains phase-level sync and checkpoint counts.
The four plain-run files retain every mount and timing vector; the
[artifact manifest](artifact-manifest.json) identifies complete traces, thread
waits, host monitors, source snapshots, SDK/binary identities, test/build logs,
the expected old-core failure, reproduction checks and cleanup records.
All 32 recorded test/measurement temporary roots are removed after shutdown,
including all 28 measured mounts. No owned run processes or mounts remain.
The production source and installed-module hashes match the baseline again;
only the isolated experimental artifacts contain the candidate implementation.

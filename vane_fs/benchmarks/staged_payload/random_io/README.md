# Transaction-control cache experiment

This isolated C++ candidate extends [selective publication](../conditional/README.md).
It reuses the connection's existing prepared-statement cache for `BEGIN`,
`BEGIN IMMEDIATE` and `COMMIT`. Transaction boundaries, rollback, FULL barriers,
publication ordering, GC leases and checkpoint scheduling remain unchanged.
Production sources and the installed package are unchanged.

## Diagnosis

Each native 4 KiB read still used two `sqlite3_exec` calls for transaction control.
SQLite documents that [exec](https://www.sqlite.org/c3ref/exec.html) wraps
prepare, step and finalize. The immediate-consumption diagnostic records 8,192
calls per 4,096 reads, with about 14 ms of exclusive scoped time for the selective
candidate. The cache removes those calls and reuses the fixed statements.
It does not remove the SQL transaction itself. Failed statements still use the
existing reset/finalize handling, and exception unwinding retains nonthrowing
rollback.

The diagnostic also distinguishes two measurement effects from production costs:

- The preceding native probe classified file descriptors with `readlink` before
  each payload read. That cost was inside the overall timer but outside its
  payload syscall timer. The new scoped probe needs no descriptor classification
  on the read path. Uninstrumented FUSE measurements never used that wrapper.
- Retaining every returned string changed allocator behavior. Consuming,
  verifying and freeing each buffer immediately models the FUSE callback more
  closely. The formerly large allocation gap largely disappears. FUSE already
  replies directly from that buffer; no new read-buffer API is introduced.

External payload reads still acquire a shared GC lease and check process identity.
For 4,096 reads, the selective and cached candidates record 4,096 lock/unlock
pairs and 12,288 process checks, versus 4,096 process checks for SQLite BLOBs.
Those protections remain. The cache is also applicable to the SQLite-BLOB path;
this experiment does not establish an intrinsic advantage of external storage.

Scoped times include instrumentation overhead. Inclusive and exclusive times
are retained separately, allocation counts cover C++ new/delete only, and the
single-caller profiler is not used in the multithreaded FUSE daemon. Hardware and
software perf events were denied by the host's `perf_event_paranoid=4`; no kernel
settings were changed. Plain binaries supply the throughput comparisons.

## Measurements and decision

The uninstrumented FUSE window uses six balanced fresh-workspace orders and
three rotating aged-workspace orders, each with a separate database and mount.
The fsync-mode workload and warmed O_RDWR read protocol match the preceding
Vane-only matrix: 1 MiB application writes, 64 MiB then 256 MiB sequential files,
a warmed 64 MiB read fixture, 4,096 aligned random reads and 512 aligned 4 KiB
overwrites. Aged workspaces retain a 32 MiB guard, remove alternate blocks and
run GC. Small-file and namespace items each perform 128 operations. Release
builds run on the shared Linux/ext4 SATA SSD without global cache eviction.
No owned builds or correctness tests overlap the measurement window.

Medians, including every sample:

| Workload | Production BLOB | Selective | Cached controls |
| --- | ---: | ---: | ---: |
| Fresh 64 MiB write + fsync/close, MiB/s | 82.98 | 152.34 | 133.00 |
| Subsequent 256 MiB write + fsync/close, MiB/s | 22.38 | 129.77 | 133.54 |
| Warm sequential read, MiB/s | 549.21 | 803.38 | 799.36 |
| Warm 4 KiB random read, MiB/s | 66.23 | 62.84 | 65.71 |
| 4 KiB overwrite + final fsync, MiB/s | 12.45 | 11.48 | 12.91 |
| Aged 4 KiB random read, MiB/s | 67.62 | 61.42 | 63.76 |
| Aged 4 KiB overwrite + final fsync, MiB/s | 12.40 | 13.55 | 13.53 |
| Aged overwrite requests before fsync, MiB/s | 17.61 | 14.00 | 13.82 |
| Aged 4 KiB file + fsync, files/s | 264.57 | 233.58 | 269.63 |
| Aged mkdir + directory fsync, ops/s | 2,136.99 | 1,844.74 | 1,743.15 |
| Aged rename + directory fsync, ops/s | 1,440.83 | 2,143.13 | 1,903.65 |
| Aged rmdir + directory fsync, ops/s | 1,530.02 | 2,231.57 | 2,270.90 |

Compared with selective publication, fresh random-read throughput improves by
4.6% and overwrite-plus-fsync throughput by 12.4%. Fresh reads are within 0.8%
of production. Aged random reads improve by 3.8% over selective, but remain 5.7%
below production. The immediate-consumption native control also shows lower
plain-binary read times with the cache; instrumented figures are not used as
throughput results.

Aged overwrite totals conceal remaining request cost: cached throughput before
the final fsync is 13.82 MiB/s versus production's 17.61, a 21.5% deficit. The
respective final-fsync medians are 6.83 and 47.66 ms. These phase medians describe
their own sample distributions and must not be added to reconstruct a median
whole operation. Request latency vectors are retained. Aged mkdir including its
barrier is also 18.4% below production. This experiment has not isolated every
remaining FUSE or aged-layout cost.

Large-write and barrier times remain unstable. Production's six 256 MiB samples
range from 9.32 to 94.54 MiB/s, selective from 22.25 to 147.21, and cached from
28.27 to 147.64. Their medians do not establish a stable large-write gain from
transaction-control caching. An unchanged production build measured 93.59 MiB/s
in the preceding fresh-workspace window; those windows remain separate. One
selective random-overwrite item spends 147 ms issuing requests and 2.85 seconds
in its final fsync. Cached small-file samples range from 19.58 to 284.21 files/s.
All slow samples remain in the report. Host-wide CPU busy time has an 11.7%
median and an 84.2% maximum; three of 472 sampled intervals exceed 50%, overlapping
selective 256 MiB items. These counters include preparation and cleanup as well
as measured work. They limit causal interpretation of small timing differences
and do not justify discarding samples or attributing contention to a specific job.

**The candidate remains experimental.** Reusing fixed SQL removes verified
compilation work and narrows the read gap, but the aged random-read and
request-only overwrite criteria still fail. Synchronization tails also remain
unresolved. The next focused investigation is the aged 4 KiB FUSE write request
path, with per-request processing separated from durability barriers. Applying
transaction-control caching to the default SQLite-BLOB path would be a separate
comparison; external-format results do not substitute for it.


## Synchronization diagnosis

A separate diagnostic runs three rotating orders for each of three states:
a fresh database, a retained 32 MiB guard with alternate blocks removed followed
by GC, and a long-lived workspace after four 64 MiB write/sync/unlink cycles.
Each measured item writes 64 MiB and calls exactly `Workspace::Sync()`; close and
unlink are outside its timers. These native instrumented results are separate
from FUSE throughput.

All 27 cases pass content verification and SQLite quick_check. The selective
candidate includes a 2.90-second payload fdatasync during Sync, and another
2.23-second payload fdatasync during the write phase. The production path
includes a 0.91-second WAL fsync and overlapping background checkpoint work.
Owned-thread samples during those events observe `rq_qos_wait`,
`folio_wait_bit_common` and `jbd2_log_wait_commit`. Event intervals and sampled
wait locations are retained; overlapping events must not be added as exclusive
wall time. Sampling identifies waiting inside persistence operations, not the
underlying device-level cause or responsibility of another job.

The cached candidate's nine cases have total write-plus-Sync times between
0.31 and 0.58 seconds in this diagnostic window. That observation is insufficient
to claim that the tails are fixed: its persistence path is unchanged, previous
windows contain stalls, and the sample is small. Cached and selective fresh-state
Sync medians are almost identical, about 176 and 174 ms respectively.

## Validation

The candidate passes 144 related Python tests with one explicit skip for the
unsupported production-v1 migration. All ten native tests pass in Release and
ASan/UBSan. Existing recovery, publication, checkpoint, GC, fork and FUSE tests
remain included, including 33 deterministic staging crash/loss cases. These
software fault tests are not physical power-cut certification.

The new native regression checks that warm read/write transactions do not
recompile control statements. It injects a failed COMMIT, verifies rollback,
exercises a real SQLITE_BUSY at BEGIN, and checks retry and FULL persistence
through reopening. The preceding selective candidate fails the cache assertion;
the new candidate passes. All 39 prepared source files match the tested copy.
Only related tests are run.

## Reproduce

```bash
.venv/bin/python vane_fs/benchmarks/staged_payload/random_io/prepare.py /tmp/vane-control-cache
cmake -S /tmp/vane-control-cache/cached -B /tmp/vane-control-cache/cached/build \
  -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/vane_fs/vcpkg_installed/x64-linux-release" \
  -DVANE_FS_BUILD_PYTHON=OFF -DVANE_FS_BUILD_TESTS=ON
cmake --build /tmp/vane-control-cache/cached/build --parallel 2
ctest --test-dir /tmp/vane-control-cache/cached/build --output-on-failure
```

The preparer inherits the preceding source-identity checks and adds `control.patch`
in a separate copy. Component wheels use persistent build directories and isolated
non-editable installations. The generated source tree matches the tested copy.

## Retained evidence

[results.json](results.json) contains every scalar result, request-latency
summaries, native profiles and synchronization diagnostics. The
[artifact manifest](artifact-manifest.json) identifies full request/event traces,
test logs, frozen sources, binaries, runners, host monitoring and cleanup records
retained locally. Slow samples are retained. Owned workspaces are removed after
their processes and mounts stop, outside measurements; the production package
and other workloads are left intact.

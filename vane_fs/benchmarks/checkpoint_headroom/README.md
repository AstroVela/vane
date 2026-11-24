# Allocation and fsync checkpoint evaluation: 2026-10-10

**Neither worker candidate is adopted.** Allocation hints do not improve the
native strict diagnostic. Fsync worker batching removes many small background
checkpoints, but shifts work into foreground barriers and retains large-write
and metadata regressions. Earlier maintenance near the admission budget does
not repair the complete workload. Production sources, the shared SQLite SDK
and installed component remain unchanged.

The baseline is `c6b7d5c0df5c319dbda7872a2e7fcda7e1bd55fc`. This follows the
[WAL capacity experiment](../strict_checkpoint/wal_reuse/README.md), which
already demonstrated that avoiding WAL extension does not eliminate every
fixed-size synchronization stall.

## Allocation probes

Two independent native windows retain six balanced rounds per setting,
18 disposable databases each. They use the unchanged strict core, the original
SQLite SDK, 4 KiB payloads/pages and 4,096-page automatic checkpoints. Split
1 MiB writes reproduce the existing FUSE request sizes. Timings are diagnostics,
not plain FUSE throughput.

One probe sets the documented database `SQLITE_FCNTL_CHUNK_SIZE` to 1 or 4 MiB;
SQLite's checkpoint size hint supplies the required size. The second uses a
private diagnostic VFS to issue `CHUNK_SIZE`/`SIZE_HINT` on a growing WAL when
it first reaches the selected chunk threshold. No custom VFS is integrated.

| Native probe / chunk | Write median, s | Cycle median, s | Cycle maximum, s |
| --- | ---: | ---: | ---: |
| Database / unchanged | 1.155 | 1.245 | 1.287 |
| Database / 1 MiB | 1.234 | 1.353 | 1.687 |
| Database / 4 MiB | 1.237 | 1.350 | 6.908 |
| WAL / unchanged | 1.165 | 1.273 | 1.304 |
| WAL / 1 MiB | 1.277 | 1.387 | 7.532 |
| WAL / 4 MiB | 1.247 | 1.360 | 1.396 |

Database-probe cycles include write, explicit sync, checkpoint and close;
WAL-probe cycles additionally include initialization. Compare settings within
their own row group. Database file sizes rise from 80,302,080 bytes unchanged
to 80,740,352/83,886,080 with the two chunk sizes. WAL hints issue 17/4 reserve
calls during each write phase. Neither probe supports adopting allocation hints.
[Results](results.json) retain every diagnostic timing and cleanup record; full
SQL/VFS vectors and runners are indexed by the artifact manifest.

## Worker candidates and tests

The [batched patch](batched.patch) applies the preceding batching idea only to
the existing **fsync-mode worker**. Before starting new work, it queries current
WAL progress with `SQLITE_CHECKPOINT_NOOP` and requires 16 MiB of unbackfilled
frames. It does not infer the generation from cached frame counts. Once a
checkpoint starts, partial/error retries continue below the batch threshold,
including after a joined Stop/Start without new commits.

The [pressure patch](pressure.patch) additionally bypasses that batching gate
when total committed frames reach 48 MiB, leaving one batch before the existing
64 MiB admission boundary. Both retain the 16 MiB allocation limit, writer FULL
barriers, reader backpressure and error handling. Strict connections retain
their original inline automatic checkpoint policy; neither candidate makes
strict commits asynchronous. Format 2 and 4,096-byte SQLite BLOBs are unchanged.

Each isolated, non-editable Release component passes **132 related Python
tests and seven C++ tests**. The original six-suite baseline also passes. The
new wrapped-checkpoint regression covers 4/8 KiB pages, FULL/NORMAL writers,
small additions to a retained generation, partial idle retry, Stop/Start, WAL
reset and injected status-probe errors. The pressure version adds a reader
pinned before backfill completes, ensuring sub-budget additions near admission
trigger maintenance. The production worker fails the small-addition assertion;
the first candidate fails the pressure assertion, as expected.

The pressure fixture initially opened its reader after complete backfill;
SQLite correctly chose a database-only read and allowed WAL reset. That test
failure, its exact source and logs are retained separately. Pinning before the
partial checkpoint finishes repairs the fixture; all seven final suites pass.
Existing component tests cover explicit barriers, write admission, mixed
writers, failure/close retries, snapshots, GC, fork and process-crash recovery.
Local ASan/UBSan is not rerun for these isolated experiments. Only related
tests run.

## Plain measurement method

Each candidate has its own contemporaneous control window: six alternating
baseline/candidate pairs on fresh workspaces and six on independent fragmented
workspaces. This is **48 plain mounts**, plus eight separate instrumented mounts.
All use VaneFS `fsync` durability and the **original SDK's `fsync` primitive**.
Do not pool controls between the two windows or compare their medians as if
host conditions were identical.

Each mount writes 64 and 256 MiB, reads a separate warmed 64 MiB fixture,
performs 4,096 seeded 4 KiB reads and 512 seeded overwrites, creates 128 small
files with per-file fsync, then performs 128 mkdir/rename/rmdir operations with
a final directory fsync per phase. Large writes include creation, application
fsync and close. Requested bytes and content hashes are verified outside those
timers. Overwrites include the final fsync. Metadata operation latency vectors
exclude the final directory barrier, whose duration is recorded separately.

The fragmented fixture writes a 32 MiB guard, zeroes alternate 4 KiB blocks,
collects garbage through an independent strict workspace and retains the guard
while allocating measured files. All 24 plain aged workspaces have **4,128
free pages after GC**. Guard hashes before/after subsequent workloads agree.
The fixture deliberately stresses free-page reuse; it does not establish how
common that fragmentation is in production.

Each window has a 20-second host baseline and one-second telemetry. No owned
builds/tests overlap measurement. Kernel caches remain enabled; no host, device,
scheduler or cache setting changes. Every sample, including slow runs, remains.
Pooled latency uses nearest-rank percentiles; combined rate is total work
divided by total elapsed time. Six runs do not establish a stable tail distribution.

## Plain results

Medians across six mounts; write/read rates are MiB/s, small-file rates are
files/s and metadata rates are operations/s. Each control belongs to its
candidate's window.

| Batched window / workload | Fresh control | Fresh batched | Aged control | Aged batched |
| --- | ---: | ---: | ---: | ---: |
| 64 MiB write + fsync + close | 56.37 | 99.90 | 94.77 | 91.24 |
| 256 MiB write + fsync + close | 95.56 | 94.36 | 96.02 | 61.61 |
| Warm sequential read | 520.27 | 538.06 | 509.87 | 507.49 |
| Warm 4 KiB random read | 65.02 | 67.02 | 67.88 | 67.44 |
| 4 KiB overwrite + final fsync | 10.23 | 10.29 | 1.47 | 9.96 |
| Small file + fsync | 144.96 | 259.92 | 106.40 | 243.77 |
| mkdir + final directory fsync | 1,914.26 | 865.08 | 2,108.03 | 672.77 |
| rename + final directory fsync | 2,299.88 | 1,539.28 | 2,447.91 | 1,481.45 |
| rmdir + final directory fsync | 2,240.00 | 1,483.64 | 2,455.82 | 1,577.66 |

The initial batching candidate improves several observed small-write rates,
but aged 256 MiB **combined** throughput falls from 96.06 to 38.35 MiB/s.
Its longest aged 1 MiB write request is 6.773 seconds, versus 0.325 seconds in
that window's control. Fresh 256 MiB combined throughput also falls,
95.30 to 70.30 MiB/s. First-write medians alone would conceal these regressions.

| Pressure window / workload | Fresh control | Fresh pressure | Aged control | Aged pressure |
| --- | ---: | ---: | ---: | ---: |
| 64 MiB write + fsync + close | 98.00 | 98.52 | 95.88 | 92.41 |
| 256 MiB write + fsync + close | 92.92 | 31.51 | 94.81 | 94.18 |
| Warm sequential read | 524.61 | 600.60 | 520.10 | 510.92 |
| Warm 4 KiB random read | 65.68 | 66.69 | 66.71 | 65.95 |
| 4 KiB overwrite + final fsync | 14.42 | 11.38 | 12.30 | 11.57 |
| Small file + fsync | 250.06 | 237.04 | 251.45 | 20.88 |
| mkdir + final directory fsync | 1,932.00 | 1,427.57 | 1,953.62 | 745.39 |
| rename + final directory fsync | 2,268.51 | 1,395.21 | 2,335.21 | 1,055.97 |
| rmdir + final directory fsync | 2,326.69 | 1,386.60 | 2,156.89 | 999.13 |

The pressure candidate's fresh 256 MiB rates are 94.54, 33.59, 93.75, 29.44,
29.12 and 16.92 MiB/s. Combined throughput falls from 52.01 to 33.63 MiB/s;
the corresponding longest request grows from 5.086 to 6.940 seconds. Aged
small-file combined throughput falls from 59.53 to 23.89 files/s, and pooled
per-file p95 increases from 81.88 to 114.84 ms. Its aged 64 MiB combined rate
also falls, 96.49 to 36.30 MiB/s. Both controls retain slow samples; these
results do not establish a universal causal effect on individual storage stalls.

## Checkpoint and barrier diagnosis

Linker wrappers record actual sync/checkpoint events on the unchanged production
FUSE path. Counts below include calls wholly contained within a phase;
boundary-crossing syncs are recorded separately. Durations on different threads
overlap and must not be added to infer wall time. Instrumented rates are not
used as plain performance results.

During the first window's fresh mkdir/rename/rmdir phases, the control completes
**39 background PASSIVE checkpoints**, versus zero in the batched candidate.
However, the candidate performs two foreground RESTART checkpoints during mkdir;
its foreground WAL synchronization totals 25.08 ms versus 4.60 ms. In the plain
aged runs, median final directory-barrier times for mkdir/rename/rmdir rise from
4.01/3.12/4.03 ms to 21.12/13.58/12.83 ms. Removing maintenance calls does not
remove the barrier's remaining work.

The pressure diagnostic still records two foreground RESTART calls during
mkdir in both fresh and aged candidate mounts. Its aged small-file phase makes
68 background PASSIVE calls versus five in its control: bypassing batching near
admission can reintroduce frequent maintenance. Its aged 256 MiB write records
4.339 seconds of background WAL sync and 3.375 seconds of background database
sync, versus 0.844/0.702 seconds in its control. These scopes overlap their
checkpoint calls and foreground admission waits. Moving I/O into a worker does
not remove a writer's wait for that worker under pressure.

All instrumented small-file phases retain 128 foreground WAL syncs. Explicit
FULL barriers remain observable; neither candidate obtains its result by
omitting fsync. Whole-device telemetry cannot attribute every request or
identify SSD-internal causes. These observations explain where the costs occur,
not the cause of every shared-host long stall.

## Lifecycle, space and retained evidence

Lifecycle medians include fixtures, validation, deletion and shutdown; aged
ones additionally include fragmentation preparation and GC. Post-close SQLite
inspection and owned-data cleanup follow the lifecycle timer. Stage medians
must not be summed to reconstruct a directly measured lifecycle.

| Window / group | Control lifecycle, s | Candidate lifecycle, s |
| --- | ---: | ---: |
| Batched / fresh | 17.56 | 18.41 |
| Batched / aged | 23.71 | 20.22 |
| Pressure / fresh | 16.31 | 17.82 |
| Pressure / aged | 20.74 | 23.92 |

All mounts pass content checks, requested-slice comparisons, SQLite quick
checks and format-2/4,096-byte payload validation. Timed fsync/close durations,
large-write WAL samples, fresh initialization/mount/shutdown allocation and
post-GC free pages remain in the eight per-window/group/variant JSON files.
Logical/allocated samples are observations at specified times, not instantaneous
peaks. The worker policies introduce no new preallocation or persistent format.

[results.json](results.json) links every raw plain vector and both complete
summaries. The [artifact manifest](artifact-manifest.json) retains full timelines,
host telemetry, SQL/VFS allocation probes, exact SDK/source/binary identities,
build/test logs, the failed pressure fixture and frozen runners. All recorded
owned run/test data is removed after its processes stop, outside measured
comparison windows. Free disk space is checked before every mount.

This evaluation does not support further increases to thresholds or retained
capacity as a production optimization. A subsequent storage-path candidate
needs to reduce actual foreground synchronization or data write amplification,
and pass fresh, fragmented, GC, barrier and complete-lifecycle comparisons.

## Reproduction

The [preparer](prepare.py) reads the recorded baseline through `git archive`,
applies the selected patch to an isolated copy and checks all 29 baseline and
30 candidate source/build-input hashes. It works after the production checkout
advances, provided the recorded commit is available locally:

```bash
python3 vane_fs/benchmarks/checkpoint_headroom/prepare.py vane_fs/build/reproduce-batching --variant batched
python3 vane_fs/benchmarks/checkpoint_headroom/prepare.py vane_fs/build/reproduce-pressure --variant pressure
```

Build each copied component into its own `build/python-release` with Release,
`VANE_FS_BUILD_PYTHON`, `VANE_FS_BUILD_TOOLS`, `VANE_FS_BUILD_FUSE` and
`VANE_FS_BUILD_TESTS` enabled. Use the original SQLite prefix, a libfuse3 SDK,
and a non-editable install into an isolated environment. Run its seven CTests
and copied component Python tests with that mount binary. Exact commands,
dependency/environment identities and frozen `measure.py`/`workload.py` runners
remain in the artifact roots listed in the manifest. The two windows must
remain separate; neither optional SQLite SDK nor strict-mode worker experiment
is used in these FUSE measurements.

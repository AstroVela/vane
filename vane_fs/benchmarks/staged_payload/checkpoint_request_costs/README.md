# FUSE request costs after WAL reuse

This report diagnoses the random-write, small-file fsync and rename regressions
in the [commit-progress experiment](../checkpoint_progress/README.md). It adds
instrumentation and analysis, with **no production runtime change or adoption**.
The preceding plain ext4 screen still fails six original checks. These new
instrumented windows cannot replace that decision.

## Two independent windows

Each window has 16 ext4 mounts: four alternating AB/BA pairs for both fresh and
independently prepared post-GC workspaces. The original workload, application
buffers, page offset 48, final barriers, content checks and indexed format 1004
remain unchanged. Both variants use the pinned SQLite 3.53.2 SDK with its
original `fsync` primitive and FUSE 3.14. Each GC run uses its own verified native
Python module. There are no hints, global cache eviction, frequency/affinity
changes, or concurrent owned builds/tests inside either window.

- **Original-control window:** the original partial-write control versus the
  latest orphan-batching + optional RESTART + commit-wake latch prototype. This
  is the same control used by the preceding failed plain FUSE screen.
- **Wake-latch causal window:** the preceding RESTART prototype versus the
  latest candidate. The only source difference is `src/checkpoint.hpp`;
  reversing `progress.patch` with fuzz zero recovers the preceding source
  exactly. The recorder sources are byte-identical. Existing verified diagnostic
  binaries are reused, with independent native bindings for GC.

Instrumentation records complete FUSE callbacks, thread CPU, SQLite VM steps,
transaction/lock stages, WAL generation resets, and individual typed syncs.
Read-only 5 ms CPU samples provide scheduler accounting and frequency estimates.
Removing only the generator's diagnostic additions recovers every original
source byte exactly. Source, SDK, module and binary identities are checked.
Nested stage durations and overlapping background syncs are not additive.

## Small-file fsync: excessive background database synchronization

The application creates 128 files, writes 4 KiB, fsyncs and closes each one.
Both variants issue 128 application WAL syncs. The primary fsync handlers execute
exactly 11,776 SQLite VM steps in each batch; extra FUSE requests do not explain
the regression. In the causal window, retaining the wake latch causes many more
fully completed small PASSIVE checkpoints:

| Causal window | Old restart | Wake latch |
| --- | ---: | ---: |
| Fresh batch, median ms | 532.75 | 622.94 (+16.9%) |
| Fresh background database syncs, median | 1 | 107 |
| Fresh application WAL sync time, median ms | 315.71 | 379.93 |
| Post-GC batch, median ms | 509.75 | 611.80 (+20.0%) |
| Post-GC background database syncs, median | 3.5 | 104.5 |
| Post-GC application WAL sync time, median ms | 297.95 | 374.02 |

The background database-sync inclusive totals also rise from 12.47 to 313.18 ms
fresh and 25.24 to 292.16 ms post-GC. These totals overlap application work and
cannot be summed into elapsed time. The application sync count does not rise;
the worker adds database synchronization beside already durable FULL commits.
[SQLite documents](https://www.sqlite.org/c3ref/wal_checkpoint_v2.html) that a
PASSIVE checkpoint syncs the database when it copies all frames.

The original-control window independently shows median background database
sync counts of 1→100 fresh and 2→104.5 post-GC. This repeated count difference
and the single-source-change causal window support the wake policy as a
specific cause of the small-file penalty. They do not prove that every remaining
tail comes from that policy.

## Rename: lookup acquires the same writer lock as RESTART

Every 128-rename phase contains 128 `rename`, 128 `lookup`, 128 `getattr` and one
final directory fsync in both variants. The primary rename VM count is exactly
101,631 in both. The original-control window's candidate lookup wall median
rises from 7.38 to 16.49 ms fresh and 6.95 to 11.72 ms post-GC per batch.

All six candidate lookups with more than 0.5 ms off-CPU time last about 8.3 ms,
while consuming only 0.12–0.16 ms CPU. Their `begin` spans occupy about 8.2 ms.
Each overlaps two actual `wal_restart` sync calls. The original control has no
such slow lookup among 1,024 lookups. The causal window retains nine additional
slow lookups across its two variants; all nine also overlap RESTART syncs.
[findings.json](findings.json) retains all 15 requests and their overlapping
sync records, including timestamps, thread identities, CPU and stage costs.
The 0.5 ms cutoff classifies diagnostic waits; it is not a new acceptance gate.

The source path is `ops.lookup` → `Session::LookupInode` →
`Transaction(*database, !snapshot)` → `BEGIN IMMEDIATE`. A writable branch
acquires the writer lock before resolving the name, including names that turn
out not to exist. This workload renames to previously absent target names.
Positive lookup needs atomic inode pinning, so it cannot simply become read-only.

RESTART obtains the SQLite writer lock and can perform real synchronization
while holding it. Setting the worker's busy timeout to zero disables its own
lock retries; it does not bound sync duration or make the writer lock available
to the foreground connection. This interpretation follows the recorded BEGIN
waits, identical request/VM counts and the
[checkpoint lock contract](https://www.sqlite.org/c3ref/wal_checkpoint_v2.html).
The trace does not count individual busy-handler sleeps.

Removing the wake latch would lose useful progress. In the causal window,
the final directory fsync median improves from 12.28 to 2.90 ms fresh and
8.17 to 2.97 ms post-GC with the latch. Complete rename phases improve from
91.13 to 68.00 ms and 82.73 to 71.20 ms. The latch mitigates an existing
RESTART/progress regression while adding the small-file cost above.

## Random write: both CPU variation and SQLite waits remain

Both variants issue 512 writes and one final fsync and execute exactly 224,448
VM steps in write handlers. The original-control window's median batch time
rises from 173 to 214 ms fresh and 199 to 241 ms post-GC. Fresh write-handler CPU
increases from 124.89 to 148.05 ms; sampled frequency falls from about 1.61 to
1.40 GHz. Post-GC BEGIN time increases from 6.69 to 40.95 ms, alongside write
off-CPU time increasing from 3.50 to 39.22 ms. Background WAL sync medians grow
from one to 35 fresh and one to 53.5 post-GC.

Median cumulative WAL-hook time is below 0.5 ms per complete batch, and median
write admission is below 1 ms. This rules out treating the atomic boolean
or the 64 MiB admission path as the dominant recorded cost. Frequencies are
sampled estimates rather than cycle counts; their correlation does not prove
that CPU frequency caused the full difference or that another workload did.

The causal window improves complete random-write medians from 264 to 237 ms
fresh and 240 to 193 ms post-GC with the latch, while reducing final fsync from
about 31–33 ms to 3–4 ms. The entire original-control regression therefore cannot
be assigned to the latch. The recorded SQL waits, completed-checkpoint frequency
and CPU behavior need to be treated separately.

## Next implementation and its checks

First test a read-only path for **missing inode lookups**, retaining write
transactions and fresh validation for successful atomic pins. Check concurrent
create/rename/unlink, directory inode identity, snapshots and open handles;
measure successful lookups as well as misses so a read-first probe does not
silently double the cost of positive lookups.

Then test coalescing of small CLEAN maintenance while retaining a fixed upper
bound on scheduling deferral, progress after the last commit, pinned-reader
retry and Stop/Start recovery. Keep OPEN publication, FULL durability and the
16/64 MiB admission thresholds intact. Do not simply turn off the wake latch.
The earlier [global 2 ms scheduling experiment](../checkpoint_batching/README.md)
was rejected on its different base; its results are not approval to repeat that
policy unchanged. The new target is the repeated completed small checkpoints
identified here.

Each change needs an independent ablation and the original plain workload and
gates. A lower diagnostic sync count alone is insufficient for adoption.
Protected payload publication remains a separate seconds-long tail target.

## Validation, tails and retention

The new typed recorder passes Release and ASAN/UBSAN validation: 25 sync calls,
six descriptor kinds, descriptor reuse, nested sites, concurrency and
EIO/EINTR/EBADF handling. The unchanged component/Python/restart/progress test
reports from the preceding builds are retained as reused evidence; they are not
claimed as newly rerun tests. No broad base/Ray suite runs.

All 32 mounts pass content, retained-data, indexed-schema and SQLite integrity
checks. All **15,169** sync calls reconcile with request counts and durations.
All **217** calls lasting at least 100 ms remain, including six at least one
second long. The original window retains overlapping candidate random-write
WAL syncs of 1.224 and 1.143 seconds. The causal window retains a 7.001-second
control payload sync outside measured phases, a 4.125-second candidate payload
sync inside a subsequent-write phase, and candidate random-write WAL syncs of
1.412 and 1.390 seconds. No sample is discarded or attributed to unrelated load.

The measured wrapper before formatting and the initial analysis preflights are
retained under [preflight](preflight/analysis-failures.json). Formatting leaves
the wrapper AST identical. Decoder adaptations cover copied non-native files
and the fact that both causal variants use RESTART; they do not alter runtime
logic or measurement data.

[audit.json](audit.json) verifies source/binary/module identities, lossless
report splits, both independent windows and all 32 removed owned data paths.
Large raw request traces, CPU/host samples and frozen generated code remain
locally with sizes and hashes in the audit. Database, WAL and payload temporary
data have been removed after each owned daemon stopped and unmounted, outside
the measured phases. About 90 GiB remained after the second window.
Production source hashes and the installed default module hash are unchanged.

Reproduce from the repository root with the preceding frozen build artifacts:

```bash
python3 vane_fs/benchmarks/staged_payload/checkpoint_request_costs/prepare.py \
  vane_fs/build/new-request-costs \
  --parent vane_fs/build/checkpoint-progress-20261012T153141Z/fuse-screen \
  --sqlite-prefix vane_fs/vcpkg_installed/x64-linux-release \
  --fuse-prefix .cache/fuse-sdk/root/usr
python3 vane_fs/benchmarks/staged_payload/checkpoint_request_costs/run.py \
  vane_fs/build/new-request-costs
python3 vane_fs/benchmarks/staged_payload/checkpoint_request_costs/run_causal.py \
  vane_fs/build/new-request-costs \
  --preceding-diagnostic vane_fs/build/restart-diagnostic-20261012T140610Z
python3 vane_fs/benchmarks/staged_payload/checkpoint_request_costs/analyze.py \
  vane_fs/build/new-request-costs
python3 vane_fs/benchmarks/staged_payload/checkpoint_request_costs/analyze.py \
  vane_fs/build/new-request-costs/causal
python3 vane_fs/benchmarks/staged_payload/checkpoint_request_costs/summarize.py \
  vane_fs/build/new-request-costs
```

The publication command also verifies the retained measured-wrapper archive
from this window. A new window should archive its own executed wrapper before
post-run formatting, rather than substitute this window's execution identity.

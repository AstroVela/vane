# FULL WAL synchronization with an active or quiesced worker

This investigates the WAL-sync cost observed by the
[pre-OPEN experiment](../pre_open_backfill/README.md). It adds a native C++
diagnostic and **does not change production scheduling**. Two independent
windows do not establish that stopping the worker is a general performance
improvement. Foreground FULL synchronization remains the main measured cost.

## Controlled difference

Both arms link the same frozen, instrumented format-1004 core and the same
verified typed-sync recorder. A link wrapper observes the registered
`Checkpointer` context while preserving the original WAL callback and context.
After common setup finishes backfilling, both arms call `Stop()` and check for
errors. The active arm calls `Start()` before measurement; the quiesced arm
leaves the worker stopped. Commit notifications, FULL barriers, the 64 MiB
admission check, data format and all file operations remain the same.

Each native phase performs 128 sequences corresponding to FUSE create, 4 KiB
write, inode validation, FULL workspace sync, flush validation and release.
Two inode references model the open handle and retained lookup; lookup
references are released after the timed phase. The open parent inode and
exclusive mount lease remain valid. This is a core-mechanism diagnostic with
no mounted FUSE filesystem, rather than an overall FUSE throughput comparison.

The three independently prepared states are:

| State | Setup | Purpose |
| --- | --- | --- |
| Cold | Newly created workspace below 16 MiB WAL | Negative control: the worker has not been warmed |
| Warm reuse | A committed 20 MiB SQLite padding table is backfilled before measurement | Exercise persistent worker wakeups after WAL generation reuse |
| Post-GC reuse | Retain live/snapshot contents, delete an 8 MiB child branch, collect it, then warm as above | Exercise a workspace with actual version/payload reclamation |

Every post-GC run reclaims 2,065 versions, 2,048 payloads and one snapshot.
Live and retained-snapshot contents are checked. The padding table is a setup
fixture, not a claim that the file payload resides in SQLite in this format.
Both arms stay below the 64 MiB admission budget; the cold state stays below
16 MiB. Exactly 128 real foreground WAL syncs occur in every phase. Quiesced
phases and both cold arms have no background checkpoint; active warm/post-GC
phases have PASSIVE calls, with no RESTART or checkpoint error.

## Two independent windows

Each window runs six alternating AB/BA pairs per state: 36 owned processes and
workspaces, 72 total. The [screen](screen/findings.json) and
[confirmation](confirmation/findings.json) use unchanged parameters and the
same executable. The second window addresses the inconsistent first result;
it repeats the complete matrix without selecting favorable states or removing
the first window's tail.

Quiesced throughput changes against the active worker:

| State | Screen median | Screen aggregate | Confirmation median | Confirmation aggregate |
| --- | ---: | ---: | ---: | ---: |
| Cold | +3.1% | +0.5% | −3.7% | −1.7% |
| Warm reuse | +8.6% | +86.1% | +0.4% | +1.6% |
| Post-GC reuse | −3.7% | +1.6% | +4.3% | +3.0% |

The first warm aggregate is dominated by a 2.40-second active phase. Its
128 foreground WAL syncs total 2.18 seconds; its 27 worker database syncs total
658 ms, and 27 foreground WAL syncs overlap worker database sync intervals.
Request CPU totals 166 ms, BEGIN totals 6.16 ms, and publication wait 0.166 ms.
There is no single two-second syscall: repeated slower syncs accumulate across
the phase. The largest is a 176 ms foreground WAL sync. All eight measured
syncs lasting at least 100 ms remain, including the 110 ms worker database sync.

The confirmation has no measured sync at least 100 ms. Warm phase medians are
440.1 ms active versus 438.5 ms quiesced, with only three of six quiesced paired
wins. Post-GC medians are 436.9 versus 418.9 ms, but this reverses the first
window's median result. Cold results also change direction. The active arms
have 109/95 warm database syncs and 67/74 post-GC database syncs across the two
windows; quiesced arms eliminate them completely. That elimination does not
produce a consistent improvement across states and windows.

All 9,216 foreground FULL WAL syncs remain. The 128 foreground syncs collectively
account for a median 67.5% and 67.4% of phase wall time in the two windows.
Worker and foreground sync intervals overlap and must not be added together as
wall time. This supports investigating foreground synchronization itself;
it does not prove that concurrent backfill caused the first window's tail or
the preceding FUSE experiment's separate tails.

## Validation and retention

Six Release and six ASAN/UBSAN probe cases pass. They reconcile the 128 actual
FULL WAL syncs, verify no checkpoint in quiesced/cold phases, verify active
progress after warmup, and check all file contents, retained live/snapshot
data, actual GC reclamation and database integrity. The unchanged runtime's
154 Python, 12 native, 12 ASAN/UBSAN and 56 checkpoint-helper validations are
reused by exact source/library identity. They are not rerun for this diagnostic.
Only related checks run.

The SDK remains SQLite 3.53.2 with its original `fsync` primitive. Production
source and its installed module remain unchanged. All builds and tests finish
before the screen; the two measurement windows do not overlap. There is no
cache eviction, affinity/frequency tuning or sample filtering. Preparation,
validation, close and deletion are outside the small-file phase. All owned
processes exit before database, WAL, payload and sidecar removal.

[audit.json](audit.json) verifies source/library/binary identities, both windows,
all samples and 90 cleaned owned directories. Complete traces, CPU samples and
preflight binaries remain local with hashes; phase/operation samples, typed
sync tails, configurations, code and validation logs are retained here.

The preflights retain two preparation mistakes: the parent inode initially
was not open, and unlinking files in the current mutable interval did not
produce deferred GC work. The corrected fixture opens the parent, acquires
the mount lease after snapshot preparation and deletes a child branch before
collection. A request-filter field typo in the first analysis was corrected.
Neither preparation failure entered a measured comparison, and no measurement
was changed by the analysis repair.

## Next investigation

Keep the worker's bounded progress and the explicit FULL barriers. Use the
existing optional Linux `fdatasync` SDK in a separate, equally durable
comparison of the actual WAL sync primitive, including syscall reconciliation
and crash recovery. Native results require a subsequent ordinary FUSE window
before making a production performance claim. A broad worker pause or another
coalescing interval change is not supported by these measurements.

## Reproduction

```bash
python3 vane_fs/benchmarks/staged_payload/full_sync_contention/build.py OUTPUT \
  --parent vane_fs/build/pre-open-backfill-20261013
python3 vane_fs/benchmarks/staged_payload/full_sync_contention/run.py OUTPUT
python3 vane_fs/benchmarks/staged_payload/full_sync_contention/analyze.py OUTPUT
mkdir OUTPUT/confirmation
ln -s ../build.json OUTPUT/confirmation/build.json
ln -s ../probe-release OUTPUT/confirmation/probe-release
python3 vane_fs/benchmarks/staged_payload/full_sync_contention/run.py OUTPUT/confirmation
python3 vane_fs/benchmarks/staged_payload/full_sync_contention/analyze.py OUTPUT/confirmation
```

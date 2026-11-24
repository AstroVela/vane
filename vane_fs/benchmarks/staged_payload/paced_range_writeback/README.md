# Paced finite-range writeback for format 1005

The requested C++ candidate is implemented and validated, but **rejected for
adoption: 9 of the unchanged 22 ordinary FUSE checks fail**. Complete large-file
aggregate throughput improves **2.7% fresh** and **13.1% after GC**. Lower final
application-fsync medians do not compensate for the remaining request tails
and nonregressions. Production format 2, its installed module and default
SQLite SDK remain unchanged.

This follows the [complete kernel-wait diagnosis](../fuse_sync_waits/README.md).
The earlier [format-1004 paced trial](../writeback_budget/README.md), the
[checkpoint-pause negative control](../checkpoint_pause_control/README.md)
and the unpaced range candidate's seven failed checks remain retained. They
are separate windows; differences in failed-check counts do not isolate the
effect of pacing.

## Implementation

[policy.patch](policy.patch) changes four internal sources relative to the
frozen retained-descriptor range candidate. The complete scheduler is in
[prewriteback.hpp](prewriteback.hpp). No public Workspace API, disk format,
checkpoint source or durability policy changes.

- Keep the original **64 MiB owner-local appended-byte trigger**. Once awake,
  drain one coalesced pending range in claims of at most **4 MiB**. Clip stale
  physical ranges at current EOF under the independent shared payload lease;
  a small internal accessor supplies the size. Include interleaved physical
  gaps conservatively and preserve ranges after GC truncation/regrowth.
- Wait at least **16 ms after a completed claim** before another claim. The
  deadline survives stop/restart after failed close. This is a nominal
  250 MiB/s claim budget per coordinator, not device-wide physical bandwidth.
  EINTR retries remain part of the same claim.
- A balanced, nestable RAII pause stops new claims during foreground FULL
  commit and managed `Staging::Publish`, including payload sync, WAL COMMIT
  and CLEAN-fence sync. Recheck pause and retirement after waiting for the
  GC lease. A pause neither cancels nor awaits an already claimed/in-kernel
  call. Other connections' workers and kernel writeback remain independent.
- Successful FULL retires unsent ranges. Late completion cannot resurrect
  them or mark later appends submitted. Failure unwinds the pause and preserves
  retry opportunities. Keep asynchronous error reporting before subsequent
  mutations/barriers and close, unsupported-hint fallback, fork abandonment
  and the payload error cursor retained before external appends.

Every hint is still finite, positive and WRITE-only. It never publishes a
fence or acknowledges durability. Final FULL retains payload synchronization,
FULL WAL metadata commit and durable CLEAN in that order. Inline NORMAL
transactions and fsync-mode snapshots retain their existing behavior.

Staging holds a non-owning pointer to its coordinator's worker so background
publication can pause it. It is attached before the checkpointer starts and
cleared before every worker destruction; close stops both workers first.
Fork abandonment closes the duplicate descriptor and abandons inherited
mutex/thread objects as before. These changes remain in an isolated
non-editable installation and have exact source reversal proofs.

## Frozen complete comparison

The contemporaneous baseline is the original format-1005 **extent control
without hints**. The old unpaced range candidate is the source parent, not a
third measured arm. Both measured components use the same original fsync
SQLite SDK and normal Release configuration. No native instrumentation or
kernel-wait sampler is added to the ordinary comparison.

The primary ext4 window has six AB/BA pairs per fresh/post-GC state: **24
mounts**. It preserves the first 256 MiB write with 8 MiB application calls,
subsequent 64 MiB write with 1 MiB calls, 48-byte buffer alignment, final FULL,
reads, 512 random overwrites and final FULL, 128 small-file fsyncs, namespace
operations, maintenance and original close. GC preparations and retained
guards remain identical apart from their assigned component.

The unchanged gate requires at least **10%** combined large-write aggregate
gain with **5/6 paired wins** in each state. Each workload retains at least
95% of control median and aggregate throughput. Pooled request p99 stays
within 1.2x or increases by at most 1 ms, and median lifecycle grows by no
more than 5%. No gate or parameter is changed after measurement.

| Window/state | Control combined MiB/s | Candidate combined MiB/s | Ratio | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| Primary, fresh | 232.96 | 239.36 | 1.027 | 5/6 |
| Primary, post-GC | 226.11 | 255.75 | 1.131 | 6/6 |
| Supplementary four cycles, fresh | 228.72 | 248.56 | 1.087 | 2/2 |
| Supplementary four cycles, post-GC | 224.27 | 247.07 | 1.102 | 2/2 |

Supplementary measurements add **8 mounts** with four consecutive write cycles
in one live workspace, GC only during preparation. Their two pairs per state
do not replace the primary gates. All **112 large-file content checks** pass.
Median complete mount lifecycle is 7.399/7.337 seconds control/candidate fresh
and 9.168/9.116 seconds post-GC. Both lifecycle gates pass, while the fresh
combined-gain gate fails.

## Where time improves, and where it remains

Median elapsed costs in the primary first-256-MiB phase:

| State | Write calls control/candidate, ms | Final application fsync control/candidate, ms | Complete phase control/candidate, ms |
| --- | ---: | ---: | ---: |
| Fresh | 423.246 / 423.975 | 608.302 / 471.443 | 1,036.014 / 904.650 |
| Post-GC | 448.962 / 445.944 | 617.517 / 464.852 | 1,067.125 / 917.492 |

The visible ordinary-path saving is in application fsync, not the write-call
timer. These are application timings; they do not separate the native payload,
WAL and fence syscalls. Fresh candidate repetition five still has a
**990.490 ms** application fsync and a **1,721.115 ms** combined 320 MiB
phase. That sample remains in the aggregate, reducing the fresh gain to 2.7%.
No large-file write/fsync exceeds one second in this window; this does not
resolve the earlier multi-second syscall tails or establish a latency bound.

Remaining failed checks include:

| Primary check | Control | Candidate | Failure |
| --- | ---: | ---: | --- |
| Fresh read aggregate, ratio | 1 | 0.929 | More than 5% loss |
| Fresh random-write aggregate, MiB/s | 10.37 | 9.23 | Median and aggregate loss |
| Fresh rename median throughput, ratio | 1 | 0.872 | Median and aggregate loss |
| Fresh combined large-write gain | 1 | 1.027 | Below 1.10 target |
| Post-GC first-256 pooled write p99, ms | 23.891 | 29.556 | Above both allowances |
| Post-GC random-write aggregate, MiB/s | 9.40 | 3.87 | Aggregate loss |
| Post-GC small-file pooled p99, ms | 5.182 | 21.107 | Aggregate loss and p99 |
| Post-GC mkdir aggregate, ratio | 1 | 0.891 | More than 5% loss |
| Post-GC rmdir aggregate, ratio | 1 | 0.905 | More than 5% loss |

Post-GC candidate repetition five's random phase lasts **2.029499 s**:
**1.885179 s** in write requests, including its very first 4 KiB write at
**1.617439 s**, and **144.319 ms** in the final fsync. Its phase median across
all six runs is 220.330 ms versus 223.533 ms control, but the full aggregate
ratio is only 0.412. A good median does not excuse that tail. The small-file
candidate phase also reaches **994.522 ms** versus 524.727 ms control maximum.

These ordinary traces do not identify whether a given delay is a checkpoint,
publication lock, syscall, scheduler or device wait. In particular, they do
not prove that this first random write waits for optional range submission,
or that extra pause notifications cause read/namespace regressions. The
preceding checkpoint-pause experiment did not improve complete work; overlap
alone remains insufficient to assign causality.

## Related validation and setup history

Only affected storage and integration tests run:

- **153 Python tests pass**, with explicit skips for the unavailable production
  v1 migration in format 1005 and absent PyAV. The existing inherited-fork
  deprecation warning remains.
- **12 Release and 12 ASan/UBSan C++ suites pass**, including the unchanged
  66 modeled staging crash/loss cases in each mode.
- **12 extent boundary/snapshot/GC cases per mode** pass.
- **8 scheduling cases per mode** check threshold, complete bounded draining,
  minimum interval, nested/disabled/null/exception pauses, active FULL plus
  later appends, EIO/ENOSPC/EINTR retry, pause after GC-lease wait, EOF clipping,
  regrowth and stop/restart deadline preservation.
- **14 inherited fault cases per mode** reuse byte-identical function bodies:
  writer progress and independent leases, barrier/close failure, ENOSPC,
  rollback, GC/snapshots, fork, strict mode, three unsupported errors, failed
  authoritative FULL, late error and a previously observed async writeback
  error that a newly opened descriptor could miss.
- **8 real FULL integration cases per mode** block payload, WAL and CLEAN
  synchronization separately in foreground commit and background publication.
  An active hint can finish, but no new claim occurs while FULL is blocked.
  Success retires pending work; payload-sync failure releases the pause and
  allows retry. Each case reopens and verifies file contents.

The initial compile called private `PayloadFile::Size`; a small internal
accessor fixes EOF clipping. No test or measurement started in that attempt.
Its complete source/command/log identities are retained. A later integration
fixture initially blocked background `Guard::Load` adopting an unfamiliar
OPEN fence, before `Publish()` and its pause. Six prior integration cases
passed; the seventh assertion failed because it watched the wrong phase.
The corrected fixture enables gating after acquiring the guards, observing
the actual CLEAN sync. Native source/libraries remain unchanged for that
fixture correction, and both complete modes are rerun. The failing source,
log, command and cleanup are retained separately.

All 32 FUSE workloads pass content, read/overwrite, small-file, retained-guard,
format/fence and SQLite integrity checks. Source/native/module identities are
checked before and after each ordinary window. No broad base/Ray suite runs.

## Evidence, cleanup and next investigation

[findings.json](findings.json) retains every gate, phase distribution and
application tail; [final-checks.json](final-checks.json) records source checks,
the 11 unchanged production identities and process/mount cleanup.
[audit.json](audit.json) indexes all raw request vectors, host observations,
native source snapshots, generated runners/workloads, commands, test/build
logs and both preflights by original size/SHA-256 and ordered gzip parts.
Concatenate an entry's `retained_as` files and decompress once to recover
identical bytes. Every published artifact is below 500 KiB.

All **44 owned data directories** are removed after their processes stop,
including the failed integration fixture. No owned mount/process remains.
Cleanup is outside measured windows. No cache eviction, CPU/device tuning,
load-based sample filtering or unowned process termination occurs. Build,
tests, primary measurement, sustained measurement and analysis are serial.

Keep this policy isolated. Before another scheduling change, attribute the
**first 4 KiB overwrite after GC** and the small-file p99 on the new candidate
with the existing typed native request recorder. Separate the write callback,
SQL/checkpoint/publication waits, actual syncs and concurrent optional hints.
Use diagnostic results only for causality, retaining these nine ordinary
failures. A later factorial control can separate pacing, FULL pausing and the
presence of a worker; this bundled two-arm result cannot do so.

```bash
here=vane_fs/benchmarks/staged_payload/paced_range_writeback
out="$PWD/vane_fs/build/paced-range-writeback-repro"
python3 "$here/generate.py" "$out" \
  --parent "$PWD/vane_fs/build/payload-range-writeback-retained-20261018"
python3 "$here/build.py" "$out"
python3 "$here/validate.py" "$out"
# Existing setup failures are retained; a clean replay records NOT_NEEDED.
python3 "$here/freeze.py" "$out"
python3 "$here/run.py" "$out/fuse"
python3 "$here/analyze.py" "$out/fuse"
python3 "$here/summarize.py" "$out"
# Record final source/identity/cleanup checks as "$out/final-checks.json".
python3 "$here/publish.py" "$out"
```

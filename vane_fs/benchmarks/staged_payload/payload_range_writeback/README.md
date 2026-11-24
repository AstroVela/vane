# Finite-range payload writeback experiment

**Keep this prototype isolated; do not adopt it.** Correctness checks pass,
but **7 of the original 22 performance checks fail**. All rates below compare
the unchanged format-1005 extent control with this candidate on the same host
and frozen workload; they are not production or cross-system rates.

| Complete 256+64 MiB writes, aggregate MiB/s | Extent control | Range writeback | Change | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| Full fresh | 117.52 | 132.85 | +13.0% | 4/6 |
| Full post-GC | 59.84 | 289.58 | +384.0% | 6/6 |
| Sustained (supplementary) fresh | 114.27 | 102.44 | -10.4% | 1/2 |
| Sustained (supplementary) post-GC | 107.97 | 190.69 | +76.6% | 2/2 |

Fresh complete writes clear the aggregate gain threshold but win only four of
six pairs, below the required five. Fresh first-256 pooled write-call p99 grows
**31.296 to 41.797 ms (+33.6%)**, failing its latency gate despite higher
throughput. Its maximum write call grows 109.774 to 465.513 ms. No large-file
write call exceeds one second in either variant across all 32 mounts.

The five other failed checks are fresh small-file-fsync throughput (median
-18.2%, aggregate -17.1%), fresh rename median throughput (-5.1%), post-GC read
median throughput (-5.7%), post-GC random-write aggregate throughput (-66.5%),
and post-GC small-file-fsync throughput and p99. In that last workload,
aggregate throughput falls **251.89 to 41.60 files/s** and p99 grows **5.434
to 154.318 ms**. All seven checks and exact reasons remain in the
[analysis](validation/analysis.json); none is waived or reclassified.

| Full first-256 measurement | Fresh control | Fresh range | Post-GC control | Post-GC range |
| --- | ---: | ---: | ---: | ---: |
| Write-loop median, seconds | 0.439 | 0.449 | 0.448 | 0.449 |
| Final-fsync median, seconds | 0.616 | 0.333 | 3.008 | 0.312 |
| Pooled write-call p99, milliseconds | 31.296 | 41.797 | 26.148 | 25.069 |
| Daemon CPU median, seconds | 0.485 | 0.520 | 0.500 | 0.545 |

Full fresh still has two large-file final fsyncs above one second per variant;
the candidate reaches 3.814 seconds. Full post-GC has five such control
barriers and none for the candidate. In the supplementary window, the
candidate still reaches **6.229 seconds fresh** and **3.800 seconds post-GC**.
Full lifecycle medians fall 9.893 to 9.397 seconds fresh (-5.0%) and 14.758
to 13.806 seconds post-GC (-6.5%). Supplementary fresh lifecycle instead grows
25.811 to 26.651 seconds (+3.3%); post-GC falls 33.083 to 32.323 seconds
(-2.3%). The much smaller lifecycle gains and slower small-file phases make
the complete large-write gain insufficient for adoption. The two-pair
supplementary window cannot establish reliable gains.

Next, attribute the fresh write-call p99 and post-GC random-write/small-file
barriers, including the transition from large writes into those later phases.
The application timings show simultaneous improvements and regressions; they
do not establish that a wait moved between phases or identify a kernel/device
cause. Keep the original gates and authoritative FULL ordering while collecting
that evidence. These results do not justify another threshold change.

Validation passes **153 related Python tests** (two skips), **12 Release +
12 ASan/UBSan component suites**, **24 extent case executions**, and **42 new
case executions** (21 in each mode). All **32 FUSE mounts** pass content,
guard, format/fence and SQLite integrity checks, including **112 large-file
content checks**. No broad base/Ray suite runs. The [audit](audit.json) covers
32 benchmark workspaces and 17 component/preflight data directories, all
removed after owned processes and mounts stop. Logs, source proofs and both
expected negative regressions are retained under [validation](validation/).

This isolated C++ trial replaces the advisory worker's whole-payload
`fdatasync` with positive, finite `sync_file_range(..., SYNC_FILE_RANGE_WRITE)`
requests. The trigger stays at 64 MiB of successful physical appends per owner.
The authoritative FULL operation still synchronizes the whole payload before
the existing WAL/CLEAN sequence. Production format 2, the default installed
module and the default SQLite SDK stay unchanged.

The preceding [wait diagnosis](../prewriteback_waits/README.md) attributed the
observed multi-second final barriers to payload synchronization. A background
whole-file sync overlapped the candidate's longest barrier, but the control
also stalled without such a worker. This motivates testing a different
advisory operation; it does not establish the kernel/device cause or a fix.
Earlier [prewriteback](../payload_prewriteback/README.md) acceptance failures
remain recorded independently.

The worker holds no Database, checkpoint or publication mutex during I/O. Its
independently opened payload descriptor holds a shared lease that permits
appends and excludes GC. Successful SQL commits transfer the writer's pending
physical minimum/maximum append offsets into one coalesced pending range.
There is at most one active range and one pending range. They may include gaps,
another owner's intervening data and successful appends from subsequently
rolled-back transactions. This is bounded bookkeeping, not a strict 64 MiB
range size or global dirty-byte budget.

The worker acquires the lease, rechecks retirement by FULL, claims the pending
range and clips its end to the current EOF before submitting it. Physical
offsets, rather than a monotonically advancing tail, handle GC truncation and
regrowth. A single transaction may exceed the trigger before notification.
FULL retires prior requests; later commits remain pending. A submitted request
never clears the writer's dirty flag or acknowledges durability.

`WRITE` starts data writeback, but can itself block. The kernel rounds ranges
to pages. It does not provide the metadata or device-cache guarantees of the
final FULL barrier. No WAIT flag or zero-length-to-EOF request is used. These
limits follow the [Linux interface documentation](https://man7.org/linux/man-pages/man2/sync_file_range.2.html).
EINTR retries the same bounded request. ENOSYS, EOPNOTSUPP and EINVAL disable
this optional hint for that descriptor; FULL remains mandatory. Other errors
are latched for the next mutation, explicit barrier or close, and the claimed
range merges back for retry unless FULL already retired it. Failed close
retains the worker's retry path. Fork abandonment closes the child's duplicate
without unlocking or joining the parent's inherited worker.

The candidate also retains a separate payload barrier descriptor from before
external writes, rather than opening one only at FULL. A fresh descriptor can
miss an asynchronous error already observed elsewhere; the retained descriptor
preserves its error observation position. See the kernel's
[errseq explanation](https://www.kernel.org/doc/html/latest/core-api/errseq.html)
and [fsync error reporting](https://man7.org/linux/man-pages/man2/fsync.2.html).
The payload sync count and payload→WAL→CLEAN ordering are unchanged. The
checkpoint source is byte-identical; the descriptor lifetime change in staging
is explicitly part of this trial. The performance comparator is the original
extent control with neither advisory worker nor this lifetime correction.

Before measurement, an initial version passed 20 cases in both Release and
ASan/UBSan. A subsequently added software model reproduced the missed-error
case: another open descriptor observes the injected error first, after which
the temporarily opened FULL descriptor reports success. That version fails
the new 21st case. The corrected candidate opens and retains its barrier
descriptor before writes. This tests an error observation model, not real
disk failure. The initial build, source snapshots, passing tests, expected
failure and cleanup records are preserved; it has no performance samples.

The 21 native cases cover the threshold; writer progress and shared leases
during blocked submission; OPEN/CLEAN and independent FULL barriers; reset and
concurrent later commits; I/O failure, retry, failed close and late errors;
rollback, pinned snapshots, GC shrink/regrowth, interleaved owners and stale
tails; fork and strict mode; unsupported hints; EINTR; ENOSPC; mandatory FULL
failure after successful hints; and the modeled already-seen asynchronous
error. The old whole-file-fdatasync worker also fails the new helper as
expected because it invokes a background payload durability barrier.

Measurement uses all 22 original acceptance checks, six AB/BA pairs in each
of fresh and independent post-GC states, plus a supplementary four-cycle
256+64 MiB workload with two pairs per state. Both variants use format 1005.
No thresholds or workloads are retuned. Complete-write timings include final
FULL; write loops, final fsyncs, lifecycle and pooled write-call p99 are kept
separately. Builds and tests finish before serial measurement; analysis starts
after both windows. No cache eviction or sample filtering is performed.

With the retained SDK and frozen parent artifacts:

```bash
python3 generate.py <artifacts> --parent <payload-prewriteback-artifacts>
python3 regression_before.py <artifacts>
python3 build.py <artifacts> --stage all
# Add the retained initial-preflight.json and modeled-error negative evidence.
python3 prepare.py <artifacts>
python3 run.py <artifacts>/fuse
python3 analyze.py <artifacts>/fuse
# After recording final source checks:
python3 publish.py <artifacts>/fuse
python3 audit_history.py <artifacts>/fuse
```

The historical negative uses `regression_async.py <initial-preflight-artifacts>`
with that earlier frozen library, not the corrected candidate. Source manifests
record every native transformation, its exact reversal and code hashes. Large
workspace data is removed only after owned processes and mounts stop; hashes,
logs, measurements, configurations and cleanup records are retained in
lossless chunks smaller than 500 KiB each.

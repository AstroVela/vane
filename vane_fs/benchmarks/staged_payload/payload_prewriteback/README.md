# Advisory payload prewriteback experiment

**Keep this prototype isolated; do not adopt it.** Correctness checks pass,
but **4 of the 22 original performance checks fail**. GC-state complete writes
improve substantially in this window; fresh complete-write aggregate throughput
regresses. All measurements compare the unchanged extent control with this
isolated candidate, with both using format 1005. They are not production or
cross-system throughput.

| Complete 256+64 MiB writes, aggregate MiB/s | Extent control | Advisory prewriteback | Change | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| Full fresh | 125.07 | 113.71 | -9.1% | 4/6 |
| Full post-GC | 76.39 | 156.77 | +105.2% | 5/6 |
| Sustained (supplementary) fresh | 112.73 | 131.74 | +16.9% | 1/2 |
| Sustained (supplementary) post-GC | 117.85 | 130.07 | +10.4% | 2/2 |

The full fresh first-256 median rate improves 248.19 to 321.11 MiB/s, while its
aggregate rate falls 115.28 to 103.85 MiB/s. Two candidate final fsyncs take
**6.106 and 4.417 seconds**; two control fsyncs take 3.582 and 4.721 seconds.
All samples count. The complete fresh 256+64 aggregate falls 9.1% and wins only
four of six pairs, missing the original 10% gain and five-win requirement.
The other failed checks are fresh first-256 aggregate throughput (-9.9%),
fresh sequential-read median throughput (-8.8%), and post-GC sequential-read
aggregate throughput (-8.4%). Exact gate reasons remain in the
[analysis](validation/analysis.json).

| Full first-256 measurement | Fresh control | Fresh prewriteback | Post-GC control | Post-GC prewriteback |
| --- | ---: | ---: | ---: | ---: |
| Write-loop median, seconds | 0.412 | 0.439 | 0.433 | 0.445 |
| Final-fsync median, seconds | 0.615 | 0.369 | 0.621 | 0.323 |
| Pooled write-call p99, milliseconds | 38.612 | 37.558 | 23.200 | 25.442 |
| Daemon CPU median, seconds | 0.445 | 0.515 | 0.485 | 0.535 |

Across all 32 mounts, neither variant has a large-file write call over one
second. The remaining multi-second stalls are final fsyncs. In the full window,
these number two versus two in fresh state and four versus one after GC. In
the supplementary window, each variant has two in each state. The candidate
still reaches 6.593-second fresh and 5.725-second post-GC sustained fsyncs.
These application timings establish where the observed stalls occur; they do
not distinguish payload synchronization, WAL work or kernel/device waits.
The native concurrency regression verifies that blocking the advisory flush
does not retain the three application mutexes or block an independent FULL
barrier before its own kernel synchronization.

Full fresh lifecycle median grows 9.439 to 9.664 seconds (+2.4%); post-GC falls
14.240 to 11.755 seconds (-17.5%). Supplementary fresh lifecycle grows 25.380
to 26.882 seconds (+5.9%) despite the complete large-write gain. Post-GC
supplementary lifecycle falls 31.750 to 27.444 seconds. The supplementary
window has only two pairs per state and cannot establish reliable adoption.

Next, attribute the remaining long final fsyncs in this candidate to payload,
WAL and actual waits, then investigate the sequential-read regression. The
current evidence does not justify another threshold change, a weaker FULL
barrier or a kernel/device causal claim. The prior
[publication-lock diagnosis](../payload_waits/README.md) motivated the separate
worker, and this trial establishes its concurrency boundaries and remaining
performance limits.

Validation passes **153 related Python tests** (two environment/format skips),
**12 Release + 12 ASan/UBSan component suites**, **24 extent case executions**
and **18 new prewriteback case executions**. All **32 FUSE mounts** pass content,
retained-guard, format/fence and SQLite integrity checks, including **112
large-file content checks**. No broad base/Ray suite runs. The [audit](audit.json)
covers 32 removed benchmark workspaces plus seven component data directories
and one baseline-regression directory. All their owned processes and mounts
stopped before cleanup. Test logs, source proofs and the expected control
failure are retained under [component validation](validation/component/).

This isolated C++ trial derives from the frozen format-1005 extent component.
A separate worker synchronizes the payload after **64 MiB** of committed,
connection-local external appends. It holds no Database, checkpoint or
publication mutex during that I/O. An independently opened payload descriptor
holds a shared flock, allowing appends while excluding GC. Reusing the writer's
open-file description would let one thread unlock another's lease.

Prewriteback is advisory: it never publishes metadata, changes the OPEN/CLEAN
fence or acknowledges durability. The existing FULL barrier still synchronizes
the payload before its original WAL/CLEAN sequence. The staging and checkpoint
sources are byte-identical to the control. WAL thresholds remain 16/64 MiB.
Production format 2 and the default module/SQLite SDK stay unchanged.

The counter includes successful physical appends from later rolled-back SQL
transactions conservatively, but notifications follow successful commits only.
It remains monotonic after GC truncation and tail reuse. FULL barriers retire
earlier local requests. An in-flight flush records only its captured target;
later commits remain pending. This is a per-owner heuristic, not global dirty
byte accounting. Cross-owner durability continues to use the authoritative
FULL path. A single transaction can exceed the budget before notification.

The worker latches I/O errors for the next mutation, explicit barrier or close,
retains its request for retry, and treats a busy GC lease as retryable. Failed
close restarts the worker. Forked children close their duplicate descriptor
without unlocking or joining the parent's inherited worker.

Related validation includes the original Release and ASan/UBSan component
suites, Python tests and extent-specific cases. Nine additional native cases
cover the threshold, ordinary writer progress during blocked preflush,
independent shared leases, OPEN/CLEAN and authoritative FULL ordering, repeated
budget/reset, a concurrent FULL and late commit, asynchronous errors and retry,
failed close, rollback, GC and pinned snapshots, tail reuse, fork abandonment
and strict-mode behavior. The unchanged extent control fails the same public
API regression because a 64 MiB append below the WAL threshold does not start
advisory prewriteback.

The complete FUSE comparison uses the original six AB/BA pairs in each of fresh
and independent post-GC states, all 22 frozen acceptance checks, and unchanged
workloads and thresholds. The supplementary comparison uses four consecutive
256+64 MiB cycles in one workspace, with two pairs per state. It cannot override
a failed complete comparison. Application timings include the final FULL
barrier; write-loop, fsync, lifecycle and pooled p99 are retained separately.
No cache eviction, tuning or sample filtering is performed.

Reproduce with the retained SDK and frozen parent artifacts:

```bash
python3 generate.py <artifacts> --parent <versioned-extents-artifacts>
python3 regression_before.py <artifacts>
python3 build.py <artifacts>
python3 prepare.py <artifacts>
python3 run.py <artifacts>/fuse
python3 analyze.py <artifacts>/fuse
# After recording final source checks:
python3 publish.py <artifacts>/fuse
```

Owned builds and tests finish before measurement. Analysis runs after both
measurement windows. Measurements, logs, frozen configurations and source/code
identities are retained; large workspace contents are removed after their
owned processes and mounts stop. Lossless chunks keep Git artifacts below
500 KiB each.

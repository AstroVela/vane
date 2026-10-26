# Ordinary and sustained FUSE evaluation of versioned extents

The format-1005 extent candidate **fails 14 of the original 22 complete FUSE
acceptance checks**. Fresh combined large-write aggregate improves 8.2%, but
post-GC aggregate falls 46.4%. Its smaller metadata and lower write CPU cost
do not establish a complete-operation improvement. The candidate remains
isolated; production format 2, its module and default SDK are unchanged.

This follows the [versioned extent count experiment](../versioned_extents/README.md).
The comparator is that experiment's frozen borrowed-buffer format-1004 core,
not a new production measurement. Both components use the same SQLite fsync
SDK, fsync durability, physical append batches, publication/fence protocol and
16/64 MiB checkpoint thresholds. Neither component is rebuilt or instrumented
for this evaluation.

## Frozen workload and gates

The ordinary window runs **24 independent ext4 mounts**, with six AB/BA pairs
in fresh workspaces and six after independent fragmentation/GC preparation.
Each mount writes 256 MiB using an 8 MiB application buffer at page offset 48,
then 64 MiB using 1 MiB calls, followed by sequential/random reads, random
overwrites, small files with fsync and directory operations. Final fsync and
file close remain inside each complete write timer.

Post-GC preparation creates a retained 32 MiB guard, zeros alternate 4 KiB
blocks and runs GC using each format's own frozen module. The guard's entire
content is checked before GC and again after the workload. Schema, payload
lengths, range coverage, allocation and SQLite integrity are inspected outside
the workload timers. Lifecycle includes initialization, workload, validation
and shutdown; post-GC lifecycle also includes guard/GC preparation.

[prepare.py](prepare.py) adapts the previously frozen ordinary runner, preserving
an exact reverse proof for every replacement. Changes bind per-format schema
checks, add out-of-timer inspection fields, label the comparison and configure
the separate sustained window. The workload source remains byte-identical.
Sources, binaries, modules, SDK, workload, monitor and original gate analyzer
are hashed before either window. See the [full plan](full/plan.json).

The original full gates remain unchanged:

- Combined 256 + 64 MiB aggregate throughput improves at least 10%, with at
  least five of six paired wins in each state.
- Every workload retains at least 95% of both median and aggregate throughput.
- Recorded pooled request p99 grows at most 20%, or at most 1 ms absolutely.
- Median complete lifecycle grows at most 5%.

All samples, including every slow control and candidate, enter the same
[original analyzer](../partial_sync/analyze_writeback.py). The supplemental
window has no replacement adoption gates. No build, test or heavy analysis
overlaps either measurement window. A read-only host monitor remains common;
there is no global cache eviction, CPU/device tuning or load-based filtering.

## Ordinary results

Aggregate rates divide all completed bytes/operations by their total time.
Medians and paired vectors are separately retained in the
[complete summary](full/summary.json).

| Combined 256 + 64 MiB | Format 1004, MiB/s | Format 1005, MiB/s | Change | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| Fresh | 81.42 | 88.12 | +8.2% | 4/6 |
| After GC | 80.19 | 42.95 | −46.4% | 1/6 |

Both primary checks fail. The fresh candidate misses both the aggregate-gain
and win thresholds; the post-GC candidate regresses substantially. The complete
throughput comparison is below, including workloads that improve.

| Aggregate MiB/s | Fresh 1004 → 1005 | After GC 1004 → 1005 |
| --- | ---: | ---: |
| First 256 MiB write | 162.77 → 133.26 | 75.40 → 66.49 |
| Subsequent 64 MiB write | 27.15 → 37.42 | 107.46 → 17.77 |
| Sequential read, 64 MiB | 796.65 → 977.69 | 737.93 → 1,001.25 |
| Random read, 4 KiB | 70.18 → 67.39 | 70.21 → 67.06 |
| Random overwrite, 4 KiB | 6.58 → 8.77 | 10.31 → 3.09 |

| Aggregate operations/s | Fresh 1004 → 1005 | After GC 1004 → 1005 |
| --- | ---: | ---: |
| Small file + fsync | 84.43 → 79.84 | 37.80 → 77.82 |
| mkdir | 1,759.04 → 1,608.29 | 1,955.68 → 1,844.98 |
| rename | 1,702.25 → 1,844.37 | 1,962.55 → 1,828.00 |
| rmdir | 2,088.80 → 2,207.38 | 2,188.03 → 1,986.34 |

Four fresh workload checks fail: first-256 aggregate, random-overwrite median,
small-file aggregate and mkdir median/aggregate. Seven post-GC workload checks
fail: both large writes, random read, random overwrite, mkdir, rename and rmdir.
Post-GC random-read aggregate retains 95.5%, but its median retains only 94.6%.
Fresh random-overwrite aggregate improves while its median falls 7.5%; the
aggregate alone does not pass that check. All recorded request-p99 checks pass.

Fresh median lifecycle falls 11.270 to 10.463 seconds. Post-GC median lifecycle
rises **16.015 to 19.325 seconds (+20.7%)**, failing the remaining check. The
[failure details](validation/analysis.json) retain each gate and its reasons;
the 14 failures are eleven workload checks, two combined-write checks and one
lifecycle check.

## Faster writes can still have slower final barriers

Post-GC first-256 request-loop median falls **1.407 to 0.444 seconds**. Observed
daemon CPU during that complete phase falls 0.895 to 0.490 seconds, including
any worker CPU in the interval. This is phase CPU, not a complete-lifecycle CPU
measurement. The preceding count experiment independently demonstrates fewer
metadata records, VM steps and WAL pwrite bytes.

Final fsync remains expensive. After GC, the candidate has first-256 final
fsyncs of 6.471, 6.029 and 6.058 seconds in three of six repetitions. Its
subsequent-64 final fsync exceeds one second in **five of six** repetitions;
the control does so in none. Median subsequent-64 fsync rises **0.174 to
3.880 seconds**, while the request-loop median falls 0.280 to 0.258 seconds.
The complete-operation regression is therefore visible inside timed writes,
outside GC preparation. GC-call wall medians themselves are 0.111 / 0.102
seconds for control / candidate.

Application timers locate these waits in final fsync. This uninstrumented
window does not distinguish payload synchronization, WAL synchronization and
checkpoint-lock acquisition within a particular fsync. The older format-1004
wait attribution cannot assign causes to individual new format-1005 samples.
Reducing metadata work does not bound durable I/O latency.

## Continuous 1.25 GiB supplement

Eight additional independent mounts run two AB/BA pairs per state, with **four
consecutive 256 + 64 MiB cycles in one workspace** before reads/overwrites and
namespace operations. Each file is fully validated, unlinked and recreated
for its next cycle; no GC runs between cycles. This exercises continued
metadata/WAL growth and subsequent checkpoint work. It is not an in-place
large-file overwrite benchmark. See the [sustained plan](sustained/plan.json).

| Aggregate across all four large-write cycles | Format 1004, MiB/s | Format 1005, MiB/s | Change | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| Fresh | 81.19 | 114.44 | +41.0% | 2/2 |
| After GC | 58.00 | 116.70 | +101.2% | 2/2 |

Supplemental lifecycle medians fall 31.187 to 25.520 seconds fresh and 43.216
to 30.236 seconds after GC. These two-pair results do not replace the original
six-pair failures. There are still late slow calls: the candidate's fresh
fourth-cycle combined aggregate falls 146.34 to 71.79 MiB/s, including a
5.992-second final fsync. The post-GC third cycle has a 6.025-second candidate
fsync. Every per-cycle vector remains in the analysis and raw mount records.

## Validation, cleanup and reproduction

All **32 mounts** pass content, retained-guard, schema, CLEAN-fence and SQLite
integrity checks, including **112 complete large-file checks**, all 32
read/overwrite fixtures and 4,096 small-file content checks. Both formats use
their own verified GC module. The existing candidate's 153 related Python
tests, 12 Release and 12 ASan/UBSan component suites, and 24 extent-case
executions are reused by exact source/binary identity; no native implementation
changes or broad base/Ray suite run in this follow-up.

The [audit](audit.json) verifies all frozen identities, complete sample retention,
separate measurement/analysis windows, stopped processes and **32 cleaned owned
data directories**. Large databases and payload files are deleted after their
processes stop. Raw phases/requests, validations, allocation records, logs,
host samples and cleanup hashes are retained losslessly in indexed files below
500 KiB. Correctness success is separate from the failed performance gate.

```bash
here=vane_fs/benchmarks/staged_payload/extent_fuse
out="$PWD/vane_fs/build/extent-fuse-repro"
component="$PWD/vane_fs/build/versioned-extents-20261018"
python3 "$here/prepare.py" "$out" --component "$component"
python3 "$here/run.py" "$out"
python3 "$here/analyze.py" "$out"
```

The frozen format-1004/1005 component builds and SDK must be available as
described by the preceding extent report. The next investigation should
attribute format-1005 final fsync during the post-GC first/subsequent writes,
preserving these failures and the original durability/gate boundaries.

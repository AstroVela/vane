# Shared 64 MiB unpublished-payload budget

This isolated C++ candidate adds a payload byte budget to the preceding
no-hint orphan-batching candidate. **It is not adopted.** Original combined-write
targets pass, but 13 non-regression checks fail. The separate comparison
against preceding orphan batching also regresses ordinary combined writes.
The production 4 KiB SQLite BLOB component and installed module are unchanged.

## Implementation and durability

Only `src/staging.hpp`, `src/checkpoint.hpp` and `src/workspace.cpp` change:

- Capture the current payload tail in the fence's CRC-covered reserved field
  at offset 80 when starting OPEN. Every participating connection loads that
  origin; worker/admission checks subtract it from the current append tail.
- Successful appends update an atomic local hint. Commit callbacks only read
  that hint, with no extra file inspection or SQLite query, and wake the worker
  at 64 MiB even when the WAL is below its former 16 MiB trigger.
- The worker rechecks shared state under the publication guard and publishes
  when either the original WAL policy or the payload budget requires it.
- Before a new external write's SQL transaction, check the shared budget and
  publish existing data if needed. A failed admission leaves the next operation
  unstarted and OPEN available for retry.

The origin is advisory only. Recovery does not read it, and it never establishes
a durable prefix. Zero legacy origins or origins past the current tail cause
conservative accounting. New OPEN batches recapture the tail after publication,
rollback and GC. All writers must participate for consistent budget scheduling.
A single atomic write can exceed the budget; this is not a file or device quota.

`BeforeFullCommit`, `AfterFullCommit`, `Publish` and `Recover` bodies are
byte-identical to preceding orphan batching. The payload file and reclaim code
are unchanged. The final whole-file sync remains under the publication guard,
with FULL WAL commit followed by CLEAN. The existing 64 MiB WAL admission rule
and prompt CLEAN checkpoint policy remain. There is no preparatory presync,
writeback hint, segmented format, or modified durability setting.

## Original adoption window

The frozen original no-hint partial-edge control remains the adoption baseline.
All 48 uninstrumented mounts retain the same workload, page-offset-48 buffers,
six alternating pairs per state, validation, timing boundaries and gates.

| Medium/state | Control aggregate MiB/s | Candidate | Change | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| tmpfs fresh | 194.77 | 328.85 | +68.8% | 6/6 |
| tmpfs post-GC | 191.47 | 294.80 | +54.0% | 6/6 |
| ext4 fresh | 95.86 | 133.17 | +38.9% | 6/6 |
| ext4 post-GC | 110.37 | 132.71 | +20.2% | 6/6 |

The fresh ext4 control retains a 4.42-second first-256 phase; it affects the
aggregate and is not removed. First-256 median rates decrease 153.01→147.11
MiB/s fresh and 153.85→148.91 post-GC, while its pooled write p99 decreases
357.5→268.5 ms and 368.5→271.6 ms. Subsequent-64 medians improve
53.23→96.04 MiB/s and 52.08→94.31, but pooled write p99 increases
70.6→150.5 ms and 73.3→147.5 ms.

All primary combined-write targets pass. Thirteen non-regression checks fail:
tmpfs random writes and rmdir in both states; fresh ext4 subsequent writes,
random writes, mkdir, rename and rmdir; post-GC ext4 subsequent writes, reads,
small-file fsync and rename. A post-GC candidate small-file phase around
4.25 seconds and a control around 2.78 seconds are retained. Detailed
checks, including median/aggregate versus latency failures, are in `summary.json`.

## Direct comparison with preceding orphan batching

Another 24 uninstrumented ext4 mounts isolate the budget change. Their reference
checks do not replace adoption gates.

| State | Combined median MiB/s, control→candidate | Aggregate MiB/s | Paired wins |
| --- | ---: | ---: | ---: |
| fresh | 138.51→134.59 (−2.8%) | 103.45→135.10 (+30.6%) | 1/6 |
| post-GC | 135.34→132.01 (−2.5%) | 135.37→94.93 (−29.9%) | 0/6 |

The fresh control retains a 6.42-second first-write phase, explaining the higher
candidate aggregate despite ordinary regression. The post-GC candidate retains
a 7.46-second first-write phase. First-256 median rates decrease
153.48→147.67 MiB/s fresh and 154.53→145.86 post-GC. Its pooled write p99 improves
380.5→272.2 ms and 362.3→272.8 ms. Subsequent-64 p99 barely changes
(147.7→148.7 ms fresh, 151.7→146.6 post-GC). Five reference checks fail, including
combined-write targets in both states.

## Typed diagnostic: more barriers, little relief of publication waiting

Sixteen separate instrumented ext4 mounts use preceding orphan batching as
control. These rates do not decide adoption. Inclusive times and overlapping
workers must not be summed.

| First-256 median cost, ms | Fresh control→candidate | Post-GC control→candidate |
| --- | ---: | ---: |
| worker protected publication | 474.4→609.9 | 457.2→624.1 |
| worker WAL/database fsync, inclusive | 134.5→175.9 | 131.6→175.6 |
| foreground publication-lock waiting | 556.5→548.1 | 541.4→555.1 |
| final application fsync | 172.4→192.1 | 189.9→209.6 |

Worker publication count doubles from median two to four. Foreground budget
inspection itself costs about 4.4–4.5 ms across the first-write phase; direct
budget-admission publication does not occur in these diagnostic phases because
the worker wins the scheduling race. The deterministic admission test covers
the fallback path with the callback suppressed.

All **5,871 syncs** reconcile exactly with their request aggregates; all **87
calls at least 100 ms** remain. Payload sync peaks at 1.589 s in the control
and 1.497 s in the candidate; WAL sync peaks at 890 ms, database sync at 924 ms,
and fence sync at 42 ms.

A post-GC candidate subsequent-64 phase lasts **5.180 seconds**, including
1.497 s worker payload sync, 650 ms foreground WAL-prefix sync, 924 ms database
checkpoint sync and 1.125 s final foreground payload sync. Their sync intervals
do not overlap, while request/publication/checkpoint spans contain them.
Foreground publication-lock waiting totals 1.846 s. The foreground write and
final-barrier CPU times total about 284 ms; sampled runqueue waiting is only
about 0.041 ms during the write window. Smaller batches have not removed storage
synchronization stalls. This trace explains that diagnostic phase, not every
separate plain-window tail or a particular device mechanism.

Do not enable this fixed budget or remove final syncs. Retain preceding orphan
batching as the next causal control. Next investigate WAL/database checkpoint
work and the metadata that needs copying/synchronizing; increasing publication
frequency has not produced a useful default-policy tradeoff.

## Related validation and retention

Seven new cases pass Release and ASAN/UBSAN: payload-only wake below the WAL
threshold, cross-connection accumulation and stale-origin reset, GC shrink with
retained snapshots, admission EIO/retry, SQL rollback after physical append,
legacy/invalid advisory origins, and crash before payload sync with discarded
page-cache bytes and repeated recovery. The old orphan control fails the wake
case as expected. Inherited CLEAN/continuous/no-query/Stop and orphan tests pass.

Original component tests remain unchanged: 11 control and 11 candidate C++
tests, 154 installed Python tests and 11 ASAN/UBSAN C++ tests pass. One existing
production-v1 migration test is skipped for experimental format 1004. Only
related tests ran.

All 88 mounts pass content, retained-guard and storage inspection, stop/unmount
and remove their temporary data. `audit.json` covers **144 removed paths**,
including helper/Python data and the initial test-fixture failure. That failure
incorrectly required a sub-16 MiB CLEAN WAL to backfill immediately; its corrected
precondition preserves the original policy. The observed setup WAL is 9,508,992
bytes. A helper hash-path typo after successful native install was also corrected
before measurements. Their logs and reports remain under `preflight/`.

No builds, tests or formatters overlap measurement windows. No global cache
eviction or CPU/device setting changes occur. Gate reconstruction, 35-file
candidate reconstruction using the original control plus orphan/budget patches,
and 669 retained/published identities verify. JSON arrays are split losslessly,
each below 500 KiB; full traces, host/CPU logs, binaries and frozen sources remain
locally with paths and hashes in `manifest.json`.

## Reproduce

`prepare.py` takes the original frozen no-hint partial-edge artifact as
`--parent` and preceding orphan batching as `--orphan-parent`. Supply the same
SQLite 3.53.2 and FUSE SDKs and test-support site-packages. It performs a
non-editable Release install under `updated/build/python-release`.

Build with `prepare.py`, validate with `validate.py`, then build diagnostics with
`prepare_diagnostic.py`. Run `run.py --medium tmpfs` and `run.py --medium ext4`,
followed by the original `partial_sync/analyze_writeback.py`. Run `run_causal.py`
and analyze its `causal/` directory with the same analyzer. Run
`run_diagnostic.py`, `analyze_budget.py` and the inherited request-cost, CPU and
maintenance analyzers against the diagnostic directory. Finally run
`summarize.py --diagnostic …` and `publish.py --diagnostic …` on the plain artifact.
Build and test steps must finish before any measurement window.

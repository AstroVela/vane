# Checkpoint worker scheduling batches: rejected candidate

This C++ experiment follows the foreground maintenance costs in the rejected
[uncopied-threshold candidate](../checkpoint_coalescing/README.md). It retains
the original CLEAN-tail progress policy and changes only scheduling in
`src/checkpoint.hpp` of the isolated format-1004 budget implementation.
The complete plain comparison still fails 11 adoption checks. This scheduling
candidate remains isolated; production code, module and SDK are unchanged.

## Fixed scheduling policy

The first pending request records a deadline two milliseconds ahead. Later
commits neither move that deadline nor signal an already-pending request again.
The worker waits until this fixed deadline before consuming the request. Time
spent in an already-active pass counts toward that interval, so queued work
does not acquire another full delay after the pass finishes. The wait holds no
checkpoint/publication lease and releases the queue mutex. Stop interrupts it.
This is a programmed batching interval, not a wall-clock latency guarantee:
OS scheduling and real I/O can still take longer.

The commit callback performs no additional SQLite status query. The original
OPEN progress threshold and timely CLEAN processing are unchanged, as are the
20 ms partial/error retry, pending work across Stop/Start, 64 MiB total-frame
admission and FULL publication barriers. The inherited optional WRITE-only
hints retain their 4 MiB size, 16 ms completion-to-next-start interval and FULL
publication pause. No public configuration option is added.

Five dedicated GNU-wrapper cases check actual notifications, real pinned WAL
generations and acknowledged content:

- While a real PASSIVE pass is paused, 32 commits produce exactly one new
  notification for pending work. The preceding budget library fails this
  deduplication assertion.
- Small CLEAN commits trigger a real partial PASSIVE pass even though uncopied
  bytes remain below 16 MiB. Releasing the reader makes the worker catch up
  without a further commit. The prior threshold candidate fails this case.
- A stream of 256 commits permits a PASSIVE pass to start before the stream
  ends; the batching deadline does not slide until the caller stops.
- The foreground commit callback performs zero checkpoint status queries. The
  prior threshold candidate fails this assertion.
- Stop is observed while a reader-pinned PASSIVE pass is paused. After its real
  partial result and reader release, Start catches up without another commit.

The current helper/source, negative-control identities and full logs are
retained. The original component tests, including OPEN batching and prompt
CLEAN progress, remain byte-identical; no scheduling expectation is relaxed.

## Comparison protocol

The original no-hint partial-edge implementation remains the adoption control.
There are six alternating AB/BA pairs per fresh/post-GC state on tmpfs and ext4
(48 plain mounts). The workload still writes first 256 MiB with 8 MiB application
buffers, then 64 MiB with 1 MiB buffers, at page offset 48 and fsync durability.
The two-millisecond parameter, complete workload and original gates are frozen
before the first comparison. Neither warmup/affinity manipulation nor
frequency, kernel/device/scheduler settings or global cache eviction is added.

An additional 24 plain ext4 mounts use the preceding 4 MiB/16 ms budget binary
as control. This isolates the scheduling change without substituting that
earlier failed candidate for the adoption control. The original analyzer's
checks are retained as references in `causal/summary.json`; this comparison
does not decide adoption. Finally, 16 separate ext4 mounts compare those same
two implementations with C++ request counters and read-only owned CPU sampling.
Diagnostic rates do not replace plain measurements.

The main decision requires both ext4 fresh/post-GC combined-write aggregates
to gain at least 10%, with at least five wins among six pairs in each state.
Every individual workload retains the original median/aggregate nonregression,
pooled p99 and lifecycle gates. All slow samples remain.

## Plain results

The two primary combined-write targets pass, but the complete decision is
[REJECT](summary.json). Combined rates divide all 320 MiB per mount by summed
complete write durations; individual medians are different statistics.

| ext4 state | No-hint control MiB/s | Candidate MiB/s | Aggregate change | Paired wins |
|---|---:|---:|---:|---:|
| Fresh | 112.52 | 126.66 | +12.6% | 5/6 |
| After GC | 109.12 | 123.87 | +13.5% | 6/6 |

Fresh ext4 random-write aggregate improves by 3.8%; mkdir, rename and rmdir
aggregates are 0.980, 0.976 and 1.038 times control. The earlier threshold-only
CLEAN deferral's large metadata regression is not reproduced at that scale.
Remaining failures include:

- Subsequent-64 MiB application-write p99 rises from 74.336 to 96.266 ms fresh,
  and 73.514 to 92.548 ms after GC. The complete-write aggregate ratios are
  0.970 and 0.997, but these p99 increases exceed the original latency gate.
- After-GC small-file fsync has nearly identical medians (250.057 versus
  250.198 files/s), while aggregate throughput falls to 0.410 times control.
  Candidate repetition 0 takes 4.944 s for 128 files (25.888 files/s), with a
  longest individual operation of 295.439 ms. It remains in the result.
- After-GC ext4 rename median falls from 1894.779 to 1771.435 operations/s;
  rmdir median falls from 2178.859 to 1933.918. Tmpfs fresh namespace operations
  and several after-GC random-write/small-file/mkdir checks also fail.

The [additional budget-control comparison](causal/summary.json) preserves a
second complete plain window. Fresh combined aggregate falls from 129.09 to
80.44 MiB/s (0/6 wins); after-GC aggregate is 126.47 versus 127.22 MiB/s (3/6).
These are new paired measurements, not ratios across the earlier window.

In that window, first-256 MiB fresh aggregate is almost unchanged (ratio
1.003), but subsequent-64 MiB fresh aggregate falls to 0.431 times control.
Candidate repetitions 1 and 5 take 5.456 and 5.485 s, including maximum
application write requests of 1.469 and 1.288 s, and final fsync of 1.687 and
3.059 s. The control also has retained after-GC random-write and small-file
tails; its slow samples make some candidate aggregates appear much faster.
All rows remain, and these reference results do not override the primary
no-hint rejection.

## Diagnostic attribution

The diagnostic baseline is the preceding budget implementation. Current-WAL
queries in commit callbacks are absent in both variants. In fresh 64 MiB phases,
query-only worker overlaps fall from 129 to 63 per mount, while both variants
retain two PASSIVE overlaps. Median cumulative foreground admission is 0.233
versus 0.248 ms, rather than the earlier threshold candidate's 141 ms. Fresh
rename retains 18–27 PASSIVE overlaps in the candidate, with median final
application fsync of 2.757 ms. The control has 26–34 overlaps and 3.442 ms fsync.
The small-tail backlog observed in the previous experiment is avoided here.

The scheduling change does not remove substantial work. The 64 MiB write's
VM count remains 1,528,000, and fresh median handler CPU rises from 198.313 to
211.896 ms at the same sampled 2.298 GHz. Median hook time falls from 0.428 to
0.057 ms, a saving too small to dominate the complete operation. Stage times
are inclusive; worker overlaps cover whole spans, not just time inside a phase.
CPU sampling is an estimate rather than a cycle count.

[Slow overlapping spans](diagnostic/slow-spans.json) retain every recorded span
at least 100 ms long that overlaps a measured phase, including callbacks that
start before its timer. They expose two separate costs:

- Fresh candidate repetition 3's first write takes 4.290 s. One FUSE write
  lasts 3.032 s with only 4.888 ms thread CPU; 2.988 s is in its publication
  wait. An overlapping checkpoint worker spends 2.923 s in payload fdatasync
  while performing publication. A real optional hint also lasts 360.425 ms.
  Overlap alone does not establish that the hint caused the entire sync stall.
- After-GC control repetition 2's 64 MiB write takes 5.231 s. Its final fsync
  lasts 3.028 s, including 2.933 s payload synchronization. A 1.080 s write
  overlaps a 1.076 s checkpoint worker with 1.043 s in fsync. Long syncs remain
  possible in the unchanged budget control too.

There is also a large CPU cost hidden from write-handler-only summaries.
In fresh repetition 0, the prior file's FUSE `forget` starts before the next
64 MiB phase and overlaps its file creation. It executes **9,830,796 SQLite VM
steps** in both variants, lasting 956.684 ms in control and 988.103 ms in the
candidate, with almost all that time on CPU. Remaining callback time overlaps
the next open for approximately 756 and 775 ms. The complete workload timer
already includes this queued work; no timing boundary is changed to remove it.

The source path is `ops.forget` → `Close` → `Session::CloseFile` →
`CleanupOrphans`. On the last inode reference, orphan cleanup walks every
visible block and calls `Put` individually to publish tombstones. This same
per-block cleanup is present in both libraries; the scheduling patch does not
address it. Its timing/VM evidence is stronger than another timer adjustment
as a target for subsequent-file latency and CPU work.

The next C++ experiment should batch orphan-block tombstone publication while
preserving one atomic transaction, branch/snapshot intervals and open-handle
visibility. The payload-sync publication lock remains a separate long-tail
target. Pre-synchronization outside the hot gate is worth testing only while
retaining the real final FULL barrier and rechecking current work under the
gate; simply moving the authoritative sync outside it can publish unsynced
concurrent appends. Neither path is implemented or adopted by this report.

## Validation and retention

The installed candidate passes 154 related Python tests with one explicit
unsupported production-v1 migration skip. Control/candidate each pass all 11
C++ suites, and the candidate passes all 11 under ASan/UBSan. The five dedicated
cases and all 24 inherited writeback cases pass in Release and sanitized builds.
The helper report has 67 executed cases, including four expected negative
controls, plus two sanitizer helper compilations. Only related suites run.

Both native and diagnostic builds finish before measurement starts. The audit
verifies disjoint windows and no owned build/test overlap, all content hashes,
format 1004, valid payload lengths, SQLite `quick_check`, stopped/unmounted
daemons, removed owned data and frozen source/module/SDK identities. Source
reconstruction uses the inherited budget patch and this scheduling patch with
fuzz 0. Split arrays preserve every parsed record and the exact primary and
causal analyzer results. The manifest indexes full local request/CPU/host
vectors, native sources, modules, libraries and test executables. Large owned
database/payload fixtures are removed outside timed phases. Each published JSON
is below 500 KiB.

## Reproduce

Use CPython 3.12 and the frozen source/build roots from the preceding reports.
The candidate installs noneditably in its own Release incremental build. The
diagnostic build uses the same pinned SDK and unchanged request recorder.

```bash
REPO=/home/kaka/astrovela_vane-fs
BENCH="$REPO/vane_fs/benchmarks/staged_payload/checkpoint_batching"
BASE="$REPO/vane_fs/build/external-partial-20261011T083153Z"
BUDGET="$REPO/vane_fs/build/writeback-budget-20261011T165013Z"
REJECTED="$REPO/vane_fs/build/checkpoint-coalescing-20261012T041450Z"
OUT="$REPO/vane_fs/build/checkpoint-batching-new"
DIAGNOSTIC_OUT="$REPO/vane_fs/build/batching-diagnostic-new"
TEST_SUPPORT=/home/kaka/astrovela_vane-batch-udf-20260930/.venv-gpu/lib/python3.12/site-packages

"$REPO/.venv/bin/python" "$BENCH/prepare.py" "$OUT" \
  --parent "$BASE" --budget-parent "$BUDGET" --rejected-parent "$REJECTED" \
  --sqlite-prefix "$REPO/vane_fs/vcpkg_installed/x64-linux-release" \
  --fuse-prefix "$REPO/.cache/fuse-sdk/root/usr" --test-support "$TEST_SUPPORT"
"$REPO/.venv/bin/python" "$BENCH/validate.py" "$OUT"
"$REPO/.venv/bin/python" "$BENCH/prepare_diagnostic.py" "$DIAGNOSTIC_OUT" \
  --parent "$OUT" --sqlite-prefix "$REPO/vane_fs/vcpkg_installed/x64-linux-release" \
  --fuse-prefix "$REPO/.cache/fuse-sdk/root/usr"
"$REPO/.venv/bin/python" "$BENCH/run.py" "$OUT" --medium tmpfs
"$REPO/.venv/bin/python" "$BENCH/run.py" "$OUT" --medium ext4
"$REPO/.venv/bin/python" "$BENCH/../partial_sync/analyze_writeback.py" "$OUT"
"$REPO/.venv/bin/python" "$BENCH/run_causal.py" "$OUT"
"$REPO/.venv/bin/python" "$BENCH/../partial_sync/analyze_writeback.py" "$OUT/causal"
"$REPO/.venv/bin/python" "$BENCH/run_diagnostic.py" "$DIAGNOSTIC_OUT"
"$REPO/.venv/bin/python" "$BENCH/../request_costs/analyze.py" "$DIAGNOSTIC_OUT"
"$REPO/.venv/bin/python" "$BENCH/../request_costs/analyze_cpu.py" "$DIAGNOSTIC_OUT"
"$REPO/.venv/bin/python" "$BENCH/analyze_diagnostic.py" "$DIAGNOSTIC_OUT"
"$REPO/.venv/bin/python" "$BENCH/publish.py" "$OUT" --diagnostic "$DIAGNOSTIC_OUT"
```

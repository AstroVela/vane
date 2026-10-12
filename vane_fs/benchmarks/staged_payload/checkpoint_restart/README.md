# Optional background WAL restart

This isolated C++ prototype attempts a lock-nonblocking WAL RESTART after a
completed large PASSIVE checkpoint. **It is not adopted:** 18 original checks
fail, and 11 reference checks fail against preceding no-hint orphan batching.
The production 4 KiB SQLite BLOB component and installed module are unchanged.

## Hypothesis and runtime change

Earlier orphan-control traces showed 200 PASSIVE calls during fresh random
writes: 199 reported equal frame/backfill counts, yet 200 WAL syncs and only
four database syncs occurred. Post-GC counts were 159, 157, 159 and three.
`rca.json` records and hashes those eight baseline traces. SQLite's reported
checkpoint head can become stale while concurrent writers advance the live
head; equal returned counts do not prove that a generation can be reset.

Only `src/checkpoint.hpp` differs from orphan batching. After the existing
protected publication and a successful PASSIVE with equal counts and at least
16 MiB of frames, the worker temporarily sets its busy timeout to zero and
attempts RESTART under the same checkpoint lease. It restores the normal
timeout, treats BUSY as an optional failure, preserves partial-progress retry,
and retains existing error reporting before the next mutation or barrier.

Zero timeout prevents waiting for SQLite reader/writer locks. It does not
prevent synchronous I/O. In the pinned SQLite 3.53.2 source,
`sqlite3WalCheckpoint` falls back to PASSIVE if the writer lock is unavailable;
the call can still backfill and sync before returning BUSY. `walCheckpoint`
syncs the database when its safe head equals the live WAL head at completion.
These are observations from the exact local amalgamation and headers, not
assumptions from WAL file length.

`staging.hpp`, `workspace.cpp` and `payload_file.hpp` remain byte-identical.
The WAL commit callback and write admission bodies, 16/64 MiB thresholds,
OPEN publication, final FULL commit, CLEAN fence, recovery, layout and GC are
unchanged. There is no new format, index removal, payload budget or writeback
hint. `restart.patch` replays with fuzz zero over preceding orphan batching;
`reproducer-proof.json` verifies all 35 reconstructed source inputs.

## Original adoption window

The original no-hint partial-edge control, workload and gates remain frozen:
48 uninstrumented mounts, tmpfs/ext4, fresh/post-GC, six alternating pairs per
medium/state. The application buffer has page offset 48. Complete file creation,
256 MiB writes in 8 MiB calls, subsequent 64 MiB writes in 1 MiB calls, final
fsync and close remain inside the large-write timers. Each variant uses its own
installed native Python module for GC; module path/hash and index-presence
checks run outside workload timers.

| Medium/state | Control combined aggregate MiB/s | Candidate | Change | Wins |
| --- | ---: | ---: | ---: | ---: |
| tmpfs fresh | 197.74 | 323.59 | +63.6% | 6/6 |
| tmpfs post-GC | 195.04 | 313.63 | +60.8% | 6/6 |
| ext4 fresh | 111.76 | 138.08 | +23.6% | 6/6 |
| ext4 post-GC | 110.14 | 96.69 | -12.2% | 5/6 |

Ext4 first-256 median rates are 153.59→153.44 MiB/s fresh and
152.82→149.16 post-GC. Subsequent-64 medians are 53.56→96.83 and
51.82→94.38, benefiting from the inherited orphan batching as well as the
new candidate. A 7.081-second post-GC candidate first-write phase remains in
the aggregate; no slow sample is excluded or rerun away.

Eighteen checks fail: fresh tmpfs random writes, small-file fsync and rmdir;
post-GC tmpfs random writes and small-file fsync; fresh ext4 subsequent-write
p99, random writes, mkdir, rename and rmdir; post-GC ext4 first-write aggregate,
subsequent-write p99, random writes, small-file fsync, all three directory
operations and combined large writes. Fresh control random-write and small-file
fsync tails also remain and inflate some aggregate candidate gains.

The unchanged gates require at least 10% ext4 combined aggregate improvement
and five wins out of six in each state; every workload needs median and
aggregate ratios of at least 0.95. Pooled p99 must be within 1.20 times the
control or within 1 ms absolute growth, and lifecycle median within 1.05 times.
Reference and diagnostic results do not replace these adoption gates.

## Independent comparison with preceding orphan batching

Another 24 uninstrumented ext4 mounts isolate RESTART over the preceding
candidate, using the same six alternating pairs and independent GC modules.

| State | Combined median MiB/s, control→candidate | Aggregate MiB/s | Wins |
| --- | ---: | ---: | ---: |
| fresh | 137.36→139.67 (+1.7%) | 105.81→140.12 (+32.4%) | 5/6 |
| post-GC | 136.02→133.52 (-1.8%) | 106.15→133.69 (+25.9%) | 3/6 |

Control first-write and subsequent-write phases of 5.929 and 4.695 seconds
raise the aggregate improvement. A 7.737-second post-GC candidate small-file
fsync phase remains. Eleven reference checks fail: random writes, small-file
fsync, mkdir, rename and rmdir in both states, plus post-GC combined writes.
Fresh directory medians decrease 33.6%, 42.1% and 36.6%; post-GC decreases
are 31.7%, 39.4% and 33.4%. Random-write medians decrease 25.3% and 23.2%.

## Typed diagnostic and root cause

Sixteen separate instrumented ext4 mounts compare orphan batching and the
prototype, with four alternating pairs per state and read-only 5 ms CPU
sampling. These rates do not decide adoption. The recorder reconciles all
5,077 sync calls against per-request counts and durations and retains all
87 calls of at least 100 ms, including calls outside workload intervals.
Sync spans can overlap; inclusive durations must not be added as elapsed time.

Across the eight candidate mounts, 360 optional RESTART attempts produce
106 successes and 254 BUSY results. Calls returning BUSY still perform 229
WAL syncs, totalling 860 ms of inclusive sync time. Successful attempts perform
44 WAL and 44 database syncs. The return code alone is not an I/O-cost measure.
`diagnostic/restart-costs.json` separates whole-mount outcomes from overlapping
phase costs, with the added restart span and `wal_restart` sync-site tag.

Successful reuse also changes subsequent small-commit behavior: the unchanged
WAL hook wakes only at 16 MiB of total generation frames. The smaller generation
therefore receives less background syncing until it grows again. This moves
work into the final barrier rather than eliminating it. Fresh mkdir and rmdir
candidate phases have median zero overlapping workers, compared with 14 and
16 in the control. The observed final application barriers are:

| Phase | Fresh control→candidate, ms | Post-GC control→candidate, ms |
| --- | ---: | ---: |
| random write | 4.03→29.70 | 4.20→35.65 |
| mkdir | 3.63→20.58 | 4.39→20.64 |
| rename | 2.65→11.04 | 3.23→7.46 |
| rmdir | 3.81→20.87 | 4.85→10.43 |

Statement counts and VM instructions remain equal for corresponding short
phases. Foreground CPU time and sampled frequency also differ: fresh mkdir
CPU is 35.29→43.55 ms and its runtime-weighted sampled frequency is
2,146→1,489 MHz. The diagnostic identifies final sync growth and changed
background activity, but does not attribute every CPU-time difference to one
cause or assume other machine load. No affinity, governor or device setting
was changed.

The post-GC control's first-write phase takes 8.166 seconds. Its checkpoint
worker spends 6.752 seconds in one payload-file sync at `payload_final`, while
foreground publication wait totals 7.011 seconds. That sync overlaps no other
recorded sync call. The final application fsync is only 179 ms. This directly
locates that tail in payload publication and its writer backpressure, rather
than WAL restart. It does not prove the same cause for every plain-window tail.

The next experiment should preserve prompt CLEAN small-commit background
progress while separating that policy from WAL-generation reuse. First use a
narrow replay to compare dirty-WAL work and final barriers. For large-file
tails, investigate payload publication batch size and publication-lock scope
with the captured sync evidence before changing another WAL policy.

## Related validation and retention

All related tests pass: 154 installed Python tests; 11 original-control,
11 candidate and 11 ASAN/UBSAN component tests; 53 positive helper cases and
one expected old-orphan failure. Eight new cases run in Release and ASAN/UBSAN:
generation reuse, current/older pinned readers, a held writer lock, an external
writer waiting for the checkpoint lease, storage-error propagation/retry,
small-WAL suppression and abrupt exit before RESTART with two cold reopens.
Inherited CLEAN/continuous/stop and orphan/snapshot/GC/recovery cases remain.
One existing production-v1 migration test is skipped for experimental format
1004. Only related tests run; no base or Ray suite runs. The additive typed
recorder separately passes Release and ASAN/UBSAN validation for 25 calls,
all six file kinds, errno, descriptor reuse, nesting and concurrency.

The initial test-only inspector had not read the schema, so SQLite returned
-1/-1 without opening the WAL. The reader case failed; the initial progress
case's less-than test was too weak. `preflight/` retains that failed report,
logs and helper source, and hashes both original helper binaries. The corrected
helper reads the schema and requires nonnegative frame counts. Native runtime
binaries remain unchanged; all related validation runs again before measuring.

All 88 mounts validate content, retained snapshot guards and SQLite integrity,
then stop, unmount and remove their data. `audit.json` verifies 148 removed
paths, including failed preflight and test data, and no build/test/formatter
overlap with measurement windows. There is no global cache eviction. JSON
vectors are split losslessly below 500 KiB; sources, binaries, SDKs, logs,
CPU samples and host measurements remain locally with hashes in `manifest.json`.
Only measurements and necessary small logs are committed.

## Reproduce

`build.json` pins the SQLite 3.53.2 fsync SDK, FUSE 3.14 SDK, three installed
native modules and exact source inputs. Given the retained original partial-edge
and orphan-batching parents, use fresh output directories:

```bash
python vane_fs/benchmarks/staged_payload/checkpoint_restart/prepare.py "$restart_out" \
  --parent "$partial_parent" --orphan-parent "$orphan_parent" \
  --sqlite-prefix "$sqlite_prefix" --fuse-prefix "$fuse_prefix" \
  --test-support "$test_support"
python vane_fs/benchmarks/staged_payload/checkpoint_restart/validate.py "$restart_out"
python vane_fs/benchmarks/staged_payload/checkpoint_restart/prepare_diagnostic.py "$restart_diag" \
  --parent "$restart_out" --sqlite-prefix "$sqlite_prefix" --fuse-prefix "$fuse_prefix"
python vane_fs/benchmarks/staged_payload/checkpoint_restart/run.py "$restart_out" --medium tmpfs
python vane_fs/benchmarks/staged_payload/checkpoint_restart/run.py "$restart_out" --medium ext4
python vane_fs/benchmarks/staged_payload/partial_sync/analyze_writeback.py "$restart_out"
python vane_fs/benchmarks/staged_payload/checkpoint_restart/run_causal.py "$restart_out"
python vane_fs/benchmarks/staged_payload/partial_sync/analyze_writeback.py "$restart_out/causal"
python vane_fs/benchmarks/staged_payload/checkpoint_restart/summarize.py "$restart_out"
python vane_fs/benchmarks/staged_payload/checkpoint_restart/run_diagnostic.py "$restart_diag"
python vane_fs/benchmarks/staged_payload/request_costs/analyze.py "$restart_diag"
python vane_fs/benchmarks/staged_payload/request_costs/analyze_cpu.py "$restart_diag"
python vane_fs/benchmarks/staged_payload/checkpoint_batching/analyze_diagnostic.py "$restart_diag"
python vane_fs/benchmarks/staged_payload/checkpoint_restart/analyze_restart.py "$restart_diag"
```

Build and validate the diagnostic before the measurement windows if reproducing
the original ordering. Do not run builds, tests or formatting concurrently with
any measured window. The publication script consolidates this frozen run,
including its archived inspector preflight, and rechecks every gate, patch,
identity and cleanup path without changing the measurement boundaries.

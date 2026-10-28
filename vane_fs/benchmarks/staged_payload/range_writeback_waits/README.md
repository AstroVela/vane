# Range-writeback tails and subsequent WAL synchronization

The slow paths differ: fresh write p99 outliers occur in **OPEN-batch WAL/fence
synchronization before the first range submission**, while the reproduced
post-GC random-write and small-file stalls occur in **foreground WAL fsync**,
often alongside the checkpoint worker's WAL synchronization. The range hints
also block for seconds in large-write phases. These are measured call paths,
not proof that concurrent calls cause every stall or a kernel/device diagnosis.

The isolated [range-writeback trial](../payload_range_writeback/README.md)
still fails **7 of its original 22 performance checks**. This matched,
instrumented four-pair-per-state window does not replace those gates or its
plain throughput numbers. Production format 2, its installed module and the
default SQLite SDK remain unchanged. Both diagnostic variants use format 1005.

## Fresh first-256 write p99

The pooled application write-call p99 is 33.127 ms in the control and 93.584 ms
in the candidate. The two largest candidate calls are both the **first 8 MiB
application write** in a new workspace. Each includes its constituent FUSE
requests and the first OPEN-batch preparation.

| Fresh candidate repetition | Application write, ms | Own WAL fdatasync, ms | Own fence fdatasync, ms | Native CPU, ms | Outside handlers, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 182.116 | 62.531 | 87.311 | 26.829 | 5.660 |
| 3 | 115.422 | 2.564 | 80.430 | 27.390 | 5.273 |

WAL/fence synchronization occupies 82.3% and 71.9% of those application
intervals. Neither overlaps a background sync or range submission. The first
range call in each workspace begins **141.554 and 139.592 ms after** the
application write ends. The 64 MiB threshold has not yet been reached during
the first 8 MiB request. Range submission in that workspace cannot directly
explain these particular outliers.

The control's two longest first-256 write calls take 36.481 and 35.199 ms,
including 5.872 and 5.128 ms of WAL/fence synchronization. Their native CPU is
25.859 and 25.239 ms. The large candidate latency difference is located in
synchronization, rather than a similar increase in native CPU or application
time outside handlers. The original uninstrumented p99 failure remains; these
four diagnostic pairs do not establish its frequency or hardware cause.

## Post-GC random overwrite and small files

Candidate repetition 1 reproduces both regressions. The seeded random phase
still performs 512 distinct 4 KiB overwrites, followed by its mandatory FULL
barrier. That final barrier takes **1.236551 s**, including **1.236091 s in one
foreground WAL fsync (99.96%)**. Caller CPU is 3.956 ms and outside-handler time
is 0.090 ms. Payload and fence synchronization are absent from this barrier.

A concurrent **checkpoint-worker WAL fsync takes 1.206295 s**, with 0.912286 s
overlapping the foreground application interval. It belongs to a PASSIVE
checkpoint whose reported backfill advances from 115 to 1,694 frames. The foreground's database,
checkpoint, publication and payload-guard acquisition times are small; the
second thread's interval must not be added to the first as independent wall
time.

The following 128 small create/write/fsync/close operations take **3.779586 s**.
Their own WAL syncs sum to **3.426403 s (90.7%)**, while handler CPU sums to
274.369 ms and outside-handler time to 91.340 ms. There are **39 operations
of at least 20 ms**; 34 overlap another thread's WAL sync and five do not.
No random-write or small-file operation in this bad repetition overlaps a
range submission. The two slowest small-file operations are:

| Application operation, ms | Own WAL fsync, ms | Handler CPU, ms | Outside handlers, ms |
| ---: | ---: | ---: | ---: |
| 190.668 | 187.893 | 2.086 | 0.820 |
| 149.016 | 146.383 | 2.072 | 0.678 |

Across the four post-GC pairs, pooled small-file p99 is 6.654 ms in the control
and 95.649 ms in the candidate. All eight post-GC small-file phases have
**128 foreground WAL fsyncs and 162,944 caller SQLite VM steps**. The bad
candidate phase and its normal control counterpart each have 70 worker WAL
syncs; their phase times are 3.780 and 0.634 s. Thus that pair's regression is
not an increased foreground sync count, worker WAL sync count or SQL work.
Worker database sync counts and schedules differ, and time overlap alone does
not isolate the cause. All post-GC random-write phases have 356,208 caller VM
steps. The candidate's write-call p99 remains below 1 ms there; its slow final
barrier dominates the bad complete phase.

The current checkpointer latches its warmed state after 16 MiB and thereafter
receives commit notifications across WAL generation reuse. It coalesces small
CLEAN maintenance on a fixed 5 ms deadline, but its PASSIVE I/O can overlap a
foreground FULL barrier after releasing the publication mutex. That mechanism
exists unchanged in both variants. SQLite documents that checkpointing also
synchronizes WAL before copying pages into the database; see its
[WAL performance discussion](https://sqlite.org/wal.html#performance_considerations).
Foreground FULL and checkpoint syncs are both real durability work. The
observed overlap is a reason to test coordination, not to remove either
operation's durability guarantees.

## Range submissions and remaining large-file barriers

All **37 range calls** use positive finite lengths, WRITE only, the payload
descriptor and successful results. Controls issue none. Worker roots have
zero publication-held and checkpoint-held spans. The longest range call takes
**3.437113 s** in the candidate's post-GC first-256 phase. In that phase, the
foreground final barrier takes **5.169058 s**, including **5.164265 s of payload
fdatasync**, and overlaps that call and a subsequent 2.059719 s range call.
The overlapping intervals are not additive costs.

Fresh candidate range calls also reach 3.283754 and 3.166056 s. Moving the
submission to a worker avoids holding the application mutexes during that
call; it does not make the syscall bounded or remove the mandatory final
payload sync. WRITE-only can itself block, as the
[Linux interface documentation](https://man7.org/linux/man-pages/man2/sync_file_range.2.html)
specifies. The control also has a 3.989711 s payload sync without a range
worker. These traces do not prove concurrent submission as the sole cause.

## Next controlled experiment

Prioritize a causal control for **checkpoint overlap with explicit FULL
barriers** before changing checkpoint timing or the 64 MiB range trigger.
Use the same prepared native workspace and fixed random/small-file sequence
with a diagnostic-only checkpoint-worker pause, compared with the unchanged
worker. Apply both settings to the extent control and range candidate so the
shared checkpoint mechanism can be separated from the range change.

Both settings must drain identical preparation before timing and retain every
foreground FULL barrier and the existing 64 MiB synchronous admission path.
Measure worker resumption, deferred maintenance and final close as well as the
foreground phase; a faster phase that merely moves work beyond the timer is
insufficient. Check WAL high-water size, progress after resumption, reads,
snapshots, fault propagation and close retry. If a coordinated implementation
is justified, it must retain bounded WAL progress and recovery ordering, then
pass the unchanged full FUSE gates. No coordination change is implemented here.

The separate cold OPEN-batch barriers remain another target. The present
evidence does not justify deleting the WAL-prefix/fence barriers. The earlier
[SQLite sync-primitive trial](../sync_primitive/README.md) is also retained;
its partial small-file gains and failed gates do not justify a default SDK
change for this candidate.

## Validation and reproduction

Sixteen serial ext4 mounts execute four AB/BA pairs in each fresh and
independently prepared post-GC state. The complete first-256/subsequent-64,
warm/random read, random overwrite, small-file and namespace workloads remain
unchanged apart from diagnostic timestamps. Every large-file hash, random
slice, overwritten fixture, small file, retained GC guard, format/fence and
SQLite integrity check passes, including **32 large-file content checks**.

Release and ASan/UBSan recorder integration tests pass, including actual range
submission and EBADF/errno preservation. Existing algorithm correctness tests
are reused by frozen identity and exact source reversals. No broad base/Ray
suite runs. **223,265 native roots and 206,118 events** pass disjoint wall/CPU
accounting, and **11,824 application operations** reconcile with handler IDs.
Every actual sync/range call in the measurements succeeds.

The initial analyzer matched random writes by time overlap only. A previous
FUSE reply's 1,242 ns trace tail overlapped the next syscall, producing two
matches with different offsets. The original analyzer remains frozen;
[analyze_intervals.py](analyze_intervals.py) adds exact logical offset/length
matching with an exact reversal proof. It reuses the already successful native
analysis without rerunning it or any measurement. The preflight, sample roots,
adapter proof and hashes are embedded in [findings.json](findings.json).

```bash
here=vane_fs/benchmarks/staged_payload/range_writeback_waits
out="$PWD/vane_fs/build/range-writeback-waits-repro"
python3 "$here/prepare.py" "$out" \
  --comparison "$PWD/vane_fs/build/payload-range-writeback-retained-20261018/fuse"
python3 "$here/run.py" "$out"
python3 "$here/analyze.py" "$out"
# If the frozen time-only matcher hits the retained reply-tail preflight,
# save its evidence as "$out/analysis-preflight.json", then finish without rerunning:
# python3 "$here/analyze_intervals.py" "$out"
# After recording final source checks:
python3 "$here/publish.py" "$out"
python3 "$here/publish_lines.py" "$out"
```

Builds and tests finish before measurement; analysis follows it. There is no
cache eviction, host tuning or load-based filtering. All samples, roots,
events, application intervals, CPU/host observations, code identities and
source/runner reversals are retained losslessly. [The audit](audit.json) covers
**18 removed data directories**, after owned processes and mounts stop.
Large workspace data is removed; indexed Git artifacts stay below 500 KiB.
The publication-only line-boundary adapter avoids splitting JSON indentation
at a chunk's EOF and verifies identical reassembled bytes; it changes no values
or measurement inputs.

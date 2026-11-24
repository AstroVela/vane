# Kernel waits in the complete format-1005 FUSE workload

The cold fence tail is reproduced and the long payload/WAL calls now have
conservative kernel wait attribution. A fresh first-8-MiB application write
takes **113.662 ms**, including **81.229 ms** inside its own fence fdatasync.
All 11 matched observations of that call report `jbd2_log_wait_commit`.
The earlier approximately 1 ms ordinary fence-allocation saving does not
explain this tail. This diagnostic introduces no storage implementation,
installed-module or default SQLite SDK change. The range candidate still has
**7 failed ordinary FUSE checks out of 22**.

## Unchanged workload and identities

Sixteen serial ext4 mounts reuse the exact native diagnostic binaries from
the [range-wait investigation](../range_writeback_waits/README.md): four AB/BA
pairs in each fresh and independently prepared post-GC state. The control
has ordinary extent writes; the candidate adds retained-FD finite-range
WRITE-only payload submission. Both use format 1005 and the original fsync
SQLite SDK. The rejected [fence-slot patch](../fence_slots/README.md) is absent.

The generated runner changes only its external observer's path, with exact
reverse proof. The workload is byte-identical to the frozen parent. Each mount
retains the full first-256/subsequent-64-MiB writes, final FULL, reads, 512 random
overwrites and final FULL, 128 small-file fsyncs and namespace operations.
Initial creation, GC preparation, unmount, maintenance and close remain in
their original timers and cleanup paths. No barrier, checkpoint trigger,
cache policy, format or SDK option changes. In particular, final FULL still
persists payload before FULL WAL metadata and the CLEAN fence.

## Matching rules and observation cost

The external sampler reads only owned daemon threads' proc entries and
read-only host counters. It brackets each observation with CLOCK_MONOTONIC
and reads syscall-before, wchan, the resolved FD and syscall-after. A point is
assigned to a native call only when:

- Its TID matches and its entire observation bracket is inside the call.
- Both syscall reads agree, including the syscall number and FD.
- The FD resolves to the same descriptor class as the native trace.
- A range call's offset, length and WRITE-only flags also match.

The analyzer keeps every native call, including calls without observations.
Application attribution additionally matches caller PID, callback type,
start time and logical write range. A preceding callback's reply tail cannot
be assigned to the following write. Four focused attribution tests cover
boundary, syscall, FD, file and range mismatches, reply tails, and excluding
device snapshots that cross an event's edges.

There are **29,887 cycles and 70,309 thread observations**. Of 15,829 stable
sync observations, **15,805 match**; 24 remain unmatched in the raw evidence.
Another 564 candidate points have unstable/unavailable syscall identity and
313 cross a call's ending boundary; they are excluded from attribution and
retained. All **344 calls of at least 20 ms**, including all **19 calls of at
least one second**, have matching observations. Point counts are not time
accounting and are not converted to percentages of a call's elapsed time.

The delay is 5 ms after processing each cycle. The median of per-mount median
periods is **7.260 ms**; the largest gap is **86.766 ms**. Corresponding median
cycle duration is **2.114 ms**, with an **81.593 ms** maximum. The sampler uses
**58.659 CPU seconds** across **215.501 seconds** of summed mount lifecycles.
This is material diagnostic overhead; this window is unsuitable for ordinary
throughput comparison or adoption. No build, test or analysis overlaps it.

Every daemon's full kernel stack read is denied with errno 13. The sampler
retains the first failure and does not retry that denied interface repeatedly.
Wchan, syscall and FD observations remain available. Function names locate
wait classes; they cannot recover a complete kernel call chain, identify the
exact block QoS policy, or determine a journal transaction's owner.

## Reproduced cold fence tail

Fresh candidate repetition three's first 8 MiB comprises the same nine FUSE
write callbacks as before. Their combined CPU time is **24.372 ms**. Its own
WAL-prefix fdatasync takes **3.405 ms**, then fence fdatasync **81.229 ms** with
only **0.077 ms** thread CPU. The fence call has 11 matching journal-wait
points. No range-writeback call overlaps any of the 16 first-8-MiB intervals.
This tail therefore is not an overlapping range submission in these traces.

The other three fresh candidate first-8-MiB intervals are 26.301, 36.294 and
27.675 ms, with fence calls 2.332, 1.941 and 1.854 ms. All four fresh control
fence calls are 1.702–1.899 ms. These are diagnostic samples, not a latency
bound or a new performance gate. The evidence locates the slow syscall's
journal wait; it does not show whether fence allocation triggered that
transaction, what else it contained, or how materializing another slot would
affect the tail. The preceding native slot experiment's negative complete
screen remains unchanged.

## Payload and WAL tails

Maximum actual payload fdatasync durations, retaining all samples:

| State | Control, ms | Range candidate, ms |
| --- | ---: | ---: |
| Fresh | 5,271.406 | 4,743.987 |
| Post-GC | 12,418.525 | 4,043.961 |

The longest call is post-GC control repetition one's subsequent-64-MiB FULL.
It uses only **19.191 ms** thread CPU. Its 1,801 matching points include 818
`rq_qos_wait`, 251 `folio_wait_bit_common`, 708 `jbd2_log_wait_commit`, 23
`do_get_write_access` and one zero wchan. This shows block QoS, folio and
journal waits inside the actual payload syscall; it is not explained by SQL
execution during those 12.419 seconds. The control has no range worker, so
concurrent optional hints cannot be the sole cause of the shared long tails.

The candidate still has long submission calls despite using no WAIT flags:
fresh repetition three's **3,235.490 ms** range request has 314 `rq_qos_wait`
and 106 `do_get_write_access` points. Its following coalesced **192 MiB**
request takes **1,833.210 ms**. The foreground final payload sync concurrently
takes **4,743.987 ms**. Post-GC repetition one's **2,610.525 ms** range call
has 183 QoS and 152 `wait_transaction_locked` points. All **39** measured
range calls succeed, with finite positive ranges and WRITE-only flags.
The existing trace checks continue to reconcile exclusive stages and held
publication/checkpoint intervals; submission does not hold those mutexes.

WAL is independently important. Post-GC control repetition two's final random
FULL contains a **1,320.295 ms WAL fsync**, with **4.463 ms** thread CPU and
106 folio plus 83 journal-wait points. A checkpointer's WAL fsync in that mount
reaches **1,347.435 ms**. Overlap is retained but does not establish that the
worker caused the foreground delay. The earlier
[complete checkpoint-pause control](../checkpoint_pause_control/README.md)
already failed to improve complete work; that negative result still applies.

## What the device counters support

Every cycle also retains all 17 `/sys/block/sda/stat` counters and their read
brackets. A call's contextual deltas use only snapshots entirely inside it,
with uncovered edges explicit. There are no device read failures or detected
counter regressions. All sampled in-flight counts are zero in this run; no
queue-depth claim is made from that field.

The 12.419 s payload call's interior covers 12.413 s, during which the whole
device records 13 completed flushes and 6,846 flush milliseconds. Those
values are not that file's flush count or elapsed flush cost. The 2.611 s range
call has a 2.599 s interior with no completed flush increment and 335 matching
QoS/journal-lock points. Zero completed flushes does not exclude a flush that
was already in flight or completed outside the interior. It also does not
make the entire syscall duration measured device-flush time.

The host uses kernel `6.11.0-24-generic`, ext4 on `/dev/sda2`, mq-deadline and
a recorded `wbt_lat_usec` of 2000. These settings are read after the run and
unchanged by the experiment. The launcher cgroup and ancestor settings are
retained; daemon cgroup paths were not captured during measurement. Do not
infer a particular cgroup policy, an SSD fault or unrelated workload from
these observations.

The upstream Linux 6.11
[ext4 sync path](https://github.com/torvalds/linux/blob/v6.11/fs/ext4/fsync.c)
waits for file data and the appropriate journal transaction and can issue a
flush. Its [rq_qos implementation](https://github.com/torvalds/linux/blob/v6.11/block/blk-rq-qos.c)
and [writeback throttling caller](https://github.com/torvalds/linux/blob/v6.11/block/blk-wbt.c)
explain why QoS is a wait category rather than proof of one configured policy.
These are upstream references, not a symbol-exact Ubuntu kernel build.
[Kernel I/O statistics documentation](https://www.kernel.org/doc/html/v6.11/admin-guide/iostats.html)
describes whole-device, potentially combined flushes and counter limitations.

## Validation, retention and next experiment

Only related tests run: **6 sampler tests**, including a real blocked owned
child and access-denial/error reporting; **4 attribution tests**; and the
unchanged **Release and ASan/UBSan recorder contract executables**. All pass.
All **221,919 roots and 202,093 events** reconcile exclusive wall/CPU
accounting. All **6,535 actual sync/range calls succeed**. All 11,824 recorded
application intervals, 32 large-file hashes, overwrite/read/small-file checks,
GC guards and SQLite integrity validations pass. No broad base/Ray suite runs.

Root formatting, lint, copyleft and all eight new source headers pass. The
repository-wide SPDX baseline retains five historical omissions, identical
at HEAD and in the working tree. An initial final-check harness assumed a
stale count of seven; its corrected assumption and original logs remain in
the validation preflight evidence. No measurement or storage code changes
were needed for that correction.

[plan.json](plan.json) freezes native/source/module identities and the complete
measurement before it starts. [findings.json](findings.json) contains the
groups, all second-scale calls, first requests and sampling limitations.
[final-checks.json](final-checks.json) records source checks, the unchanged
11 production identities and stopped-process/mount checks.
[audit.json](audit.json) indexes every raw native trace, sampler/device/host
row, all per-call joins including unmatched calls, generated runner/workload,
logs, commands, native source snapshots and cleanup manifest. Concatenate an
entry's ordered `retained_as` chunks, gzip-decompress once, and verify the
original size/SHA-256. Every published artifact is below 500 KiB.

All **18 owned data directories** are removed after their processes stop:
two recorder fixtures and 16 mounted workspaces. No owned mount/process
remains. Data cleanup is outside the comparison timers. There is no cache
eviction, host/device tuning, load filtering or unowned process termination.

The next bounded software experiment can apply **small, paced range claims
and no new claims during FULL** to the current format-1005 candidate. Its
purpose is to test the observed coalesced-submission/QoS exposure, retaining
the original final FULL and complete lifecycle. A pause cannot cancel an
already active syscall or eliminate the control's independent payload/WAL
tails. A similar [format-1004 trial](../writeback_budget/README.md) failed nine
nonregression checks; preserve that result and do not assume the new extent
layout will pass. Require the same unchanged complete FUSE gates before
adoption. This diagnostic itself implements no new writeback policy.

```bash
here=vane_fs/benchmarks/staged_payload/fuse_sync_waits
out="$PWD/vane_fs/build/fuse-sync-waits-repro"
# Requires the frozen parent diagnostics, original SDK and non-editable venvs.
python3 "$here/prepare.py" "$out"
python3 "$here/run.py" "$out"
python3 "$here/test_analyze.py" -v
python3 "$here/analyze.py" "$out"
# Record final source/identity/cleanup checks as "$out/final-checks.json".
python3 "$here/publish.py" "$out"
```

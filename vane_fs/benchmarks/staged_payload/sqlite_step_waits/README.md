# SQLite transaction-start and sync wait diagnosis

The slow-SQL hypothesis is now narrower: **104 BEGIN IMMEDIATE calls invoke
the default SQLite busy handler**, making **331 actual nanosleep calls**.
All 104 transaction-start intervals and all 331 sleeps overlap another
thread's RESTART checkpoint. The sleep requests are exactly 1, 2, 5 and
10 ms. SQL identity and actual callback stack frames establish busy backoff
in this window; this is not an inference from low CPU time.

This supports an isolated transaction-start/RESTART coordination experiment.
It does not explain every slow operation: small-file tails still spend about
92% of application time in WAL fsync, and two payload fdatasync calls take
**3.078 and 3.806 seconds**. No performance policy is changed here. The
[paced candidate](../paced_range_writeback/README.md) retains **all nine
ordinary FUSE gate failures**, and the historical uninstrumented 1.62-second
first post-GC write remains unattributed.

## Frozen comparison and recorder

Both arms retain format 1005, the original fsync SQLite SDK, original extent
control versus 4 MiB/16 ms paced hints, complete workloads and four AB/BA
pairs per fresh/post-GC state. There are **16 mounts** and **32 large-file
hash checks**. No native algorithm, scheduling, disk format, publication
barrier, default installed module or production SDK changes.

The diagnostic source transformations reverse exactly to both frozen native
implementations. The previous recorder's 28 stages remain, with one added
sleep stage. Exclusive wall/CPU accounting still partitions each native
root. SQLite's busy handler and timeout are not replaced.

- Intern only the unexpanded SQL text associated with retained events.
  Record the statement and API identity, per-thread invocation identity,
  VM steps and return code for slow step/exec calls.
- Wrap the SDK's actual nanosleep reference. Preserve arguments, remainder,
  return value and errno. Record every sleep and native stack PCs, normalized
  to the containing mount image. Symbolize using that exact binary's nm
  table after measurement. No live symbolization subprocess runs.
- Add stack PCs to actual fsync/fdatasync/range events. The existing event
  columns remain in place; SQL/context/API/VM-step/stack fields are appended.
- Expose current sync/sleep calls through a **4096-byte memfd** with 16
  cache-line-sized seqlocked slots. The observer maps it **read-only**.
  Only calls still active after **20 ms** trigger a kernel observation,
  at most once per thread per 20 ms. The original 5 ms CPU/frequency census
  remains. The marker makes no database or payload writes and disappears
  with process exit.

Accept a kernel point only when the marker is unchanged, TID/kind/native
begin/FD/arguments match, the complete observation bracket lies within the
native call, and syscall number/FD/file class remain consistent. Retain
changed, unavailable and rejected points. A nearby syscall or a previous
callback's reply tail cannot be substituted.

The 210.941-second window contains **224,220 native roots, 211,496 events,
11,824 application intervals, 7,007 sync/sleep calls and 2,609 slow SQL calls**.
Every complete workload passes content, retained-GC-guard, read/overwrite,
namespace, fence and SQLite integrity checks.

## The measured SQL wait is transaction acquisition

All observed sleeps have the actual stack path through unixSleep,
sqliteDefaultBusyCallback, btreeBeginTrans, sqlite3_step, Statement::Step and
Transaction construction. Their SQL is **BEGIN IMMEDIATE**, not a data
lookup/insert or COMMIT. There are no observed sleeps from the separate
quadratic WAL-read retry path in this window.

| Request | Number |
| --- | ---: |
| 1 ms | 104 |
| 2 ms | 104 |
| 5 ms | 104 |
| 10 ms | 19 |

Total requested sleep is **1,022 ms**, actual sleep **1,049.650 ms**.
For one fresh candidate write, the BEGIN lasts **18.925 ms**, including
**18.422 ms** in its four sleeps and only **0.602 ms** thread CPU. The whole
application overwrite takes **23.698 ms**, including a later WAL sync.

Every one of these 104 BEGIN intervals, and every one of the 331 sleeps,
overlaps another thread's checkpoint API call with RESTART mode.
[contention evidence](evidence/contention.json/000.gz.part) retains the
exact SQL and checkpoint intervals and event references. The linked SQLite
code needs the writer lock for a WAL write transaction and also for RESTART;
the Vane worker already uses busy_timeout(0) around its RESTART attempt.
That avoids waiting inside the worker but does not prevent it acquiring the
writer lock first and sending the foreground BEGIN into default backoff.

This identifies a specific contention mechanism, supported by the SQL,
callback frames, checkpoint mode and source semantics. It does not timestamp
the exact internal SQLite unlock. Across the 104 cases, BEGIN completes
0.189–9.955 ms after the last overlapping RESTART returns, median 1.818 ms.
That gap includes final retry/work and is an opportunity indicator, not a
promised optimization gain.

Not every long COMMIT is backoff. One 31.795 ms random write spends
**31.044 ms in a 24-byte WAL pwrite64**, with no own sync or sleep. The
typed write event and disjoint counters distinguish that I/O from SQL work.

## Actual synchronization remains a separate limit

| Observed call | Wall | CPU | Contained, matched named wait points |
| --- | ---: | ---: | --- |
| Fresh candidate, 64 MiB final payload fdatasync | 3.078109 s | 16.165 ms | rq_qos_wait 58; folio_wait_bit_common 51; jbd2_log_wait_commit 7 |
| Post-GC control, 256 MiB final payload fdatasync | 3.806198 s | 80.133 ms | rq_qos_wait 98; do_get_write_access 22; folio_wait_bit_common 19 |
| Post-GC candidate WAL fsync in a barrier COMMIT | 304.773 ms | 2.147 ms | jbd2_log_wait_commit 11 |

Payload stacks pass through Staging::BeforeFullCommit, Transaction::Commit
and Workspace::Sync. WAL commit stacks pass through unixSync, pagerWalFrames
and the SQLite commit path. Background checkpoint WAL syncs have a distinct
walCheckpoint stack and checkpoint API context. These are actual call
durations, rather than attributing an entire application fsync to one file.

The kernel points establish where sampled calls waited. They do not
identify a unique device/firmware cause or prove that a concurrent hint or
checkpoint caused the wait. A no-hint control still has a multi-second
payload sync. All 175 candidate hints remain finite WRITE-only calls; their
maximum observed duration is 21.107 ms.

All 2,048 small-file operations still have exactly one foreground WAL fsync
each and no own payload/database/fence sync. The candidate's **seven >=20 ms
small-file operations** spend **219.925 of 239.242 ms, 91.93%**, in WAL fsync.
Post-GC small-file p99 is **8.529 ms candidate / 6.276 ms control** in this
diagnostic window. The four candidate first post-GC overwrites take
**3.143–10.307 ms**; their syncs are now identified as part of COMMIT.
The earlier 1.62-second first-write sample is not reproduced or explained
retrospectively by this trace.

## Observer cost and attribution limits

There are **835 targeted kernel points: 788 accepted and 47 rejected**
(45 unstable/unavailable syscall observations, two changed markers). Every
point has a matching native call identity; rejected points remain retained.
Kernel stack reads are denied with errno 13 and that denial is visible.

The complete observer consumes **30.915 CPU seconds across 163.536 seconds
of summed observed process time**, or **18.90% of one CPU core**. The CPU
census and frequency reads remain continuous; native recorder overhead is
additional. Targeting kernel reads reduces the earlier observer's scope but
does not make this window an ordinary throughput comparison. These results
are for attribution only and cannot clear any of the nine ordinary failures.

## Next isolated change

Test a dedicated gate between **foreground transaction acquisition** and
**the worker's RESTART attempt**. Release the foreground gate as soon as
BEGIN succeeds. Have the worker try the gate without waiting and defer its
RESTART attempt if transaction acquisition is already in progress.

Keep PASSIVE copying independent, retain the existing WAL admission budget
and let unrelated connections retain SQLite busy handling. Do not hold the
whole checkpoint mutex across ordinary requests: that would make writers
wait for PASSIVE disk I/O that need not own the SQLite writer lock. Check
publication/GC lease ordering, failed-close retries, external writers,
FULL bypass and progress before running the unchanged ordinary gates.
This proposal is not implemented or validated by the present trace, and
cannot remove the independent journal/payload synchronization cost.

## Validation, analysis history and retention

Both original recorder contracts and the new SQL/sleep contracts pass in
**Release and ASan/UBSan: four native executions**. The new contracts exercise
real default busy retry and timeout, a custom SQL sleep that is not busy
backoff, SQL control-character escaping, EINTR/remainder and EINVAL behavior.
The read-only observer additionally matches eight real blocked-child points.
**Six new observer/matching tests and four inherited attribution tests pass**.
Parent algorithm suites are inherited by exact source reversal, not claimed
as newly run. Only related tests run.

Initial analysis completed all request joins but stopped while reading an
omitted workload-proof.json. The workload itself was copied byte-identically
from the earlier frozen diagnosis. The initial native analysis, failing logs,
command records and request SHA remain retained. Copying the unchanged
provenance file permits analysis to finish; the repeated request JSON is
byte-identical. No measurement or native build is repeated for this correction.
[provenance.py](provenance.py) makes that metadata step explicit in a replay.

[diagnosis.json](diagnosis.json) summarizes SQL/sleep classification, stack
examples, application tails and observer accounting. [final-checks.json](final-checks.json)
checks frozen/default identities and cleanup. [audit.json](audit.json) indexes
all native events, application vectors, symbols, samples, logs, source proofs,
contracts, the analysis preflight and reference-source excerpts losslessly.
Concatenate an entry's ordered gzip parts and decompress once. Every
published file is below 500 KiB.

All **22 owned directories** are removed after processes stop: 16 mounts,
four native contracts, one observer probe and one syntax fixture. No owned
process/mount remains. Build/tests, measurement and analysis are separate;
cleanup is outside application timers. No cache eviction, device/CPU tuning,
load-based filtering or unowned process termination occurs.

~~~bash
here=vane_fs/benchmarks/staged_payload/sqlite_step_waits
out="$PWD/vane_fs/build/sqlite-step-waits-repro"
python3 "$here/prepare.py" "$out" \
  --comparison "$PWD/vane_fs/build/paced-range-writeback-20261019-v2/fuse"
python3 "$here/verify.py" "$out"
python3 "$here/run.py" "$out"
python3 "$here/provenance.py" "$out"
python3 "$here/analyze.py" "$out"
python3 "$here/attribute.py" "$out"
# Record final source/identity/process checks as "$out/final-checks.json".
python3 "$here/publish.py" "$out"
~~~

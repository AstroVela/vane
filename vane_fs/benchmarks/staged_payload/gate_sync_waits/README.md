# Matched gate, checkpoint and durable-sync attribution

This diagnostic locates newly captured multi-second pauses in the same paced
parent and BEGIN/RESTART-gate candidate as the [ordinary matched comparison](../begin_gate_matched/README.md).
It does not promote either candidate or change the default implementation. All
**nine failed ordinary performance checks remain**. The exact historical
4.666/6.930-second samples were uninstrumented; the traces below describe new
samples, rather than assigning causes retrospectively to those old calls.

The strongest result is that two fresh-workspace final fsync pauses spend
**99.5%/99.9% of their application wall time in actual payload fdatasync**.
The new gate also produces measurable millisecond waits during random writes,
but its wait in those two final fsyncs is only about 1.7 microseconds.

| State / arm / repetition | Application operation | Application time, ms | Located work |
| --- | --- | ---: | --- |
| Fresh / gate / 0 | 64 MiB final fsync | 3,898.729 | Payload fdatasync 3,878.871 ms; gate wait 0.001674 ms |
| Fresh / gate / 5 | 256 MiB final fsync | 7,117.282 | Payload fdatasync 7,111.421 ms; gate wait 0.001648 ms |
| Fresh / gate / 0 | First 64 MiB write call | 1,071.750 | BeforeWrite -> PrepareExternal PASSIVE checkpoint 906.212 ms, including WAL fsync 501.285 ms and database fsync 316.926 ms; gate wait 0.006417 ms |
| Post-GC / parent / 1 | Random-write final fsync | 1,298.259 | COMMIT WAL fsync 1,297.769 ms |

The first write row includes more than the PrepareExternal checkpoint: the complete
handler has 975.677 ms exclusive fsync and 80.114 ms exclusive fdatasync across
its files. Its handler CPU is 17.280 ms, with only 0.693 ms outside handlers.
Its native stack and checkpoint mode locate the slow call in the synchronous
CLEAN-prefix preparation before an external write. This is PASSIVE, not the
worker's gated RESTART or evidence that the 64 MiB admission limit was reached.
For the two slow large final fsyncs, handler CPU is 18.348/71.546 ms and time
outside handlers is 0.082/0.080 ms. These measurements separate native I/O waits
from application/FUSE scheduling and SQL CPU work.

Actual payload calls have conservative kernel observations in `rq_qos_wait`,
`folio_wait_bit_common` and `do_get_write_access`; the 64 MiB call also has
`jbd2_log_wait_commit` observations. Their accepted point counts are 63 and
116. The parent WAL COMMIT call has 21 accepted points in rq_qos/folio/jbd2
waits. Native stacks, SQL identities, syscall return values, file classes and
all accepted/rejected points are retained. Point counts are observations, not
a time breakdown or proof of a unique device, scheduler or competing-load cause.

## What the gate changes

Both arms preserve the 4 MiB claims, 16 ms completion-to-next-start pacing,
64 MiB activation threshold, FULL pause, SQLite SDK, publication order and WAL
admission limit. The only algorithm difference remains checkpoint.hpp and
workspace.cpp from the already frozen gate candidate. Diagnostic wrappers
restore both source trees byte-for-byte; the new gate spans only measure the
real mutex acquisition and the successful worker hold around RESTART.

Within the **8,868 mapped candidate application intervals**, 37 have at least
1 ms total gate wait. The maximum is **22.705 ms** fresh / **14.213 ms** post-GC.
Recorded slow waits total 321.994 ms; 320.257 ms (99.46%) overlaps the measured
worker gate holds and 320.071 ms overlaps its RESTART calls. The largest
23.276 ms random-write operation spends 22.705 ms waiting for the gate,
overlapping 22.654 ms of RESTART. A zero SQLite busy timeout prevents SQLite
lock retries; the RESTART call can still perform synchronous I/O while holding
the local gate.

The parent has **236 actual SQLite default-busy nanosleeps** (53 fresh / 183
post-GC); the gate candidate has zero. Removing that backoff does not remove
waiting for the RESTART I/O. For fresh 4 KiB random writes in this diagnostic,
mean application time is 519.45 -> 580.88 microseconds: the candidate averages
59.19 microseconds of gate wait, while the parent averages 29.18 microseconds
in actual busy sleep. Handler CPU also changes from 417.57 to 444.17
microseconds. This is a decomposition of the instrumented window, not proof
that all the earlier ordinary regression comes from one lock. Fresh 256 MiB
write-call medians here are similar (14.466 -> 14.490 ms); the previous
ordinary median regression is not consistently reproduced by this trace.

## Payload submission and WAL tails

All 24 observed 64 MiB write phases submit just **one 4 MiB WRITE-only hint**
before their final fsync begins. The 64 MiB activation threshold leaves little
opportunity to submit this phase early. In the slow 256 MiB candidate phase,
only four hints request 16 MiB before final fsync, and the submission calls
themselves take 299.407 ms. The other candidate fresh 256 MiB phases request
72--84 MiB in 18--21 calls. WRITE-only submission can itself block; return
does not certify completion or durability. These are requests during the
application write window, not a measurement of remaining dirty bytes or
proof that a particular amount reached stable storage.

Post-GC small-file bursts also remain on both arms. Parent repetition 1 has
41 operations >=20 ms; candidate repetition 4 has 59. Their maximum complete
operations are 276.471/238.486 ms. The largest actual WAL COMMIT fsyncs are
273.750/235.887 ms, with folio/jbd2 observations. The earlier uninstrumented
fresh parent burst is not retroactively attributed by these post-GC samples.

The next isolated experiment should start finite payload submission earlier
and evaluate a bounded schedule driven by queued ranges, retaining the FULL
pause and every payload/WAL/fence durability barrier. Measure requested ranges
and submission latency as well as final-sync tails: faster nominal pacing is
not enough when WRITE-only submission itself blocks. Separately, test fewer
opportunistic RESTART attempts during active writers while retaining PASSIVE
progress, cross-process safety and the 64 MiB admission bound. Neither is
implemented or claimed as a demonstrated improvement in this report.

## Protocol and verification

Local artifacts: `vane_fs/build/gate-sync-waits-20261019`.
The control is the matched **paced parent**, not the older no-hint extent
control. A recorded metadata adapter selects that frozen source without
changing its files. Twenty-four full FUSE mounts run serially on ext4: fresh
and post-GC each use six alternating AB/BA pairs. The diagnostic workload is
byte-identical to the earlier application-interval workload; original workload
reversal and generated runner reversal are retained. Each GC preparation uses
its own variant's isolated non-editable native module and the unchanged SDK.
There is no cache eviction, CPU/device tuning, load filtering or overlapping
owned build/test/analysis work during the measurement window.

The native recorder has 31 stages, adding actual BEGIN/RESTART mutex wait and
worker hold to the existing SQL/sync accounting. All **336,479 roots and
317,113 events** satisfy exact exclusive wall/CPU accounting; all **17,736
application intervals** map to native handlers. There are 502 WRITE-only range
calls, 236 actual busy sleeps and 3,820 slow SQL events. Every sample remains,
including the four application intervals >=1 second.

The observer reads the native memfd marker every 20 ms, considers calls active
for >=20 ms, and observes a TID at most once per 50 ms. It omits the old
continuous thread/frequency census. Every accepted point still requires exact
native sequence, TID, call kind, FD/class/arguments, syscall and time-bracket
agreement. **735 points pass and 30 are rejected**, all retained. Observer
thread CPU is **1.910 seconds across 252.806 seconds of observation**, or
**0.755% of one core**. Native recording and host monitoring add separate
overhead, so these diagnostic throughputs are not ordinary benchmark results.

Only related tests run: six observer units, the real-child read-only marker
probe, and six C++ contract executions across Release and ASan/UBSan. The
contracts cover actual publication/holds, SQL busy versus non-busy stacks,
errno/remainder, nested accounting, mutex contention and exception unwinding.
Unmodified algorithm suites are inherited by exact frozen identity and source
reversal. All **24 mount validations and 48 large-file hashes pass**.

Twelve native-source reversals and four recorder reversals are checked again
after measurement, along with frozen inputs and 11 unchanged defaults.
All **31 owned data directories** (seven contract/probe and 24 mount fixtures)
are removed after their processes stop, outside application timers; manifests
retain file sizes and hashes. Source/SDK/build environments remain available.
The report-write setup failure occurred before measurement, created no data
fixture and changed no native identity; its original source, build manifest,
failure and exact runner-only correction are retained.

`findings.json` and `final-checks.json` provide compact results. `audit.json`
indexes deterministic gzip parts below 500 KiB containing all raw traces,
kernel points, SQL/stack symbols, source snapshots, inputs, logs, post-window
queries, setup failure and cleanup manifests. Concatenate each stream's
`retained_as` parts in order and decompress to recover its original bytes.
The post-window `inspect.py` and `attribution.json` retain query inputs and
hashes in addition to the generic application joins. No candidate is adopted.

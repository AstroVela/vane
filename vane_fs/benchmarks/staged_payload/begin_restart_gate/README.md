# Local BEGIN/RESTART coordination

A dedicated C++ gate removes the reproduced SQLite busy-backoff mechanism
between a connection's foreground BEGIN IMMEDIATE and its own worker's
RESTART checkpoint. The combined paced-plus-gate candidate passes **19 of
the original 22 ordinary FUSE performance checks**. It remains isolated:
fresh rename, post-GC random-write p99 and post-GC rmdir still fail, and a
supplementary candidate fsync still takes **4.190 seconds**.

The [preceding SQL/sleep diagnosis](../sqlite_step_waits/README.md) identified
104 BEGIN calls making 331 default-busy-handler sleeps, all overlapping
RESTART. This experiment implements that proposed coordination, rather than
changing the SQLite SDK or the durable publication protocol.

## Native change and lock ordering

Only checkpoint.hpp and workspace.cpp differ from the frozen
[paced parent](../paced_range_writeback/README.md). The
[patch](policy.patch) and retained source manifest reverse exactly to that
parent. Format 1005, 4 MiB WRITE-only range hints, 16 ms completion delay,
64 MiB payload trigger, retained barrier descriptor and balanced FULL pauses
remain unchanged. Production code, installed default module, SQLite SDK and
durability defaults retain their original identities.

The foreground performs existing admission/error checks, acquires the new
gate before publication/payload guards, and releases it immediately after
successful SQL BEGIN. RAII releases it on failed guard/pragma/BEGIN paths.
It is released before external writes can request a checkpoint lease.

The worker leaves PASSIVE outside this gate. Only its opportunistic RESTART
uses try_lock; an unsuccessful attempt skips that optional restart and
releases its checkpoint lease. It never waits for the new mutex while
holding that lease. Existing zero-timeout RESTART handling, busy-timeout
restoration, 16/64 MiB WAL start/admission budgets, FULL bypass and
payload-to-WAL-to-CLEAN ordering remain in place. Subsequent commits still
wake the warmed worker. Foreign connections retain SQLite busy handling.

## Controlled mechanism tests

The regression VFS pauses a real RESTART **after SQLite obtains the actual
WAL writer lock**. It does not substitute a fake SQLITE_BUSY result for
that contention. The unchanged paced parent enters BEGIN and makes actual
default-handler nanosleeps in both NORMAL and FULL controls. The candidate
does not enter BEGIN until RESTART releases the gate, and makes zero busy
sleeps in those same cases. Both Release and ASan/UBSan execute these controls.

Seven candidate cases pass in each build:

- NORMAL and FULL foreground exclusion during the held real RESTART.
- Foreground write progress during blocked real PASSIVE database I/O.
- A held BEGIN gate makes background RESTART skip and release its lease;
  the following committed tail is subsequently backfilled.
- The gate is released immediately after BEGIN, before later SQL work.
- Failed FULL BEGIN releases guards and permits retry and checkpoint progress.
- A real foreign SQLite writer still causes bounded Busy and atomic retry.

The initial progress fixture incorrectly required a later RESTART call.
A retained probe shows the writer completed and the checkpoint lease was
released with zero restarts. SQLite can reuse a fully backfilled WAL without
that separate call. The corrected assertion observes actual backfill of the
new committed tail. Original failing logs, source/fixture snapshots, binaries,
cleanup and the probe remain retained. Only the external contract fixture
and retry bookkeeping changed; native source/binary identities remained
identical, so successful component builds/tests were reused.

## Ordinary performance window

The contemporaneous control is the original **no-hint extent** implementation,
also format 1005. The candidate combines the already paced parent with the
new gate. These throughput results do **not** isolate incremental gate gains
from pacing, and the historical nine paced failures are not a matched third
arm. Controlled tests establish the coordination mechanism separately.

Freeze the unchanged workloads, durability settings and original 22 gates
before measurement. The primary window has six AB/BA pairs per fresh/post-GC
state, **24 mounts**. The supplementary window has two pairs per state and
four consecutive 256+64 MiB cycles per live workspace, **8 mounts**. All
**112 complete large-file hashes**, retained-GC-guard hashes, overwrite/read,
namespace, fence, format and SQLite integrity checks pass.

Complete 320 MiB phase rates include all requests, final fsync and close:

| Primary window | No-hint median MiB/s | Candidate median MiB/s | Paired wins |
| --- | ---: | ---: | ---: |
| Fresh workspace | 227.55 | 256.34 | 6/6 |
| After GC | 225.84 | 256.13 | 6/6 |

Median improvements are **12.65% fresh and 13.41% after GC**. The retained
fresh aggregate is **151.68 to 259.28 MiB/s**, strongly affected by a control
64 MiB final fsync of **4.266 s**. The post-GC aggregate is **227.32 to
256.79 MiB/s, +12.97%**. No slow sample is removed.

For the 256 MiB phase, median final fsync is **612.026 to 476.352 ms** fresh
and **623.467 to 467.663 ms** after GC. Its median complete throughput is
**247.51 to 281.39 MiB/s** fresh and **238.75 to 280.54 MiB/s** after GC.
These are measurements of the combined candidate, not isolated mutex effects.

Three unchanged primary checks fail:

| Check | No-hint control | Candidate | Failure |
| --- | ---: | ---: | --- |
| Fresh rename median, operations/s | 2,012.19 | 1,825.99 | -9.25%; aggregate also -5.24% |
| Post-GC random-write pooled p99, ms | 0.631 | 3.383 | Exceeds 20% and 1 ms allowances |
| Post-GC rmdir median, operations/s | 2,133.54 | 1,970.62 | -7.64% |

Post-GC random-write aggregate improves from **6.24 to 8.14 MiB/s** despite
the p99 failure. Fresh small-file aggregate is **263.32 to 255.31 files/s**;
post-GC is **256.73 to 264.77 files/s**. Both small-file checks pass this
window, but that does not erase the remaining failures.

Supplementary four-cycle aggregate rates are **229.24 to 258.52 MiB/s**
fresh and **143.28 to 172.15 MiB/s** after GC. The latter has only **one of
two paired wins**: the control has a **5.567 s** final fsync, and the
candidate has a **4.190 s** final fsync. Thus the retained aggregate does
not establish consistent post-GC long-tail improvement. Ordinary application
timings do not attribute those individual pauses to a syscall/device or
identify the cause of the three failed checks.

## Validation and retention

Only related tests run: **153 Python passed, two explicit skips**, 12 C++
component suites in Release and 12 in ASan/UBSan, plus per build 12 extent,
eight pacing, 14 inherited fault, eight actual FULL-pause and seven new gate
cases. The two parent negative controls also pass per build. The skips are
production-v1 migration unavailable for isolated format 1005 and optional
PyAV unavailable; the inherited-fork warning remains explicit.

[findings.json](findings.json) retains every failed check and complete phase
accounting. [final-checks.json](final-checks.json) audits frozen/default
identities and source reversal. [audit.json](audit.json) indexes raw vectors,
host samples, source/fixture histories, commands, logs and cleanup hashes
losslessly. Concatenate an entry's ordered gzip parts and decompress once;
each published file is below 500 KiB.

All **50 owned data directories** are removed after their processes stop, outside
application timers. Sources, SDKs, candidate environments and incremental
builds remain reproducible. Build/tests, ordinary measurement and analysis
do not overlap. No global cache eviction, CPU/device tuning, load-based
sample filtering or unowned process termination occurs.

The gate's contention fix is validated; the combined candidate still fails
adoption criteria. A matched paced-parent versus gate-only performance
window would isolate its incremental effect before any independent backport.
The remaining metadata/p99 failures and multi-second durable barriers need
their own evidence rather than further tuning of this mutex.

~~~bash
here=vane_fs/benchmarks/staged_payload/begin_restart_gate
out="$PWD/vane_fs/build/begin-restart-gate-repro"
python3 "$here/generate.py" "$out" \
  --parent "$PWD/vane_fs/build/paced-range-writeback-20261019-v2"
python3 "$here/build.py" "$out"
python3 "$here/verify.py" "$out"
python3 "$here/full_pauses.py" "$out"
python3 "$here/prepare.py" "$out"
python3 "$here/run.py" "$out/fuse"
python3 "$here/analyze.py" "$out/fuse"
python3 "$here/summarize.py" "$out"
git add "$here"
python3 "$here/final_checks.py" "$out"
python3 "$here/publish.py" "$out"
~~~

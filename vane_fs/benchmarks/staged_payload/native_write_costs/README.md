# Native write CPU costs and borrowed input buffers

Borrowing the current write's input removes two intermediate data copies and
passes the complete native CPU screen. Across 48 paired native processes,
caller write CPU time falls **10.7–15.1%**, with all 24 pairs improving.
Ordinary FUSE acceptance nevertheless fails **10 checks**: fresh combined
large-write aggregate falls 20.7%, and both states win only four of six pairs.
The C++ change remains an isolated format-1004 prototype. Production format 2,
its installed module, SDK and scheduling remain unchanged.

This follows the [allocation investigation](../payload_allocation/README.md).
Small explicit reservations increased physical fragmentation and did not
eliminate long barriers. This investigation first separates software work
from durable-device waiting, then implements and measures a bounded buffer
change. It does not change the 64-block SQL batches or durability boundaries.

## Matched native attribution

The unchanged staged core and default SQLite fsync SDK perform fresh 64/256 MiB
writes in 1 MiB calls on ext4 and tmpfs. Six alternating medium orders per
size produce **24 processes**, each with an independent workspace. Timing
includes workspace construction, root/mount setup, file creation, all writes,
the final FULL `Workspace::Sync()`, flush and file release. Validation and
workspace close are measured separately; buffers are prepared outside timing.
The worker remains enabled with its original 16/64 MiB thresholds.

`CLOCK_THREAD_CPUTIME_ID` measures the calling thread; process CPU includes
other threads. These clocks have distinct documented
[meanings](https://man7.org/linux/man-pages/man3/clock_gettime.3.html).
Every phase and request retains wall, thread and process CPU times. A read-only
5 ms sampler records owned-thread runtime, runqueue delay, migration and CPU
frequency. The old interval estimator is reused with monotonic timestamps.
Frequency values are sampled estimates; no result is frequency-normalized.

| Unchanged core, median | Writes wall, ms | Caller write CPU, ms | Final barrier wall, ms |
| --- | ---: | ---: | ---: |
| ext4 / 64 MiB | 147.77 | 143.80 | 178.96 |
| tmpfs / 64 MiB | 133.56 | 133.55 | 0.17 |
| ext4 / 256 MiB | 1,246.73 | 559.82 | 130.77 |
| tmpfs / 256 MiB | 505.01 | 474.08 | 0.16 |

The 64 MiB tmpfs write phase is almost entirely caller CPU. Moving storage
to tmpfs does not remove software costs. The 256 MiB ext4 write phase also
contains substantial off-CPU time; its final barrier alone is not the whole
durability cost because managed publication/checkpoint work can occur earlier.
Wall-minus-thread-CPU is not itself an attribution to a particular syscall.

Eight later diagnostic processes add source scopes, real API/syscall wrappers,
and a regular C++ `operator new` counter. The source generator reverses exactly
to the frozen original; only `workspace.cpp`, `payload_file.hpp` and an added
counter header differ. Exclusive wall/CPU counters subtract complete child
spans and sum exactly to the phase root. Inclusive SQLite, checkpoint and I/O
times must not be added together. SQLite C allocator activity is outside the
C++ allocation counter, and worker CPU is outside caller stage totals.

The direct component statements retain actual
[SQLite VM-step counts](https://sqlite.org/c3ref/c_stmtstatus_counter.html).
The separate `sqlite3_exec` scope covers control SQL; its internal VM steps
are not claimed as part of the direct-statement counters.

| Diagnostic caller write CPU share | ext4 64/256 MiB | tmpfs 64/256 MiB |
| --- | ---: | ---: |
| Version insert execution, excluding nested I/O | 25.2% / 26.1% | 28.8% / 29.6% |
| Block construction and other WriteBytes work | 7.2% / 7.3% | 7.8% / 7.5% |
| Concatenate external data | 6.0% / 5.7% | 6.3% / 5.8% |
| Build batch SQL text | 2.1% / 2.1% | 2.3% / 2.3% |
| Prepare direct statements | 0.20% / 0.06% | 0.22% / 0.06% |

The block-construction remainder includes copying, allocation, zero checks
and other C++ work, not a pure memcpy timer. SQL re-preparation is small under
the existing cache; changing batch size does not follow from that observation.
There is a concrete copy path: construct one owned string per new 4 KiB block,
then concatenate 64 strings into a fresh 256 KiB payload string.

Diagnostic write CPU medians are 12.6–19.4% above the separate unlogged-core
window. Instrumentation and differing sampled CPU states prevent treating
that ratio as a causal overhead estimate. Diagnostic rates are not used to
claim an optimization; the implementation gets a subsequent paired window
with the uninstrumented cores.

## C++ buffer change

The [patch](borrow.patch) changes only the isolated core's `workspace.cpp`
and `payload_file.hpp`, verified by an exact reverse patch:

- Pending new blocks carry views into the immutable input of the current call.
  At most two padded boundary blocks use separate owned strings whose lifetime
  covers every pending view.
- A complete group of 64 consecutive file blocks borrows its span directly
  only when the whole span lies inside that input. Partial boundaries and
  skipped zero blocks retain the owned gather path.
- Payload append consumes a string view through the same synchronous pwrite
  loop, error handling and dirty flag. SQL bindings retain `SQLITE_TRANSIENT`
  copying. Views do not escape the call or reach the background worker.

The external threshold, batch count, rowids, interval/version algorithms,
token format, schema, publication/fence order, full barriers, worker and GC
remain unchanged. Occupied intervals retain their original owned data path.

An independent 4 MiB comparison links the regular allocation counter with
each uninstrumented core. Counted C++ requests fall from **2,417 to 1,377**;
requested bytes fall from **8,544,943 to 151,199**. The removed 1,040 requests
are 1,024 block strings and 16 concatenation buffers. Direct payload/version
VM counts remain exactly **21,664 / 70,864**. This is allocation evidence,
not a throughput estimate or a count of SQLite's C allocations.

## Paired native result

Six AB/BA pairs per medium/size produce **48 fresh native processes**. The
same probe links the unchanged or borrowed-buffer Release core and the same
SDK. Neither core has the internal stage/new counters. CPU clocks and the
read-only external sampler remain common. Source, library, wheel, binary,
workload and environment identities are frozen before the window.

Before measurement, the CPU screen requires at least 10% gains in median and
aggregate write CPU throughput and at least five paired wins per group.
Complete-operation median/aggregate throughput must retain at least 95%;
pooled request p99 stays within 1.2 times control or a 1 ms absolute increase;
median close time may grow at most 5%. All **28 checks pass**, with six CPU
wins in every group. All slower controls and candidates remain included.

| Medium / bytes | Caller write CPU before → after, ms | CPU time reduction | Complete median MiB/s before → after |
| --- | ---: | ---: | ---: |
| ext4 / 64 MiB | 145.94 → 130.27 | 10.7% | 182.74 → 187.91 |
| ext4 / 256 MiB | 572.95 → 503.77 | 12.1% | 185.13 → 192.69 |
| tmpfs / 64 MiB | 135.17 → 116.56 | 13.8% | 453.05 → 520.70 |
| tmpfs / 256 MiB | 486.01 → 412.41 | 15.1% | 486.17 → 566.60 |

Ext4 complete median throughput improves 2.8% / 4.1%; tmpfs improves
14.9% / 16.5%. Ext4 256 MiB aggregate improves 58.2%, influenced by a
retained slow control; the median is the more modest result above. CPU-rate
gain and CPU-time reduction are different statistics. All vectors and gates
are retained in [compare-summary.json](compare-summary.json).

## Ordinary FUSE acceptance

The unchanged reference workload runs **24 independent ext4 mounts**: six
AB/BA pairs fresh and after independent fragmentation/GC preparation. Each
variant uses its own non-editable module for GC. The first file is 256 MiB
using an 8 MiB application buffer at page offset 48, followed by 64 MiB using
1 MiB calls, reads, overwrites, small files and namespace operations. Every
content, retained guard, SQLite integrity and format check passes.

The original complete gates remain: combined large-write aggregate gains
at least 10% with at least five wins in each state; all workloads retain 95%
of median/aggregate throughput; request p99 stays within 20% or 1 ms; median
lifecycle grows at most 5%. Native CPU success does not replace these gates.

| Combined 256 + 64 MiB, aggregate MiB/s | Original | Borrowed | Change | Wins |
| --- | ---: | ---: | ---: | ---: |
| Fresh | 70.65 | 56.02 | −20.7% | 4/6 |
| After GC | 48.87 | 58.06 | +18.8% | 4/6 |

Ten checks fail. Fresh first-256-MiB aggregate falls 115.88 to 60.25 MiB/s;
fresh small-file aggregate falls 245.57 to 81.21 files/s. After GC, subsequent
64 MiB aggregate falls 37.35 to 19.44 MiB/s and random-overwrite aggregate
falls 8.60 to 4.52 MiB/s. Small-file, mkdir and rename gates also fail.
Fresh median lifecycle rises 10.71 to 13.52 seconds. Full results are in
[fuse-summary.json](fuse-summary.json).

Both variants retain long calls. A fresh candidate first-write phase takes
13.642 seconds, with 13.478 seconds in application writes and only 0.160
seconds in final fsync. Another candidate's subsequent-write phase takes
5.913 seconds, including 4.206 seconds in fsync. A control takes 5.805 seconds,
including 3.461 seconds in fsync. The after-GC candidate also has a 6.451-second
subsequent write, including 4.164 seconds in fsync. These application timers
locate the wait; this plain window does not identify the responsible native
mutex, file sync or device request. CPU savings cannot be used to erase these
failures or assign their causes.

## Validation, scope and reproduction

The candidate passes **153 related Python tests**, 12 C++ component suites,
12 ASan/UBSan suites and **24 new boundary/snapshot/GC cases** across strict
and fsync modes and Release/sanitized builds. New cases use distinct data,
partial and occupied boundaries, skipped zero blocks, and input mutation
immediately after write and before fsync. Both source scopes and API wrappers
are separately checked for nested accounting, real VM counts and preserved
errno. The measurement probes also validate full content and format on every
run. Only related tests run.

Two Python skips are explicit: format 1004 has no production-v1 migration,
and the optional image/video comparison lacks PyAV. The inherited-fork test
emits CPython's expected multithreaded-fork deprecation warning. A retained
environment preflight first lacks pytest, then discovers that the host's
Arrow 22 is below Vane's declared Arrow 25 requirement; missing runtime
dependencies also initially skip integration cases. Dependencies are restored
only inside the candidate environment, and the Python suite is rerun.
Unchanged successful native results are reused by source/binary identity.
All original logs, harness versions and the final package freeze are retained.

The [audit](audit.json) verifies 80 native measurements, 24 FUSE mounts,
nonoverlapping build/test/measurement windows, unchanged default identities,
and **116 unique cleaned owned directories**. Processes stop before deletion.
All phase/request vectors, counter values, physical data hashes, CPU/host
samples, source/binary identities and gate failures remain in indexed files
below 500 KiB. Reconstructing every split record must equal the original JSON.
Large databases and payload outputs are removed outside timed phases.

Reproduction starts from the frozen component pair in the
[preceding matched-SDK report](../sync_primitive/README.md):

```bash
RUN="$PWD/vane_fs/build/native-write-costs-reproduction"
BENCH="$PWD/vane_fs/benchmarks/staged_payload/native_write_costs"
python3 "$BENCH/prepare.py" "$RUN" --parent "$PWD/vane_fs/build/sync-primitive-20261013"
python3 "$BENCH/run.py" "$RUN"
python3 "$BENCH/run.py" "$RUN" --profile
python3 "$BENCH/analyze.py" "$RUN"
python3 "$BENCH/borrow.py" "$RUN"
python3 "$BENCH/compare.py" "$RUN"
python3 "$BENCH/compare.py" "$RUN" --analyze
python3 "$BENCH/fuse.py" "$RUN" --template "$PWD/vane_fs/build/checkpoint-progress-20261012T153141Z/fuse-screen"
python3 "$BENCH/publish.py" "$RUN"
```

The publisher refreshes this directory's evidence. Benchmark processes never
overlap owned builds or correctness tests. Global caches, CPU policy/affinity
and device settings are unchanged, and no sample is removed based on load.
Keep the buffer patch isolated. A next software design investigation should
quantify reducing per-4-KiB version/payload metadata, which still accounts for
substantial SQLite execution and WAL work; actual FUSE stalls need separate
wait attribution before another adoption claim.

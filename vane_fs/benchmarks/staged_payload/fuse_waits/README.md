# FUSE write and synchronization wait attribution

The long calls in the isolated format-1004 buffer comparison now have direct
request-level attribution. A 6.585-second write spends 6.577 seconds acquiring
the checkpoint mutex while the worker synchronizes payload. Another write
spends 6.371 seconds performing a foreground PASSIVE checkpoint. A final
application fsync spends 3.684 of its 4.480 seconds synchronizing payload and
another 0.794 seconds synchronizing WAL. Reducing C++ copying cannot remove
these observed waits.

This is a diagnosis of the two frozen components from
[native write costs](../native_write_costs/README.md), including their existing
CLEAN-prefix preparation policy. Both are experimental format 1004. This does
not change production format 2, the installed default, scheduling, durability
or the previous ten failed plain FUSE performance checks. Instrumented rates
are not an adoption comparison.

## Matched request timelines

Four balanced AB/BA pairs in each state produce **16 independent ext4 mounts**:
fresh and after independent fragmentation/GC preparation. The complete existing
workload is retained: first 256 MiB in 8 MiB calls, subsequent 64 MiB in 1 MiB
calls, reads, random I/O, small-file fsync and namespace operations. Application
buffers keep page offset 48. Both variants retain the default SQLite fsync SDK,
16/64 MiB WAL thresholds, publication ordering and the final FULL barrier.

The diagnostic sources add scopes around all 21 FUSE handlers, worker passes,
database/checkpoint/publication/lease waits, SQL execution, publication stages
and actual I/O calls. Reversing every insertion reproduces the frozen original
files exactly. Separate non-editable Release installations use persistent
`build/python-release` directories. The installed diagnostic modules are used
for their own GC preparation, with tracing disabled in those processes.

Each application large write and fsync has a `CLOCK_MONOTONIC` interval.
FUSE write handlers additionally record caller, inode, file offset and length.
Every application write reconciles with its exact contiguous handler byte
ranges, including all split requests. The handler includes the reply call;
it can end just after the application returns. Correlation clips interval
coverage and checks offsets rather than assuming strict time containment.

Both variants and all 16 mounts receive the same request shapes:

| Application phase | Applications | FUSE writes and byte lengths | Total handlers |
| --- | ---: | --- | ---: |
| 256 MiB | 32 × 8 MiB | 32 × 1,048,528; 224 × 1,048,576; 32 × 48 | 288 |
| 64 MiB | 64 × 1 MiB | 64 × 1,048,528; 64 × 48 | 128 |

The difference from aligned 1 MiB native writes is real. Partial padded edges
still require owned gather buffers, while FUSE also constructs the call's input
string. The previous native CPU gain is not a prediction of FUSE wall time.
The inherited [partial-edge handling](../partial_edges/README.md) already
batches vacant subintervals; split requests do not imply that every block
falls back to an inline BLOB.

Each handler has inclusive and exclusive wall/thread-CPU counters. Exclusive
children partition their root exactly; inclusive admission/checkpoint/SQL/I/O
times must not be added together. All **226,407 roots** pass both partition
identities. Worker intervals overlap foreground waits and are separate evidence
of the same elapsed time.

All actual sync/checkpoint calls are retained. Successful I/O and lock-span
events of at least 1 ms, plus syscall failures, retain typed intervals, thread,
request identity, file kind, result and errno. Complete per-root counters also
include shorter calls. This produces **121,761 events**. Records stay in memory
until process exit. The external 5 ms owned-CPU sampler and 1-second host monitor
are read-only; no global cache eviction, CPU/device setting or affinity changes
occur. Builds and tests finish before the diagnostic window; analysis starts
after it ends.

Instrumentation and teardown logging can affect timings and subsequent mounts.
These separate diagnostic samples do not attribute an individual slow call in
the preceding plain performance window.

## Three concrete slow paths

The examples below are selected deterministically from large-write timers in
[findings.json](findings.json). They are individual requests, not phase averages.

| Request | Handler wall | Caller CPU | Direct attribution |
| --- | ---: | ---: | --- |
| After-GC original, first-256 write at offset 102,760,400 | 6.585 s | 5.81 ms | 6.577 s checkpoint-mutex acquisition |
| Fresh original, first request of subsequent-64 write | 7.199 s | 75.66 ms | 6.371 s foreground PASSIVE checkpoint; 0.778 s checkpoint-mutex acquisition |
| Fresh borrowed-buffer, subsequent-64 final fsync | 4.480 s | 21.18 ms | 3.684 s payload fdatasync; 0.794 s WAL fsync |

The first request enters `BeforeWrite → PrepareExternal` before its SQL
transaction and waits on `checkpoint_mutex`. The worker holds that mutex over
publication and checkpoint I/O. Its payload `fdatasync` lasts 6.508 seconds;
the union of overlapping worker sync intervals covers 6.532 seconds of the
6.577-second mutex acquisition. These are overlapping descriptions of the
same stall. The sampled foreground runqueue delta is zero over the complete
sampling intervals, while direct caller CPU is only 5.81 ms. The trace supports
checkpoint serialization behind real I/O, rather than a six-second C++ compute
operation.

The second request exposes another path. Once `PrepareExternal` acquires the
mutex, its existing CLEAN-prefix policy performs PASSIVE backfill on the
foreground thread. The call returns all 5,417 WAL frames copied. Actual WAL
fsync and database fsync take 2.663 and 3.652 seconds respectively. Together
with the prior mutex wait, the admission scope lasts 7.150 seconds. This is
maintenance before the new file's first data transaction; the final application
fsync alone would miss it. The existing SQLite timeout does not bound these
successful I/O calls.

The third request has negligible mutex waiting. It runs the required
payload-before-WAL FULL publication sequence on the foreground thread. Its
long time is inside the actual synchronization calls, with a small final
fence sync. This establishes that removing a checkpoint mutex wait would not
remove every long call.

There are **16 write/fsync handlers lasting at least one second inside large
write phases**: eleven have checkpoint acquisition as their largest exclusive
cost group; five are dominated by foreground file synchronization. Eleven of
the 32 large phases last at least four seconds. All phase/application vectors
remain included; these thresholds select diagnostic summaries, not replacement
benchmark samples or acceptance gates.

The largest summed application time outside its correlated write/fsync
handlers is 107.1 ms across a complete large phase. That residual combines
kernel copying, transport, caller scheduling and measurement overhead; it is
not an exact queue-delay measurement. The multi-second examples are measured
inside handlers. The **315 roots and 166 syscalls lasting at least 100 ms**
include preparation, validation and lifecycle as well as workload timers;
their raw intervals preserve that distinction.

## Consequence for the next implementation

Keep the original plain performance failures. Moving a sync to another thread
or to an earlier stage can move its wait without reducing complete work. The
earlier [payload presync](../payload_presync/README.md),
[payload budget](../payload_budget/README.md) and
[worker-quiescence](../full_sync_contention/README.md) experiments already fail
to establish a general whole-workload improvement. The current trace supplies
precise request paths for these frozen components; it does not establish the
SSD, journal or block-layer mechanism behind an individual sync.

A next bounded software design should reduce contiguous-file **version and
payload metadata per 4 KiB**, rather than just move checkpoint waiting. Native
attribution still assigns roughly 25–30% of caller write CPU to version insertion;
the FUSE trace now shows the WAL/database maintenance that surrounds those
records. A versioned extent representation can be evaluated first as an
isolated format: contiguous external spans get one version record, while
overwrites, boundaries, holes and inherited views retain explicit semantics.
All sixteen first-256 phases contain two worker payload syncs before their
final application barrier; subsequent-64 phases contain none. A smaller WAL
could reduce those intermediate publications under the same thresholds, but
the larger final dirty payload still needs its original durability barrier.
It needs snapshot/branch/GC/recovery proofs and count-only WAL/VM measurements
before another complete performance screen. Fewer metadata rows are a testable
hypothesis; they do not guarantee bounded payload/device sync latency.

## Validation and retained artifacts

The new C++ recorder passes Release and ASan/UBSan tests for nested accounting,
thread separation, exception unwinding, real SQLite steps and failed-syscall
errno preservation. These are **two recorder executions**, not a rerun of the
parent's component/Python suites. Original source identity is established by
the exact reverse transformations. Every full FUSE workload passes content,
retained-guard, schema, fence and SQLite integrity checks. Only related checks
run.

[audit.json](audit.json) verifies the build/test/measurement/analysis separation,
frozen inputs, stopped processes, absent mounts and **18 cleaned owned
directories**: two recorder fixtures and sixteen mounted workspaces. Large
database/payload data is deleted only after process exit and unmount. All raw
roots, actual events, CPU/host samples, original workload rows, application
intervals, source transformations, logs and diagnostic summaries are retained in
indexed JSON chunks below 500 KiB. Reconstructing each split record must equal
the original parsed JSON. Build products remain local with their identities.

```bash
BENCH="$PWD/vane_fs/benchmarks/staged_payload/fuse_waits"
RUN="$PWD/vane_fs/build/fuse-waits-reproduction"
python3 "$BENCH/prepare.py" "$RUN" \
  --parent "$PWD/vane_fs/build/native-write-costs-20261017"
python3 "$BENCH/run.py" "$RUN"
python3 "$BENCH/analyze.py" "$RUN"
python3 "$BENCH/summarize.py" "$RUN"
python3 "$BENCH/publish.py" "$RUN"
```

The publisher refreshes this report's evidence and refuses changed identities,
unremoved owned paths or overlapping owned build/test/measurement windows.

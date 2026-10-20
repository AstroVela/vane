# Final-fsync attribution for versioned extents

The reproduced format-1005 large-file fsync stalls are dominated by actual
payload `fdatasync`. A post-GC first-256 fsync lasts **6.909 seconds**, including
**6.904 seconds in payload fdatasync**, 3.156 ms in WAL fsync and 0.972 ms in
fence synchronization. A subsequent-64 fsync lasts **3.999 seconds**, including
**3.601 seconds in payload fdatasync** and 0.397 seconds in WAL fsync. Both
requests have zero measured checkpoint-lock wait and only microseconds in the
other instrumented locks. SQL execution is not the main cost of these calls.

This diagnoses the [ordinary extent FUSE regression](../extent_fuse/README.md).
All **14 original failures out of 22 checks** remain reported. Diagnostic
rates do not replace that uninstrumented acceptance window, and production
format 2 remains unchanged. The traced controls are the same frozen
borrowed-buffer format 1004 and versioned-extent format 1005.

## Matched diagnosis

Sixteen independent ext4 mounts run four AB/BA pairs fresh and four after the
same retained-guard fragmentation/GC preparation. They reuse the complete
first-256/subsequent-64, read/overwrite, small-file and namespace workload from
the preceding ordinary evaluation. The 8 MiB / 1 MiB application buffers stay
at page offset 48. Each format uses its own verified GC module, with the same
SQLite fsync SDK, durability and 16/64 MiB WAL thresholds.

The existing [C++ wait recorder](../fuse_waits/wait_trace.cpp) is reused without
changes. Exclusive wall/CPU stages partition each handler or worker root
exactly. Every real sync/checkpoint and slow I/O/lock event retains timestamps,
thread, request, file role, return value and errno. Application monotonic
intervals reconcile with native write coverage and final fsync handlers.
Overlapping worker sync and foreground lock intervals are not added as
independent costs. No trace output is written during requests; teardown writes
the trace after the component closes.

[prepare.py](prepare.py) reuses the old format-1004 diagnostic binary only after
verifying that regenerating every instrumentation change produces the same
source bytes. Format 1005 builds separately through a non-editable incremental
Release install. Removing the recorded replacements reconstructs both frozen
originals exactly. The runner adapts the tested recorder harness to the
per-format workload template; both reverse proofs are retained.

The recorder's Release and ASan/UBSan tests are rebuilt and pass, covering
threads, nested exclusive accounting, exception unwinding, actual SQLite VM
steps and syscall errno preservation. Algorithm correctness suites are reused
by original source identity and exact instrumentation reversals. Builds/tests
finish before the diagnostic window; analysis starts after all mounts and the
host monitor stop. There is no cache eviction, CPU/device tuning or sample
filtering.

## Actual final-barrier costs

All **32 large-file final fsyncs** are matched individually. Seven last at least
one second, and all seven have payload synchronization as their largest
measured component. The [complete findings](findings.json) include every final
fsync, actual syscall events, disjoint stage totals and overlapping-thread
wait intervals.

| Post-GC format-1005 example | First 256 MiB | Subsequent 64 MiB |
| --- | ---: | ---: |
| Repetition | 2 | 3 |
| Native fsync wall, seconds | 6.908566 | 3.998930 |
| Payload fdatasync, seconds | 6.903907 | 3.600554 |
| WAL fsync, seconds | 0.003156 | 0.396687 |
| Fence synchronization, seconds | 0.000972 | 0.001063 |
| Checkpoint-lock wait, seconds | 0 | 0 |
| Caller CPU, milliseconds | 77.869 | 21.786 |

These are separate deterministic examples, selected as the longest candidate
post-GC final fsync for each workload. Another subsequent-64 candidate call in
repetition 2 takes 3.989 seconds, including 3.607 seconds in payload fdatasync.
The ordinary window's 0.174 → 3.880 second median remains an observation from
that earlier window, not a statistic recomputed from these instrumented calls.

| Final fsyncs at least one second / four repetitions | Format 1004 | Format 1005 |
| --- | ---: | ---: |
| Fresh first-256 | 0/4 | 0/4 |
| Fresh subsequent-64 | 2/4 | 1/4 |
| Post-GC first-256 | 0/4 | 1/4 |
| Post-GC subsequent-64 | 1/4 | 2/4 |

Five additional sync roots outside the two large-file phases last at least one
second. Four are payload dominated; one post-GC candidate fsync spends 1.086
seconds almost entirely in WAL fsync, with no payload synchronization. The
payload finding therefore applies to the reproduced large-file stalls, rather
than every latency in the workload. These additional roots also remain in the
findings and complete raw traces.

## Progress before the final barrier

For **every one of the eight format-1004 first-256 phases**, the worker performs
two payload sync calls before the application's final fsync. For **every one
of the eight format-1005 phases**, it performs zero. Both formats have zero
worker payload sync calls during their subsequent-64 write loops in this
window. Format-1004 first-256 phases retain foreground checkpoint waits; one
post-GC phase accumulates 11.543 seconds there. The extent candidate's matching
large-write phases have zero checkpoint-wait time.

Together with the preceding row/WAL count results, this supports a narrower
explanation: reducing metadata growth also delays background publication under
a WAL-byte trigger, while the physical payload byte volume remains unchanged.
The candidate avoids those observed midstream waits but leaves synchronization
for its final barrier. This is evidence about application-level scheduling and
real syscall durations. It does not identify why the kernel/device takes
multiple seconds, or establish that an extra background sync will improve the
complete operation.

The next bounded experiment should add an independent dirty-payload byte
signal to the existing controlled background publication path. It must retain
payload-before-WAL ordering, OPEN/CLEAN recovery, explicit FULL barriers and
the current WAL thresholds. Its complete fresh/post-GC FUSE gates must include
new waits that might move into ordinary writes; shorter final fsync alone is
not acceptance. No such policy change is implemented in this diagnosis.

## Evidence and reproduction

All 16 mounts pass content, retained-guard, schema/fence and SQLite integrity
checks. **226,316 handler/worker roots** satisfy exact exclusive wall/CPU
accounting; **121,775 events** retain the typed call and wait evidence. There
are 32 complete large-file content checks, with all remaining workload
validations unchanged. The two recorder test executions pass; no broad
base/Ray suite runs.

The [audit](audit.json) verifies frozen inputs, nonoverlapping build/test,
diagnostic and analysis windows, stopped processes and **18 cleaned owned
data directories**, including the two recorder fixtures. All roots, events,
application intervals, CPU/host samples, source/runner proofs, logs and cleanup
hashes are retained losslessly in indexed files below 500 KiB. Large database
and payload outputs are deleted after their processes stop.

```bash
here=vane_fs/benchmarks/staged_payload/extent_waits
out="$PWD/vane_fs/build/extent-waits-repro"
python3 "$here/prepare.py" "$out" \
  --comparison "$PWD/vane_fs/build/extent-fuse-20261018" \
  --recorder "$PWD/vane_fs/build/fuse-waits-20261018"
python3 "$here/run.py" "$out"
python3 "$here/analyze.py" "$out"
```

The preceding frozen component/SDK builds and recorder must be present. The
source generator and proof manifests describe how those inputs are matched.

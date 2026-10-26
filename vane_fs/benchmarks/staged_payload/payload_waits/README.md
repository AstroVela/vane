# Payload publication: synchronization and lock attribution

The 64 MiB publication policy adds synchronous work to the write path through
**`PrepareExternal()` waiting for `checkpoint_mutex`**. The background worker
holds that mutex across payload publication and checkpointing. In this matched
diagnostic window, **99.3% of fresh and 99.4% of post-GC write checkpoint-wait
time overlaps synchronization performed by the recorded mutex holder**.
Publication-lock acquisition during ordinary writes remains around 1 ms per
256 MiB phase; checkpoint waits grow to roughly half a second in a typical
phase, with multi-second outliers.

This explains the failed [ordinary payload-progress trial](../payload_progress/README.md).
That trial's **11 failures out of 22 performance checks remain unchanged**.
These instrumented measurements provide attribution, not adoption throughput.
Both components use format 1005: the original extent control and the frozen
64 MiB payload-publication candidate. Production format 2, its installed native
module and the default SQLite SDK remain unchanged.

## Actual synchronization counts

All eight first-256 phases of each variant have the following actual counts,
including their final barrier:

| Calls per complete first-256 phase | Extent control | Payload publication |
| --- | ---: | ---: |
| Payload `fdatasync` | 1 | 4 |
| WAL `fsync`, foreground + worker | 2 | 16 |
| Fence `fdatasync`, foreground + worker | 2 | 8 |
| Worker payload syncs before final fsync | 0 | 3 |

The candidate performs three worker payload syncs during the write loop and
one during the final-barrier interval in every repetition. In this window the
worker also wins ownership of that last publication; ownership can race with
the explicit barrier. Its WAL calls split into eight foreground and eight
worker calls. The worker adds three or four database syncs per first-256 phase.
These counts include WAL-prefix capture, FULL publication and checkpoint work;
a payload trigger therefore incurs more than a data flush.

Both variants perform one payload sync in each subsequent-64 phase, with zero
worker payload syncs during its write loop. The control performs that sync in
the caller; the candidate's worker performs it during the barrier interval.
The [findings](findings.json) retain all 32 phase vectors and final fsyncs.

## Direct write blocking

These are medians of the four independent repetitions per state. Handler wall
and CPU totals include all native chunks for the 256 MiB phase. Medians of
components are separate statistics and must not be added to reconstruct a
median total.

| First-256 native measurement | Fresh control | Fresh candidate | Post-GC control | Post-GC candidate |
| --- | ---: | ---: | ---: | ---: |
| Write-handler wall, ms | 448.641 | 1063.211 | 413.631 | 1093.074 |
| Write-handler CPU, ms | 405.718 | 522.225 | 409.505 | 538.065 |
| Checkpoint acquisition wait, ms | 0.474 | 528.653 | 0.468 | 539.444 |
| Publication acquisition wait, ms | 0.951 | 1.174 | 0.966 | 1.204 |
| Holder-sync overlap with checkpoint waits, ms | 0 | 517.099 | 0 | 527.746 |

The candidate's accumulated checkpoint waits are **6.351 seconds fresh** and
**7.996 seconds post-GC**, over four phases in each state. Actual sync calls by
the mutex holder cover **6.306** and **7.950 seconds**, respectively. Payload
sync alone covers **5.743** and **7.504 seconds**; WAL, database and fence syncs
account for most of the rest. Each wait-to-holder link retains the wait, held
interval, owner thread/root and actual syscall timestamps under
[the indexed links](links/index.json). Overlapping worker I/O and caller wait
are two views of the same time and are not added as separate costs.

The largest application write takes **4.517 seconds** at offset 64 MiB in
fresh candidate repetition 0. It includes **4.410 seconds of checkpoint wait**,
of which **4.406 seconds** overlaps the worker's syncs, including **4.009
seconds of payload fdatasync**. Native handler CPU totals only **24.746 ms**
and the application interval outside handlers is **5.260 ms**. The matching
native request has one `PrepareExternal` call containing the checkpoint wait.
Post-GC candidate repetition 2 similarly takes **4.449 seconds** at offset
64 MiB, including **4.332 seconds checkpoint wait** and **4.100 seconds payload
sync** by its holder. Both examples and all slow application writes are kept
in the findings; every native interval remains in the complete traces.

Foreground SQLite VM work is unchanged: **306,644 steps fresh** and **306,647
post-GC** for each variant's first-256 phase. Caller CPU nevertheless rises,
including increased measured pwrite CPU. The trace establishes the additional
serialized I/O and its lock blocking; it does not identify a kernel/device
reason for the increased pwrite CPU or the long synchronization calls.

## Final fsync can wait on another thread

The longest candidate post-GC subsequent-64 final fsync takes **3.989 seconds**
(repetition 3), including **3.910 seconds waiting for the publication mutex**.
Its own WAL fsync takes **78.275 ms**; it performs no payload sync itself.
The holder performs **3.490 seconds payload fdatasync**, **418.459 ms WAL
fsync** and **0.922 ms fence sync** while the caller waits. Their union covers
**3.909 seconds** of the publication wait. Native caller CPU is **0.776 ms**.
The [complete example](case-final-wait.json) retains the source root, wait,
actual calls and overlapping publication-held interval. It is selected by the
largest candidate post-GC subsequent-64 fsync, not used to estimate a rate.

The next bounded experiment should separate **payload prewriteback** from
**FULL metadata publication**. A background data-flush operation can be
evaluated without retaining checkpoint/publication mutexes for its I/O, with
the payload/GC lease still protecting its file. The existing FULL barrier must
continue to establish durability and advance OPEN/CLEAN using its current
ordering. This diagnosis does not implement that policy or establish that it
will improve complete writes. A candidate would still need the original fresh,
post-GC, p99, lifecycle and sustained acceptance checks.

## Validation and reproduction

Sixteen independent ext4 mounts run four AB/BA pairs per state with the same
complete first-256/subsequent-64, read/overwrite, small-file and namespace
workload as the ordinary trial. Buffer page offset 48, application request
sizes, per-variant GC modules, SQLite fsync SDK, durability and WAL thresholds
remain fixed. There is no cache eviction, CPU/device tuning or sample filtering.
Owned builds/tests finish before measurement, and analysis starts after all
mounts and monitors stop.

Both diagnostics are rebuilt with identical recorders. The original stage and
syscall recorder is extended with publication-held and checkpoint-held spans.
Removing all recorded instrumentation changes reconstructs each frozen
component exactly. Each mutex-held span begins just after acquisition and ends
just before unlock; the publication span also covers its file lease and fence
adoption. Every hold is emitted, including fast holds; acquisition-wait events
retain the original 1 ms threshold. Counters retain faster waits as well.
Nested hold spans are inclusive descriptions and must not be summed.

The new recorder integration tests pass in **Release and ASan/UBSan**, covering
real payload/FULL publication, nested holds, exception unwinding, multiple
threads, VM steps and syscall errno. Original component correctness suites are
reused by frozen source identity and exact reversals; they are not rerun as a
new broad suite. All **16 mounts and 32 large-file content checks pass**.
**224,359 roots and 209,964 events** pass disjoint wall/CPU validation; every
hold event also matches its root's inclusive count, wall and CPU counters.
There are **24 recorded ordinary-write waits**, all with retained holder links.

The [audit](audit.json) retains all samples, native roots/events, CPU/host
observations, configurations, hashes, source/runner proofs and test logs in
indexed files below 500 KiB. **18 diagnostic/test and 2 preflight data
directories** are cleaned after owned processes stop. The preflight recorder
covered publication holds and was extended to include checkpoint holds before
any measurement; its original sources and test results remain retained.

```bash
here=vane_fs/benchmarks/staged_payload/payload_waits
out="$PWD/vane_fs/build/payload-holds-repro"
python3 "$here/prepare.py" "$out" \
  --comparison "$PWD/vane_fs/build/payload-progress-20261018/fuse"
python3 "$here/run.py" "$out"
python3 "$here/analyze.py" "$out"
```

The preceding frozen components and SDK must be present. The runner retains
all failed ordinary gates; native timing attribution does not change them.

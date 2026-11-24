# WAL/checkpoint interference evaluation: 2026-10-11

**The 16 MiB admission candidate is not enabled.** It increases synchronization
frequency and regresses fresh and post-GC writes. Production retains its 64 MiB
admission budget. This evaluation first separates storage synchronization from
SQLite and FUSE, then tests the smaller budget in the real C++ component. The
original SQLite SDK, 4 KiB BLOB format and durability barriers are unchanged.

## Direct-file controls

The [C++ probe](io_probe.cpp) uses ordinary `pwrite`, `pread` and `fsync` on two
owned files on the same ext4 filesystem as the preceding VaneFS measurements.
It does not execute SQLite or FUSE operations. The common timeline wrapper
links SQLite only to satisfy its unused SQLite wrapper symbols.

Every trial writes and validates 256 MiB. One file represents a WAL, reused
after each group; the other grows as pages are copied into it. Each group
syncs the WAL after writing its first quarter, copies that quarter, appends
the remainder, syncs the WAL again, copies the remainder, then syncs the
destination. Concurrent mode overlaps only the first WAL sync/copy with the
remaining appends. Recorded intervals verify positive overlap in every group.
Serial mode finishes that sync/copy before appending. Byte counts and barriers
match within each group size and write granularity.

Large writes use 1 MiB syscalls. Framed writes issue a 24-byte header and a
4 KiB payload separately, then copy payloads in 4 KiB calls; this is an I/O
shape, not a valid SQLite WAL. Paced runs wait until at least 10 ms have elapsed
per MiB of append work, approximating a producer capped at 100 MiB/s. Timers
include those waits, all copies and barriers. Complete byte comparisons and
SHA256 hashes are recorded after timing. No cache, kernel or device settings
are changed.

The first window has six alternating forward/reverse orders of six variants:
36 trials and 432 barriers with 64 MiB groups. The second uses eight balanced
orders of paced serial/concurrent variants with 16/64 MiB groups: 32 trials.
Smaller groups intentionally increase the number of barriers from 12 to 48
per trial. Controls from different windows are not pooled.

These are instrumented diagnostics. The synthetic copy pattern omits SQL,
version indexes, readers, FUSE and SQLite's checkpoint scheduling. Their rates
are not VaneFS throughput predictions.

### Findings

The first window reproduces seconds-long `fsync` calls **without SQLite, FUSE
or concurrent writers**. Large serial writes reach 2.86 seconds; framed serial
writes reach 3.58 seconds; paced serial writes reach 4.02 seconds. Fifteen of
432 barriers exceed one second. Concurrent paced writes also reproduce stalls.
Concurrency is therefore not necessary for this class of stall on this host.
The earlier serial control used only 16 MiB barriers and did not reproduce it;
the expanded workload includes larger barriers and repeated WAL reuse.

Samples during these direct-file stalls include `rq_qos_wait`,
`folio_wait_bit_common` and `jbd2_log_wait_commit`, as in the earlier VaneFS
traces. Some device-wide counter intervals contain only about 48–65 MiB of
writes despite taking several seconds. These counters include other host I/O;
sample counts are not exact time allocations. Neither the observations nor the
reported `wbt_lat_usec=2000` uniquely identify a controller, device fault, SSD
garbage collection or competing workload as the cause. No host tuning is applied.

The second window tests whether smaller batches help:

| Paced direct-file variant | Median trial, s | Maximum trial, s | Combined logical MiB/s | Maximum fsync, s |
| --- | ---: | ---: | ---: | ---: |
| Serial, 64 MiB | 4.624 | 10.469 | 38.92 | 3.574 |
| Serial, 16 MiB | 4.518 | 5.081 | 55.78 | 0.647 |
| Concurrent, 64 MiB | 4.062 | 10.102 | 42.28 | 3.655 |
| Concurrent, 16 MiB | 4.217 | 9.777 | 41.90 | 1.109 |

Each row has eight trials. A smaller maximum barrier does not ensure better
complete-workload throughput: concurrent 16 MiB batching mostly distributes
the waits across more barriers. The serial result motivates the C++ candidate,
but is not sufficient evidence to enable it.

## C++ admission candidate

The isolated [candidate](admission-16.patch) changes only the fsync-mode WAL
admission limit from 64 to 16 MiB. Its worker still starts at 16 MiB and retained
WAL allocation stays at 16 MiB. The existing checkpoint mutex and restart path
wait before admitting another mutation; explicit FULL barriers still bypass
admission. SQL, caches, payloads, atomic transactions and the SDK are unchanged.
Existing bounded-WAL and pinned-reader assertions are tightened to the proposed
budget. A single atomic transaction may still exceed it.

The candidate passes **132 related Python tests and six C++ suites**; the
unchanged control passes its six native suites. These include reader contention,
error/retry handling, slow checkpoints, synchronization syscall checks and crash
recovery. No full Vane or release suite is run.

## Real FUSE comparison

Six alternating pairs on fresh workspaces and six on independent post-GC
workspaces give **24 plain mounts**. The retained 32 MiB post-GC guard has
alternate 4 KiB blocks zeroed; an independent strict collector leaves 4,128
free pages before measurement. Every mount measures 64 and 256 MiB writes with
1 and 8 MiB application requests, then warm/sequential and seeded random reads,
512 random overwrites, 128 per-file fsyncs and directory operations with a final
directory fsync. Creation, fsync and close are included in write timings.
Content, guard, format-2 and SQLite quick checks all pass.

| Workload | Fresh 64 MiB budget | Fresh 16 MiB budget | Post-GC 64 MiB budget | Post-GC 16 MiB budget |
| --- | ---: | ---: | ---: | ---: |
| 64 MiB, 1 MiB requests, MiB/s | 96.78 | 40.56 | 98.40 | 59.51 |
| 64 MiB, 8 MiB requests, MiB/s | 95.50 | 57.04 | 94.11 | 56.37 |
| 256 MiB, 1 MiB requests, MiB/s | 95.99 | 63.53 | 94.45 | 60.29 |
| 256 MiB, 8 MiB requests, MiB/s | 89.02 | 56.40 | 89.48 | 25.20 |
| 4 KiB overwrite + final fsync, MiB/s | 11.65 | 5.97 | 12.00 | 6.40 |
| Small-file fsync, files/s | 249.64 | 237.46 | 256.50 | 243.48 |
| Complete lifecycle, seconds | 23.99 | 31.27 | 28.68 | 36.76 |

These are medians. Complete lifecycle includes preparation, validation,
deletion and shutdown, excluding final artifact inspection and cleanup.
Its summed time rises from 151.53 to 186.88 seconds fresh, and from 176.85 to
230.95 seconds post-GC. No owned builds or functional tests overlap these windows.

Some tails improve. Fresh 256 MiB writes with 1 MiB requests reduce maximum
request latency from 323 to 167 ms, but requests over 50 ms rise from five to
19 per file in every round. For 8 MiB requests, the post-GC maximum decreases
from 5.55 to 2.27 seconds while combined throughput falls from 59.20 to
24.95 MiB/s. Post-GC random-overwrite combined throughput improves from 1.91 to
6.27 MiB/s because the control has two very slow samples, despite the candidate's
lower median. These results are retained; they do not offset the complete-workload
regression. No slow samples are discarded.

Four additional instrumented mounts run in a separate window. In the fresh
256 MiB/1 MiB-request phase, the control has five complete database syncs and
23 WAL syncs; the candidate has 19 and 40. Restart checkpoint calls rise from
five to 21. These counts support the observed increase in synchronization
frequency. Admission waits on the worker's mutex **before** calling SQLite's
restart checkpoint, so short restart-call durations do not measure that waiting.
Boundary-crossing events are recorded separately; overlapping thread durations
must not be added as wall time. Instrumented throughput is not pooled with the
plain comparison.

## Reproduction and evidence

Use a new output directory for each run. The [runner](run.py) builds the probe
with the original component SQLite SDK, records its compiler/source identities,
samples host counters and owned thread wait locations, validates data and removes
owned data after processes exit. `smoke` runs four content checks; it is not a
performance window.

```bash
python3 vane_fs/benchmarks/io_interference/run.py vane_fs/build/io-repro direct
python3 vane_fs/benchmarks/io_interference/run.py vane_fs/build/io-repro bounded
python3 vane_fs/benchmarks/io_interference/run.py vane_fs/build/io-smoke smoke
python3 vane_fs/benchmarks/io_interference/prepare.py vane_fs/build/admission-repro
```

The [preparer](prepare.py) reconstructs the baseline at
`f4f76ddce4f52e8818316c09e7378899d7c8c74b` and the complete candidate, verifying
all 29 native, binding, build and test source inputs per variant. Build each
component with the [component instructions](../../README.md).

[Results](results.json), [summaries](summary.json), raw variant files and the
[artifact manifest](artifact-manifest.json) retain every sample, source/binary
identity, host observation, validation hash and cleanup record. The original
probe sources are frozen locally alongside the exact build commands; the
published runner also passes four standalone content checks after formatting.
All 100 owned benchmark data directories and the component's Python test
directory are removed after their processes stop. Production sources, the
shared SDK and installed component retain their original hashes.

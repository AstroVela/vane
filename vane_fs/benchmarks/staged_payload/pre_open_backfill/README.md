# Selective CLEAN backfill before external writes

This implements the directed opportunity identified by the
[follow-through traces](../checkpoint_followthrough/README.md), preserving the
original 5 ms CLEAN policy. **The candidate is not adopted:** its subsequent
64 MiB write median improves, but retained tails and post-GC regressions fail
the unchanged checks. Production format 2 sources and the installed default
module remain unchanged.

## Implementation and safety

[prepare.patch](prepare.patch) forwards the existing `MayExternalize` flag to
`Checkpointer::BeforeWrite`. Before `BEGIN IMMEDIATE`, a potentially external
write with more than 256 KiB of uncopied CLEAN WAL attempts one PASSIVE backfill.
The checkpoint mutex, publication guard and checkpoint lease use the existing
lock order. The guard refreshes the fence and WAL status; an OPEN batch is
excluded. The lease protects copying while the publication guard is released,
permitting concurrent FULL barriers. This path never publishes a payload batch,
restarts a WAL generation or retries a partial/busy PASSIVE call. Storage errors
are reported before the incoming mutation or payload append.

The entire background worker is byte-identical to the control. Its fixed,
non-sliding 5 ms CLEAN window, 16 MiB OPEN budget, reader retry, last-commit
progress and Stop/Start handling remain. Admission at 64 MiB and FULL barriers
retain their existing behavior. The preparation is real synchronous I/O;
neither the 256 KiB selector nor a SQLite timeout bounds disk latency. Later
workloads can inherit a different WAL state even though their code is unchanged.

The builder reverses both changed files with fuzz zero and verifies every
other runtime file against the frozen control. Both native installations are
non-editable, use the pinned SQLite 3.53.2 SDK with its original `fsync` primitive,
and use FUSE 3.14. No parameters or gates change after measurement begins.

## Plain comparison

[The ordinary ext4 window](plain/summary.json) runs six alternating AB/BA pairs
in each of independently prepared fresh and post-GC workspaces: 24 mounts.
The original workload, barriers, buffer offset 48, request sizes and acceptance
checks are unchanged. Each variant uses its own installed module for GC.

Median throughput changes against the previous 5 ms CLEAN policy:

| Workload | Fresh | Post-GC |
| --- | ---: | ---: |
| Initial 256 MiB write | −0.4% | +0.3% |
| Subsequent 64 MiB write | +12.5% | +11.7% |
| 4 KiB random write | +10.3% | −20.2% |
| Small-file fsync | −2.9% | +3.2% |
| Mkdir | +0.4% | −8.0% |
| Rename | +0.1% | −12.2% |

The 64 MiB write medians rise 96.91→109.02 MiB/s fresh and 93.22→104.13 MiB/s
post-GC. Its fresh aggregate ratio is only 0.441: a candidate request lasts
1.60 seconds and its final fsync lasts 4.17 seconds. The post-GC candidate has
a 4.37-second initial-stream request; the control has a 2.90-second 64 MiB
request and a 2.84-second final fsync. All five samples are retained.

Nine ordinary workload checks and both combined-write goals fail. Fresh read
aggregate and small-file fsync tail checks also fail; post-GC random-write,
directory and initial-stream results fail. The combined-write aggregate ratios
are 0.734 fresh and 1.162 post-GC, with four of six paired wins in each state.
The candidate does not satisfy the original five-of-six paired-win goal.
No further original-control or tmpfs acceptance window follows this rejection.

## What the sync trace establishes

A separate 16-mount diagnostic window uses the verified typed-sync recorder.
Removing only instrumentation restores both original implementations exactly;
recorder code and its preceding Release/ASAN validation are reused unchanged.
Instrumented results diagnose costs and are separate from ordinary acceptance
throughput. It records 5,778 actual syncs, 109,498 WAL calls and 2,048 paired
lookup/rename observations. Raw records remain local with hashes.

[Preparation findings](trace/preparation-findings.json) confirm that all eight
candidate 64 MiB phases perform one CLEAN prefix backfill and have zero
midstream payload publications, compared with eight publications in the eight
control phases. The prefix contains approximately 11.6 MiB, and median PASSIVE
duration is 82.4 ms fresh and 99.9 ms post-GC. Eight additional preparations
occur during workload setup. All 16 attempts copy their observed prefix.

Complete diagnostic 64 MiB phase medians fall 623→560 ms fresh and 662→653 ms
post-GC. Summed actual sync durations remain approximately 267→273 ms fresh and
267→281 ms post-GC. Those overlapping syscall durations are not additive wall
time. The preparation changes scheduling and eliminates the identified
midstream publication; it does not demonstrate less total synchronous I/O.

The candidate also has one 7.14-second small-file phase. Its 128 foreground WAL
syncs total 6.91 seconds, with a 74.9 ms median and 256.6 ms maximum. C++ request
CPU totals 198 ms, BEGIN totals 8.39 ms, publication-lock wait totals 0.062 ms,
and pwrite totals 27.6 ms. During that phase the worker performs 31 database
syncs totaling 1.97 seconds; 33 foreground WAL sync intervals overlap those
database syncs. This identifies the slow phase as WAL synchronization, while
the overlap alone does not prove why device completion slows.

All 101 syncs lasting at least 100 ms remain. Two real syncs last at least one
second: a 2.18-second payload barrier and a 1.70-second RESTART WAL sync, both
in the control's workload setup outside timed phases. The ordinary window's
five long request/barrier samples did not recur as the same traced requests;
these separate observations cannot attribute their individual causes. No
sample is removed or explained away as unrelated host load.

## Validation and retention

The candidate passes 154 installed Python tests with one unsupported format-v1
migration skip, 12 C++ suites, 12 ASAN/UBSAN suites and 56 Release/sanitized
checkpoint helpers. Twelve control C++ suites pass. The new cases exercise
large-write preparation through the public API, small writes, a tiny prefix,
OPEN exclusion, pinned-reader partial progress, busy handling and pre-mutation
I/O failure. Three behavioral assertions fail against the control as expected.
Existing progress, restart, concurrency, hook, recovery and small-burst tests
remain. Only related tests run.

All 40 mounts pass content, retained-data, indexed-schema and integrity checks.
[audit.json](audit.json) verifies frozen source/module/binary identities, all
measurements, the unchanged default and owned-data cleanup. Builds and tests
finish before the ordinary window; ordinary and diagnostic windows do not
overlap. No cache eviction or affinity/frequency tuning is used. Owned daemons
stop and unmount before temporary database, WAL and payload deletion, outside
timers. Raw host samples, full typed traces, large test logs and build products
remain local with hashes.

Changed-source formatting, Ruff, whitespace, copyleft and SPDX-header checks
pass. The optional whole-root SPDX scan reports five missing headers in older
experimental files, all unchanged from HEAD; `validation/final-checks.json`
records them separately.

Two harness issues are retained in `preflight`: the first diagnostic launch
needed the already-passed test report copied to `trace/tests.json` and stopped
before creating a mount; the first additional analysis used a nonexistent
publication `workload` field rather than `phase_begin_ns`. Neither changed the
candidate, recorder or measurements.

## Next investigation

Isolate concurrent foreground FULL WAL sync and background CLEAN PASSIVE sync
with the same durability barriers and bounded WAL admission. Compare complete
small-file phases and all sync calls, including a worker-quiesced diagnostic
control, before changing the scheduling policy. Preserving small-operation
source code did not preserve their inherited WAL state or measured performance.
The separate RESTART writer-lock contention also remains.

## Reproduction

```bash
python3 vane_fs/benchmarks/staged_payload/pre_open_backfill/candidate.py OUTPUT \
  --parent vane_fs/build/lookup-read-20261013T012000Z
python3 vane_fs/benchmarks/staged_payload/pre_open_backfill/validate.py OUTPUT
python3 vane_fs/benchmarks/staged_payload/pre_open_backfill/diagnostic.py OUTPUT \
  --recorder vane_fs/build/checkpoint-followthrough-20261013 \
  --template vane_fs/build/checkpoint-progress-20261012T153141Z/fuse-screen \
  --mode build
cp OUTPUT/tests.json OUTPUT/trace/tests.json
python3 vane_fs/benchmarks/staged_payload/pre_open_backfill/run.py OUTPUT \
  --template vane_fs/build/checkpoint-progress-20261012T153141Z/fuse-screen
python3 vane_fs/benchmarks/staged_payload/pre_open_backfill/diagnostic.py OUTPUT \
  --recorder vane_fs/build/checkpoint-followthrough-20261013 \
  --template vane_fs/build/checkpoint-progress-20261012T153141Z/fuse-screen \
  --mode run
python3 vane_fs/benchmarks/staged_payload/pre_open_backfill/analyze.py OUTPUT/trace
```

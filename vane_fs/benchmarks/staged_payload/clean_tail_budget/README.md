# Isolating the CLEAN dirty-tail budget

This separates the two scheduling restrictions introduced by the
[follow-through investigation](../checkpoint_followthrough/README.md).
**The tail-only candidate is not adopted:** the targeted 64 MiB write recovers,
but random-write, directory and tail checks still fail. Default source and the
installed default module remain unchanged.

## Implementation and controls

[budget.patch](budget.patch) limits the fixed 5 ms CLEAN coalescing window to
uncopied WAL bytes at most 256 KiB. It retains coalescing of small tails in large
WAL generations. OPEN publication still uses the existing 16 MiB uncopied budget;
the 64 MiB admission limit, FULL barriers, non-sliding deadline, reader retry,
last-commit progress and Stop/Start handling are unchanged. The bound applies
to scheduling, not actual I/O or lock duration.

Two independent plain ext4 windows run six alternating AB/BA pairs in each of
fresh and independently prepared post-GC workspaces: 24 mounts each, 48 total.
Both controls and the candidate are frozen before either window. The original
workload, buffer offset 48, request sizes, barriers and gates are unchanged.
Each component uses its own installed native module for GC.

| Window | Control | Candidate | Isolated difference |
| --- | --- | --- | --- |
| [Whole-generation restriction](whole-generation/summary.json) | Both restrictions | Tail-only restriction | Remove the whole-generation condition |
| [Dirty-tail restriction](tail/summary.json) | Previous 5 ms CLEAN policy | Tail-only restriction | Add the 256 KiB uncopied cap |

For the first window, every other C++ token is identical after ignoring comments;
the source proof removes exactly `Bytes(frames) < START_BYTES &&`. All other
runtime source files are byte-identical. For the second, reversing the patch
with fuzz zero recovers the control header byte-for-byte. The 256 KiB cap and
5 ms window are held fixed; no parameter sweep or gate recalibration is used.

## Results

Removing the whole-generation restriction does not eliminate the earlier
regression. Fresh random-write median throughput falls 11.7%; post-GC mkdir
falls 10.3%. The 64 MiB write is approximately level, at −1.9% fresh and +2.1%
post-GC. Fresh rename median improves 3.2% but its aggregate falls 12.4%. Five
ordinary checks and both original combined-large-write goals fail.

Adding only the dirty-tail restriction gives the following median changes
against the previous 5 ms policy:

| Workload | Fresh | Post-GC |
| --- | ---: | ---: |
| Subsequent 64 MiB write | +10.1% | +21.3% |
| Small-file fsync | +3.3% | +4.0% |
| 4 KiB random write | −3.7% | −14.8% |
| Mkdir | +5.6% | −6.8% |
| Rename | +6.1% | +1.8% |

The 64 MiB write medians rise from 103.30→113.69 MiB/s fresh and
92.70→112.41 MiB/s after GC. Its fresh aggregate ratio is only 0.412 because the
candidate has a 5.23-second final fsync. Fresh random-write aggregate falls 5.6%;
post-GC median and aggregate fall 14.8% and 9.8%. Several directory aggregate
checks also fail. Six ordinary checks and both combined-large-write goals fail.
The paired combined-write wins are five of six in each state, but aggregate
ratios remain 0.713 fresh and 0.993 post-GC; paired wins alone are insufficient.

All latency samples remain, including a 6.69-second candidate 256 MiB write in
the first window and 7.72-second candidate / 7.22-second control requests in the
second. Five request/barrier samples last at least one second. These plain
windows do not identify the individual sync syscall responsible for each tail.
No tail is discarded or attributed to unrelated machine load.

The independent windows do not support attributing the entire regression to
the whole-generation restriction. The dirty-tail cap alone repeats the
random-write/aggregate failure while improving the large-write median. A global
CLEAN cap therefore remains too broad for adoption on these measurements.
There is no additional original-control or tmpfs window after rejection.

## Next target

Preserve the existing small-file and namespace coalescing policy, and test a
directed backfill opportunity before a large write changes CLEAN to OPEN. The
previous trace identifies a large CLEAN prefix that becomes expensive to publish
midstream; ordinary small writes need not change their checkpoint frequency to
handle that transition. Record the actual payload/WAL/database sync and writer
intervals for the seconds-long publication/barrier tails before claiming a cause.
The separate RESTART writer-lock contention also remains: moving lookup to a
read transaction merely transfers that wait to rename.

## Validation and retention

The new candidate passes 154 installed Python tests with one unsupported
format-v1 migration skip, 12 native suites, 12 ASAN/UBSAN suites and 42
Release/sanitized progress, restart, policy and budget helpers. Twelve native
control suites pass. The previous policy fails the large-prefix urgency check
as expected. The candidate's large-generation small-tail case still coalesces
(two probes before PASSIVE), while its 1 MiB dirty prefix bypasses the timer
(one probe). The 80-commit small FULL burst still performs 16 PASSIVE calls.
These are helper counts, not measured FUSE sync counts. Only related tests run.

The pinned SQLite 3.53.2 SDK uses its original `fsync` primitive; FUSE is 3.14.
All 48 mounts pass content, retained-data, indexed-schema and integrity checks.
No owned build/test overlaps either window, and no cache eviction or
frequency/affinity tuning is used. [audit.json](audit.json) verifies frozen
identities, all measurements, the unchanged default and cleanup after each owned
daemon stops/unmounts. Raw host samples, large test logs and build products remain
local with hashes; database, WAL and payload data are removed outside timers.
The two wrapper preflights are retained; they stopped before runtime testing or
measurement and changed only generator import/marker handling.

## Reproduction

```bash
python3 vane_fs/benchmarks/staged_payload/clean_tail_budget/prepare.py OUTPUT \
  --parent vane_fs/build/lookup-read-20261013T012000Z
python3 vane_fs/benchmarks/staged_payload/clean_tail_budget/validate.py OUTPUT
python3 vane_fs/benchmarks/staged_payload/clean_tail_budget/run.py OUTPUT \
  --template vane_fs/build/checkpoint-progress-20261012T153141Z/fuse-screen \
  --generation-control vane_fs/build/checkpoint-followthrough-20261013/budget/updated \
  --window whole-generation
python3 vane_fs/benchmarks/staged_payload/clean_tail_budget/run.py OUTPUT \
  --template vane_fs/build/checkpoint-progress-20261012T153141Z/fuse-screen \
  --generation-control vane_fs/build/checkpoint-followthrough-20261013/budget/updated \
  --window tail
```

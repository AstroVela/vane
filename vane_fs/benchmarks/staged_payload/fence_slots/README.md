# Materialize both fence slots before the first OPEN

The first OPEN in a new format-1005 workspace allocates and extends its fence
from 4 KiB to 8 KiB. An isolated candidate materializes the second, invalid
slot under the existing initialization barrier. It saves about **1 ms** in
ordinary first-OPEN fence synchronization in this native window, but **does
not pass the frozen complete-workload screen**. Keep it experimental.
Production format 2, its installed module and the default SQLite SDK remain
unchanged. The preceding range candidate still has **7 failed FUSE checks out
of 22**; this experiment does not clear them.

## Implementation and durability

[materialize.patch](materialize.patch) changes only fresh initialization.
Before the existing `Save(false)`, it writes zero padding into slot two at
offset 4096, retrying EINTR and short writes and reporting zero/error results.
The normal initial CLEAN record is then written into slot one. Its existing
fdatasync persists both slots and the allocation; no extra barrier is added.
The padding has no magic, sequence or checksum and remains an invalid record.

Every first OPEN still captures the committed WAL prefix, synchronizes it,
writes the next checksummed OPEN record, and synchronizes the fence before
appending external payload bytes. Every final FULL still synchronizes payload,
commits FULL WAL metadata, then persists CLEAN. Recovery, cross-connection
adoption, checkpoint policy, the 64 MiB admission limit and the disk format are
unchanged. Existing 4 KiB fences are accepted without rewriting at open; their
first OPEN follows the old extension path. Existing two-slot fences are also
accepted. The same reader and record-validity code is byte-identical.

Both the uninstrumented component and instrumented diagnostic have exact patch
reversal proofs. Every other copied source remains identical to its frozen
parent. The candidate uses a non-editable isolated installation; the installed
production module and default SDK are untouched.

## Measured mechanism

The recorder observes every pwrite and sync with descriptor class, offsets,
byte counts, return/errno, wall/thread CPU time, and file size/allocated bytes
before and after the call. It retains the original 28-stage ABI and adds only
private event fields. Both measured native cores compile with the same
`-std=c++17 -O2 -DNDEBUG` flags and use the same recorder object and fsync SDK.

All 12 fresh controls show the OPEN fence write growing **4096→8192 bytes**
and allocated space **4096→8192 bytes**. All 12 candidates show
**8192→8192 bytes** for both, because initialization already wrote the padding.
The first request in both variants captures exactly **50 WAL frames** in a
**206,032-byte WAL**, with zero backfilled, and performs **738 SQLite VM steps**.
Median caller CPU is **3.358 ms control versus 3.306 ms candidate**. The
WAL-prefix fdatasync medians remain **2.125 versus 2.100 ms**.

The first-OPEN fence fdatasync median falls **1.825→0.826 ms**, with all 12
paired calls faster. The entire first native request falls **7.086→6.052 ms**.
This supports a small ordinary cost from allocating the second slot in this
workload. The earlier **80–87 ms fence tails** are not reproduced: the largest
fresh control OPEN fence sync here is **2.246 ms**. This result cannot assign
those earlier tails to allocation or establish a kernel/device cause.

## Complete native comparison

Forty-eight serial processes run 12 AB/BA pairs fresh, six reused and six
independently prepared post-GC pairs. Each writes the first 8 MiB using the
preceding FUSE trace's exact nine core requests: **1 MiB minus 48 bytes, seven
1 MiB requests, then 48 bytes**. It keeps mount/open/lookup references valid,
performs the original final FULL, releases references and closes the workspace.
This is a native first-write diagnostic; it omits kernel FUSE/request/reply
cost and does not replace the ordinary 256+64 MiB workload or its gates.

Total starts before workspace construction and includes initialization, all
writes, final FULL and close. Preparation is separately measured and also
included in the complete lifecycle. Reopening for correctness verification
and removing owned data happen after these timers. No maintenance is deferred
beyond the timed original close. Reused/post-GC preparations already have two
materialized fence slots in both variants. GC reclaims versions and payloads
while retaining different live/snapshot contents.

Each table entry is the median elapsed time; lower is better. Aggregate ratio
is total control time divided by total candidate time; above one favors the
candidate. All samples are retained.

| State | First 8 MiB control / candidate, ms | Total control / candidate, ms | First-8 aggregate ratio | Total aggregate ratio | Candidate total wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh | 19.201 / 18.028 | 69.106 / 68.447 | 1.094 | 1.064 | 10/12 |
| Reused | 23.791 / 23.478 | 61.028 / 61.934 | 1.045 | 0.963 | 4/6 |
| Post-GC | 24.616 / 24.931 | 60.203 / 64.145 | 0.982 | 0.921 | 2/6 |

Fresh first-8 elapsed median improves **6.1%**, with **11/12 paired wins**,
but complete total median improves only **0.95%**. Initialization medians
are **21.208/21.434 ms** and final FULL **24.309/24.749 ms**. They are included,
so the first-write timer alone is insufficient to judge the change.

The frozen native screen required fresh first-8 aggregate gain of at least
1.05 with nine of twelve wins, and total aggregate ratio of at least 0.98 in
every state. The fresh criteria pass; reused and post-GC totals fail. No
ordinary FUSE adoption window follows this rejection, and no gate is relaxed.

## Remaining barriers and retained tails

The reused candidate's first total is **84.538 ms**, including **46.277 ms**
in the final payload fdatasync. The post-GC candidate's first total is
**88.003 ms**, including **49.504 ms** in that syscall. Both synchronize a
9,441,280-byte payload file with thread CPU of **1.681 and 1.605 ms**.
These calls, rather than first-OPEN fence sync, account for much of the
negative-control aggregate loss. They remain in every comparison.

The fresh control's repetition-three total is **97.505 ms**, with **30.675 ms**
in a closing WAL fsync. That retained tail improves the candidate's fresh
aggregate result more than the 0.95% median improvement. Full sync sequences
and all individual calls are in [barrier-costs.json](barrier-costs.json).
No second-scale or 80 ms fence sync appears in this window; its lack does not
establish a latency bound or resolve the preceding complete FUSE tails.

## Validation and provenance

Only related storage checks run:

- **96 Python tests pass**, with one production-v1 migration case skipped
  because format 1005 has no such migration.
- **12 Release and 12 ASan/UBSan C++ suites pass**, including staging's
  **66 crash/loss cases in each mode**, durability, external faults and retry.
- **12 matched workload probes pass** across both variants, all three states
  and both modes; every one reopens and verifies all 8 MiB, retained guards,
  snapshots where applicable and SQLite integrity.
- Two new slot-test executables cover invalid padding, unchanged initial
  fence barrier count, EINTR/short writes, zero/EIO/sync failures and legacy
  one-slot publication. Two recorder contracts check observed growth, successful
  errno preservation and EIO/EINTR/EBADF, including invalid-descriptor stats.

All **648 roots and 42,168 events** in the measurements reconcile exclusive
wall/CPU accounting. Every measured actual sync succeeds. WAL-before-fence,
OPEN-before-payload and final payload-before-FULL-before-CLEAN ordering is
checked in every run. No range writeback hint is issued.

Preparation history is retained. First, GNU patch made an `.orig` backup
after an offset match; the equality audit rejected the extra file before any
native build. Second, a test launch could not find pytest, whose dependency
directory was supplied by the parent's nonrecursive `support.pth`. Its full
explicit path closure is now included; no test case ran in that failed launch.
Both original runner/source snapshots and logs remain.

Before measurement, the frozen control archive was found to use CMake's
`-O3`, while the new candidate core used `-O2`. The control was rebuilt at the
same `-O2` flags, relinked and all three affected Release probes rerun. The
earlier binary, passing probe traces, original harness, exact source diff and
commands remain as compiler preflight history. Every measured process uses
the matched binary. The builder now compiles both cores the same way directly;
the recorded one-time matching adapter is unnecessary for a new reproduction.

[findings.json](findings.json) retains every phase/group sample and explicit
screen failure. [plan.json](plan.json) freezes requests, orders, thresholds,
source/library/binary identities and all measurement inputs. The default
module/SDK identity checks and source checks are in
[final-checks.json](final-checks.json). [audit.json](audit.json) indexes every
original trace, CPU/host sample, build/test log and source-history byte stream
by size/SHA-256 and ordered gzip chunks. Concatenate an entry's `retained_as`
files, decompress once, then check its original hash and size.

All **71 owned data directories** are removed after their processes stop,
including the failed test-launch directory and compiler preflight probes.
No owned mount or measurement process remains. Artifacts are individually
below 500 KiB. Builds/tests, measurement and analysis do not overlap. There is
no cache eviction, host/device tuning, load-based filtering or unowned process
termination. No broad base/Ray suite runs.

## Next step and reproduction

Keep the small patch isolated. Before changing synchronization further,
reproduce the earlier cold-request fence tail under the complete FUSE workload
and distinguish journal/data/flush waits with read-only observations. The
existing OPEN/WAL/fence ordering remains required. The independently slow
final payload barriers remain a larger target than the approximately 1 ms
ordinary fence allocation cost found here.

```bash
here=vane_fs/benchmarks/staged_payload/fence_slots
out="$PWD/vane_fs/build/fence-slots-repro"
python3 "$here/build.py" "$out" \
  --parent "$PWD/vane_fs/build/range-writeback-waits-20261018" \
  --component "$PWD/vane_fs/build/versioned-extents-20261018/candidate"
python3 "$here/run.py" "$out"
python3 "$here/analyze.py" "$out"
# The recorded barrier-costs helper is in the lossless retained_sources index.
# Record final source/identity/cleanup checks as "$out/final-checks.json".
python3 "$here/publish.py" "$out"
```

The optional `--preflight` argument preserves an earlier failed preparation
directory; new reproductions need not create one. Source validation success
is distinct from the failed native performance screen.

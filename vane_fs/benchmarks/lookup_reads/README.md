# Read-first inode lookup: implemented, not adopted

This experiment implements the next step from the
[request-cost investigation](../staged_payload/checkpoint_request_costs/README.md).
It passes functional checks and improves individual misses and retained-inode
lookups. **The complete production FUSE comparison fails five non-regression
checks, so the default runtime was restored.** The complete implementation and
new native tests remain in [lookup.patch](lookup.patch).

## Implementation and correctness

Resolve the name in a read transaction. A missing name returns without acquiring
SQLite's writer lock. For an inode already pinned by this connection, increment
the transactional temporary reference count while holding the database mutex;
the existing durable pin protects the inode throughout. Snapshot lookup remains
read-only.

A first reference ends the read transaction and acquires a normal write
transaction, including the existing checkpoint admission and retired-pin cleanup.
It validates the session and checks both `SQLITE_FCNTL_DATA_VERSION` and
`sqlite3_total_changes64`. Reuse the resolved inode only when neither changed;
otherwise resolve the parent, branch and entry again before atomically publishing
the pin. This avoids upgrading a stale SQLite read snapshot. The
[SQLite file-control contract](https://www.sqlite.org/c3ref/c_fcntl_begin_atomic_write.html)
describes detection of changes from the same or another connection; the total
change count also conservatively detects local temporary-reference updates.

Tests cover another connection holding the writer lock, first-pin failure and
retry, invalid names and parents, dot/dot-dot, snapshots, directory identity after
rename/replacement, unlink/GC lifetime, and concurrent namespace changes. A
separate native test interleaves deletion, inode replacement, parent-pin release
and a pin-failure trigger precisely between the two transactions, through both
the same and another SQLite connection. The original implementation fails the
held-writer missing-name regression as expected.

## Native measurement

Six alternating AB/BA pairs for each durability mode and medium give 48 runs.
Each run measures 4,096 misses, 4,096 retained-pin lookups and 512 first pins.
First-pin release and integrity validation are outside the request timer. The
directory/database pages are warm; “first” means a new durable pin, not cold
storage. Values below are medians of six per-run mean latencies, in microseconds.

| ext4 case | Before | Candidate | Change |
| --- | ---: | ---: | ---: |
| Fsync miss | 24.86 | 20.92 | −15.8% |
| Fsync retained pin | 29.89 | 23.79 | −20.4% |
| Fsync first pin | 56.59 | 61.84 | +9.3% / +5.25 µs |
| Strict miss | 25.26 | 23.35 | −7.5% |
| Strict retained pin | 27.20 | 24.23 | −10.9% |
| Strict first pin | 2,361.97 | 2,451.03 | +3.8% |

All tmpfs results and every individual request sample are retained in
[native](native/summary.json). These measurements do not establish an end-to-end
improvement. The earlier preflights exposed a roughly 35% first-pin penalty
without version validation; their raw samples remain locally with hashes in the
shared audit and are not used for the final decision.

## Complete FUSE comparisons

The production comparison uses the original 4 KiB SQLite BLOB implementation,
with lookup as its only runtime change. It runs the unchanged plain ext4
workload: six AB/BA pairs each for new and independently prepared post-GC
workspaces, 24 mounts. The 256 MiB + 64 MiB write sequence, application buffer
offset 48, request sizes, barriers, content checks and cleanup are unchanged.

Before measurement, [the plan](production/plan.json) requires the original
per-workload median and aggregate rate floors of 0.95, p99 allowance of 20% or
1 ms, and lifecycle median increase of at most 5%. The external-payload
experiment's additional 10% large-write improvement target is explicitly not
a lookup adoption condition. No threshold is relaxed after observing results.

[The decision](production/decision.json) is **REJECT**. Fresh random-write median
rate is 0.898× the control. Post-GC aggregate rates are 0.562× for the initial
large write, 0.570× for the subsequent write and 0.928× for reads; lifecycle
median is 1.224×. Fresh rename improves about 8%, while post-GC rename is nearly
unchanged. Large-write medians after GC are much closer to control than their
aggregate rates: the retained long samples matter. They are not discarded or
attributed to another workload without evidence.

A separate format-1004 lookup-only ablation also fails non-regression checks:
fresh large-write aggregate throughput and post-GC random-write/rename. See the
[CLEAN coalescing report](../staged_payload/clean_coalescing/README.md) for that
window and the independently evaluated follow-up.

The tests establish that read-only misses avoid the writer lock. They do not
establish that a rename batch avoids the same wait in its subsequent mutation.
Trace lookup and the immediately following writer transaction together before
claiming that the complete operation became faster.

## Validation, restoration and reproduction

The measured production candidate passes 132 installed Python tests, seven C++
suites and seven ASAN/UBSAN suites. The related isolated candidates and all 96
FUSE mounts across four windows pass content/integrity checks. The shared
[audit](../staged_payload/clean_coalescing/audit.json) retains the full validation
matrix, identities, samples and cleanup proof. Only related tests run.

After rejection, production source files are restored byte-for-byte and the
default package is reinstalled with the incremental, non-editable workflow.
All six original native suites and 132 installed Python tests pass again. The rebuilt module has a new binary
hash; the audit records both hashes and does not claim byte-identical binaries.
The exact measured candidate sources, module and binaries are archived locally
before restoration, with a verified mapping from their recorded measurement paths.

To reproduce, prepare two separate component checkouts and virtual environments
with the pinned SQLite SDK and FUSE SDK. Apply `lookup.patch` with `patch -p1`
inside the candidate's `vane_fs` directory, then incrementally install both
components with tools/FUSE/tests enabled. The patch includes its CMake test
registration and both native test files. Run the candidate's related tests first.

```bash
python3 vane_fs/benchmarks/lookup_reads/run.py OUTPUT \
  --baseline BASELINE_COMPONENT --updated CANDIDATE_COMPONENT \
  --sqlite-prefix vane_fs/vcpkg_installed/x64-linux-release
```

Run `fuse.py` from the candidate checkout, whose component is installed in its
repository `.venv`; `--baseline` must have its own `venv` and release build:

```bash
python3 vane_fs/benchmarks/lookup_reads/fuse.py FUSE_OUTPUT \
  --baseline BASELINE_COMPONENT --validation VALIDATION_JSON
```

`VALIDATION_JSON` must record successful related checks. The runner freezes code
and installed-module identities and refuses changes during the measured window.

# Versioned extent metadata prototype

The isolated C++ format-1005 prototype replaces one external payload/version
record per 4 KiB block with ranges of at most **64 blocks / 256 KiB**. Matched
native request replays reduce version rows by **98.0–98.4%**, lifecycle WAL
pwrite bytes by **57.7–73.9%**, and write-phase direct SQLite VM steps by
**94.2–96.8%**. All 30 processes verify every content byte and SQLite integrity.
These are count results, without a FUSE throughput or tail-latency claim.

This follows the [FUSE wait diagnosis](../fuse_waits/README.md), which found
checkpoint-lock waits overlapping multi-second payload synchronization and
foreground WAL/database synchronization. Reducing the amount of version
metadata addresses repeated SQL/WAL work. It does not bound payload or device
sync duration. Production format 2, its installed module and default SDK remain
unchanged; this candidate builds separately from the frozen format-1004
[borrowed-buffer prototype](../native_write_costs/README.md).

## Implementation and invariants

[generate.py](generate.py) produces a separate component checkout. The
[source manifest](validation/source.json) records ten reversible changes to
existing files, two added headers, original/default identities and all source
hashes. Reversing each recorded replacement reconstructs the frozen original
bytes exactly. No production source file changes.

- `block_versions` adds `count` and `payload_offset`. A single eight-byte
  external token covers a physical 256 KiB append. Contiguous logical blocks
  form one version range; logical gaps create separate ranges sharing that
  token with different offsets. Underfilled batches retain scalar inline
  payloads. The physical append threshold and 64-block batch limit stay fixed.
- A block mutation splits overlapping rectangles in both logical-block space
  and branch-coordinate space. Left/right spatial fragments preserve their
  original coordinate interval; the changed block preserves coordinate prefix
  and suffix versions. SQL rollback covers all range splits and new metadata.
- Range reads join once per extent, reject invalid/overlapping ranges and
  missing payloads, clip partial edges and coalesce contiguous physical reads.
  Snapshot, diff and merge references retain both payload identity and offset.
- GC keeps referenced physical 4 KiB slices, including slices retained by old
  snapshots and nested branches. It can punch unused parts of a shared
  allocation and truncate a completely dead tail without losing retained data.

The payload-before-WAL publication order, OPEN/CLEAN fence protocol, FULL
barriers, background checkpointer, 16/64 MiB thresholds, error/retry paths and
recovery algorithms are unchanged. Format and sidecar identifiers are distinct.
Formats 1004 and 1005 reject each other before changing file bytes. The related
suite also checks rejection against production format 2. There is no migration.

Scalar mutation, truncate, merge and tree enumeration still use block-based
compatibility paths and can materialize many individual records. This prototype
targets new sequential writes; it does not establish general extent-efficient
overwrite/merge performance or a production-ready storage migration.

## Matched count experiment

The same [probe](probe.cpp) links each Release core to the same default-fsync
SQLite static SDK. Each process creates a fresh workspace in fsync durability
mode on ext4. Three shapes cover aligned 1 MiB calls and native replay of the
previously observed page-offset-48 FUSE splits:

| Shape | Calls for 256 MiB | Native request lengths |
| --- | ---: | --- |
| aligned | 256 | 1 MiB |
| fuse8 | 288 | For each 8 MiB: 1 MiB − 48, seven × 1 MiB, 48 |
| fuse1 | 512 | For each 1 MiB: 1 MiB − 48, 48 |

The `fuse1` 64 MiB replay has 128 native calls, matching the preceding FUSE
diagnosis. Its 256 MiB case deliberately repeats that request pattern at the
larger size. These are direct native calls, not additional mounted FUSE runs.

Six 4 MiB validation processes precede 24 larger count processes: two AB/BA
pairs per shape and size (64/256 MiB). Buffers contain deterministic, distinct
block data and are prepared before writes. Every process performs all writes,
an explicit FULL sync, full byte comparison and content hashing, handle/mount
release, workspace close, then read-only SQLite checks. Both repetition orders
and every sample remain in [counts.json](counts.json); no slow-sample exclusion
or throughput acceptance rule applies to this count-only study.

Process-wide atomic counters include foreground and checkpoint-worker calls.
Real pwrite/pwrite64 calls retain successful bytes and failures, separated by
WAL, database, payload, fence and other descriptors. Real fsync/fdatasync and
checkpoint calls are retained per phase. Attribution uses the phase at call
entry; workspace close joins the worker before totals are read. Direct
`sqlite3_step` counters retain actual VM steps, excluding SQLite-internal
`sqlite3_exec` execution. The wrappers preserve syscall return values and errno;
both binaries pass a separate failure/SQL-counter self-test.

## Results

Each value is the median of the two retained repetitions. Lifecycle counts
include setup, writes, final barrier, content validation and explicit close;
read-only SQL inspection is separate. WAL bytes count successful API writes,
including overwritten/reused frames, rather than physical device writes.

| 256 MiB shape | Version rows before → after | Lifecycle WAL MiB before → after | WAL reduction | Write VM-step reduction |
| --- | ---: | ---: | ---: | ---: |
| aligned | 65,536 → 1,024 | 39.99 → 10.56 | 73.6% | 96.8% |
| fuse8 | 65,536 → 1,280 | 41.99 → 14.05 | 66.5% | 94.9% |
| fuse1 | 65,536 → 1,280 | 49.17 → 20.80 | 57.7% | 94.2% |

At 64 MiB, version rows fall from 16,384 to 256 aligned or 320 with either
split pattern. Lifecycle WAL byte reductions are 73.9%, 66.8% and 57.8%.
Both sizes reduce external payload records by exactly 64 times. Split patterns
also retain one scalar inline payload per logical MiB, and replaced payload
records remain until normal GC; total payload rows are distinct from version
rows. The [analysis](validation/analysis.json) retains every count and reduction.

Physical payload pwrite bytes are identical between variants: exactly the
logical data size during writes, plus the shared 4 KiB header during setup.
Every final barrier still performs one payload fdatasync and one WAL fsync.
The explicit close remains counted; savings are not obtained by dropping
its writes or moving them outside the accounting window.

At 256 MiB, lifecycle PASSIVE checkpoint calls have medians 5 / 5 / 7.5 in
the control and 0 / 0 / 3 in the extent candidate. Both policies retain the
same thresholds. Smaller metadata stays below the worker trigger longer,
reducing observed midstream publications and database backfill. Worker
scheduling can affect these counts. Unchanged `NO_CKPT_ON_CLOSE` also leaves
durable WAL for subsequent work: this short fresh-workspace window does not
establish steady-state checkpoint cost or eliminate deferred work. A complete
fresh/after-GC FUSE workload must validate that boundary next.

## Validation and evidence

Only related tests run. The candidate passes **153 Python tests**, **12 Release
C++ component suites**, **12 ASan/UBSan component suites**, and **24 new extent
case executions** across Release/sanitized builds and strict/fsync modes.
[test_extents.cpp](test_extents.cpp) covers cross-extent writes, clipped reads,
input lifetime, snapshots, nested branch-coordinate splits, merge/diff,
truncate/zero extension, skipped-zero logical gaps, partial-slice GC, tail
reclamation/reuse, rollback after spatial splitting and physical append,
malformed offsets and overlapping ranges. Two additional cross-format
rejection executions preserve all existing file bytes; both counter binaries
pass their self-tests. Logs and exact commands are in
[validation/logs.json](validation/logs.json) and the build manifests.

The two Python skips are unchanged: the isolated format has no production-v1
migration, and optional image/video coverage lacks PyAV. The inherited-fork
test emits the expected multithreaded-fork deprecation warning.

Initial fixture runs are retained: the checkpoint test first misses its WAL
budget because it creates padding per payload record; Python fault injection
first fails to reach the old 100-record trigger after coalescing. C++/Python
trigger adapters count covered logical bytes/blocks, keeping the original
backpressure budget, multi-batch rollback assertions and error semantics.
An initial setup reverse-proof collision and a cross-format harness catching
the wrong Python exception are also preserved under [preflight](preflight).
All original failures are retained; every collected count run passes.

The [audit](audit.json) verifies frozen source/binary/default identities,
nonoverlapping build/test/count/analysis windows, complete sample retention,
stopped processes and **40 cleaned owned data directories**, including the
failed setup attempt. Cleanup records
retain content hashes and removed paths; generated databases/payloads are not
committed. Host samples, failed preflights, harness/source histories and logs
are retained losslessly in files below 500 KiB. No global cache eviction,
CPU/device tuning or default-module reinstall occurs.

Reproduce with the frozen parent component and SDK available:

```bash
here=vane_fs/benchmarks/staged_payload/versioned_extents
out="$PWD/vane_fs/build/versioned-extents-repro"
parent="$PWD/vane_fs/build/native-write-costs-20261017"
python3 "$here/generate.py" "$out" --parent "$parent"
python3 "$here/build.py" "$out" --stage all
python3 "$here/count.py" "$out" --stage prepare
python3 "$here/count.py" "$out" --stage run
python3 "$here/analyze.py" "$out"
```

The parent supplies its verified borrowed-buffer core and separate dependencies;
its reproduction procedure is in the preceding report. Candidate installation
is non-editable and incremental, inside its own venv and build directory.
Run builds/tests completely before the count window. Full ordinary FUSE
acceptance and sustained fragmentation/GC workloads remain the next step before
any adoption decision.

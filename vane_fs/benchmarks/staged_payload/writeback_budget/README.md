# Budgeted background payload writeback

This follows the [background submission diagnosis](../background_writeback/README.md).
Its worker removed foreground submission blocking, but coalescing produced
48 MiB requests and an observed 2.480 s submission. An optional hint also
overlapped the final whole-file data barrier. This experiment tests a smaller,
paced submission budget and pauses new claims during FULL publication.

The control remains the format-1004 partial-edge candidate **without hints**.
The preceding background candidate is used for correctness negative controls,
not as the throughput baseline. Production format 2 and its default durability
policy are unchanged.

**Decision: REJECT for adoption.** Both ext4 large-write targets pass, but
nine frozen nonregression checks fail. The measured native implementation
remains isolated; the submission parameters were not tuned after seeing results.

## Implementation

[`budget.patch`](budget.patch) changes only `src/staging.hpp` and
`src/workspace.cpp` relative to that no-hint control:

- Keep one coalesced pending range and at most one active submission. Enqueue
  after 16 MiB accumulates, then drain the range in requests of at most 4 MiB.
- Wait at least 16 ms after each completed request before claiming another.
  The idle wait is interruptible by pausing, shutdown or cancellation. This
  limits sustained requested bytes to a nominal 250 MiB/s per coordinator;
  it is not a device-wide physical-I/O limit. The deadline survives a worker
  restart after failed close.
- Use a balanced RAII pause for FULL transaction commits, managed checkpoint
  publication, initialization and close. A pause prevents another queue claim;
  an already claimed/in-kernel request can still start or finish. Publication
  does not wait for it or hold the queue mutex during a syscall.
- Keep pausing through whole-payload `fdatasync`, FULL SQLite COMMIT and the
  durable CLEAN fence. Successful publication cancels redundant unsent hints;
  exception unwinding resumes the queue. An `fsync`-mode `Snapshot()` remains
  a NORMAL metadata transaction and does not acquire this FULL pause.
- Preserve the pre-append payload descriptor, unsupported-hint fallback,
  EINTR retry, asynchronous EIO/ENOSPC reporting, close retry and fork abandonment
  from the preceding experiment. Hints use WRITE-only flags and never publish
  metadata or a fence.

Pausing is local to one Database coordinator. Workers in other connections or
processes remain independent, as does kernel writeback. The final durability
barriers retain their original order and do not rely on hint completion.

## Frozen comparison

Each medium/state has six alternating AB/BA pairs: tmpfs and ext4, fresh and
independently prepared after-GC workspaces, for 48 plain mounts. Durability is
`fsync`; application buffers start 48 bytes into a page. The workload and
prepared runner hashes match the preceding complete comparison.

The target includes the complete first 256 MiB write (8 MiB application calls)
and subsequent 64 MiB write (1 MiB calls), including their synchronization.
Both ext4 states must improve aggregate throughput by at least 10% and win
at least 5/6 pairs. Every workload must retain at least 95% of control median
and aggregate throughput. Pooled request p99 must stay within 1.2x or increase
by at most 1 ms; median mount lifecycle time may increase by at most 5%.

All slow samples remain. No owned build or correctness test overlaps a plain
or diagnostic window. There is no global cache eviction or kernel/device
configuration change. Source, binary and SDK identities are checked before
and after each plain medium.

## Results

| Medium/state | Control combined MiB/s | Candidate combined MiB/s | Ratio | Paired wins |
|---|---:|---:|---:|---:|
| tmpfs, fresh | 196.98 | 200.90 | 1.020 | 3/6 |
| tmpfs, after GC | 193.66 | 191.96 | 0.991 | 3/6 |
| ext4, fresh | 111.62 | 130.80 | 1.172 | 6/6 |
| ext4, after GC | 110.69 | 128.01 | 1.156 | 6/6 |

Fresh ext4 first-256-MiB aggregate rises from 152.25 to 193.89 MiB/s;
after-GC aggregate rises from 156.56 to 193.13. Subsequent-64-MiB aggregate
rises from 53.99 to 56.83 and from 50.97 to 54.50 respectively. These are
complete phase timings, not the write-call timer alone.

Both request time and final synchronization contribute to the first-write
gain. First-write fsync medians fall from 159.36 to 95.15 ms fresh and
162.87 to 89.93 ms after GC. Subsequent-write fsync medians fall from
177.41 to 107.94 and 204.30 to 126.58 ms. The largest subsequent-write
fsync in this window falls from 225.09 to 133.05 ms fresh and 211.53 to
133.18 ms after GC. Neither control nor candidate reobserves the historical
multi-second payload stalls here; this does not establish a worst-case bound.

The remaining failures include:

| Measurement | Control | Candidate |
|---|---:|---:|
| Fresh tmpfs subsequent-write request p99, ms | 21.04 | 26.78 |
| Fresh tmpfs read aggregate, MiB/s | 796.39 | 729.94 |
| After-GC tmpfs read aggregate, MiB/s | 818.11 | 736.10 |
| Fresh tmpfs random-write aggregate, MiB/s | 19.60 | 16.62 |
| After-GC tmpfs random-read aggregate, MiB/s | 71.25 | 67.64 |
| Fresh ext4 random-write aggregate, MiB/s | 13.81 | 11.95 |
| Fresh ext4 rename aggregate, operations/s | 2025.05 | 1899.05 |
| Fresh ext4 rmdir aggregate, operations/s | 2305.85 | 2071.04 |
| After-GC ext4 rmdir aggregate, operations/s | 2085.45 | 1865.68 |

For fresh ext4 random writes, median request time rises from 107.78 to
145.81 ms while the final fsync median changes from 23.93 to 22.55 ms.
Its regression cannot be attributed uniformly to the last fsync. After-GC
random writes improve, but this does not cancel a failed fresh-state gate.
Read and namespace request paths need separate attribution before adoption.

[`plain/summary.json`](plain/summary.json) retains every check and distribution.
The split arrays in `plain/` retain every request latency and scalar without
rounding or sampling; [`plain/windows.json`](plain/windows.json) retains the
remaining configuration, monitoring and cleanup records. The published
analyzer reconstructs exactly the same decision.

## Separate diagnosis

The independent eight-mount ext4 window runs after all plain processes stop.
All 229 real hints succeed, use WRITE-only flags, run on distinct submission
threads and request at most 4 MiB. The minimum observed completion-to-next-start
gap is 16.061 ms; the maximum submission takes 12.980 ms.

There are 568 retained FULL-pause intervals and no new hint starts inside them.
Two hints already active at pause entry overlap it, as the design permits.
This is observed evidence; a request claimed just before a pause can still
enter the syscall, and an EINTR retry remains part of that claimed request.
The blocking fixtures independently verify that another claim is prevented
through payload, WAL and CLEAN-fence synchronization, and that error unwinding
resumes pending work.

The candidate's four longest durable syscalls are first-write payload
`fdatasync` calls of 109–125 ms. Wait samples still identify `rq_qos_wait`,
`folio_wait_bit_common` and `jbd2_log_wait_commit`. A budget reduces optional
submission bursts; it does not remove device or journal waits at a mandatory
barrier. Whole-device counters do not identify per-file writes, another host
workload or an SSD fault.

This window diagnoses the mechanism and does not decide throughput adoption.
Nested/cross-thread intervals must not be added. The recorder filters ordinary
spans shorter than 1 ms: missing queue/shutdown markers are bounds, not measured
zero or complete enqueue counts. Actual hint syscalls are always retained.
See [`diagnostic/background-diagnostic.json`](diagnostic/background-diagnostic.json)
and [`diagnostic/diagnostic-summary.json`](diagnostic/diagnostic-summary.json).

## Related validation

The installed candidate passes 154 related Python tests; one unsupported
production-v1 migration check is skipped. The control and candidate each pass
11 C++ component suites, and the candidate passes 11 under ASan/UBSan. The
unchanged staging tests include 66 software crash/loss cases.

All 24 dedicated cases pass in Release and ASan/UBSan: normal/small writes,
three unsupported-hint fallbacks, EINTR, EIO, ENOSPC, payload-barrier failure,
blocked foreground progress, complete bounded draining, close blocking and
error/restart, fork, GC, late hint error, process exit with an active hint,
payload/WAL/CLEAN-fence pausing, failed payload/fence publication recovery,
NORMAL snapshot progress and strict-mode publication. Four negative cases
linked to the preceding background library fail as expected for excessive
submission/pacing and missing payload/WAL/fence pausing.

The initial fixture incorrectly treated a NORMAL snapshot as a FULL barrier.
The negative control also exposed an exact-counter polling race; its wait was
changed to a monotonic predicate before checking for a forbidden second call.
Initial reports and diagnostic logs are retained. These corrections changed
the fixture, not the measured native implementation. Software fault checks
do not certify physical power-loss behavior.

## Reproduction

Use the retained no-hint partial-edge source/build and preceding background
source/build, with the same SQLite and FUSE SDKs. All installs are non-editable.
Native builds use two jobs and an incremental `build/python-release` directory.

```bash
PYTHON=.venv/bin/python
OUT=vane_fs/build/writeback-budget-reproduction
"$PYTHON" vane_fs/benchmarks/staged_payload/writeback_budget/prepare.py "$OUT" \
  --parent "$(cat /tmp/vane-external-partial-current)" \
  --unbudgeted-parent "$(cat /tmp/vane-background-writeback-current)" \
  --sqlite-prefix "$PWD/vane_fs/vcpkg_installed/x64-linux-release" \
  --fuse-prefix "$PWD/.cache/fuse-sdk/root/usr" \
  --test-support /home/kaka/astrovela_vane-batch-udf-20260930/.venv-gpu/lib/python3.12/site-packages
"$PYTHON" vane_fs/benchmarks/staged_payload/writeback_budget/validate.py "$OUT"
"$PYTHON" vane_fs/benchmarks/staged_payload/writeback_budget/run.py "$OUT" --medium tmpfs
"$PYTHON" vane_fs/benchmarks/staged_payload/writeback_budget/run.py "$OUT" --medium ext4
"$PYTHON" vane_fs/benchmarks/staged_payload/partial_sync/analyze_writeback.py "$OUT"
```

After both plain windows stop, prepare and run the separate eight-mount ext4
diagnosis. Its syscall wrappers preserve the original result/errno and timing
does not decide throughput adoption.

```bash
TRACE=vane_fs/build/writeback-budget-trace-reproduction
"$PYTHON" vane_fs/benchmarks/staged_payload/partial_sync/prepare.py "$TRACE" \
  --parent "$OUT" --writeback --background-writeback --writeback-budget \
  --sqlite-prefix "$PWD/vane_fs/vcpkg_installed/x64-linux-release" \
  --fuse-prefix "$PWD/.cache/fuse-sdk/root/usr"
"$PYTHON" vane_fs/benchmarks/staged_payload/partial_sync/run.py "$TRACE" --pairs 1
"$PYTHON" vane_fs/benchmarks/staged_payload/writeback_budget/analyze_diagnostic.py "$TRACE"
"$PYTHON" vane_fs/benchmarks/staged_payload/writeback_budget/publish.py "$OUT" --diagnostic "$TRACE"
```

Publication reconstructs the native patch with fuzz 0, checks every native,
Python and test source identity, and reproduces the same plain summary from
the split raw arrays. It verifies stopped processes, removed owned data and
non-overlapping measurement/build/test windows. Large DB, payload, WAL and
temporary outputs are removed outside measured timers. Full local timelines,
wait/host logs, binaries and source inputs remain indexed by content hash.
See [`audit.json`](audit.json), [`reproducer-proof.json`](reproducer-proof.json)
and [`manifest.json`](manifest.json) for verification and cleanup evidence.

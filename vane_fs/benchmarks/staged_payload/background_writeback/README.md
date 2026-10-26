# Background payload writeback experiment

This follows the [foreground writeback diagnosis](../partial_sync/README.md).
WRITE-only `sync_file_range` can block while a request is allocated. The earlier
prototype made that call inside the SQLite transaction and publication guard.
This C++ candidate moves submission to a dedicated thread, preserving the
format-1004 partial-edge control and its final durability barriers.

**Decision: REJECT for adoption.** Blocking isolation passes, but the complete
48-mount plain comparison fails 16 frozen checks. Production format 2, the
installed production component, default SQLite SDK and durability policy are
unchanged. These measurements compare two isolated format-1004 prototypes.

## Implementation and lifecycle

[`background.patch`](background.patch) changes only `src/staging.hpp` and
`src/workspace.cpp` relative to the previously measured partial-edge candidate:

- Open the payload error-observation description before the first external
  append and keep it through the final barrier and worker shutdown.
- Accumulate a range until it spans at least 16 MiB. Keep one active range and
  one coalesced pending range. Queue memory is constant; a coalesced request can
  exceed 16 MiB and include gaps.
- Submit only `SYNC_FILE_RANGE_WRITE`, without holding the SQLite transaction,
  publication guard, payload guard or queue mutex in the submission thread.
- Disable unsupported hints on ENOSYS/EOPNOTSUPP/EINVAL, retry EINTR and retain
  EIO/ENOSPC for a subsequent mutation, barrier or close to report. An ordinary
  NORMAL commit can return before a hint fails; reporting that error does not
  undo the already visible commit.
- Keep the whole-payload `fdatasync`, FULL WAL commit and durable CLEAN fence
  in their existing order. Hints never publish a fence. A successful whole-file
  barrier discards unsent hints; it does not depend on an active hint finishing.
- Join the active worker before closing its descriptor. A failed close remains
  retryable. Fork children close their inherited descriptors and abandon the
  parent's thread and mutexes through the existing Database abandonment path.

The worker's error is checked before and after the final payload barrier.
An error reported after a successful barrier is still reported by a later
operation; the actual barrier, rather than completion of the optional hint,
determines durability.

## Plain comparison

Each medium/state has six alternating AB/BA pairs. The control is the original
partial-edge candidate without hints; the candidate uses the same SQLite SDK,
256 KiB externalization threshold, SQL batching, checkpoint settings and FUSE
cache policy. Fixed application buffers start 48 bytes into a page. Durability
is `fsync`. Every slow sample is retained. No owned build or correctness test
overlaps either plain comparison window.

The target is the complete first 256 MiB write plus the subsequent 64 MiB write,
including their synchronization. The frozen gate requires at least 10% higher
aggregate throughput and wins in at least 5/6 pairs in **both** ext4 states.
Individual median/aggregate throughput must remain at least 95% of control;
request p99 must remain within 1.2x or increase by no more than 1 ms. Median
mount lifecycle time may increase by at most 5%.

| Medium/state | Control combined MiB/s | Candidate combined MiB/s | Ratio | Paired wins |
|---|---:|---:|---:|---:|
| tmpfs, fresh | 200.91 | 199.55 | 0.993 | 4/6 |
| tmpfs, after GC | 191.34 | 194.80 | 1.018 | 4/6 |
| ext4, fresh | 85.46 | 82.48 | 0.965 | 4/6 |
| ext4, after GC | 68.86 | 73.89 | 1.073 | 3/6 |

The fresh ext4 first-write median rises from 155.66 to 216.34 MiB/s, but its
aggregate falls from 155.07 to 123.61 MiB/s after retaining the slow first-write
sample. Looking only at the median would give the wrong adoption decision.

Other material regressions:

| Measurement | Control | Candidate |
|---|---:|---:|
| Fresh ext4 random-write aggregate, MiB/s | 11.95 | 2.61 |
| Fresh ext4 small-file/fsync aggregate, files/s | 259.22 | 52.18 |
| Fresh ext4 small-file/fsync pooled p99, ms | 13.10 | 124.53 |
| Fresh ext4 subsequent-write request p99, ms | 44.10 | 72.68 |
| After-GC ext4 subsequent-write aggregate, MiB/s | 31.71 | 21.17 |
| After-GC ext4 random-write aggregate, MiB/s | 12.37 | 5.83 |
| Fresh tmpfs read aggregate, MiB/s | 802.75 | 709.02 |

Fresh ext4 median lifecycle time also rises 41.2%. The failed checks and full
statistics are in [`plain/summary.json`](plain/summary.json). Raw per-run request
vectors, phase timers, CPU measurements, content checks and cleanup records are
split by medium/state/variant in `plain/`. Compact JSON preserves every vector
entry without sampling or rounding. [`plain/windows.json`](plain/windows.json)
retains the remaining monitor/configuration data. The published analyzer
reconstructs exactly the same summary from these split files.

## Separate diagnosis

An independent eight-mount instrumented ext4 window runs AB/BA for fresh and
after-GC workspaces, after the plain comparison and its processes finish.
It retains actual syscalls/results, file descriptions, thread identities and
nominal 2 ms wait samples. It does not decide throughput adoption.

All 88 hints use WRITE-only flags, return success and appear on threads distinct
from the retained publication/admission markers. Queue/shutdown spans below
1 ms are intentionally filtered by the reused recorder; missing markers give
only a **less-than-1-ms bound**, not a measured zero or a complete enqueue count.
The forced-blocking tests independently prove foreground progress while a hint
remains blocked.

The fresh candidate's slow subsequent 64 MiB phase takes 5.829 s:

- Whole-payload `fdatasync`: 3.604 s.
- WAL `fsync` interval union: 0.997 s.
- Background hint interval union: 3.518 s, including a 48 MiB request taking
  2.480 s. These intervals overlap and must not be added.

Its other fresh run's small-file phase takes 6.336 s, with 6.091 s in WAL
`fsync` interval union. The slow after-GC subsequent-write phase includes a
3.489 s payload barrier. Wait samples identify page writeback, request-QoS
allocation and ext4 journal waits (`folio_wait_bit_common`, `rq_qos_wait`,
`wait_transaction_locked`, `do_get_write_access`, `jbd2_log_wait_commit`).
Both variants also exhibit slow WAL synchronization; this window does not
establish hints as the sole cause of every tail.

Moving submission to a thread removes direct foreground submission blocking.
It does not remove physical writeback and journal waits at durable barriers.
A useful next experiment is a bounded submission budget and pausing optional
hints during FULL synchronization, retaining the current final barriers and
the same complete adoption gates. Device counters are whole-device observations;
they do not identify per-file writes, an SSD fault or another host workload.

[`diagnostic/background-diagnostic.json`](diagnostic/background-diagnostic.json)
contains thread/range checks; [`diagnostic/diagnostic-summary.json`](diagnostic/diagnostic-summary.json)
contains interval and wait analysis. Full syscall timelines and wait logs are
retained locally and indexed with their hashes in the manifest.

## Related validation and reproduction

The candidate passes 154 related Python tests with one explicit unsupported
production-v1 migration skip. Control and candidate each pass 11 C++ component
suites; the candidate passes the same 11 under ASan/UBSan. The unchanged staging
suite includes 66 software crash/loss cases.

All **17 dedicated cases** pass in Release and ASan/UBSan: small/normal writes,
unsupported hints, EINTR, EIO, ENOSPC, barrier failure, blocked foreground
progress, coalescing, close waiting/error retry, fork, GC, a late hint error after
a successful barrier, and process exit with a blocked hint. The latter verifies
an OPEN fence, rollback to the last durable content, repeated cold recovery and
GC. These are software-fault checks, not physical power-loss certification.
The same blocked-hint fixture linked to the preceding foreground prototype
fails as expected. A control without hints also fails the expected-hint assertion.

Two initial fixture failures are retained with their repairs: a child can clean
the shared fence using its own descriptor, and GC separately flushes hole
punching/truncation after durable metadata deletion. Neither required native
source changes. The initial diagnostic analyzer's incorrect demand for
sub-1-ms enqueue markers is also recorded; fixing that interpretation did not
rerun or discard any diagnostic measurement.

Reuse the frozen partial-edge candidate from
[`../partial_edges/README.md`](../partial_edges/README.md), then run:

```bash
.venv/bin/python -I vane_fs/benchmarks/staged_payload/background_writeback/prepare.py \
  /tmp/vane-background --parent /tmp/vane-external-partial \
  --sqlite-prefix "$PWD/vane_fs/vcpkg_installed/x64-linux-release" \
  --fuse-prefix "$PWD/.cache/fuse-sdk/root/usr" \
  --test-support /path/to/existing/test/site-packages
.venv/bin/python -I vane_fs/benchmarks/staged_payload/background_writeback/validate.py \
  /tmp/vane-background
.venv/bin/python -I vane_fs/benchmarks/staged_payload/background_writeback/run.py \
  /tmp/vane-background --medium tmpfs
.venv/bin/python -I vane_fs/benchmarks/staged_payload/background_writeback/run.py \
  /tmp/vane-background --medium ext4
.venv/bin/python -I vane_fs/benchmarks/staged_payload/partial_sync/analyze_writeback.py \
  /tmp/vane-background
.venv/bin/python -I vane_fs/benchmarks/staged_payload/partial_sync/prepare.py \
  /tmp/vane-background-trace --parent /tmp/vane-background \
  --sqlite-prefix "$PWD/vane_fs/vcpkg_installed/x64-linux-release" \
  --fuse-prefix "$PWD/.cache/fuse-sdk/root/usr" --writeback --background-writeback
.venv/bin/python -I vane_fs/benchmarks/staged_payload/partial_sync/run.py \
  /tmp/vane-background-trace --pairs 1
.venv/bin/python -I vane_fs/benchmarks/staged_payload/background_writeback/analyze_diagnostic.py \
  /tmp/vane-background-trace
```

The noneditable component install uses its own incremental Release build
directory. `--helpers-only` reuses already passed unchanged component suites when
only the dedicated fixture changes; an initial validation runs every related
suite. Do not run builds/tests inside measured windows.

All 56 measured mount processes stopped and unmounted before generated databases
and payloads were removed. Necessary logs, full timing vectors, source/binary/SDK
identities, content hashes and cleanup manifests remain. See
[`audit.json`](audit.json), [`reproducer-proof.json`](reproducer-proof.json) and
[`manifest.json`](manifest.json). No production or host tuning is applied.

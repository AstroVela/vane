# External payload synchronization and early writeback

**The 16 MiB early-writeback candidate is rejected and remains isolated.**
It passes its correctness checks, but foreground writeback submission can
block and the complete comparison fails its frozen performance gates.
Production format 2, the installed component, the strict FULL default and
checkpoint settings are unchanged.

## Locate the existing stalls

The first diagnostic compares the original cached format-1004 control with
the [partial-edge candidate](../partial_edges/README.md), on local ext4.
Four alternating AB/BA pairs in each fresh and independently GC-aged state
produce 16 mounts. This is an instrumented diagnostic, not a throughput
adoption window. It records publication and checkpoint mutex intervals,
per-file synchronization calls, nominal 2 ms owned-thread wait samples and
whole-device counters. Nested and overlapping intervals must not be added.

In the partial-edge candidate, payload `fdatasync` takes **3.623 seconds**
during a subsequent 64 MiB write. Samples locate waits in `rq_qos_wait`,
`folio_wait_bit_common` and `jbd2_log_wait_commit`. Another payload sync takes
3.440 seconds outside workload timers, inside a 4.152-second publication
interval, while a foreground publication-lock wait takes 4.151 seconds.
Those are overlapping observations of one stalled publication, not three
independent costs. WAL and database synchronization also have long calls.

This window does not reproduce the earlier candidate's first-256-MiB
6.573-second request stall. Several long sync calls occur during lifecycle
work outside workload timers; the report labels them explicitly. The
observations locate writeback, journal and publication waits. They neither
identify a device fault nor establish competing host workload as the cause.

## Implement an isolated writeback candidate

The plain comparison uses the **previous partial-edge candidate as its
control**. Both variants have identical external allocation, format 1004,
256 KiB externalization threshold, default SQLite SDK and durability policy.
This control differs from the original cached control in the first diagnostic.

The C++ patch submits `sync_file_range(..., SYNC_FILE_RANGE_WRITE)` after
16 MiB of contiguous appended payload. It uses no WAIT flags. An open payload
description survives through publication, and the original whole-file
`fdatasync` still precedes the FULL WAL commit. Submission never marks the
fence clean. Unsupported ENOSYS/EOPNOTSUPP/EINVAL disables the hint; EINTR
retries; I/O and space failures propagate. Fork abandonment closes the
description using the existing inherited-owner handling.

WRITE-only requests can still block while allocating write requests, and
they do not flush device caches or replace durability barriers. See the
[Linux 6.11 implementation and API comments](https://github.com/torvalds/linux/blob/v6.11/fs/sync.c).
The retained description provides an observation lifetime for asynchronous
errors; see [kernel errseq documentation](https://www.kernel.org/doc/html/v6.11/core-api/errseq.html).

## Complete performance window

The plan is frozen before the first plain comparison: six alternating AB/BA
pairs in each state on tmpfs and ext4, totaling **48 independent mounts**.
The first write is 256 MiB with an 8 MiB application buffer at page offset 48;
the subsequent write is 64 MiB with a 1 MiB buffer. Rates include each phase's
final fsync and close. Reads, random overwrites, small-file fsync, directory
operations and retained GC guards use the unchanged acceptance workload.
Separate lifecycle gates include preparation, validation and mount teardown.
All slow samples are retained. No owned builds or correctness tests overlap
either plain window, and no global cache, kernel or device settings change.

The primary target requires at least 10% improvement in combined large-write
aggregate throughput and at least 5/6 paired wins in **both ext4 states**.
Every individual workload must retain 95% of median and aggregate throughput;
request/operation p99 must stay within 20%, with a 1 ms absolute allowance.
Median lifecycle duration may increase at most 5%. These are conservative
acceptance gates, not statistical significance tests.

| Combined 256 + 64 MiB writes | Control, MiB/s | Candidate, MiB/s | Paired wins |
| --- | ---: | ---: | ---: |
| tmpfs, fresh | 198.50 | 193.51 | 3/6 |
| tmpfs, GC-aged | 190.41 | 190.89 | 3/6 |
| ext4, fresh | 66.29 | 61.67 | 4/6 |
| ext4, GC-aged | 47.04 | 67.69 | 3/6 |

The fresh ext4 first-256-MiB aggregate improves from **70.42 to 156.17 MiB/s**,
while the subsequent-64-MiB aggregate falls from **53.70 to 18.03 MiB/s**.
Its request p99 grows from 71.3 to 762.0 ms, and one final fsync takes
**6.173 seconds**. The two phases combined are 7.0% slower. A faster first
write therefore does not establish a faster complete sequence.

GC-aged combined ext4 throughput improves by 43.9%, but only three of six
pairs win. Its first-write request p99 rises from 366.9 to 1,040.4 ms.
GC-aged random-overwrite aggregate falls from **12.79 to 5.37 MiB/s** on ext4
and from **21.96 to 17.44 MiB/s** on tmpfs. Other directory, small-file and
read gates also fail; all 14 failed checks are retained in the summary.

## Diagnose the new submission path

A separate, subsequent **8-mount diagnostic** traces the partial-edge
control and writeback candidate, using one AB/BA cycle in each state. It
preserves the actual syscalls and records every hint's offset, range, flags,
result and duration. All **92 observed hints** use WRITE only and return
success; controls submit none. The totals include fixture/lifecycle work,
not just the two large-write timers.

One fresh candidate first-write phase takes **7.091 seconds**. Its WRITE-only
submissions occupy **4.738 seconds** in total, including one **1.057-second**
call whose owned-thread samples are all `rq_qos_wait`. Payload sync occupies
0.857 seconds in that same phase; WAL fsync occupies 0.070 seconds. These
labels describe separate interval unions, and nested publication/lock spans
must not be added to them. Instrumentation is unsuitable for comparing these
rates with the plain window, but it directly locates foreground submission
as a remaining stall.

The next candidate should remove hint submission from the foreground SQL
transaction and publication guard, while preserving the payload-before-WAL
barrier, asynchronous error observation, fork handling and GC lifecycle.
A bounded background mechanism needs its own correctness and full-sequence
comparison before adoption. Increasing or decreasing the current threshold
alone does not address the observed blocking call.

## Correctness and reproduction

Both plain variants pass their 11 native component suites. The candidate passes
**154 related Python tests**, with one explicit unsupported production-v1
migration skip, plus all 11 native suites under ASan/UBSan. The existing
66 staging software crash/loss cases remain covered; they do not certify
physical power-loss behavior or every asynchronous hint failure.

Nine additional native cases check normal submission, small writes, three
unsupported-call fallbacks, EINTR, EIO, ENOSPC and final-barrier failure.
They verify retained-description lifetime, payload-before-WAL ordering,
rollback/retry and snapshot content using injected hints and real final
barriers. All nine also pass under ASan/UBSan. Formatted Release helpers
reconstruct byte-identically; the sanitized helper's embedded source locations
change, and its nine cases are repeated successfully. The four diagnostic
mount programs also reconstruct byte-identically from formatted helpers.
Only related tests run.

Reuse the completed partial-edge source/build directory, and substitute local
SDK and test-support paths:

```bash
.venv/bin/python vane_fs/benchmarks/staged_payload/partial_sync/prepare.py \
  /tmp/vane-sync-diag --parent /tmp/vane-external-partial \
  --sqlite-prefix "$PWD/vane_fs/vcpkg_installed/x64-linux-release" \
  --fuse-prefix "$PWD/.cache/fuse-sdk/root/usr"
.venv/bin/python vane_fs/benchmarks/staged_payload/partial_sync/run.py /tmp/vane-sync-diag
.venv/bin/python vane_fs/benchmarks/staged_payload/partial_sync/analyze.py /tmp/vane-sync-diag

.venv/bin/python vane_fs/benchmarks/staged_payload/partial_sync/prepare_writeback.py \
  /tmp/vane-writeback --parent /tmp/vane-external-partial \
  --sqlite-prefix "$PWD/vane_fs/vcpkg_installed/x64-linux-release" \
  --fuse-prefix "$PWD/.cache/fuse-sdk/root/usr" \
  --test-support /path/to/existing/test/site-packages
.venv/bin/python vane_fs/benchmarks/staged_payload/partial_sync/validate_writeback.py /tmp/vane-writeback
.venv/bin/python vane_fs/benchmarks/staged_payload/partial_sync/run_writeback.py /tmp/vane-writeback --medium tmpfs
.venv/bin/python vane_fs/benchmarks/staged_payload/partial_sync/run_writeback.py /tmp/vane-writeback --medium ext4
.venv/bin/python vane_fs/benchmarks/staged_payload/partial_sync/analyze_writeback.py /tmp/vane-writeback

.venv/bin/python vane_fs/benchmarks/staged_payload/partial_sync/prepare.py \
  /tmp/vane-hint-diag --parent /tmp/vane-writeback --writeback \
  --sqlite-prefix "$PWD/vane_fs/vcpkg_installed/x64-linux-release" \
  --fuse-prefix "$PWD/.cache/fuse-sdk/root/usr"
.venv/bin/python vane_fs/benchmarks/staged_payload/partial_sync/run.py /tmp/vane-hint-diag --pairs 1
.venv/bin/python vane_fs/benchmarks/staged_payload/partial_sync/analyze.py /tmp/vane-hint-diag
```

`run_writeback.py` freezes the published gates and reuses its parent's prepared
runner identities. When relocating the checkout, regenerate that parent's
prepared-runner hash for the new absolute path before any measurements.
The published `writeback/` split samples regenerate the complete performance
summary with `analyze_writeback.py`. Diagnostic summaries and raw workload
samples are published under `diagnostic/` and `hint_trace/`; regeneration of
wait analysis additionally uses the retained local timelines and wait logs,
whose hashes and paths appear in the artifact manifest.

The initial diagnostic fixture lacked a CLI symlink and failed before its
first mount. The initial hint test used an aligned overwrite that never
reached an external span; it was corrected to offset 1 before any plain
measurement. Both failures and repairs are retained. They are not discarded
slow performance samples. Generated databases, payloads and temporary test
data are removed only after their owners stop and mounts disappear; content
hashes, configurations, logs, source/binary identities and cleanup records
remain in the audit and manifests.

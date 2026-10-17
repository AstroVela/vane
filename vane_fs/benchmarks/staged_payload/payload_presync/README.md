# Preparatory payload sync outside the publication lock

**The candidate is not adopted.** It preserves the publication contract and
reduces first-write lock waiting in diagnostics, but retains severe sync tails.
The original adoption comparison fails fourteen gates. Against the preceding
orphan-batching candidate, fresh combined throughput falls 14.7%; post-GC rises
11.3%. Production source, installed module, format 2 and defaults are unchanged.

## Cause and isolated implementation

The preceding orphan-batching experiment removes most of the previous file's
last-reference cleanup delay. Its next 64 MiB write still waits for background
publication: roughly 73 ms cumulative publication-lock wait in the fresh
diagnostic phase, with worker payload synchronization inside that lock.

`presync.patch` changes only `src/staging.hpp` and `src/checkpoint.hpp` over
the frozen no-hint orphan-batching source. An OPEN worker pass that is eligible
for publication first releases the publication guard and synchronizes the
payload file. It then reacquires the guard, reloads the fence, rechecks committed
WAL progress, and uses the original publication sequence. A foreground FULL
barrier or GC may complete meanwhile; if the fence is now CLEAN, publication
does not redundantly synchronize the payload. CLEAN metadata tails retain their
original prompt PASSIVE backfill, and the 16 MiB trigger, 64 MiB admission limit,
retry/error rules and per-commit notifications remain unchanged.

The preparatory sync changes no fence and never authorizes checkpointing or
acknowledges durability. Concurrent appends require the final protected sync.
`BeforeFullCommit`, `AfterFullCommit` and `Publish` bodies are byte-identical to
the preceding control, verified by `analyze_presync.py`. The checkpoint
serialization mutex remains held across this work; releasing the publication
lock does not remove all possible waiting or make a storage sync latency bound.
Orphan cleanup is unchanged, and neither rejected hints nor scheduler delays
are inherited.

## Correctness and controls

Seven dedicated native scenarios pass in Release and ASan/UBSan:

- Append a new external file while the worker's preparatory sync is blocked;
  OPEN persists and a later protected sync still occurs.
- Complete a foreground FULL barrier while presync is blocked; the worker
  adopts CLEAN without issuing another payload publication.
- Complete GC in that interval, preserving retained snapshot bytes and valid
  payload references despite hole punching/truncation.
- Inject an EIO into preparatory synchronization, preserve OPEN and retry,
  and surface the worker error before a subsequent mutation.
- Inject an EIO into the final protected sync, checking rollback, error
  propagation, OPEN retention and a successful retry.
- Backfill a small CLEAN tail without any payload presync.
- Complete presync, append more data, and kill the process before the final
  protected sync. Restore only the payload image captured by presync; repeated
  recovery must retain earlier acknowledged content/snapshots and discard the
  unpublished WAL references. GC then succeeds. This is a software loss model,
  not physical power-loss certification.

The preceding control is an expected negative for foreground progress during
the blocked payload sync. A separately built unsafe control removes only the
final protected sync and fails the concurrent-append barrier assertion. Neither
control is used as a production change. Their exact failure messages are checked.

The installed candidate passes **154 related Python tests**, with one explicit
unsupported production-v1 migration skip. Original control and candidate each
pass 11 C++ component suites; the candidate also passes all 11 under ASan/UBSan.
Four inherited checkpoint-policy cases verify prompt CLEAN progress, progress
during continuous commits, no extra commit-hook queries, and Stop/Start without
another commit. Seven inherited orphan cases verify bounded SQL work, sparse
holes/tombstones, snapshots/parent retention, future versions, rollback/retry,
lease release and owner recovery. The helper report records **54 executions**,
including two expected negatives, and three sanitizer helper compilations.
Only related tests were run.

## Complete original-control comparison

The original workload, six alternating AB/BA pairs per state/medium, page-offset
48 buffers and gates remain frozen: 48 uninstrumented mounts on tmpfs/ext4,
fresh/post-GC, with complete first-256 plus subsequent-64 writes, reads, random
I/O, small-file fsync and directory operations. Every content/schema/guard and
SQLite check passes. No builds, tests or format hooks overlap measurement
windows; global caches and CPU/device settings are unchanged. Slow samples are
retained without replacing runs or loosening any gate.

Combined 320 MiB aggregate throughput includes creation, all writes, final
fsync and close:

| Medium/state | Control, MiB/s | Candidate, MiB/s | Change | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| tmpfs/fresh | 194.93 | 307.83 | +57.9% | 6/6 |
| tmpfs/post-GC | 193.56 | 312.50 | +61.5% | 6/6 |
| ext4/fresh | 111.87 | 101.24 | −9.5% | 4/6 |
| ext4/post-GC | 110.44 | 154.59 | +40.0% | 6/6 |

Fresh first-256 median throughput improves from 155.94 to 188.15 MiB/s, but
its aggregate falls from 155.02 to 99.86 because candidate rounds retain
approximately 5.52 and 4.44 second complete writes. First-write p99 also rises
from 353.17 to 371.80 ms. Subsequent-64 median throughput rises from 52.82 to
106.28 MiB/s, while request p99 rises from 69.93 to 153.92 ms. These cumulative
gains include the inherited orphan batching, not just presync.

Fourteen adoption gates fail: fresh tmpfs random writes; post-GC tmpfs random
writes, small-file fsync, mkdir and rmdir; fresh ext4 first-write, subsequent-write,
random writes, mkdir, rename and combined-write target; post-GC ext4
subsequent-write, random writes and small-file fsync. Post-GC candidate small-file
fsync retains a phase around 6.11 seconds, and the control retains its own slow
random-write/fsync phases. These observations do not identify another workload
or a device fault as their cause.

## Direct comparison with preceding orphan batching

Another 24 uninstrumented ext4 mounts use the immediately preceding candidate
as their baseline, six alternating pairs per state. This separates presync from
the inherited cleanup improvement; its reference gates never replace adoption
gates.

| State | Orphan control, MiB/s | Presync, MiB/s | Change | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| fresh | 138.64 | 118.20 | −14.7% | 4/6 |
| post-GC | 137.90 | 153.52 | +11.3% | 5/6 |

Fresh first-256 median rates improve from 155.06 to 189.54 MiB/s, but a subsequent
64 MiB candidate phase lasts about 4.53 seconds. Its aggregate subsequent-write
rate falls from 96.93 to 50.47 MiB/s. Fresh subsequent-write, sequential-read and
combined-write reference checks fail. Post-GC reference checks pass, including
their retained slow fsync phases in both variants. This is insufficient to
adopt a candidate that regresses the fresh state.

## Separate trace: less first-write lock waiting, persistent sync tails

Sixteen ext4 diagnostic mounts use preceding orphan batching as control, four
alternating pairs per state. The existing recorder adds a `presync` stage on
both variants, zero in the control. Read-only owned-CPU sampling stays at 5 ms.
Inclusive span times are not additive, and these rates do not decide adoption.

During first-256 writes, cumulative foreground publication-lock wait medians
fall from 691 to 272 ms fresh and 542 to 280 ms post-GC. Worker publication count
falls from two to one per phase in the median, with one unlocked presync. Final
application fsync medians increase from 158 to 205 ms fresh and 163 to 207 ms
post-GC. During subsequent-64 writes, publication wait barely changes fresh
(67.3 to 65.5 ms) and increases post-GC (69.6 to 85.2 ms). Worker cumulative
`fdatasync` time increases from 54.6 to 100.2 ms fresh and 57.6 to 132.3 ms
post-GC, including preparatory work and fence syncs. CPU frequency and instruction counts are
retained in the reports; settings were not modified.

One fresh candidate subsequent-write phase lasts **about 5.10 seconds**. A
background worker spends 3.927 seconds in unlocked payload presync while the
foreground final fsync spends 3.787 seconds in its protected publication sync.
Those spans overlap and must not be summed. An earlier write also spends
930 ms in `fdatasync`. Foreground FULL publication establishes CLEAN before the
worker's later recheck, so the worker skips a redundant protected publication.
Releasing the mutex cannot eliminate this shared-file/storage synchronization
cost. The trace explains that retained stall, not every plain-window tail.

Do not enable this presync policy or simply remove the final sync. The next
investigation should attribute payload, WAL and fence synchronization separately
and evaluate a bounded payload publication batch. Further overlap work must
preserve a durable WAL recovery prefix and prevent GC from reclaiming referenced
data; this experiment does not implement that design.

## Reproduction and retention

`prepare.py` requires the frozen original partial-edge control as `--parent`
and the frozen orphan-batching experiment as `--orphan-parent`. It builds a
non-editable installed candidate, reuses the two controls, and builds the unsafe
negative control with separate source/library hashes. Supply the same SQLite
and FUSE SDKs and an existing test-support site-packages directory as in earlier
experiments. Run `validate.py` before measurements, and prepare the diagnostic
copy before any measured window.

```bash
.venv/bin/python vane_fs/benchmarks/staged_payload/payload_presync/prepare.py \
  /tmp/vane-presync --parent /path/to/frozen-partial-edge-control \
  --orphan-parent /path/to/frozen-orphan-batching \
  --sqlite-prefix "$PWD/vane_fs/vcpkg_installed/x64-linux-release" \
  --fuse-prefix "$PWD/.cache/fuse-sdk/root/usr" \
  --test-support /path/to/test/site-packages
.venv/bin/python vane_fs/benchmarks/staged_payload/payload_presync/validate.py /tmp/vane-presync
.venv/bin/python vane_fs/benchmarks/staged_payload/payload_presync/prepare_diagnostic.py \
  /tmp/vane-presync-trace --parent /tmp/vane-presync \
  --sqlite-prefix "$PWD/vane_fs/vcpkg_installed/x64-linux-release" \
  --fuse-prefix "$PWD/.cache/fuse-sdk/root/usr"
.venv/bin/python vane_fs/benchmarks/staged_payload/payload_presync/run.py /tmp/vane-presync --medium tmpfs
.venv/bin/python vane_fs/benchmarks/staged_payload/payload_presync/run.py /tmp/vane-presync --medium ext4
.venv/bin/python vane_fs/benchmarks/staged_payload/partial_sync/analyze_writeback.py /tmp/vane-presync
.venv/bin/python vane_fs/benchmarks/staged_payload/payload_presync/run_causal.py /tmp/vane-presync
.venv/bin/python vane_fs/benchmarks/staged_payload/partial_sync/analyze_writeback.py /tmp/vane-presync/causal
.venv/bin/python vane_fs/benchmarks/staged_payload/payload_presync/run_diagnostic.py /tmp/vane-presync-trace
.venv/bin/python vane_fs/benchmarks/staged_payload/request_costs/analyze.py /tmp/vane-presync-trace
.venv/bin/python vane_fs/benchmarks/staged_payload/request_costs/analyze_cpu.py /tmp/vane-presync-trace
.venv/bin/python vane_fs/benchmarks/staged_payload/checkpoint_batching/analyze_diagnostic.py /tmp/vane-presync-trace
.venv/bin/python vane_fs/benchmarks/staged_payload/payload_presync/analyze_presync.py \
  /tmp/vane-presync --diagnostic /tmp/vane-presync-trace
.venv/bin/python vane_fs/benchmarks/staged_payload/payload_presync/publish.py \
  /tmp/vane-presync --diagnostic /tmp/vane-presync-trace
```

Primary and causal split arrays reproduce their exact analyzer decisions.
Diagnostic phase records and slow overlaps are split losslessly; each published
JSON remains below 500 KiB. Manifests retain source, installed module, SDK,
library, executable, runner, test and measurement identities. All **88 mounts**
validate content and stop/unmount, and **143 owned temporary paths** are removed
after their owners stop. Large generated databases and payloads are not retained.

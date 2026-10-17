# Orphan tombstone batching within exact intervals

**The candidate remains isolated.** The patch removes most of the CPU cost
of reclaiming a newly written large orphan, but ten original performance gates
still fail. Production format 2, its installed module and defaults are unchanged.
This candidate starts from the original no-hint partial-edge control. It does
not inherit the rejected hint budget, threshold coalescing or 2 ms timer.

## Cause and implementation

The low-level FUSE daemon processes `forget` on its request thread. After a
256 MiB file is unlinked, releasing its final inode reference publishes 65,536
block tombstones through individual `Put` calls. Each call queries overlapping
versions, deletes them, reconstructs preserved intervals and inserts a tombstone.
The prior trace measured 9,830,796 SQLite VM instructions and roughly a second
of handler CPU time. The next file's create waits behind this cleanup; the
existing application timer already includes that wait.

`batch.patch` changes only `CleanupOrphans` in `src/workspace.cpp`. It checks
every overlapping block version, including deleted and future versions. When
all overlaps have exactly the writable interval's point and high bound, one
`UPDATE` sets live blocks to the current writer, `deleted=1` and `payload=NULL`.
Existing tombstones and non-overlapping versions remain untouched. Any interval
that needs splitting uses the original `Visible`/`Put` path. The enclosing
transaction, inode tombstone, generation increment, open-reference rules,
payload GC and durability ordering are unchanged. There is no format or index
change, and no improvement is claimed for inherited versions requiring splits.

## SQL work and lifetime checks

A separate native probe creates a 256 MiB open-unlinked file, counts only the
final close's `sqlite3_step` calls and VM instructions, compares every resulting
block version to the old per-key semantics, and checks GC. Its timing is a
diagnostic sample on tmpfs, not an adoption throughput result.

| Final close, 65,536 blocks | Control | Candidate |
| --- | ---: | ---: |
| `sqlite3_step` calls | 327,701 | 22 |
| VM instructions | 9,830,796 | 4,850,078 |
| Probe wall time, ms | 818.600 | 232.641 |

VM work falls by 50.7%. The old control fails the explicit bounded-SQL-work
assertion, after passing its content, interval, generation and GC checks. Six
other helper scenarios pass on both variants: existing tombstones/sparse holes,
snapshot and parent retention, future versions, rollback and retry, lease release
and abandoned-owner recovery. Applicable lifetime cases run in both strict and
fsync modes. Failure injection checks partial tombstone publication and a later
generation-update failure; the durable pin and local reference remain retryable.

The candidate's installed module passes **154 related Python tests**, with one
explicit unsupported production-v1 migration skip. Both variants pass their
11 C++ component suites. The candidate also passes all 11 and all seven helper
scenarios under ASan/UBSan. No broad Vane/Ray test suite was run.

## Complete uninstrumented comparison

The original workload and gates are frozen before measurement: six alternating
AB/BA pairs in fresh and independently GC-aged workspaces, on tmpfs and ext4,
for 48 independent mounts. First writes are 256 MiB with an 8 MiB application
buffer at page offset 48; subsequent writes are 64 MiB with a 1 MiB buffer.
Rates include creation, write calls, final fsync and close. Content hashes,
retained guards, schema, payload reference lengths and SQLite checks all pass.
All slow samples remain, and no builds, tests, format hooks, global cache eviction
or CPU/device setting changes overlap comparison windows.

Combined 320 MiB throughput is total bytes divided by total complete time:

| Medium/state | Control, MiB/s | Candidate, MiB/s | Change | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| tmpfs/fresh | 193.74 | 324.23 | +67.4% | 6/6 |
| tmpfs/post-GC | 192.17 | 308.40 | +60.5% | 6/6 |
| ext4/fresh | 111.69 | 138.29 | +23.8% | 6/6 |
| ext4/post-GC | 86.21 | 136.16 | +57.9% | 6/6 |

The post-GC aggregate is affected by a retained control first-write sample of
about 6.04 seconds. First-256 MiB ext4 median rates change only from 152.86 to
154.56 MiB/s fresh and 150.17 to 151.88 MiB/s post-GC; this does not establish
that orphan batching fixes cold large-file writes or every storage sync tail.

For the subsequent 64 MiB write, fresh ext4 median throughput rises from 53.12
to 96.00 MiB/s and post-GC from 53.03 to 94.82. The residual complete time after
subtracting write calls, fsync and close falls from 749 to 104 ms fresh and 771
to 141 ms post-GC. This includes create/queue and loop overhead, not a pure
create syscall measurement. Write calls themselves become slower: fresh medians
267 to 444 ms and post-GC 252 to 412 ms. Thus the complete write improves while
pooled application-write p99 rises from 72.57 to 155.77 ms fresh and 68.04 to
147.62 ms post-GC.

Ten gates fail: fresh tmpfs subsequent-write p99, random writes and rmdir;
post-GC tmpfs random writes; fresh ext4 subsequent-write p99, random writes,
mkdir, rename and rmdir; and post-GC ext4 subsequent-write p99. Fresh ext4 random
write aggregate falls from 12.54 to 10.86 MiB/s. The original 95% median/aggregate
floors, p99 allowance, lifecycle limit and combined-write gates are unchanged.
The primary combined-write gates pass, but the overall decision is **REJECT**.

## Separate causal trace

Sixteen ext4 diagnostic mounts use the same two implementations, four alternating
pairs per state, the existing inclusive C++ recorder and read-only 5 ms owned-CPU
sampling. These timings explain behavior and do not decide adoption.

Fresh large-file `forget` median wall time falls from 985 to 301 ms, with nearly
equal handler CPU times. Post-GC it falls from 1,076 to 339 ms. Every corresponding
trace has exactly the same VM counts as the native probe. The median portion of
that handler extending beyond the next write's start falls from 783 to 97 ms
fresh and 869 to 134 ms post-GC. This supports the cleanup/queue attribution.

The next bottleneck is visible too: cumulative foreground publication-lock wait
during the subsequent 64 MiB phase grows from about 0.005 to 72.9 ms fresh and
0.006 to 67.2 ms post-GC. Worker payload `fdatasync` count changes from zero to
two in those phases; foreground SQL-step counts remain 2,880. The candidate's
request-weighted CPU frequency is also lower in that phase (fresh 2.00 versus
2.30 GHz); settings were not changed. During random writes, foreground SQL work
is unchanged, while worker checkpoint calls increase substantially in this
diagnostic window. These observations identify maintenance overlap worth
isolating; they do not prove the cause of every plain regression or justify
attributing it to another workload.

The next experiment should reduce payload synchronization while holding the
publication lock. A presync outside that lock may be evaluated, but the final protected
sync, WAL publication, CLEAN fence and revalidation must remain authoritative:
concurrent appends cannot be treated as durable merely because an earlier
presync completed. Small CLEAN-tail backfill must remain prompt.

## Reproduction and retention

Reuse the frozen no-hint partial-edge control as `--parent`; build and validate
an empty output directory before either measurement window:

```bash
.venv/bin/python vane_fs/benchmarks/staged_payload/orphan_batching/prepare.py \
  /tmp/vane-orphans --parent /path/to/frozen-partial-edge-control \
  --sqlite-prefix "$PWD/vane_fs/vcpkg_installed/x64-linux-release" \
  --fuse-prefix "$PWD/.cache/fuse-sdk/root/usr" \
  --test-support /path/to/test/site-packages
.venv/bin/python vane_fs/benchmarks/staged_payload/orphan_batching/validate.py \
  /tmp/vane-orphans
.venv/bin/python vane_fs/benchmarks/staged_payload/orphan_batching/prepare_diagnostic.py \
  /tmp/vane-orphan-trace --parent /tmp/vane-orphans \
  --sqlite-prefix "$PWD/vane_fs/vcpkg_installed/x64-linux-release" \
  --fuse-prefix "$PWD/.cache/fuse-sdk/root/usr"
.venv/bin/python vane_fs/benchmarks/staged_payload/orphan_batching/run.py \
  /tmp/vane-orphans --medium tmpfs
.venv/bin/python vane_fs/benchmarks/staged_payload/orphan_batching/run.py \
  /tmp/vane-orphans --medium ext4
.venv/bin/python vane_fs/benchmarks/staged_payload/partial_sync/analyze_writeback.py \
  /tmp/vane-orphans
.venv/bin/python vane_fs/benchmarks/staged_payload/orphan_batching/run_diagnostic.py \
  /tmp/vane-orphan-trace
.venv/bin/python vane_fs/benchmarks/staged_payload/request_costs/analyze.py \
  /tmp/vane-orphan-trace
.venv/bin/python vane_fs/benchmarks/staged_payload/request_costs/analyze_cpu.py \
  /tmp/vane-orphan-trace
.venv/bin/python vane_fs/benchmarks/staged_payload/checkpoint_batching/analyze_diagnostic.py \
  /tmp/vane-orphan-trace
.venv/bin/python vane_fs/benchmarks/staged_payload/orphan_batching/analyze_orphans.py \
  /tmp/vane-orphans --diagnostic /tmp/vane-orphan-trace
.venv/bin/python vane_fs/benchmarks/staged_payload/orphan_batching/publish.py \
  /tmp/vane-orphans --diagnostic /tmp/vane-orphan-trace
```

`summary.json` retains every original gate decision. Primary and diagnostic
mount arrays, per-phase maintenance records and slow overlaps are split
losslessly below 500 KiB, with reconstruction checks. Source, installed module,
native library, executable, SQLite/FUSE SDK, runner and log hashes are retained.
The reproducer reconstructs all 35 candidate inputs from the exact control and
the patch with zero fuzz. Cleanup records cover 64 stopped/unmounted workloads,
21 helper directories and the Python test directory. Owned databases, payloads
and staging files are removed after their owners stop; retained measurements and
logs remain available without keeping each workload's large generated outputs.

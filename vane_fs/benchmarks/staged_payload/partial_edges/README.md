# External payload batching across partial edges

**The candidate remains isolated.** It fixes an external-payload allocation
cliff and passes its correctness checks, but the complete physical-storage
window fails the predeclared performance gates. Production format 2, its
installed component, strict default and checkpoint settings are unchanged.

## Cause and change

The original format-1004 control already caches transaction statements and
uses selective publication. Its `WriteBytes` batch path requires the entire
replacement interval to be vacant. An 8 MiB application buffer at page offset
48 produces a first 1,048,528-byte FUSE write, seven 1 MiB writes and a final
48-byte write. The existing leading block makes later intervals occupied;
individual new blocks fall below the 256 KiB externalization threshold and
remain 4 KiB SQLite BLOBs. This does not involve the rejected UPDATE policy or
the single-block read-query candidate.

The C++ patch batches a vacant subinterval while processing occupied edges
through the existing version logic. A partial edge is included only if its
whole replacement interval is vacant, including future versions and
tombstones. This preserves 64-block groups for externalization. Allocation
starts after an automatic leading-edge rowid; pending explicit IDs flush
before an automatic trailing-edge insert. Rowid exhaustion retains its
existing fallback. Publication guards and payload-before-WAL durability
ordering are unchanged.

## Allocation evidence

A separate native probe reproduces the traced split sizes for 32 MiB, verifies
all content and counts successful `pwrite`/`pwrite64` bytes through explicit
Sync and Close. The instrumented probe is not a throughput benchmark.

| Split-write metric | Control | Candidate |
| --- | ---: | ---: |
| Live external block references | 1,020 / 8,192 | 8,160 / 8,192 |
| Live inline 4 KiB BLOBs | 7,172 | 32 |
| WAL write bytes, MiB | 36.60 | 5.00 |
| Database write bytes during this scope, MiB | 31.58 | 0 |
| Payload write bytes, MiB | 4 | 32 |

WAL syscall bytes fall by **86.3%**, and external references rise from 12.45%
to 99.61%. Aligned writes have identical allocation and syscall counts in
both variants. These are file-level syscall bytes, not physical device writes.
The smaller WAL also avoids a checkpoint during this scope; this does not
measure fully settled database write amplification.
Repeating the published counter runner reproduces both live-block counts and
WAL bytes exactly. Its control database writes rise to 34.06 MiB because the
background checkpoint's progress within this scope varies; both counter runs
are retained.

## Complete workloads

The frozen plan uses six alternating AB/BA pairs in both fresh and independently
GC-aged workspaces, separately on tmpfs and local ext4: 48 independent mounts.
The first workload writes 256 MiB using an 8 MiB buffer with page offset 48.
Each rate includes final fsync and close. Subsequent writes, reads, random
overwrites, small-file fsync and directory operations use the same prior
acceptance workload. Every content, retained-guard and schema check passes.
No owned builds or correctness tests overlap either plain comparison window.
All slow samples remain in the reports; global caches are not evicted.

First-write throughput, MiB/s; aggregate is all bytes divided by all elapsed
write time, including barriers:

| Medium/state | Control median | Candidate median | Control aggregate | Candidate aggregate | Paired wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| tmpfs/fresh | 157.38 | 352.77 | 158.54 | 353.17 | 6/6 |
| tmpfs/GC-aged | 155.00 | 356.59 | 154.96 | 359.76 | 6/6 |
| ext4/fresh | 62.48 | 156.05 | 62.42 | 68.86 | 4/6 |
| ext4/GC-aged | 63.63 | 154.47 | 63.40 | 113.10 | 5/6 |

Fresh ext4 fails the required 5/6 paired wins and its write-request p99 rises
from 0.335 to 0.921 seconds, with a maximum of 6.573 seconds. The GC-aged tmpfs
random-overwrite aggregate falls from 18.77 to 16.70 MiB/s, and the subsequent
64 MiB write's request p99 rises from 19.7 to 26.1 ms. Fresh ext4 directory
operations and GC-aged ext4 sequential reads/mkdir also miss the 95% floors.
These are conservative acceptance checks, not statistical significance tests.
End-to-end timings alone do not locate the remaining stalls or establish that
this patch causes every regression. The dedicated-host results do not justify
attributing them to competing workloads.

Next investigation should separately trace payload and WAL synchronization in
this candidate's slow writes, then isolate the GC-aged small-write cost. Keep
the reduced BLOB/WAL volume while testing any remedy against the same full
workload gates before enabling an experimental format.

## Correctness and reproduction

The installed candidate passes **154 related Python tests**, with one explicit
unsupported production-v1 migration skip. Both controls pass their 11 native
component suites; the candidate also passes all 11 under ASan/UBSan. Staging
tests now exercise 66 software crash/loss cases, including 33 new partial-edge
cases and a forced SQL spill before the first external publication. They check
original/published bytes, repeated recovery and GC; they are not physical
power-loss certification. Partial-edge tests additionally cover snapshots,
branches, sparse writes, tombstones, rowid limits and rollback injection.

Reuse the original external control built by `point_read/prepare.py` (the
`external` directory, without the equality-query patch). Then run:

```bash
.venv/bin/python vane_fs/benchmarks/staged_payload/partial_edges/prepare.py \
  /tmp/vane-external-partial --parent /tmp/vane-point-read \
  --sqlite-prefix "$PWD/vane_fs/vcpkg_installed/x64-linux-release" \
  --fuse-prefix "$PWD/.cache/fuse-sdk/root/usr" \
  --test-support /path/to/existing/test/site-packages
.venv/bin/python vane_fs/benchmarks/staged_payload/partial_edges/validate.py \
  /tmp/vane-external-partial
.venv/bin/python vane_fs/benchmarks/staged_payload/partial_edges/diagnose.py \
  /tmp/vane-external-partial
.venv/bin/python vane_fs/benchmarks/staged_payload/partial_edges/run.py \
  /tmp/vane-external-partial --medium tmpfs
.venv/bin/python vane_fs/benchmarks/staged_payload/partial_edges/run.py \
  /tmp/vane-external-partial --medium ext4
.venv/bin/python vane_fs/benchmarks/staged_payload/partial_edges/analyze.py \
  /tmp/vane-external-partial
```

The published source/runner hashes lock this host's measured configuration.
When moving the checkout, regenerate the prepared-runner hash for its new
absolute source path before starting a new comparison; retain that new plan
separately. Published raw samples are split by medium, state and variant to
keep individual files small. `analyze.py` also accepts this report directory.
Generated databases, sidecars and test temporary data are removed after their
owners stop and mounts disappear. The manifests retain code, binary, SDK,
configuration, log and measurement identities, plus cleanup records.

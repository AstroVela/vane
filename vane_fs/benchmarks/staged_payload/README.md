# Recoverable external-payload staging experiment

This is an isolated C++/Linux format-1004 prototype. Production continues to
use SQLite BLOB payloads. The previous format-1003 experiment synchronized
external bytes before every metadata commit, including NORMAL commits. This
experiment makes those commits immediately visible while allowing multiple
appends to share a payload barrier. It adds a recovery protocol; it does not
change SQLite's durability guarantees by itself.

## Protocol and scope

The prepared component retains the previous immutable external extents, 4 KiB
blocks, 256 KiB externalization threshold, inline small updates, coalesced
reads, snapshot retention and physical GC. Its application ID is `0x56465834`;
the payload header is `VANE-STAGED-1004:` followed by the workspace UUID. Other
formats refuse it. There is no in-place migration or production format change.

`<database>.stage-fence` contains two alternating 4 KiB records. Each records a
sequence, workspace UUID, CLEAN/OPEN state, WAL header, committed frame count
and final committed frame header. CRC64-ECMA covers the 152 meaningful bytes;
the unused padding separates slots. A record is written completely and passed
through `fdatasync` before the writer relies on it. Directory synchronization
protects initial sidecar creation. Torn writes can fall back to the previous
valid record. The CRC is not authentication or general bit-rot recovery.

1. **CLEAN:** every committed external reference points to synchronized bytes.
   Inline writes and namespace changes use their normal SQLite durability mode.
2. **Open a batch:** under the publication gate, capture the last committed WAL
   prefix, synchronize the WAL, then durably write OPEN. Only then append external
   bytes and commit their references in NORMAL mode. Further transactions see
   the new data immediately and share that OPEN batch.
3. **Publish:** fsync, synchronous inode writes, strict transactions, explicit
   close and managed checkpoints synchronize the whole payload file, commit a
   SQLite FULL barrier, then synchronize CLEAN. Success is acknowledged last.
   A connection's local payload-dirty flag cannot represent other writers.
4. **Cold recovery:** with no managed connection alive, OPEN truncates the WAL
   to its recorded committed prefix before SQLite opens/replays it. If SQLite
   has restarted the WAL generation, that previous generation was already
   backfilled and the unpublished new generation is discarded. Recovery checks
   the retained frame boundary and synchronizes the truncation before writing
   CLEAN. Unpublished namespace changes roll back with their file references.
5. **Checkpoint:** publish OPEN before copying any of its metadata into the
   database. Disable automatic checkpoints and last-close checkpoints on all
   managed SQLite connections. The worker's PASSIVE checkpoint holds a separate
   checkpoint gate, releasing the publication gate after publication. New OPEN
   batches wait; FULL barriers can proceed while a PASSIVE checkpoint is paused.
   RESTART admission keeps writers excluded until WAL reuse is established.

A separate observer connection obtains the committed frame count through
`SQLITE_CHECKPOINT_NOOP`: that API returns `SQLITE_LOCKED` on the connection
already running the write transaction. Physical WAL length is not a committed
prefix: it can include rolled-back, spilled or reused frames. Tests exercise
spill before the first external append, abort/retry and WAL reuse.

A valid marker read from page cache is not evidence that another connection's
sync succeeded. Each coordinator synchronizes an unfamiliar sequence before
trusting either OPEN or CLEAN. Tests inject failed OPEN and CLEAN syncs and
verify that the next operation/connection cannot silently trust that record.

The protocol uses only ordinary files, `flock`, public SQLite APIs and the
documented WAL layout; it does not modify SQLite or its shared-memory index.
The implementation was checked against the official SQLite 3.53.2 amalgamation,
the [WAL documentation](https://www.sqlite.org/wal.html),
[WAL format](https://www.sqlite.org/walformat.html) and
[checkpoint API](https://www.sqlite.org/c3ref/wal_checkpoint_v2.html).

All writers, checkpointers and GC must use this native component. Raw SQLite
writers/checkpoints are outside the supported protocol; test inspectors make
specific controlled changes only to construct fault fixtures. A raw SQLite
reader also does not hold the external-payload GC lease. `.stage-init` serializes
constructor initialization; `.stage-live` prevents cold recovery while native
connections remain alive. `.stage-publish` and `.stage-checkpoint` coordinate
publication/checkpoint operations across connections and processes. Explicit
close releases its liveness lease only after successful publication and SQLite
close. Inherited handles must be reopened after fork, as in production.

This remains a local-filesystem experiment. It has no S3/DuckDB filesystem
adapter, online backup protocol or automatic cleanup of interrupted *initial*
workspace creation. A failed initial creation may leave sidecars which are
refused rather than overwritten. All database, WAL and sidecar files belong to
one workspace; copying or deleting them individually is not supported. The
extra publication locks introduce retryable contention between connections.
Do not use it for existing or important workspaces.

The `batched` candidate applies the previous experiment's checkpoint trigger
to this recovery protocol: a new PASSIVE checkpoint starts after 16 MiB of
*uncopied* WAL, instead of repeatedly checkpointing small additions after total
WAL length crosses the threshold. Already-started checkpoints continue retrying
on incomplete progress or storage errors. FULL barriers and the 64 MiB RESTART
admission rule are unchanged. `staged` retains the original trigger, making
this cost independently measurable.

## Reproduce

From the repository root, prepare a new directory:

```bash
.venv/bin/python vane_fs/benchmarks/staged_payload/prepare.py /tmp/vane-stage-experiment
cmake -S /tmp/vane-stage-experiment/staged -B /tmp/vane-stage-experiment/staged/build \
  -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/vane_fs/vcpkg_installed/x64-linux-release" \
  -DVANE_FS_BUILD_PYTHON=OFF -DVANE_FS_BUILD_TESTS=ON -DVANE_FS_BUILD_TOOLS=ON
cmake --build /tmp/vane-stage-experiment/staged/build --parallel 2
ctest --test-dir /tmp/vane-stage-experiment/staged/build --output-on-failure
```

`prepare.py` checks/reuses the previous experiment's frozen source inputs and
prepares baseline, immediate-sync external, staged and batched copies. The patches only
apply to these copies. `staging.hpp` implements the protocol; `storage.patch`
wires it into transactions, close and checkpoint paths. `fixtures.patch`
adapts the existing related tests and registers the additional native tests.
`batching.patch` adds the alternative trigger and its regression test.
Python wheels were built non-editably in persistent per-candidate build trees
and installed into isolated environments. Exact commands and dependency paths
are retained in the local build records.

## Validation

Both staged candidates pass 144 Python tests each, with the production-v1
migration test explicitly skipped for this isolated format. Seven native tests
pass for `staged`, and eight for `batched`, in Release and under
AddressSanitizer/UndefinedBehaviorSanitizer. The checkpoint-trigger regression
fails against `staged` and passes against `batched`.
The staging native test additionally passes 33 deterministic crash/loss cases
in both configurations, including a final forced-spill recovery case.

Coverage includes eight publication boundaries, process kill and a loss model
that retains all latest SQLite bytes but only successfully synchronized payload
and marker images, with and without forced WAL reuse. It also covers
cross-connection visibility and barriers, a surviving connection during another
writer's crash, synchronous handles, close failure/retry, partial marker writes,
failed marker/data synchronization, publication-lock timeout, GC and repeated
recovery. Missing, truncated, mismatched or symlinked marker files, and files
with no valid checksum record, are refused without changing database, WAL or
payload bytes.

The existing checkpoint suite still covers progress, pinned readers and bounded
admission, concurrent writers, storage errors/retry, slow checkpoint I/O, fork
and a paused checkpoint during foreground fsync. Its budget test now retries
publication-gate contention *below* the WAL budget while checking rejected-write
atomicity; a transient lock timeout is not evidence that the WAL limit was hit.
The durable-image fixture captures direct sidecar syncs separately from SQLite
VFS syncs because CLEAN follows the FULL WAL commit.

These are deterministic software fault tests and sanitizer checks, not physical
power-cut certification.

## Measurements

The production source remains at `7e554c80a6`, with
`src/workspace.cpp` SHA256
`cae97ec42ce136c902d6fda9b9d1c1cd9dc60844c1c6c993be11367d464fc02b`.
The production installed module also remains unchanged. Tests and measurements
use isolated binaries; [results.json](results.json) records all per-run scalar
measurements and summaries, and [artifact-manifest.json](artifact-manifest.json)
identifies full request traces, sources/binaries, build commands, logs, host
counters and cleanup records retained locally.

There are three serial measurement windows on the same Linux/ext4/SATA-SSD
host, using SQLite 3.53.2, Release builds and fsync mode:

- `initial`: baseline / immediate-sync external / staged; six order permutations
  for FUSE, three cyclic orders for native and aged workloads.
- `followup`: baseline / staged / batched, with the same orders and workloads.
- `cold_256`: baseline / batched, four alternating paired orders; a new workspace
  directly receives one 256 MiB file, without the preceding 64 MiB workload.

Each six-order FUSE sequence writes 64 MiB, writes 256 MiB, creates/warms a
64 MiB read fixture, then performs sequential reads, 4 KiB random reads and
4 KiB overwrites. The 256 MiB item is a new file but uses a workspace that has
already processed the 64 MiB item. Writes include fsync and close. Application
writes are 1 MiB; the native probe also reproduces their 1,048,528 + 48 byte
FUSE request split. Aged workspaces retain a 32 MiB guard with alternate blocks
removed and run GC before measuring reads, small files and directory changes.

The original staging trigger does not improve large FUSE writes: initial-window
medians are 95.23 / 84.66 / 64.09 MiB/s for baseline / external / staged. The
batched candidate has a repeatable large-write gain in the follow-up window,
but substantial regressions elsewhere:

| FUSE workload | Baseline | Staged | Batched |
| --- | ---: | ---: | ---: |
| 256 MiB write after the 64 MiB item, MiB/s | 95.99 | 63.73 | 150.20 |
| Warm 64 MiB sequential read, MiB/s | 524.40 | 750.93 | 719.79 |
| Warm 4 KiB random read, MiB/s | 68.51 | 58.98 | 60.71 |
| 4 KiB overwrite + final fsync, MiB/s | 13.03 | 9.55 | 8.94 |
| Aged 4 KiB file + fsync, files/s | 267.00 | 238.99 | 254.26 |
| Aged mkdir + final directory fsync, ops/s | 1,974.01 | 1,845.19 | 793.06 |
| Aged rename + final directory fsync, ops/s | 2,365.83 | 2,188.74 | 1,353.57 |
| Aged rmdir + final directory fsync, ops/s | 2,429.38 | 2,149.22 | 1,572.23 |

The standalone fresh-workspace 256 MiB check also improves its median by 59.5%:

| Candidate | All four samples, MiB/s | Median |
| --- | --- | ---: |
| Baseline | 94.00, 93.10, 93.89, 18.70 | 93.50 |
| Batched | 150.65, 19.39, 153.68, 147.55 | 149.10 |

These are not tail-latency guarantees. Both candidates have a severe stall in
the fresh-workspace check. The first 64 MiB item is particularly unstable:
follow-up medians are 39.12 / 83.19 / 18.46 MiB/s, with baseline ranging
9.63–107.51 and batched ranging 16.15–19.15. The initial window's baseline
64 MiB median was 96.86 MiB/s. No sample is discarded or substituted.

Native I/O counts explain the large-write mechanism. For 256 MiB of application
data, total software write volume including setup/readback/close falls from
646.57 MiB (2.526×) to 324.34 MiB (1.267×). Counts are medians over three runs:

| Synchronization layer/file | Baseline | Staged | Batched |
| --- | ---: | ---: | ---: |
| SQLite database VFS sync | 7 | 33 | 4 |
| SQLite WAL VFS sync | 25 | 68 | 10 |
| Payload `fdatasync` | 0 | 33 | 5 |
| Recovery marker `fdatasync` | 0 | 65 | 9 |
| Direct WAL prefix `fdatasync` | 0 | 32 | 4 |

The immediate-sync external candidate performs 257 payload syncs, including
initialization. Batched performs five, also including initialization: four
data-publication barriers replace 256. Marker and direct-WAL synchronization
are additional work, not free barriers. The native probe counts SQLite VFS and
direct I/O separately to avoid double-counting the same call; it does not count
directory syncs or claim to measure physical SSD write amplification.

Native timings still expose long tails: batched write+Sync takes
7.881 / 1.464 / 6.419 seconds, compared with baseline
2.628 / 2.578 / 2.561 seconds. Payload sync alone accounts for 5.281 and
5.029 seconds in the two slow batched native runs. This identifies the software
wait location; it does not establish the kernel/device cause or prove that
batching caused each stall. FUSE timings use ordinary uninstrumented binaries.

All windows use host monitoring and a 30-second pre-measurement baseline.
No owned builds or correctness tests overlap measurement, and no global cache
drops are used. Across whole windows, active CPU medians are approximately
5.2–6.3% over 36 logical CPUs and available memory stays above 54.7 GiB;
iowait peaks around 12–13%. These counters include untimed cleanup and other
host activity, so they cannot attribute device traffic to one candidate.

Validation retains 152 recorded FUSE hash results, 221,184 random-slice checks,
4.5 GiB of native readback, 206 SQLite quick checks, and further sparse/snapshot
content checks. All 98 benchmark workspaces are removed after their processes
and mounts stop, outside measured workloads. Source/binary identities, raw
samples, traces, necessary logs and cleanup manifests remain.

**Decision: keep both candidates experimental.** Batch publication establishes
a useful large-file result and passes the stated recovery tests, but the
batched trigger loses 31% on fresh random overwrites and 35–60% on aged directory
operations. The original trigger avoids most directory regression but loses
large-write throughput. Neither replaces the production format or policy.
Before adoption, investigate the sync tails and whether batching can be limited
to pending external data while promptly draining WAL for subsequent small and
metadata operations. That is a follow-up hypothesis, not a tested fix.

# Conditional checkpoint and publication experiment

These isolated C++ candidates refine the format-1004
[staging experiment](../README.md). Production still stores payloads in SQLite
BLOBs. Neither candidate changes the production binary, default durability,
on-disk format, or installed Vane package.

## Evidence and changes

The preceding batched policy deferred every new checkpoint until 16 MiB of WAL
remained uncopied. This also deferred maintenance after publication was CLEAN,
when only inline or namespace transactions followed a large write. A pinned-WAL
test reproduces that scheduling behavior independently of storage timing.

The **conditional** candidate coalesces new checkpoints only while external
publication is OPEN. After CLEAN, the original committed-WAL wake policy applies.
The state check and checkpoint lease acquisition share the publication gate.
Already-started work still retries, and the 64 MiB restart admission rule and
FULL barriers are unchanged. This fixes scheduling; it does not guarantee that
each workload becomes faster.

Separate native instrumentation found another cost: 512 inline random writes
acquired the publication gate and reloaded its fence 512 times, spending about
10–12 ms in fence loading. The **selective** candidate additionally lets inline
NORMAL transactions proceed without that gate or fence reload. SQLite's writer
lock serializes them with the WAL-prefix capture and FULL transactions. Their
payload GC lease remains active. Transactions that might append external data,
strict transactions, synchronous writes and barriers still acquire publication
before their SQL writer lock.

Classification is conservative and happens before BEGIN IMMEDIATE. It includes
both padded boundary blocks: an unaligned 253,954-byte request starting at byte
4,095 can produce 64 external blocks, despite containing less than 256 KiB of
application data. Taking publication only after discovering those blocks inside
the SQL transaction would invert the checkpoint lock order. The boundary test
checks rejection before mutation while the gate is held.

The final instrumented runs record zero foreground publication acquisitions or
fence loads during each 512-operation selective random-write phase. Admission
or a FULL barrier can still require publication; this is not a lock-free writer.
The protocol still requires managed native writers, checkpoints and GC. It has
no production migration, remote backend, or online-backup protocol.

## Local FUSE measurements

Two serial comparison windows retain every sample. The first compares production,
the previous unconditional batching policy, and conditional checkpoints. The
second compares production, conditional checkpoints, and selective publication.
Each has six balanced fresh-workspace orders and three rotating aged-workspace
orders. No owned builds or correctness tests overlap the timers, and no global
caches are evicted. Sources, binaries, host counters and runner hashes are retained.

The workload sizes and preparation match the preceding experiment: 64 MiB then
256 MiB writes, a warmed 64 MiB read fixture, 4,096 random reads, 512 aligned 4 KiB
overwrites, and 128 small-file or directory operations. Writes include fsync;
large writes also include open and close. Aged workspaces retain a 32 MiB guard
with alternate blocks removed, followed by GC. This is Linux/ext4 on the shared
SATA SSD, with 4 KiB logical blocks, Release builds and fsync durability.

Second-window medians:

| Workload | Production | Conditional | Selective |
| --- | ---: | ---: | ---: |
| First 64 MiB write, MiB/s | 54.41 | 146.90 | 84.17 |
| Subsequent 256 MiB write, MiB/s | 93.59 | 144.88 | 142.02 |
| Warm sequential read, MiB/s | 525.41 | 747.06 | 725.67 |
| Warm 4 KiB random read, MiB/s | 67.08 | 60.36 | 59.28 |
| 4 KiB overwrite + final fsync, MiB/s | 13.59 | 11.01 | 11.15 |
| Aged random overwrite + final fsync, MiB/s | 14.86 | 1.86 | 13.04 |
| Aged 4 KiB file + fsync, files/s | 258.18 | 30.48 | 254.53 |
| Aged mkdir + directory fsync, ops/s | 1,847.12 | 1,750.78 | 1,840.20 |
| Aged rename + directory fsync, ops/s | 2,036.03 | 1,998.12 | 1,958.23 |
| Aged rmdir + directory fsync, ops/s | 2,293.84 | 2,075.80 | 2,270.73 |

Selective publication reaches about 52% higher 256 MiB write throughput than
production in this window. Small-file fsync and directory medians are within
about 0–4% of production. However, random writes remain about 12–18% lower and
random reads about 12–13% lower. Conditional's aged overwrite and small-file
medians contain severe stalls; their ratios are not stable selective-publication
speedups. Instrumentation establishes removal of the per-inline publication
work, not attribution of every timing difference to that work.

The first 64 MiB writes remain particularly variable. Even the unchanged
conditional binary has a 256 MiB median of 90.31 MiB/s in the first window and
144.88 in the second. Three of its first-window large writes stall. These windows
must not be merged, filtered, or substituted to manufacture an improvement.
The JSON report retains the full ranges and all scalar samples.

## Synchronization tails

A separate diagnostic samples only owned native threads every 10 ms. During
fdatasync on the payload, WAL and database files, it observes
`rq_qos_wait`, `folio_wait_bit_common` and `jbd2_log_wait_commit`. One conditional
run has 314 payload samples at `rq_qos_wait`; production WAL/database syncs also
wait in these kernel paths. The Linux
[request-QoS implementation](https://github.com/torvalds/linux/blob/v6.11/block/blk-rq-qos.c)
waits for an in-flight token in that function.

These are sampled wait locations, not exact accumulated syscall durations.
Full kernel stacks were unavailable to the sampler. The evidence does not
identify the device-level cause, attribute contention to another job, or show
that either policy has eliminated synchronization tails. Diagnostic timing is
kept separate from uninstrumented FUSE throughput.

## Validation and decision

The final selective component passes 144 Python tests, with one explicit skip
for unsupported production-v1 migration, and all nine native tests in Release
and ASan/UBSan. Existing publication fault/recovery, external data/GC, paused
checkpoint, fork, snapshot and FUSE cases remain included. The staging test
retains its 33 deterministic crash/loss cases; these are software fault tests,
not physical power-cut certification.

The new checkpoint test fails under both preceding policies: unbatched staging
restarts work during OPEN, while unconditional batching fails to resume CLEAN
maintenance. It also checks reader release, retry without another foreground
commit, and WAL reuse. A corrected test fixture keeps the original WAL generation
pinned until admission is reached; the initial fixture failure is retained.

The inline-publication test fails on the conditional-only candidate. It checks
inline progress while publication is held, OPEN-state inline overwrites and
namespace changes, gating of FULL/O_SYNC/external operations, unaligned boundary
writes, concurrent connections and reopening after publication. Prepared sources
match the tested components byte for byte.

**Both candidates remain experimental.** Directory and small-file regressions
are substantially reduced, but random I/O and synchronization tails still fail
the adoption criteria. Production keeps its existing SQLite payload format and
checkpoint policy.

## Reproduce

From the repository root, prepare a new output directory:

```bash
.venv/bin/python vane_fs/benchmarks/staged_payload/conditional/prepare.py /tmp/vane-conditional
cmake -S /tmp/vane-conditional/selective -B /tmp/vane-conditional/selective/build \
  -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/vane_fs/vcpkg_installed/x64-linux-release" \
  -DVANE_FS_BUILD_PYTHON=OFF -DVANE_FS_BUILD_TESTS=ON
cmake --build /tmp/vane-conditional/selective/build --parallel 2
ctest --test-dir /tmp/vane-conditional/selective/build --output-on-failure
```

`prepare.py` retains the parent experiment's source-identity checks and applies
`checkpoint.patch`, then `inline.patch` in a separate copy. Component wheels are
built non-editably with persistent per-candidate build directories and installed
into isolated environments. The production installation is unchanged.

## Same-window local reference comparison

A final, separate six-permutation window compares production, selective staging
and Drive9 `3334e43`, using independent TiDB 8.5.6 unistore and MinIO services on
loopback. Drive9 uses `--durability=fsync --profile=none
--trust-process-local-events --no-auto-unpack`, with the same 1% cache free-space
guard as the prior local runs. Each round creates a new directory in the same
mounted workspace; backing metadata and retained versions carry over. There is
no intervening Vane GC or global cache eviction. This window has different state
and cache protocols from the fresh/aged workspace matrices above.

Read cache paths are specified before measurement and verified by a separate
FUSE-debug run:

- **Writable handle:** open with O_RDWR, perform reads only, run one complete
  warm pass, then measure through that same handle. Drive9 returns FUSE DIRECT;
  all three targets issue 129 FUSE READ requests for the 64 MiB sequential pass
  (including EOF) and 4,096 for the random pass.
- **Settled readonly handle:** wait at least three seconds after file fsync,
  reopen O_RDONLY, then run two complete warm passes. Drive9 returns KEEP_CACHE
  and both measured passes issue zero FUSE READ requests. Vane's live mounts
  continue to issue 129 and 4,096 requests respectively.

Both protocols use warm data; FUSE DIRECT is not an O_DIRECT disk benchmark.
An earlier diagnostic showed that immediately reopening readonly after fsync
already used KEEP_CACHE, so elapsed time alone did not identify the read path.
That original diagnostic and the revised protocol logs are both retained.
Debug logging is disabled for the throughput runs. Random reads use the same
4,096 deterministic, potentially unaligned offsets on every target.

Medians over all six orders:

| Workload | Production | Selective prototype | Drive9 local |
| --- | ---: | ---: | ---: |
| 64 MiB write + fsync/close, MiB/s | 110.31 | 86.88 | 83.29 |
| 256 MiB write + fsync/close, MiB/s | 88.70 | 121.81 | 108.27 |
| Warm sequential read through writable handle, MiB/s | 526.83 | 784.72 | 218.71 |
| Warm 4 KiB random read through writable handle, MiB/s | 62.04 | 60.88 | 0.35 |
| Warm readonly sequential read, MiB/s | 526.07 | 751.68 | 4,906.71 |
| Warm readonly 4 KiB random read, MiB/s | 61.73 | 59.03 | 1,208.27 |
| 4 KiB overwrite + final fsync, MiB/s | 1.52 | 12.44 | 1.81 |
| 4 KiB file + fsync, files/s | 78.33 | 252.05 | 23.38 |
| mkdir + directory fsync, ops/s | 1,220.92 | 1,264.83 | 70.18 |
| rename + directory fsync, ops/s | 1,344.03 | 1,303.61 | 30.79 |
| rmdir + directory fsync, ops/s | 1,324.72 | 1,285.91 | 48.41 |

The readonly Drive9 results measure kernel page-cache hits. They do not imply
that MinIO serves those bytes at the reported rate. The writable-handle case
also selects Drive9's writable-handle read/prefetch policy; it is not a universal
ranking of all read modes. The workloads cover this storage follow-up, without
rerunning stat or Git macrobenchmarks.

Local SQLite FULL barriers and remote upload/metadata commits have different
persistence boundaries. Drive9's remote directory fsync returns success without
an additional barrier; its metadata API operations have their own commit path.
Content verification through the remote API follows drain outside each timer.
These results apply to this local development deployment, not distributed TiKV
or a production S3 service.

Production random-overwrite and small-file results in this long-lived window
include repeated stalls. They differ substantially from the fresh/aged matrix,
which still shows a selective-prototype random-I/O regression. Both windows and
every slow sample are retained; this reference comparison does not override the
decision to keep the prototype experimental.

## Retained evidence and cleanup

[results.json](results.json) retains both Vane-only windows, every scalar sample,
latency summaries and native/kernel diagnostics. [reference-results.json](reference-results.json)
retains the final comparison, explicit read protocols, all samples and validation
counts. [artifact-manifest.json](artifact-manifest.json) identifies full request
traces, source/binary/wheel hashes, runners, build/test commands, host monitoring,
logs and cleanup records retained locally.

The two Vane matrices validate 144 recorded content hashes and
221,184 random-read slices. They and the native diagnostics record
102 successful SQLite quick checks. All 54 matrix workspaces and 30
instrumented workspaces are removed after their owners stop. The final comparison
retains 198 scalar results and verifies 147,456 random-read slices,
2,304 small files, and 24 remote content results. Its two Vane databases
pass quick_check after clean unmount. Both cache-diagnostic deployments, the
comparison's services, mounts, caches and generated database/payload data are
removed outside measurement; final absence checks and removed paths are recorded.
No owned native builds or correctness tests overlap formal throughput windows.

The next adoption gate remains consistent 4 KiB read/write performance across
fresh and aged workspaces, alongside the existing recovery and synchronization
checks. The current large-write gain alone is insufficient to change the default.

The [transaction-control cache follow-up](../random_io/README.md) separates native
probe overhead from real read-path work, tests cached BEGIN/COMMIT statements,
and repeats random-I/O and synchronization measurements without changing the
production package.

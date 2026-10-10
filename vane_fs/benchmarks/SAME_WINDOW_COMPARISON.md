# Local same-window comparison: 2026-10-05 UTC

This experiment remeasures VaneFS and Drive9 in three alternating pairs on the
same host, using one Python process and the unchanged
[Drive9 workload functions](https://github.com/mem9-ai/drive9/blob/3334e43d47e00d1411c32defe837a87a969d1e56/tools/bench/drive9_fuse_bench.py).
It covers the remaining sequential-write, hot-stat and Git/macro costs; it
does not rerun every workload from earlier comparisons.

VaneFS uses the source at `9ee791a625` plus the
[4,096-page checkpoint change](WRITE_OPTIMIZATION.md), with its frozen source
and binary hashes recorded. Drive9 is pinned to
`3334e43d47e00d1411c32defe837a87a969d1e56`, with an independent local TiDB 8.5.6
`unistore` instance and real MinIO on loopback. Container images and executable
hashes are retained. This is a local development deployment, not distributed
TiKV or a production-scale storage comparison.

## Configuration

VaneFS retains FULL commits per mutation, direct file I/O, no kernel writeback
and a 60-second positive entry/attribute cache. Drive9 uses `--durability=fsync`,
`--profile=none`, `--trust-process-local-events` and `--no-auto-unpack` with its
read cache, 60-second attributes/entries and 10-second directory cache.
The process-local event setting applies to this isolated single-server setup.
The cache free-space floor is 1%; the runner checks for at least 2 GiB free
before samples on the shared partition.

Sequential writes include fsync. Drive9 waits for remote commit at fsync, while
ordinary writes/closes can finish before upload; VaneFS acknowledges each
mutation after its FULL commit. Git macros do not add fsync. These differing
durability and cache boundaries remain relevant even when timing in the same
window. Both mounts use the same fixture sizes and path depth, with fresh
workload directories for each sample. Preparation, validation, upload-queue
drains and cleanup are outside timers. No agent-owned tests/builds overlap;
unrelated jobs remain active on the shared host.

## Three-run medians

| Workload | VaneFS | Drive9 |
| --- | ---: | ---: |
| 64 MiB sequential write, MiB/s | 45.91 | 87.97 |
| Warm stat over 1,000 files, operations/s | 110,304.89 | 143,680.16 |
| Git clone, seconds | 1.656 | 3.135 |
| Warm Git status, seconds | 0.0654 | 0.1771 |
| Warm Git diff, seconds | 0.0164 | 0.0539 |
| Warm find, seconds | 0.01210 | 0.00965 |
| Warm Go build, seconds | 0.1077 | 0.1688 |

Git clone uses `--no-local` with 100 payload files totaling approximately
4 MiB. The complete Git repository includes metadata and a Go fixture.

Sequential write remains the largest gap: VaneFS reaches 52.2% of Drive9's
throughput. Hot stat reaches 76.8%. Find takes about 2.4 milliseconds longer
(25.3%). The other medians favor VaneFS in this window, but Git diff is noisy:
VaneFS takes 0.0074/0.0164/0.0439 seconds and Drive9 takes
0.0539/0.0067/0.0556 seconds. These overlapping ranges do not support a stable
ranking for diff. VaneFS clone also varies from 1.590 to 3.262 seconds. Three
samples on a shared host do not establish tail-latency bounds or universal
performance rankings.

## Validation and cleanup

All six 64 MiB files passed size/hash checks, including remote API reads after
draining Drive9 writes. Stat fixtures passed 6,000 exact-name/content checks.
All six Git fixtures passed fsck, the expected dirty-file check and 600 payload
content checks; Drive9's remote Git HEAD matched each local clone. The VaneFS
database passed its SQLite quick check after unmounting.

All owned mount/server processes exited with status zero; both containers and
mounts were removed. Drive9's unmount CLI returned status one, but the mount
was already gone at the fallback check; that command error is retained. Cleanup
removed 849,558,666 bytes of generated data after stopping its owners. Existing
user services were left running, and no global caches were evicted.

The [JSON artifact](same-window-20261005.json) retains all 42 timing samples,
per-sample host load, frozen identities, configurations, validation and cleanup
manifests. Runners, binaries and redacted logs remain in
`vane_fs/build/drive9-comparison/same-window-20261005/`. Historical measurements
from a different window are not substituted for either side of this table.

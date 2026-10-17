# Single-block equality-query evaluation

**This candidate is not enabled.** The query uses `block = ?` when a read,
after EOF clipping, lies within one logical block. Multi-block reads keep the
original range query. Both use the same LEFT JOIN, sparse zero filling,
payload validation and duplicate-visible-version checks. Production sources,
durability and checkpoint policy are unchanged.

The isolated production and format-1004 external pairs start from
`4e83455308faea9d5f6fc45200df20a608ec6f25`. Only `ReadBytes` changes in each
candidate. The external control retains selective publication and cached
transaction-control statements; it does not include the rejected UPDATE policy.

## Diagnosis and results

SQLite's measured plan for the range query constrains `(inode, block range)`.
With equality on block, it can also constrain `low` in the primary key. A
synthetic SQL-only control records about 36 versus 32 VM instructions per
ordinary single-block lookup, and 126 versus 92 with 16 retained intervals.
Its six balanced orders show median query times of 57.10 versus 56.21 ms
without history, and 134.15 versus 121.51 ms with history. These are query
microbenchmarks, not filesystem throughput.

The plain native matrix uses six balanced orders per fresh, GC-aged and
history state on tmpfs: 72 independent databases. It consumes each returned
buffer immediately and measures foreground thread CPU. The history case
retains 16 versions of the first MiB; old-snapshot reads address that MiB.
Ordinary live single-block CPU changes by roughly 0–1%. Old-snapshot CPU
medians improve by 9.4% for BLOB storage and 6.3% for external storage, below
the frozen 10% requirement. Cross-block controls retain the range query.

A separate ext4/FUSE window uses six balanced orders in fresh and GC-aged
states: 48 independent mounts. Each prepares a 64 MiB fixture with a fixed
buffer offset of 48 bytes, opens O_RDWR, warms through the same handle and
measures sequential, aligned 4 KiB, within-block 128-byte and crossing 4 KiB
reads. Every returned slice is checked. No owned tests or builds overlap either
plain measurement window; global caches are not evicted.

Aligned random-read medians, MiB/s:

| State | BLOB control | BLOB equality | External control | External equality |
| --- | ---: | ---: | ---: | ---: |
| Fresh | 61.59 | 62.65 | 63.00 | 63.10 |
| After GC | 61.76 | 63.60 | 67.61 | 63.04 |

The full gates reject both pairs. Besides the insufficient old-snapshot CPU
gain, GC-aged sequential aggregate throughput falls from 391.99 to 357.88 MiB/s
for BLOBs and 519.28 to 485.77 for external storage. The external aged random
read also falls below the 95% median and aggregate floors. Sequential SQL is
unchanged; this window does not identify the cause of those timing differences.
All samples remain in the reports. These gates are conservative acceptance
checks, not a statistical significance test.

## Validation and reproduction

The BLOB candidate passes 136 related Python tests. The external candidate
passes 148, with one explicit skip for unsupported production-v1 migration.
Both pairs pass their native component suites. A bounded-VM regression fails
under the preceding query and passes with equality, while checking snapshots,
holes, clipped EOF, inherited partial writes and pinned readers after GC.
New Python cases verify missing payloads, overlapping visible versions and
query reset after repeated errors for one-block reads.

One external test initially lacks its production-interpreter fixture variable.
Its other 147 cases pass; only that fixture case is rerun after supplying the
variable. The native measurement initially lacks a required inode mount lease
and fails before any read timer. The corrected preparation releases the lease
while creating history, then reopens the same inode before timing. Initial
failures and their repairs remain in the local artifact archive.

```bash
.venv/bin/python vane_fs/benchmarks/point_read/prepare.py /tmp/vane-point-read \
  --sqlite-prefix "$PWD/vane_fs/vcpkg_installed/x64-linux-release" \
  --fuse-prefix "$PWD/.cache/fuse-sdk/root/usr" \
  --test-support /path/to/existing/test/site-packages
.venv/bin/python vane_fs/benchmarks/point_read/validate.py /tmp/vane-point-read
# Copy the published plan.json into the fresh artifact directory before runs.
.venv/bin/python vane_fs/benchmarks/point_read/run.py /tmp/vane-point-read --kind native
.venv/bin/python vane_fs/benchmarks/point_read/run.py /tmp/vane-point-read --kind fuse
.venv/bin/python vane_fs/benchmarks/point_read/analyze.py /tmp/vane-point-read
```

Generated databases, payload sidecars and test temporary directories are removed
after owned processes exit and mounts disappear. Retained artifacts contain raw
samples, configuration, cleanup records, source/binary identities and logs.
The source report publishes scalar samples and summaries. All per-request
latency vectors remain in the local archive, indexed by its artifact manifest;
recomputing the FUSE summary requires those vectors or a fresh run.

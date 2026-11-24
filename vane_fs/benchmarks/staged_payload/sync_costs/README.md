# Sync attribution and bounded-batch hypotheses

This investigation changes diagnostic copies and benchmark tools only. It
compares the preceding no-hint orphan-batching candidate with the rejected
whole-file presync candidate. Removing added call-site tags produces
byte-identical runtime sources from the preceding diagnostic build, except
for the recorder header. The default component and installed module remain
unchanged. No new policy has passed adoption gates or been enabled.

## What stalls

Sixteen ext4 mounts use four alternating pairs per fresh/post-GC state and
the unchanged 256 MiB + 64 MiB diagnostic workload. SQLite 3.53.2 still uses
its original `fsync`. No cache, frequency, affinity or kernel settings change.
Every content check and `quick_check` passes.

The recorder resolves each current descriptor through `/proc/self/fd`, captures
file size, call site, syscall result/errno, wall and thread CPU time, and links
the call to its original request. The read-only classification probe costs a
median 22.1 µs, p99 40.5 µs, maximum 107.5 µs; these costs are separate from
the syscall times below. Per-request sync counts and nanoseconds reconcile
exactly with individual calls.

| File | Calls | Median sync, ms | Maximum, ms | Calls ≥100 ms | Calls ≥1 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| payload | 120 | 133.622 | 4,500.007 | 67 | 1 |
| WAL | 5,690 | 2.402 | 1,425.686 | 29 | 4 |
| fence | 208 | 0.878 | 78.322 | 0 | 0 |
| SQLite database | 370 | 2.606 | 945.860 | 8 | 0 |
| directory | 56 | 0.002 | 1.781 | 0 | 0 |
| other | 16 | 2.229 | 6.645 | 0 | 0 |

All **6,460 calls**, including startup/shutdown, and all **104 calls lasting
at least 100 ms** are retained. Overlapping call times are not additive.

Two presync-variant random-write phases show the same pattern:

- Fresh rep 1: worker WAL sync lasts 1.290 s; foreground final-barrier WAL
  sync lasts 1.214 s, overlapping almost its entire duration.
- Post-GC rep 0: worker WAL sync lasts 1.426 s; foreground WAL sync lasts
  1.348 s, again overlapping almost its entire duration.

Their sync thread CPU times are only 2.5–5.5 ms. Subsequent small-file phases
also contain database checkpoint syncs up to 946 ms overlapping foreground
WAL syncs. Payload batching alone cannot remove these WAL/database stalls.
The fence has no second-scale stall in this window.

Post-GC control rep 3 additionally has a **4.500 s protected payload sync**
with 15.3 ms thread CPU, at a 436,203,520-byte file size. It occurs during file
preparation outside recorded workload phases. It is retained, and is not
included in measured sequential-phase rates.

Final payload sync medians for first-256 increase from 134.5 to 175.1 ms fresh
and 135.4 to 178.0 ms post-GC. For subsequent-64 they decrease from 109.4 to
55.1 ms fresh and 94.9 to 51.4 ms post-GC, while worker cumulative payload
sync medians increase from 55.2 to 99.3 ms and 58.3 to 102.1 ms. These are
inclusive diagnostic observations, not adoption throughput.

## Is bounding payload work useful?

A separate native C++ probe writes 320 MiB using the same 8 MiB / 1 MiB
buffers at page offset 48. Six balanced orders per state compare a growing
file synced every 128 MiB, that file synced every 16 MiB, and independently
created/sealed 16 MiB files. Every intermediate sync completes before further
writes. Complete timings include payload syncs, create/close, one final 4 KiB
marker sync and directory fsync.

The reclaimed-layout state first writes 192 MiB and reclaims alternate 16 MiB
ranges, retaining and checking a 96 MiB guard. Single-file variants punch
holes; the sealed variant removes old segments. This is a raw layout
hypothesis, **not VaneFS GC, SQLite recovery, concurrent publication or an
adoption benchmark**.

| State | Variant | Median MiB/s | Aggregate MiB/s | Sync p99, ms | Maximum, ms |
| --- | --- | ---: | ---: | ---: | ---: |
| fresh | growing, 128 MiB | 318.68 | 190.24 | 4,331.35 | 4,331.35 |
| fresh | growing, 16 MiB | 268.87 | 268.91 | 68.04 | 68.58 |
| fresh | sealed, 16 MiB | 274.53 | 274.09 | 67.16 | 68.17 |
| reclaimed layout | growing, 128 MiB | 310.96 | 301.41 | 548.80 | 548.80 |
| reclaimed layout | growing, 16 MiB | 266.28 | 266.02 | 67.72 | 67.77 |
| reclaimed layout | sealed, 16 MiB | 271.49 | 271.08 | 67.36 | 67.43 |

Fresh growing-128 rep 4 retains a **4.331 s `fdatasync` of 128 MiB dirty data**,
reducing its complete run to 63.54 MiB/s. A plain native file reproduces this
stall without SQLite or FUSE. These measurements establish storage sync
waiting, but do not identify a particular kernel or device mechanism.

Smaller batches have no second-scale sync in these 36 runs, with a maximum
around 69 ms. This sample size does not establish an upper latency bound.
Growing-16 median throughput is 14–16% lower than growing-128 because of
more serial barriers. Sealing adds only about 2% over the growing 16 MiB
variant; this evidence does not justify a new segmented storage format.

The next prototype should budget **unpublished payload bytes**, independently
of WAL frames, while retaining final protected sync and recovery/GC leases.
First assess an intermediate budget such as 64 MiB and its foreground-barrier
tradeoff. A byte budget needs a safe shared start point for the OPEN batch;
total file size is not dirty-byte count, and multiple connections and GC must
be covered. Do not apply the segmentation probe directly to the managed format.

WAL/database checkpoint interference needs separate work. The worker's PASSIVE
checkpoint sync overlaps explicit foreground FULL commits. Suppressing redundant
work must preserve prompt CLEAN progress and the 64 MiB admission rule; previous
threshold/coalescing failures remain relevant. Original adoption gates are
unchanged; presync still has 14 failing gates in `reference-adoption.json`.

## Validation and retained evidence

Recorder contracts pass Release and ASAN/UBSAN for all six file classes, errno
preservation, EIO/EINTR, invalid descriptors, descriptor reuse, deleted files,
nested site restoration, concurrent threads and unscoped calls: 25 calls each.
Only related checks ran; prior component tests were not repeated.

All 16 FUSE mounts pass contents/storage inspection. All 36 raw-file runs pass
SHA-256, reclaimed-zero ranges, retained guards, marker contents, write counts
and dirty-byte budgets. `audit.json` records **54 removed temporary paths**
after owned processes stopped. Measurements do not overlap owned builds/tests,
and free space is checked before each native run. One initial raw-probe preflight
failed on a monitor path before plan creation or any measured samples; its
correction and logs are retained.

Published chunks preserve every sync vector, corresponding parent row and
original request index losslessly. Full request traces, CPU samples, host/build
logs, binaries and sources remain locally with paths, byte sizes and SHA-256
identities in `retention.json`. Every JSON is below 500 KiB. No timing vector
was rounded and no slow sample was removed.

## Reproduce

Use the frozen presync artifact as `--parent`, with its preceding diagnostic
build for byte-equivalence checks. Supply the same SQLite and FUSE SDKs.

```bash
.venv/bin/python vane_fs/benchmarks/staged_payload/sync_costs/prepare.py \
  /tmp/vane-sync-costs --parent /path/to/frozen-presync-artifact \
  --sqlite-prefix "$PWD/vane_fs/vcpkg_installed/x64-linux-release" \
  --fuse-prefix "$PWD/.cache/fuse-sdk/root/usr"
.venv/bin/python vane_fs/benchmarks/staged_payload/sync_costs/validate.py \
  /tmp/vane-sync-costs \
  --sqlite-prefix "$PWD/vane_fs/vcpkg_installed/x64-linux-release"
.venv/bin/python vane_fs/benchmarks/staged_payload/sync_costs/run.py /tmp/vane-sync-costs
.venv/bin/python vane_fs/benchmarks/staged_payload/sync_costs/analyze.py \
  /tmp/vane-sync-costs --prior-diagnostic /path/to/frozen-presync-diagnostic
.venv/bin/python vane_fs/benchmarks/staged_payload/request_costs/analyze.py /tmp/vane-sync-costs
.venv/bin/python vane_fs/benchmarks/staged_payload/request_costs/analyze_cpu.py /tmp/vane-sync-costs
.venv/bin/python vane_fs/benchmarks/staged_payload/checkpoint_batching/analyze_diagnostic.py \
  /tmp/vane-sync-costs
.venv/bin/python vane_fs/benchmarks/staged_payload/sync_costs/probe_batches.py /tmp/vane-sync-costs
.venv/bin/python vane_fs/benchmarks/staged_payload/sync_costs/publish.py /tmp/vane-sync-costs
```

Build and validate before any measurement window. Diagnostic throughput never
replaces uninstrumented original-workload adoption checks.

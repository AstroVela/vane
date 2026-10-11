# Partial write batching acceptance

This follow-up separates the preserved [partial-write candidate](../README.md)
from physical-storage synchronization costs. It compares the same production
format-2 baseline and candidate; checkpoint thresholds, SQLite SDK, cache limits,
transaction boundaries and durability settings remain fixed. The baseline native
sources match production at `61ca3015360396fec354524e2720aa93f98c5e95`.

**The candidate remains experimental.** Software work decreases, but the
physical-storage comparison fails the predeclared complete-workload gates.
Neither the production C++ implementation nor the installed component changes.

The native probes use the previously built Release libraries without SQL or I/O
wrappers. Separate counter probes measure SQLite calls and WAL bytes. Both run on
tmpfs, which isolates software costs but does not measure durable-storage speed.
All content comparisons happen outside the write/sync timers.

## Protocol fixed before measurement

[plan.json](plan.json) retains the acceptance criteria and sample counts:

- Eight alternating native pairs for each initial offset: 0, 1 and 4,095 bytes.
  Each writes 64 MiB in 1 MiB calls using strict durability. Three additional
  pairs per offset 0 and 1 use the existing 16 MiB SQL counter probe; their
  timings are excluded from throughput comparisons.
- Six alternating FUSE pairs per medium and state: tmpfs/ext4, fresh/post-GC.
  Every mount first writes a 256 MiB file in 8 MiB application requests, then a
  64 MiB file in 1 MiB requests. An mmap buffer fixes its page offset at 48 bytes
  for both variants. The 256 MiB file therefore starts in a genuinely new
  workspace, or an independently prepared aged workspace. The 64 MiB item uses
  that same workspace afterward. Neither window is pooled with earlier tests.
- The unchanged post-GC fixture retains a 32 MiB guard with alternate 4 KiB
  payloads removed. Warm reads, 512 seeded 4 KiB overwrites, 128 small files with
  individual fsync and 128 operations of each directory type follow large writes.
  Large-write throughput includes create, writes, fsync and close. Ordinary file
  close is not a separate durability barrier in fsync mode.
- Separate instrumented ext4 mounts record sync calls, checkpoint calls, the
  admission mutex wait and complete admission duration. FUSE callback sizes and
  offsets verify the request split. These timings do not replace plain results.
- Ordinary-file controls bracket the ext4 window: three serial large-call and
  three paced framed-call trials on each side, using the previous
  [I/O reproducer](../../io_interference/README.md). They contain no SQLite or
  FUSE calls. Their schedule is retained in [controls-plan.json](controls-plan.json).

Native acceptance requires at least a 5% unaligned CPU reduction in the median,
with a reduction in at least six of eight pairs, and no more than a 5% aligned
regression. FUSE acceptance requires median and aggregate throughput within 5%
of baseline or better for every workload, no more than a 20% request-p99 increase
when the absolute increase exceeds 1 ms, and lifecycle totals within 5%. These
are conservative adoption gates, not claims of statistical significance. Every
slow sample is retained; controls do not excuse a candidate failing its gate.

## Software cost on tmpfs

Plain native write-phase process CPU medians:

| Initial offset | Baseline, seconds | Candidate, seconds | Change | Pairs with lower CPU |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.237401 | 0.221980 | -6.5% | 4/8 |
| 1 | 0.318974 | 0.228376 | -28.4% | 8/8 |
| 4,095 | 0.319302 | 0.234535 | -26.5% | 8/8 |

The unchanged aligned algorithm is a control, so its noisy CPU difference is
not attributed to the optimization. Both unaligned cases pass the native gate.

Every 16 MiB offset-one counter run retains the previous 16,752-to-592 SQLite
step reduction. Both variants write exactly 21,370,472 WAL bytes and 16,424,960
database bytes during that phase. At offset zero, both use 320 SQLite steps and
write 21,230,392 WAL bytes. Those equal counts do not establish identical page
order or checkpoint scheduling. Counter-probe timings are excluded above.

Plain FUSE medians on tmpfs:

| Workload | Fresh baseline | Fresh candidate | Post-GC baseline | Post-GC candidate |
| --- | ---: | ---: | ---: | ---: |
| First 256 MiB write, MiB/s | 186.82 | 264.79 | 191.91 | 257.71 |
| Subsequent 64 MiB write, MiB/s | 69.01 | 70.04 | 67.40 | 66.76 |
| Warm sequential read, MiB/s | 525.47 | 530.65 | 522.01 | 529.43 |
| Warm 4 KiB random read, MiB/s | 72.68 | 73.25 | 71.50 | 72.94 |
| 4 KiB overwrite + final fsync, MiB/s | 21.58 | 22.25 | 21.86 | 22.18 |
| Small file + fsync, files/s | 1,430.42 | 1,406.72 | 1,492.26 | 1,511.95 |

Large-write median throughput improves by 41.7% fresh and 34.3% after GC.
Fresh small-file median throughput is 1.7% lower, and aggregate throughput is
5.2% lower, narrowly failing the predeclared gate. The remaining tmpfs workload
and lifecycle gates pass. This establishes the target-path software benefit;
it does not establish an overall production performance improvement.

## Physical-storage result

The host has one physical disk, with ext4 on `/dev/sda2`, and approximately
95 GiB free at the start. No separate physical device was available for an
independent storage comparison. The tmpfs and ext4 windows use the same libraries,
SDK and application workload, but their measurements are kept separate.

Plain ext4 FUSE medians, including every sample:

| Workload | Fresh baseline | Fresh candidate | Post-GC baseline | Post-GC candidate |
| --- | ---: | ---: | ---: | ---: |
| First 256 MiB write, MiB/s | 91.04 | 95.15 | 88.91 | 94.88 |
| Subsequent 64 MiB write, MiB/s | 47.41 | 46.82 | 44.36 | 28.13 |
| Warm sequential read, MiB/s | 519.26 | 519.25 | 522.06 | 509.06 |
| Warm 4 KiB random read, MiB/s | 65.66 | 67.08 | 67.04 | 70.08 |
| 4 KiB overwrite + final fsync, MiB/s | 9.97 | 9.40 | 5.51 | 10.92 |
| Small file + fsync, files/s | 131.49 | 134.34 | 44.62 | 259.94 |
| mkdir + directory fsync, ops/s | 1,789.41 | 1,729.66 | 1,805.03 | 1,840.09 |
| rename + directory fsync, ops/s | 1,961.14 | 1,893.83 | 2,137.54 | 2,028.92 |
| rmdir + directory fsync, ops/s | 2,133.95 | 2,178.14 | 2,262.22 | 2,019.56 |

The fresh 256 MiB median improves 4.5%, but aggregate throughput falls from
90.88 to 80.69 MiB/s, an 11.2% regression. Its maximum application-write latency
grows from 335 ms to 3.299 s. After GC, the 256 MiB aggregate improves from
89.07 to 92.25 MiB/s, while the subsequent 64 MiB aggregate falls from 24.75 to
16.14 MiB/s. Both variants stall in the latter workload, with maximum request
latencies of 6.753 and 6.331 s respectively. Request-p99 rises from 290 to 392 ms.
The roughly 6-second maxima alone therefore conceal the different frequency and
total cost of pauses.

Fresh overwrite aggregate throughput falls from 5.30 to 3.60 MiB/s, while aged
overwrite aggregate rises from 2.68 to 11.27. These totals include final fsync;
request-only elapsed totals are close (fresh 0.980/0.985 s, aged 0.945/0.927 s).
The aged final-fsync sums are 3.531/0.138 s, for baseline/candidate. Likewise,
baseline aged small-file operations repeatedly stall while this candidate window
has no comparable pause. These observations do not establish intrinsic gains of
that magnitude from partial-block batching. All samples, request latencies,
barrier times and the opposite-direction regressions remain in the report.

Complete lifecycle totals are 88.23/91.84 s fresh and 134.10/102.71 s after GC.
They include preparation, validation, deletion and shutdown, and exclude the
final artifact hashing/removal. Better lifecycle totals do not override failed
individual workload gates. Fresh stream/overwrite/rename and aged 64 MiB write,
rename/rmdir gates fail, alongside the narrow tmpfs small-file failure.

## Where requests wait

All eight separate diagnostic mounts verify the same 256 MiB stream split in
both variants: 32 callbacks of 1,048,528 bytes at block-aligned offsets,
224 callbacks of 1,048,576 bytes at offset 4,048 within a block, and 32 trailing
48-byte callbacks. Buffer alignment is therefore not a variant-dependent
confounder in this comparison.

One aged candidate stream waits 6.273 s acquiring the admission mutex. The wait
spans two worker PASSIVE checkpoints; during the second, WAL fsync takes 3.670 s
and database fsync takes 1.735 s. This directly measures waiting that would be
missed by timing only the subsequent SQLite RESTART call. Other baseline and
candidate phases wait in WAL synchronization, including baseline overwrite
WAL fsync calls of 1.315 s. The separate diagnostic does not reproduce every
plain-window slowdown, so it does not causally explain each failed sample.
Nested admission, checkpoint and sync durations overlap and must not be added.

The ordinary-file controls retain all 12 trials and 144 fsync calls. Eight
barriers exceed one second; the maximum is 2.852 s before the FUSE window and
3.470 s afterward. Serial file operations alone therefore reproduce this class
of stall in the same session. That does not identify its kernel/device cause,
attribute it to another workload, or prove that the candidate cannot regress.

**Decision:** preserve the candidate and its verified software benefit without
enabling it. The fixed acceptance gates fail. Further storage acceptance needs
an independent physical device or machine; this round stops without another
checkpoint-parameter trial or selectively repeating unfavorable samples.

## Validation

The preserved candidate passes **142 related Python tests and six C++ suites**;
the control passes its six native suites. Existing tests cover partial edges,
snapshots, branches, GC, rollback, retries and payload-ID exhaustion. No broader
Python or release suite runs.

All 48 plain native trials, 12 native counter trials, 48 plain FUSE mounts,
eight diagnostic mounts and 12 ordinary-file controls pass their content and
database checks. The FUSE runs verify 229,376 random-read slices as well as full
file and retained-guard content. Every post-GC fixture has 4,128 free SQLite pages
before its measured workloads.

The published preparer rebuilds all six plain/counter/diagnostic executables
byte-identically to those measured. [Reproducer validation](reproducer-validation.json)
also checks the measured and published workload function bodies, allowing the
recorded formatting/import cleanup, overwritten defaults and one local-variable
rename. Frozen original runners remain available with the artifacts.

No owned builds or correctness tests overlap the measurement windows. Across
the monitored windows, host-wide CPU busy medians are about 5–6%, peaks remain
below 16%, and available memory stays above 53 GiB. Those aggregate counters
include preparation and cleanup; they do not identify the source of I/O waiting.
No host, scheduler, device or global-cache setting is changed.

After all owners stop, **128 generated benchmark data directories** and the
component Python test directory are removed. All recorded FUSE processes have
exited and mounts are absent. The production sources, SQLite SDK and installed
component retain their original identities.

## Reproduction and retained evidence

The parent [preparer](../prepare.py) reconstructs and checks both component source
trees. Build each in `build/python-release`, with tools, FUSE and native tests
enabled, and install its wheel non-editably into its own `venv`. Use the same
SQLite SDK for both, following the [component instructions](../../../README.md).
Then build the plain probes and separate instrumented mounts:

```bash
python vane_fs/benchmarks/partial_write/acceptance/prepare.py /tmp/partial-check \
  --baseline /tmp/partial-components/baseline \
  --candidate /tmp/partial-components/partial-write \
  --sqlite-prefix "$PWD/vane_fs/vcpkg_installed/x64-linux-release" \
  --fuse-prefix "$PWD/.cache/fuse-sdk/root/usr"
python vane_fs/benchmarks/partial_write/acceptance/validate.py /tmp/partial-check
python vane_fs/benchmarks/partial_write/acceptance/run_native.py /tmp/partial-check
python vane_fs/benchmarks/partial_write/acceptance/run_fuse.py /tmp/partial-check --medium tmpfs
python vane_fs/benchmarks/partial_write/acceptance/run_controls.py /tmp/partial-check /path/to/io-probe --phase before
python vane_fs/benchmarks/partial_write/acceptance/run_fuse.py /tmp/partial-check --medium ext4
python vane_fs/benchmarks/partial_write/acceptance/run_controls.py /tmp/partial-check /path/to/io-probe --phase after
python vane_fs/benchmarks/partial_write/acceptance/run_fuse.py /tmp/partial-check --medium ext4 --diagnostic
python vane_fs/benchmarks/partial_write/acceptance/analyze.py /tmp/partial-check
```

Use a new output directory. The runner writes a separate result file for each
window, verifies content and the SQLite schema, stops owned mounts/processes,
and removes generated workspace data outside the timers. It never evicts global
caches or changes host storage settings. Instrumented events may overlap across
threads or cross a phase boundary; their durations are not additive wall time.

Build `io-probe` using the linked I/O-reproducer instructions. [Results](results.json)
retain the configurations, window timestamps, host summaries and cleanup records;
[summary.json](summary.json) contains all acceptance outcomes and diagnostic
distributions. The adjacent native, per-medium/state/variant FUSE and control
JSON files retain every scalar sample and request-latency vector. The
[artifact manifest](artifact-manifest.json) records source/binary identities,
full timelines, frozen runners, logs and validation evidence retained locally.

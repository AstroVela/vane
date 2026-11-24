# Checkpoint admission priority evaluation

**The candidate remains isolated and is not enabled.** Foreground priority
changes checkpoint handoff as intended, but fails the fixed complete-workload
performance gates on the dedicated host. It substantially reduces mutex waiting
while moving checkpoint I/O into the foreground RESTART call.

The control is production revision
`8e21f48ea64c7811893d847116ccc4fbf204f81d`. The
[candidate patch](priority.patch) changes only `src/checkpoint.hpp` in a prepared
component copy. It does not include the earlier partial-write candidate, change
the 4 KiB BLOB format, weaken synchronization, or change any WAL threshold.
Production sources and the installed component retain their original hashes.

## Candidate and scope

Before waiting for the checkpoint mutex, admission announces its claim through
an atomic counter. A scope releases that claim after RESTART, including exceptions.
After acquiring the mutex, the worker yields its next PASSIVE checkpoint when a
claim is pending. It uses the existing 20 ms retry wait and preserves pending
maintenance across Stop/Start. An already running checkpoint is allowed to finish.
Priority applies to this connection's admission, not every writer in the database.
Explicit FULL barriers still bypass WAL admission.

The [frozen plan](plan.json) requires every correctness check to pass. For both
ext4 states, first-file aggregate throughput must improve at least 5%, with faster
complete writes in at least five of six pairs. Every plain workload's median and
aggregate rate must retain at least 95% of the control, pooled request p99 must
stay within 120% unless the increase is at most 1 ms, and lifecycle totals must
stay within 105%. These conservative gates are not a significance test. Every
slow sample is retained; no checkpoint parameter is tuned after seeing results.

The host owner confirms dedicated hardware. Both variants use the same SATA SSD,
ext4 on `/dev/sda2`, original SQLite 3.53.2 fsync SDK, fsync durability mode and
fixed application-buffer page offset of 48 bytes. The worker starts at 16 MiB,
admission applies at 64 MiB, and retained WAL allocation is 16 MiB. No global
cache eviction, device configuration or kernel setting changes.

Plain tmpfs and ext4 windows each contain six alternating pairs for fresh and
independently prepared post-GC workspaces: **48 plain mounts** in total. Each
starts with a 256 MiB file written through 8 MiB application requests, then a
64 MiB file through 1 MiB requests, followed by reads, random overwrites,
small-file fsync and directory operations. Eight separate ext4 diagnostic mounts
use the same workload. Twelve serial ordinary-file controls bracket the plain
ext4 window. No owned builds or correctness tests overlap measurement windows.

## Complete write results

Rates include create/write/fsync/close. Aggregate rate is total logical bytes
divided by total measured time across all six trials, including slow trials.

| Medium/state | Workload | Control median MiB/s | Candidate median MiB/s | Control aggregate MiB/s | Candidate aggregate MiB/s |
|---|---|---:|---:|---:|---:|
| tmpfs fresh | First 256 MiB | 180.34 | 181.43 | 180.37 | 181.56 |
| tmpfs post-GC | First 256 MiB | 188.85 | 187.76 | 186.99 | 186.91 |
| ext4 fresh | First 256 MiB | 90.81 | 88.25 | 91.48 | 88.07 |
| ext4 post-GC | First 256 MiB | 89.26 | 85.93 | 64.82 | 63.40 |
| ext4 fresh | Subsequent 64 MiB | 47.24 | 44.85 | 29.18 | 28.72 |
| ext4 post-GC | Subsequent 64 MiB | 44.19 | 45.42 | 26.40 | 45.42 |

The candidate wins zero of six complete first-file writes on fresh ext4 and two
of six after GC. The post-GC subsequent-file aggregate improves, but does not
override failed target and other-workload gates.

| ext4 workload | Fresh control aggregate | Fresh candidate aggregate | Post-GC control aggregate | Post-GC candidate aggregate |
|---|---:|---:|---:|---:|
| Random 4 KiB write + final fsync, MiB/s | 5.67 | 3.73 | 5.95 | 4.04 |
| 4 KiB file + fsync, files/s | 55.14 | 42.34 | 64.31 | 46.29 |
| mkdir, operations/s | 1633.88 | 1886.20 | 1847.10 | 1717.31 |
| rename, operations/s | 1928.74 | 1975.94 | 2080.63 | 1930.94 |

Fresh lifecycle totals increase from 88.65 to 95.40 s; post-GC totals increase
from 114.37 to 122.30 s. Both exceed the 5% lifecycle gate. Post-GC tmpfs random
overwrite aggregate throughput also fails its gate. Full distributions, paired
wins, tails and every workload are in [summary.json](summary.json).

## Where waiting moves

The following sums cover both diagnostic first-file trials in each state.
Admission timing starts after the initial WAL-status probe and covers mutex
acquisition through completion of the subsequent checkpoint work.

| State/variant | Mutex wait, s | Foreground RESTART, s | Whole admission scope, s | PASSIVE starts inside mutex waits |
|---|---:|---:|---:|---:|
| Fresh control | 7.035 | 0.000213 | 7.035 | 10 |
| Fresh candidate | 1.200 | 6.540 | 7.740 | 0 |
| Post-GC control | 2.461 | 0.000262 | 2.462 | 10 |
| Post-GC candidate | 0.283 | 1.923 | 2.206 | 0 |

Candidate yield events confirm the intended handoff. The fresh mutex wait falls
82.9%, but whole admission time rises 10.0%. This directly demonstrates why a
smaller mutex-wait metric alone cannot establish a write performance gain.

One candidate fresh-file admission lasts 5.701 s: 0.862 s waiting for the mutex
and 4.839 s in RESTART. Baseline samples also contain long synchronization;
one post-GC subsequent-file admission waits 5.952 s while the worker performs
checkpoint I/O. These are separate trials, not controlled reproductions of an
identical device stall. The traces locate and transfer waiting without uniquely
attributing each plain-window regression to the policy, kernel or SSD.

Serial ordinary-file controls contain 144 fsync calls, seven exceeding one
second. Maxima are 2.673 s before the plain ext4 window and 3.411 s afterward.
They reproduce synchronization stalls without SQLite, FUSE or concurrent
controlled writers. Competing-workload attribution remains unproven. Nested
checkpoint, fsync and admission timings overlap and must not be added as
exclusive wall time.

## Validation and retention

The [scheduling probe](test_priority.cpp) pauses an announced admission before
mutex acquisition and controls the first checkpoint's completion. It checks
ordering, release after Busy and maintenance after Stop/Start. All three probes
reproduce the control's second PASSIVE preceding admission (expected exit 1);
the candidate passes all three. Probe-only header copies contain observation
hooks and mocked checkpoints; plain and storage diagnostic libraries do not
contain these test hooks.

Both variants pass their **six real C++ suites**. The candidate passes **132
related Python tests**, including FUSE, recovery and filesystem behavior. No
broader Vane or release suite runs. The initial Python launcher found no pytest
module and collected no tests; the environment was repaired using the same
explicit dependency support path as the preceding component runs. That attempt,
the successful Python retry and unchanged native-suite results are retained in
[tests.json](tests.json) and [environment repair](python-environment-fix.json).

All 48 plain mounts, eight diagnostic mounts and 12 ordinary-file controls pass
their content/schema checks. The new analyzer reproduces 280 aggregate,
distribution and lifecycle results from the preceding acceptance report exactly.
The published preparer rebuilds all **eight measured executables byte-identically**,
including both scheduling probes; see [reproducer validation](reproducer-validation.json).

After all owners stop, **68 generated measurement directories** and the Python
test directory are removed. All 56 recorded FUSE processes have exited and
mounts are absent. Configurations, raw latency samples, code/binary identities,
logs, host counters, full timelines and cleanup hashes are retained through the
[artifact manifest](artifact-manifest.json) and [audit](audit.json). Full database
and generated payload files are not retained. Preparation failures occurred
before performance windows and remain in the local artifacts.

## Reproduce

Prepare a new artifact directory from the frozen revision with the existing
SQLite and FUSE SDKs. `--test-support` identifies a Python 3.12 site-packages
directory containing the existing pytest, pytest-timeout and fsspec dependencies.
The candidate receives a non-editable incremental Release installation in its
own environment; the root installation is not replaced.

```bash
.venv/bin/python vane_fs/benchmarks/checkpoint_priority/prepare.py <new-output> \
  --sqlite-prefix "$PWD/vane_fs/vcpkg_installed/x64-linux-release" \
  --fuse-prefix "$PWD/.cache/fuse-sdk/root/usr" \
  --test-support <existing-test-site-packages>
.venv/bin/python vane_fs/benchmarks/checkpoint_priority/run.py <new-output> \
  --control-binary <existing-io-probe>
.venv/bin/python vane_fs/benchmarks/checkpoint_priority/analyze.py <new-output>
```

The ordinary-file probe source and configuration are preserved in
[the preceding I/O evaluation](../io_interference/README.md).
`--resume` is restricted to a validation repair before any performance window
exists. It verifies retained scheduling results, successful repaired validation
and unchanged component sources. It cannot selectively rerun performance trials.

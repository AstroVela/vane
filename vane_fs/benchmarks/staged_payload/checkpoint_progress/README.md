# Commit progress after WAL reuse

This isolated C++ prototype preserves background progress for small CLEAN
commits after a large WAL generation is reused. **It is not adopted:** the
native replay passes its screening checks, but the original ext4 FUSE workload
still fails six non-regression checks. Production sources, the installed
4 KiB SQLite BLOB module and its strict default remain unchanged.

## Evidence and minimal change

The preceding [restart experiment](../checkpoint_restart/README.md) found
that reusing a large WAL leaves later small commits below the unchanged
16 MiB wake threshold. Background syncing decreases and more work reaches
the final application fsync. This also occurs after an explicit reusable
generation in the preceding orphan-batching implementation.

`progress.patch` changes only the commit callback and adds one atomic boolean
in `src/checkpoint.hpp` over the restart prototype. A real large-generation
callback activates the boolean; subsequent commits continue waking the worker
even if the WAL has become small. The callback performs no SQL/status query
or sync. The state remains local to that worker and survives Stop/Start;
opening a new cold workspace does not activate it.

Worker scheduling, partial/error retry, optional zero-timeout RESTART, 16/64 MiB
thresholds and write admission are byte-identical to the preceding restart
candidate. OPEN publication coalescing and the publication/FULL/CLEAN/recovery
bodies, workspace operations, payload layout, GC and component tests also remain
byte-identical. There is no index removal, presync, hint, new format or delay.
Fuzz-zero replay of the partial-edge source plus orphan, restart and progress
patches reconstructs all 35 source inputs.

## Real progress tests

Five new cases pass in Release and ASAN/UBSAN:

- Small cold commits leave the original background-I/O policy intact.
- A real 20 MiB commit activates maintenance. After a reusable generation and
  a small commit while stopped, Start backfills without another commit.
- A running worker also backfills a small post-reuse commit without more writes.
- An older reader pins a small tail; releasing it lets the existing retry finish
  without another commit.
- Background work begins during 256 continuous small commits below the original
  threshold, rather than waiting for the stream to finish.

Each case checks real WAL progress, acknowledged SQL bytes and SQLite integrity;
callback wrappers verify zero status queries and syncs. Both preceding orphan
and restart controls fail the post-reuse progress assertion as expected, while
their cold cases pass. The two failures and their exact messages are retained.

Related validation also passes: 154 installed Python tests, 11 original-control
and 11 candidate C++ component suites, 11 ASAN/UBSAN component suites, 53 inherited
positive helper cases and one inherited expected negative. Including the new
cases and two cold controls, there are 65 positive helpers and three expected
negatives. One unsupported production-v1 migration test is skipped for format
1004. No broad base or Ray suite runs. The unmodified typed recorder's prior
Release/ASAN tests are reused only after checking every source, SDK and helper
binary identity; the new replay independently reconciles all recorded syncs.

## Frozen native replay

Fifty-four instrumented native API runs compare orphan batching, the old restart
prototype and the new candidate. Six permutations balance all three variants
within each state. Every phase contains 128 directory operations and a final
`Workspace::Sync()` inside the timer; setup, inspection, close and cleanup remain
outside. This is screening evidence, not FUSE adoption throughput.

The three states are a cold small workspace, a real 20 MiB WAL made reusable
before measurement, and a reader retaining an older partially backfilled WAL.
The frozen checks require complete-phase median throughput within 5% of orphan
batching for every state/workload, plus at least 20% lower final-barrier median
than the old restart prototype after reuse. All twelve checks pass.

| Reused state | Old restart final fsync, ms | Candidate | Complete throughput vs orphan |
| --- | ---: | ---: | ---: |
| mkdir | 14.09 | 5.29 | +19.8% |
| rename | 12.79 | 3.91 | +20.8% |
| rmdir | 10.61 | 3.06 | +27.9% |

After reuse the controls have zero background WAL syncs in these phases.
Candidate median counts are 8.5, 10.5 and 9. Corresponding inclusive WAL-sync
time is 25.86, 31.37 and 26.94 ms. These overlapping background durations cannot
be added to application elapsed time. The native replay does not show extra
foreground header syncs: all three variants have one ordinary-mkdir WAL sync
and one observed generation reset, and none during ordinary rename or rmdir.
That rules out claiming repeated header-sync cost as this native replay's cause.

All 1,141 sync calls reconcile with request counts and durations. Nine calls
of at least 100 ms remain, including post-phase maintenance: retained candidate
round 0 has foreground WAL syncs of 320, 403 and 248 ms and later background
syncs of 1.31 and 1.40 seconds. Retained orphan round 1 has its own long syncs.
No sample is dropped. The screening checks use medians and do not certify tails.

## Original ext4 FUSE screen

After the native screen, 24 uninstrumented ext4 mounts use the original no-hint
partial-edge control and the existing workload: six alternating pairs per
fresh/post-GC state, page-offset-48 buffers, complete first-256 MiB and subsequent
64 MiB writes including final fsync and close, reads, random I/O, small-file
fsync and namespace operations. Each variant uses its own verified native
module for GC; index/content/schema/retained-guard checks run outside timers.

The ext4 subset retains every original gate constant, including 10% combined
aggregate gain, five wins out of six, median/aggregate ratios at least 0.95,
pooled p99 within 1.20 times control or 1 ms absolute growth, and lifecycle
median within 1.05 times. It is explicitly ineligible for full adoption; tmpfs
is not run after ext4 rejects the candidate.

| State | Combined median MiB/s, control→candidate | Combined aggregate | Wins |
| --- | ---: | ---: | ---: |
| fresh | 112.93→145.96 | 109.53→144.63 (+32.0%) | 6/6 |
| post-GC | 110.01→143.96 | 70.38→103.57 (+47.2%) | 5/6 |

These gains include inherited orphan batching and RESTART; the native
three-variant replay isolates the additional progress change. First-256 medians
are 156.30→154.15 MiB/s fresh and 151.99→152.49 post-GC. Subsequent-64 medians
increase 53.64→120.53 and 52.18→115.40. Subsequent-write pooled p99 is
73.25→51.61 ms fresh and 68.84→51.69 post-GC, passing the original latency gate.

Six checks still fail, in both states:

| Workload | Fresh median, control→candidate | Post-GC median |
| --- | ---: | ---: |
| 4 KiB random write, MiB/s | 13.64→9.51 (-30.3%) | 12.11→11.28 (-6.8%) |
| Small file + fsync, files/s | 257.10→226.17 (-12.0%) | 249.29→228.11 (-8.5%) |
| rename, operations/s | 2232.92→1832.28 (-17.9%) | 2398.72→1855.82 (-22.6%) |

The post-GC control retains a 5.011-second subsequent-write phase and a
7.830-second first-write phase. The candidate retains a 5.777-second
subsequent-write phase. These affect aggregate ratios; none is excluded or
attributed to another load. The plain FUSE screen cannot assign its remaining
rename/random-write regressions to a specific syscall. Native API benefits
must not be assumed to carry unchanged through FUSE.

## Payload publication lock review

`payload-lock-review.json` verifies that the final payload sync occurs under
the publication guard and inside the worker's SQLite IMMEDIATE transaction,
before the FULL SQL commit and CLEAN fence. The OPEN recovery record retains
the durable WAL prefix captured when the batch began. Releasing protection
around the final sync would allow new unsynced external bytes to acquire
metadata in that FULL commit unless a stable publishable boundary is added.

The earlier [unlocked preparatory sync](../payload_presync/README.md) already
reduced first-write lock wait but retained severe sync tails and failed its
adoption gates. This iteration reviews that evidence and keeps the final
publication contract intact; it adds no payload runtime change.

Next collect FUSE request/CPU and typed-sync costs for the three remaining
regressions before altering another checkpoint policy. For payload tails,
define and fault-test a stable publishable payload/WAL boundary before reducing
the protected sync's lock scope. The current small-WAL progress fix provides
useful evidence, but does not satisfy whole-filesystem performance requirements.

## Build, retention and reproduction

The candidate uses a non-editable Release installation in its isolated
incremental build directory, with the same SQLite 3.53.2 fsync and FUSE 3.14
SDKs. The first recorder-helper link omitted the `pread64`/`pwrite` wrappers;
its log, recipe and report remain under `link-preflight/`. Recovery also records
a missing helper-output-directory failure. Only helpers are relinked; the
runtime binaries are reused after identity verification. The corrected builder
includes both flags and creates those directories for fresh reproduction.

All 54 native runs and 24 FUSE mounts validate and remove their databases and
payloads after processes stop/unmount. `audit.json` checks 147 removed paths,
fuzz-zero source reconstruction, lossless FUSE gate reconstruction and no owned
build/test/formatter overlap with measurement windows. There is no global cache
eviction or CPU/device setting change. JSON is retained losslessly below
500 KiB. Native traces, necessary logs and measurements are committed; sources,
SDKs, binaries and host measurements remain locally with identities in
`manifest.json`. No large generated workspace data is retained.

With a retained preceding restart artifact and fresh output directory:

```bash
python vane_fs/benchmarks/staged_payload/checkpoint_progress/prepare.py "$progress_out" \
  --restart-parent "$restart_parent"
python vane_fs/benchmarks/staged_payload/checkpoint_progress/validate.py "$progress_out"
python vane_fs/benchmarks/staged_payload/checkpoint_progress/run.py "$progress_out"
python vane_fs/benchmarks/staged_payload/checkpoint_progress/analyze.py "$progress_out"
python vane_fs/benchmarks/staged_payload/checkpoint_progress/screen.py "$progress_out"
```

The builder derives original controls and SDK paths from the preceding frozen
build, and freezes its adapted generator. `screen.py` reuses the exact original
FUSE workload/adapter and requires the native checks to pass before starting.
`publish.py` consolidates this recorded run and its retained link preflight.

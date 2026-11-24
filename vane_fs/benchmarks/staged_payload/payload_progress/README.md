# Independent payload progress experiment

**Reject this policy for adoption.** Correctness checks pass, but **11 of the
22 original performance checks fail**. Shorter final fsync does not produce
a consistent improvement in complete writes. Production format 2 and the
default installed native module/SQLite SDK are unchanged.

| Complete large-write aggregate, MiB/s | Extent control | Payload trigger | Change | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| Full fresh | 152.04 | 79.65 | -47.6% | 1/6 |
| Full post-GC | 50.72 | 86.43 | +70.4% | 3/6 |
| Sustained fresh (supplementary) | 113.99 | 75.77 | -33.5% | 0/2 |
| Sustained post-GC (supplementary) | 115.67 | 85.46 | -26.1% | 0/2 |

These are matched results from this window, comparing two isolated extent
components. They are not production throughput or a cross-system comparison.
The full fresh lifecycle median grows **7.485 to 12.968 seconds (+73.3%)**.
Post-GC lifecycle falls 17.225 to 12.316 seconds, but the primary paired-win
requirement and multiple individual workload checks still fail. All failures
and their exact reasons remain in [the analysis](validation/analysis.json).

The first-256 phase illustrates the tradeoff:

| Full first-256 measurement | Fresh control | Fresh trigger | Post-GC control | Post-GC trigger |
| --- | ---: | ---: | ---: | ---: |
| Write-loop median, seconds | 0.437 | 1.105 | 0.437 | 1.127 |
| Final-fsync median, seconds | 0.611 | 0.159 | 3.337 | 0.159 |
| Pooled write-call p99, milliseconds | 30.554 | 389.749 | 31.751 | 216.476 |
| Daemon phase CPU median, seconds | 0.480 | 0.640 | 0.490 | 0.690 |

Thus the final barrier improves while ordinary writes acquire more latency
and CPU cost. In the full fresh window, control has zero large-file writes
over one second and the candidate has two. In the supplementary window the
candidate has four fresh and five post-GC write calls over one second, versus
zero in each control; both sustained aggregate rates regress. The mechanism
still holds the publication gate during payload synchronization, and external
writers require that gate. These application timings are consistent with
moving work into writes, but this uninstrumented window does not identify
individual native waits or the kernel/device cause of long syncs. Native
sync-count and held-lock attribution is the next diagnostic step before
considering a separate prewriteback path. No threshold retuning is included.

This isolated C++ trial derives from the frozen format-1005 extent component.
It tests one scheduling hypothesis from the
[final-fsync diagnosis](../extent_waits/README.md): compact metadata no longer
reaches the WAL wake threshold during a large write, so payload publication
waits until the application's final barrier.

The candidate adds a fixed **64 MiB append-distance trigger** to the existing
publication worker. WAL thresholds remain **16 MiB / 64 MiB**. A successful
external commit checks the independent payload budget while retaining its
publication guard. The worker rechecks the budget under that guard; either
budget can publish an OPEN batch. It uses the original payload `fdatasync`,
SQLite FULL transaction and CLEAN-fence ordering, error latch and retry path.
An aborted SQL transaction does not issue the new notification.

Reserved OPEN-fence bytes 80–87 hold the physical payload append origin,
covered by the existing CRC64. This hint is shared between owners and is
ignored by recovery. Older format-1005 OPEN records contain zero there, which
can cause an earlier publication. A new OPEN captures a new origin after
background or explicit publication, including across connections and GC.
Append distance includes padding and abandoned rollback bytes; it is a
conservative maintenance signal, not an exact count of logical dirty bytes.
The threshold is checked after each complete transaction, so a single large
transaction can exceed it. No durable format migration or production default
change is included.

The control is the unchanged extent component, also format 1005. The frozen
full comparison retains all original workloads, six AB/BA pairs per state,
fresh and independent post-GC workspaces, FULL application barriers and all
22 acceptance checks. These require at least 10% combined large-write gain
and five of six paired wins in each state, at most 5% median/aggregate loss
in individual workloads, the original pooled-p99 allowances, and at most 5%
median lifecycle growth. The supplementary window keeps four consecutive
256+64 MiB cycles in one workspace, with two pairs per state; it cannot
override a failed full comparison. No tuning or sample filtering follows
the measurements.

Related validation reuses the component Release, ASan/UBSan and Python suites,
the extent-specific cases, and adds six scheduling/failure cases per native
build: threshold and reset behavior below the WAL budget, shared owner
accounting, rollback/GC, and background payload/WAL/CLEAN synchronization
failures. The old component fails the same scheduling regression: its WAL
remains below 16 MiB and a completed 64 MiB append does not trigger background
publication. The new policy passes the corresponding regression.

Validation passes **153 related Python tests** (two environment/format skips),
**12 Release + 12 ASan/UBSan component suites**, **24 extent case executions**
and **12 new scheduling/fault case executions**. All **32 FUSE mounts** pass
content, retained-guard, fence/schema and SQLite integrity checks, including
**112 large-file content checks**. No broad base/Ray suite runs. Source proofs,
test logs and the failed-baseline regression are retained under
[component validation](validation/component/).

Reproduce with the retained SDK and frozen parent artifacts:

```bash
python3 generate.py <artifacts> --parent <versioned-extents-artifacts>
python3 build.py <artifacts>
python3 prepare.py <artifacts>
python3 run.py <artifacts>/fuse
python3 analyze.py <artifacts>/fuse
# After recording final source checks:
python3 publish.py <artifacts>/fuse
```

All owned builds and tests finish before measurement; analysis follows both
measurement windows. Raw samples, configurations, hashes, host observations,
test logs and cleanup manifests are retained. Generated workspace contents
are removed after their owned mounts and processes stop.
The [audit](audit.json) covers **32 cleaned benchmark workspaces** and **10
component/preflight data directories**, stopped processes and unchanged
identities. A harness preflight stopped before mounting because inherited
PASS markers were omitted; its failure is retained in
[the preflight record](validation/preflight/fuse-harness.json). A test-helper
preflight similarly exposed an uninitialized SQLite inspector and is retained
with the component logs. Neither produced performance samples. All retained
files are below 500 KiB; large database/payload outputs are removed.

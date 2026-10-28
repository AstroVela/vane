# Checkpoint dispatch coalescing: rejected candidate

This C++ experiment follows the redundant CLEAN checkpoint work identified in
the [request-cost diagnosis](../request_costs/README.md). It reduces worker
dispatch, but fails 17 frozen adoption checks. The candidate remains an isolated
patch over the format-1004 [budgeted-writeback experiment](../writeback_budget/README.md).
Production sources, the installed production component and the SQLite SDK are
unchanged.

## Native change and safety checks

The runtime change is confined to `src/checkpoint.hpp`. Once total committed
WAL frames reach 16 MiB, the commit hook queries current uncopied progress and
suppresses a wake if less than 16 MiB remains. The worker also defers small CLEAN
tails, using the same progress rule as OPEN tails. This deliberately changes
maintenance scheduling; it does not supply a new durability barrier.

The [SQLite WAL-hook contract](https://sqlite.org/c3ref/wal_hook.html) permits
checkpoint calls after commit. The
[NOOP checkpoint API](https://sqlite.org/c3ref/wal_checkpoint_v2.html) supplies
current frame/backfill counts without performing checkpoint work. Counts are
queried for the current generation, rather than cached across a WAL restart.
Busy, invalid, unknown or failed status queries conservatively retain the wake;
their error is not returned from a hook after an acknowledged commit.

Partial/failed worker passes keep their existing retry path, including reader
release without another commit and failed-close restart. The 64 MiB admission
limit still uses **total committed frames**, not uncopied bytes. FULL barriers
retain payload synchronization, FULL WAL commit and durable CLEAN-fence
ordering. The inherited WRITE-only hints retain their 4 MiB submission limit,
16 ms completion-to-next-start interval and balanced FULL-publication pause.

Seven dedicated cases use real WAL generations and GNU symbol wrappers:
retained/backfilled WAL with small CLEAN commits, crossing the work threshold,
restart into a large new generation, and busy/I/O/unknown/invalid hook status.
The preceding budget library fails the two small-dispatch negative controls.
Release and ASan/UBSan cases verify acknowledged content and reopen it after a
real barrier. The existing coalescing suite now verifies the new OPEN/CLEAN
dispatch policy and waits for an actual partial PASSIVE result before releasing
its pinned reader. Its admission/retry checks remain.

## Frozen plain comparison

There are six alternating AB/BA pairs for each fresh/post-GC state on tmpfs and
ext4: 48 complete mounts. The adoption control remains the original no-hint
partial-edge implementation; it is not replaced with the preceding failed
budget candidate. The original first-256 MiB/subsequent-64 MiB workload, 8/1 MiB
application buffers, page offset 48, fsync durability and gates remain frozen.
No owned build or correctness test overlaps a measurement window. There are no
frequency-policy, affinity, device, scheduler or global-cache changes.

The ext4 combined rate uses all 320 MiB written per mount divided by the sum of
complete write durations, including barriers and close. Both states must gain
at least 10% with at least five paired wins; every individual workload must also
meet the original median/aggregate and p99 nonregression gates.

| ext4 state | Control combined MiB/s | Candidate combined MiB/s | Change | Paired wins |
|---|---:|---:|---:|---:|
| Fresh | 112.23 | 119.71 | +6.7% | 6/6 |
| After GC | 109.34 | 95.37 | −12.8% | 5/6 |

Selected individual **medians** illustrate the tradeoff; they do not replace
the all-sample aggregates above:

| ext4 phase | Fresh control | Fresh candidate | After-GC control | After-GC candidate |
|---|---:|---:|---:|---:|
| First 256 MiB write, MiB/s | 153.19 | 188.79 | 153.30 | 191.29 |
| Subsequent 64 MiB write, MiB/s | 53.23 | 49.38 | 50.61 | 48.13 |
| 4 KiB random write, MiB/s | 12.36 | 10.42 | 13.10 | 10.44 |
| mkdir, operations/s | 1787.13 | 1206.62 | 1823.64 | 1185.25 |
| rename, operations/s | 2014.08 | 1303.43 | 2074.62 | 1284.89 |
| rmdir, operations/s | 2277.06 | 1435.08 | 2144.19 | 1330.88 |

The after-GC candidate's repetition 3 first write takes 5.617 s and runs at
45.575 MiB/s, including one 4.179 s application write and a 0.379 s final fsync.
Keeping that sample makes first-write aggregate throughput regress by 18.9%
despite the faster median. Tmpfs random-write aggregate also regresses by 27.1%
fresh and 21.4% after GC. All samples remain in [the lossless arrays](windows.json),
and [the unchanged analyzer's decision](summary.json) is `REJECT`.

## Separate diagnosis

Another 16 ext4 mounts run the existing C++ request recorder with read-only
owned-thread CPU sampling. They use four alternating pairs in each state, with
unchanged workload buffers. Diagnostic timing does not replace the plain gates.
[Maintenance analysis](diagnostic/maintenance-analysis.json) separates operation
handlers from the final file/directory fsync, while
[request costs](diagnostic/request-costs-summary.json) and
[CPU sampling](diagnostic/cpu-summary.json) retain their earlier definitions.
Worker costs cover whole spans overlapping a phase; nested stage times are
inclusive and must not be added together.

The worker reduction is real, but work moves to foreground barriers:

- Fresh 128-operation rename phases overlap 18–34 PASSIVE worker spans in the
  control and one in the candidate. Median final application fsync rises from
  2.889 to 13.879 ms; its measured fsync syscalls account for 2.681 and 13.646 ms.
  Fresh mkdir drops from 23–28 PASSIVE overlaps to zero, while its final barrier
  rises from 3.656 to 20.764 ms. After-GC phases show the same direction.
- Subsequent 64 MiB write handlers spend a median 0.188 ms in admission in the
  control and 141.410 ms in the candidate. The candidate performs foreground
  publication and restart work, with four fsync and six fdatasync calls; the
  control handlers perform zero and two. Their SQLite VM/statement work is
  unchanged (1,528,000 VM steps per phase). Foreground pwrite counts increase
  from 6,539 to 7,073 in fresh repetition 0. Median handler CPU time rises from
  153.334 to 261.277 ms, while sampled request frequency is 2.298 GHz in both.
  Thus this particular regression cannot be dismissed as slower sampled CPU
  frequency alone.
- For 512 random writes, the candidate performs 1,024 foreground checkpoint
  API calls instead of 512: the extra status query repeats admission's query.
  Total hook time rises from 0.129 to 0.612 ms, too small to explain the whole
  regression. Its final barrier rises from 14.263 to 31.294 ms. VM and statement
  counts remain 224,448 and 10,230. Fresh repetition 0 performs 10,715 versus
  10,714 foreground pwrite calls, so that count is not identical in this
  experiment.
- Metadata phases also have different sampled CPU states: fresh rename runs at
  a median runtime-weighted 2.123 GHz in the control and 1.308 GHz in the
  candidate. Runqueue waits are small. This is a sampled association, not proof
  that all residual differences are caused by frequency or unrelated load.

The diagnosis retains seconds-scale tails in both variants: the after-GC
control's first write takes 6.556 s with a 5.216 s application request; candidate
repetition 3 random-write fsync takes 1.295 s. These separately instrumented
samples neither invalidate nor replace the frozen plain comparison.

The evidence rules out adopting this threshold-only CLEAN deferral. A follow-up
should preserve timely CLEAN-tail progress and test bounded worker-side wake
batching without another foreground status query. It should also include the
preceding budget implementation as a causal control while retaining the
original no-hint adoption gates. Reducing worker counts alone is insufficient.

## Validation and retained evidence

The final installed candidate passes 154 related Python tests with one explicit
unsupported production-v1 migration skip. Control and candidate each pass all
11 C++ component suites; the candidate also passes all 11 under ASan/UBSan.
The dedicated helper report contains 66 executed Release/sanitized cases,
including three expected negative-control failures, plus two sanitizer helper
compilations. No broad release/Ray suite was run.

Two initial preparation failures remain reviewable. The first used the system
SQLite header when building a helper, which lacks the SDK's NOOP constant;
adding the SDK include path fixes the build. The second passed the dedicated
cases but failed the old component test's per-commit wake expectation. The
runtime implementation is unchanged between that test failure and the final
build; the policy-specific expectation was updated as described above. Original
failed source trees, logs and reports are retained under `initial-0/` and
`initial-1/` and in the local manifest. Performance gates were not changed.

[The audit](audit.json) checks all 64 mounts' hashes, format 1004, SQLite
`quick_check`, payload lengths, stopped daemons, absent mounts and removed owned
data directories. It verifies disjoint windows and build/test timing, default
source/module/SDK identity, and test cleanup. The
[reconstruction proof](reproducer-proof.json) applies both patches with fuzz 0
and matches all 35 source inputs. Published arrays reconstruct the exact parsed
measurements and analyzer decision without rounding. [The manifest](manifest.json)
indexes published files and local full request vectors, CPU/host samples,
generated source, installed modules, test executables and libraries. Large
database/payload fixtures are removed after validation, outside timed phases.
Each published JSON is below 500 KiB.

## Reproduce

Use CPython 3.12 and the frozen source/build roots from the preceding reports.
The candidate installs noneditably in its own Release incremental build; the
diagnostic libraries are built separately before measurements.

```bash
REPO=/home/kaka/astrovela_vane-fs
BENCH="$REPO/vane_fs/benchmarks/staged_payload/checkpoint_coalescing"
BASE="$REPO/vane_fs/build/external-partial-20261011T083153Z"
BUDGET="$REPO/vane_fs/build/writeback-budget-20261011T165013Z"
OUT="$REPO/vane_fs/build/checkpoint-coalescing-new"
DIAGNOSTIC_OUT="$REPO/vane_fs/build/coalescing-diagnostic-new"
TEST_SUPPORT=/home/kaka/astrovela_vane-batch-udf-20260930/.venv-gpu/lib/python3.12/site-packages

"$REPO/.venv/bin/python" "$BENCH/prepare.py" "$OUT" \
  --parent "$BASE" --budget-parent "$BUDGET" \
  --sqlite-prefix "$REPO/vane_fs/vcpkg_installed/x64-linux-release" \
  --fuse-prefix "$REPO/.cache/fuse-sdk/root/usr" --test-support "$TEST_SUPPORT"
"$REPO/.venv/bin/python" "$BENCH/validate.py" "$OUT"
"$REPO/.venv/bin/python" "$BENCH/prepare_diagnostic.py" "$DIAGNOSTIC_OUT" \
  --parent "$OUT" --sqlite-prefix "$REPO/vane_fs/vcpkg_installed/x64-linux-release" \
  --fuse-prefix "$REPO/.cache/fuse-sdk/root/usr"
"$REPO/.venv/bin/python" "$BENCH/run.py" "$OUT" --medium tmpfs
"$REPO/.venv/bin/python" "$BENCH/run.py" "$OUT" --medium ext4
"$REPO/.venv/bin/python" "$BENCH/../partial_sync/analyze_writeback.py" "$OUT"
"$REPO/.venv/bin/python" "$BENCH/run_diagnostic.py" "$DIAGNOSTIC_OUT"
"$REPO/.venv/bin/python" "$BENCH/../request_costs/analyze.py" "$DIAGNOSTIC_OUT"
"$REPO/.venv/bin/python" "$BENCH/../request_costs/analyze_cpu.py" "$DIAGNOSTIC_OUT"
"$REPO/.venv/bin/python" "$BENCH/analyze_diagnostic.py" "$DIAGNOSTIC_OUT"
"$REPO/.venv/bin/python" "$BENCH/publish.py" "$OUT" --diagnostic "$DIAGNOSTIC_OUT"
```

`publish.py` refuses changed frozen inputs, missing validation, incomplete
cleanup or oversized JSON. The recorded failed preparations can additionally
be supplied with `--failed-preparations`; they are not replayed or discarded.

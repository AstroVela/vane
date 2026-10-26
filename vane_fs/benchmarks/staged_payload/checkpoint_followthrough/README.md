# Follow-through after lookup and CLEAN coalescing

This investigates the two hypotheses left by the
[previous comparison](../clean_coalescing/README.md), then implements a narrower
CLEAN scheduling policy in an isolated format-1004 candidate. **The bounded
candidate is not adopted:** it improves the targeted 64 MiB write but still
fails throughput and tail checks. Production source and the installed default
module remain unchanged.

## Lookup moves the wait into rename

The first diagnostic window compares the commit-progress prototype with its
read-first lookup candidate. It runs four alternating AB/BA pairs in each of
fresh and independently prepared post-GC workspaces: 16 mounts and 2,048 paired
lookup/rename operations. The serial FUSE trace verifies one lookup before each
rename and retains the pair's entire elapsed interval, CPU and BEGIN costs.

Median accumulated BEGIN time per 128-operation batch:

| State | Operation | Before lookup change | After lookup change |
| --- | --- | ---: | ---: |
| Fresh | Lookup | 9.08 ms | 0.23 ms |
| Fresh | Rename | 0.79 ms | 9.06 ms |
| Post-GC | Lookup | 8.94 ms | 0.19 ms |
| Post-GC | Rename | 0.75 ms | 8.96 ms |

All 13 pairs with at least 0.5 ms of BEGIN time overlap actual RESTART syncs:
seven old-policy lookups and six candidate renames. The long BEGIN spans are
about 8.2–8.3 ms; each pair overlaps WAL and database synchronization at the
`wal_restart` site. This confirms that removing the lookup's writer lock does
not eliminate the subsequent mutation's lock contention.

The complete rename phase is approximately 72→75 ms fresh and 68→69 ms after
GC. These are instrumented diagnostic costs, not new adoption throughput.
[Paired samples](lookup-diagnostic/rename-pairs.json) and
[findings](lookup-diagnostic/findings.json) retain every pair and overlapping
checkpoint/sync interval. Nested stages and concurrent sync durations are not
additive. The trace does not count individual SQLite busy-handler sleeps.

## CLEAN can miss an opportunity to backfill a large prefix

The second independent 16-mount diagnostic compares lookup-only with the previous
5 ms CLEAN coalescing candidate. Its sources differ only in `checkpoint.hpp`.
The recorder now retains every checkpoint call and each worker decision's
generation size, copied frames, OPEN/CLEAN state, retry state, deferral and
remaining deadline. Removing diagnostic additions recovers the original sources
byte-for-byte.

The hypothesized large-generation deferral occurs: 2,060 observations in 112
episodes defer CLEAN maintenance of a WAL generation at least 16 MiB in size.
However, the subsequent 64 MiB write regression has a more specific mechanism.
An early CLEAN prefix of 2,948 frames, only eight already copied, contains about
11.6 MiB of outstanding WAL data. The old predicate considers it small because
it is below 16 MiB. During its 5 ms delay, the next external write turns the
workspace OPEN. The worker then defers publication until the uncopied OPEN
budget reaches 16 MiB, forcing payload publication during the write stream.

In a retained fresh sample, the deferred CLEAN decision is followed by an OPEN
publication that spends 53.6 ms syncing payload and 44.0 ms syncing the database.
A foreground write lasts 143.7 ms and overlaps that worker; its publication-lock
stage alone is 66.4 ms. The control first backfills the CLEAN prefix and avoids
that midstream payload publication. These inclusive measurements describe
overlapping work and must not be summed.

Across the window, median foreground publication-lock time in the 64 MiB phase
rises from about 0.006 ms to 74.8 ms fresh and 67.1 ms after GC. Admission remains
below 0.4 ms: this is not principally the 64 MiB admission limit. The 64 MiB phase
median grows from 530→644 ms fresh and 581→659 ms after GC. The same candidate
reduces small-file background database sync medians from 107.5→17.5 fresh and
97.5→22 after GC, explaining why simply removing all coalescing loses useful work.
See the retained `midstream_publications` and full intervals in
[CLEAN findings](clean-diagnostic/findings.json).

## Bounded candidate

[budget.patch](budget.patch) permits the fixed 5 ms CLEAN window only when:

- The entire WAL generation is below 16 MiB, keeping large generations eligible
  for prompt backfill/reuse.
- Uncopied WAL bytes are at most 256 KiB, preventing a multi-MiB CLEAN prefix
  from waiting while the next external batch opens.

The 256 KiB cap matches the prototype's external allocation unit. It was frozen
before validation and measurement, not swept over the observed results. OPEN
publication continues to use its existing 16 MiB uncopied budget; the 64 MiB
admission threshold, FULL durability, non-sliding deadline, final-commit progress,
reader retry and Stop/Start handling are unchanged. A timer bounds scheduling
deferral, not real I/O or lock duration.

The new regression independently constructs a large generation with a small
tail and a small generation with a 1 MiB tail. Both old-policy negative controls
perform two status probes before their first PASSIVE call and fail the urgency
assertion. Both new-policy cases perform one and pass, including ASAN/UBSAN.
The existing 80-commit small FULL burst still takes 16 PASSIVE calls. This helper
count is not a FUSE sync count.

## Plain comparison and decision

The [plain ablation](budget-ablation/summary.json) compares the previous 5 ms
CLEAN candidate against the two-budget candidate. It uses six alternating AB/BA
pairs in each state, 24 mounts total, and the unchanged original workload and
gates. Each component's own installed module performs GC.

| Workload | Fresh median change | Post-GC median change |
| --- | ---: | ---: |
| Subsequent 64 MiB write | +17.1% | +24.0% |
| Small-file fsync | +5.8% | −4.5% |
| 4 KiB random write | −16.4% | −3.2% |
| Rename | −10.3% | +9.5% |

The 64 MiB write rates recover from 99.39→116.36 MiB/s fresh and
91.47→113.39 MiB/s after GC; aggregate improvements are 21.0% and 24.6%.
However, five ordinary fresh-workspace checks fail: the 256 MiB write's
aggregate/tail, read throughput, random-write throughput, mkdir aggregate and
rename throughput. The original combined-large-write goal also fails in both
states. The fresh combined aggregate ratio is 0.763 despite five of six paired
wins, and the post-GC ratio is 1.061 with six wins. All samples, including the
candidate's slow 256 MiB request, are retained.

This rejects the candidate without extending a failing ablation to another
original-control or tmpfs window. The two new restrictions were measured together;
the remaining random-write/rename regression cannot yet be assigned to either
one individually. The next comparison should isolate the 256 KiB uncopied cap
from the whole-generation cap. The larger lock-contention target remains the
RESTART sync itself, since lookup merely transfers its wait to rename.

## Validation and retention

The candidate passes 154 installed Python tests with one unsupported format-v1
migration skip, 12 native suites, 12 ASAN/UBSAN suites, and 42 Release/sanitized
policy, progress, restart and budget helpers. Twelve native control suites also
pass. The two expected negative cases reproduce the old timer behavior. The
extended recorder passes Release and ASAN/UBSAN tests for 25 actual sync calls,
six descriptor types, descriptor reuse, errors, concurrency, nested scopes,
worker decision fields and checkpoint intervals. Only related tests run.

The two diagnostic windows retain all 32 mounts, 4,096 rename pairs, 17,973
individual syncs and 187,889 checkpoint calls. All 217 syncs lasting at least
100 ms remain, including six lasting at least one second; the maximum is 3.95 s.
The plain comparison adds 24 mounts, for 56 total. The same pinned SQLite 3.53.2 `fsync` SDK, FUSE 3.14,
original application buffer offset 48, request sizes, barriers, content hashes
and workload are used. Instrumentation is separate from plain performance
comparisons. No cache eviction or frequency/affinity tuning is used; no owned
build or test overlaps a measured window.

[audit.json](audit.json) verifies source, binary, module and recorder identities,
all retained samples, and cleanup after each owned daemon stops and unmounts.
Large raw traces, CPU/host samples and build products remain local with sizes and
hashes. Test preflights are retained: the recorder test initially guessed the
NOOP enum incorrectly, then was corrected to the pinned header's value of -1;
no runtime policy or measurement changed in that correction. The exact executed
C++ regression source is retained before a formatting-only lambda line wrap.

## Reproduction

Run from the repository root with the preceding frozen artifacts available:

```bash
python3 vane_fs/benchmarks/staged_payload/checkpoint_followthrough/prepare.py OUTPUT \
  --diagnostic-parent vane_fs/build/checkpoint-request-costs-20261012T163511Z \
  --candidates vane_fs/build/lookup-read-20261013T012000Z
python3 vane_fs/benchmarks/staged_payload/checkpoint_followthrough/run.py OUTPUT \
  --baseline progress --updated lookup
python3 vane_fs/benchmarks/staged_payload/checkpoint_followthrough/run.py OUTPUT \
  --baseline lookup --updated coalesced
python3 vane_fs/benchmarks/staged_payload/checkpoint_followthrough/analyze.py OUTPUT/progress-lookup
python3 vane_fs/benchmarks/staged_payload/checkpoint_followthrough/analyze.py OUTPUT/lookup-coalesced
python3 vane_fs/benchmarks/staged_payload/checkpoint_followthrough/candidate.py OUTPUT/budget \
  --parent vane_fs/build/lookup-read-20261013T012000Z
python3 vane_fs/benchmarks/staged_payload/checkpoint_followthrough/validate.py OUTPUT/budget
```

For plain comparisons, reuse `../clean_coalescing/screen.py` with the previous
`checkpoint-progress-20261012T153141Z/fuse-screen` template and the new
`OUTPUT/budget/tests.json` validation. The bounded component is
`OUTPUT/budget/updated`; the ablation control is the preceding `coalesced`
component. The `clean-ablation` role names this scheduling comparison; the plan
records the exact component paths and hashes. Original gates are unchanged.

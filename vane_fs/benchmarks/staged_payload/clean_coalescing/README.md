# Small CLEAN maintenance: implemented, not adopted

This implements the second proposal from the
[request-cost investigation](../checkpoint_request_costs/README.md), on top of
the [read-first lookup candidate](../../lookup_reads/README.md). **The candidate
is not adopted:** small-file fsync improves, but complete FUSE comparisons still
fail the original gates. Production sources are restored; both implementations
remain reproducible isolated patches.

## Policy and checks

[clean.patch](clean.patch) adds a fixed 5 ms scheduling window for a CLEAN tail
whose uncopied WAL bytes are below 16 MiB. The deadline starts at the first
observed request and is never extended by later commits. New commits can wake
the worker to recheck urgency. The timer holds no publication/checkpoint lock.
Large uncopied tails and retries bypass this scheduling deferral; OPEN publication,
FULL barriers, WAL-hook behavior, optional RESTART, and 16/64 MiB admission
thresholds remain unchanged. The bound is on scheduling deferral, not actual I/O.

The deadline survives Stop/Start. A final small commit still progresses without
another commit; a pinned reader is retried after release. Existing cold/reset/
live/reader/stream, restart/error/crash and publication tests pass. In the new
80-commit FULL burst test, the previous policy fails the “fewer than 40 passes”
assertion, while this candidate performs 16 PASSIVE passes in both Release and
ASAN/UBSAN. This is a helper workload, not a measured FUSE sync count.

This differs from the rejected global 2 ms scheduling experiment: the new delay
is selected in the worker for small CLEAN maintenance. Its different scope does
not exempt it from the original throughput, tail and lifecycle checks.

## Three independent plain windows

Each uses six alternating AB/BA pairs for both fresh and independently prepared
post-GC workspaces: 24 mounts per window, 72 total. The original plain ext4
workload, source data, buffer offset 48, final barriers and gate constants are
unchanged. SQLite 3.53.2 uses its original `fsync` primitive; FUSE is 3.14.
Each variant's own installed module performs GC. No owned build/test overlaps a
window; no cache eviction or frequency/affinity tuning is used.

| Window | Control | Candidate |
| --- | --- | --- |
| [Lookup ablation](lookup-ablation/summary.json) | Previous commit-progress prototype | Lookup only |
| [CLEAN ablation](clean-ablation/summary.json) | Lookup only | Lookup + CLEAN window |
| [Original control](original-control/summary.json) | Original partial-write control | Lookup + CLEAN window |

The two CLEAN-ablation sources differ only in `src/checkpoint.hpp`. Complete
patches reconstruct all three measured candidates byte-for-byte, including the
native regression tests. The isolated format remains indexed format 1004.

## Results and decision

The CLEAN ablation improves median small-file fsync throughput **6.2% fresh and
9.2% after GC**; aggregate improvements are 7.1% and 10.1%. However, subsequent
64 MiB write medians fall 9.0% and 15.3%. Fresh rename median falls 6.3%; post-GC
rename aggregate falls 10.6%. Fresh random-write aggregate and additional
directory/tail checks also fail. Improved checkpoint counts alone are insufficient.

Against the original control, combined large-write aggregate throughput is
114.10→102.91 MiB/s fresh (−9.8%, despite 5/6 paired wins) and 109.85→135.09 MiB/s
after GC (+23.0%, 6/6 wins). Fresh random-write and rename medians fall 16.0%
and 19.6%; after GC they fall 14.0% and 21.8%. Small-file fsync is roughly level
with the original control. Eight original checks fail, including large-write
tails. The ext4 subset is not full adoption evidence; tmpfs is not added after
this rejection.

The independent [production-format evaluation](../../lookup_reads/production/decision.json)
also rejects lookup-only adoption. This is why the common lookup change is kept
as a patch instead of entering the default implementation.

All request samples and barriers remain in the split raw reports. Slow requests
occur on both sides; medians and aggregate rates can disagree substantially.
No sample is removed, no gate is recalibrated, and no tail is assigned to unrelated
machine load. These windows establish rejection and observed costs, not the
precise cause of every long syscall.

## Next investigation

The current CLEAN predicate limits **uncopied bytes**, so it can also delay the
nearly completed tail of a large WAL generation. That may delay RESTART/reuse
before the next large write. Test this hypothesis by recording generation size,
uncopied bytes and the subsequent writer's wait; then consider restricting
coalescing to generations that are themselves below 16 MiB. Do not assume the
5 ms timer alone explains seconds-long tails.

For lookup, record the missing-name read and following rename/write admission
as one operation. Removing one lock acquisition can move the wait to the next
writer, which the current plain measurements cannot distinguish.

## Validation and retained evidence

New checks: 132 installed Python tests on the production lookup candidate;
154 on the combined isolated candidate, with one unsupported migration skip;
seven production and 12+12 isolated C++ suites; seven production and 12 isolated
ASAN/UBSAN suites; 38 Release/sanitized policy/progress helpers and one expected
negative helper. The original-source missing-lookup regression also fails as
expected. After restoring defaults, six original native suites and 132 installed
Python tests pass again.
No broad base/Ray suite runs.

[audit.json](audit.json) verifies 48 native runs, all 96 FUSE mounts across four
windows, measured source/binary/module identities, exact patch reconstruction,
lossless raw-result splits and owned-data cleanup. Daemons stop and unmount before
database/WAL/payload deletion, outside measured phases. Logs, configurations,
content hashes, all latency samples and host-monitor identities are retained.
Large build artifacts and the archived measured production candidate remain local.
The default source is unchanged; its non-editable rebuild's new module hash is
recorded explicitly.

The original executed builder and narrower patch are archived under `executed`:
the published lookup patch was expanded to include tests when the runtime change
was rejected. Applying the complete published patch reconstructs exactly the same
measured candidates; this is checked rather than inferred.

## Reproduction

From the repository root, with the preceding frozen parent available:

```bash
python3 vane_fs/benchmarks/staged_payload/clean_coalescing/prepare.py OUTPUT \
  --parent vane_fs/build/checkpoint-progress-20261012T153141Z
python3 vane_fs/benchmarks/staged_payload/clean_coalescing/validate.py OUTPUT
```

Then use `screen.py OUTPUT_WINDOW --template TEMPLATE --validation OUTPUT/tests.json
--baseline CONTROL_COMPONENT --updated CANDIDATE_COMPONENT --role ROLE`.
`TEMPLATE` is the preceding verified `fuse-screen`; roles are `lookup-ablation`,
`clean-ablation` and `original-control`, with controls/candidates from the table.
The default-format reproduction is documented with the lookup patch.

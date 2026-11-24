# Matched paced-parent versus BEGIN/RESTART gate

The isolated gate fixes the controlled busy-backoff scenario, but this matched
ordinary window **does not establish a stable overall performance gain**.
Complete-large-phase median throughput changes **-2.96% fresh and +0.73%
after GC**. The candidate has **4.666 s and 6.930 s final fsync calls**.
Seven regression checks and the two original large-write improvement targets
fail. No implementation is promoted.

This separates the incremental gate from the
[previous combined candidate](../begin_restart_gate/README.md), whose
contemporaneous control had no paced hints. Its earlier 12.7%/13.4% median
large-phase gains cannot be credited to the mutex from that comparison.

## Comparison frozen before measurement

Both arms already contain format 1005, identical extents, 4 MiB WRITE-only
range claims, a 16 ms completion delay, the 64 MiB hint trigger, retained
payload barrier descriptors and balanced FULL pauses. They share the original
fsync SQLite SDK and the 16/64 MiB WAL start/admission budgets. Publication,
FULL bypass, payload-to-WAL-to-CLEAN ordering and durability settings are
unchanged.

[comparison.json](comparison.json) identifies both native implementations.
Of 44 frozen source files per arm, **42 are byte-identical**. Only
checkpoint.hpp and workspace.cpp differ, and the gate transformation reverses
exactly to the paced parent. Existing native libraries, mount binaries and
Python modules are reused by source/binary identity; no native rebuild occurs.
Each GC preparation imports its own arm's verified native module.

The primary window uses six alternating AB/BA pairs per fresh/post-GC state,
**24 mounts**. The supplementary window uses two pairs per state, four
consecutive 256+64 MiB cycles per live workspace, **8 mounts**. Workload bytes
are copied unchanged. The runner differs from the prior runner only in its
comparison description. The original 22 checks and their thresholds are
unchanged, including the 10% combined-large-write target. No relaxed adoption
criterion is introduced for the gate-only comparison.

## Primary matched results

Rates below are medians of six complete samples. Large writes include final
fsync and close; the combined phase contains 256+64 MiB. Throughput is MiB/s
except small files, which use files/s. Wins compare candidate and parent
within the same AB/BA pair.

| Workload | State | Paced parent | Plus gate | Candidate wins |
| --- | --- | ---: | ---: | ---: |
| Combined 320 MiB | Fresh | 261.95 | 254.21 | 1/6 |
| Combined 320 MiB | Post-GC | 254.48 | 256.34 | 4/6 |
| 256 MiB write | Fresh | 294.00 | 277.83 | 0/6 |
| 256 MiB write | Post-GC | 280.61 | 279.11 | 1/6 |
| 64 MiB write | Fresh | 187.32 | 190.30 | 5/6 |
| 64 MiB write | Post-GC | 181.99 | 193.59 | 5/6 |
| 4 KiB random write | Fresh | 9.31 | 8.13 | 2/6 |
| 4 KiB random write | Post-GC | 8.64 | 8.71 | 4/6 |
| Small file plus fsync | Fresh | 250.62 | 257.80 | 4/6 |
| Small file plus fsync | Post-GC | 250.40 | 257.22 | 4/6 |

The 64 MiB case improves **1.59% fresh and 6.37% post-GC**. Small-file median
rates improve **2.86% and 2.72%**, with four of six wins each. Fresh 256 MiB
throughput declines **5.50%**, and fresh random writes decline **12.61%**.
These observed tradeoffs do not support a general gate-only speedup.

Candidate large-phase aggregates are much worse than their medians because
the two multi-second final barriers remain in the data:

| Primary phase | Paced aggregate MiB/s | Gate aggregate MiB/s | Long candidate call |
| --- | ---: | ---: | --- |
| Fresh 320 MiB | 262.55 | 162.50 | Rep 1, 256 MiB final fsync: 4.666 s |
| Post-GC 320 MiB | 255.24 | 137.33 | Rep 3, 256 MiB final fsync: 6.930 s |

Median 256 MiB final fsync is **476.772 to 474.414 ms** fresh and **460.995
to 471.267 ms** after GC. Ordinary application timings identify the final
fsync interval but do not identify whether an individual pause waits on the
new mutex, publication, payload/WAL/fence sync, a checkpoint or device I/O.

Seven original regression checks fail:

| Check | Retained failure |
| --- | --- |
| Fresh 256 MiB | Median -5.50%, aggregate -46.62%; pooled write p99 22.235 to 31.926 ms |
| Fresh random write | Median -12.61%, aggregate -14.43% |
| Fresh rename | Pooled p99 0.725 to 3.023 ms despite higher median throughput |
| Post-GC 256 MiB | Aggregate -54.38%; pooled write p99 24.032 to 29.424 ms |
| Post-GC mkdir | Aggregate -12.77% |
| Post-GC rename | Aggregate -7.44% |
| Post-GC rmdir | Aggregate -13.33% |

Both combined-large-write improvement targets also fail, making **9 of 22**
failed checks. The nonpassing 10% targets alone do not explain this result:
there are seven additional regressions under the unchanged allowances.

## Retained bursts and supplementary cycles

Fresh small-file pooled p99 improves **149.700 to 6.386 ms**, and aggregate
throughput improves **78.41 to 255.45 files/s**. A retained parent round
dominates those numbers: rep 5 takes **7.065 s for 128 files**, with **72
operations at least 20 ms**, including 13 at least 100 ms. Parent rep 2 also
contains a **259.572 ms** operation. The matched small-file median gain is
only 2.86%; application-only timings do not establish which native waits
produced the parent burst. No operation or round is dropped.

Supplementary fresh aggregate is **168.20 to 257.83 MiB/s**, two of two wins,
including a **5.422 s parent final fsync** in rep 1 cycle 3. Supplementary
post-GC aggregate is **254.27 to 253.10 MiB/s**, zero of two candidate wins.
The smaller supplementary window retains all cycles and does not replace
the six-pair primary checks. Multi-second barriers occur in both versions.

## Validation, retention and next evidence

All **32 complete workloads and 112 large-file hashes pass**, along with
retained-GC-guard hashes, read/overwrite, namespace, format/fence and SQLite
integrity checks. Measurement is ordinary uninstrumented FUSE. Only the
existing host monitor runs; builds, tests and analysis are separate.

The existing gate binaries are reexecuted in Release and ASan/UBSan:
**seven candidate cases and two parent negative controls per build**, four
native executions. The parent still reproduces actual default-handler busy
sleeps under a held real WAL writer lock, and the candidate has zero sleeps
in that controlled scenario. PASSIVE concurrency, try-lock skip/progress,
post-BEGIN release, failed BEGIN retry and foreign-writer Busy remain valid.
Component, Python, fault and FULL-pause validation results are inherited by
the exact frozen native source/binary identities.

[findings.json](findings.json) contains every paired effect and failed check.
[final-checks.json](final-checks.json) audits frozen/default identities,
measurement separation and **36 cleaned owned data directories**: 32 FUSE
directories and four related native-test directories. [audit.json](audit.json)
retains every raw request/phase vector, host sample, log, proof, source
snapshot and cleanup hash losslessly. Concatenate an entry's ordered gzip
parts and decompress once. Every published file is below 500 KiB.

Production/default SDK, native algorithms and durability remain unchanged.
No global cache eviction, CPU/device tuning, load-based sample removal or
unowned process termination occurs. Owned processes stop before cleanup;
data cleanup is outside application timers. Sources, SDKs, environments and
incremental builds remain for reproduction.

The next useful evidence is native-stage attribution of the final-fsync
pauses and fresh write regressions, covering the new gate alongside
RESTART/publication and actual payload/WAL/fence barriers. Keep observer CPU
overhead measured separately. The current application timings support
neither a unique cause for those pauses nor an independent production
backport based on overall performance.

~~~bash
here=vane_fs/benchmarks/staged_payload/begin_gate_matched
out="$PWD/vane_fs/build/begin-gate-matched-repro"
python3 "$here/prepare.py" "$out" \
  --component "$PWD/vane_fs/build/begin-restart-gate-20261019"
python3 "$here/run.py" "$out"
python3 "$here/analyze.py" "$out"
git add "$here"
python3 "$here/checks.py" "$out"
python3 "$here/publish.py" "$out"
~~~

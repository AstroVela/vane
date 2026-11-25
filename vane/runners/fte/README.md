# Scan task planning

Ray FTE uses `ScanBatchSplitAssigner` for unordered fragments with a single
scan source and no exchange sources. This includes CSV and NDJSON scans.
Ordered, multi-source, broadcast, and hash-distributed fragments retain their
existing assigners.

The planning unit is all split events for a fragment in one scheduler submission
batch. This is **not a whole-query metadata barrier**: the event source currently
chunks submissions into 8 events by default. A transport event can contain many
splits. The assigner additionally caps each planning window at 4,096 splits.
At the end of a submission, even a partial window is flushed, so scan work can
start without waiting for source exhaustion. No new file metadata requests are
needed.

Reader-produced splits are indivisible. A split may represent a safe byte range
or a whole unsplittable file. The planner uses its estimated bytes, falling back
to 64 MiB when unknown; known zero bytes remain zero. Estimates represent input
work, not worker memory reservations or guaranteed execution time.

For each window:

1. Sum estimated work excluding indivisible splits larger than 256 MiB, which
   get standalone tasks. Aim for four regular tasks per worker slot, using
   `VANE_DISTRIBUTED_WORKER_SLOTS` (one slot if absent or invalid). Limit that
   target by a 64 MiB minimum average task size and a 256 MiB maximum task size.
2. Sort splits by descending cost, preserving arrival order for ties. Separate
   catalog, host, and remote-access requirements. For a split with several
   eligible hosts, select the least-loaded eligible host deterministically.
3. Within each compatible group, preallocate task bins based on the target size
   and split-count limit. Assign each split to the lightest task with capacity.
   Create another task if necessary. Standalone oversized splits do not count
   toward these bins. The minimum size is a target; locality, indivisible splits,
   and batch boundaries can produce smaller tasks.
4. Assign partition IDs in descending order of final task size across all
   compatible groups, including standalone oversized tasks. Seal each task
   with its complete split list. Existing worker admission
   dispatches queued tasks when resources become available. Retry replays the
   same descriptor and stable split IDs; task planning does not bind a worker.

`VANE_FTE_DYNAMIC_SCAN_MAX_SPLITS_PER_PARTITION` continues to limit split count
per task (2,048 by default). Four waves is an initial heuristic, not a measured
universal optimum. Tune with mixed file sizes, slow workers, and representative
storage; more waves offer more opportunities for dynamic load balancing, but
increase task overhead. Because windows are bounded, a tiny input arriving over
many submission batches may produce more tasks than a whole-scan plan would.

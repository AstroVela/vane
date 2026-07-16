from __future__ import annotations

from duckdb.runners.fte.backend import (
    FragmentTemplateStore as FragmentTemplateStore,
    ResourceSnapshotProvider as ResourceSnapshotProvider,
    TaskResultHandle as TaskResultHandle,
    TaskResultPoll as TaskResultPoll,
    TaskResultState as TaskResultState,
    WorkerHandle as WorkerHandle,
    WorkerManagerBackend as WorkerManagerBackend,
)
from duckdb.runners.fte.fte_attempts import (
    ExecutionClassTransition as ExecutionClassTransition,
    ReadyTask as ReadyTask,
    RevokedAttempt as RevokedAttempt,
    ScheduledAttempt as ScheduledAttempt,
)
from duckdb.runners.fte.fte_config import (
    FTE_WORKER_RUNTIME as FTE_WORKER_RUNTIME,
)
from duckdb.runners.fte.fte_config import (
    FteWorkerAdmissionConfig as FteWorkerAdmissionConfig,
)
from duckdb.runners.fte.fte_config import (
    fte_split_queue_max_buffered_splits as fte_split_queue_max_buffered_splits,
)
from duckdb.runners.fte.fte_config import (
    fte_status_wait_timeout_s as fte_status_wait_timeout_s,
)
from duckdb.runners.fte.fte_descriptor import (
    FteTaskUpdateRequest as FteTaskUpdateRequest,
    TaskDescriptor as TaskDescriptor,
    TaskDescriptorStorage as TaskDescriptorStorage,
)
from duckdb.runners.fte.fte_exchange import (
    FteExchangeSourceOutputSelector as FteExchangeSourceOutputSelector,
    FteExchangeTracker as FteExchangeTracker,
    SpoolingExchangeManager as SpoolingExchangeManager,
    collect_spooling_output_stats as collect_spooling_output_stats,
    derive_exchange_sink_instance_for_attempt as derive_exchange_sink_instance_for_attempt,
)
from duckdb.runners.fte.fte_execution import (
    FteFragmentExecution as FteFragmentExecution,
    FteWorkerControlFailure as FteWorkerControlFailure,
    FteWorkerReservationUnavailable as FteWorkerReservationUnavailable,
)
from duckdb.runners.fte.fte_scheduler import (
    FteAttemptStatusWatcher as FteAttemptStatusWatcher,
    FteEventDrivenTaskSource as FteEventDrivenTaskSource,
    FteEventHandlers as FteEventHandlers,
    FteQueryScheduler as FteQueryScheduler,
    FteSchedulerRegistry as FteSchedulerRegistry,
    FteSchedulerStats as FteSchedulerStats,
    FteWorkerCommandExecutor as FteWorkerCommandExecutor,
)
from duckdb.runners.fte.fte_split_assigner import (
    ArbitrarySplitAssigner as ArbitrarySplitAssigner,
    AssignmentResult as AssignmentResult,
    HashSplitAssigner as HashSplitAssigner,
    HashTaskPartition as HashTaskPartition,
    NodeRequirements as NodeRequirements,
    PartitionInfo as PartitionInfo,
    PartitionUpdate as PartitionUpdate,
    SingleSplitAssigner as SingleSplitAssigner,
    SplitAssigner as SplitAssigner,
)
from duckdb.runners.fte.fte_state import (
    FtePartitionState as FtePartitionState,
    FteTaskExecutionClass as FteTaskExecutionClass,
    FteTaskState as FteTaskState,
    fte_task_execution_class_from_metadata as fte_task_execution_class_from_metadata,
    fte_task_execution_class_metadata_present as fte_task_execution_class_metadata_present,
)
from duckdb.runners.fte.fte_types import (
    FteSplit as FteSplit,
    FteTaskAttemptId as FteTaskAttemptId,
    FteTaskId as FteTaskId,
    _check_non_negative as _check_non_negative,
    validate_fte_status_identity as validate_fte_status_identity,
)
from duckdb.runners.fte.fte_worker_runtime import (
    FteTaskExecution as FteTaskExecution,
    FteWorkerTaskManager as FteWorkerTaskManager,
    materialize_task_inputs as materialize_task_inputs,
)

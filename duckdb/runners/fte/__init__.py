# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from duckdb.runners.fte.backend import (
    FragmentTemplateStore as FragmentTemplateStore,
)
from duckdb.runners.fte.backend import (
    ResourceSnapshotProvider as ResourceSnapshotProvider,
)
from duckdb.runners.fte.backend import (
    TaskResultHandle as TaskResultHandle,
)
from duckdb.runners.fte.backend import (
    TaskResultPoll as TaskResultPoll,
)
from duckdb.runners.fte.backend import (
    TaskResultState as TaskResultState,
)
from duckdb.runners.fte.backend import (
    WorkerHandle as WorkerHandle,
)
from duckdb.runners.fte.backend import (
    WorkerManagerBackend as WorkerManagerBackend,
)
from duckdb.runners.fte.fte_attempts import (
    ExecutionClassTransition as ExecutionClassTransition,
)
from duckdb.runners.fte.fte_attempts import (
    ReadyTask as ReadyTask,
)
from duckdb.runners.fte.fte_attempts import (
    RevokedAttempt as RevokedAttempt,
)
from duckdb.runners.fte.fte_attempts import (
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
)
from duckdb.runners.fte.fte_descriptor import (
    TaskDescriptor as TaskDescriptor,
)
from duckdb.runners.fte.fte_descriptor import (
    TaskDescriptorStorage as TaskDescriptorStorage,
)
from duckdb.runners.fte.fte_exchange import (
    FteExchangeSourceOutputSelector as FteExchangeSourceOutputSelector,
)
from duckdb.runners.fte.fte_exchange import (
    FteExchangeTracker as FteExchangeTracker,
)
from duckdb.runners.fte.fte_exchange import (
    SpoolingExchangeManager as SpoolingExchangeManager,
)
from duckdb.runners.fte.fte_exchange import (
    collect_spooling_output_stats as collect_spooling_output_stats,
)
from duckdb.runners.fte.fte_exchange import (
    derive_exchange_sink_instance_for_attempt as derive_exchange_sink_instance_for_attempt,
)
from duckdb.runners.fte.fte_execution import (
    FteFragmentExecution as FteFragmentExecution,
)
from duckdb.runners.fte.fte_execution import (
    FteWorkerControlFailure as FteWorkerControlFailure,
)
from duckdb.runners.fte.fte_execution import (
    FteWorkerReservationUnavailable as FteWorkerReservationUnavailable,
)
from duckdb.runners.fte.fte_scheduler import (
    FteAttemptStatusWatcher as FteAttemptStatusWatcher,
)
from duckdb.runners.fte.fte_scheduler import (
    FteEventDrivenTaskSource as FteEventDrivenTaskSource,
)
from duckdb.runners.fte.fte_scheduler import (
    FteEventHandlers as FteEventHandlers,
)
from duckdb.runners.fte.fte_scheduler import (
    FteQueryScheduler as FteQueryScheduler,
)
from duckdb.runners.fte.fte_scheduler import (
    FteSchedulerRegistry as FteSchedulerRegistry,
)
from duckdb.runners.fte.fte_scheduler import (
    FteSchedulerStats as FteSchedulerStats,
)
from duckdb.runners.fte.fte_scheduler import (
    FteWorkerCommandExecutor as FteWorkerCommandExecutor,
)
from duckdb.runners.fte.fte_split_assigner import (
    ArbitrarySplitAssigner as ArbitrarySplitAssigner,
)
from duckdb.runners.fte.fte_split_assigner import (
    AssignmentResult as AssignmentResult,
)
from duckdb.runners.fte.fte_split_assigner import (
    HashSplitAssigner as HashSplitAssigner,
)
from duckdb.runners.fte.fte_split_assigner import (
    HashTaskPartition as HashTaskPartition,
)
from duckdb.runners.fte.fte_split_assigner import (
    NodeRequirements as NodeRequirements,
)
from duckdb.runners.fte.fte_split_assigner import (
    PartitionInfo as PartitionInfo,
)
from duckdb.runners.fte.fte_split_assigner import (
    PartitionUpdate as PartitionUpdate,
)
from duckdb.runners.fte.fte_split_assigner import (
    SingleSplitAssigner as SingleSplitAssigner,
)
from duckdb.runners.fte.fte_split_assigner import (
    SplitAssigner as SplitAssigner,
)
from duckdb.runners.fte.fte_state import (
    FtePartitionState as FtePartitionState,
)
from duckdb.runners.fte.fte_state import (
    FteTaskExecutionClass as FteTaskExecutionClass,
)
from duckdb.runners.fte.fte_state import (
    FteTaskState as FteTaskState,
)
from duckdb.runners.fte.fte_state import (
    fte_task_execution_class_from_metadata as fte_task_execution_class_from_metadata,
)
from duckdb.runners.fte.fte_state import (
    fte_task_execution_class_metadata_present as fte_task_execution_class_metadata_present,
)
from duckdb.runners.fte.fte_types import (
    FteSplit as FteSplit,
)
from duckdb.runners.fte.fte_types import (
    FteTaskAttemptId as FteTaskAttemptId,
)
from duckdb.runners.fte.fte_types import (
    FteTaskId as FteTaskId,
)
from duckdb.runners.fte.fte_types import (
    _check_non_negative as _check_non_negative,
)
from duckdb.runners.fte.fte_types import (
    validate_fte_status_identity as validate_fte_status_identity,
)
from duckdb.runners.fte.fte_worker_runtime import (
    FteTaskExecution as FteTaskExecution,
)
from duckdb.runners.fte.fte_worker_runtime import (
    FteWorkerTaskManager as FteWorkerTaskManager,
)
from duckdb.runners.fte.fte_worker_runtime import (
    materialize_task_inputs as materialize_task_inputs,
)

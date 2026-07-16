// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/distributed/plan/fte_split_queue.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/distributed/common_types.hpp"
#include "duckdb/parallel/interrupt.hpp"

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace duckdb {

class PhysicalPlan;

namespace distributed {

class FteSplitQueue {
public:
	enum class GetResult { SPLIT, BLOCKED, FINISHED, CANCELED };

	struct GetNextResult {
		GetResult state = GetResult::BLOCKED;
		TaskInput input;
		idx_t split_id = DConstants::INVALID_INDEX;

		bool HasSplit() const {
			return state == GetResult::SPLIT;
		}
	};

	void AddSplits(std::vector<TaskInput> inputs);
	void AddSplit(TaskInput input);
	void NoMoreSplits();
	void Cancel();
	GetNextResult TryGetNext();
	GetNextResult WaitForNext();
	bool RegisterBlockedTask(const duckdb::InterruptState &interrupt_state);

	idx_t BufferedSplits() const;
	idx_t BufferedBytes() const;
	idx_t SubmittedRows() const;
	idx_t SubmittedInputBytes() const;
	idx_t ConsumedRows() const;
	idx_t ConsumedInputBytes() const;
	idx_t SubmittedSplits() const;
	idx_t ConsumedSplits() const;
	idx_t CompletedSplits() const;
	idx_t CompletedRows() const;
	idx_t CompletedInputBytes() const;
	idx_t QueueWaitMillis() const;
	idx_t ExchangeSourcePartitionCount() const;
	idx_t ExchangeSourceTaskCount() const;
	void CompleteConsumedSplits();

private:
	struct SplitProgress {
		idx_t rows = 0;
		idx_t bytes = 0;
	};

	struct QueuedSplit {
		idx_t split_id = DConstants::INVALID_INDEX;
		TaskInput input;
		SplitProgress progress;
	};

	void WakeBlockedTasks(std::vector<duckdb::InterruptState> tasks);
	void CompleteSplitLocked(idx_t split_id);

	mutable std::mutex mutex_;
	std::condition_variable cv_;
	std::deque<QueuedSplit> splits_;
	std::vector<duckdb::InterruptState> blocked_tasks_;
	std::unordered_map<idx_t, SplitProgress> active_splits_;
	idx_t buffered_bytes_ = 0;
	idx_t submitted_splits_ = 0;
	idx_t consumed_splits_ = 0;
	idx_t completed_splits_ = 0;
	idx_t submitted_rows_ = 0;
	idx_t submitted_input_bytes_ = 0;
	idx_t consumed_rows_ = 0;
	idx_t consumed_input_bytes_ = 0;
	idx_t completed_rows_ = 0;
	idx_t completed_input_bytes_ = 0;
	idx_t queue_wait_micros_ = 0;
	idx_t exchange_source_partition_count_ = 0;
	idx_t exchange_source_task_count_ = 0;
	idx_t next_split_id_ = 1;
	bool no_more_splits_ = false;
	bool canceled_ = false;
};

const char *FteSplitQueueGetResultName(FteSplitQueue::GetResult result);

bool ApplyFteExchangeSourceQueuesToPlan(duckdb::PhysicalPlan &plan,
                                        const std::unordered_map<idx_t, std::shared_ptr<FteSplitQueue>> &queues,
                                        std::string *error = nullptr);

} // namespace distributed
} // namespace duckdb

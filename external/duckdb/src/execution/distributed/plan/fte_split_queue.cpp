// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/distributed/plan/fte_split_queue.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb/execution/distributed/plan/fte_split_queue.hpp"

#include "duckdb/execution/operator/exchange/physical_remote_exchange_source.hpp"
#include "duckdb/execution/distributed/plan/exchange_source_task.hpp"
#include "duckdb/execution/distributed/plan/scan_task.hpp"
#include "duckdb/execution/physical_plan.hpp"
#include "duckdb/parallel/interrupt.hpp"

#include <chrono>
#include <algorithm>
#include <limits>
#include <utility>

namespace duckdb {
namespace distributed {

namespace {

struct TaskInputProgressStats {
	idx_t rows = 0;
	idx_t bytes = 0;
	idx_t exchange_source_partition_count = 0;
	idx_t exchange_source_task_count = 0;
};

idx_t SaturatingAdd(idx_t lhs, idx_t rhs) {
	if (std::numeric_limits<idx_t>::max() - lhs < rhs) {
		return std::numeric_limits<idx_t>::max();
	}
	return lhs + rhs;
}

idx_t SaturatingMicroseconds(std::chrono::steady_clock::duration duration) {
	auto micros = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
	if (micros <= 0) {
		return 0;
	}
	auto max_value = static_cast<unsigned long long>(std::numeric_limits<idx_t>::max());
	auto value = static_cast<unsigned long long>(micros);
	return static_cast<idx_t>(value > max_value ? max_value : value);
}

idx_t TaskInputBufferedBytes(const TaskInput &input) {
	switch (input.kind) {
	case TaskInput::Kind::ScanTask:
		return input.scan_task_bytes.size();
	case TaskInput::Kind::ExchangeSourceTask:
		return input.exchange_source_task_bytes.size();
	default:
		return 0;
	}
}

TaskInputProgressStats TaskInputProgress(const TaskInput &input) {
	TaskInputProgressStats stats;
	try {
		switch (input.kind) {
		case TaskInput::Kind::ScanTask: {
			auto descriptor = ScanTaskDescriptor::DeserializeFromBytes(input.scan_task_bytes);
			stats.rows = descriptor.estimated_cardinality;
			stats.bytes = descriptor.estimated_bytes;
			break;
		}
		case TaskInput::Kind::ExchangeSourceTask: {
			auto descriptor = ExchangeSourceTaskDescriptor::DeserializeFromBytes(input.exchange_source_task_bytes);
			stats.exchange_source_partition_count = descriptor.source_partition_count;
			stats.exchange_source_task_count = descriptor.source_task_count;
			for (const auto &handle : descriptor.source_handles) {
				for (const auto &file : handle.files) {
					stats.rows = SaturatingAdd(stats.rows, file.rows);
					stats.bytes = SaturatingAdd(stats.bytes, static_cast<idx_t>(file.file_size));
				}
			}
			break;
		}
		default:
			break;
		}
	} catch (...) {
		return TaskInputProgressStats();
	}
	return stats;
}

FteSplitQueue::GetNextResult MakeGetNextResult(FteSplitQueue::GetResult state, TaskInput input = TaskInput(),
                                               idx_t split_id = DConstants::INVALID_INDEX) {
	FteSplitQueue::GetNextResult result;
	result.state = state;
	result.input = std::move(input);
	result.split_id = split_id;
	return result;
}

} // namespace

void FteSplitQueue::WakeBlockedTasks(std::vector<duckdb::InterruptState> tasks) {
	for (auto &task : tasks) {
		task.Callback();
	}
}

void FteSplitQueue::AddSplits(std::vector<TaskInput> inputs) {
	if (inputs.empty()) {
		return;
	}
	std::vector<duckdb::InterruptState> tasks;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if (canceled_) {
			return;
		}
		if (no_more_splits_) {
			throw std::runtime_error("cannot add splits after no_more_splits");
		}
		for (auto &input : inputs) {
			QueuedSplit queued;
			queued.split_id = next_split_id_++;
			auto progress = TaskInputProgress(input);
			queued.progress.rows = progress.rows;
			queued.progress.bytes = progress.bytes;
			exchange_source_partition_count_ =
			    std::max(exchange_source_partition_count_, progress.exchange_source_partition_count);
			exchange_source_task_count_ = std::max(exchange_source_task_count_, progress.exchange_source_task_count);
			queued.input = std::move(input);
			buffered_bytes_ += TaskInputBufferedBytes(queued.input);
			submitted_splits_ = SaturatingAdd(submitted_splits_, 1);
			submitted_rows_ = SaturatingAdd(submitted_rows_, queued.progress.rows);
			submitted_input_bytes_ = SaturatingAdd(submitted_input_bytes_, queued.progress.bytes);
			splits_.push_back(std::move(queued));
		}
		tasks.swap(blocked_tasks_);
	}
	cv_.notify_all();
	WakeBlockedTasks(std::move(tasks));
}

void FteSplitQueue::AddSplit(TaskInput input) {
	std::vector<TaskInput> inputs;
	inputs.push_back(std::move(input));
	AddSplits(std::move(inputs));
}

void FteSplitQueue::NoMoreSplits() {
	std::vector<duckdb::InterruptState> tasks;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		no_more_splits_ = true;
		tasks.swap(blocked_tasks_);
	}
	cv_.notify_all();
	WakeBlockedTasks(std::move(tasks));
}

void FteSplitQueue::Cancel() {
	std::vector<duckdb::InterruptState> tasks;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		canceled_ = true;
		tasks.swap(blocked_tasks_);
	}
	cv_.notify_all();
	WakeBlockedTasks(std::move(tasks));
}

FteSplitQueue::GetNextResult FteSplitQueue::TryGetNext() {
	std::lock_guard<std::mutex> lock(mutex_);
	if (canceled_) {
		return MakeGetNextResult(GetResult::CANCELED);
	}
	if (!splits_.empty()) {
		auto queued = std::move(splits_.front());
		buffered_bytes_ -= TaskInputBufferedBytes(queued.input);
		consumed_splits_ = SaturatingAdd(consumed_splits_, 1);
		consumed_rows_ = SaturatingAdd(consumed_rows_, queued.progress.rows);
		consumed_input_bytes_ = SaturatingAdd(consumed_input_bytes_, queued.progress.bytes);
		auto split_id = queued.split_id;
		active_splits_[split_id] = queued.progress;
		auto input = std::move(queued.input);
		splits_.pop_front();
		return MakeGetNextResult(GetResult::SPLIT, std::move(input), split_id);
	}
	if (no_more_splits_) {
		return MakeGetNextResult(GetResult::FINISHED);
	}
	return MakeGetNextResult(GetResult::BLOCKED);
}

FteSplitQueue::GetNextResult FteSplitQueue::WaitForNext() {
	std::unique_lock<std::mutex> lock(mutex_);
	auto wait_start = std::chrono::steady_clock::now();
	cv_.wait(lock, [&]() { return canceled_ || no_more_splits_ || !splits_.empty(); });
	queue_wait_micros_ =
	    SaturatingAdd(queue_wait_micros_, SaturatingMicroseconds(std::chrono::steady_clock::now() - wait_start));
	if (canceled_) {
		return MakeGetNextResult(GetResult::CANCELED);
	}
	if (!splits_.empty()) {
		auto queued = std::move(splits_.front());
		buffered_bytes_ -= TaskInputBufferedBytes(queued.input);
		consumed_splits_ = SaturatingAdd(consumed_splits_, 1);
		consumed_rows_ = SaturatingAdd(consumed_rows_, queued.progress.rows);
		consumed_input_bytes_ = SaturatingAdd(consumed_input_bytes_, queued.progress.bytes);
		auto split_id = queued.split_id;
		active_splits_[split_id] = queued.progress;
		auto input = std::move(queued.input);
		splits_.pop_front();
		return MakeGetNextResult(GetResult::SPLIT, std::move(input), split_id);
	}
	return MakeGetNextResult(GetResult::FINISHED);
}

bool FteSplitQueue::RegisterBlockedTask(const duckdb::InterruptState &interrupt_state) {
	std::lock_guard<std::mutex> lock(mutex_);
	if (canceled_ || no_more_splits_ || !splits_.empty()) {
		return false;
	}
	blocked_tasks_.push_back(interrupt_state);
	return true;
}

idx_t FteSplitQueue::BufferedSplits() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return splits_.size();
}

idx_t FteSplitQueue::BufferedBytes() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return buffered_bytes_;
}

idx_t FteSplitQueue::SubmittedRows() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return submitted_rows_;
}

idx_t FteSplitQueue::SubmittedInputBytes() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return submitted_input_bytes_;
}

idx_t FteSplitQueue::ConsumedRows() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return consumed_rows_;
}

idx_t FteSplitQueue::ConsumedInputBytes() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return consumed_input_bytes_;
}

idx_t FteSplitQueue::SubmittedSplits() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return submitted_splits_;
}

idx_t FteSplitQueue::ConsumedSplits() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return consumed_splits_;
}

idx_t FteSplitQueue::CompletedSplits() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return completed_splits_;
}

idx_t FteSplitQueue::CompletedRows() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return completed_rows_;
}

idx_t FteSplitQueue::CompletedInputBytes() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return completed_input_bytes_;
}

idx_t FteSplitQueue::QueueWaitMillis() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return queue_wait_micros_ / 1000;
}

idx_t FteSplitQueue::ExchangeSourcePartitionCount() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return exchange_source_partition_count_;
}

idx_t FteSplitQueue::ExchangeSourceTaskCount() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return exchange_source_task_count_;
}

void FteSplitQueue::CompleteSplitLocked(idx_t split_id) {
	if (split_id == DConstants::INVALID_INDEX) {
		return;
	}
	auto entry = active_splits_.find(split_id);
	if (entry == active_splits_.end()) {
		return;
	}
	completed_splits_ = SaturatingAdd(completed_splits_, 1);
	completed_rows_ = SaturatingAdd(completed_rows_, entry->second.rows);
	completed_input_bytes_ = SaturatingAdd(completed_input_bytes_, entry->second.bytes);
	active_splits_.erase(entry);
}

void FteSplitQueue::CompleteConsumedSplits() {
	std::lock_guard<std::mutex> lock(mutex_);
	while (!active_splits_.empty()) {
		CompleteSplitLocked(active_splits_.begin()->first);
	}
}

const char *FteSplitQueueGetResultName(FteSplitQueue::GetResult result) {
	switch (result) {
	case FteSplitQueue::GetResult::SPLIT:
		return "SPLIT";
	case FteSplitQueue::GetResult::BLOCKED:
		return "BLOCKED";
	case FteSplitQueue::GetResult::FINISHED:
		return "FINISHED";
	case FteSplitQueue::GetResult::CANCELED:
		return "CANCELED";
	default:
		return "UNKNOWN";
	}
}

namespace {

bool ApplyFteExchangeSourceQueuesToOperator(PhysicalOperator &op,
                                            const std::unordered_map<idx_t, std::shared_ptr<FteSplitQueue>> &queues,
                                            std::string *error, idx_t &applied) {
	if (op.type == PhysicalOperatorType::EXCHANGE_SOURCE) {
		auto *source = dynamic_cast<PhysicalRemoteExchangeSource *>(&op);
		if (source && source->RuntimeSourceNodeId().IsValid()) {
			const auto node_id = source->RuntimeSourceNodeId().GetIndex();
			auto entry = queues.find(node_id);
			if (entry == queues.end()) {
				if (error) {
					*error =
					    "missing FTE exchange source split queue for runtime_source_node_id=" + std::to_string(node_id);
				}
				return false;
			}
			if (!entry->second) {
				if (error) {
					*error =
					    "null FTE exchange source split queue for runtime_source_node_id=" + std::to_string(node_id);
				}
				return false;
			}
			source->ApplyRuntimeSplitQueue(entry->second);
			applied++;
		}
	}
	for (auto &child : op.children) {
		if (!ApplyFteExchangeSourceQueuesToOperator(child.get(), queues, error, applied)) {
			return false;
		}
	}
	return true;
}

} // namespace

bool ApplyFteExchangeSourceQueuesToPlan(duckdb::PhysicalPlan &plan,
                                        const std::unordered_map<idx_t, std::shared_ptr<FteSplitQueue>> &queues,
                                        std::string *error) {
	if (!plan.HasRoot()) {
		if (error) {
			*error = "plan has no root";
		}
		return false;
	}
	if (queues.empty()) {
		if (error) {
			*error = "FTE exchange source queue map is empty";
		}
		return false;
	}
	idx_t applied = 0;
	if (!ApplyFteExchangeSourceQueuesToOperator(plan.Root(), queues, error, applied)) {
		return false;
	}
	if (applied == 0) {
		if (error) {
			*error = "no FTE exchange source queues applied";
		}
		return false;
	}
	return true;
}

} // namespace distributed
} // namespace duckdb

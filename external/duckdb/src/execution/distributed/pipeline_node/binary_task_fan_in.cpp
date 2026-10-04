// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/binary_task_fan_in.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"

namespace duckdb {
namespace distributed {

namespace {

BinaryTaskPoll EndOfBinaryTaskStream() {
	return std::make_pair(false, BinarySubmittableTask());
}

} // namespace

void MoveTaskInputsOrThrow(TaskInputs &target, TaskInputs &source, const std::string &node_name) {
	for (auto &entry : source) {
		auto inserted = target.emplace(entry.first, std::move(entry.second));
		if (!inserted.second) {
			throw InvalidInputException("%s received duplicate source node id %llu", node_name,
			                            static_cast<unsigned long long>(entry.first));
		}
	}
	source.clear();
}

BinaryFragmentInputFanInStream::BinaryFragmentInputFanInStream(SubmittableTaskStream<WorkerTask> left,
                                                               SubmittableTaskStream<WorkerTask> right,
                                                               BinaryInitialTaskBuilder build_initial_task,
                                                               std::shared_ptr<TaskIDCounter> task_id_counter,
                                                               uint16_t query_idx, NodeID node_id,
                                                               std::string node_name)
    : left_(std::move(left)), right_(std::move(right)), build_initial_task_(std::move(build_initial_task)),
      task_id_counter_(std::move(task_id_counter)), query_idx_(query_idx), node_id_(node_id),
      node_name_(std::move(node_name)) {
}

BinaryTaskPoll BinaryFragmentInputFanInStream::poll_next() {
	if (finished_) {
		return EndOfBinaryTaskStream();
	}
	if (!initialized_) {
		return PollInitialTask();
	}

	while (true) {
		PollReadyInputs();
		auto ready = EmitContinuationTask();
		if (ready.first) {
			return ready;
		}
		if (left_exhausted_ && right_exhausted_) {
			finished_ = true;
			return EndOfBinaryTaskStream();
		}

		if (ShouldBlockOnLeft()) {
			PollLeft(true);
		} else {
			PollRight(true);
		}
	}
}

BinaryTaskPoll BinaryFragmentInputFanInStream::try_poll_next() {
	if (finished_) {
		return EndOfBinaryTaskStream();
	}
	if (!initialized_) {
		PollReadyInputs();
		return TryEmitInitialTask();
	}

	PollReadyInputs();
	auto ready = EmitContinuationTask();
	if (ready.first) {
		return ready;
	}
	if (left_exhausted_ && right_exhausted_) {
		finished_ = true;
	}
	return EndOfBinaryTaskStream();
}

BinaryTaskPoll BinaryFragmentInputFanInStream::PollInitialTask() {
	while (true) {
		PollReadyInputs();
		auto ready = TryEmitInitialTask();
		if (ready.first || finished_) {
			return ready;
		}

		if (!pending_left_ && !left_exhausted_) {
			PollLeft(true);
			continue;
		}
		if (!pending_right_ && !right_exhausted_) {
			PollRight(true);
			continue;
		}
	}
}

BinaryTaskPoll BinaryFragmentInputFanInStream::TryEmitInitialTask() {
	if (pending_left_ && pending_right_) {
		auto left_task = std::move(*pending_left_);
		auto right_task = std::move(*pending_right_);
		pending_left_.reset();
		pending_right_.reset();

		auto combined_task = build_initial_task_(std::move(left_task), std::move(right_task));
		continuation_template_ = combined_task.task()->clone();
		continuation_template_->mutable_inputs().clear();
		initialized_ = true;
		return std::make_pair(true, std::move(combined_task));
	}
	if (pending_left_ && right_exhausted_) {
		throw InvalidInputException("%s received an empty right task stream", node_name_);
	}
	if (pending_right_ && left_exhausted_) {
		throw InvalidInputException("%s received an empty left task stream", node_name_);
	}
	if (left_exhausted_ && right_exhausted_) {
		finished_ = true;
	}
	return EndOfBinaryTaskStream();
}

BinaryTaskPoll BinaryFragmentInputFanInStream::EmitContinuationTask() {
	if (!pending_left_ && !pending_right_) {
		return EndOfBinaryTaskStream();
	}

	TaskInputs inputs;
	if (pending_left_) {
		MoveTaskInputsOrThrow(inputs, pending_left_->task()->mutable_inputs(), node_name_);
		pending_left_.reset();
	}
	if (pending_right_) {
		MoveTaskInputsOrThrow(inputs, pending_right_->task()->mutable_inputs(), node_name_);
		pending_right_.reset();
	}

	TaskContext task_context = TaskContext::from_node_context(query_idx_, node_id_, task_id_counter_->next());
	WorkerTask continuation_task(task_context, continuation_template_->plan(), continuation_template_->config(),
	                             continuation_template_->context(), continuation_template_->name(), std::move(inputs));
	return std::make_pair(true, BinarySubmittableTask(std::move(continuation_task)));
}

void BinaryFragmentInputFanInStream::PollReadyInputs() {
	PollLeft(false);
	PollRight(false);
}

void BinaryFragmentInputFanInStream::PollLeft(bool blocking) {
	PollSide(left_, pending_left_, left_exhausted_, blocking);
}

void BinaryFragmentInputFanInStream::PollRight(bool blocking) {
	PollSide(right_, pending_right_, right_exhausted_, blocking);
}

void BinaryFragmentInputFanInStream::PollSide(SubmittableTaskStream<WorkerTask> &stream,
                                              std::unique_ptr<BinarySubmittableTask> &pending, bool &exhausted,
                                              bool blocking) {
	if (pending || exhausted) {
		return;
	}

	auto next = blocking ? stream.poll_next() : stream.try_poll_next();
	if (next.first) {
		pending.reset(new BinarySubmittableTask(std::move(next.second)));
		if (stream.is_exhausted()) {
			exhausted = true;
		}
		return;
	}
	if (blocking || stream.is_exhausted()) {
		exhausted = true;
	}
}

bool BinaryFragmentInputFanInStream::ShouldBlockOnLeft() {
	if (left_exhausted_) {
		return false;
	}
	if (right_exhausted_) {
		return true;
	}
	const bool block_on_left = prefer_left_;
	prefer_left_ = !prefer_left_;
	return block_on_left;
}

} // namespace distributed
} // namespace duckdb

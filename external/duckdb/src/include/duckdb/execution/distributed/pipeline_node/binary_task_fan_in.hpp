// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"

namespace duckdb {
namespace distributed {

class TaskIDCounter;

using BinarySubmittableTask = SubmittableTask<WorkerTask>;
using BinaryTaskPoll = std::pair<bool, BinarySubmittableTask>;
using BinaryInitialTaskBuilder = std::function<BinarySubmittableTask(BinarySubmittableTask, BinarySubmittableTask)>;

//! Move source-id keyed task inputs into target while rejecting collisions.
void MoveTaskInputsOrThrow(TaskInputs &target, TaskInputs &source, const std::string &node_name);

//! Fan two independently advancing input streams into one logical worker fragment.
//!
//! The first task from each side is passed to build_initial_task. Later tasks are
//! emitted as input-only updates that reuse the initial task's plan. This is not
//! a positional zip: exchange source descriptors on either side may arrive at
//! different rates.
class BinaryFragmentInputFanInStream {
public:
	BinaryFragmentInputFanInStream(SubmittableTaskStream<WorkerTask> left, SubmittableTaskStream<WorkerTask> right,
	                               BinaryInitialTaskBuilder build_initial_task,
	                               std::shared_ptr<TaskIDCounter> task_id_counter, uint16_t query_idx, NodeID node_id,
	                               std::string node_name);

	BinaryTaskPoll poll_next();
	BinaryTaskPoll try_poll_next();

	bool is_exhausted() const {
		return finished_;
	}

private:
	BinaryTaskPoll PollInitialTask();
	BinaryTaskPoll TryEmitInitialTask();
	BinaryTaskPoll EmitContinuationTask();

	void PollReadyInputs();
	void PollLeft(bool blocking);
	void PollRight(bool blocking);
	static void PollSide(SubmittableTaskStream<WorkerTask> &stream, std::unique_ptr<BinarySubmittableTask> &pending,
	                     bool &exhausted, bool blocking);
	bool ShouldBlockOnLeft();

private:
	SubmittableTaskStream<WorkerTask> left_;
	SubmittableTaskStream<WorkerTask> right_;
	BinaryInitialTaskBuilder build_initial_task_;
	std::shared_ptr<TaskIDCounter> task_id_counter_;
	uint16_t query_idx_;
	NodeID node_id_;
	std::string node_name_;
	std::unique_ptr<BinarySubmittableTask> pending_left_;
	std::unique_ptr<BinarySubmittableTask> pending_right_;
	std::unique_ptr<WorkerTask> continuation_template_;
	bool left_exhausted_ = false;
	bool right_exhausted_ = false;
	bool initialized_ = false;
	bool finished_ = false;
	bool prefer_left_ = true;
};

} // namespace distributed
} // namespace duckdb

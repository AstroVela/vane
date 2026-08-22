// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/union.hpp"

#include <limits>

#include "duckdb/common/exception.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"

namespace duckdb {
namespace distributed {

namespace {

using UnionTask = SubmittableTask<WorkerTask>;
using UnionTaskPoll = std::pair<bool, UnionTask>;

UnionTaskPoll EndOfUnionTaskStream() {
	return std::make_pair(false, UnionTask());
}

UnionTask TagUnionTask(UnionTask task, const PipelineNodeContext &union_context) {
	auto *worker_task = task.task();
	if (!worker_task || !worker_task->plan() || !worker_task->plan()->HasRoot() || !worker_task->config()) {
		throw InvalidInputException("UnionNode received an invalid branch task");
	}

	auto task_context = worker_task->task_context();
	if (task_context.query_idx() != union_context.query_idx()) {
		throw InvalidInputException("UnionNode received a branch task from a different query index");
	}
	task_context.add_node_id(union_context.node_id());
	auto merged_context = MergeTaskContext(worker_task->context(), union_context.to_hashmap());
	auto inputs = std::move(worker_task->mutable_inputs());
	WorkerTask tagged_task(task_context, worker_task->plan(), worker_task->config(), std::move(merged_context),
	                       worker_task->name(), std::move(inputs));
	return std::move(task).with_new_task(std::move(tagged_task));
}

class UnionTaskFanInStream {
public:
	UnionTaskFanInStream(std::vector<DistributedPipelineNodeRef> children, PlanExecutionContext &plan_context,
	                     PipelineNodeContext union_context, bool allow_out_of_order)
	    : children_(std::move(children)), plan_context_(&plan_context), union_context_(std::move(union_context)),
	      allow_out_of_order_(allow_out_of_order) {
		if (children_.empty()) {
			finished_ = true;
			return;
		}
		if (allow_out_of_order_) {
			streams_.reserve(children_.size());
			for (idx_t child_idx = 0; child_idx < children_.size(); child_idx++) {
				streams_.push_back(CreateChildStream(child_idx));
			}
			active_stream_count_ = streams_.size();
		}
	}

	UnionTaskPoll poll_next() {
		return allow_out_of_order_ ? PollUnordered() : PollOrdered(true);
	}

	UnionTaskPoll try_poll_next() {
		return allow_out_of_order_ ? TryPollUnordered() : PollOrdered(false);
	}

	bool is_exhausted() const {
		return finished_;
	}

private:
	std::unique_ptr<SubmittableTaskStream<WorkerTask>> CreateChildStream(idx_t child_idx) {
		if (!plan_context_ || child_idx >= children_.size() || !children_[child_idx]) {
			throw InvalidInputException("UnionNode cannot create an invalid branch task stream");
		}
		auto stream = children_[child_idx]->produce_tasks(*plan_context_);
		return std::unique_ptr<SubmittableTaskStream<WorkerTask>>(
		    new SubmittableTaskStream<WorkerTask>(std::move(stream)));
	}

	UnionTaskPoll PollOrdered(bool blocking) {
		while (!finished_) {
			if (!ordered_stream_) {
				if (ordered_child_idx_ >= children_.size()) {
					finished_ = true;
					return EndOfUnionTaskStream();
				}
				ordered_stream_ = CreateChildStream(ordered_child_idx_);
			}

			auto next = blocking ? ordered_stream_->poll_next() : ordered_stream_->try_poll_next();
			if (next.first) {
				if (ordered_stream_->is_exhausted()) {
					ordered_stream_.reset();
					ordered_child_idx_++;
					finished_ = ordered_child_idx_ >= children_.size();
				}
				return std::make_pair(true, TagUnionTask(std::move(next.second), union_context_));
			}

			if (blocking || ordered_stream_->is_exhausted()) {
				ordered_stream_.reset();
				ordered_child_idx_++;
				continue;
			}
			return EndOfUnionTaskStream();
		}
		return EndOfUnionTaskStream();
	}

	UnionTaskPoll TryPollUnordered() {
		if (finished_) {
			return EndOfUnionTaskStream();
		}
		const auto stream_count = streams_.size();
		for (idx_t offset = 0; offset < stream_count; offset++) {
			const auto stream_idx = (round_robin_idx_ + offset) % stream_count;
			auto &stream = streams_[stream_idx];
			if (!stream) {
				continue;
			}

			auto next = stream->try_poll_next();
			if (next.first) {
				RetireStreamIfExhausted(stream_idx);
				round_robin_idx_ = (stream_idx + 1) % stream_count;
				finished_ = active_stream_count_ == 0;
				return std::make_pair(true, TagUnionTask(std::move(next.second), union_context_));
			}
			RetireStreamIfExhausted(stream_idx);
		}
		finished_ = active_stream_count_ == 0;
		return EndOfUnionTaskStream();
	}

	UnionTaskPoll PollUnordered() {
		while (!finished_) {
			auto ready = TryPollUnordered();
			if (ready.first || finished_) {
				return ready;
			}

			const auto stream_count = streams_.size();
			for (idx_t offset = 0; offset < stream_count; offset++) {
				const auto stream_idx = (round_robin_idx_ + offset) % stream_count;
				auto &stream = streams_[stream_idx];
				if (!stream) {
					continue;
				}

				auto next = stream->poll_next();
				if (next.first) {
					RetireStreamIfExhausted(stream_idx);
					round_robin_idx_ = (stream_idx + 1) % stream_count;
					finished_ = active_stream_count_ == 0;
					return std::make_pair(true, TagUnionTask(std::move(next.second), union_context_));
				}
				RetireStream(stream_idx);
				break;
			}
			finished_ = active_stream_count_ == 0;
		}
		return EndOfUnionTaskStream();
	}

	void RetireStreamIfExhausted(idx_t stream_idx) {
		if (streams_[stream_idx] && streams_[stream_idx]->is_exhausted()) {
			RetireStream(stream_idx);
		}
	}

	void RetireStream(idx_t stream_idx) {
		if (!streams_[stream_idx]) {
			return;
		}
		streams_[stream_idx].reset();
		if (active_stream_count_ == 0) {
			throw InternalException("UnionNode task stream accounting underflow");
		}
		active_stream_count_--;
	}

private:
	std::vector<DistributedPipelineNodeRef> children_;
	PlanExecutionContext *plan_context_;
	PipelineNodeContext union_context_;
	bool allow_out_of_order_;
	std::vector<std::unique_ptr<SubmittableTaskStream<WorkerTask>>> streams_;
	std::unique_ptr<SubmittableTaskStream<WorkerTask>> ordered_stream_;
	idx_t ordered_child_idx_ = 0;
	idx_t round_robin_idx_ = 0;
	idx_t active_stream_count_ = 0;
	bool finished_ = false;
};

size_t CountUnionPartitions(const std::vector<DistributedPipelineNodeRef> &children) {
	size_t partition_count = 0;
	for (const auto &child : children) {
		if (!child || child->is_statically_empty_result()) {
			continue;
		}
		auto clustering = child->config().clustering_spec();
		if (!clustering || clustering->num_partitions() == 0) {
			throw InvalidInputException("UnionNode requires positive clustering metadata for every branch");
		}
		const auto child_partitions = clustering->num_partitions();
		if (child_partitions > std::numeric_limits<size_t>::max() - partition_count) {
			throw InvalidInputException("UnionNode partition count overflow");
		}
		partition_count += child_partitions;
	}
	return partition_count == 0 ? 1 : partition_count;
}

} // namespace

UnionNode::UnionNode(NodeID node_id, const PlanConfig &plan_config, std::vector<DistributedPipelineNodeRef> children,
                     SchemaRef schema, bool allow_out_of_order)
    : context_(plan_config.query_idx, plan_config.query_id, node_id, "Union"), children_(std::move(children)),
      allow_out_of_order_(allow_out_of_order), statically_empty_(true) {
	if (children_.empty()) {
		throw InvalidInputException("UnionNode requires at least one branch");
	}
	if (!schema) {
		throw InvalidInputException("UnionNode requires an output schema");
	}

	const auto output_types = GetSchemaTypes(schema);
	if (output_types.empty()) {
		throw InvalidInputException("UnionNode requires a non-empty output schema");
	}
	DuckDBExecutionConfigRef execution_config = plan_config.config;
	for (const auto &child : children_) {
		if (!child || !child->config().schema() || !child->config().clustering_spec()) {
			throw InvalidInputException("UnionNode received an invalid branch");
		}
		if (child->context().query_idx() != context_.query_idx() ||
		    child->context().query_id() != context_.query_id()) {
			throw InvalidInputException("UnionNode received a branch from a different query");
		}
		if (GetSchemaTypes(child->config().schema()) != output_types) {
			throw InvalidInputException("UnionNode branch schema does not match its output schema");
		}
		if (!execution_config) {
			execution_config = child->config().execution_config();
		}
		if (!child->is_statically_empty_result()) {
			statically_empty_ = false;
		}
	}
	if (!execution_config) {
		throw InvalidInputException("UnionNode requires an execution configuration");
	}

	config_ = PipelineNodeConfig(std::move(schema), std::move(execution_config),
	                             ClusteringSpec::unknown_with_num_partitions(CountUnionPartitions(children_)));
}

std::vector<PipelineNodeRef> UnionNode::children() const {
	std::vector<PipelineNodeRef> result;
	result.reserve(children_.size());
	for (const auto &child : children_) {
		result.push_back(child->inner());
	}
	return result;
}

SubmittableTaskStream<WorkerTask> UnionNode::produce_tasks(PlanExecutionContext &plan_context) {
	std::vector<DistributedPipelineNodeRef> active_children;
	active_children.reserve(children_.size());
	for (const auto &child : children_) {
		if (!child->is_statically_empty_result()) {
			active_children.push_back(child);
		}
	}
	// A parent such as an ungrouped aggregate needs one logical empty input task.
	// A root all-empty UNION is skipped by PlanRunner via is_statically_empty_result().
	if (active_children.empty()) {
		active_children.push_back(children_.front());
	}

	UnionTaskFanInStream stream(std::move(active_children), plan_context, context_, allow_out_of_order_);
	return SubmittableTaskStream<WorkerTask>(boxed<UnionTask>(std::move(stream)));
}

std::vector<std::string> UnionNode::multiline_display(bool /*verbose*/) const {
	return {"Union", "Branches: " + std::to_string(children_.size()),
	        std::string("Order: ") + (allow_out_of_order_ ? "out-of-order" : "branch order")};
}

} // namespace distributed
} // namespace duckdb

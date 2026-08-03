// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/limit.hpp"
#include "duckdb/execution/distributed/pipeline_node/materialized_coordinator.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"
#include "duckdb/execution/operator/helper/physical_limit.hpp"
#include "duckdb/execution/operator/helper/physical_streaming_limit.hpp"
#include "duckdb/execution/operator/helper/physical_limit_percent.hpp"

#include <limits>

namespace duckdb {
namespace distributed {

BoundLimitNode CopyBoundLimitNode(const BoundLimitNode &node) {
	switch (node.Type()) {
	case LimitNodeType::UNSET:
		return BoundLimitNode();
	case LimitNodeType::CONSTANT_VALUE:
		return BoundLimitNode::ConstantValue(static_cast<int64_t>(node.GetConstantValue()));
	case LimitNodeType::CONSTANT_PERCENTAGE:
		return BoundLimitNode::ConstantPercentage(node.GetConstantPercentage());
	case LimitNodeType::EXPRESSION_VALUE: {
		auto expr = node.GetValueExpression().Copy();
		return BoundLimitNode::ExpressionValue(std::move(expr));
	}
	case LimitNodeType::EXPRESSION_PERCENTAGE: {
		auto expr = node.GetPercentageExpression().Copy();
		return BoundLimitNode::ExpressionPercentage(std::move(expr));
	}
	}
	return BoundLimitNode();
}

namespace {
using PlanBuilder = MaterializedPlanBuilder;
using PerTaskBuilderFactory = PerTaskMaterializedPlanBuilderFactory;

static std::string BoundLimitNodeToString(const BoundLimitNode &node) {
	switch (node.Type()) {
	case LimitNodeType::UNSET:
		return "<unset>";
	case LimitNodeType::CONSTANT_VALUE:
		return std::to_string(node.GetConstantValue());
	case LimitNodeType::CONSTANT_PERCENTAGE:
		return std::to_string(node.GetConstantPercentage());
	case LimitNodeType::EXPRESSION_VALUE:
		return node.GetValueExpression().GetName();
	case LimitNodeType::EXPRESSION_PERCENTAGE:
		return node.GetPercentageExpression().GetName();
	}
	return "<unknown>";
}

static std::pair<bool, idx_t> TryGetConstantLimitRows(const BoundLimitNode &node, bool unset_as_zero) {
	if (node.Type() == LimitNodeType::UNSET) {
		return unset_as_zero ? std::make_pair(true, idx_t(0)) : std::make_pair(false, idx_t(0));
	}
	if (node.Type() != LimitNodeType::CONSTANT_VALUE) {
		return std::make_pair(false, idx_t(0));
	}
	auto constant_val = node.GetConstantValue();
	if (constant_val <= 0) {
		return std::make_pair(true, idx_t(0));
	}
	return std::make_pair(true, static_cast<idx_t>(constant_val));
}

static std::pair<bool, idx_t> ComputeLocalLimitRows(const BoundLimitNode &limit_val, const BoundLimitNode &offset_val) {
	auto limit_rows = TryGetConstantLimitRows(limit_val, false);
	if (!limit_rows.first) {
		return std::make_pair(false, idx_t(0));
	}

	auto offset_rows = TryGetConstantLimitRows(offset_val, true);
	if (!offset_rows.first) {
		return std::make_pair(false, idx_t(0));
	}

	if (limit_rows.second > std::numeric_limits<idx_t>::max() - offset_rows.second) {
		return std::make_pair(true, std::numeric_limits<idx_t>::max());
	}
	return std::make_pair(true, limit_rows.second + offset_rows.second);
}

static BoundLimitNode ConstantLimitRows(idx_t rows) {
	const auto max_i64 = static_cast<idx_t>(std::numeric_limits<int64_t>::max());
	if (rows > max_i64) {
		rows = max_i64;
	}
	return BoundLimitNode::ConstantValue(static_cast<int64_t>(rows));
}

} // namespace

LimitNode::LimitNode(NodeID node_id, PipelineNodeRef child, BoundLimitNode limit_val, BoundLimitNode offset_val,
                     std::shared_ptr<ExchangeManager> exchange_mgr)
    : ctx_(InheritPipelineNodeContext(child, node_id, "Limit")),
      config_(child ? child->config() : PipelineNodeConfig()), child_(std::move(child)),
      limit_val_(std::move(limit_val)), offset_val_(std::move(offset_val)), exchange_mgr_(std::move(exchange_mgr)) {
}

SubmittableTaskStream<WorkerTask> LimitNode::produce_tasks(PlanExecutionContext &plan_context) {
	auto limit_template = std::make_shared<BoundLimitNode>(CopyBoundLimitNode(limit_val_));
	auto offset_template = std::make_shared<BoundLimitNode>(CopyBoundLimitNode(offset_val_));
	auto final_plan_builder = [limit_template, offset_template](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
		auto limit_val = CopyBoundLimitNode(*limit_template);
		auto offset_val = CopyBoundLimitNode(*offset_template);

		auto types = input_plan->Root().GetTypes();
		idx_t estimated_cardinality = input_plan->Root().estimated_cardinality;

		auto &old_root = input_plan->Root();
		auto &limit_op = input_plan->Make<duckdb::PhysicalLimit>(types, std::move(limit_val), std::move(offset_val),
		                                                         estimated_cardinality);
		limit_op.children.push_back(old_root);
		input_plan->SetRoot(limit_op);
		return input_plan;
	};

	if (!ChildHasMultiplePartitions(child_)) {
		auto input_stream = child_->produce_tasks(plan_context);
		return input_stream.pipeline_instruction(shared_from_this(), final_plan_builder, plan_context.client_context());
	}

	PerTaskBuilderFactory per_task_builder_factory;
	auto local_limit_rows = ComputeLocalLimitRows(limit_val_, offset_val_);
	if (local_limit_rows.first) {
		per_task_builder_factory = [local_limit_rows](idx_t /*unused*/) -> PlanBuilder {
			return [local_limit_rows](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
				auto limit_val = ConstantLimitRows(local_limit_rows.second);
				auto offset_val = BoundLimitNode();

				auto types = input_plan->Root().GetTypes();
				idx_t estimated_cardinality = input_plan->Root().estimated_cardinality;

				auto &old_root = input_plan->Root();
				auto &limit_op = input_plan->Make<duckdb::PhysicalLimit>(types, std::move(limit_val),
				                                                         std::move(offset_val), estimated_cardinality);
				limit_op.children.push_back(old_root);
				input_plan->SetRoot(limit_op);
				return input_plan;
			};
		};
	}

	return ProduceWithMaterializedCoordinator(
	    plan_context, child_, std::static_pointer_cast<PipelineNodeImpl>(shared_from_this()),
	    std::move(final_plan_builder), std::move(per_task_builder_factory), exchange_mgr_);
}

std::vector<std::string> LimitNode::multiline_display(bool /*verbose*/) const {
	return {std::string("Limit: ") + BoundLimitNodeToString(limit_val_) +
	        " Offset: " + BoundLimitNodeToString(offset_val_)};
}

StreamingLimitNode::StreamingLimitNode(NodeID node_id, PipelineNodeRef child, BoundLimitNode limit_val,
                                       BoundLimitNode offset_val, bool parallel,
                                       std::shared_ptr<ExchangeManager> exchange_mgr)
    : ctx_(InheritPipelineNodeContext(child, node_id, "StreamingLimit")),
      config_(child ? child->config() : PipelineNodeConfig()), child_(std::move(child)),
      limit_val_(std::move(limit_val)), offset_val_(std::move(offset_val)), parallel_(parallel),
      exchange_mgr_(std::move(exchange_mgr)) {
}

SubmittableTaskStream<WorkerTask> StreamingLimitNode::produce_tasks(PlanExecutionContext &plan_context) {
	auto limit_template = std::make_shared<BoundLimitNode>(CopyBoundLimitNode(limit_val_));
	auto offset_template = std::make_shared<BoundLimitNode>(CopyBoundLimitNode(offset_val_));
	const bool parallel = parallel_;

	auto final_plan_builder = [limit_template, offset_template,
	                           parallel](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
		auto limit_val = CopyBoundLimitNode(*limit_template);
		auto offset_val = CopyBoundLimitNode(*offset_template);

		auto types = input_plan->Root().GetTypes();
		idx_t estimated_cardinality = input_plan->Root().estimated_cardinality;

		auto &old_root = input_plan->Root();
		auto &limit_op = input_plan->Make<duckdb::PhysicalStreamingLimit>(
		    types, std::move(limit_val), std::move(offset_val), estimated_cardinality, parallel);
		limit_op.children.push_back(old_root);
		input_plan->SetRoot(limit_op);
		return input_plan;
	};

	if (!ChildHasMultiplePartitions(child_)) {
		auto input_stream = child_->produce_tasks(plan_context);
		return input_stream.pipeline_instruction(shared_from_this(), final_plan_builder, plan_context.client_context());
	}

	PerTaskBuilderFactory per_task_builder_factory;
	auto local_limit_rows = ComputeLocalLimitRows(limit_val_, offset_val_);
	if (local_limit_rows.first) {
		per_task_builder_factory = [local_limit_rows, parallel](idx_t /*unused*/) -> PlanBuilder {
			return [local_limit_rows, parallel](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
				auto limit_val = ConstantLimitRows(local_limit_rows.second);
				auto offset_val = BoundLimitNode();

				auto types = input_plan->Root().GetTypes();
				idx_t estimated_cardinality = input_plan->Root().estimated_cardinality;

				auto &old_root = input_plan->Root();
				auto &limit_op = input_plan->Make<duckdb::PhysicalStreamingLimit>(
				    types, std::move(limit_val), std::move(offset_val), estimated_cardinality, parallel);
				limit_op.children.push_back(old_root);
				input_plan->SetRoot(limit_op);
				return input_plan;
			};
		};
	}

	return ProduceWithMaterializedCoordinator(
	    plan_context, child_, std::static_pointer_cast<PipelineNodeImpl>(shared_from_this()),
	    std::move(final_plan_builder), std::move(per_task_builder_factory), exchange_mgr_);
}

std::vector<std::string> StreamingLimitNode::multiline_display(bool /*verbose*/) const {
	return {std::string("StreamingLimit: ") + BoundLimitNodeToString(limit_val_) +
	        " Offset: " + BoundLimitNodeToString(offset_val_) + " Parallel: " + (parallel_ ? "true" : "false")};
}

LimitPercentNode::LimitPercentNode(NodeID node_id, PipelineNodeRef child, BoundLimitNode limit_val,
                                   BoundLimitNode offset_val, std::shared_ptr<ExchangeManager> exchange_mgr)
    : ctx_(InheritPipelineNodeContext(child, node_id, "LimitPercent")),
      config_(child ? child->config() : PipelineNodeConfig()), child_(std::move(child)),
      limit_val_(std::move(limit_val)), offset_val_(std::move(offset_val)), exchange_mgr_(std::move(exchange_mgr)) {
}

SubmittableTaskStream<WorkerTask> LimitPercentNode::produce_tasks(PlanExecutionContext &plan_context) {
	auto limit_template = std::make_shared<BoundLimitNode>(CopyBoundLimitNode(limit_val_));
	auto offset_template = std::make_shared<BoundLimitNode>(CopyBoundLimitNode(offset_val_));

	auto final_plan_builder = [limit_template, offset_template](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
		auto limit_val = CopyBoundLimitNode(*limit_template);
		auto offset_val = CopyBoundLimitNode(*offset_template);

		auto types = input_plan->Root().GetTypes();
		idx_t estimated_cardinality = input_plan->Root().estimated_cardinality;

		auto &old_root = input_plan->Root();
		auto &limit_op = input_plan->Make<duckdb::PhysicalLimitPercent>(types, std::move(limit_val),
		                                                                std::move(offset_val), estimated_cardinality);
		limit_op.children.push_back(old_root);
		input_plan->SetRoot(limit_op);
		return input_plan;
	};

	if (!ChildHasMultiplePartitions(child_)) {
		auto input_stream = child_->produce_tasks(plan_context);
		return input_stream.pipeline_instruction(shared_from_this(), final_plan_builder, plan_context.client_context());
	}

	return ProduceWithMaterializedCoordinator(plan_context, child_,
	                                          std::static_pointer_cast<PipelineNodeImpl>(shared_from_this()),
	                                          std::move(final_plan_builder), PerTaskBuilderFactory(), exchange_mgr_);
}

std::vector<std::string> LimitPercentNode::multiline_display(bool /*verbose*/) const {
	return {std::string("LimitPercent: ") + BoundLimitNodeToString(limit_val_) +
	        " Offset: " + BoundLimitNodeToString(offset_val_)};
}

} // namespace distributed
} // namespace duckdb

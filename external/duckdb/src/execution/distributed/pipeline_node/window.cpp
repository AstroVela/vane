// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/window.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/execution/operator/aggregate/physical_streaming_window.hpp"
#include "duckdb/execution/operator/aggregate/physical_window.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"
#include "duckdb/planner/expression/bound_window_expression.hpp"

namespace duckdb {
namespace distributed {

static duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>
CopyWindowSelectList(const std::vector<ExpressionRef> &select_list) {
	duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> copies;
	copies.reserve(select_list.size());
	for (const auto &expr_ref : select_list) {
		if (!expr_ref) {
			continue;
		}
		auto copy = expr_ref->Copy();
		copies.push_back(duckdb::unique_ptr<duckdb::Expression>(copy.release()));
	}
	return copies;
}

static const BoundWindowExpression *GetCommonWindowDefinition(const std::vector<ExpressionRef> &select_list,
                                                              const char *node_name) {
	const BoundWindowExpression *first = nullptr;
	for (const auto &expr : select_list) {
		if (!expr || expr->GetExpressionClass() != ExpressionClass::BOUND_WINDOW) {
			throw InvalidInputException("%s requires bound window expressions", node_name);
		}
		const auto &window = expr->Cast<BoundWindowExpression>();
		if (!first) {
			first = &window;
			continue;
		}
		if (!first->PartitionsAreEquivalent(window)) {
			throw InvalidInputException("%s received incompatible partition definitions", node_name);
		}
	}
	return first;
}

static bool WindowPartitionKeysMatch(const ClusteringSpecRef &clustering,
                                     const vector<unique_ptr<Expression>> &required_partitions) {
	if (!clustering || clustering->type() != ClusteringSpec::Type::Hash) {
		return false;
	}
	auto actual_partitions = clustering->partition_by();
	if (actual_partitions.size() != required_partitions.size()) {
		return false;
	}

	std::vector<bool> matched(actual_partitions.size(), false);
	for (const auto &required : required_partitions) {
		bool found = false;
		for (idx_t index = 0; index < actual_partitions.size(); index++) {
			if (!matched[index] && required && actual_partitions[index] &&
			    Expression::Equals(*required, *actual_partitions[index])) {
				matched[index] = true;
				found = true;
				break;
			}
		}
		if (!found) {
			return false;
		}
	}
	return true;
}

static void ValidateWindowDistribution(const PipelineNodeRef &child, const std::vector<ExpressionRef> &select_list) {
	auto window = GetCommonWindowDefinition(select_list, "Window");
	if (!window) {
		return;
	}
	if (!child || !child->config().clustering_spec()) {
		throw InvalidInputException("Window child is missing clustering metadata");
	}
	auto clustering = child->config().clustering_spec();
	if (clustering->num_partitions() <= 1) {
		return;
	}
	if (window->partitions.empty()) {
		throw InvalidInputException("Global Window requires single-partition input");
	}
	if (!WindowPartitionKeysMatch(clustering, window->partitions)) {
		throw InvalidInputException("Partitioned Window requires matching hash-partitioned input");
	}
}

static void ValidateStreamingWindowDistribution(const PipelineNodeRef &child,
                                                const std::vector<ExpressionRef> &select_list) {
	auto window = GetCommonWindowDefinition(select_list, "StreamingWindow");
	if (!window) {
		return;
	}
	if (!window->partitions.empty() || !window->orders.empty() || !window->arg_orders.empty() ||
	    window->exclude_clause != WindowExcludeMode::NO_OTHER) {
		throw InvalidInputException("StreamingWindow requires an empty OVER clause");
	}
	if (!child || !child->config().clustering_spec()) {
		throw InvalidInputException("StreamingWindow child is missing clustering metadata");
	}
	if (child->config().clustering_spec()->num_partitions() > 1) {
		throw InvalidInputException("StreamingWindow requires single-partition input");
	}
}

static PipelineNodeConfig BuildWindowConfig(const PipelineNodeRef &child, const std::vector<ExpressionRef> &select_list,
                                            const std::vector<LogicalType> &output_types) {
	auto execution_config = child ? child->config().execution_config() : DuckDBExecutionConfigRef();
	auto clustering = child ? child->config().clustering_spec() : ClusteringSpec::unknown_with_num_partitions(1);
	if (output_types.empty()) {
		return PipelineNodeConfig(nullptr, std::move(execution_config), std::move(clustering));
	}
	if (output_types.size() < select_list.size()) {
		return PipelineNodeConfig(MakeSchemaRef(output_types), std::move(execution_config), std::move(clustering));
	}

	const auto input_column_count = output_types.size() - select_list.size();
	auto output_names = child ? GetSchemaNames(child->config().schema()) : vector<string> {};
	if (output_names.size() != input_column_count) {
		output_names.clear();
		output_names.reserve(output_types.size());
		for (idx_t index = 0; index < input_column_count; index++) {
			output_names.push_back("c" + std::to_string(index));
		}
	} else {
		output_names.reserve(output_types.size());
	}
	for (idx_t index = input_column_count; index < output_types.size(); index++) {
		output_names.push_back("c" + std::to_string(index));
	}
	return PipelineNodeConfig(MakeSchemaRef(output_types, output_names), std::move(execution_config),
	                          std::move(clustering));
}

WindowNode::WindowNode(NodeID node_id, PipelineNodeRef child, std::vector<ExpressionRef> select_list,
                       std::vector<LogicalType> output_types)
    : ctx_(InheritPipelineNodeContext(child, node_id, "Window")),
      config_(BuildWindowConfig(child, select_list, output_types)), child_(std::move(child)),
      select_list_(std::move(select_list)), output_types_(std::move(output_types)) {
	ValidateWindowDistribution(child_, select_list_);
}

std::vector<PipelineNodeRef> WindowNode::children() const {
	return {child_};
}

SubmittableTaskStream<WorkerTask> WindowNode::produce_tasks(PlanExecutionContext &plan_context) {
	auto input_stream = child_->produce_tasks(plan_context);
	const auto *select_list_ptr = &select_list_;
	const auto *output_types_ptr = &output_types_;
	return input_stream.pipeline_instruction(
	    shared_from_this(),
	    [select_list_ptr, output_types_ptr](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
		    if (select_list_ptr->empty()) {
			    return input_plan;
		    }
		    auto select_list = CopyWindowSelectList(*select_list_ptr);
		    auto types = *output_types_ptr;
		    idx_t estimated_cardinality = input_plan->Root().estimated_cardinality;

		    auto &old_root = input_plan->Root();
		    auto &window_op = input_plan->Make<duckdb::PhysicalWindow>(std::move(types), std::move(select_list),
		                                                               estimated_cardinality);
		    window_op.children.push_back(old_root);
		    input_plan->SetRoot(window_op);
		    return input_plan;
	    },
	    plan_context.client_context());
}

std::vector<std::string> WindowNode::multiline_display(bool /*verbose*/) const {
	std::string exprs;
	for (size_t i = 0; i < select_list_.size(); ++i) {
		if (i > 0) {
			exprs += ", ";
		}
		exprs += select_list_[i] ? select_list_[i]->GetName() : std::string("<none>");
	}
	return {std::string("Window: ") + exprs};
}

StreamingWindowNode::StreamingWindowNode(NodeID node_id, PipelineNodeRef child, std::vector<ExpressionRef> select_list,
                                         std::vector<LogicalType> output_types)
    : ctx_(InheritPipelineNodeContext(child, node_id, "StreamingWindow")),
      config_(BuildWindowConfig(child, select_list, output_types)), child_(std::move(child)),
      select_list_(std::move(select_list)), output_types_(std::move(output_types)) {
	ValidateStreamingWindowDistribution(child_, select_list_);
}

std::vector<PipelineNodeRef> StreamingWindowNode::children() const {
	return {child_};
}

SubmittableTaskStream<WorkerTask> StreamingWindowNode::produce_tasks(PlanExecutionContext &plan_context) {
	auto input_stream = child_->produce_tasks(plan_context);
	const auto *select_list_ptr = &select_list_;
	const auto *output_types_ptr = &output_types_;
	return input_stream.pipeline_instruction(
	    shared_from_this(),
	    [select_list_ptr, output_types_ptr](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
		    if (select_list_ptr->empty()) {
			    return input_plan;
		    }
		    auto select_list = CopyWindowSelectList(*select_list_ptr);
		    auto types = *output_types_ptr;
		    idx_t estimated_cardinality = input_plan->Root().estimated_cardinality;

		    auto &old_root = input_plan->Root();
		    auto &window_op = input_plan->Make<duckdb::PhysicalStreamingWindow>(
		        std::move(types), std::move(select_list), estimated_cardinality);
		    window_op.children.push_back(old_root);
		    input_plan->SetRoot(window_op);
		    return input_plan;
	    },
	    plan_context.client_context());
}

std::vector<std::string> StreamingWindowNode::multiline_display(bool /*verbose*/) const {
	std::string exprs;
	for (size_t i = 0; i < select_list_.size(); ++i) {
		if (i > 0) {
			exprs += ", ";
		}
		exprs += select_list_[i] ? select_list_[i]->GetName() : std::string("<none>");
	}
	return {std::string("StreamingWindow: ") + exprs};
}

} // namespace distributed
} // namespace duckdb

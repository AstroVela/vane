// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/translator.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/execution/distributed/pipeline_node/filter.hpp"
#include "duckdb/execution/distributed/pipeline_node/local_exchange_passthrough.hpp"
#include "duckdb/execution/distributed/pipeline_node/projection.hpp"
#include "duckdb/execution/distributed/pipeline_node/shuffles/repartition.hpp"
#include "duckdb/execution/distributed/pipeline_node/vllm.hpp"
#include "duckdb/execution/operator/exchange/physical_local_exchange.hpp"
#include "duckdb/execution/operator/exchange/physical_repartition.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/projection/physical_vllm.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

namespace duckdb {
namespace distributed {

namespace {

PipelineNodeRef FirstChildImpl(const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	if (children.empty()) {
		return nullptr;
	}
	return children[0]->inner();
}

} // namespace

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateFilter(
    const PhysicalFilter &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	auto child_impl = FirstChildImpl(children);
	ExpressionRef predicate_ref = nullptr;
	if (op.expression) {
		auto pred_copy = op.expression->Copy();
		predicate_ref = ExpressionRef(pred_copy.release());
	}
	return std::make_shared<FilterNode>(get_next_pipeline_node_id(), child_impl, predicate_ref);
}

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateProjection(
    const PhysicalProjection &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	auto child_impl = FirstChildImpl(children);
	std::vector<ExpressionRef> projections;
	auto child_schema = child_impl ? child_impl->config().schema() : nullptr;
	auto child_names = duckdb::distributed::GetSchemaNames(child_schema);
	std::vector<std::string> projection_names;
	projection_names.reserve(op.select_list.size());
	for (auto &expr : op.select_list) {
		if (!expr) {
			continue;
		}
		auto copy = expr->Copy();
		if (!child_names.empty() && copy->GetExpressionClass() == ExpressionClass::BOUND_REF) {
			auto &ref = copy->Cast<BoundReferenceExpression>();
			if (ref.index < child_names.size()) {
				const auto &child_name = child_names[ref.index];
				const auto &alias = copy->GetAlias();
				if (!child_name.empty() && (alias.empty() || (!alias.empty() && alias[0] == '#'))) {
					copy->SetAlias(child_name);
				}
			}
		}
		projection_names.push_back(copy->GetName());
		projections.emplace_back(ExpressionRef(copy.release()));
	}
	SchemaRef schema = nullptr;
	if (!op.GetTypes().empty()) {
		if (!projection_names.empty() && projection_names.size() == op.GetTypes().size()) {
			schema = MakeSchemaRef(op.GetTypes(), projection_names);
		} else {
			schema = MakeSchemaRef(op.GetTypes());
		}
	}
	return std::make_shared<ProjectionNode>(get_next_pipeline_node_id(), child_impl, std::move(projections),
	                                        std::move(projection_names), schema);
}

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateLocalExchange(
    const PhysicalLocalExchange &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	auto child_impl = FirstChildImpl(children);
	if (!child_impl) {
		throw InternalException("LOCAL_EXCHANGE requires a child node");
	}
	SchemaRef schema = nullptr;
	if (!op.GetTypes().empty()) {
		schema = MakeSchemaRef(op.GetTypes());
	}
	auto spec = op.repartition_spec;
	if (!spec) {
		spec = RepartitionSpec::create_random(0);
	}
	return std::make_shared<LocalExchangePassthroughNode>(get_next_pipeline_node_id(), child_impl, std::move(spec),
	                                                      std::move(schema), op.estimated_cardinality);
}

std::shared_ptr<DistributedPipelineNode> PhysicalPlanToPipelineNodeTranslator::TranslateRepartition(
    const PhysicalRepartition &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	if (children.empty() || !children[0]) {
		throw InternalException("REPARTITION requires a child node");
	}
	SchemaRef schema = nullptr;
	if (!op.GetTypes().empty()) {
		schema = MakeSchemaRef(op.GetTypes());
	}
	auto spec = op.repartition_spec;
	if (!spec) {
		spec = RepartitionSpec::create_random(0);
	}
	auto shuffle_result = gen_shuffle_node(spec, schema, children[0]);
	if (!shuffle_result) {
		throw InternalException("Failed to create RepartitionNode for REPARTITION: %s", shuffle_result.error().what());
	}
	return shuffle_result.value();
}

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateVLLMProject(
    const PhysicalVLLM &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	auto child_impl = FirstChildImpl(children);
	ExpressionRef prompt_expr_ref = nullptr;
	if (op.prompt_expr) {
		auto copy = op.prompt_expr->Copy();
		prompt_expr_ref = ExpressionRef(copy.release());
	}
	duckdb::vector<LogicalType> output_types;
	output_types.reserve(op.GetTypes().size());
	for (auto &type : op.GetTypes()) {
		output_types.push_back(type);
	}
	return std::make_shared<VLLMProjectNode>(get_next_pipeline_node_id(), child_impl, std::move(prompt_expr_ref),
	                                         op.model, op.options, op.output_column_name, std::move(output_types));
}

} // namespace distributed
} // namespace duckdb

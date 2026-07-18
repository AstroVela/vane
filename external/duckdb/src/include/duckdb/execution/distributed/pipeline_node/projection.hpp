// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/operator/exchange/repartition.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/common/types.hpp"
#include <unordered_map>

namespace duckdb {
namespace distributed {

class ProjectionNode : public PipelineNodeImpl, public std::enable_shared_from_this<ProjectionNode> {
public:
	ProjectionNode(NodeID node_id, PipelineNodeRef child, std::vector<ExpressionRef> projection,
	               std::vector<std::string> projection_names, SchemaRef schema)
	    : ctx_(InheritPipelineNodeContext(child, node_id, "Projection")),
	      config_(std::move(schema), child ? child->config().execution_config() : DuckDBExecutionConfigRef(),
	              child ? child->config().clustering_spec() : ClusteringSpec::unknown_with_num_partitions(1)),
	      child_(std::move(child)), projection_(std::move(projection)), projection_names_(std::move(projection_names)) {
	}

	std::string name() const override {
		return "Projection";
	}
	NodeID node_id() const override {
		return ctx_.node_id();
	}
	const PipelineNodeContext &context() const override {
		return ctx_;
	}
	const PipelineNodeConfig &config() const override {
		return config_;
	}

	std::vector<PipelineNodeRef> children() const override {
		return {child_};
	}

	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) override {
		auto input_stream = child_->produce_tasks(plan_context);
		auto input_schema = child_ ? child_->config().schema() : SchemaRef();

		auto projection = projection_;
		auto projection_names = projection_names_;
		auto schema = config_.schema();
		auto input_schema_capture = input_schema;
		return input_stream.pipeline_instruction(
		    shared_from_this(),
		    [projection, projection_names, schema,
		     input_schema_capture](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
			    if (projection.empty()) {
				    return input_plan;
			    }
			    auto input_types = input_plan->Root().GetTypes();
			    auto input_names = duckdb::distributed::GetSchemaNames(input_schema_capture);
			    std::unordered_map<std::string, idx_t> name_to_index;
			    if (!input_names.empty()) {
				    name_to_index.reserve(input_names.size());
				    for (idx_t idx = 0; idx < input_names.size(); ++idx) {
					    const auto &name = input_names[idx];
					    if (!name.empty() && name_to_index.find(name) == name_to_index.end()) {
						    name_to_index.emplace(name, idx);
					    }
				    }
			    }

			    auto schema_types = [&](const SchemaRef &schema) -> duckdb::vector<duckdb::LogicalType> {
				    duckdb::vector<duckdb::LogicalType> types;
				    if (!schema) {
					    return types;
				    }
				    if (schema->id() == duckdb::LogicalTypeId::STRUCT) {
					    const auto &children = duckdb::StructType::GetChildTypes(*schema);
					    types.reserve(children.size());
					    for (const auto &child : children) {
						    types.push_back(child.second);
					    }
				    } else {
					    types.push_back(*schema);
				    }
				    return types;
			    };

			    std::function<void(duckdb::Expression &, const duckdb::vector<duckdb::LogicalType> &)> fix_ref_types;
			    fix_ref_types = [&](duckdb::Expression &expr,
			                        const duckdb::vector<duckdb::LogicalType> &types) -> void {
				    if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
					    auto &ref = expr.Cast<duckdb::BoundReferenceExpression>();
					    if (ref.index < types.size() && ref.return_type != types[ref.index]) {
						    ref.return_type = types[ref.index];
					    }
				    }
				    duckdb::ExpressionIterator::EnumerateChildren(
				        expr, [&](duckdb::Expression &child) { fix_ref_types(child, types); });
			    };
			    std::function<void(duckdb::Expression &)> fix_ref_names;
			    fix_ref_names = [&](duckdb::Expression &expr) -> void {
				    if (!name_to_index.empty() && expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
					    auto &ref = expr.Cast<duckdb::BoundReferenceExpression>();
					    const auto old_index = ref.index;
					    if (old_index < input_types.size() && ref.return_type == input_types[old_index]) {
						    return;
					    }
					    const auto &alias = ref.GetAlias();
					    if (!alias.empty()) {
						    auto it = name_to_index.find(alias);
						    if (it != name_to_index.end()) {
							    ref.index = it->second;
							    if (ref.index < input_types.size()) {
								    ref.return_type = input_types[ref.index];
							    }
						    }
					    }
				    }
				    duckdb::ExpressionIterator::EnumerateChildren(
				        expr, [&](duckdb::Expression &child) { fix_ref_names(child); });
			    };

			    // Build select_list and output types from projection expressions
			    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> select_list;
			    duckdb::vector<duckdb::LogicalType> out_types;
			    auto expected_types = schema_types(schema);
			    idx_t expected_count = 0;
			    for (auto &expr_ref : projection) {
				    if (expr_ref) {
					    expected_count++;
				    }
			    }
			    const bool use_expected_types = !expected_types.empty() && expected_types.size() == expected_count;
			    idx_t out_idx = 0;
			    for (auto &expr_ref : projection) {
				    if (!expr_ref)
					    continue;
				    auto copy = expr_ref->Copy();
				    if (out_idx < projection_names.size()) {
					    const auto &proj_name = projection_names[out_idx];
					    if (!proj_name.empty() && copy->GetAlias().empty()) {
						    copy->SetAlias(proj_name);
					    }
				    }
				    fix_ref_names(*copy);
				    fix_ref_types(*copy, input_types);
				    if (use_expected_types) {
					    const auto &expected_type = expected_types[out_idx];
					    if (copy->return_type != expected_type) {
						    copy = duckdb::BoundCastExpression::AddDefaultCastToType(std::move(copy), expected_type);
					    }
				    }
				    out_types.push_back(copy->return_type);
				    select_list.push_back(std::move(copy));
				    out_idx++;
			    }
			    idx_t estimated_cardinality = input_plan->Root().estimated_cardinality;
			    // Attach the existing root as the child of the new projection so
			    // plan structure remains valid. Use the current root before
			    // replacing it with the projection.
			    auto &old_root = input_plan->Root();
			    auto &proj = input_plan->Make<::duckdb::PhysicalProjection>(out_types, std::move(select_list),
			                                                                estimated_cardinality);
			    proj.children.push_back(old_root);
			    input_plan->SetRoot(proj);
			    // Return the updated plan (not the operator) to match the
			    // expected DuckPhysicalPlanRef return type for the
			    // plan-builder lambda.
			    return input_plan;
		    },
		    plan_context.client_context());
	}

	std::vector<std::string> multiline_display(bool verbose) const override {
		std::string projections;
		for (size_t i = 0; i < projection_.size(); ++i) {
			if (i > 0)
				projections += ", ";
			projections += projection_[i] ? projection_[i]->GetName() : std::string("<none>");
		}
		return {std::string("Project: ") + projections};
	}

private:
	PipelineNodeContext ctx_;
	PipelineNodeConfig config_;
	PipelineNodeRef child_;
	std::vector<ExpressionRef> projection_;
	std::vector<std::string> projection_names_;
};

} // namespace distributed
} // namespace duckdb

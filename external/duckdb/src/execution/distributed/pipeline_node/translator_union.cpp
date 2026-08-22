// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/translator.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/execution/distributed/pipeline_node/union.hpp"
#include "duckdb/execution/operator/set/physical_union.hpp"

namespace duckdb {
namespace distributed {

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateUnion(
    const PhysicalUnion &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	if (op.children.empty() || children.size() != op.children.size()) {
		throw InvalidInputException("Distributed UNION requires at least one fully translated branch");
	}
	if (op.GetTypes().empty()) {
		throw InvalidInputException("Distributed UNION requires a non-empty output schema");
	}

	for (idx_t child_idx = 0; child_idx < children.size(); child_idx++) {
		const auto &child = children[child_idx];
		if (!child || !child->config().schema() || !child->config().clustering_spec()) {
			throw InvalidInputException("Distributed UNION branch %llu is missing pipeline metadata",
			                            static_cast<unsigned long long>(child_idx));
		}
		if (op.children[child_idx].get().GetTypes() != op.GetTypes() ||
		    GetSchemaTypes(child->config().schema()) != op.GetTypes()) {
			throw InvalidInputException("Distributed UNION branch %llu schema does not match its output schema",
			                            static_cast<unsigned long long>(child_idx));
		}
	}

	auto output_names = GetSchemaNames(children.front()->config().schema());
	SchemaRef schema;
	if (output_names.size() == op.GetTypes().size()) {
		schema = MakeSchemaRef(op.GetTypes(), output_names);
	} else {
		schema = MakeSchemaRef(op.GetTypes());
	}

	return std::make_shared<UnionNode>(get_next_pipeline_node_id(), plan_config_, children, std::move(schema),
	                                   op.allow_out_of_order);
}

} // namespace distributed
} // namespace duckdb

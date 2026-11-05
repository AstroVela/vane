// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/translator.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/environment.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/multi_file/multi_file_states.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/distributed/pipeline_node/empty_result_source.hpp"
#include "duckdb/execution/distributed/pipeline_node/materialized_cte.hpp"
#include "duckdb/execution/operator/set/physical_cte.hpp"
#include "duckdb/execution/distributed/pipeline_node/scan_source.hpp"
#include "duckdb/execution/distributed/pipeline_node/translator_scan.hpp"
#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/execution/operator/scan/physical_dummy_scan.hpp"
#include "duckdb/execution/operator/scan/physical_empty_result.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"

namespace duckdb {
namespace distributed {

namespace {

unique_ptr<duckdb::ColumnDataCollection> CopyColumnDataCollection(const duckdb::ColumnDataCollection &collection) {
	auto copy = duckdb::make_uniq<duckdb::ColumnDataCollection>(Allocator::DefaultAllocator(), collection.Types());
	for (auto &chunk : collection.Chunks()) {
		copy->Append(chunk);
	}
	return copy;
}

DuckPhysicalPlanRef MakeDummyScanPlan(const duckdb::PhysicalDummyScan &scan) {
	Allocator &alloc = Allocator::DefaultAllocator();
	auto plan = std::make_shared<duckdb::PhysicalPlan>(alloc);
	auto &dummy = plan->Make<duckdb::PhysicalDummyScan>(scan.GetTypes(), scan.estimated_cardinality);
	plan->SetRoot(dummy);
	return plan;
}

DuckPhysicalPlanRef MakeEmptyResultPlan(const duckdb::PhysicalEmptyResult &empty_result) {
	Allocator &alloc = Allocator::DefaultAllocator();
	auto plan = std::make_shared<duckdb::PhysicalPlan>(alloc);
	auto &empty = plan->Make<duckdb::PhysicalEmptyResult>(empty_result.GetTypes(), empty_result.estimated_cardinality);
	plan->SetRoot(empty);
	return plan;
}

DuckPhysicalPlanRef MakeColumnDataScanPlan(const duckdb::PhysicalColumnDataScan &scan) {
	Allocator &alloc = Allocator::DefaultAllocator();
	auto plan = std::make_shared<duckdb::PhysicalPlan>(alloc);

	auto collection = [&]() {
		if (!scan.collection) {
			return optionally_owned_ptr<duckdb::ColumnDataCollection>();
		}
		return optionally_owned_ptr<duckdb::ColumnDataCollection>(CopyColumnDataCollection(*scan.collection));
	}();

	// A standalone owned collection has no native CTE writer in its worker fragment.
	auto scan_type = scan.type == PhysicalOperatorType::CTE_SCAN ? PhysicalOperatorType::COLUMN_DATA_SCAN : scan.type;
	auto &scan_op = plan->Make<duckdb::PhysicalColumnDataScan>(scan.GetTypes(), scan_type, scan.estimated_cardinality,
	                                                           std::move(collection))
	                    .Cast<duckdb::PhysicalColumnDataScan>();
	scan_op.cte_index = scan.cte_index;
	scan_op.delim_index = scan.delim_index;
	plan->SetRoot(scan_op);
	return plan;
}

DuckDBExecutionConfigRef ResolveExecutionConfig(const PlanConfig &plan_config) {
	auto exec_cfg = plan_config.config;
	if (!exec_cfg) {
		exec_cfg = std::make_shared<duckdb::distributed::DuckDBExecutionConfig>(
		    duckdb::distributed::DuckDBExecutionConfig::from_env());
	}
	return exec_cfg;
}

SchemaRef MakeSchemaFromTypes(const duckdb::vector<LogicalType> &types) {
	if (types.empty()) {
		return nullptr;
	}
	return MakeSchemaRef(types);
}

std::shared_ptr<DistributedPipelineNode> MakeScanSourceNode(PipelineNodeContext context, DuckPhysicalPlanRef scan_plan,
                                                            std::vector<duckdb::distributed::ScanSplit> scan_splits,
                                                            SchemaRef schema, DuckDBExecutionConfigRef exec_cfg,
                                                            bool require_scan_splits) {
	auto scan_node = std::make_shared<ScanSourceNode>(std::move(context), std::move(scan_plan), std::move(scan_splits),
	                                                  std::move(schema), std::move(exec_cfg), require_scan_splits);
	return std::make_shared<DistributedPipelineNode>(scan_node);
}

} // namespace

void PhysicalPlanToPipelineNodeTranslator::VisitMaterializedCTE(PhysicalOperator &op) {
	const EnvironmentVariable setting("VANE_RAY_MATERIALIZED_CTE");
	if (setting.GetValue()) {
		auto enabled = StringUtil::Lower(setting.GetValue());
		if (enabled == "0" || enabled == "false" || enabled == "off" || enabled == "no") {
			throw NotImplementedException(
			    "Distributed materialized CTE execution is disabled by VANE_RAY_MATERIALIZED_CTE");
		}
	}
	auto &cte = op.Cast<PhysicalCTE>();
	if (cte.children.size() != 2) {
		throw InternalException("Distributed materialized CTE requires a definition and a consumer");
	}
	VisitOperator(cte.children[0]);
	auto definition = std::move(node_stack_.back());
	node_stack_.pop_back();
	auto input = MakeMaterializedCTEInput(
	    MakePipelineNodeContext(plan_config_.query_idx, plan_config_.query_id, get_next_pipeline_node_id(), "CTEInput"),
	    std::move(definition), ordered_ctes_.count(cte.table_index) != 0);
	if (!cte_inputs_.emplace(cte.table_index, input).second) {
		throw InternalException("Duplicate distributed CTE binding %llu", cte.table_index);
	}
	VisitOperator(cte.children[1]);
	auto body = std::move(node_stack_.back());
	node_stack_.pop_back();
	cte_inputs_.erase(cte.table_index);
	node_stack_.push_back(MakeMaterializedCTEScope(
	    MakePipelineNodeContext(plan_config_.query_idx, plan_config_.query_id, get_next_pipeline_node_id(), "CTE"),
	    std::move(input), std::move(body)));
}

std::shared_ptr<DistributedPipelineNode>
PhysicalPlanToPipelineNodeTranslator::TranslateDummyScanSource(const PhysicalDummyScan &op) {
	auto scan_plan = MakeDummyScanPlan(op);
	std::vector<duckdb::distributed::ScanSplit> scan_splits;
	auto schema = MakeSchemaFromTypes(op.GetTypes());
	auto exec_cfg = ResolveExecutionConfig(plan_config_);
	return MakeScanSourceNode(MakePipelineNodeContext(plan_config_.query_idx, plan_config_.query_id,
	                                                  get_next_pipeline_node_id(), "ScanSource"),
	                          std::move(scan_plan), std::move(scan_splits), schema, exec_cfg, false);
}

std::shared_ptr<DistributedPipelineNode>
PhysicalPlanToPipelineNodeTranslator::TranslateEmptyResultSource(const PhysicalEmptyResult &op) {
	auto empty_plan = MakeEmptyResultPlan(op);
	auto schema = MakeSchemaFromTypes(op.GetTypes());
	auto exec_cfg = ResolveExecutionConfig(plan_config_);
	auto empty_result = std::make_shared<EmptyResultSourceNode>(
	    MakePipelineNodeContext(plan_config_.query_idx, plan_config_.query_id, get_next_pipeline_node_id(),
	                            "EmptyResultSource"),
	    std::move(empty_plan), std::move(schema), std::move(exec_cfg));
	return std::make_shared<DistributedPipelineNode>(std::move(empty_result));
}

std::shared_ptr<DistributedPipelineNode>
PhysicalPlanToPipelineNodeTranslator::TranslateColumnDataScanSource(const PhysicalColumnDataScan &op) {
	if (!op.collection) {
		throw NotImplementedException("Distributed pipeline does not support %s without a collection", op.GetName());
	}

	auto scan_plan = MakeColumnDataScanPlan(op);
	std::vector<duckdb::distributed::ScanSplit> scan_splits;
	auto schema = MakeSchemaFromTypes(op.GetTypes());
	auto exec_cfg = ResolveExecutionConfig(plan_config_);

	int scan_node_id;
	if (op.source_node_id.IsValid()) {
		scan_node_id = static_cast<int>(op.source_node_id.GetIndex());
		if (scan_node_id > pipeline_node_id_counter_) {
			pipeline_node_id_counter_ = scan_node_id;
		}
	} else {
		scan_node_id = get_next_pipeline_node_id();
	}

	return MakeScanSourceNode(
	    MakePipelineNodeContext(plan_config_.query_idx, plan_config_.query_id, scan_node_id, "ScanSource"),
	    std::move(scan_plan), std::move(scan_splits), schema, exec_cfg, false);
}

std::shared_ptr<DistributedPipelineNode>
PhysicalPlanToPipelineNodeTranslator::TranslateTableScanSource(PhysicalTableScan &op) {
	auto exec_cfg = ResolveExecutionConfig(plan_config_);
	int scan_node_id;
	DuckPhysicalPlanRef scan_plan;
	std::vector<duckdb::distributed::ScanSplit> scan_splits;

	if (op.extra_info.scan_node_id.IsValid()) {
		scan_node_id = static_cast<int>(op.extra_info.scan_node_id.GetIndex());
		if (scan_node_id > pipeline_node_id_counter_) {
			pipeline_node_id_counter_ = scan_node_id;
		}
	} else {
		scan_node_id = get_next_pipeline_node_id();
		op.extra_info.scan_node_id = optional_idx(static_cast<idx_t>(scan_node_id));
	}
	if (!op.extra_info.scan_group_id.IsValid()) {
		op.extra_info.scan_group_id = op.extra_info.scan_node_id;
	}

	scan_plan = MakeTableScanPlan(op);
	scan_splits = MakeTableScanSplits(op, *exec_cfg, plan_config_.db, client_context_);
	if (!scan_plan || !scan_plan->HasRoot()) {
		throw std::runtime_error(
		    "[translate.cpp] TABLE_SCAN: MakeTableScanPlan returned null/empty plan for function " + op.function.name);
	}
	if (scan_plan && scan_plan->HasRoot()) {
		auto &root = scan_plan->Root();
		if (root.type == PhysicalOperatorType::TABLE_SCAN) {
			auto &scan_root = root.Cast<PhysicalTableScan>();
			scan_root.extra_info.scan_node_id = optional_idx(static_cast<idx_t>(scan_node_id));
			scan_root.extra_info.scan_group_id = op.extra_info.scan_group_id;
		}
	}

	auto schema = MakeTableScanSchema(op, op.GetTypes());
	return MakeScanSourceNode(
	    MakePipelineNodeContext(plan_config_.query_idx, plan_config_.query_id, scan_node_id, "ScanSource"),
	    std::move(scan_plan), std::move(scan_splits), schema, exec_cfg, true);
}

} // namespace distributed
} // namespace duckdb

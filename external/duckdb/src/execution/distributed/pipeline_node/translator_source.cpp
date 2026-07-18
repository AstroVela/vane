// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/translator.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/distributed/pipeline_node/scan_source.hpp"
#include "duckdb/execution/distributed/pipeline_node/translator_scan.hpp"
#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/execution/operator/scan/physical_dummy_scan.hpp"
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

DuckPhysicalPlanRef MakeColumnDataScanPlan(const duckdb::PhysicalColumnDataScan &scan) {
	Allocator &alloc = Allocator::DefaultAllocator();
	auto plan = std::make_shared<duckdb::PhysicalPlan>(alloc);

	auto collection = [&]() {
		if (!scan.collection) {
			return optionally_owned_ptr<duckdb::ColumnDataCollection>();
		}
		return optionally_owned_ptr<duckdb::ColumnDataCollection>(CopyColumnDataCollection(*scan.collection));
	}();

	auto &scan_op = plan->Make<duckdb::PhysicalColumnDataScan>(scan.GetTypes(), scan.type, scan.estimated_cardinality,
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

std::shared_ptr<DistributedPipelineNode>
MakeScanSourceNode(PipelineNodeContext context, DuckPhysicalPlanRef scan_plan,
                   std::vector<duckdb::distributed::ScanTaskDescriptor> scan_tasks, SchemaRef schema,
                   DuckDBExecutionConfigRef exec_cfg, bool is_external_scan) {
	auto scan_node = std::make_shared<ScanSourceNode>(std::move(context), std::move(scan_plan), std::move(scan_tasks),
	                                                  std::move(schema), std::move(exec_cfg), is_external_scan);
	return std::make_shared<DistributedPipelineNode>(scan_node);
}

} // namespace

std::shared_ptr<DistributedPipelineNode>
PhysicalPlanToPipelineNodeTranslator::TranslateCTESource(PhysicalOperator &op) {
	DuckPhysicalPlanRef scan_plan;
	if (plan_ && plan_->HasRoot() && &plan_->Root() == &op) {
		scan_plan = plan_;
	} else {
		Allocator &alloc = Allocator::DefaultAllocator();
		scan_plan = std::make_shared<duckdb::PhysicalPlan>(alloc);
		scan_plan->SetRoot(op);
	}

	std::vector<duckdb::distributed::ScanTaskDescriptor> scan_tasks;
	auto schema = MakeSchemaFromTypes(op.GetTypes());
	auto exec_cfg = ResolveExecutionConfig(plan_config_);
	return MakeScanSourceNode(MakePipelineNodeContext(plan_config_.query_idx, plan_config_.query_id,
	                                                  get_next_pipeline_node_id(), "ScanSource"),
	                          std::move(scan_plan), std::move(scan_tasks), schema, exec_cfg, false);
}

std::shared_ptr<DistributedPipelineNode>
PhysicalPlanToPipelineNodeTranslator::TranslateDummyScanSource(const PhysicalDummyScan &op) {
	auto scan_plan = MakeDummyScanPlan(op);
	std::vector<duckdb::distributed::ScanTaskDescriptor> scan_tasks;
	auto schema = MakeSchemaFromTypes(op.GetTypes());
	auto exec_cfg = ResolveExecutionConfig(plan_config_);
	return MakeScanSourceNode(MakePipelineNodeContext(plan_config_.query_idx, plan_config_.query_id,
	                                                  get_next_pipeline_node_id(), "ScanSource"),
	                          std::move(scan_plan), std::move(scan_tasks), schema, exec_cfg, false);
}

std::shared_ptr<DistributedPipelineNode>
PhysicalPlanToPipelineNodeTranslator::TranslateColumnDataScanSource(const PhysicalColumnDataScan &op) {
	if (!op.collection) {
		throw NotImplementedException("Distributed pipeline does not support %s without a collection", op.GetName());
	}

	auto scan_plan = MakeColumnDataScanPlan(op);
	std::vector<duckdb::distributed::ScanTaskDescriptor> scan_tasks;
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
	    std::move(scan_plan), std::move(scan_tasks), schema, exec_cfg, false);
}

std::shared_ptr<DistributedPipelineNode>
PhysicalPlanToPipelineNodeTranslator::TranslateTableScanSource(PhysicalTableScan &op) {
	auto exec_cfg = ResolveExecutionConfig(plan_config_);
	auto scan_node_id = get_next_pipeline_node_id();
	DuckPhysicalPlanRef scan_plan;
	std::vector<duckdb::distributed::ScanTaskDescriptor> scan_tasks;

	if (op.extra_info.scan_node_id.IsValid()) {
		scan_node_id = static_cast<int>(op.extra_info.scan_node_id.GetIndex());
		if (scan_node_id > pipeline_node_id_counter_) {
			pipeline_node_id_counter_ = scan_node_id;
		}
	} else {
		op.extra_info.scan_node_id = optional_idx(static_cast<idx_t>(scan_node_id));
	}
	if (!op.extra_info.scan_group_id.IsValid()) {
		op.extra_info.scan_group_id = op.extra_info.scan_node_id;
	}

	scan_plan = MakeTableScanPlan(op);
	scan_tasks = MakeTableScanTasks(op, *exec_cfg, plan_config_.db);
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
	    std::move(scan_plan), std::move(scan_tasks), schema, exec_cfg, true);
}

} // namespace distributed
} // namespace duckdb

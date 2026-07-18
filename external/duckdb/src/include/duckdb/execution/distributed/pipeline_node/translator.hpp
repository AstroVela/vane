// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once
#include <memory>
#include <vector>
#include <string>

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/physical_operator_visitor.hpp"
#include "duckdb/execution/distributed/plan/plan_config.hpp"
#include "duckdb/execution/distributed/pipeline_node/aggregate.hpp"

namespace duckdb {
class RepartitionSpec;
class PhysicalBatchCopyToFile;
class PhysicalCopyToFile;
class PhysicalOperator;
class PhysicalDelimJoin;
class PhysicalHashJoin;
class PhysicalNestedLoopJoin;
class PhysicalHashAggregate;
class PhysicalColumnDataScan;
class PhysicalDummyScan;
class PhysicalExpressionScan;
class PhysicalFilter;
class PhysicalLimit;
class PhysicalLimitPercent;
class PhysicalLocalExchange;
class PhysicalOrder;
class PhysicalPartitionedAggregate;
class PhysicalPerfectHashAggregate;
class PhysicalPivot;
class PhysicalProjection;
class PhysicalRepartition;
class PhysicalReservoirSample;
class PhysicalStreamingLimit;
class PhysicalStreamingSample;
class PhysicalStreamingUDF;
class PhysicalStreamingWindow;
class PhysicalTableInOutFunction;
class PhysicalTableScan;
class PhysicalTopN;
class PhysicalUngroupedAggregate;
class PhysicalUnnest;
class PhysicalVLLM;
class PhysicalWindow;
} // namespace duckdb

namespace duckdb {
class ClientContext;
namespace distributed {
class ExchangeManager;

// Use the PhysicalOperatorVisitor to traverse a DuckDB PhysicalPlan's operator tree.
class PhysicalPlanToPipelineNodeTranslator : public ::duckdb::PhysicalOperatorVisitor {
private:
	PlanConfig plan_config_;
	int pipeline_node_id_counter_ = 0;
	DuckPhysicalPlanRef plan_;
	std::shared_ptr<ExchangeManager> exchange_mgr_;

	// 辅助方法：获取下一个节点ID
	int get_next_pipeline_node_id() {
		return ++pipeline_node_id_counter_;
	}

public:
	PhysicalPlanToPipelineNodeTranslator(PlanConfig plan_config, DuckPhysicalPlanRef plan,
	                                     ClientContext *client_context = nullptr);

	// Static helper: convert a DuckDB PhysicalPlan into a DistributedPipelineNode.
	// This mirrors the Rust helper `physical_plan_to_pipeline_node` but implemented
	// as a C++ static method on the translator class. Implementation uses the
	// PhysicalOperatorVisitor to traverse the operator tree in post-order.
	static DuckDBResult<std::shared_ptr<DistributedPipelineNode>>
	physical_plan_to_pipeline_node(PlanConfig plan_config, DuckPhysicalPlanRef plan,
	                               ClientContext *client_context = nullptr);

	// Override VisitOperator (post-order): children are visited first by calling
	// the base traversal helper, then we peek/pop child results from node_stack_
	void VisitOperator(::duckdb::PhysicalOperator &op) override;

private:
	// stack of constructed DistributedPipelineNodes corresponding to visited operators
	std::vector<std::shared_ptr<DistributedPipelineNode>> node_stack_;

	// 生成 shuffle 节点（实现自 Rust 逻辑）
	DuckDBResult<std::shared_ptr<DistributedPipelineNode>>
	gen_shuffle_node(std::shared_ptr<RepartitionSpec> repartition_spec, SchemaRef schema,
	                 std::shared_ptr<DistributedPipelineNode> child);

	// 生成无预聚合的聚合节点（GroupBy/Shuffle/Gather）
	std::shared_ptr<DistributedPipelineNode> gen_without_pre_agg(std::shared_ptr<DistributedPipelineNode> input_node,
	                                                             const std::vector<BoundExpr> &group_by,
	                                                             const std::vector<BoundAggExpr> &aggregations,
	                                                             SchemaRef output_schema,
	                                                             const std::vector<BoundExpr> &partition_by);

	// 生成有预聚合的聚合节点（两阶段聚合：pre-agg -> shuffle -> final agg -> project）
	std::shared_ptr<DistributedPipelineNode> gen_with_pre_agg(std::shared_ptr<DistributedPipelineNode> input_node,
	                                                          const GroupByAggSplit &split_details,
	                                                          SchemaRef output_schema);

	// 生成聚合节点（主入口，选择有或无预聚合路径）
	std::shared_ptr<DistributedPipelineNode> gen_agg_nodes(std::shared_ptr<DistributedPipelineNode> input_node,
	                                                       const std::vector<BoundExpr> &group_by,
	                                                       const std::vector<BoundAggExpr> &aggregations,
	                                                       SchemaRef output_schema,
	                                                       const std::vector<BoundExpr> &partition_by);

	// 生成 gather 节点（使用 RepartitionNode with num_partitions=1）
	std::shared_ptr<DistributedPipelineNode> gen_gather_node(std::shared_ptr<DistributedPipelineNode> input_node);

	std::shared_ptr<PipelineNodeImpl>
	TranslateHashJoin(const PhysicalHashJoin &op,
	                  const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<PipelineNodeImpl>
	TranslateDelimJoin(const PhysicalDelimJoin &op,
	                   const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<PipelineNodeImpl>
	TranslateNestedLoopJoin(const PhysicalNestedLoopJoin &op,
	                        const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<DistributedPipelineNode>
	TranslateHashGroupBy(const PhysicalHashAggregate &op,
	                     const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<DistributedPipelineNode>
	TranslatePerfectHashGroupBy(const PhysicalPerfectHashAggregate &op,
	                            const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<DistributedPipelineNode>
	TranslatePartitionedAggregate(const PhysicalPartitionedAggregate &op,
	                              const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<DistributedPipelineNode>
	TranslateUngroupedAggregate(const PhysicalUngroupedAggregate &op,
	                            const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<DistributedPipelineNode> TranslateCTESource(PhysicalOperator &op);

	std::shared_ptr<DistributedPipelineNode> TranslateDummyScanSource(const PhysicalDummyScan &op);

	std::shared_ptr<DistributedPipelineNode> TranslateColumnDataScanSource(const PhysicalColumnDataScan &op);

	std::shared_ptr<DistributedPipelineNode> TranslateTableScanSource(PhysicalTableScan &op);

	std::shared_ptr<PipelineNodeImpl>
	TranslateFilter(const PhysicalFilter &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<PipelineNodeImpl>
	TranslateProjection(const PhysicalProjection &op,
	                    const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<PipelineNodeImpl>
	TranslateLocalExchange(const PhysicalLocalExchange &op,
	                       const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<DistributedPipelineNode>
	TranslateRepartition(const PhysicalRepartition &op,
	                     const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<PipelineNodeImpl>
	TranslateVLLMProject(const PhysicalVLLM &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<PipelineNodeImpl>
	TranslateLimit(const PhysicalLimit &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<PipelineNodeImpl>
	TranslateStreamingLimit(const PhysicalStreamingLimit &op,
	                        const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<PipelineNodeImpl>
	TranslateLimitPercent(const PhysicalLimitPercent &op,
	                      const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<PipelineNodeImpl>
	TranslateOrderBy(const PhysicalOrder &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<PipelineNodeImpl>
	TranslateTopN(const PhysicalTopN &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<PipelineNodeImpl>
	TranslateReservoirSample(const PhysicalReservoirSample &op,
	                         const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<PipelineNodeImpl>
	TranslateStreamingSample(const PhysicalStreamingSample &op,
	                         const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<PipelineNodeImpl>
	TranslateCopyToFile(const PhysicalCopyToFile &op,
	                    const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<PipelineNodeImpl>
	TranslateBatchCopyToFile(const PhysicalBatchCopyToFile &op,
	                         const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<PipelineNodeImpl>
	TranslatePivot(const PhysicalPivot &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<PipelineNodeImpl>
	TranslateUnnest(const PhysicalUnnest &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<PipelineNodeImpl>
	TranslateTableInOut(const PhysicalTableInOutFunction &op,
	                    const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<PipelineNodeImpl>
	TranslateStreamingUDF(const PhysicalStreamingUDF &op,
	                      const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<PipelineNodeImpl>
	TranslateWindow(const PhysicalWindow &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<PipelineNodeImpl>
	TranslateStreamingWindow(const PhysicalStreamingWindow &op,
	                         const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);

	std::shared_ptr<PipelineNodeImpl>
	TranslateExpressionScan(const PhysicalExpressionScan &op,
	                        const std::vector<std::shared_ptr<DistributedPipelineNode>> &children);
};

// Backwards-compatible free function wrapper that delegates to the
// translator static helper. Keeps existing call sites simple.
inline DuckDBResult<std::shared_ptr<DistributedPipelineNode>>
physical_plan_to_pipeline_node(PlanConfig plan_config, DuckPhysicalPlanRef plan,
                               ClientContext *client_context = nullptr) {
	return PhysicalPlanToPipelineNodeTranslator::physical_plan_to_pipeline_node(std::move(plan_config), std::move(plan),
	                                                                            client_context);
}
} // namespace distributed
} // namespace duckdb

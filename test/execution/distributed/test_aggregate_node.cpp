// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb/execution/distributed/pipeline_node/aggregate.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/distributed/common_types.hpp"

using namespace duckdb;
using namespace duckdb::distributed;

// Minimal stub PipelineNodeImpl used for testing
class DummyNode : public PipelineNodeImpl {
public:
	DummyNode()
	    : config_(nullptr, nullptr, ClusteringSpec::unknown_with_num_partitions(4)), context_(0, "", 0, "dummy") {
	}

	const PipelineNodeContext &context() const override {
		return context_;
	}
	const PipelineNodeConfig &config() const override {
		return config_;
	}
	std::vector<PipelineNodeRef> children() const override {
		return {};
	}
	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) override {
		// Not used in these tests
		return SubmittableTaskStream<WorkerTask>(nullptr);
	}
	std::vector<std::string> multiline_display(bool verbose) const override {
		return {};
	}

private:
	PipelineNodeConfig config_;
	PipelineNodeContext context_;
};

TEST_CASE("AggregateNode uses node_name helper and config clustering_spec", "[distributed][aggregate]") {
	PlanConfig plan_cfg(0, "q", std::make_shared<DuckDBExecutionConfig>());

	// 1) empty group_by -> name is "Aggregate"
	{
		AggregateNode node(1, plan_cfg, {}, {}, nullptr, nullptr);
		REQUIRE(node.context().query_id() == plan_cfg.query_id);
		REQUIRE(node.context().node_name() == AggregateNode::node_name({}));
	}

	// 2) non-empty group_by -> name is "GroupBy Aggregate"
	{
		std::vector<BoundExprRef> group_by = {nullptr};
		AggregateNode node(2, plan_cfg, group_by, {}, nullptr, nullptr);
		REQUIRE(node.context().node_name() == AggregateNode::node_name(group_by));
	}

	// 3) when child provides clustering spec, config should pick it up
	{
		auto dummy_impl = std::make_shared<DummyNode>();
		auto child = std::make_shared<DistributedPipelineNode>(dummy_impl);
		AggregateNode node(3, plan_cfg, {}, {}, nullptr, child);
		REQUIRE(node.config().clustering_spec() != nullptr);
		REQUIRE(node.config().clustering_spec()->num_partitions() == 4);
	}
}

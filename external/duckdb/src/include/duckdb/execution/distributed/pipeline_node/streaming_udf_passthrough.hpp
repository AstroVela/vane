// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

// StreamingUDFPassthroughNode keeps STREAMING_UDF in the worker's
// native physical plan so DuckDB can build concurrent source/sink pipelines.
#pragma once

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/operator/projection/physical_udf_inout.hpp"
#include "duckdb/common/column_index.hpp"
#include "duckdb/common/optional_idx.hpp"

namespace duckdb {
namespace distributed {

class StreamingUDFPassthroughNode : public PipelineNodeImpl,
                                    public std::enable_shared_from_this<StreamingUDFPassthroughNode> {
public:
	StreamingUDFPassthroughNode(NodeID node_id, PipelineNodeRef child, TableFunction function,
	                            unique_ptr<FunctionData> bind_data, vector<ColumnIndex> column_ids,
	                            vector<column_t> projected_input, optional_idx ordinality_idx,
	                            vector<LogicalType> output_types, idx_t estimated_cardinality)
	    : ctx_(InheritPipelineNodeContext(child, node_id, "StreamingUDF")),
	      config_(BuildSchema(output_types), child ? child->config().execution_config() : DuckDBExecutionConfigRef(),
	              child ? child->config().clustering_spec() : ClusteringSpec::unknown_with_num_partitions(1)),
	      child_(std::move(child)), function_(std::move(function)), bind_data_(std::move(bind_data)),
	      column_ids_(std::move(column_ids)), projected_input_(std::move(projected_input)),
	      ordinality_idx_(ordinality_idx), output_types_(std::move(output_types)),
	      estimated_cardinality_(estimated_cardinality) {
	}

	std::string name() const override {
		return "StreamingUDF";
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
	const FunctionData *bind_data() const {
		return bind_data_.get();
	}

	std::vector<PipelineNodeRef> children() const override {
		return {child_};
	}

	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) override {
		auto input_stream = child_->produce_tasks(plan_context);
		auto self = shared_from_this();
		auto function = function_;
		auto bind_data_ptr = bind_data_.get();
		auto column_ids = column_ids_;
		auto projected_input = projected_input_;
		auto ordinality_idx = ordinality_idx_;
		auto output_types = output_types_;
		auto estimated_cardinality = estimated_cardinality_;

		return input_stream.pipeline_instruction(
		    self,
		    [function, bind_data_ptr, column_ids, projected_input, ordinality_idx, output_types,
		     estimated_cardinality](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
			    auto bind_copy = bind_data_ptr ? bind_data_ptr->Copy() : nullptr;
			    auto out_types = output_types;
			    if (out_types.empty()) {
				    out_types = input_plan->Root().GetTypes();
			    }
			    auto &old_root = input_plan->Root();
			    auto &streaming = input_plan->Make<::duckdb::PhysicalStreamingUDF>(
			        std::move(out_types), function, std::move(bind_copy), column_ids, estimated_cardinality,
			        projected_input);
			    auto &streaming_ref = streaming.Cast<::duckdb::PhysicalStreamingUDF>();
			    streaming_ref.ordinality_idx = ordinality_idx;
			    streaming_ref.children.push_back(old_root);
			    input_plan->SetRoot(streaming_ref);
			    return input_plan;
		    },
		    plan_context.client_context());
	}

	std::vector<std::string> multiline_display(bool) const override {
		return {"StreamingUDF (passthrough)"};
	}

private:
	static SchemaRef BuildSchema(const vector<LogicalType> &output_types) {
		if (output_types.empty()) {
			return nullptr;
		}
		return std::make_shared<duckdb::LogicalType>(output_types[0]);
	}

	PipelineNodeContext ctx_;
	PipelineNodeConfig config_;
	PipelineNodeRef child_;
	TableFunction function_;
	unique_ptr<FunctionData> bind_data_;
	vector<ColumnIndex> column_ids_;
	vector<column_t> projected_input_;
	optional_idx ordinality_idx_;
	vector<LogicalType> output_types_;
	idx_t estimated_cardinality_;
};

} // namespace distributed
} // namespace duckdb

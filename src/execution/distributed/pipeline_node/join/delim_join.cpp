// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/join/delim_join.hpp"

#include <utility>

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/operator/join/physical_left_delim_join.hpp"
#include "duckdb/execution/operator/join/physical_right_delim_join.hpp"
#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"

namespace duckdb {
namespace distributed {

static SchemaRef MakeSchema(const vector<LogicalType> &types) {
	return MakeSchemaRef(types);
}

DelimJoinNode::DelimJoinNode(NodeID node_id, const PhysicalDelimJoin &delim_join,
                             std::shared_ptr<DistributedPipelineNode> child, SchemaRef schema,
                             shared_ptr<DatabaseInstance> db)
    : context_(InheritPipelineNodeContext(child, node_id, delim_join.GetName())), child_(std::move(child)),
      delim_type_(delim_join.type), output_types_(delim_join.GetTypes()), delim_idx_(delim_join.delim_idx),
      estimated_cardinality_(delim_join.estimated_cardinality), join_bytes_(SerializeOperator(delim_join.join)),
      distinct_bytes_(SerializeOperator(delim_join.distinct)), db_(std::move(db)) {
	if (!schema) {
		schema = MakeSchema(output_types_);
	}
	if (child_) {
		config_ = PipelineNodeConfig(schema, child_->config().execution_config(), child_->config().clustering_spec());
	} else {
		config_ =
		    PipelineNodeConfig(schema, DuckDBExecutionConfigRef(), ClusteringSpec::unknown_with_num_partitions(1));
	}
}

std::shared_ptr<DistributedPipelineNode> DelimJoinNode::into_node() {
	return std::make_shared<DistributedPipelineNode>(shared_from_this());
}

std::vector<PipelineNodeRef> DelimJoinNode::children() const {
	if (child_) {
		return {child_->inner()};
	}
	return {};
}

std::string DelimJoinNode::SerializeOperator(const PhysicalOperator &op) {
	MemoryStream stream(Allocator::DefaultAllocator());
	SerializationOptions options;
	options.serialization_compatibility = SerializationCompatibility::Latest();
	options.serialize_default_values = true;
	BinarySerializer serializer(stream, options);
	serializer.Begin();
	op.Serialize(serializer);
	serializer.End();
	const auto size = stream.GetPosition();
	const auto *data = stream.GetData();
	return std::string(reinterpret_cast<const char *>(data), size);
}

unique_ptr<PhysicalOperator> DelimJoinNode::DeserializeOperator(const std::string &data, PhysicalPlan &plan,
                                                                shared_ptr<DatabaseInstance> db) {
	auto *buffer = reinterpret_cast<data_ptr_t>(const_cast<char *>(data.data()));
	MemoryStream stream(buffer, data.size());
	BinaryDeserializer deserializer(stream);
	// Use a local connection for deserialization to avoid relying on the caller's
	// ClientContext, but keep the same DatabaseInstance when available so UDFs remain visible.
	unique_ptr<DuckDB> local_db;
	unique_ptr<Connection> local_conn;
	if (db) {
		local_conn = make_uniq<Connection>(*db);
	} else {
		local_db = make_uniq<DuckDB>(nullptr);
		local_conn = make_uniq<Connection>(*local_db);
	}

	unique_ptr<PhysicalOperator> op;
	auto &local_ctx = *local_conn->context;
	local_ctx.RunFunctionInTransaction([&]() {
		auto &db = DatabaseInstance::GetDatabase(local_ctx);
		deserializer.Set<DatabaseInstance &>(db);
		deserializer.Set<ClientContext &>(local_ctx);
		deserializer.Begin();
		op = PhysicalOperator::Deserialize(deserializer, plan);
		deserializer.End();
	});
	return op;
}

void DelimJoinNode::GatherDelimScans(PhysicalOperator &op, vector<const_reference<PhysicalOperator>> &delim_scans,
                                     optional_idx delim_idx) {
	if (op.type == PhysicalOperatorType::DELIM_SCAN) {
		auto &scan = op.Cast<PhysicalColumnDataScan>();
		if (!delim_idx.IsValid() || scan.delim_index == delim_idx) {
			delim_scans.push_back(op);
		}
	}
	for (auto &child : op.children) {
		GatherDelimScans(child, delim_scans, delim_idx);
	}
}

SubmittableTaskStream<WorkerTask> DelimJoinNode::produce_tasks(PlanExecutionContext &plan_context) {
	if (!child_) {
		return SubmittableTaskStream<WorkerTask>::from_receiver(Receiver<SubmittableTask<WorkerTask>>());
	}
	auto input_stream = child_->produce_tasks(plan_context);
	auto join_bytes = join_bytes_;
	auto distinct_bytes = distinct_bytes_;
	auto delim_idx = delim_idx_;
	auto output_types = output_types_;
	auto estimated_cardinality = estimated_cardinality_;
	auto delim_type = delim_type_;
	auto db = db_;

	return input_stream.pipeline_instruction(
	    shared_from_this(),
	    [join_bytes, distinct_bytes, delim_idx, output_types, estimated_cardinality, delim_type,
	     db](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
		    if (!input_plan || !input_plan->HasRoot()) {
			    return input_plan;
		    }

		    auto join_op = DeserializeOperator(join_bytes, *input_plan, db);
		    auto distinct_op = DeserializeOperator(distinct_bytes, *input_plan, db);
		    if (!join_op || !distinct_op) {
			    return input_plan;
		    }

		    auto *join_ptr = join_op.get();
		    auto *distinct_ptr = distinct_op.get();
		    input_plan->TakeOwnership(std::move(join_op));
		    input_plan->TakeOwnership(std::move(distinct_op));

		    vector<const_reference<PhysicalOperator>> delim_scans;
		    GatherDelimScans(*join_ptr, delim_scans, delim_idx);
		    optional_idx resolved_idx = delim_idx;
		    if (!resolved_idx.IsValid() && !delim_scans.empty()) {
			    auto &scan = delim_scans[0].get().Cast<PhysicalColumnDataScan>();
			    resolved_idx = scan.delim_index;
		    }
		    if (delim_scans.empty()) {
			    throw NotImplementedException("Delim join translation failed: no DELIM_SCAN nodes found");
		    }

		    auto &old_root = input_plan->Root();
		    PhysicalOperator &delim_join =
		        delim_type == PhysicalOperatorType::LEFT_DELIM_JOIN
		            ? input_plan->Make<PhysicalLeftDelimJoin>(DelimJoinDeserializeTag {}, output_types, *join_ptr,
		                                                      *distinct_ptr, delim_scans, estimated_cardinality,
		                                                      resolved_idx)
		            : input_plan->Make<PhysicalRightDelimJoin>(DelimJoinDeserializeTag {}, output_types, *join_ptr,
		                                                       *distinct_ptr, delim_scans, estimated_cardinality,
		                                                       resolved_idx);
		    delim_join.children.push_back(old_root);
		    input_plan->SetRoot(delim_join);
		    return input_plan;
	    },
	    plan_context.client_context());
}

std::vector<std::string> DelimJoinNode::multiline_display(bool /*verbose*/) const {
	std::vector<std::string> lines;
	lines.push_back(delim_type_ == PhysicalOperatorType::LEFT_DELIM_JOIN ? "Left Delim Join" : "Right Delim Join");
	if (delim_idx_.IsValid()) {
		lines.push_back("Delim index: " + std::to_string(delim_idx_.GetIndex()));
	}
	return lines;
}

} // namespace distributed
} // namespace duckdb

// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/helper/physical_limit.hpp"
#include "duckdb/execution/operator/helper/physical_streaming_limit.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/execution/operator/order/physical_order.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/projection/physical_tableinout_function.hpp"
#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/planner/bound_result_modifier.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"

#include <chrono>

namespace duckdb {
namespace distributed {
namespace {

static unique_ptr<PhysicalOperator> DeserializePlanRoot(BinaryDeserializer &deserializer, PhysicalPlan &plan,
                                                        ClientContext &context, bound_parameter_map_t &parameters) {
	auto &db = DatabaseInstance::GetDatabase(context);
	deserializer.Set<DatabaseInstance &>(db);
	deserializer.Set<ClientContext &>(context);
	deserializer.Set<bound_parameter_map_t &>(parameters);
	deserializer.Begin();
	auto root = PhysicalOperator::Deserialize(deserializer, plan);
	deserializer.End();
	deserializer.Unset<bound_parameter_map_t>();
	deserializer.Unset<ClientContext>();
	deserializer.Unset<DatabaseInstance>();
	return root;
}

} // namespace

DuckPhysicalPlanRef ClonePhysicalPlanOrThrow(const DuckPhysicalPlanRef &plan, const char *reason_context,
                                             ClientContext *client_context) {
	if (!plan || !plan->HasRoot()) {
		throw InvalidInputException("ClonePhysicalPlanOrThrow: plan missing root (context=%s)",
		                            reason_context ? reason_context : "unknown");
	}

	Allocator &alloc = Allocator::DefaultAllocator();
	auto new_plan = std::make_shared<duckdb::PhysicalPlan>(alloc);
	auto &new_root = ClonePhysicalPlanRootIntoPlanOrThrow(plan, *new_plan, reason_context, client_context);
	new_plan->SetRoot(new_root);
	return new_plan;
}

PhysicalOperator &ClonePhysicalPlanRootIntoPlanOrThrow(const DuckPhysicalPlanRef &source_plan,
                                                       PhysicalPlan &destination_plan, const char *reason_context,
                                                       ClientContext *client_context) {
	if (!source_plan || !source_plan->HasRoot()) {
		throw InvalidInputException("ClonePhysicalPlanRootIntoPlanOrThrow: plan missing root (context=%s)",
		                            reason_context ? reason_context : "unknown");
	}

	MemoryStream stream(Allocator::DefaultAllocator());
	SerializationOptions options;
	options.serialization_compatibility = SerializationCompatibility::Latest();
	options.serialize_default_values = true;
	BinarySerializer serializer(stream, options);
	serializer.Begin();
	source_plan->Root().Serialize(serializer);
	serializer.End();
	stream.Rewind();
	BinaryDeserializer deserializer(stream);

	bound_parameter_map_t parameters;
	unique_ptr<PhysicalOperator> new_root;

	try {
		unique_ptr<DuckDB> local_db;
		unique_ptr<Connection> local_conn;
		if (client_context) {
			auto &db = DatabaseInstance::GetDatabase(*client_context);
			local_conn = make_uniq<Connection>(db);
		} else {
			local_db = make_uniq<DuckDB>(nullptr);
			local_conn = make_uniq<Connection>(*local_db);
		}
		auto &ctx = *local_conn->context;
		ctx.RunFunctionInTransaction(
		    [&]() { new_root = DeserializePlanRoot(deserializer, destination_plan, ctx, parameters); });
	} catch (const std::exception &ex) {
		throw InvalidInputException("ClonePhysicalPlanRootIntoPlanOrThrow: clone failed (context=%s): %s",
		                            reason_context ? reason_context : "unknown", ex.what());
	}

	if (!new_root) {
		throw InvalidInputException("ClonePhysicalPlanRootIntoPlanOrThrow: deserialized null root (context=%s)",
		                            reason_context ? reason_context : "unknown");
	}
	auto *root_ptr = new_root.get();
	destination_plan.TakeOwnership(std::move(new_root));
	return *root_ptr;
}

SubmittableTask<WorkerTask>
append_plan_to_existing_task(SubmittableTask<WorkerTask> submittable_task, const PipelineNodeRef &node,
                             const std::function<DuckPhysicalPlanRef(DuckPhysicalPlanRef)> &plan_builder,
                             ClientContext *client_context) {
	auto append_profile_start = std::chrono::steady_clock::now();
	auto old_task = submittable_task.task();

	auto clone_profile_start = std::chrono::steady_clock::now();
	auto plan_ref = old_task->plan();
	auto plan_ref_count = plan_ref.use_count();
	auto working_plan = ClonePhysicalPlanOrThrow(plan_ref, "append_plan_to_existing_task", client_context);
	auto clone_profile_end = std::chrono::steady_clock::now();

	auto builder_profile_start = std::chrono::steady_clock::now();
	auto new_plan = plan_builder(std::move(working_plan));
	auto builder_profile_end = std::chrono::steady_clock::now();
	TaskContext ctx = old_task->task_context();
	ctx.add_node_id(node->node_id());
	auto merged_ctx = MergeTaskContext(old_task->context(), node->context().to_hashmap());
	WorkerTask new_task(ctx, new_plan, old_task->config(), std::move(merged_ctx));
	new_task.mutable_inputs() = old_task->inputs();
	auto append_profile_end = std::chrono::steady_clock::now();
	return SubmittableTask<WorkerTask>(std::move(new_task));
}

std::unordered_map<std::string, std::string>
MergeTaskContext(const std::unordered_map<std::string, std::string> &base,
                 const std::unordered_map<std::string, std::string> &extra) {
	std::unordered_map<std::string, std::string> merged = base;
	for (const auto &kv : extra) {
		auto entry = merged.find(kv.first);
		if (entry == merged.end()) {
			merged.emplace(kv.first, kv.second);
			continue;
		}
		if (entry->second.empty() && !kv.second.empty()) {
			entry->second = kv.second;
		}
	}
	return merged;
}

} // namespace distributed
} // namespace duckdb

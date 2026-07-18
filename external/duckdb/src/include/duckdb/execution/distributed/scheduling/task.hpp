// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

/**
 * @file task.hpp
 * @brief Task types and interfaces for distributed execution
 *
 * Translated from DuckDB's DuckDB-distributed/src/scheduling/task.rs to C++20.
 * Provides Task traits, TaskContext, TaskResourceRequest, and related types.
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "duckdb/execution/distributed/common_types.hpp"

namespace duckdb {
class PhysicalPlan; // forward-declare DuckDB native PhysicalPlan

namespace distributed {

//------------------------------------------------------------------------------
// Task Resource Request
//------------------------------------------------------------------------------

/**
 * @brief TaskResourceRequest - resource requirements for a task
 * (Rust: TaskResourceRequest)
 */
class TaskResourceRequest {
public:
	TaskResourceRequest() = default;

	explicit TaskResourceRequest(ResourceRequest resource_request) : resource_request_(std::move(resource_request)) {
	}

	/// Get number of CPUs (defaults to 1.0)
	double num_cpus() const {
		double v = resource_request_.num_cpus();
		return v > 0.0 ? v : 1.0;
	}

	/// Get number of GPUs (defaults to 0.0)
	double num_gpus() const {
		double v = resource_request_.num_gpus();
		return v >= 0.0 ? v : 0.0;
	}

	/// Get memory bytes (defaults to 0)
	size_t memory_bytes() const {
		return resource_request_.memory_bytes();
	}

	/// Get underlying resource request
	const ResourceRequest &resource_request() const {
		return resource_request_;
	}

private:
	ResourceRequest resource_request_;
};

// TaskContext and MaterializedOutput are defined in common_types.hpp to avoid
// circular include dependencies between scheduling/task.hpp and
// pipeline_node/pipeline_node.hpp.

//------------------------------------------------------------------------------
// Worker Task
//------------------------------------------------------------------------------

/**
 * @brief WorkerTask - concrete task implementation
 * (Rust: WorkerTask)
 */
class WorkerTask {
public:
	/// Default constructor for pair compatibility
	WorkerTask() : task_context_(), task_name_("WorkerTask") {
	}

	WorkerTask(TaskContext task_context, DuckPhysicalPlanRef plan, DuckDBExecutionConfigRef config,
	           std::unordered_map<std::string, std::string> context, std::string task_name = "WorkerTask",
	           TaskInputs inputs = {})
	    : task_context_(std::move(task_context)), plan_(std::move(plan)), config_(std::move(config)),
	      inputs_(std::move(inputs)), context_(std::move(context)), task_name_(std::move(task_name)) {
		// Add task_id to context
		context_["task_id"] = std::to_string(task_context_.task_id());
	}

	TaskContext task_context() const {
		return task_context_;
	}

	std::unique_ptr<WorkerTask> clone() const {
		return std::unique_ptr<WorkerTask>(
		    new WorkerTask(task_context_, plan_, config_, context_, task_name_, inputs_));
	}

	/// Get plan (DuckDB native PhysicalPlan)
	DuckPhysicalPlanRef plan() const {
		return plan_;
	}

	/// Get config
	const DuckDBExecutionConfigRef &config() const {
		return config_;
	}

	/// Get task inputs (worker-side data routing, analogous to Vane's inputs)
	const TaskInputs &inputs() const {
		return inputs_;
	}

	/// Get mutable task inputs (for populating during task construction)
	TaskInputs &mutable_inputs() {
		return inputs_;
	}

	/// Get context
	const std::unordered_map<std::string, std::string> &context() const {
		return context_;
	}

	/// Get name
	std::string name() const {
		return task_name_;
	}

private:
	TaskContext task_context_;
	DuckPhysicalPlanRef plan_;
	DuckDBExecutionConfigRef config_;
	TaskInputs inputs_; // worker execution routing (analogous to Vane's inputs)
	std::unordered_map<std::string, std::string> context_;
	std::string task_name_;
};

} // namespace distributed
} // namespace duckdb

// Hash specialization for TaskContext
namespace std {
template <>
struct hash<duckdb::distributed::TaskContext> {
	size_t operator()(const duckdb::distributed::TaskContext &ctx) const {
		return ctx.hash();
	}
};
} // namespace std

// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

// Core distributed pipeline node interfaces.

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "duckdb/execution/distributed/common_types.hpp"
#include "duckdb/execution/operator/exchange/repartition.hpp"
#include "duckdb/execution/distributed/common/display/lib.hpp"
#include "duckdb/execution/distributed/scheduling/task.hpp"
#include "duckdb/execution/distributed/utils/channel.hpp"

namespace duckdb {
class ClientContext;
namespace distributed {

struct ScanTaskDescriptor;

struct MaterializeResult;

// MaterializedOutput is defined in `common_types.hpp` to avoid include cycles.

//------------------------------------------------------------------------------
// Pipeline node config and context
//------------------------------------------------------------------------------

/// PipelineNodeConfig: configuration for a pipeline node
class PipelineNodeConfig {
public:
	PipelineNodeConfig() = default;

	PipelineNodeConfig(SchemaRef schema, DuckDBExecutionConfigRef execution_config, ClusteringSpecRef clustering_spec)
	    : schema_(std::move(schema)), execution_config_(std::move(execution_config)),
	      clustering_spec_(std::move(clustering_spec)) {
	}

	SchemaRef schema() const {
		return schema_;
	}
	DuckDBExecutionConfigRef execution_config() const {
		return execution_config_;
	}
	ClusteringSpecRef clustering_spec() const {
		return clustering_spec_;
	}

private:
	SchemaRef schema_;
	DuckDBExecutionConfigRef execution_config_;
	ClusteringSpecRef clustering_spec_;
};

/// PipelineNodeContext: context for a pipeline node during planning/execution
class PipelineNodeContext {
public:
	PipelineNodeContext() = default;

	PipelineNodeContext(uint16_t query_idx, std::string query_id, NodeID node_id, NodeName node_name)
	    : query_idx_(query_idx), query_id_(std::move(query_id)), node_id_(node_id), node_name_(std::move(node_name)) {
	}

	uint16_t query_idx() const {
		return query_idx_;
	}
	const std::string &query_id() const {
		return query_id_;
	}
	NodeID node_id() const {
		return node_id_;
	}
	const NodeName &node_name() const {
		return node_name_;
	}

	std::unordered_map<std::string, std::string> to_hashmap() const {
		auto result = std::unordered_map<std::string, std::string> {
		    {"query_id", query_id_}, {"node_id", std::to_string(node_id_)}, {"node_name", node_name_}};
		if (!query_id_.empty()) {
			result.emplace("resource_query_id", query_id_);
			result.emplace("resource_stage_id", "stage:" + query_id_ + ":node:" + std::to_string(node_id_) + ":fte");
		}
		return result;
	}

private:
	uint16_t query_idx_ = 0;
	std::string query_id_ = "";
	NodeID node_id_ = 0;
	NodeName node_name_ = "";
};

class DistributedPipelineNode;
using DistributedPipelineNodeRef = std::shared_ptr<DistributedPipelineNode>;
// SubmittableTaskStream is a template; forward declaration removed to avoid
// hiding the template definition below.
class Exchange;
class PlanExecutionContext;
class PipelineNodeContext;
class PipelineNodeImpl;
class WorkerTask;

using PipelineNodeRef = std::shared_ptr<PipelineNodeImpl>;

class FteTaskSubmitter {
public:
	virtual ~FteTaskSubmitter() = default;

	virtual DuckDBResult<void> submit_fte_task_events(std::vector<WorkerTask> tasks) = 0;
	virtual DuckDBResult<void> task_input_stream_exhausted(const std::string &query_id,
	                                                       const std::unordered_set<SourceNodeId> &source_node_ids) = 0;
	virtual DuckDBResult<std::vector<MaterializedOutput>> wait_query_finished(const std::string &query_id,
	                                                                          double timeout_s) = 0;
	virtual DuckDBResult<std::vector<MaterializedOutput>>
	wait_query_finished(const std::string &query_id, double timeout_s, MaterializedOutputCallback on_output) {
		auto res = wait_query_finished(query_id, timeout_s);
		if (res.is_err()) {
			return res;
		}
		if (on_output) {
			for (const auto &output : res.value()) {
				auto callback_res = on_output(output);
				if (callback_res.is_err()) {
					return DuckDBResult<std::vector<MaterializedOutput>>::err(callback_res.error());
				}
			}
		}
		return res;
	}
	virtual DuckDBResult<std::vector<MaterializedOutput>>
	wait_query_finished(const std::string &query_id, double timeout_s,
	                    const std::unordered_set<TaskContext, TaskContextHash> &task_contexts,
	                    MaterializedOutputCallback on_output) {
		(void)task_contexts;
		auto res = wait_query_finished(query_id, timeout_s);
		if (res.is_err()) {
			return res;
		}
		if (on_output) {
			for (const auto &output : res.value()) {
				auto callback_res = on_output(output);
				if (callback_res.is_err()) {
					return DuckDBResult<std::vector<MaterializedOutput>>::err(callback_res.error());
				}
			}
		}
		return res;
	}
};

//===----------------------------------------------------------------------===//
// PipelineNodeImpl - 流水线节点接口
// 对应 DuckDB 的 PipelineNodeImpl trait
// 所有具体节点类型必须实现此接口
//===----------------------------------------------------------------------===//
// Forward declarations needed by the stream types
template <typename TaskT>
class SubmittableTask;
template <typename T>
class Receiver;

/**
 * A planned task emitted by distributed pipeline nodes.
 *
 * FTE is now the only distributed Ray execution path, so this wrapper only
 * carries the WorkerTask through pipeline rewrites until PlanRunner or
 * materialize() forwards it as an FTE task event.
 */
template <typename TaskT>
class SubmittableTask {
public:
	SubmittableTask() : task_() {
	}

	explicit SubmittableTask(TaskT task) : task_(std::move(task)) {
	}

	SubmittableTask(SubmittableTask &&) = default;
	SubmittableTask &operator=(SubmittableTask &&) = default;
	SubmittableTask(const SubmittableTask &) = delete;
	SubmittableTask &operator=(const SubmittableTask &) = delete;

	const TaskT *task() const {
		return &task_;
	}
	TaskT *task() {
		return &task_;
	}

	TaskT take_task() && {
		return std::move(task_);
	}

	SubmittableTask with_new_task(TaskT new_task) && {
		task_ = std::move(new_task);
		return std::move(*this);
	}

private:
	TaskT task_;
};

// 可提交任务流类
template <typename TaskT>
class SubmittableTaskStream {
public:
	// 从接收器构造（对应 From<Receiver> trait 实现）
	static SubmittableTaskStream from_receiver(Receiver<SubmittableTask<TaskT>> receiver);

	// 构造函数
	SubmittableTaskStream(std::unique_ptr<BoxStream<SubmittableTask<TaskT>>> task_stream)
	    : task_stream_(std::move(task_stream)) {
	}

	// 物化方法 - 返回 MaterializeResult (synchronous for C++11)
	MaterializeResult materialize(FteTaskSubmitter *fte_task_submitter = nullptr,
	                              MaterializedOutputCallback on_output = {});

	// 管道指令方法
	template <typename F>
	SubmittableTaskStream pipeline_instruction(std::shared_ptr<PipelineNodeImpl> node, F plan_builder,
	                                           ::duckdb::ClientContext *client_context = nullptr) {
		// Only supported for WorkerTask at the moment
		static_assert(std::is_same<TaskT, WorkerTask>::value, "pipeline_instruction is only supported for WorkerTask");

		// Move the inner boxed stream out
		auto inner_box = std::move(task_stream_);

		// Map stream: apply append_plan_to_existing_task on each emitted task.
		struct MapStream {
			std::unique_ptr<BoxStream<SubmittableTask<TaskT>>> inner;
			PipelineNodeRef node;
			F plan_builder; // templated capture
			::duckdb::ClientContext *client_context;

			MapStream(std::unique_ptr<BoxStream<SubmittableTask<TaskT>>> i, PipelineNodeRef n, F pb,
			          ::duckdb::ClientContext *cc)
			    : inner(std::move(i)), node(std::move(n)), plan_builder(std::move(pb)), client_context(cc) {
			}

			std::pair<bool, SubmittableTask<TaskT>> map_one(std::pair<bool, SubmittableTask<TaskT>> opt) {
				if (!opt.first) {
					return std::make_pair(false, SubmittableTask<TaskT>());
				}
				auto task = std::move(opt.second);
				auto result = append_plan_to_existing_task(std::move(task), node, plan_builder, client_context);
				return std::make_pair(true, std::move(result));
			}

			std::pair<bool, SubmittableTask<TaskT>> poll_next() {
				if (!inner) {
					return std::make_pair(false, SubmittableTask<TaskT>());
				}
				return map_one(inner->poll_next());
			}

			std::pair<bool, SubmittableTask<TaskT>> try_poll_next() {
				if (!inner) {
					return std::make_pair(false, SubmittableTask<TaskT>());
				}
				return map_one(inner->try_poll_next());
			}

			bool is_exhausted() const {
				return !inner || inner->is_exhausted();
			}

			// Provide simple iterator adapters for the boxed helper
			struct Iterator {
				MapStream *parent = nullptr;
				std::pair<bool, SubmittableTask<TaskT>> cur;
				Iterator() = default;
				explicit Iterator(MapStream *p) : parent(p) {
					++(*this);
				}
				SubmittableTask<TaskT> operator*() {
					return std::move(cur.second);
				}
				Iterator &operator++() {
					cur = parent ? parent->poll_next() : std::make_pair(false, SubmittableTask<TaskT>());
					return *this;
				}
				bool equals(const Iterator &other) const {
					return !cur.first && !other.cur.first;
				}
				bool operator==(const Iterator &other) const {
					return equals(other);
				}
				bool operator!=(const Iterator &other) const {
					return !equals(other);
				}
			};

			Iterator begin() {
				return Iterator(this);
			}
			Iterator end() {
				return Iterator();
			}
		} map_stream(std::move(inner_box), std::move(node), std::move(plan_builder), client_context);

		auto boxed_stream = boxed<SubmittableTask<TaskT>>(std::move(map_stream));
		return SubmittableTaskStream(std::move(boxed_stream));
	}

	template <typename F>
	SubmittableTaskStream map_tasks(F mapper) {
		auto inner_box = std::move(task_stream_);

		struct MapStream {
			std::unique_ptr<BoxStream<SubmittableTask<TaskT>>> inner;
			F mapper;

			MapStream(std::unique_ptr<BoxStream<SubmittableTask<TaskT>>> i, F m)
			    : inner(std::move(i)), mapper(std::move(m)) {
			}

			std::pair<bool, SubmittableTask<TaskT>> poll_next() {
				if (!inner) {
					return std::make_pair(false, SubmittableTask<TaskT>());
				}
				auto opt = inner->poll_next();
				if (!opt.first) {
					return std::make_pair(false, SubmittableTask<TaskT>());
				}
				auto mapped_result = mapper(std::move(opt.second));
				return std::make_pair(true, std::move(mapped_result));
			}

			std::pair<bool, SubmittableTask<TaskT>> try_poll_next() {
				if (!inner) {
					return std::make_pair(false, SubmittableTask<TaskT>());
				}
				auto opt = inner->try_poll_next();
				if (!opt.first) {
					return std::make_pair(false, SubmittableTask<TaskT>());
				}
				auto mapped_result = mapper(std::move(opt.second));
				return std::make_pair(true, std::move(mapped_result));
			}

			bool is_exhausted() const {
				return !inner || inner->is_exhausted();
			}

			struct Iterator {
				MapStream *parent = nullptr;
				std::pair<bool, SubmittableTask<TaskT>> cur;
				Iterator() = default;
				explicit Iterator(MapStream *p) : parent(p) {
					++(*this);
				}
				SubmittableTask<TaskT> operator*() {
					return std::move(cur.second);
				}
				Iterator &operator++() {
					cur = parent ? parent->poll_next() : std::make_pair(false, SubmittableTask<TaskT>());
					return *this;
				}
				bool equals(const Iterator &other) const {
					return !cur.first && !other.cur.first;
				}
				bool operator==(const Iterator &other) const {
					return equals(other);
				}
				bool operator!=(const Iterator &other) const {
					return !equals(other);
				}
			};

			Iterator begin() {
				return Iterator(this);
			}
			Iterator end() {
				return Iterator();
			}
		} map_stream(std::move(inner_box), std::move(mapper));

		auto boxed_stream = boxed<SubmittableTask<TaskT>>(std::move(map_stream));
		return SubmittableTaskStream(std::move(boxed_stream));
	}

	// 流接口的轮询方法
	std::pair<bool, SubmittableTask<TaskT>> poll_next() {
		return task_stream_->poll_next();
	}

	std::pair<bool, SubmittableTask<TaskT>> try_poll_next() {
		if (!task_stream_) {
			return std::make_pair(false, SubmittableTask<TaskT>());
		}
		return task_stream_->try_poll_next();
	}

	bool is_exhausted() const {
		bool exhausted = !task_stream_ || task_stream_->is_exhausted();
		return exhausted;
	}

private:
	std::unique_ptr<BoxStream<SubmittableTask<TaskT>>> task_stream_;
};

//===----------------------------------------------------------------------===//
// PipelineNodeImpl - 流水线节点接口
// 对应 DuckDB 的 PipelineNodeImpl trait
// 所有具体节点类型必须实现此接口
//===----------------------------------------------------------------------===//
class PipelineNodeImpl {
public:
	// 纯虚函数：必须在派生类中实现
	virtual const PipelineNodeContext &context() const = 0;
	virtual const PipelineNodeConfig &config() const = 0;

	virtual std::vector<PipelineNodeRef> children() const = 0;
	//! 产生可提交的任务流
	//! 对应 Rust: fn produce_tasks(...) -> BoxStream<'static, SubmittableTask<WorkerTask>>
	//! C++20 协程版本返回 Generator，使用 co_yield 产出任务
	virtual SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) = 0;
	virtual NodeName name() const {
		return this->context().node_name();
	}
	virtual NodeID node_id() const {
		return this->context().node_id();
	}
	/// Returns true if this node is a finalizable sink (e.g. CopyFinish).
	/// When true, run_plan collects all outputs and calls finalize() instead of streaming.
	virtual bool is_sink() const {
		return false;
	}
	virtual std::vector<std::string> multiline_display(bool verbose) const = 0;
};

using PipelineNodeImplRef = std::shared_ptr<PipelineNodeImpl>;

template <typename Self>
class DynTreeNode {
public:
	virtual ~DynTreeNode() = default;

	virtual std::vector<std::shared_ptr<Self>> arc_children() const = 0;
	virtual DuckDBResult<std::shared_ptr<Self>> with_new_children(std::vector<std::shared_ptr<Self>> new_children) = 0;
};

class TreeDisplay {
public:
	virtual ~TreeDisplay() = default;
	virtual std::string display_as(DisplayLevel level) const = 0;
	virtual std::string repr_json() const = 0;
	virtual std::vector<const TreeDisplay *> get_children() const = 0;
	virtual std::string get_name() const = 0;
};

class DistributedPipelineNode : public PipelineNodeImpl,
                                public DynTreeNode<DistributedPipelineNode>,
                                public TreeDisplay,
                                public std::enable_shared_from_this<DistributedPipelineNode> {
public:
	// 构造函数
	DistributedPipelineNode(std::shared_ptr<PipelineNodeImpl> op) : op_(std::move(op)) {
		// Defensive: if op_ is null, do not attempt to inspect children.
		if (!op_)
			return;

		// Build distributed children wrappers from underlying pipeline node children
		auto children_snapshot = op_->children();
		for (auto &child_impl : children_snapshot) {
			// Some pipeline nodes may have null children (placeholders). Skip null children.
			if (!child_impl)
				continue;
			children_.push_back(std::make_shared<DistributedPipelineNode>(child_impl));
		}
	}

	// 拷贝构造函数（对应 #[derive(Clone)]）
	DistributedPipelineNode(const DistributedPipelineNode &other) : op_(other.op_), children_(other.children_) {
	}

	// 获取上下文
	const PipelineNodeContext &context() const {
		return op_->context();
	}

	// 获取配置
	const PipelineNodeConfig &config() const {
		return op_->config();
	}

	// 获取节点ID
	NodeID node_id() const {
		return op_->node_id();
	}

	// 获取节点名称
	NodeName name() const {
		return op_->name();
	}

	const PipelineNodeRef &implementation() const {
		return op_;
	}

	// 获取分区数量
	size_t num_partitions() const;

	// If this node is a ScanSource with scan tasks, return them.
	bool try_get_scan_tasks(std::vector<ScanTaskDescriptor> &out) const;

	// 生产任务流
	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) override {
		auto result = op_->produce_tasks(plan_context);
		return std::move(result);
	}

	// 转换为树显示接口
	const TreeDisplay *as_tree_display() const {
		return this;
	}

	// PipelineNodeImpl interface: forward to underlying op implementation
	std::vector<PipelineNodeRef> children() const override {
		return op_->children();
	}

	// DynTreeNode interface - return arc children (shared_ptrs)
	std::vector<DistributedPipelineNodeRef> arc_children() const override;

	// For dynamic tree traversal we expose `arc_children()` via DynTreeNode

	// Create a new DistributedPipelineNode with new arc children
	DuckDBResult<DistributedPipelineNodeRef>
	with_new_children(std::vector<DistributedPipelineNodeRef> new_children) override {
		// std::make_shared cannot access the private constructor here when
		// used from a dependent context in some standard library
		// implementations; construct via `new` within class scope instead.
		auto node = std::shared_ptr<DistributedPipelineNode>(new DistributedPipelineNode(op_, std::move(new_children)));
		return DuckDBResult<DistributedPipelineNodeRef>::ok(std::move(node));
	}

	// TreeDisplay 接口实现
	std::string display_as(DisplayLevel level) const {
		switch (level) {
		case DisplayLevel::Compact:
			return get_name();
		case DisplayLevel::Default: {
			auto display_lines = op_->multiline_display(false);
			// 将字符串向量连接为单个字符串
			std::string result;
			for (const auto &line : display_lines) {
				result += line + "\n";
			}
			if (!result.empty())
				result.pop_back(); // 移除最后一个换行符
			return result;
		}
		case DisplayLevel::Verbose: {
			auto display_lines = op_->multiline_display(true);
			std::string result;
			for (const auto &line : display_lines) {
				result += line + "\n";
			}
			if (!result.empty())
				result.pop_back();
			return result;
		}
		}
		return get_name();
	}

	// Forward multiline_display to underlying impl
	std::vector<std::string> multiline_display(bool verbose) const override {
		return op_->multiline_display(verbose);
	}

	// JSON 表示（简化版，实际需要完整的 JSON 库支持）
	std::string repr_json() const {
		// 简化实现，实际需要使用 JSON 库
		return "{\"id\":\"" + std::to_string(node_id()) + "\",\"type\":\"" + name() + "\",\"name\":\"" + get_name() +
		       "\"}";
	}

	std::vector<const TreeDisplay *> get_children() const {
		std::vector<const TreeDisplay *> result;
		result.reserve(children_.size());
		for (const auto &child : children_) {
			result.push_back(child->as_tree_display());
		}
		return result;
	}

	std::string get_name() const {
		return context().node_name();
	}

	// Expose inner implementation pointer
	PipelineNodeRef inner() const {
		return op_;
	}

private:
	// 私有构造函数，用于 with_new_children
	// Private constructor used by with_new_children
	DistributedPipelineNode(std::shared_ptr<PipelineNodeImpl> op, std::vector<DistributedPipelineNodeRef> children)
	    : op_(op), children_(std::move(children)) {
	}

	std::shared_ptr<PipelineNodeImpl> op_;
	std::vector<DistributedPipelineNodeRef> children_;
};

inline PipelineNodeContext MakePipelineNodeContext(uint16_t query_idx, std::string query_id, NodeID node_id,
                                                   NodeName node_name) {
	return PipelineNodeContext(query_idx, std::move(query_id), node_id, std::move(node_name));
}

template <typename T>
inline PipelineNodeContext InheritPipelineNodeContext(const std::shared_ptr<T> &child, NodeID node_id,
                                                      NodeName node_name) {
	if (!child) {
		return PipelineNodeContext(0, "", node_id, std::move(node_name));
	}
	return PipelineNodeContext(child->context().query_idx(), child->context().query_id(), node_id,
	                           std::move(node_name));
}

// ---------------------------------------------------------------------------
// Visualization helpers
// ---------------------------------------------------------------------------
std::string viz_distributed_pipeline_mermaid(const DistributedPipelineNodeRef &root, DisplayLevel display_type,
                                             bool bottom_up, const std::string &subgraph_options = "");

std::string viz_distributed_pipeline_ascii(const DistributedPipelineNodeRef &root, bool simple);

// ---------------------------------------------------------------------------
// Pipeline task construction helpers
// ---------------------------------------------------------------------------
// Clone a PhysicalPlan for distributed pipeline rewrites.
// Throws on unsupported operators (no fallback).
DuckPhysicalPlanRef ClonePhysicalPlanOrThrow(const DuckPhysicalPlanRef &plan, const char *reason_context,
                                             ::duckdb::ClientContext *client_context = nullptr);

// Clone a plan root into an existing PhysicalPlan and transfer ownership of the
// cloned tree to that plan. This is required when composing operators from two
// independently-owned plans (for example, the left and right sides of a join).
::duckdb::PhysicalOperator &ClonePhysicalPlanRootIntoPlanOrThrow(const DuckPhysicalPlanRef &source_plan,
                                                                 ::duckdb::PhysicalPlan &destination_plan,
                                                                 const char *reason_context,
                                                                 ::duckdb::ClientContext *client_context = nullptr);

SubmittableTask<WorkerTask>
append_plan_to_existing_task(SubmittableTask<WorkerTask> submittable_task, const PipelineNodeRef &node,
                             const std::function<DuckPhysicalPlanRef(DuckPhysicalPlanRef)> &plan_builder,
                             ::duckdb::ClientContext *client_context = nullptr);

// Merge task context maps. Keys in base take precedence over keys in extra.
std::unordered_map<std::string, std::string>
MergeTaskContext(const std::unordered_map<std::string, std::string> &base,
                 const std::unordered_map<std::string, std::string> &extra);

// Record remote exchange sink completions from materialized task outputs.
// Runtime sink instance metadata is required because attempts may finish out of
// submission order.
void RecordRemoteExchangeFinishedSinks(Exchange &exchange, const std::vector<MaterializedOutput> &outputs,
                                       const char *mismatch_context);

} // namespace distributed
} // namespace duckdb

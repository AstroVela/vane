// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

/**
 * @file common_types.hpp
 * @brief Common types and utilities for the distributed execution framework
 *
 * Translated from DuckDB's duckdb-distributed module.
 * This file contains foundational types used across the distributed system.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>
#include "duckdb/planner/expression.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/execution/distributed/exchange/exchange_handles.hpp"

namespace duckdb {

class Expression;
class LogicalType;

namespace distributed {

//------------------------------------------------------------------------------
// Forward Declarations
//------------------------------------------------------------------------------

class ResultPartition;
// Note: ResultPartitionRef is defined as std::shared_ptr<ResultPartition> below
class ResourceRequest;
class DuckDBExecutionConfig;

//------------------------------------------------------------------------------
// DuckDB Forward Declarations
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Type Aliases
//------------------------------------------------------------------------------

/// Query index type (Rust: uint16_t = u16)
using uint16_t = uint16_t;

/// Task identifier type (Rust: TaskID = u32)
using TaskID = uint32_t;

/// Node name type
using NodeName = std::string;

/// Node identifier type
using NodeID = int32_t;

/// Worker identifier type (Rust: WorkerId = Arc<str>)
using WorkerId = std::shared_ptr<std::string>;

/// Helper to create WorkerId from string
inline WorkerId make_worker_id(const std::string &id) {
	return std::make_shared<std::string>(id);
}

struct WorkerIdHash {
	size_t operator()(const WorkerId &worker_id) const noexcept {
		return worker_id ? std::hash<std::string> {}(*worker_id) : 0;
	}
};

struct WorkerIdEqual {
	bool operator()(const WorkerId &lhs, const WorkerId &rhs) const noexcept {
		if (lhs == rhs) {
			return true;
		}
		if (!lhs || !rhs) {
			return false;
		}
		return *lhs == *rhs;
	}
};

/// Expression reference type
using ExpressionRef = std::shared_ptr<duckdb::Expression>;
using BoundExpr = ExpressionRef;
using BoundAggExpr = ExpressionRef;

/// Schema reference type (using LogicalType for now as placeholder for Schema)
using SchemaRef = std::shared_ptr<duckdb::LogicalType>;

template <class TypeList>
inline SchemaRef MakeSchemaRefImpl(const TypeList &types, const std::vector<std::string> *names = nullptr) {
	if (types.empty()) {
		return nullptr;
	}
	const bool use_names = names && names->size() == types.size();
	if (types.size() == 1 && !use_names) {
		return std::make_shared<duckdb::LogicalType>(types[0]);
	}
	child_list_t<LogicalType> children;
	children.reserve(types.size());
	for (idx_t i = 0; i < types.size(); i++) {
		std::string name = use_names ? (*names)[i] : ("c" + std::to_string(i));
		children.emplace_back(std::move(name), types[i]);
	}
	return std::make_shared<duckdb::LogicalType>(LogicalType::STRUCT(std::move(children)));
}

inline SchemaRef MakeSchemaRef(const std::vector<LogicalType> &types) {
	return MakeSchemaRefImpl(types);
}

inline SchemaRef MakeSchemaRef(const duckdb::vector<LogicalType> &types) {
	return MakeSchemaRefImpl(types);
}

inline SchemaRef MakeSchemaRef(const std::vector<LogicalType> &types, const std::vector<std::string> &names) {
	return MakeSchemaRefImpl(types, &names);
}

inline SchemaRef MakeSchemaRef(const duckdb::vector<LogicalType> &types, const duckdb::vector<std::string> &names) {
	return MakeSchemaRefImpl(types, &names);
}

inline duckdb::vector<std::string> GetSchemaNames(const SchemaRef &schema) {
	duckdb::vector<std::string> names;
	if (!schema) {
		return names;
	}
	if (schema->id() == duckdb::LogicalTypeId::STRUCT) {
		const auto &children = duckdb::StructType::GetChildTypes(*schema);
		names.reserve(children.size());
		for (const auto &child : children) {
			names.push_back(child.first);
		}
	}
	return names;
}

//------------------------------------------------------------------------------
// Error Types
//------------------------------------------------------------------------------

/// DuckDBError equivalent - base exception class for distributed framework
class DuckDBError : public std::runtime_error {
public:
	enum class Type { InternalError, ExternalError, IoError, ValueError, InvalidStateError };

	/// Default constructor for pair compatibility
	DuckDBError() : std::runtime_error("uninitialized") {
	}

	explicit DuckDBError(Type type, const std::string &message) : std::runtime_error(format_message(type, message)) {
	}

	explicit DuckDBError(const std::string &message) : DuckDBError(Type::InternalError, message) {
	}

	static DuckDBError internal_error(const std::string &msg) {
		return DuckDBError(Type::InternalError, msg);
	}

	static DuckDBError external_error(const std::string &msg) {
		return DuckDBError(Type::ExternalError, msg);
	}

	static DuckDBError io_error(const std::string &msg) {
		return DuckDBError(Type::IoError, msg);
	}

	static DuckDBError value_error(const std::string &msg) {
		return DuckDBError(Type::ValueError, msg);
	}

	static DuckDBError invalid_state_error(const std::string &msg) {
		return DuckDBError(Type::InvalidStateError, msg);
	}

private:
	static std::string format_message(Type type, const std::string &msg) {
		const char *prefix = "DuckDBError";
		switch (type) {
		case Type::InternalError:
			prefix = "DuckDBError::InternalError";
			break;
		case Type::ExternalError:
			prefix = "DuckDBError::ExternalError";
			break;
		case Type::IoError:
			prefix = "DuckDBError::IoError";
			break;
		case Type::ValueError:
			prefix = "DuckDBError::ValueError";
			break;
		case Type::InvalidStateError:
			prefix = "DuckDBError::InvalidStateError";
			break;
		}
		return std::string(prefix) + " " + msg;
	}
};

//------------------------------------------------------------------------------
// Result Type
//------------------------------------------------------------------------------

/// DuckDBResult equivalent - Result type that can hold either a value or an error
template <typename T>
class DuckDBResult {
public:
	using value_type = T;

	/// Success result
	static DuckDBResult<T> ok(T value) {
		DuckDBResult<T> result;
		result.has_value_ = true;
		result.value_ = std::move(value);
		return result;
	}

	/// Error result
	static DuckDBResult<T> err(DuckDBError error) {
		DuckDBResult<T> result;
		result.has_error_ = true;
		result.error_ = std::move(error);
		return result;
	}

	/// Check if result is success
	bool is_ok() const noexcept {
		return has_value_;
	}

	/// Check if result is error
	bool is_err() const noexcept {
		return has_error_;
	}

	/// Get value (throws if error)
	T &value() & {
		if (is_err()) {
			throw error_;
		}
		return value_;
	}

	const T &value() const & {
		if (is_err()) {
			throw error_;
		}
		return value_;
	}

	T &&value() && {
		if (is_err()) {
			throw error_;
		}
		return std::move(value_);
	}

	/// Get error (throws if success)
	const DuckDBError &error() const {
		if (is_ok()) {
			throw std::logic_error("Called error() on Ok result");
		}
		return error_;
	}

	/// Operator bool for convenience
	explicit operator bool() const noexcept {
		return is_ok();
	}

	/// Default constructor for pair compatibility (public)
	DuckDBResult() : has_value_(false), has_error_(false), value_(), error_() {
	}

private:
	bool has_value_ = false;
	bool has_error_ = false;
	T value_;
	DuckDBError error_;

	template <typename U>
	friend class DuckDBResult;
};

/// Specialization for void
template <>
class DuckDBResult<void> {
public:
	static DuckDBResult<void> ok() {
		DuckDBResult<void> result;
		result.is_ok_ = true;
		return result;
	}

	static DuckDBResult<void> err(DuckDBError error) {
		DuckDBResult<void> result;
		result.is_ok_ = false;
		result.has_error_ = true;
		result.error_ = std::move(error);
		return result;
	}

	bool is_ok() const noexcept {
		return is_ok_;
	}
	bool is_err() const noexcept {
		return !is_ok_;
	}

	void value() const {
		if (is_err() && has_error_) {
			throw error_;
		}
	}

	const DuckDBError &error() const {
		if (is_ok()) {
			throw std::logic_error("Called error() on Ok result");
		}
		return error_;
	}

	explicit operator bool() const noexcept {
		return is_ok_;
	}

private:
	bool is_ok_ = false;
	bool has_error_ = false;
	DuckDBError error_;
	DuckDBResult() : is_ok_(false), has_error_(false), error_("uninitialized") {
	}
};

//------------------------------------------------------------------------------
// ResultPartition Types
//------------------------------------------------------------------------------

/// ResultPartition - interface for result partition implementations
class ResultPartition {
public:
	virtual ~ResultPartition() = default;

	/// Get the size in bytes (returns 0 if unknown)
	virtual DuckDBResult<size_t> size_bytes() const = 0;

	/// Get the number of rows
	virtual DuckDBResult<size_t> num_rows() const = 0;

	/// Convert this fragment to a ColumnDataCollection for consumption.
	/// Returns nullptr if this fragment doesn't carry tabular data.
	virtual std::shared_ptr<duckdb::ColumnDataCollection> to_column_data() const {
		return nullptr;
	}
};

/// Shared pointer to ResultPartition
using ResultPartitionRef = std::shared_ptr<ResultPartition>;

//------------------------------------------------------------------------------
// SourceNodeId & TaskInput (analogous to Vane's SourceId & Input)
//------------------------------------------------------------------------------

/// Stable source node identifier, assigned during plan translation,
/// preserved through serialization. Used for worker-side data routing.
using SourceNodeId = int;

/// Worker-side data input type (analogous to Vane's Input enum).
/// Each variant carries the data needed by a specific source node.
struct TaskInput {
	enum Kind { ScanTask, ExchangeSourceTask };
	Kind kind = ScanTask;
	/// For Kind::ScanTask — raw serialized ScanTaskDescriptor bytes
	std::string scan_task_bytes;
	/// For Kind::ExchangeSourceTask — raw serialized ExchangeSourceTaskDescriptor bytes
	std::string exchange_source_task_bytes;

	static TaskInput make_scan_task(std::string bytes) {
		TaskInput input;
		input.kind = Kind::ScanTask;
		input.scan_task_bytes = std::move(bytes);
		return input;
	}

	static TaskInput make_exchange_source_task(std::string bytes) {
		TaskInput input;
		input.kind = Kind::ExchangeSourceTask;
		input.exchange_source_task_bytes = std::move(bytes);
		return input;
	}
};

/// Map from source_node_id → TaskInput (worker execution routing)
using TaskInputs = std::unordered_map<SourceNodeId, TaskInput>;

class ColumnDataResultPartition : public ResultPartition {
public:
	explicit ColumnDataResultPartition(std::shared_ptr<duckdb::ColumnDataCollection> collection)
	    : collection_(std::move(collection)) {
	}

	DuckDBResult<size_t> size_bytes() const override {
		if (!collection_) {
			return DuckDBResult<size_t>::ok(0);
		}
		return DuckDBResult<size_t>::ok(collection_->SizeInBytes());
	}

	DuckDBResult<size_t> num_rows() const override {
		if (!collection_) {
			return DuckDBResult<size_t>::ok(0);
		}
		return DuckDBResult<size_t>::ok(collection_->Count());
	}

	std::shared_ptr<duckdb::ColumnDataCollection> to_column_data() const override {
		return collection_;
	}

private:
	std::shared_ptr<duckdb::ColumnDataCollection> collection_;
};

//------------------------------------------------------------------------------
// Task Context (moved from task.hpp to avoid include cycles)
//------------------------------------------------------------------------------

/**
 * @brief TaskContext - context information for a task
 * (Rust: TaskContext)
 */
class TaskContext {
public:
	TaskContext() = default;

	TaskContext(uint16_t query_idx, NodeID last_node_id, TaskID task_id, std::vector<NodeID> node_ids)
	    : query_idx_(query_idx), last_node_id_(last_node_id), task_id_(task_id), node_ids_(std::move(node_ids)) {
	}

	/// Create from PipelineNodeContext and TaskID
	static TaskContext from_node_context(uint16_t query_idx, NodeID node_id, TaskID task_id) {
		return TaskContext(query_idx, node_id, task_id, {node_id});
	}

	uint16_t query_idx() const {
		return query_idx_;
	}
	NodeID last_node_id() const {
		return last_node_id_;
	}
	TaskID task_id() const {
		return task_id_;
	}
	const std::vector<NodeID> &node_ids() const {
		return node_ids_;
	}

	void add_node_id(NodeID node_id) {
		node_ids_.push_back(node_id);
	}

	bool operator==(const TaskContext &other) const {
		return query_idx_ == other.query_idx_ && last_node_id_ == other.last_node_id_ && task_id_ == other.task_id_ &&
		       node_ids_ == other.node_ids_;
	}

	size_t hash() const {
		size_t h = std::hash<uint16_t> {}(query_idx_);
		h ^= std::hash<NodeID> {}(last_node_id_) + 0x9e3779b9 + (h << 6) + (h >> 2);
		h ^= std::hash<TaskID> {}(task_id_) + 0x9e3779b9 + (h << 6) + (h >> 2);
		return h;
	}

private:
	uint16_t query_idx_ = 0;
	NodeID last_node_id_ = 0;
	TaskID task_id_ = 0;
	std::vector<NodeID> node_ids_;
};

// Hash specialization for TaskContext
struct TaskContextHash {
	size_t operator()(const TaskContext &ctx) const {
		return ctx.hash();
	}
};

//------------------------------------------------------------------------------
// MaterializedOutput (moved here from pipeline_node/pipeline_node.hpp)
//------------------------------------------------------------------------------

/// MaterializedOutput - result of a task execution
class MaterializedOutput {
public:
	MaterializedOutput() = default;

	MaterializedOutput(std::vector<ResultPartitionRef> fragments, WorkerId worker_id)
	    : fragments_(std::move(fragments)), worker_id_(std::move(worker_id)) {
	}

	MaterializedOutput(std::vector<ResultPartitionRef> fragments, WorkerId worker_id, std::vector<NodeID> node_ids)
	    : fragments_(std::move(fragments)), worker_id_(std::move(worker_id)), node_ids_(std::move(node_ids)) {
	}

	const std::vector<ResultPartitionRef> &fragments() const {
		return fragments_;
	}
	std::vector<ResultPartitionRef> &fragments() {
		return fragments_;
	}

	const WorkerId &worker_id() const {
		return worker_id_;
	}

	const std::vector<NodeID> &node_ids() const {
		return node_ids_;
	}

	int flight_port() const {
		return flight_port_;
	}
	void set_flight_port(int port) {
		flight_port_ = port;
	}

	bool has_exchange_sink_instance() const {
		return has_exchange_sink_instance_;
	}
	const ExchangeSinkInstanceHandle &exchange_sink_instance() const {
		return exchange_sink_instance_;
	}
	void set_exchange_sink_instance(ExchangeSinkInstanceHandle instance) {
		exchange_sink_instance_ = std::move(instance);
		has_exchange_sink_instance_ = true;
	}

	bool has_node_id(NodeID node_id) const {
		for (const auto &id : node_ids_) {
			if (id == node_id) {
				return true;
			}
		}
		return false;
	}

private:
	std::vector<ResultPartitionRef> fragments_;
	WorkerId worker_id_;
	std::vector<NodeID> node_ids_;
	int flight_port_ = 0;
	bool has_exchange_sink_instance_ = false;
	ExchangeSinkInstanceHandle exchange_sink_instance_;
};

using MaterializedOutputCallback = std::function<DuckDBResult<void>(const MaterializedOutput &)>;

//------------------------------------------------------------------------------
// Resource Request Types
//------------------------------------------------------------------------------

/// ResourceRequest - request for compute resources
class ResourceRequest {
public:
	ResourceRequest() : num_cpus_(-1.0), num_gpus_(-1.0), memory_bytes_(0) {
	}

	ResourceRequest(double num_cpus, double num_gpus, size_t memory_bytes)
	    : num_cpus_(num_cpus), num_gpus_(num_gpus), memory_bytes_(memory_bytes) {
	}

	double num_cpus() const {
		return num_cpus_;
	}
	double num_gpus() const {
		return num_gpus_;
	}
	size_t memory_bytes() const {
		return memory_bytes_;
	}

	void set_num_cpus(double value) {
		num_cpus_ = value;
	}
	void set_num_gpus(double value) {
		num_gpus_ = value;
	}
	void set_memory_bytes(size_t value) {
		memory_bytes_ = value;
	}

private:
	double num_cpus_;
	double num_gpus_;
	size_t memory_bytes_;
};

//------------------------------------------------------------------------------
// Execution Config
//------------------------------------------------------------------------------

class DuckDBExecutionConfig {
private:
	std::string shuffle_algorithm_;
	int min_cpu_per_task_;
	bool scan_task_size_grouping_enabled_;
	uint64_t scan_task_min_bytes_;
	uint64_t scan_task_max_bytes_;
	uint64_t scan_task_open_cost_bytes_;
	size_t scan_task_min_partition_num_;
	size_t distributed_node_count_;
	size_t distributed_worker_slots_;
	size_t scan_task_backlog_;

	// 环境变量名常量
	static constexpr const char *ENV_VANE_SHUFFLE_ALGORITHM = "VANE_SHUFFLE_ALGORITHM";
	static constexpr const char *ENV_VANE_MIN_CPU_PER_TASK = "VANE_MIN_CPU_PER_TASK";
	static constexpr const char *ENV_SCAN_TASK_SIZE_GROUPING = "VANE_RAY_SCAN_TASK_SIZE_GROUPING";
	static constexpr const char *ENV_SCAN_TASK_MIN_BYTES = "VANE_RAY_SCAN_TASK_MIN_BYTES";
	static constexpr const char *ENV_SCAN_TASK_MAX_BYTES = "VANE_RAY_SCAN_TASK_MAX_BYTES";
	static constexpr const char *ENV_DISTRIBUTED_NODE_COUNT = "VANE_DISTRIBUTED_NODE_COUNT";
	static constexpr const char *ENV_DISTRIBUTED_WORKER_SLOTS = "VANE_DISTRIBUTED_WORKER_SLOTS";
	static constexpr const char *ENV_SCAN_TASK_BACKLOG = "VANE_RAY_MAX_TASK_BACKLOG";
	static constexpr const char *ENV_SCAN_TASK_OPEN_COST_BYTES = "VANE_RAY_SCAN_TASK_OPEN_COST_BYTES";
	static constexpr const char *ENV_SCAN_TASK_MIN_PARTITION_NUM = "VANE_RAY_SCAN_TASK_MIN_PARTITION_NUM";

public:
	// 默认构造函数
	DuckDBExecutionConfig()
	    : shuffle_algorithm_(""), min_cpu_per_task_(1), scan_task_size_grouping_enabled_(true),
	      scan_task_min_bytes_(96ULL * 1024 * 1024), scan_task_max_bytes_(384ULL * 1024 * 1024),
	      scan_task_open_cost_bytes_(4ULL * 1024 * 1024), scan_task_min_partition_num_(0), distributed_node_count_(0),
	      distributed_worker_slots_(0), scan_task_backlog_(0) {
	}

	// 从环境变量创建配置
	static DuckDBExecutionConfig from_env() {
		DuckDBExecutionConfig cfg;

		// 解析字符串环境变量
		{
			auto val = parse_string_from_env(ENV_VANE_SHUFFLE_ALGORITHM, true);
			if (val.first) {
				cfg.shuffle_algorithm_ = val.second;
			}
		}

		// 解析整数环境变量
		{
			auto val = parse_int_from_env(ENV_VANE_MIN_CPU_PER_TASK);
			if (val.first) {
				cfg.min_cpu_per_task_ = val.second;
			}
		}

		{
			auto val = parse_bool_from_env(ENV_SCAN_TASK_SIZE_GROUPING);
			if (val.first) {
				cfg.scan_task_size_grouping_enabled_ = val.second;
			}
		}

		{
			auto val = parse_bytes_from_env(ENV_SCAN_TASK_MIN_BYTES);
			if (val.first) {
				cfg.scan_task_min_bytes_ = val.second;
			}
		}
		{
			auto val = parse_bytes_from_env(ENV_SCAN_TASK_MAX_BYTES);
			if (val.first) {
				cfg.scan_task_max_bytes_ = val.second;
			}
		}
		{
			auto val = parse_bytes_from_env(ENV_SCAN_TASK_OPEN_COST_BYTES);
			if (val.first) {
				cfg.scan_task_open_cost_bytes_ = val.second;
			}
		}
		{
			auto val = parse_int_from_env(ENV_SCAN_TASK_MIN_PARTITION_NUM);
			if (val.first && val.second > 0) {
				cfg.scan_task_min_partition_num_ = static_cast<size_t>(val.second);
			}
		}
		{
			auto val = parse_int_from_env(ENV_DISTRIBUTED_NODE_COUNT);
			if (val.first && val.second > 0) {
				cfg.distributed_node_count_ = static_cast<size_t>(val.second);
			}
		}
		{
			auto val = parse_int_from_env(ENV_DISTRIBUTED_WORKER_SLOTS);
			if (val.first && val.second > 0) {
				cfg.distributed_worker_slots_ = static_cast<size_t>(val.second);
			}
		}
		{
			auto val = parse_int_from_env(ENV_SCAN_TASK_BACKLOG);
			if (val.first && val.second > 0) {
				cfg.scan_task_backlog_ = static_cast<size_t>(val.second);
			}
		}

		return cfg;
	}

	// Getter 方法
	const std::string &shuffle_algorithm() const {
		return shuffle_algorithm_;
	}
	void set_shuffle_algorithm(std::string algo) {
		shuffle_algorithm_ = std::move(algo);
	}
	int min_cpu_per_task() const {
		return min_cpu_per_task_;
	}
	bool scan_task_size_grouping_enabled() const {
		return scan_task_size_grouping_enabled_;
	}
	uint64_t scan_task_min_bytes() const {
		return scan_task_min_bytes_;
	}
	uint64_t scan_task_max_bytes() const {
		return scan_task_max_bytes_;
	}
	uint64_t scan_task_open_cost_bytes() const {
		return scan_task_open_cost_bytes_;
	}
	size_t scan_task_min_partition_num() const {
		return scan_task_min_partition_num_;
	}
	size_t distributed_node_count() const {
		return distributed_node_count_;
	}
	void set_distributed_node_count(size_t value) {
		distributed_node_count_ = value;
	}
	size_t distributed_worker_slots() const {
		return distributed_worker_slots_;
	}
	void set_distributed_worker_slots(size_t value) {
		distributed_worker_slots_ = value;
	}
	size_t scan_task_backlog() const {
		return scan_task_backlog_;
	}

private:
	// 解析字符串环境变量
	static std::pair<bool, std::string> parse_string_from_env(const char *env_name, bool trim_whitespace = false) {
		const char *val = std::getenv(env_name);
		if (val == nullptr) {
			return std::make_pair(false, std::string());
		}

		std::string result(val);
		if (trim_whitespace) {
			result.erase(0, result.find_first_not_of(" \t\n\r\f\v"));
			result.erase(result.find_last_not_of(" \t\n\r\f\v") + 1);
		}

		return std::make_pair(true, result);
	}

	// 解析布尔值环境变量
	static std::pair<bool, bool> parse_bool_from_env(const char *env_name) {
		const char *val = std::getenv(env_name);
		if (val == nullptr) {
			return std::make_pair(false, false);
		}

		std::string str_val(val);
		std::transform(str_val.begin(), str_val.end(), str_val.begin(), ::tolower);

		if (str_val == "1" || str_val == "true" || str_val == "yes" || str_val == "on") {
			return std::make_pair(true, true);
		} else if (str_val == "0" || str_val == "false" || str_val == "no" || str_val == "off") {
			return std::make_pair(true, false);
		}

		return std::make_pair(false, false);
	}

	// 解析整数环境变量
	static std::pair<bool, int> parse_int_from_env(const char *env_name) {
		const char *val = std::getenv(env_name);
		if (val == nullptr) {
			return std::make_pair(false, 0);
		}

		try {
			size_t pos;
			int result = std::stoi(val, &pos);
			if (pos == strlen(val)) {
				return std::make_pair(true, result);
			}
		} catch (const std::exception &) {
		}

		return std::make_pair(false, 0);
	}

	static std::pair<bool, uint64_t> parse_bytes_from_env(const char *env_name) {
		const char *val = std::getenv(env_name);
		if (val == nullptr) {
			return std::make_pair(false, uint64_t(0));
		}

		std::string str_val(val);
		str_val.erase(0, str_val.find_first_not_of(" \t\n\r\f\v"));
		if (str_val.empty()) {
			return std::make_pair(false, uint64_t(0));
		}
		str_val.erase(str_val.find_last_not_of(" \t\n\r\f\v") + 1);

		std::string lower = str_val;
		std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
		if (lower == "auto") {
			return std::make_pair(true, uint64_t(0));
		}

		uint64_t multiplier = 1;
		if (lower.size() >= 2 && lower.back() == 'b') {
			char unit = lower[lower.size() - 2];
			if (unit == 'k' || unit == 'm' || unit == 'g' || unit == 't') {
				lower.resize(lower.size() - 2);
			} else {
				unit = '\0';
			}
			switch (unit) {
			case 'k':
				multiplier = 1024ULL;
				break;
			case 'm':
				multiplier = 1024ULL * 1024ULL;
				break;
			case 'g':
				multiplier = 1024ULL * 1024ULL * 1024ULL;
				break;
			case 't':
				multiplier = 1024ULL * 1024ULL * 1024ULL * 1024ULL;
				break;
			default:
				break;
			}
		} else {
			char unit = lower.back();
			if (unit == 'k' || unit == 'm' || unit == 'g' || unit == 't') {
				lower.pop_back();
				switch (unit) {
				case 'k':
					multiplier = 1024ULL;
					break;
				case 'm':
					multiplier = 1024ULL * 1024ULL;
					break;
				case 'g':
					multiplier = 1024ULL * 1024ULL * 1024ULL;
					break;
				case 't':
					multiplier = 1024ULL * 1024ULL * 1024ULL * 1024ULL;
					break;
				default:
					break;
				}
			}
		}

		if (lower.empty()) {
			return std::make_pair(false, uint64_t(0));
		}

		try {
			size_t pos;
			auto base = std::stoull(lower, &pos);
			if (pos != lower.size()) {
				return std::make_pair(false, uint64_t(0));
			}
			if (multiplier != 0 && base > std::numeric_limits<uint64_t>::max() / multiplier) {
				return std::make_pair(false, uint64_t(0));
			}
			return std::make_pair(true, base * multiplier);
		} catch (const std::exception &) {
			return std::make_pair(false, uint64_t(0));
		}
	}
};

using DuckDBExecutionConfigRef = std::shared_ptr<DuckDBExecutionConfig>;
using ExecutionConfigRef = DuckDBExecutionConfigRef;

//------------------------------------------------------------------------------
// Local Physical Plan - forward declaration only
// Full definition is in pipeline_node/local_physical_plan.hpp
//------------------------------------------------------------------------------

// DuckPhysicalPlanRef is the new plan type for all distributed nodes
// For the purposes of building the distributed module and tests, we always
// represent DuckPhysicalPlanRef as a shared_ptr to a real duckdb::PhysicalPlan.
// This avoids inconsistencies when translation units are compiled with different
// BUILD_DISTRIBUTED_PIPELINE_NODE configurations across targets.
using DuckPhysicalPlanRef = std::shared_ptr<duckdb::PhysicalPlan>;

//------------------------------------------------------------------------------
// Atomic Counter Utilities
//------------------------------------------------------------------------------

/// Atomic counter for generating unique IDs
template <typename T>
class AtomicCounter {
public:
	explicit AtomicCounter(T initial = 0) : counter_(initial) {
	}

	/// Get and increment
	T fetch_add(T delta = 1) {
		return counter_.fetch_add(delta, std::memory_order_relaxed);
	}

private:
	std::atomic<T> counter_;
};

/// Global query index counter (Rust: QUERY_IDX_COUNTER)
inline AtomicCounter<uint16_t> &get_query_idx_counter() {
	static AtomicCounter<uint16_t> counter(0);
	return counter;
}

} // namespace distributed
} // namespace duckdb

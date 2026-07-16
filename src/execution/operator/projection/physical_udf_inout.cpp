// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/projection/physical_udf_inout.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb/execution/operator/projection/physical_udf_inout.hpp"
#include "duckdb/execution/distributed/common_types.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/limits.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/types/selection_vector.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/execution/udf_executor.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/execution/external_block.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/function/function_serialization.hpp"
#include "duckdb/function/scalar/udf_functions.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/parallel/thread_context.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <thread>
#include <unordered_map>
#include <unistd.h>

namespace duckdb {

struct UDFPendingOutput {
	ExecutionBatch batch;
	unique_ptr<DataChunk> rows;
	bool raw_udf_output = false;
	std::function<void()> release_output_lease;
};

struct UDFOutputLeaseOwnership {
	explicit UDFOutputLeaseOwnership(std::function<void()> release_output_lease_p)
	    : release_output_lease(std::move(release_output_lease_p)) {
	}

	~UDFOutputLeaseOwnership() {
		if (!release_output_lease) {
			return;
		}
		try {
			auto release = std::move(release_output_lease);
			release();
		} catch (...) {
		}
	}

	std::function<void()> release_output_lease;
};

struct LazyInputReplayToken {
	bool valid = false;
	const LazyRefDataChunk *ptr = nullptr;
	idx_t cardinality = 0;
	idx_t block_count = 0;
	vector<const void *> object_ref_ptrs;
	vector<idx_t> start_offsets;
	vector<idx_t> end_offsets;
};

namespace {

static bool DebugEnvFlagEnabled(const char *name) {
	const char *value = std::getenv(name);
	if (!value || !*value) {
		return false;
	}
	auto to_lower = [](unsigned char c) {
		return static_cast<char>(std::tolower(c));
	};
	if (value[0] == '0' && value[1] == '\0') {
		return false;
	}
	if (to_lower(value[0]) == 'n' && to_lower(value[1]) == 'o' && value[2] == '\0') {
		return false;
	}
	if (to_lower(value[0]) == 'f' && to_lower(value[1]) == 'a' && to_lower(value[2]) == 'l' &&
	    to_lower(value[3]) == 's' && to_lower(value[4]) == 'e' && value[5] == '\0') {
		return false;
	}
	return true;
}

static bool StreamingUDFDebugEnabled() {
	static int cached = -1;
	if (cached != -1) {
		return cached == 1;
	}
	cached = DebugEnvFlagEnabled("DUCKDB_DISTRIBUTED_DEBUG") ? 1 : 0;
	return cached == 1;
}

static void StreamingUDFDebugLog(const string &message) {
	if (!StreamingUDFDebugEnabled()) {
		return;
	}
	std::cerr << "[vane-streaming-udf pid=" << getpid() << " tid=" << std::this_thread::get_id() << "] " << message
	          << std::endl;
}

static bool UDFWorkerSlotDebugEnabled() {
	static int cached = -1;
	if (cached != -1) {
		return cached == 1;
	}
	cached = (DebugEnvFlagEnabled("VANE_UDF_WORKER_SLOT_DEBUG") || StreamingUDFDebugEnabled()) ? 1 : 0;
	return cached == 1;
}

static void UDFWorkerSlotDebugLog(const string &message) {
	if (!UDFWorkerSlotDebugEnabled()) {
		return;
	}
	std::cerr << "[vane-udf-worker-slots pid=" << getpid() << " tid=" << std::this_thread::get_id() << "] " << message
	          << std::endl;
}

static atomic<uint64_t> g_streaming_udf_debug_tick {0};

static LazyInputReplayToken MakeLazyInputReplayToken(const LazyRefDataChunk &bundle) {
	LazyInputReplayToken token;
	token.valid = true;
	token.ptr = &bundle;
	token.cardinality = bundle.cardinality;
	token.block_count = bundle.blocks.size();
	token.object_ref_ptrs.reserve(bundle.blocks.size());
	token.start_offsets.reserve(bundle.blocks.size());
	token.end_offsets.reserve(bundle.blocks.size());
	for (auto &block : bundle.blocks) {
		token.object_ref_ptrs.push_back(block.object_ref.get());
		token.start_offsets.push_back(block.StartOffset());
		token.end_offsets.push_back(block.EndOffset());
	}
	return token;
}

static bool LazyInputReplayTokenMatches(const LazyInputReplayToken &left, const LazyInputReplayToken &right) {
	return left.valid && right.valid && left.ptr == right.ptr && left.cardinality == right.cardinality &&
	       left.block_count == right.block_count && left.object_ref_ptrs == right.object_ref_ptrs &&
	       left.start_offsets == right.start_offsets && left.end_offsets == right.end_offsets;
}

const Value *GetStructChild(const Value &payload, const string &name) {
	if (payload.IsNull() || payload.type().id() != LogicalTypeId::STRUCT) {
		return nullptr;
	}
	const auto &children = StructValue::GetChildren(payload);
	const auto child_count = StructType::GetChildCount(payload.type());
	for (idx_t i = 0; i < child_count; i++) {
		if (StructType::GetChildName(payload.type(), i) != name) {
			continue;
		}
		if (i >= children.size() || children[i].IsNull()) {
			return nullptr;
		}
		return &children[i];
	}
	return nullptr;
}

std::pair<bool, string> GetStructStringField(const Value &payload, const string &name) {
	auto child = GetStructChild(payload, name);
	if (!child) {
		return std::make_pair(false, string());
	}
	if (child->type().id() == LogicalTypeId::VARCHAR) {
		return std::make_pair(true, StringValue::Get(*child));
	}
	return std::make_pair(true, child->ToString());
}

std::pair<bool, bool> GetStructBoolField(const Value &payload, const string &name) {
	auto child = GetStructChild(payload, name);
	if (!child) {
		return std::make_pair(false, false);
	}
	if (child->type().id() == LogicalTypeId::BOOLEAN) {
		return std::make_pair(true, BooleanValue::Get(*child));
	}
	return std::make_pair(false, false);
}

bool HasStructField(const Value &payload, const string &name) {
	auto child = GetStructChild(payload, name);
	return child != nullptr && !child->IsNull();
}

std::pair<bool, idx_t> GetStructIntField(const Value &payload, const string &name) {
	auto child = GetStructChild(payload, name);
	if (!child) {
		return std::make_pair(false, idx_t(0));
	}
	switch (child->type().id()) {
	case LogicalTypeId::INTEGER:
		return std::make_pair(true, static_cast<idx_t>(IntegerValue::Get(*child)));
	case LogicalTypeId::BIGINT:
		return std::make_pair(true, static_cast<idx_t>(BigIntValue::Get(*child)));
	default:
		return std::make_pair(false, idx_t(0));
	}
}

vector<string> GetStructStringListField(const Value &payload, const string &name) {
	vector<string> result;
	auto child = GetStructChild(payload, name);
	if (!child) {
		return result;
	}
	if (child->type().id() != LogicalTypeId::LIST) {
		throw InvalidInputException("udf payload field '%s' must be a LIST", name);
	}
	auto &values = ListValue::GetChildren(*child);
	result.reserve(values.size());
	for (auto &value : values) {
		if (value.IsNull()) {
			throw InvalidInputException("udf payload field '%s' cannot contain NULL", name);
		}
		if (value.type().id() == LogicalTypeId::VARCHAR) {
			result.push_back(StringValue::Get(value));
		} else {
			result.push_back(value.ToString());
		}
	}
	return result;
}

struct PayloadOutputSchema {
	vector<string> names;
	vector<LogicalType> types;
};

vector<idx_t> ParseOutputSchemaTensorShape(const Value &entry) {
	auto shape_child = GetStructChild(entry, "shape");
	if (!shape_child) {
		throw InvalidInputException("udf output_schema tensor entry is missing shape");
	}
	if (shape_child->type().id() != LogicalTypeId::LIST) {
		throw InvalidInputException("udf output_schema tensor shape must be LIST<BIGINT>");
	}
	vector<idx_t> shape;
	auto &values = ListValue::GetChildren(*shape_child);
	shape.reserve(values.size());
	for (auto &value : values) {
		if (value.IsNull()) {
			throw InvalidInputException("udf output_schema tensor shape cannot contain NULL");
		}
		auto dim = value.DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>();
		if (dim <= 0) {
			throw InvalidInputException("udf output_schema tensor shape dimensions must be positive");
		}
		shape.push_back(NumericCast<idx_t>(dim));
	}
	if (shape.empty()) {
		throw InvalidInputException("udf output_schema tensor shape must be non-empty");
	}
	return shape;
}

PayloadOutputSchema ParsePayloadOutputSchema(const Value &payload) {
	PayloadOutputSchema schema;
	auto output_schema = GetStructChild(payload, "output_schema");
	if (!output_schema) {
		return schema;
	}
	if (output_schema->type().id() != LogicalTypeId::LIST) {
		throw InvalidInputException("udf payload field 'output_schema' must be a LIST");
	}
	auto &entries = ListValue::GetChildren(*output_schema);
	schema.names.reserve(entries.size());
	schema.types.reserve(entries.size());
	for (auto &entry : entries) {
		if (entry.IsNull() || entry.type().id() != LogicalTypeId::STRUCT) {
			throw InvalidInputException("udf output_schema entries must be STRUCT values");
		}
		auto name = GetStructStringField(entry, "name");
		if (!name.first || name.second.empty()) {
			throw InvalidInputException("udf output_schema entry is missing name");
		}
		auto kind = GetStructStringField(entry, "kind");
		if (!kind.first || kind.second.empty() || StringUtil::CIEquals(kind.second, "duckdb_type")) {
			auto type_name = GetStructStringField(entry, "type");
			if (!type_name.first || type_name.second.empty()) {
				throw InvalidInputException("udf output_schema duckdb_type entry is missing type");
			}
			schema.names.push_back(name.second);
			schema.types.push_back(DBConfig::ParseLogicalType(type_name.second));
			continue;
		}
		if (StringUtil::CIEquals(kind.second, "tensor")) {
			auto dtype = GetStructStringField(entry, "dtype");
			if (!dtype.first || dtype.second.empty()) {
				throw InvalidInputException("udf output_schema tensor entry is missing dtype");
			}
			schema.names.push_back(name.second);
			schema.types.push_back(
			    TensorType::Create(DBConfig::ParseLogicalType(dtype.second), ParseOutputSchemaTensorShape(entry)));
			continue;
		}
		throw InvalidInputException("unsupported udf output_schema kind '%s'", kind.second);
	}
	return schema;
}

std::pair<bool, string> GetStructNumericFieldString(const Value &payload, const string &name) {
	auto child = GetStructChild(payload, name);
	if (!child) {
		return std::make_pair(false, string());
	}
	switch (child->type().id()) {
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
		return std::make_pair(true, child->ToString());
	default:
		return std::make_pair(false, string());
	}
}

idx_t SaturatingMultiply(idx_t left, idx_t right) {
	if (left == 0 || right == 0) {
		return 0;
	}
	auto max_value = std::numeric_limits<idx_t>::max();
	if (left > max_value / right) {
		return max_value;
	}
	return left * right;
}

idx_t SaturatingAdd(idx_t left, idx_t right) {
	auto max_value = std::numeric_limits<idx_t>::max();
	if (left > max_value - right) {
		return max_value;
	}
	return left + right;
}

idx_t ResolveUDFRuntimeWorkerSlots(const Value &payload, idx_t task_operator_width) {
	auto backend = GetStructStringField(payload, "execution_backend");
	if (!backend.first || backend.second.empty()) {
		throw InvalidInputException("udf payload is missing execution_backend");
	}
	if (backend.second == "subprocess_task") {
		return MaxValue<idx_t>(idx_t(1), task_operator_width);
	}
	if (backend.second == "subprocess_actor") {
		auto actor_number = GetStructIntField(payload, "actor_number");
		if (!actor_number.first || actor_number.second == 0) {
			throw InvalidInputException("actor_number is required for execution_backend='%s'", backend.second);
		}
		return actor_number.second;
	}
	throw InvalidInputException("unsupported udf execution_backend '%s'", backend.second);
}

Value ReplaceStructFields(const Value &payload, child_list_t<Value> replacements) {
	if (payload.IsNull() || payload.type().id() != LogicalTypeId::STRUCT) {
		return payload;
	}
	const auto &existing_children = StructValue::GetChildren(payload);
	const auto child_count = StructType::GetChildCount(payload.type());
	child_list_t<Value> children;
	children.reserve(child_count + replacements.size());
	for (idx_t i = 0; i < child_count; i++) {
		auto child_name = StructType::GetChildName(payload.type(), i);
		bool replaced = false;
		for (auto &replacement : replacements) {
			if (replacement.first == child_name) {
				replaced = true;
				break;
			}
		}
		if (!replaced && i < existing_children.size()) {
			children.emplace_back(child_name, existing_children[i]);
		}
	}
	for (auto &replacement : replacements) {
		children.emplace_back(replacement.first, std::move(replacement.second));
	}
	return Value::STRUCT(std::move(children));
}

Value ResolveUDFRuntimePayload(const Value &payload, idx_t task_operator_width) {
	auto backend = GetStructStringField(payload, "execution_backend");
	if (!backend.first || backend.second.empty()) {
		throw InvalidInputException("udf payload is missing execution_backend");
	}
	if (backend.second == "ray_task" || backend.second == "ray_actor") {
		return payload;
	}
	auto worker_slots = ResolveUDFRuntimeWorkerSlots(payload, task_operator_width);
	child_list_t<Value> replacements;
	replacements.emplace_back("udf_worker_slots", Value::BIGINT(static_cast<int64_t>(worker_slots)));
	return ReplaceStructFields(payload, std::move(replacements));
}

string UDFDebugNameFromPayload(const Value &payload) {
	auto udf_name = GetStructStringField(payload, "udf_name");
	return udf_name.first ? udf_name.second : string("<missing>");
}

} // namespace

namespace {

static bool IsRowPreservingPythonUDFLayoutPayload(const Value &payload);

} // namespace

struct UDFOperatorState : public OperatorState {
	UDFOperatorState(ClientContext &context, const vector<unique_ptr<Expression>> &arg_exprs_p, Value payload_p,
	                 shared_ptr<void> actor_handles_p = nullptr)
	    : arg_executor(context), payload(std::move(payload_p)), actor_handles(std::move(actor_handles_p)) {
		for (auto &expr : arg_exprs_p) {
			arg_executor.AddExpression(*expr);
		}
		arg_types.reserve(arg_exprs_p.size());
		for (auto &expr : arg_exprs_p) {
			arg_types.push_back(expr->return_type);
		}
		// is_flat_map = "result-only table UDF mode": no passthrough rows, output = UDF result only.
		// row-preserving batch UDFs also carry output_schema, so classify by call_mode instead.
		is_flat_map = ClassifyUDFMode(payload) == UDFMode::RESULT_ONLY_BATCH;
		if (is_flat_map) {
			auto schema = ParsePayloadOutputSchema(payload);
			output_names = std::move(schema.names);
			output_types_declared = std::move(schema.types);
		}
		if (IsRowPreservingPythonUDFLayoutPayload(payload) && payload.type().id() == LogicalTypeId::STRUCT) {
			auto &children = StructValue::GetChildren(payload);
			auto child_count = StructType::GetChildCount(payload.type());
			for (idx_t i = 0; i < child_count; i++) {
				auto &field_name = StructType::GetChildName(payload.type(), i);
				if (field_name == "ref_output_types" && i < children.size() && !children[i].IsNull()) {
					auto &type_values = ListValue::GetChildren(children[i]);
					for (auto &v : type_values) {
						output_types_declared.push_back(DBConfig::ParseLogicalType(StringValue::Get(v)));
					}
				}
			}
		}
	}

	ExpressionExecutor arg_executor;
	vector<LogicalType> arg_types;
	UDFConfig config;
	Value payload;
	shared_ptr<void> actor_handles;
	unique_ptr<UDFExecutor> executor;
	bool finished_submitting = false;
	std::deque<unique_ptr<DataChunk>> pending_inputs;
	std::deque<UDFPendingOutput> pending_outputs;
	idx_t submitted_batches = 0;
	idx_t completed_batches = 0;
	bool logged_execute_enter = false;
	idx_t execute_calls = 0;
	idx_t final_calls = 0;
	idx_t blocked_returns = 0;
	idx_t soft_yield_returns = 0;
	idx_t need_more_returns = 0;
	idx_t have_more_returns = 0;
	idx_t total_output_rows = 0;
	bool input_consumed = false;
	LazyInputReplayToken consumed_lazy_input;
	bool is_flat_map = false;
	vector<string> output_names;
	vector<LogicalType> output_types_declared;
};

namespace {

static bool UDFRequiresRowPreservingOutput(const UDFOperatorState &state) {
	if (!state.is_flat_map) {
		return true;
	}
	auto mode = GetStructStringField(state.payload, "cardinality_mode");
	if (mode.first && mode.second == "block_producing") {
		return false;
	}
	auto row_preserving = GetStructBoolField(state.payload, "row_preserving");
	if (row_preserving.first) {
		return row_preserving.second;
	}
	return true;
}

static void ReleaseUDFOutputLease(std::function<void()> &callback) {
	if (!callback) {
		return;
	}
	auto release = std::move(callback);
	callback = nullptr;
	release();
}

static void EnsureExecutor(ExecutionContext &context, UDFOperatorState &state);
static bool TakeReadyResultOnce(ExecutionContext &context, UDFOperatorState &state);
static unique_ptr<DataChunk> BuildOutputChunk(DataChunk &rows, DataChunk &outputs, bool is_flat_map,
                                              const vector<string> &output_names = {},
                                              const vector<LogicalType> &output_types_declared = {});
static void PreserveFlatMapLazyOutputShape(UDFOperatorState &state, LazyRefDataChunk &bundle);
static void PreserveScalarMapLazyOutputShape(UDFOperatorState &state, LazyRefDataChunk &bundle);
static bool PopPendingOutputBatch(ExecutionContext &context, UDFOperatorState &state, ExecutionBatch &output);
static UDFWakeupRegistrationResult RegisterAsyncWakeup(ExecutionContext &context, UDFOperatorState &state);

struct UDFGlobalState;

static void PushMaterializedOutputBatchPieces(std::deque<UDFPendingOutput> &queue, unique_ptr<DataChunk> output_chunk,
                                              std::function<void()> release_output_lease = nullptr);
static idx_t EstimateStreamingVarlenBytes(const Vector &vec, const idx_t count) {
	if (count == 0) {
		return 0;
	}
	Vector scan_vec(vec.GetType());
	scan_vec.Reference(const_cast<Vector &>(vec));
	UnifiedVectorFormat vdata;
	scan_vec.ToUnifiedFormat(count, vdata);
	const auto strings = UnifiedVectorFormat::GetData<string_t>(vdata);
	idx_t total = 0;
	for (idx_t row_idx = 0; row_idx < count; row_idx++) {
		auto idx = vdata.sel->get_index(row_idx);
		if (!vdata.validity.RowIsValid(idx)) {
			continue;
		}
		auto &str = strings[idx];
		if (!str.IsInlined()) {
			total += str.GetSize();
		}
	}
	return total;
}

static idx_t EstimateStreamingChunkBytes(const DataChunk &chunk) {
	auto count = chunk.size();
	idx_t total = chunk.GetAllocationSize();
	for (auto &vec : chunk.data) {
		if (vec.GetType().InternalType() == PhysicalType::VARCHAR) {
			total += EstimateStreamingVarlenBytes(vec, count);
		}
	}
	return total;
}

static bool TrySubmitRefBundleRaw(ExecutionContext &context, UDFOperatorState &state, LazyRefDataChunk &bundle,
                                  idx_t &submit_id, idx_t retained_input_bytes = 0);
static OperatorResultType UDFInOutExecuteBatch(ExecutionContext &context, TableFunctionInput &data,
                                               ExecutionBatch &input, ExecutionBatch &output);
static OperatorFinalizeResultType UDFInOutFinalBatch(ExecutionContext &context, TableFunctionInput &data,
                                                     ExecutionBatch &output);

static idx_t LogicalInflightBatches(const UDFOperatorState &state) {
	if (state.submitted_batches <= state.completed_batches) {
		return 0;
	}
	return state.submitted_batches - state.completed_batches;
}

static UDFWakeupRegistrationResult RegisterBackpressureWakeup(ExecutionContext &context, UDFOperatorState &state) {
	if (!state.executor) {
		return UDFWakeupRegistrationResult::UNSUPPORTED;
	}
	return RegisterAsyncWakeup(context, state);
}

static bool CanUseAsyncWakeup(ExecutionContext &context, UDFOperatorState &state) {
	if (!context.interrupt_state) {
		return false;
	}
	if (!state.executor || !state.executor->SupportsAsyncWakeup()) {
		return false;
	}
	return true;
}

[[noreturn]] static void ThrowNonRefAwareUDFBoundary(const UDFOperatorState &state, const char *reason,
                                                     const char *detail = nullptr) {
	if (detail) {
		throw InvalidInputException("udf lazy/ref-bundle execution reached non-ref-aware boundary '%s' (%s). "
		                            "Implicit materialization inside UDF is disabled; planner must connect this edge "
		                            "through a ref-aware UDF path or insert an explicit materialize boundary.",
		                            reason ? reason : "unknown", detail);
	}
	throw InvalidInputException("udf lazy/ref-bundle execution reached non-ref-aware boundary '%s'. "
	                            "Implicit materialization inside UDF is disabled; planner must connect this edge "
	                            "through a ref-aware UDF path or insert an explicit materialize boundary. "
	                            "is_flat_map=%s executor_supports_ref_bundle_input=%s",
	                            reason ? reason : "unknown", state.is_flat_map ? "true" : "false",
	                            (state.executor && state.executor->SupportsRefBundleInput()) ? "true" : "false");
}

static UDFPendingOutput MakeMaterializedPendingOutput(unique_ptr<DataChunk> chunk) {
	UDFPendingOutput pending;
	pending.batch.kind = ExecutionBatchKind::MATERIALIZED_CHUNK;
	if (chunk) {
		pending.batch.rows = chunk->size();
		pending.batch.estimated_bytes = chunk->GetAllocationSize();
	}
	pending.batch.materialized = std::move(chunk);
	return pending;
}

static unique_ptr<DataChunk> TakeMaterializedPendingUDFOutput(ExecutionContext &context, UDFOperatorState &state,
                                                              UDFPendingOutput &pending) {
	if (pending.batch.kind == ExecutionBatchKind::MATERIALIZED_CHUNK) {
		if (!pending.batch.materialized) {
			throw InternalException("udf pending materialized output is null");
		}
		return std::move(pending.batch.materialized);
	}
	if (pending.batch.kind != ExecutionBatchKind::LAZY_DATA_CHUNK) {
		throw InternalException("udf pending output has unsupported ExecutionBatch kind");
	}
	if (!pending.batch.lazy) {
		throw InternalException("udf pending lazy output is null");
	}
	ThrowNonRefAwareUDFBoundary(state, "non_ref_aware_inout_queue");
}

static bool PopPendingOutput(ExecutionContext &context, UDFOperatorState &state, DataChunk &output) {
	while (!state.pending_outputs.empty()) {
		auto pending = std::move(state.pending_outputs.front());
		state.pending_outputs.pop_front();
		auto output_chunk = TakeMaterializedPendingUDFOutput(context, state, pending);
		if (!output_chunk) {
			continue;
		}
		if (output_chunk->size() > STANDARD_VECTOR_SIZE) {
			std::deque<UDFPendingOutput> pieces;
			PushMaterializedOutputBatchPieces(pieces, std::move(output_chunk), std::move(pending.release_output_lease));
			if (pieces.empty()) {
				continue;
			}
			auto first = std::move(pieces.front());
			pieces.pop_front();
			while (!pieces.empty()) {
				state.pending_outputs.push_front(std::move(pieces.back()));
				pieces.pop_back();
			}
			output_chunk = std::move(first.batch.materialized);
		}
		state.total_output_rows += output_chunk->size();
		output.Move(*output_chunk);
		ReleaseUDFOutputLease(pending.release_output_lease);
		return true;
	}
	return false;
}

static void StoreMaterializedExecutionBatchOutput(ExecutionBatch &batch, unique_ptr<DataChunk> chunk) {
	batch = ExecutionBatch();
	batch.kind = ExecutionBatchKind::MATERIALIZED_CHUNK;
	if (chunk) {
		batch.rows = chunk->size();
		batch.estimated_bytes = chunk->GetAllocationSize();
	}
	batch.materialized = std::move(chunk);
}

static void StoreLazyExecutionBatchOutput(ExecutionBatch &batch, unique_ptr<LazyRefDataChunk> lazy) {
	batch = ExecutionBatch();
	batch.kind = ExecutionBatchKind::LAZY_DATA_CHUNK;
	if (lazy) {
		lazy->RecomputeCardinality();
		batch.rows = lazy->cardinality;
		batch.estimated_bytes = lazy->EstimatedBytes();
	}
	batch.lazy = std::move(lazy);
}

static bool PopPendingOutputBatch(ExecutionContext &context, UDFOperatorState &state, ExecutionBatch &output) {
	while (!state.pending_outputs.empty()) {
		auto pending = std::move(state.pending_outputs.front());
		state.pending_outputs.pop_front();
		if (pending.batch.kind == ExecutionBatchKind::MATERIALIZED_CHUNK) {
			auto output_chunk = TakeMaterializedPendingUDFOutput(context, state, pending);
			if (!output_chunk) {
				continue;
			}
			if (output_chunk->size() > STANDARD_VECTOR_SIZE) {
				std::deque<UDFPendingOutput> pieces;
				PushMaterializedOutputBatchPieces(pieces, std::move(output_chunk),
				                                  std::move(pending.release_output_lease));
				if (pieces.empty()) {
					continue;
				}
				auto first = std::move(pieces.front());
				pieces.pop_front();
				while (!pieces.empty()) {
					state.pending_outputs.push_front(std::move(pieces.back()));
					pieces.pop_back();
				}
				output = std::move(first.batch);
			} else {
				StoreMaterializedExecutionBatchOutput(output, std::move(output_chunk));
				ReleaseUDFOutputLease(pending.release_output_lease);
			}
			state.total_output_rows += output.rows;
			return true;
		}
		if (pending.batch.kind != ExecutionBatchKind::LAZY_DATA_CHUNK) {
			throw InternalException("udf pending output has unsupported ExecutionBatch kind");
		}
		if (!pending.batch.lazy || pending.batch.lazy->Empty()) {
			continue;
		}
		auto total_rows = pending.batch.lazy->cardinality;
		if (total_rows > STANDARD_VECTOR_SIZE) {
			auto first = SliceLazyDataChunk(*pending.batch.lazy, 0, STANDARD_VECTOR_SIZE);
			auto rest =
			    SliceLazyDataChunk(*pending.batch.lazy, STANDARD_VECTOR_SIZE, total_rows - STANDARD_VECTOR_SIZE);
			StoreLazyExecutionBatchOutput(output, std::move(first));
			StoreLazyExecutionBatchOutput(pending.batch, std::move(rest));
			state.pending_outputs.push_front(std::move(pending));
		} else {
			output = std::move(pending.batch);
			ReleaseUDFOutputLease(pending.release_output_lease);
		}
		state.total_output_rows += output.rows;
		return true;
	}
	return false;
}

static void BufferPendingInput(UDFOperatorState &state, DataChunk &input) {
	if (input.size() == 0) {
		return;
	}
	auto buffered = make_uniq<DataChunk>();
	buffered->Initialize(Allocator::DefaultAllocator(), input.GetTypes(), input.size());
	input.Copy(*buffered, 0);
	state.pending_inputs.push_back(std::move(buffered));
}

static void BufferDeferredPipelineInput(UDFOperatorState &state, DataChunk &input, bool using_buffered_input) {
	if (!using_buffered_input || input.size() == 0) {
		return;
	}
	BufferPendingInput(state, input);
}

static OperatorResultType ReturnExecuteOutputForCurrentInput(UDFOperatorState &state) {
	if (state.input_consumed && state.pending_outputs.empty() && LogicalInflightBatches(state) == 0) {
		state.input_consumed = false;
		state.need_more_returns++;
		return OperatorResultType::NEED_MORE_INPUT;
	}
	state.have_more_returns++;
	return OperatorResultType::HAVE_MORE_OUTPUT;
}

static bool BufferExecuteInputBeforeOutput(UDFOperatorState &state, DataChunk &input) {
	if (!state.input_consumed && input.size() > 0) {
		BufferPendingInput(state, input);
		state.need_more_returns++;
		return true;
	}
	return false;
}

static bool BufferExecuteBatchInputBeforeOutput(UDFOperatorState &state, ExecutionBatch &input) {
	if (!state.input_consumed && input.kind == ExecutionBatchKind::MATERIALIZED_CHUNK && input.materialized &&
	    input.materialized->size() > 0) {
		BufferPendingInput(state, *input.materialized);
		state.need_more_returns++;
		return true;
	}
	return false;
}

static UDFWakeupRegistrationResult RegisterAsyncWakeup(ExecutionContext &context, UDFOperatorState &state) {
	if (!CanUseAsyncWakeup(context, state)) {
		return UDFWakeupRegistrationResult::UNSUPPORTED;
	}
	return state.executor->RegisterWakeup(*context.interrupt_state);
}

static unique_ptr<DataChunk> BuildOutputChunk(DataChunk &rows, DataChunk &outputs, bool is_flat_map,
                                              const vector<string> &output_names,
                                              const vector<LogicalType> &output_types_declared) {
	if (is_flat_map) {
		// flat_map/map_batches mode: output is UDF result only, no passthrough rows.
		if (outputs.ColumnCount() <= 1) {
			// Single column — pass through as-is
			auto output_ptr = make_uniq<DataChunk>();
			output_ptr->Move(outputs);
			return output_ptr;
		}
		// Multi-column: wrap into a single STRUCT column to match the plan's
		// expected output type (STRUCT(col1 T1, col2 T2, ...)).
		// Use declared output types when available to avoid type mismatches
		// (e.g., pyarrow returns int64 but schema declares int32).
		auto row_count = outputs.size();
		child_list_t<LogicalType> struct_children;
		for (idx_t i = 0; i < outputs.ColumnCount(); i++) {
			auto name = i < output_names.size() ? output_names[i] : StringUtil::Format("c%d", i);
			auto type = i < output_types_declared.size() ? output_types_declared[i] : outputs.data[i].GetType();
			struct_children.push_back(make_pair(name, type));
		}
		auto struct_type = LogicalType::STRUCT(std::move(struct_children));

		auto output_ptr = make_uniq<DataChunk>();
		vector<LogicalType> output_types;
		output_types.push_back(struct_type);
		output_ptr->Initialize(Allocator::DefaultAllocator(), output_types, row_count);

		// Set up STRUCT vector: cast children to declared types if needed
		auto &struct_vector = output_ptr->data[0];
		auto &struct_entries = StructVector::GetEntries(struct_vector);
		for (idx_t i = 0; i < outputs.ColumnCount(); i++) {
			auto declared_type =
			    i < output_types_declared.size() ? output_types_declared[i] : outputs.data[i].GetType();
			if (outputs.data[i].GetType() == declared_type) {
				struct_entries[i]->Reference(outputs.data[i]);
			} else {
				// Cast to declared type (e.g., BIGINT -> INTEGER)
				VectorOperations::DefaultCast(outputs.data[i], *struct_entries[i], row_count);
			}
		}
		output_ptr->SetCardinality(row_count);
		return output_ptr;
	}
	if (rows.size() != outputs.size()) {
		throw InvalidInputException("udf output count (%d) does not match input rows (%d)", outputs.size(),
		                            rows.size());
	}

	DataChunk output;
	output.Move(rows);
	output.Fuse(outputs);

	auto output_ptr = make_uniq<DataChunk>();
	output_ptr->Move(output);
	return output_ptr;
}

static LogicalType BuildFlatMapStructOutputType(const vector<string> &output_names,
                                                const vector<LogicalType> &output_types) {
	child_list_t<LogicalType> struct_children;
	for (idx_t i = 0; i < output_types.size(); i++) {
		auto name = i < output_names.size() ? output_names[i] : StringUtil::Format("c%d", i);
		struct_children.push_back(make_pair(name, output_types[i]));
	}
	return LogicalType::STRUCT(std::move(struct_children));
}

static vector<LogicalType> GetUDFBatchOutputTypes(const UDFOperatorState &state,
                                                  optional_ptr<const FunctionData> bind_data) {
	vector<LogicalType> output_types;
	if (bind_data) {
		if (IsRowPreservingPythonUDFLayoutPayload(state.payload) && !state.output_types_declared.empty()) {
			return state.output_types_declared;
		}
		output_types.push_back(bind_data->Cast<UDFFunctionData>().return_type);
		return output_types;
	}
	if (!state.is_flat_map) {
		return output_types;
	}
	if (state.output_types_declared.size() <= 1) {
		output_types = state.output_types_declared;
		return output_types;
	}
	output_types.push_back(BuildFlatMapStructOutputType(state.output_names, state.output_types_declared));
	return output_types;
}

static unique_ptr<DataChunk> MakeEmptyUDFBatchOutput(ClientContext &context, const UDFOperatorState &state,
                                                     optional_ptr<const FunctionData> bind_data = nullptr) {
	auto chunk = make_uniq<DataChunk>();
	auto output_types = GetUDFBatchOutputTypes(state, bind_data);
	// Empty batches can still flow through downstream expressions such as
	// struct_extract. Initialize the vectors so nested types have auxiliary
	// child storage even when cardinality is zero.
	chunk->Initialize(context, output_types, 0);
	chunk->SetCardinality(0);
	return chunk;
}

static idx_t UDFExecutionBatchSize(const ExecutionBatch &batch) {
	if (batch.kind == ExecutionBatchKind::MATERIALIZED_CHUNK) {
		return batch.materialized ? batch.materialized->size() : batch.rows;
	}
	if (batch.kind == ExecutionBatchKind::LAZY_DATA_CHUNK) {
		return batch.lazy ? batch.lazy->cardinality : batch.rows;
	}
	return batch.rows;
}

static unique_ptr<DataChunk> MakeEmptyUDFInputChunk(ClientContext &context, const vector<LogicalType> &types) {
	auto chunk = make_uniq<DataChunk>();
	chunk->InitializeEmpty(types);
	chunk->SetCardinality(0);
	return chunk;
}

static unique_ptr<DataChunk> ReferenceMaterializedUDFInputBatch(ExecutionContext &context, UDFOperatorState &state,
                                                                ExecutionBatch &input, const char *reason) {
	if (input.kind == ExecutionBatchKind::MATERIALIZED_CHUNK) {
		if (!input.materialized) {
			return MakeEmptyUDFInputChunk(context.client, state.arg_types);
		}
		auto chunk = make_uniq<DataChunk>();
		chunk->InitializeEmpty(input.materialized->GetTypes());
		chunk->Reference(*input.materialized);
		return chunk;
	}
	if (input.kind == ExecutionBatchKind::LAZY_DATA_CHUNK) {
		if (!input.lazy) {
			return MakeEmptyUDFInputChunk(context.client, state.arg_types);
		}
		ThrowNonRefAwareUDFBoundary(state, reason ? reason : "udf_lazy_input_boundary");
	}
	throw InternalException("udf input has unsupported ExecutionBatch kind");
}

static void StoreEmptyUDFBatchOutput(ExecutionContext &context, UDFOperatorState &state,
                                     optional_ptr<const FunctionData> bind_data, ExecutionBatch &output) {
	StoreMaterializedExecutionBatchOutput(output, MakeEmptyUDFBatchOutput(context.client, state, bind_data));
}

static void PreserveFlatMapLazyOutputShape(UDFOperatorState &state, LazyRefDataChunk &bundle) {
	if (!state.is_flat_map || bundle.logical_types.size() <= 1) {
		return;
	}

	vector<LogicalType> child_types;
	if (state.output_types_declared.size() == bundle.logical_types.size()) {
		child_types = state.output_types_declared;
	} else {
		child_types = bundle.logical_types;
	}

	bundle.logical_types.clear();
	bundle.logical_types.push_back(BuildFlatMapStructOutputType(state.output_names, child_types));
	bundle.names.clear();
	bundle.names.push_back("__udf_result");
	bundle.wrap_columns_as_struct = true;
}

static void PreserveScalarMapLazyOutputShape(UDFOperatorState &state, LazyRefDataChunk &bundle) {
	if (!IsRowPreservingPythonUDFLayoutPayload(state.payload)) {
		return;
	}
	if (!state.output_types_declared.empty() && state.output_types_declared.size() == bundle.logical_types.size()) {
		bundle.logical_types = state.output_types_declared;
	}
}

static unique_ptr<LazyRefDataChunk> CanonicalizeWrappedStructLazyInputForRefSubmit(const Value &payload,
                                                                                   const LazyRefDataChunk &bundle,
                                                                                   bool project_input_names = true) {
	if (!bundle.wrap_columns_as_struct || bundle.logical_types.size() != 1 ||
	    bundle.logical_types[0].id() != LogicalTypeId::STRUCT) {
		return nullptr;
	}

	auto &struct_type = bundle.logical_types[0];
	auto child_count = StructType::GetChildCount(struct_type);
	vector<LogicalType> raw_types;
	vector<string> raw_names;
	raw_types.reserve(child_count);
	raw_names.reserve(child_count);
	for (idx_t child_idx = 0; child_idx < child_count; child_idx++) {
		raw_types.push_back(StructType::GetChildType(struct_type, child_idx));
		raw_names.push_back(StructType::GetChildName(struct_type, child_idx));
	}

	auto raw_input = bundle;
	raw_input.logical_types = std::move(raw_types);
	raw_input.names = raw_names;
	raw_input.wrap_columns_as_struct = false;

	if (!project_input_names) {
		auto output = make_uniq<LazyRefDataChunk>();
		*output = std::move(raw_input);
		output->RecomputeCardinality();
		return output;
	}

	auto input_names = GetStructStringListField(payload, "input_names");
	if (input_names.empty()) {
		auto output = make_uniq<LazyRefDataChunk>();
		*output = std::move(raw_input);
		output->RecomputeCardinality();
		return output;
	}

	vector<idx_t> column_ids;
	column_ids.reserve(input_names.size());
	for (auto &input_name : input_names) {
		bool found = false;
		for (idx_t child_idx = 0; child_idx < raw_names.size(); child_idx++) {
			if (!StringUtil::CIEquals(raw_names[child_idx], input_name)) {
				continue;
			}
			column_ids.push_back(child_idx);
			found = true;
			break;
		}
		if (!found) {
			throw InvalidInputException("udf input column '%s' was not found in wrapped struct lazy input", input_name);
		}
	}
	return ProjectLazyDataChunk(raw_input, column_ids, input_names);
}

static bool IsRowPreservingPythonUDFLayoutPayload(const Value &payload) {
	return UDFModePreservesRows(ClassifyUDFMode(payload)) && HasStructField(payload, "scalar_arg_count") &&
	       HasStructField(payload, "ref_output_types");
}

static bool UsesCompleteRowPreservingBlockLayout(const Value &payload) {
	auto ray_block_stream = GetStructBoolField(payload, "produce_ray_block_stream");
	auto local_ref_bundle = GetStructBoolField(payload, "produce_ref_bundle_output");
	return (ray_block_stream.first && ray_block_stream.second) || (local_ref_bundle.first && local_ref_bundle.second);
}

static idx_t GetRowPreservingUDFArgCount(const Value &payload) {
	auto arg_count = GetStructIntField(payload, "scalar_arg_count");
	if (!arg_count.first) {
		throw InvalidInputException("row-preserving udf payload is missing scalar_arg_count");
	}
	return arg_count.second;
}

static void EnsureExecutor(ExecutionContext &context, UDFOperatorState &state) {
	if (state.executor) {
		return;
	}

	auto factory = GetUDFExecutorFactory();
	if (!factory) {
		throw InvalidInputException("UDF executor factory is not registered");
	}
	state.executor = factory(context.client, state.payload, state.config, state.actor_handles);
	if (!state.executor) {
		throw InternalException("UDF executor factory returned NULL");
	}
}

static void CompleteMaterializedSubmitResult(UDFOperatorState &state, const UDFResult &result) {
	if (!result.submit_complete) {
		return;
	}
	state.completed_batches++;
}

static bool TakeReadyResultOnce(ExecutionContext &context, UDFOperatorState &state) {
	if (!state.executor) {
		return false;
	}
	auto result = state.executor->TakeReadyResult(context.client);
	if (!result.first) {
		return false;
	}
	CompleteMaterializedSubmitResult(state, result.second);
	if (result.second.ref_outputs) {
		if (state.is_flat_map) {
			PreserveFlatMapLazyOutputShape(state, *result.second.ref_outputs);
		} else {
			PreserveScalarMapLazyOutputShape(state, *result.second.ref_outputs);
		}
		result.second.ref_outputs->RecomputeCardinality();
		if (result.second.handoff_output_lease) {
			auto handoff = std::move(result.second.handoff_output_lease);
			try {
				handoff();
			} catch (...) {
				ReleaseUDFOutputLease(result.second.release_output_lease);
				throw;
			}
		}
		if (result.second.release_output_lease) {
			auto token = make_shared_ptr<UDFOutputLeaseOwnership>(std::move(result.second.release_output_lease));
			for (auto &block : result.second.ref_outputs->blocks) {
				block.ownership_tokens.push_back(token);
			}
		}
		UDFPendingOutput pending;
		StoreLazyExecutionBatchOutput(pending.batch, std::move(result.second.ref_outputs));
		state.pending_outputs.push_back(std::move(pending));
		return true;
	}
	if (!result.second.rows || !result.second.outputs) {
		throw InvalidInputException("UDF executor returned a malformed result without rows or output payload");
	}
	auto output_chunk = BuildOutputChunk(*result.second.rows, *result.second.outputs, state.is_flat_map,
	                                     state.output_names, state.output_types_declared);
	PushMaterializedOutputBatchPieces(state.pending_outputs, std::move(output_chunk),
	                                  std::move(result.second.release_output_lease));
	return true;
}

// Drain one ready result from the executor without waiting.
static bool DrainReadyResult(ExecutionContext &context, UDFOperatorState &state) {
	if (!state.executor) {
		return false;
	}
	return TakeReadyResultOnce(context, state);
}

template <class OUTPUT, class POP_OUTPUT>
static bool DrainReadyResultsUntilLogicalDone(ExecutionContext &context, UDFOperatorState &state, POP_OUTPUT pop_output,
                                              OUTPUT &output) {
	while (state.submitted_batches != state.completed_batches || LogicalInflightBatches(state) != 0) {
		if (!DrainReadyResult(context, state)) {
			return false;
		}
		if (pop_output(context, state, output)) {
			return true;
		}
	}
	return false;
}

static void ReferenceColumnRange(ClientContext &context, DataChunk &input, idx_t column_offset, idx_t column_count,
                                 DataChunk &output) {
	if (column_offset + column_count > input.ColumnCount()) {
		throw InternalException("udf column range exceeds input column count");
	}
	vector<LogicalType> types;
	types.reserve(column_count);
	for (idx_t i = 0; i < column_count; i++) {
		types.push_back(input.data[column_offset + i].GetType());
	}
	output.InitializeEmpty(types);
	output.SetCardinality(input.size());
	for (idx_t i = 0; i < column_count; i++) {
		output.data[i].Reference(input.data[column_offset + i]);
	}
}

static bool TrySubmitArgsRaw(ExecutionContext &context, UDFOperatorState &state, DataChunk &input, idx_t &submit_id) {
	if (!state.executor) {
		throw InternalException("UDF executor is not initialized");
	}
	input.Flatten();
	bool submitted = false;
	if (state.is_flat_map) {
		DataChunk empty_rows;
		empty_rows.SetCardinality(0);
		submitted = state.executor->TrySubmitWithRetainedBytes(input, empty_rows, context.client,
		                                                       EstimateStreamingChunkBytes(input), submit_id);
	} else if (IsRowPreservingPythonUDFLayoutPayload(state.payload)) {
		auto arg_count = GetRowPreservingUDFArgCount(state.payload);
		if (arg_count > input.ColumnCount()) {
			throw InvalidInputException("scalar_arg_count %llu exceeds input column count %llu",
			                            static_cast<unsigned long long>(arg_count),
			                            static_cast<unsigned long long>(input.ColumnCount()));
		}
		if (UsesCompleteRowPreservingBlockLayout(state.payload)) {
			// A streamed block is the complete downstream batch. Send the scalar
			// argument prefix and passthrough suffix together so materialized
			// and lazy-ref inputs produce the same immutable output schema.
			DataChunk empty_rows;
			empty_rows.SetCardinality(0);
			submitted = state.executor->TrySubmitWithRetainedBytes(input, empty_rows, context.client,
			                                                       EstimateStreamingChunkBytes(input), submit_id);
		} else {
			DataChunk args;
			DataChunk rows;
			ReferenceColumnRange(context.client, input, 0, arg_count, args);
			ReferenceColumnRange(context.client, input, arg_count, input.ColumnCount() - arg_count, rows);
			submitted = state.executor->TrySubmitWithRetainedBytes(args, rows, context.client,
			                                                       EstimateStreamingChunkBytes(args), submit_id);
		}
	} else {
		submitted = state.executor->TrySubmitWithRetainedBytes(input, input, context.client,
		                                                       EstimateStreamingChunkBytes(input), submit_id);
	}
	if (!submitted) {
		submit_id = 0;
		return false;
	}
	if (submit_id == 0) {
		throw InternalException("streaming UDF submit returned invalid submit_id 0");
	}
	state.submitted_batches++;
	return true;
}

static bool TrySubmitMaterializedEnvelopeRaw(ExecutionContext &context, UDFOperatorState &state,
                                             vector<unique_ptr<DataChunk>> &chunks, idx_t retained_input_bytes,
                                             idx_t &submit_id) {
	if (!state.executor) {
		throw InternalException("UDF executor is not initialized");
	}
	if (chunks.empty()) {
		submit_id = 0;
		return false;
	}
	if (IsRowPreservingPythonUDFLayoutPayload(state.payload) && !UsesCompleteRowPreservingBlockLayout(state.payload)) {
		throw InvalidInputException("streaming UDF row-preserving layout does not support materialized envelope input");
	}
	for (auto &chunk : chunks) {
		if (!chunk) {
			throw InternalException("streaming UDF materialized envelope contains null chunk");
		}
		chunk->Flatten();
	}
	DataChunk empty_rows;
	empty_rows.SetCardinality(0);
	if (!state.executor->TrySubmitEnvelopeWithRetainedBytes(chunks, empty_rows, context.client, retained_input_bytes,
	                                                        submit_id)) {
		submit_id = 0;
		return false;
	}
	if (submit_id == 0) {
		throw InternalException("streaming UDF materialized envelope submit returned invalid submit_id 0");
	}
	state.submitted_batches++;
	return true;
}

static bool TrySubmitRefBundleRaw(ExecutionContext &context, UDFOperatorState &state, LazyRefDataChunk &bundle,
                                  idx_t &submit_id, idx_t retained_input_bytes) {
	if (!state.executor || !state.executor->SupportsRefBundleInput()) {
		throw InvalidInputException("udf executor does not support ref-bundle input");
	}
	const bool row_preserving = IsRowPreservingPythonUDFLayoutPayload(state.payload);
	auto canonical_bundle = CanonicalizeWrappedStructLazyInputForRefSubmit(state.payload, bundle, !row_preserving);
	auto &submit_bundle = canonical_bundle ? *canonical_bundle : bundle;
	submit_bundle.RecomputeCardinality();
	DataChunk passthrough_rows;
	passthrough_rows.SetCardinality(row_preserving ? submit_bundle.cardinality : 0);
	if (!state.executor->TrySubmitRefBundleWithRetainedBytes(submit_bundle, passthrough_rows, context.client,
	                                                         retained_input_bytes, submit_id)) {
		submit_id = 0;
		return false;
	}
	if (submit_id == 0) {
		throw InternalException("udf ref-bundle submit returned invalid submit_id 0");
	}
	state.submitted_batches++;
	return true;
}

// ─── UDFLocalState: wraps UDFOperatorState for INOUT path ─────────
struct UDFLocalState : public LocalTableFunctionState {
	UDFOperatorState inner;

	UDFLocalState(ClientContext &context, Value payload_p, shared_ptr<void> actor_handles_p = nullptr)
	    : inner(context, {}, std::move(payload_p), std::move(actor_handles_p)) {
	}
};

struct UDFGlobalState : public GlobalTableFunctionState {
	explicit UDFGlobalState(Value payload_p) : original_payload(std::move(payload_p)) {
	}

	idx_t MaxThreads(idx_t source_max_threads) override {
		return source_max_threads;
	}

	void PipelineMaxThreadsResolved(idx_t max_threads) override {
		UDFWorkerSlotDebugLog(StringUtil::Format(
		    "pipeline_max_threads_resolved udf_name=%s max_threads=%llu runtime_resolved=%s resolved_width=%llu",
		    DebugUDFName().c_str(), static_cast<unsigned long long>(max_threads), runtime_resolved ? "true" : "false",
		    static_cast<unsigned long long>(resolved_task_operator_width)));
		ResolveRuntime(max_threads, "pipeline_resolved");
	}

	const Value &ResolvedPayload() {
		if (!runtime_resolved) {
			throw InternalException("UDF runtime payload requested before pipeline max threads were resolved");
		}
		return resolved_payload;
	}

private:
	string DebugUDFName() const {
		return UDFDebugNameFromPayload(original_payload);
	}

	void ResolveRuntime(idx_t task_operator_width, const char *reason) {
		lock_guard<mutex> guard(resolve_lock);
		task_operator_width = MaxValue<idx_t>(idx_t(1), task_operator_width);
		if (runtime_resolved) {
			UDFWorkerSlotDebugLog(StringUtil::Format(
			    "resolve_runtime_skip udf_name=%s reason=%s requested_width=%llu resolved_width=%llu",
			    DebugUDFName().c_str(), reason ? reason : "<missing>",
			    static_cast<unsigned long long>(task_operator_width),
			    static_cast<unsigned long long>(resolved_task_operator_width)));
			return;
		}
		UDFWorkerSlotDebugLog(StringUtil::Format("resolve_runtime_commit udf_name=%s reason=%s width=%llu",
		                                         DebugUDFName().c_str(), reason ? reason : "<missing>",
		                                         static_cast<unsigned long long>(task_operator_width)));
		resolved_payload = ResolveUDFRuntimePayload(original_payload, task_operator_width);
		resolved_task_operator_width = task_operator_width;
		runtime_resolved = true;
	}

public:
	Value original_payload;
	Value resolved_payload;
	idx_t resolved_task_operator_width = 0;
	bool runtime_resolved = false;
	mutex resolve_lock;
};

static void PushMaterializedOutputBatchPieces(std::deque<UDFPendingOutput> &queue, unique_ptr<DataChunk> output_chunk,
                                              std::function<void()> release_output_lease) {
	if (!output_chunk || output_chunk->size() == 0) {
		ReleaseUDFOutputLease(release_output_lease);
		return;
	}
	if (output_chunk->size() <= STANDARD_VECTOR_SIZE) {
		auto pending = MakeMaterializedPendingOutput(std::move(output_chunk));
		pending.release_output_lease = std::move(release_output_lease);
		queue.push_back(std::move(pending));
		return;
	}
	output_chunk->Flatten();
	auto types = output_chunk->GetTypes();
	idx_t total = output_chunk->size();
	idx_t offset = 0;
	while (offset < total) {
		idx_t count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, total - offset);
		auto piece = make_uniq<DataChunk>();
		piece->Initialize(Allocator::DefaultAllocator(), types, count);
		for (idx_t col = 0; col < output_chunk->ColumnCount(); col++) {
			VectorOperations::Copy(output_chunk->data[col], piece->data[col], offset + count, offset, 0);
		}
		piece->SetCardinality(count);
		auto pending = MakeMaterializedPendingOutput(std::move(piece));
		if (offset + count >= total) {
			pending.release_output_lease = std::move(release_output_lease);
		}
		queue.push_back(std::move(pending));
		offset += count;
	}
}

// ─── INOUT Execute callback ─────────────────────────────────────────────────
static OperatorResultType UDFInOutExecute(ExecutionContext &context, TableFunctionInput &data, DataChunk &input,
                                          DataChunk &output) {
	auto &local_state = data.local_state->Cast<UDFLocalState>();
	auto &state = local_state.inner;
	state.execute_calls++;
	bool using_buffered_input = false;
	DataChunk *current_input = &input;
	if (!state.pending_inputs.empty()) {
		current_input = state.pending_inputs.front().get();
		using_buffered_input = true;
	}

	// 1. Return pending output if available
	if (!state.pending_outputs.empty() && BufferExecuteInputBeforeOutput(state, input)) {
		output.SetCardinality(0);
		return OperatorResultType::NEED_MORE_INPUT;
	}
	if (PopPendingOutput(context, state, output)) {
		return ReturnExecuteOutputForCurrentInput(state);
	}

	// 2. Drain one ready result, if the executor already has one queued.
	if (state.executor) {
		DrainReadyResult(context, state);
		if (!state.pending_outputs.empty() && BufferExecuteInputBeforeOutput(state, input)) {
			output.SetCardinality(0);
			return OperatorResultType::NEED_MORE_INPUT;
		}
		if (PopPendingOutput(context, state, output)) {
			return ReturnExecuteOutputForCurrentInput(state);
		}
	}

	// 3. Submit new input. TrySubmit is the task-admission boundary: it returns
	// false until the executor owns a query task lease for this input.
	if (current_input->size() > 0 && !state.input_consumed) {
		EnsureExecutor(context, state);
		idx_t submit_id = 0;
		if (!TrySubmitArgsRaw(context, state, *current_input, submit_id)) {
			output.SetCardinality(0);
			auto wakeup_result = RegisterBackpressureWakeup(context, state);
			if (wakeup_result == UDFWakeupRegistrationResult::ARMED) {
				return OperatorResultType::BLOCKED;
			}
			return OperatorResultType::HAVE_MORE_OUTPUT;
		}
		BufferDeferredPipelineInput(state, input, using_buffered_input);
		if (using_buffered_input) {
			state.pending_inputs.pop_front();
		}
		state.input_consumed = true;
	}

	// 4. Always request more input and let the downstream ready-result/finalize path
	// drain async results. Avoid returning BLOCKED here; the in/out pipeline
	// resume path is not preserving the consumed-input state correctly.
	if (state.input_consumed) {
		output.SetCardinality(0);
		state.input_consumed = false;
		state.need_more_returns++;
		return OperatorResultType::NEED_MORE_INPUT;
	}

	output.SetCardinality(0);
	state.need_more_returns++;
	return OperatorResultType::NEED_MORE_INPUT;
}

static OperatorResultType UDFInOutExecuteBatch(ExecutionContext &context, TableFunctionInput &data,
                                               ExecutionBatch &input, ExecutionBatch &output) {
	auto &local_state = data.local_state->Cast<UDFLocalState>();
	auto &state = local_state.inner;

	state.execute_calls++;
	bool using_buffered_input = false;
	unique_ptr<DataChunk> materialized_input;
	DataChunk *current_input = nullptr;
	LazyRefDataChunk *current_lazy_input = nullptr;
	if (!state.pending_inputs.empty()) {
		current_input = state.pending_inputs.front().get();
		using_buffered_input = true;
	} else if (input.kind == ExecutionBatchKind::LAZY_DATA_CHUNK && input.lazy && UDFExecutionBatchSize(input) > 0) {
		current_lazy_input = input.lazy.get();
		current_lazy_input->RecomputeCardinality();
	} else {
		materialized_input = ReferenceMaterializedUDFInputBatch(context, state, input, "udf_batch_input_barrier");
		current_input = materialized_input.get();
	}
	const auto current_rows =
	    current_lazy_input ? current_lazy_input->cardinality : (current_input ? current_input->size() : idx_t(0));
	LazyInputReplayToken current_lazy_token;
	bool current_lazy_already_consumed = false;
	if (current_lazy_input) {
		current_lazy_token = MakeLazyInputReplayToken(*current_lazy_input);
		current_lazy_already_consumed = LazyInputReplayTokenMatches(state.consumed_lazy_input, current_lazy_token);
	}
	const auto submit_rows = current_lazy_already_consumed ? idx_t(0) : current_rows;

	if (!state.pending_outputs.empty() && BufferExecuteBatchInputBeforeOutput(state, input)) {
		StoreEmptyUDFBatchOutput(context, state, data.bind_data, output);
		return OperatorResultType::NEED_MORE_INPUT;
	}
	if (PopPendingOutputBatch(context, state, output)) {
		return ReturnExecuteOutputForCurrentInput(state);
	}

	if (state.executor) {
		DrainReadyResult(context, state);
		if (!state.pending_outputs.empty() && BufferExecuteBatchInputBeforeOutput(state, input)) {
			StoreEmptyUDFBatchOutput(context, state, data.bind_data, output);
			return OperatorResultType::NEED_MORE_INPUT;
		}
		if (PopPendingOutputBatch(context, state, output)) {
			return ReturnExecuteOutputForCurrentInput(state);
		}
	}

	if (submit_rows > 0 && !state.input_consumed) {
		EnsureExecutor(context, state);
		auto can_submit_ref_bundle = current_lazy_input && state.executor->SupportsRefBundleInput() &&
		                             (state.is_flat_map || IsRowPreservingPythonUDFLayoutPayload(state.payload));
		bool submitted = false;
		idx_t submit_id = 0;
		if (can_submit_ref_bundle) {
			submitted = TrySubmitRefBundleRaw(context, state, *current_lazy_input, submit_id);
			if (submitted) {
				state.consumed_lazy_input = std::move(current_lazy_token);
			}
		} else {
			if (!current_input) {
				materialized_input =
				    ReferenceMaterializedUDFInputBatch(context, state, input, "udf_lazy_input_barrier");
				current_input = materialized_input.get();
			}
			submitted = TrySubmitArgsRaw(context, state, *current_input, submit_id);
		}
		if (!submitted) {
			StoreEmptyUDFBatchOutput(context, state, data.bind_data, output);
			auto wakeup_result = RegisterBackpressureWakeup(context, state);
			if (wakeup_result == UDFWakeupRegistrationResult::ARMED) {
				return OperatorResultType::BLOCKED;
			}
			return OperatorResultType::HAVE_MORE_OUTPUT;
		}

		if (using_buffered_input) {
			auto deferred_input =
			    ReferenceMaterializedUDFInputBatch(context, state, input, "udf_deferred_lazy_input_barrier");
			BufferPendingInput(state, *deferred_input);
			state.pending_inputs.pop_front();
		}
		state.input_consumed = true;
	}

	if (state.input_consumed) {
		StoreEmptyUDFBatchOutput(context, state, data.bind_data, output);
		state.input_consumed = false;
		state.need_more_returns++;
		return OperatorResultType::NEED_MORE_INPUT;
	}

	StoreEmptyUDFBatchOutput(context, state, data.bind_data, output);
	state.need_more_returns++;
	return OperatorResultType::NEED_MORE_INPUT;
}

// ─── INOUT FinalExecute callback ────────────────────────────────────────────
template <class OUTPUT, class POP_OUTPUT, class STORE_EMPTY_OUTPUT>
static OperatorFinalizeResultType UDFInOutFinalCommon(ExecutionContext &context, TableFunctionInput &data,
                                                      OUTPUT &output, POP_OUTPUT pop_output,
                                                      STORE_EMPTY_OUTPUT store_empty_output) {
	auto &local_state = data.local_state->Cast<UDFLocalState>();
	auto &state = local_state.inner;
	state.final_calls++;

	if (pop_output(context, state, output)) {
		return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
	}

	if (!state.pending_inputs.empty()) {
		EnsureExecutor(context, state);
		while (!state.pending_inputs.empty()) {
			DrainReadyResult(context, state);
			if (pop_output(context, state, output)) {
				return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
			}
			auto pending_input = std::move(state.pending_inputs.front());
			state.pending_inputs.pop_front();
			idx_t submit_id = 0;
			if (!TrySubmitArgsRaw(context, state, *pending_input, submit_id)) {
				state.pending_inputs.push_front(std::move(pending_input));
				auto wakeup_result = RegisterBackpressureWakeup(context, state);
				if (wakeup_result == UDFWakeupRegistrationResult::READY) {
					continue;
				}
				store_empty_output(context, state, data.bind_data, output);
				if (wakeup_result == UDFWakeupRegistrationResult::ARMED) {
					return OperatorFinalizeResultType::BLOCKED;
				}
				return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
			}
		}
	}

	if (!state.executor) {
		store_empty_output(context, state, data.bind_data, output);
		return OperatorFinalizeResultType::FINISHED;
	}

	if (!state.finished_submitting) {
		state.finished_submitting = true;
		state.executor->FinishedSubmitting(context.client);
	}

	DrainReadyResult(context, state);
	if (pop_output(context, state, output)) {
		return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
	}

	bool all_done = state.executor->AllTasksFinished(context.client);
	const auto logical_done = state.submitted_batches == state.completed_batches && LogicalInflightBatches(state) == 0;
	if (all_done && !logical_done) {
		if (DrainReadyResultsUntilLogicalDone(context, state, pop_output, output)) {
			return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
		}
	}
	const auto logical_done_after_drain =
	    state.submitted_batches == state.completed_batches && LogicalInflightBatches(state) == 0;
	if (all_done && logical_done_after_drain) {
		DrainReadyResult(context, state);
		if (pop_output(context, state, output)) {
			return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
		}
		store_empty_output(context, state, data.bind_data, output);
		return OperatorFinalizeResultType::FINISHED;
	}
	if (all_done && !logical_done_after_drain) {
		throw InvalidInputException("UDF executor finished before completing all submitted batches");
	}

	auto wakeup_result = RegisterAsyncWakeup(context, state);
	if (wakeup_result == UDFWakeupRegistrationResult::ARMED) {
		store_empty_output(context, state, data.bind_data, output);
		return OperatorFinalizeResultType::BLOCKED;
	}
	if (wakeup_result == UDFWakeupRegistrationResult::UNSUPPORTED) {
		store_empty_output(context, state, data.bind_data, output);
		return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
	}
	DrainReadyResult(context, state);
	if (pop_output(context, state, output)) {
		return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
	}
	store_empty_output(context, state, data.bind_data, output);
	all_done = state.executor->AllTasksFinished(context.client);
	if (all_done && state.submitted_batches == state.completed_batches && LogicalInflightBatches(state) == 0) {
		return OperatorFinalizeResultType::FINISHED;
	}
	if (all_done) {
		throw InvalidInputException("UDF executor finished before completing all submitted batches");
	}
	return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
}

static OperatorFinalizeResultType UDFInOutFinal(ExecutionContext &context, TableFunctionInput &data,
                                                DataChunk &output) {
	auto pop_output = [](ExecutionContext &context, UDFOperatorState &state, DataChunk &output) {
		return PopPendingOutput(context, state, output);
	};
	auto store_empty_output = [](ExecutionContext &, UDFOperatorState &, optional_ptr<const FunctionData>,
	                             DataChunk &output) {
		output.SetCardinality(0);
	};
	return UDFInOutFinalCommon(context, data, output, pop_output, store_empty_output);
}

static OperatorFinalizeResultType UDFInOutFinalBatch(ExecutionContext &context, TableFunctionInput &data,
                                                     ExecutionBatch &output) {
	auto pop_output = [](ExecutionContext &context, UDFOperatorState &state, ExecutionBatch &output) {
		return PopPendingOutputBatch(context, state, output);
	};
	auto store_empty_output = [](ExecutionContext &context, UDFOperatorState &state,
	                             optional_ptr<const FunctionData> bind_data, ExecutionBatch &output) {
		StoreEmptyUDFBatchOutput(context, state, bind_data, output);
	};
	return UDFInOutFinalCommon(context, data, output, pop_output, store_empty_output);
}

// ─── INOUT InitLocal callback ───────────────────────────────────────────────
static unique_ptr<LocalTableFunctionState> UDFInitLocal(ExecutionContext &context, TableFunctionInitInput &input,
                                                        GlobalTableFunctionState *global_state) {
	auto &bind_data = input.bind_data->Cast<UDFFunctionData>();
	auto payload = bind_data.payload;
	if (global_state) {
		auto &udf_global_state = global_state->Cast<UDFGlobalState>();
		payload = udf_global_state.ResolvedPayload();
	}
	auto local_state = make_uniq<UDFLocalState>(context.client, std::move(payload), bind_data.actor_handles);
	return std::move(local_state);
}

static unique_ptr<GlobalTableFunctionState> UDFInitGlobal(ClientContext &, TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<UDFFunctionData>();
	return make_uniq<UDFGlobalState>(bind_data.payload);
}

} // namespace

static void AppendUDFExecutionConfigParams(InsertionOrderPreservingMap<string> &result, const Value &payload) {
	auto actor_number = GetStructIntField(payload, "actor_number");
	if (actor_number.first && actor_number.second > 0) {
		result["actor_number"] = std::to_string(actor_number.second);
	}
	auto actor_pool_size = GetStructIntField(payload, "actor_pool_size");
	if (actor_pool_size.first && actor_pool_size.second > 0) {
		result["actor_pool_size"] = std::to_string(actor_pool_size.second);
	}
	auto ray_actor_thread_policy = GetStructStringField(payload, "ray_actor_thread_policy");
	if (ray_actor_thread_policy.first && !ray_actor_thread_policy.second.empty()) {
		result["ray_actor_thread_policy"] = ray_actor_thread_policy.second;
	}
	auto worker_slots = GetStructIntField(payload, "udf_worker_slots");
	if (worker_slots.first && worker_slots.second > 0) {
		result["udf_worker_slots"] = std::to_string(worker_slots.second);
	}
	auto min_task_batch_size = GetStructIntField(payload, "min_task_batch_size");
	if (min_task_batch_size.first && min_task_batch_size.second > 0) {
		result["min_task_batch_size"] = std::to_string(min_task_batch_size.second);
	}
	auto preserve_compute_batch_boundaries = GetStructBoolField(payload, "preserve_compute_batch_boundaries");
	if (preserve_compute_batch_boundaries.first && preserve_compute_batch_boundaries.second) {
		result["preserve_compute_batch_boundaries"] = "true";
	}
	auto target_max_batch_bytes = GetStructIntField(payload, "udf_target_max_batch_bytes");
	if (target_max_batch_bytes.first && target_max_batch_bytes.second > 0) {
		result["udf_target_max_batch_bytes"] = std::to_string(target_max_batch_bytes.second);
	}
	auto task_input_max_bytes = GetStructIntField(payload, "udf_task_input_max_bytes");
	if (task_input_max_bytes.first && task_input_max_bytes.second > 0) {
		result["udf_task_input_max_bytes"] = std::to_string(task_input_max_bytes.second);
	}
	auto output_target_max_bytes = GetStructIntField(payload, "udf_output_target_max_bytes");
	if (output_target_max_bytes.first && output_target_max_bytes.second > 0) {
		result["udf_output_target_max_bytes"] = std::to_string(output_target_max_bytes.second);
	}
	auto cpus = GetStructNumericFieldString(payload, "cpus");
	if (cpus.first) {
		result["cpus"] = cpus.second;
	}
	auto gpus = GetStructNumericFieldString(payload, "gpus");
	if (gpus.first) {
		result["gpus"] = gpus.second;
	}
}

static void AppendUDFExecutorStatsParams(InsertionOrderPreservingMap<string> &result, UDFExecutor *executor) {
	if (!executor) {
		return;
	}
	auto stats = executor->Stats();
	for (const auto &entry : stats) {
		if (StringUtil::StartsWith(entry.first, "udf_")) {
			result[entry.first] = entry.second;
		}
	}
}

static InsertionOrderPreservingMap<string> UDFTableFunctionToString(TableFunctionToStringInput &input) {
	InsertionOrderPreservingMap<string> result;
	result["Name"] = input.table_function.name.empty() ? "udf" : input.table_function.name;
	if (!input.bind_data) {
		return result;
	}
	auto &bind_data = input.bind_data->Cast<UDFFunctionData>();
	auto udf_name = GetStructStringField(bind_data.payload, "udf_name");
	if (udf_name.first && !udf_name.second.empty()) {
		result["udf_name"] = udf_name.second;
	}
	auto call_mode = GetStructStringField(bind_data.payload, "call_mode");
	if (call_mode.first && !call_mode.second.empty()) {
		result["call_mode"] = call_mode.second;
	}
	auto execution_backend = GetStructStringField(bind_data.payload, "execution_backend");
	if (execution_backend.first && !execution_backend.second.empty()) {
		result["execution_backend"] = execution_backend.second;
	}
	auto row_preserving = GetStructBoolField(bind_data.payload, "row_preserving");
	if (row_preserving.first) {
		result["row_preserving"] = row_preserving.second ? "true" : "false";
	}
	AppendUDFExecutionConfigParams(result, bind_data.payload);
	if (IsRowPreservingPythonUDFLayoutPayload(bind_data.payload)) {
		auto arg_count = GetStructIntField(bind_data.payload, "scalar_arg_count");
		if (arg_count.first) {
			result["scalar_arg_count"] = std::to_string(arg_count.second);
		}
	}
	auto produce_ref_bundle = GetStructBoolField(bind_data.payload, "produce_ref_bundle_output");
	auto produce_ray_block_stream = GetStructBoolField(bind_data.payload, "produce_ray_block_stream");
	if ((produce_ref_bundle.first && produce_ref_bundle.second) ||
	    (produce_ray_block_stream.first && produce_ray_block_stream.second)) {
		result["lazy_ref_boundary"] = "strict_ref_aware";
	}
	if (produce_ref_bundle.first && produce_ref_bundle.second) {
		result["ref_bundle_output"] = "invalid_non_streaming";
	}
	if (produce_ray_block_stream.first && produce_ray_block_stream.second) {
		result["ray_block_stream_output"] = "direct_block_metadata_pair";
	}
	return result;
}

static InsertionOrderPreservingMap<string> UDFTableFunctionDynamicToString(TableFunctionDynamicToStringInput &input) {
	InsertionOrderPreservingMap<string> result;
	if (!input.global_state && input.local_state) {
		auto &local_state = input.local_state->Cast<UDFLocalState>();
		AppendUDFExecutorStatsParams(result, local_state.inner.executor.get());
	}
	return result;
}

struct StreamingUDFConfig {
	idx_t compute_batch_rows = 0;
	// Soft lower bound matching Ray Data's min_rows_per_bundle: preserve
	// complete upstream blocks and coalesce only undersized blocks until this
	// row count is reached. EOS and byte pressure may still submit a short tail.
	idx_t min_task_batch_rows = 0;
	idx_t task_input_max_bytes = 0;
	idx_t output_target_bytes = 0;
};

struct StreamingSubmitPlan {
	// target_rows ends at the earliest useful upstream work-unit boundary.
	// Only an EOS tail or a byte-forced split may be an incomplete compute batch.
	idx_t target_rows = 0;
	bool allow_incomplete_compute_batch = false;

	explicit operator bool() const {
		return target_rows > 0;
	}
};

struct StreamingPendingInputPiece {
	unique_ptr<DataChunk> chunk;
	idx_t bytes = 0;
	idx_t row_offset = 0;
};

struct StreamingPendingLazyInput {
	unique_ptr<LazyRefDataChunk> bundle;
	idx_t rows = 0;
	idx_t bytes = 0;
};

struct StreamingPendingMaterializedEnvelope {
	vector<unique_ptr<DataChunk>> chunks;
	idx_t rows = 0;
	idx_t bytes = 0;
};

struct StreamingInflightBatch {
	idx_t submit_id = 0;
	idx_t total_rows = 0;
	idx_t bytes = 0;
	idx_t emitted_rows = 0;
};

struct StreamingReadyOutput {
	ExecutionBatch batch;
	idx_t bytes = 0;
	idx_t submit_id = 0;
	std::function<void()> handoff_output_lease;
	std::function<void()> release_output_lease;
	bool lease_handed_off = false;
};

struct StreamingHandoffCounters {
	atomic<idx_t> outputs {0};
	atomic<idx_t> rows {0};
	atomic<idx_t> bytes {0};
	atomic<idx_t> max_outputs {0};
	atomic<idx_t> max_rows {0};
	atomic<idx_t> max_bytes {0};
};

static StreamingUDFConfig ResolveStreamingUDFConfig(const Value &payload, idx_t task_operator_width = 1);

struct StreamingUDFState : public StateWithBlockableTasks {
	StreamingUDFState(Value payload_p, shared_ptr<void> actor_handles_p)
	    : original_payload(std::move(payload_p)), actor_handles(std::move(actor_handles_p)) {
		payload = original_payload;
		config = ResolveStreamingUDFConfig(original_payload);
		UDFWorkerSlotDebugLog(StringUtil::Format("streaming_ctor_unresolved udf_name=%s initial_config_width=1",
		                                         UDFDebugNameFromPayload(original_payload).c_str()));
	}

	void ResolveRuntime(const unique_lock<mutex> &guard, idx_t task_operator_width, bool operator_width_resolved) {
		VerifyLock(guard);
		task_operator_width = MaxValue<idx_t>(idx_t(1), task_operator_width);
		UDFWorkerSlotDebugLog(StringUtil::Format(
		    "streaming_resolve_request udf_name=%s requested_width=%llu operator_width_resolved=%s "
		    "runtime_resolved=%s runtime_operator_width_resolved=%s current_width=%llu has_op=%s "
		    "submitted_batches=%llu",
		    UDFDebugNameFromPayload(original_payload).c_str(), static_cast<unsigned long long>(task_operator_width),
		    operator_width_resolved ? "true" : "false", runtime_resolved ? "true" : "false",
		    runtime_operator_width_resolved ? "true" : "false",
		    static_cast<unsigned long long>(resolved_task_operator_width), op ? "true" : "false",
		    static_cast<unsigned long long>(submitted_batches.load(std::memory_order_relaxed))));
		if (runtime_resolved && runtime_operator_width_resolved) {
			UDFWorkerSlotDebugLog(StringUtil::Format(
			    "streaming_resolve_skip udf_name=%s reason=already_operator_width_resolved requested_width=%llu "
			    "current_width=%llu",
			    UDFDebugNameFromPayload(original_payload).c_str(), static_cast<unsigned long long>(task_operator_width),
			    static_cast<unsigned long long>(resolved_task_operator_width)));
			return;
		}
		if (runtime_resolved && !operator_width_resolved) {
			UDFWorkerSlotDebugLog(StringUtil::Format(
			    "streaming_resolve_skip udf_name=%s reason=non_operator_request_after_resolved requested_width=%llu "
			    "current_width=%llu",
			    UDFDebugNameFromPayload(original_payload).c_str(), static_cast<unsigned long long>(task_operator_width),
			    static_cast<unsigned long long>(resolved_task_operator_width)));
			return;
		}
		if (runtime_resolved && resolved_task_operator_width == task_operator_width) {
			runtime_operator_width_resolved = runtime_operator_width_resolved || operator_width_resolved;
			UDFWorkerSlotDebugLog(StringUtil::Format(
			    "streaming_resolve_skip udf_name=%s reason=same_width requested_width=%llu "
			    "operator_width_resolved=%s current_width=%llu",
			    UDFDebugNameFromPayload(original_payload).c_str(), static_cast<unsigned long long>(task_operator_width),
			    operator_width_resolved ? "true" : "false",
			    static_cast<unsigned long long>(resolved_task_operator_width)));
			return;
		}
		if (op || submitted_batches.load(std::memory_order_relaxed) > 0) {
			UDFWorkerSlotDebugLog(StringUtil::Format(
			    "streaming_resolve_skip udf_name=%s reason=executor_or_submissions_started requested_width=%llu "
			    "current_width=%llu has_op=%s submitted_batches=%llu",
			    UDFDebugNameFromPayload(original_payload).c_str(), static_cast<unsigned long long>(task_operator_width),
			    static_cast<unsigned long long>(resolved_task_operator_width), op ? "true" : "false",
			    static_cast<unsigned long long>(submitted_batches.load(std::memory_order_relaxed))));
			return;
		}
		UDFWorkerSlotDebugLog(StringUtil::Format(
		    "streaming_resolve_commit udf_name=%s width=%llu operator_width_resolved=%s previous_width=%llu",
		    UDFDebugNameFromPayload(original_payload).c_str(), static_cast<unsigned long long>(task_operator_width),
		    operator_width_resolved ? "true" : "false", static_cast<unsigned long long>(resolved_task_operator_width)));
		payload = ResolveUDFRuntimePayload(original_payload, task_operator_width);
		config = ResolveStreamingUDFConfig(payload, task_operator_width);
		resolved_task_operator_width = task_operator_width;
		runtime_resolved = true;
		runtime_operator_width_resolved = operator_width_resolved;
	}

	Value original_payload;
	Value payload;
	shared_ptr<void> actor_handles;
	StreamingUDFConfig config;
	idx_t resolved_task_operator_width = 0;
	bool runtime_resolved = false;
	bool runtime_operator_width_resolved = false;
	unique_ptr<UDFOperatorState> op;
	bool wakeup_registered = false;
	bool output_consumer_registered = false;
	bool sink_finished = false;
	bool executor_finished_submitting = false;
	bool source_finished = false;
	bool dispatcher_finished = false;
	bool has_error = false;
	string error;
	std::deque<StreamingPendingInputPiece> pending_inputs;
	std::deque<StreamingPendingLazyInput> pending_lazy_inputs;
	// Once task admission has registered a waiter, its input identity and byte
	// commitment are immutable until that exact submit is granted or cancelled.
	StreamingPendingMaterializedEnvelope planned_materialized_submit;
	StreamingPendingLazyInput planned_lazy_submit;
	idx_t pending_rows = 0;
	idx_t reserved_rows = 0;
	idx_t reserved_bytes = 0;
	idx_t pending_bytes = 0;
	std::unordered_map<idx_t, StreamingInflightBatch> inflight_batches;
	idx_t inflight_rows = 0;
	idx_t inflight_bytes = 0;
	std::deque<StreamingReadyOutput> ready_outputs;
	idx_t ready_rows = 0;
	idx_t ready_bytes = 0;
	shared_ptr<StreamingHandoffCounters> handoff_counters = make_shared_ptr<StreamingHandoffCounters>();
	std::deque<StreamingReadyOutput> deferred_outputs;
	idx_t deferred_output_rows = 0;
	idx_t deferred_output_bytes = 0;
	atomic<idx_t> output_capacity_rows_snapshot {0};
	atomic<idx_t> output_capacity_bytes_snapshot {0};
	atomic<idx_t> output_capacity_item_bytes_snapshot {0};
	mutex output_event_lock;
	std::deque<UDFOutputEvent> pending_output_events;
	atomic<idx_t> sink_calls {0};
	atomic<idx_t> source_calls {0};
	atomic<idx_t> submitted_batches {0};
	atomic<idx_t> completed_batches {0};
	atomic<idx_t> blocked_sinks {0};
	atomic<idx_t> blocked_sources {0};
	atomic<idx_t> source_have_more {0};
	atomic<idx_t> source_blocked_empty {0};
	atomic<idx_t> result_callbacks {0};
	atomic<idx_t> deferred_output_events {0};
	atomic<idx_t> ready_empty_to_nonempty {0};
	atomic<idx_t> dispatcher_finished_notifications {0};
	atomic<idx_t> notify_space_available {0};
	atomic<idx_t> source_wake_one {0};
	atomic<idx_t> source_wake_all {0};
	atomic<idx_t> queued_output_events {0};
	atomic<idx_t> stats_tick {0};
	atomic<idx_t> accepted_input_rows {0};
	atomic<idx_t> accepted_input_bytes {0};
	atomic<idx_t> submitted_input_rows {0};
	atomic<idx_t> submitted_input_bytes {0};
	atomic<idx_t> completed_input_rows {0};
	atomic<idx_t> completed_input_bytes {0};
	atomic<idx_t> produced_output_rows {0};
	atomic<idx_t> produced_output_bytes {0};
	atomic<idx_t> emitted_output_rows {0};
	atomic<idx_t> emitted_output_bytes {0};
	atomic<double> upstream_progress_done {0.0};
	atomic<double> upstream_progress_total {0.0};
	atomic<bool> upstream_progress_valid {false};
	atomic<idx_t> resolved_source_threads {0};
	atomic<idx_t> resolved_sink_threads {0};
	idx_t max_pending_rows = 0;
	idx_t max_ready_rows = 0;
	idx_t max_active_batches = 0;
	idx_t max_outstanding_rows = 0;
	bool streaming_can_block = true;
	vector<InterruptState> blocked_source_tasks;
	vector<InterruptState> blocked_control_tasks;
	std::weak_ptr<StreamingUDFState> self;
};

static void RefreshStreamingOutputCapacitySnapshotLocked(StreamingUDFState &state, const unique_lock<mutex> &guard);

static void UpdateStreamingOutputBytesLocked(StreamingUDFState &state, const unique_lock<mutex> &guard) {
	state.VerifyLock(guard);
	RefreshStreamingOutputCapacitySnapshotLocked(state, guard);
}

static void AtomicAddStreamingCounter(atomic<idx_t> &counter, idx_t value) {
	if (value == 0) {
		return;
	}
	auto current = counter.load(std::memory_order_relaxed);
	while (true) {
		auto next =
		    std::numeric_limits<idx_t>::max() - current < value ? std::numeric_limits<idx_t>::max() : current + value;
		if (counter.compare_exchange_weak(current, next, std::memory_order_relaxed)) {
			return;
		}
	}
}

static void AtomicMaxStreamingCounter(atomic<idx_t> &counter, idx_t value) {
	auto current = counter.load(std::memory_order_relaxed);
	while (current < value && !counter.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
	}
}

static void StreamingUDFDebugState(StreamingUDFState &state, const char *where, bool force = false,
                                   bool has_capacity = false, idx_t capacity = 0) {
	if (!StreamingUDFDebugEnabled()) {
		return;
	}
	auto tick = g_streaming_udf_debug_tick.fetch_add(1, std::memory_order_relaxed) + 1;
	if (!force && tick % 200 != 0) {
		return;
	}
	auto capacity_text =
	    has_capacity ? StringUtil::Format(" capacity=%llu", static_cast<unsigned long long>(capacity)) : string();
	StreamingUDFDebugLog(StringUtil::Format(
	    "state tick=%llu udf_name=%s where=%s%s pending_rows=%llu pending_bytes=%llu reserved_rows=%llu "
	    "reserved_bytes=%llu pending_lazy=%llu inflight_batches=%llu inflight_rows=%llu inflight_bytes=%llu "
	    "planned_materialized_rows=%llu planned_lazy_rows=%llu "
	    "ready_outputs=%llu ready_rows=%llu ready_bytes=%llu handoff_outputs=%llu handoff_rows=%llu "
	    "handoff_bytes=%llu deferred_outputs=%llu deferred_rows=%llu "
	    "deferred_bytes=%llu queued_events=%llu sink_finished=%s "
	    "source_finished=%s dispatcher_finished=%s executor_finished=%s submitted_batches=%llu completed_batches=%llu "
	    "result_callbacks=%llu notify_space=%llu source_calls=%llu sink_calls=%llu blocked_sources=%llu "
	    "blocked_sinks=%llu source_have_more=%llu source_blocked_empty=%llu wake_one=%llu wake_all=%llu "
	    "blocked_source_tasks=%llu blocked_control_tasks=%llu max_pending_rows=%llu max_ready_rows=%llu "
	    "max_active_batches=%llu config_compute_batch_rows=%llu config_min_task_batch_rows=%llu "
	    "config_task_input_max_bytes=%llu "
	    "config_output_target_bytes=%llu",
	    static_cast<unsigned long long>(tick), UDFDebugNameFromPayload(state.original_payload).c_str(), where,
	    capacity_text, static_cast<unsigned long long>(state.pending_rows),
	    static_cast<unsigned long long>(state.pending_bytes), static_cast<unsigned long long>(state.reserved_rows),
	    static_cast<unsigned long long>(state.reserved_bytes),
	    static_cast<unsigned long long>(state.pending_lazy_inputs.size()),
	    static_cast<unsigned long long>(state.inflight_batches.size()),
	    static_cast<unsigned long long>(state.inflight_rows), static_cast<unsigned long long>(state.inflight_bytes),
	    static_cast<unsigned long long>(state.planned_materialized_submit.rows),
	    static_cast<unsigned long long>(state.planned_lazy_submit.rows),
	    static_cast<unsigned long long>(state.ready_outputs.size()), static_cast<unsigned long long>(state.ready_rows),
	    static_cast<unsigned long long>(state.ready_bytes),
	    static_cast<unsigned long long>(state.handoff_counters->outputs.load(std::memory_order_relaxed)),
	    static_cast<unsigned long long>(state.handoff_counters->rows.load(std::memory_order_relaxed)),
	    static_cast<unsigned long long>(state.handoff_counters->bytes.load(std::memory_order_relaxed)),
	    static_cast<unsigned long long>(state.deferred_outputs.size()),
	    static_cast<unsigned long long>(state.deferred_output_rows),
	    static_cast<unsigned long long>(state.deferred_output_bytes),
	    static_cast<unsigned long long>(state.queued_output_events.load(std::memory_order_relaxed)),
	    state.sink_finished ? "true" : "false", state.source_finished ? "true" : "false",
	    state.dispatcher_finished ? "true" : "false", state.executor_finished_submitting ? "true" : "false",
	    static_cast<unsigned long long>(state.submitted_batches.load(std::memory_order_relaxed)),
	    static_cast<unsigned long long>(state.completed_batches.load(std::memory_order_relaxed)),
	    static_cast<unsigned long long>(state.result_callbacks.load(std::memory_order_relaxed)),
	    static_cast<unsigned long long>(state.notify_space_available.load(std::memory_order_relaxed)),
	    static_cast<unsigned long long>(state.source_calls.load(std::memory_order_relaxed)),
	    static_cast<unsigned long long>(state.sink_calls.load(std::memory_order_relaxed)),
	    static_cast<unsigned long long>(state.blocked_sources.load(std::memory_order_relaxed)),
	    static_cast<unsigned long long>(state.blocked_sinks.load(std::memory_order_relaxed)),
	    static_cast<unsigned long long>(state.source_have_more.load(std::memory_order_relaxed)),
	    static_cast<unsigned long long>(state.source_blocked_empty.load(std::memory_order_relaxed)),
	    static_cast<unsigned long long>(state.source_wake_one.load(std::memory_order_relaxed)),
	    static_cast<unsigned long long>(state.source_wake_all.load(std::memory_order_relaxed)),
	    static_cast<unsigned long long>(state.blocked_source_tasks.size()),
	    static_cast<unsigned long long>(state.blocked_control_tasks.size()),
	    static_cast<unsigned long long>(state.max_pending_rows), static_cast<unsigned long long>(state.max_ready_rows),
	    static_cast<unsigned long long>(state.max_active_batches),
	    static_cast<unsigned long long>(state.config.compute_batch_rows),
	    static_cast<unsigned long long>(state.config.min_task_batch_rows),
	    static_cast<unsigned long long>(state.config.task_input_max_bytes),
	    static_cast<unsigned long long>(state.config.output_target_bytes)));
}

struct StreamingUDFGlobalSinkState : public GlobalSinkState {
	explicit StreamingUDFGlobalSinkState(std::shared_ptr<StreamingUDFState> state_p) : state(std::move(state_p)) {
	}

	idx_t MaxThreads(idx_t source_max_threads) override {
		return source_max_threads;
	}

	void PipelineMaxThreadsResolved(idx_t max_threads) override {
		auto guard = state->Lock();
		state->resolved_sink_threads.store(MaxValue<idx_t>(idx_t(1), max_threads), std::memory_order_relaxed);
		state->ResolveRuntime(guard, max_threads, true);
	}

	std::shared_ptr<StreamingUDFState> state;
};

struct StreamingUDFLocalSinkState : public LocalSinkState {
	bool consumed_blocked_input = false;
};

struct StreamingUDFGlobalSourceState : public GlobalSourceState {
	explicit StreamingUDFGlobalSourceState(std::shared_ptr<StreamingUDFState> state_p) : state(std::move(state_p)) {
	}

	idx_t MaxThreads() override {
		// This source drains one stateful output queue. Multiple pipeline tasks can
		// each pull a block and then park at a backpressured downstream sink,
		// moving those blocks outside the queue's byte accounting. Keep exactly
		// one drain task; stage/slot parallelism remains independent.
		state->resolved_source_threads.store(1, std::memory_order_relaxed);
		return 1;
	}

	std::shared_ptr<StreamingUDFState> state;
};

struct StreamingUDFLocalSourceState : public LocalSourceState {};

static bool StreamingReadyFull(const StreamingUDFState &state);
static bool StreamingOutputBackpressured(const StreamingUDFState &state);
static idx_t StreamingOutputByteCapacity(const StreamingUDFState &state);
static idx_t StreamingOutputItemByteCapacity(const StreamingUDFState &state);
static idx_t StreamingOutputEventCapacity(const StreamingUDFState &state);

static StreamingUDFConfig ResolveStreamingUDFConfig(const Value &payload, idx_t task_operator_width) {
	StreamingUDFConfig config;
	auto streaming_breaker = GetStructBoolField(payload, "streaming_breaker");
	if (!streaming_breaker.first || !streaming_breaker.second) {
		throw InvalidInputException("streaming UDF requires payload.streaming_breaker=true");
	}
	auto async_mode = GetStructBoolField(payload, "async_mode");
	if (async_mode.first && async_mode.second) {
		throw InvalidInputException("streaming_breaker=True does not support async_mode=True");
	}
	auto execution_backend = GetStructStringField(payload, "execution_backend");
	const bool supports_streaming_backend =
	    execution_backend.first &&
	    (execution_backend.second == "ray_actor" || execution_backend.second == "ray_task" ||
	     execution_backend.second == "subprocess_task" || execution_backend.second == "subprocess_actor");
	if (!supports_streaming_backend) {
		throw InvalidInputException(
		    "streaming UDF requires execution_backend=ray_actor, ray_task, subprocess_task or subprocess_actor");
	}
	if (!HasStructField(payload, "output_schema")) {
		throw InvalidInputException("streaming UDF requires map_batches/flat_map output_schema");
	}

	auto batch_size = GetStructIntField(payload, "batch_size");
	if (batch_size.first && batch_size.second > 0) {
		config.compute_batch_rows = batch_size.second;
	}
	auto min_task_batch_size = GetStructIntField(payload, "min_task_batch_size");
	if (min_task_batch_size.first && min_task_batch_size.second > 0) {
		if (config.compute_batch_rows == 0) {
			throw InvalidInputException("streaming UDF min_task_batch_size requires batch_size");
		}
		if (min_task_batch_size.second < config.compute_batch_rows) {
			throw InvalidInputException("streaming UDF min_task_batch_size must be at least batch_size");
		}
		config.min_task_batch_rows = min_task_batch_size.second;
	}

	auto task_input_max_bytes = GetStructIntField(payload, "udf_task_input_max_bytes");
	if (!task_input_max_bytes.first || task_input_max_bytes.second <= 0) {
		throw InvalidInputException("streaming UDF requires positive udf_task_input_max_bytes");
	}
	config.task_input_max_bytes = task_input_max_bytes.second;
	auto output_target_max_bytes = GetStructIntField(payload, "udf_output_target_max_bytes");
	if (!output_target_max_bytes.first || output_target_max_bytes.second <= 0) {
		throw InvalidInputException("streaming UDF requires positive udf_output_target_max_bytes");
	}
	config.output_target_bytes = output_target_max_bytes.second;
	return config;
}

static void SetStreamingError(StreamingUDFState &state, const string &msg);
static bool StreamingTerminalReady(const StreamingUDFState &state);

static bool RunStreamingWakeCallbacksUnlocked(StreamingUDFState &state, vector<InterruptState> callbacks,
                                              unique_lock<mutex> &guard) {
	if (callbacks.empty()) {
		return false;
	}
	auto shared_state = state.self.lock();
	if (!shared_state) {
		throw InternalException("streaming UDF deferred wakeup queue is not initialized");
	}
	if (!state.op || !state.op->executor) {
		// A zero-input UDF can reach its terminal state without ever creating an
		// executor. Its blocked source tasks are already descheduled, so wake them
		// directly outside the state lock. Every non-terminal path still requires
		// the joined executor dispatcher for deferred wakeup ordering.
		if (!StreamingTerminalReady(state)) {
			throw InternalException("streaming UDF deferred wakeup queue is not initialized");
		}
		guard.unlock();
		for (auto &entry : callbacks) {
			try {
				entry.Callback();
			} catch (const std::exception &ex) {
				SetStreamingError(*shared_state,
				                  StringUtil::Format("streaming UDF terminal wakeup callback failed: %s", ex.what()));
			} catch (...) {
				SetStreamingError(*shared_state,
				                  "streaming UDF terminal wakeup callback failed with an unknown exception");
			}
		}
		guard.lock();
		return true;
	}
	guard.unlock();
	try {
		// Every queued function owns at least one blocked task callback, so queue
		// growth is bounded by the query's pipeline task count. The joined UDF
		// dispatcher executes it on its next event-loop turn, after the target task
		// has had a chance to finish descheduling.
		state.op->executor->EnqueueDeferredWakeup([shared_state = std::move(shared_state),
		                                           callbacks = std::move(callbacks)]() mutable {
			for (auto &entry : callbacks) {
				try {
					entry.Callback();
				} catch (const std::exception &ex) {
					SetStreamingError(*shared_state,
					                  StringUtil::Format("streaming UDF wakeup callback failed: %s", ex.what()));
				} catch (...) {
					SetStreamingError(*shared_state, "streaming UDF wakeup callback failed with an unknown exception");
				}
			}
		});
	} catch (...) {
		guard.lock();
		throw;
	}
	guard.lock();
	return true;
}

static bool BlockStreamingTask(StreamingUDFState &state, const unique_lock<mutex> &guard,
                               const InterruptState &interrupt_state, bool source) {
	state.VerifyLock(guard);
	if (!state.streaming_can_block) {
		return false;
	}
	if (source) {
		state.blocked_source_tasks.push_back(interrupt_state);
	} else {
		state.blocked_control_tasks.push_back(interrupt_state);
	}
	return true;
}

static vector<InterruptState> TakeStreamingSourceTasks(StreamingUDFState &state, const unique_lock<mutex> &guard) {
	state.VerifyLock(guard);
	vector<InterruptState> callbacks;
	callbacks.swap(state.blocked_source_tasks);
	return callbacks;
}

static vector<InterruptState> TakeStreamingControlTasks(StreamingUDFState &state, const unique_lock<mutex> &guard) {
	state.VerifyLock(guard);
	vector<InterruptState> callbacks;
	callbacks.swap(state.blocked_control_tasks);
	return callbacks;
}

static void PreventStreamingBlocking(StreamingUDFState &state, const unique_lock<mutex> &guard) {
	state.VerifyLock(guard);
	state.streaming_can_block = false;
	state.PreventBlocking(guard);
}

static void WakeStreamingUDFTasks(StreamingUDFState &state, unique_lock<mutex> &guard) {
	auto callbacks = TakeStreamingControlTasks(state, guard);
	RunStreamingWakeCallbacksUnlocked(state, std::move(callbacks), guard);
}

static bool WakeOneStreamingSource(StreamingUDFState &state, unique_lock<mutex> &guard) {
	state.VerifyLock(guard);
	// Source tasks are blocked through StateWithBlockableTasks, the same path
	// local exchange uses. Waking all blocked tasks is intentional: a callback
	// can be a no-op when the original task has already gone away, so a single
	// custom source wake can leave ready output stranded forever.
	auto callbacks = TakeStreamingSourceTasks(state, guard);
	if (RunStreamingWakeCallbacksUnlocked(state, std::move(callbacks), guard)) {
		state.source_wake_one.fetch_add(1);
		return true;
	}
	return false;
}

static void WakeAllStreamingSources(StreamingUDFState &state, unique_lock<mutex> &guard) {
	state.VerifyLock(guard);
	auto callbacks = TakeStreamingSourceTasks(state, guard);
	if (RunStreamingWakeCallbacksUnlocked(state, std::move(callbacks), guard)) {
		state.source_wake_all.fetch_add(1);
	}
}

static SourceResultType BlockStreamingSource(StreamingUDFState &state, const unique_lock<mutex> &guard,
                                             const InterruptState &interrupt_state) {
	state.VerifyLock(guard);
	return BlockStreamingTask(state, guard, interrupt_state, true) ? SourceResultType::BLOCKED
	                                                               : SourceResultType::FINISHED;
}

static SinkResultType BlockStreamingSink(StreamingUDFState &state, const unique_lock<mutex> &guard,
                                         const InterruptState &interrupt_state) {
	state.VerifyLock(guard);
	return BlockStreamingTask(state, guard, interrupt_state, false) ? SinkResultType::BLOCKED
	                                                                : SinkResultType::FINISHED;
}

static SinkFinalizeType BlockStreamingFinalize(StreamingUDFState &state, const unique_lock<mutex> &guard,
                                               const InterruptState &interrupt_state) {
	state.VerifyLock(guard);
	return BlockStreamingTask(state, guard, interrupt_state, false) ? SinkFinalizeType::BLOCKED
	                                                                : SinkFinalizeType::READY;
}

static void AcceptStreamingEvent(StreamingUDFState &state, UDFOutputEvent &&event);
static bool DrainStreamingOutputEventsLocked(StreamingUDFState &state, unique_lock<mutex> &guard);
static void QueueStreamingOutputEvent(StreamingUDFState &state, UDFOutputEvent &&event);
static void SetStreamingError(StreamingUDFState &state, const string &msg);
static void NotifyStreamingDispatcherFinished(StreamingUDFState &state);

static void TryWakeStreamingTasksForQueuedEvent(StreamingUDFState &state) {
	auto guard = state.Lock();
	// DATA events are consumed by the source side. Waking control tasks here can
	// race with task descheduling and strand the dispatcher in RescheduleTask.
	WakeOneStreamingSource(state, guard);
}

static void QueueStreamingOutputEvent(StreamingUDFState &state, UDFOutputEvent &&event) {
	{
		lock_guard<mutex> event_guard(state.output_event_lock);
		state.pending_output_events.push_back(std::move(event));
		state.queued_output_events.fetch_add(1, std::memory_order_relaxed);
	}
	state.output_capacity_rows_snapshot.store(0, std::memory_order_relaxed);
	state.output_capacity_bytes_snapshot.store(0, std::memory_order_relaxed);
	state.output_capacity_item_bytes_snapshot.store(0, std::memory_order_relaxed);
	TryWakeStreamingTasksForQueuedEvent(state);
}

static void EnsureStreamingExecutor(ExecutionContext &context, StreamingUDFState &state,
                                    const unique_lock<mutex> &guard) {
	state.VerifyLock(guard);
	if (!state.runtime_resolved) {
		throw InternalException("streaming UDF executor requested before pipeline max threads were resolved");
	}
	if (!state.op) {
		vector<unique_ptr<Expression>> empty_args;
		state.op = make_uniq<UDFOperatorState>(context.client, empty_args, state.payload, state.actor_handles);
		EnsureExecutor(context, *state.op);
	}
	if (!state.wakeup_registered && state.op->executor && state.op->executor->SupportsAsyncWakeup()) {
		auto weak_state = state.self;
		state.op->executor->RegisterWakeupCallback([weak_state]() {
			auto shared = weak_state.lock();
			if (!shared) {
				return;
			}
			auto guard = shared->Lock();
			StreamingUDFDebugState(*shared, "executor_wakeup", true);
			WakeStreamingUDFTasks(*shared, guard);
			WakeOneStreamingSource(*shared, guard);
		});
		state.wakeup_registered = true;
	}
	if (!state.output_consumer_registered && state.op->executor) {
		if (!state.op->executor->SupportsOutputConsumer()) {
			throw InvalidInputException("streaming UDF requires an executor that supports output consumers");
		}
		auto weak_state = state.self;
		UDFOutputConsumer consumer;
		consumer.data_capacity = [weak_state]() {
			auto shared = weak_state.lock();
			if (!shared) {
				return idx_t(0);
			}
			return shared->output_capacity_rows_snapshot.load(std::memory_order_relaxed);
		};
		consumer.data_byte_capacity = [weak_state]() {
			auto shared = weak_state.lock();
			if (!shared) {
				return idx_t(0);
			}
			return shared->output_capacity_bytes_snapshot.load(std::memory_order_relaxed);
		};
		consumer.data_item_byte_capacity = [weak_state]() {
			auto shared = weak_state.lock();
			if (!shared) {
				return idx_t(0);
			}
			return shared->output_capacity_item_bytes_snapshot.load(std::memory_order_relaxed);
		};
		consumer.accept_event = [weak_state](UDFOutputEvent &&event) {
			auto shared = weak_state.lock();
			if (!shared) {
				return;
			}
			QueueStreamingOutputEvent(*shared, std::move(event));
		};
		consumer.accept_error = [weak_state](const string &msg) {
			auto shared = weak_state.lock();
			if (!shared) {
				return;
			}
			UDFOutputEvent event;
			event.kind = UDFOutputEventKind::ERROR;
			event.submit_complete = true;
			event.error = msg;
			QueueStreamingOutputEvent(*shared, std::move(event));
		};
		consumer.notify_finished = [weak_state]() {
			auto shared = weak_state.lock();
			if (!shared) {
				return;
			}
			UDFOutputEvent event;
			event.kind = UDFOutputEventKind::FINISHED;
			event.submit_complete = true;
			QueueStreamingOutputEvent(*shared, std::move(event));
		};
		state.op->executor->RegisterOutputConsumer(std::move(consumer));
		state.output_consumer_registered = true;
		RefreshStreamingOutputCapacitySnapshotLocked(state, guard);
	}
}

static bool StreamingReadyFull(const StreamingUDFState &state) {
	return state.ready_bytes >= state.config.output_target_bytes;
}

static bool StreamingOutputBackpressured(const StreamingUDFState &state) {
	return StreamingReadyFull(state) || !state.deferred_outputs.empty();
}

static idx_t StreamingOutputByteCapacity(const StreamingUDFState &state) {
	if (state.has_error || state.source_finished || StreamingOutputBackpressured(state) ||
	    state.queued_output_events.load(std::memory_order_relaxed) > 0) {
		return idx_t(0);
	}
	if (state.ready_bytes >= state.config.output_target_bytes) {
		return idx_t(0);
	}
	return state.config.output_target_bytes - state.ready_bytes;
}

static idx_t StreamingOutputItemByteCapacity(const StreamingUDFState &state) {
	auto byte_capacity = StreamingOutputByteCapacity(state);
	if (byte_capacity == 0) {
		return idx_t(0);
	}
	// A producer block may use the full registered target.  A smaller residual
	// window is temporary backpressure, not a smaller legal block size.
	if (byte_capacity < state.config.output_target_bytes) {
		return idx_t(0);
	}
	return state.config.output_target_bytes;
}

static idx_t StreamingOutputEventCapacity(const StreamingUDFState &state) {
	auto byte_capacity = StreamingOutputByteCapacity(state);
	if (byte_capacity == 0) {
		return idx_t(0);
	}
	if (byte_capacity == std::numeric_limits<idx_t>::max()) {
		return byte_capacity;
	}
	auto item_bytes = StreamingOutputItemByteCapacity(state);
	if (item_bytes == 0 || byte_capacity < item_bytes) {
		return idx_t(0);
	}
	return byte_capacity / item_bytes;
}

static void RefreshStreamingOutputCapacitySnapshotLocked(StreamingUDFState &state, const unique_lock<mutex> &guard) {
	state.VerifyLock(guard);
	const auto bytes = StreamingOutputByteCapacity(state);
	const auto item_bytes = bytes == 0 ? idx_t(0) : StreamingOutputItemByteCapacity(state);
	const auto rows = bytes == 0 ? idx_t(0) : StreamingOutputEventCapacity(state);
	state.output_capacity_bytes_snapshot.store(bytes, std::memory_order_relaxed);
	state.output_capacity_item_bytes_snapshot.store(item_bytes, std::memory_order_relaxed);
	state.output_capacity_rows_snapshot.store(rows, std::memory_order_relaxed);
}

struct StreamingOutputHandoffLease {
	StreamingOutputHandoffLease(shared_ptr<StreamingHandoffCounters> counters_p, idx_t rows_p, idx_t bytes_p,
	                            std::function<void()> release_output_lease_p)
	    : counters(std::move(counters_p)), rows(rows_p), bytes(bytes_p),
	      release_output_lease(std::move(release_output_lease_p)) {
		auto current_outputs = counters->outputs.fetch_add(1, std::memory_order_relaxed) + 1;
		auto current_rows = counters->rows.fetch_add(rows, std::memory_order_relaxed) + rows;
		auto current_bytes = counters->bytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
		AtomicMaxStreamingCounter(counters->max_outputs, current_outputs);
		AtomicMaxStreamingCounter(counters->max_rows, current_rows);
		AtomicMaxStreamingCounter(counters->max_bytes, current_bytes);
	}

	~StreamingOutputHandoffLease() {
		// Output-lease callbacks are idempotent and enqueue their Python work on
		// the dispatcher. This destructor can run under an unrelated downstream
		// operator lock, so handoff metrics are deliberately lock-free.
		try {
			ReleaseUDFOutputLease(release_output_lease);
		} catch (...) {
		}
		counters->outputs.fetch_sub(1, std::memory_order_relaxed);
		counters->rows.fetch_sub(rows, std::memory_order_relaxed);
		counters->bytes.fetch_sub(bytes, std::memory_order_relaxed);
	}

	shared_ptr<StreamingHandoffCounters> counters;
	idx_t rows;
	idx_t bytes;
	std::function<void()> release_output_lease;
};

static void StartStreamingOutputHandoffLocked(StreamingUDFState &state, StreamingReadyOutput &ready,
                                              const unique_lock<mutex> &guard) {
	state.VerifyLock(guard);
	if (ready.batch.kind != ExecutionBatchKind::LAZY_DATA_CHUNK || !ready.batch.lazy ||
	    ready.batch.lazy->blocks.empty()) {
		throw InternalException("streaming UDF output handoff requires a non-empty lazy block bundle");
	}
	// This callback only transfers the producer-side liveness credit.  The
	// release callback is moved into the descriptor token below and continues
	// to own/account the physical ObjectRef until the final downstream copy is
	// destroyed.
	if (ready.handoff_output_lease) {
		auto handoff = std::move(ready.handoff_output_lease);
		ready.handoff_output_lease = nullptr;
		handoff();
	}
	const auto rows = UDFExecutionBatchSize(ready.batch);
	auto token = make_shared_ptr<StreamingOutputHandoffLease>(state.handoff_counters, rows, ready.bytes,
	                                                          std::move(ready.release_output_lease));
	for (auto &block : ready.batch.lazy->blocks) {
		block.ownership_tokens.push_back(token);
	}
}

static StreamingReadyOutput TakeStreamingReadyOutputLocked(StreamingUDFState &state, unique_lock<mutex> &guard) {
	state.VerifyLock(guard);
	if (state.ready_outputs.empty()) {
		throw InternalException("streaming UDF ready output queue is empty");
	}
	auto ready = std::move(state.ready_outputs.front());
	state.ready_outputs.pop_front();
	const auto total_rows = UDFExecutionBatchSize(ready.batch);
	state.ready_rows = state.ready_rows >= total_rows ? state.ready_rows - total_rows : 0;
	state.ready_bytes = state.ready_bytes >= ready.bytes ? state.ready_bytes - ready.bytes : 0;
	if (!ready.lease_handed_off) {
		StartStreamingOutputHandoffLocked(state, ready, guard);
		ready.lease_handed_off = true;
	}
	if (total_rows <= STANDARD_VECTOR_SIZE) {
		UpdateStreamingOutputBytesLocked(state, guard);
		return ready;
	}
	if (ready.batch.kind != ExecutionBatchKind::LAZY_DATA_CHUNK || !ready.batch.lazy) {
		throw InternalException("oversized streaming UDF output must be a lazy data chunk");
	}

	auto head = SliceLazyDataChunk(*ready.batch.lazy, 0, STANDARD_VECTOR_SIZE);
	auto tail = SliceLazyDataChunk(*ready.batch.lazy, STANDARD_VECTOR_SIZE, total_rows - STANDARD_VECTOR_SIZE);
	StreamingReadyOutput remainder;
	remainder.batch.kind = ExecutionBatchKind::LAZY_DATA_CHUNK;
	remainder.batch.rows = tail->cardinality;
	remainder.batch.estimated_bytes = tail->EstimatedBytes();
	remainder.batch.lazy = std::move(tail);
	remainder.bytes = remainder.batch.estimated_bytes;
	remainder.submit_id = ready.submit_id;
	remainder.lease_handed_off = true;
	state.ready_rows += remainder.batch.rows;
	state.ready_bytes += remainder.bytes;
	state.ready_outputs.push_front(std::move(remainder));

	ready.batch.rows = head->cardinality;
	ready.batch.estimated_bytes = head->EstimatedBytes();
	ready.batch.lazy = std::move(head);
	ready.bytes = ready.batch.estimated_bytes;
	UpdateStreamingOutputBytesLocked(state, guard);
	return ready;
}

static void ThrowIfStreamingError(const StreamingUDFState &state) {
	if (state.has_error) {
		throw InvalidInputException("streaming UDF async error: %s", state.error);
	}
}

static bool StreamingTerminalReady(const StreamingUDFState &state) {
	const bool no_executor = !state.op || !state.op->executor;
	return state.sink_finished && state.pending_rows == 0 && state.inflight_batches.empty() &&
	       state.ready_outputs.empty() && state.ready_rows == 0 && state.deferred_outputs.empty() &&
	       state.deferred_output_rows == 0 && state.queued_output_events.load(std::memory_order_relaxed) == 0 &&
	       (state.dispatcher_finished || no_executor);
}

static bool StreamingHasReadyOutput(const StreamingUDFState &state) {
	return !state.ready_outputs.empty() || !state.deferred_outputs.empty();
}

static void ReleaseQueuedStreamingOutputsLocked(StreamingUDFState &state, unique_lock<mutex> &guard) {
	state.VerifyLock(guard);
	while (!state.ready_outputs.empty()) {
		auto ready = std::move(state.ready_outputs.front());
		state.ready_outputs.pop_front();
		ReleaseUDFOutputLease(ready.release_output_lease);
	}
	while (!state.deferred_outputs.empty()) {
		auto ready = std::move(state.deferred_outputs.front());
		state.deferred_outputs.pop_front();
		ReleaseUDFOutputLease(ready.release_output_lease);
	}
	state.ready_rows = 0;
	state.ready_bytes = 0;
	state.deferred_output_rows = 0;
	state.deferred_output_bytes = 0;
	UpdateStreamingOutputBytesLocked(state, guard);
}

static void ReleaseQueuedStreamingEvents(StreamingUDFState &state) {
	std::deque<UDFOutputEvent> events;
	{
		lock_guard<mutex> event_guard(state.output_event_lock);
		events.swap(state.pending_output_events);
		state.queued_output_events.store(0, std::memory_order_relaxed);
	}
	for (auto &event : events) {
		ReleaseUDFOutputLease(event.release_output_lease);
	}
}

static void AbortStreamingInflightSubmitsLocked(StreamingUDFState &state, const unique_lock<mutex> &guard) {
	state.VerifyLock(guard);
	if (state.op) {
		state.op->completed_batches = state.op->submitted_batches;
	}
	state.inflight_batches.clear();
	state.inflight_rows = 0;
	state.inflight_bytes = 0;
}

static void SetStreamingErrorLocked(StreamingUDFState &state, unique_lock<mutex> &guard, const string &msg) {
	state.VerifyLock(guard);
	if (!state.has_error) {
		AbortStreamingInflightSubmitsLocked(state, guard);
		ReleaseQueuedStreamingOutputsLocked(state, guard);
		ReleaseQueuedStreamingEvents(state);
		state.has_error = true;
		state.error = msg;
		RefreshStreamingOutputCapacitySnapshotLocked(state, guard);
		PreventStreamingBlocking(state, guard);
	}
	WakeAllStreamingSources(state, guard);
	WakeStreamingUDFTasks(state, guard);
}

static void SetStreamingError(StreamingUDFState &state, const string &msg) {
	auto guard = state.Lock();
	SetStreamingErrorLocked(state, guard, msg);
}

static void NotifyStreamingDispatcherFinished(StreamingUDFState &state) {
	auto guard = state.Lock();
	state.dispatcher_finished = true;
	state.dispatcher_finished_notifications.fetch_add(1);
	if (StreamingTerminalReady(state)) {
		PreventStreamingBlocking(state, guard);
		WakeAllStreamingSources(state, guard);
		WakeStreamingUDFTasks(state, guard);
	}
}

static bool StreamingInputWouldBlock(const StreamingUDFState &state, idx_t incoming_bytes = 0) {
	if (StreamingOutputBackpressured(state)) {
		return true;
	}
	const auto queued_bytes = state.pending_bytes + state.reserved_bytes;
	if (state.pending_rows + state.reserved_rows == 0 && queued_bytes == 0) {
		return false;
	}
	if (incoming_bytes == 0) {
		return queued_bytes >= state.config.task_input_max_bytes;
	}
	return SaturatingAdd(queued_bytes, incoming_bytes) > state.config.task_input_max_bytes;
}

static bool StreamingLazyInputWouldBlockBeforeAccept(const StreamingUDFState &state, idx_t incoming_bytes) {
	if (StreamingOutputBackpressured(state)) {
		return true;
	}
	const auto queued_bytes = state.pending_bytes + state.reserved_bytes;
	if (state.pending_rows + state.reserved_rows == 0 && queued_bytes == 0) {
		// A single descriptor must enter so it can be sliced by the submit byte
		// limit; otherwise an oversized descriptor can never make progress.
		return false;
	}
	if (incoming_bytes == 0) {
		return queued_bytes >= state.config.task_input_max_bytes;
	}
	// Lazy descriptors remain covered by their upstream output leases. Accept
	// one whole descriptor while the existing queue is below the derived byte
	// window, even if that descriptor crosses it, so compute batches can span
	// upstream block boundaries. The next descriptor blocks until an aligned
	// submit drains the overshoot.
	if (queued_bytes >= state.config.task_input_max_bytes) {
		return true;
	}
	return false;
}

static idx_t RowsWithinByteLimit(idx_t candidate_rows, idx_t candidate_bytes, idx_t remaining_bytes,
                                 bool allow_oversized_single_row) {
	if (candidate_rows == 0) {
		return 0;
	}
	if (candidate_bytes <= remaining_bytes) {
		return candidate_rows;
	}
	const auto bytes_per_row =
	    MaxValue<idx_t>(idx_t(1), candidate_bytes / candidate_rows + (candidate_bytes % candidate_rows == 0 ? 0 : 1));
	auto rows = remaining_bytes / bytes_per_row;
	if (rows == 0 && allow_oversized_single_row) {
		rows = 1;
	}
	return MinValue<idx_t>(candidate_rows, rows);
}

static bool StreamingHasLazyInput(const StreamingUDFState &state) {
	return state.planned_lazy_submit.bundle || !state.pending_lazy_inputs.empty();
}

static StreamingSubmitPlan StreamingIncompleteSubmitPlan(const StreamingUDFState &state, bool flush_tail) {
	StreamingSubmitPlan plan;
	if (state.pending_rows == 0) {
		return plan;
	}
	const bool byte_forced = state.pending_bytes >= state.config.task_input_max_bytes;
	if (!flush_tail && !byte_forced) {
		return plan;
	}
	plan.target_rows = state.pending_rows;
	plan.allow_incomplete_compute_batch = true;
	return plan;
}

// A deque entry is one upstream work unit. Preserve complete upstream blocks
// while coalescing undersized blocks to the configured soft minimum; without
// one, submit a compute-batch-aligned prefix.
static StreamingSubmitPlan PlanStreamingLazySubmit(const StreamingUDFState &state, bool flush_tail) {
	StreamingSubmitPlan plan;
	if (state.pending_rows == 0 || state.pending_lazy_inputs.empty()) {
		return plan;
	}
	idx_t rows_through_boundary = 0;
	for (const auto &entry : state.pending_lazy_inputs) {
		if (!entry.bundle || entry.rows == 0) {
			continue;
		}
		if (state.config.min_task_batch_rows > 0) {
			for (const auto &block : entry.bundle->blocks) {
				const auto block_rows = block.RowCount();
				if (block_rows == 0) {
					continue;
				}
				rows_through_boundary = SaturatingAdd(rows_through_boundary, block_rows);
				if (rows_through_boundary >= state.config.min_task_batch_rows) {
					plan.target_rows = rows_through_boundary;
					plan.allow_incomplete_compute_batch = true;
					return plan;
				}
			}
			continue;
		}
		rows_through_boundary = SaturatingAdd(rows_through_boundary, entry.rows);
		if (state.config.compute_batch_rows == 0) {
			plan.target_rows = rows_through_boundary;
			return plan;
		}
		const auto aligned_rows = rows_through_boundary - (rows_through_boundary % state.config.compute_batch_rows);
		if (aligned_rows > 0) {
			plan.target_rows = aligned_rows;
			return plan;
		}
	}
	return StreamingIncompleteSubmitPlan(state, flush_tail);
}

static StreamingSubmitPlan PlanStreamingMaterializedSubmit(const StreamingUDFState &state, bool flush_tail) {
	StreamingSubmitPlan plan;
	if (state.pending_rows == 0 || state.pending_inputs.empty()) {
		return plan;
	}
	idx_t rows_through_boundary = 0;
	for (const auto &entry : state.pending_inputs) {
		if (!entry.chunk || entry.row_offset >= entry.chunk->size()) {
			continue;
		}
		rows_through_boundary = SaturatingAdd(rows_through_boundary, entry.chunk->size() - entry.row_offset);
		if (state.config.min_task_batch_rows > 0) {
			if (rows_through_boundary >= state.config.min_task_batch_rows) {
				plan.target_rows = rows_through_boundary;
				plan.allow_incomplete_compute_batch = true;
				return plan;
			}
			continue;
		}
		if (state.config.compute_batch_rows == 0) {
			plan.target_rows = rows_through_boundary;
			return plan;
		}
		const auto aligned_rows = rows_through_boundary - (rows_through_boundary % state.config.compute_batch_rows);
		if (aligned_rows > 0) {
			plan.target_rows = aligned_rows;
			return plan;
		}
	}
	return StreamingIncompleteSubmitPlan(state, flush_tail);
}

static void ValidateStreamingLazyBatchSchema(const LazyRefDataChunk &expected, const LazyRefDataChunk &actual) {
	if (expected.logical_types != actual.logical_types || expected.names != actual.names ||
	    expected.wrap_columns_as_struct != actual.wrap_columns_as_struct) {
		throw InvalidInputException("streaming UDF lazy batch received incompatible schemas");
	}
}

static StreamingPendingLazyInput TakeStreamingLazyInputBatch(StreamingUDFState &state, const StreamingSubmitPlan &plan,
                                                             idx_t compute_batch_rows, idx_t max_bytes) {
	StreamingPendingLazyInput submit;
	if (state.pending_lazy_inputs.empty() || !plan) {
		return submit;
	}
	const bool byte_forced_partial = max_bytes > 0 && state.pending_bytes >= max_bytes;
	const auto desired_rows = MinValue<idx_t>(plan.target_rows, state.pending_rows);
	if (desired_rows == 0) {
		return submit;
	}

	auto output = make_uniq<LazyRefDataChunk>();
	idx_t remaining = desired_rows;
	idx_t accumulated_bytes = 0;
	bool has_schema = false;
	while (remaining > 0 && !state.pending_lazy_inputs.empty()) {
		auto &front = state.pending_lazy_inputs.front();
		if (!front.bundle) {
			state.pending_lazy_inputs.pop_front();
			continue;
		}
		front.bundle->RecomputeCardinality();
		front.rows = front.bundle->cardinality;
		front.bytes = front.bundle->EstimatedBytes();
		if (front.bundle->Empty()) {
			state.pending_lazy_inputs.pop_front();
			continue;
		}
		if (max_bytes > 0 && front.bytes == 0) {
			throw InvalidInputException("streaming UDF lazy batch is missing byte metadata");
		}

		idx_t take_rows = MinValue<idx_t>(remaining, front.rows);
		auto byte_limited_rows = take_rows;
		if (max_bytes > 0) {
			const auto remaining_bytes = accumulated_bytes >= max_bytes ? idx_t(0) : max_bytes - accumulated_bytes;
			const auto candidate_bytes =
			    front.rows > 0 ? MaxValue<idx_t>(idx_t(1), SaturatingMultiply(front.bytes, take_rows) / front.rows)
			                   : front.bytes;
			byte_limited_rows =
			    RowsWithinByteLimit(take_rows, candidate_bytes, remaining_bytes, output->blocks.empty());
			if (byte_limited_rows == 0) {
				break;
			}
		}
		take_rows = byte_limited_rows;
		auto piece = SliceLazyDataChunk(*front.bundle, 0, take_rows);
		piece->RecomputeCardinality();
		if (piece->cardinality != take_rows) {
			throw InternalException("streaming UDF lazy batch split emitted row count mismatch");
		}
		const auto piece_bytes = piece->EstimatedBytes();
		if (!has_schema) {
			output->logical_types = piece->logical_types;
			output->names = piece->names;
			output->wrap_columns_as_struct = piece->wrap_columns_as_struct;
			has_schema = true;
		} else {
			ValidateStreamingLazyBatchSchema(*output, *piece);
		}
		for (auto &block : piece->blocks) {
			output->blocks.push_back(std::move(block));
		}
		accumulated_bytes += piece_bytes;

		remaining -= take_rows;
		if (take_rows == front.rows) {
			state.pending_lazy_inputs.pop_front();
		} else {
			const auto tail_rows = front.rows - take_rows;
			auto tail = SliceLazyDataChunk(*front.bundle, take_rows, tail_rows);
			tail->RecomputeCardinality();
			if (tail->cardinality != tail_rows) {
				throw InternalException("streaming UDF lazy tail split emitted row count mismatch");
			}
			front.bundle = std::move(tail);
			front.rows = front.bundle->cardinality;
			front.bytes = front.bundle->EstimatedBytes();
		}
	}

	output->RecomputeCardinality();
	if (output->Empty()) {
		return submit;
	}
	if (output->cardinality > desired_rows) {
		throw InternalException("streaming UDF lazy batch emitted row count mismatch");
	}
	if (!plan.allow_incomplete_compute_batch && compute_batch_rows > 0) {
		const auto aligned_rows = output->cardinality - (output->cardinality % compute_batch_rows);
		if (aligned_rows < output->cardinality && !(aligned_rows == 0 && byte_forced_partial)) {
			unique_ptr<LazyRefDataChunk> head;
			unique_ptr<LazyRefDataChunk> tail;
			if (aligned_rows > 0) {
				head = SliceLazyDataChunk(*output, 0, aligned_rows);
				tail = SliceLazyDataChunk(*output, aligned_rows, output->cardinality - aligned_rows);
			} else {
				tail = std::move(output);
			}
			StreamingPendingLazyInput restored;
			restored.rows = tail->cardinality;
			restored.bytes = tail->EstimatedBytes();
			restored.bundle = std::move(tail);
			state.pending_lazy_inputs.push_front(std::move(restored));
			output = std::move(head);
		}
		if (!output || output->Empty()) {
			return submit;
		}
	}
	submit.rows = output->cardinality;
	submit.bytes = output->EstimatedBytes();
	submit.bundle = std::move(output);
	return submit;
}

static void PushStreamingReadyOutputLocked(StreamingUDFState &state, StreamingReadyOutput &&ready,
                                           unique_lock<mutex> &guard) {
	state.VerifyLock(guard);
	const bool was_empty = state.ready_outputs.empty();
	const auto ready_rows = UDFExecutionBatchSize(ready.batch);
	const auto ready_bytes = ready.bytes;
	const auto submit_id = ready.submit_id;
	state.ready_outputs.push_back(std::move(ready));
	state.ready_rows += ready_rows;
	state.ready_bytes += ready_bytes;
	state.max_ready_rows = MaxValue<idx_t>(state.max_ready_rows, state.ready_rows);
	UpdateStreamingOutputBytesLocked(state, guard);
	if (was_empty) {
		state.ready_empty_to_nonempty.fetch_add(1);
		WakeOneStreamingSource(state, guard);
	}
}

static void FlushDeferredStreamingOutputsLocked(StreamingUDFState &state, unique_lock<mutex> &guard) {
	state.VerifyLock(guard);
	while (!state.deferred_outputs.empty() && !StreamingReadyFull(state)) {
		auto ready = std::move(state.deferred_outputs.front());
		state.deferred_outputs.pop_front();
		const auto ready_rows = UDFExecutionBatchSize(ready.batch);
		state.deferred_output_rows =
		    state.deferred_output_rows >= ready_rows ? state.deferred_output_rows - ready_rows : 0;
		state.deferred_output_bytes =
		    state.deferred_output_bytes >= ready.bytes ? state.deferred_output_bytes - ready.bytes : 0;
		PushStreamingReadyOutputLocked(state, std::move(ready), guard);
	}
}

static bool EnqueueStreamingDataEventLocked(StreamingUDFState &state, UDFOutputEvent &&event, unique_lock<mutex> &guard,
                                            std::function<void()> &release_after_enqueue) {
	state.VerifyLock(guard);
	if (event.kind != UDFOutputEventKind::DATA) {
		ReleaseUDFOutputLease(event.release_output_lease);
		SetStreamingErrorLocked(state, guard, "streaming UDF internal error: non-DATA event in data enqueue");
		return false;
	}
	if (!event.ref_outputs) {
		ReleaseUDFOutputLease(event.release_output_lease);
		SetStreamingErrorLocked(state, guard, "streaming UDF DATA event received null lazy/ref-bundle output");
		return false;
	}
	if (state.has_error || state.source_finished) {
		ReleaseUDFOutputLease(event.release_output_lease);
		return false;
	}

	if (state.op->is_flat_map) {
		PreserveFlatMapLazyOutputShape(*state.op, *event.ref_outputs);
	}
	event.ref_outputs->RecomputeCardinality();
	const auto lazy_rows = event.ref_outputs->cardinality;
	const auto lazy_bytes = event.ref_outputs->EstimatedBytes();
	auto entry = state.inflight_batches.find(event.submit_id);
	if (entry == state.inflight_batches.end()) {
		ReleaseUDFOutputLease(event.release_output_lease);
		SetStreamingErrorLocked(state, guard,
		                        StringUtil::Format("streaming UDF DATA event received unknown submit_id %llu",
		                                           static_cast<unsigned long long>(event.submit_id)));
		return false;
	}
	auto &inflight_ref = entry->second;
	const bool row_preserving_output = UDFRequiresRowPreservingOutput(*state.op);
	if (row_preserving_output && inflight_ref.emitted_rows + lazy_rows > inflight_ref.total_rows) {
		ReleaseUDFOutputLease(event.release_output_lease);
		SetStreamingErrorLocked(
		    state, guard,
		    StringUtil::Format("streaming UDF partial lazy output rows (%d) exceed input rows (%d) for "
		                       "submit_id %llu",
		                       inflight_ref.emitted_rows + lazy_rows, inflight_ref.total_rows,
		                       static_cast<unsigned long long>(event.submit_id)));
		return false;
	}
	inflight_ref.emitted_rows += lazy_rows;

	if (lazy_rows > 0) {
		AtomicAddStreamingCounter(state.produced_output_rows, lazy_rows);
		AtomicAddStreamingCounter(state.produced_output_bytes, lazy_bytes);
		auto release_output_lease = std::move(event.release_output_lease);
		StreamingReadyOutput ready;
		ready.batch.kind = ExecutionBatchKind::LAZY_DATA_CHUNK;
		ready.batch.rows = lazy_rows;
		ready.batch.estimated_bytes = lazy_bytes;
		ready.batch.lazy = std::move(event.ref_outputs);
		ready.bytes = lazy_bytes;
		ready.submit_id = event.submit_id;
		ready.handoff_output_lease = std::move(event.handoff_output_lease);
		ready.release_output_lease = std::move(release_output_lease);
		FlushDeferredStreamingOutputsLocked(state, guard);
		if (StreamingReadyFull(state) || !state.deferred_outputs.empty()) {
			state.deferred_outputs.push_back(std::move(ready));
			state.deferred_output_rows += lazy_rows;
			state.deferred_output_bytes += lazy_bytes;
			state.deferred_output_events.fetch_add(1);
			UpdateStreamingOutputBytesLocked(state, guard);
		} else {
			PushStreamingReadyOutputLocked(state, std::move(ready), guard);
		}
	} else {
		release_after_enqueue = std::move(event.release_output_lease);
	}
	StreamingUDFDebugState(state, "accept_data");
	if (StreamingTerminalReady(state)) {
		PreventStreamingBlocking(state, guard);
		WakeAllStreamingSources(state, guard);
	}
	WakeStreamingUDFTasks(state, guard);
	return true;
}

static bool CompleteStreamingSubmitLocked(StreamingUDFState &state, unique_lock<mutex> &guard, idx_t submit_id,
                                          const char *event_name) {
	state.VerifyLock(guard);
	auto entry = state.inflight_batches.find(submit_id);
	if (entry == state.inflight_batches.end()) {
		if (state.has_error || state.source_finished) {
			return false;
		}
		SetStreamingErrorLocked(state, guard,
		                        StringUtil::Format("streaming UDF %s event received unknown submit_id %llu", event_name,
		                                           static_cast<unsigned long long>(submit_id)));
		return false;
	}
	auto inflight_ref = entry->second;
	state.inflight_rows =
	    state.inflight_rows >= inflight_ref.total_rows ? state.inflight_rows - inflight_ref.total_rows : 0;
	state.inflight_bytes = state.inflight_bytes >= inflight_ref.bytes ? state.inflight_bytes - inflight_ref.bytes : 0;
	state.inflight_batches.erase(entry);
	if (state.has_error || state.source_finished) {
		return false;
	}
	const bool row_preserving_output = UDFRequiresRowPreservingOutput(*state.op);
	if (row_preserving_output && inflight_ref.emitted_rows != inflight_ref.total_rows) {
		SetStreamingErrorLocked(
		    state, guard,
		    StringUtil::Format("streaming UDF lazy output count (%d) does not match input rows (%d) for submit_id %llu",
		                       inflight_ref.emitted_rows, inflight_ref.total_rows,
		                       static_cast<unsigned long long>(submit_id)));
		return false;
	}
	AtomicAddStreamingCounter(state.completed_input_rows, inflight_ref.total_rows);
	AtomicAddStreamingCounter(state.completed_input_bytes, inflight_ref.bytes);
	state.completed_batches.fetch_add(1);
	state.op->completed_batches++;
	StreamingUDFDebugState(state, event_name, true);
	if (state.pending_rows > 0 && !StreamingOutputBackpressured(state)) {
		// COMPLETE can be the only signal that frees submit capacity. If
		// sources are blocked with no ready output, wake one to submit the
		// pending rows/lazy bundles.
		WakeOneStreamingSource(state, guard);
	}
	return true;
}

static bool TrySubmitStreamingLazyInput(ExecutionContext &context, StreamingUDFState &state,
                                        const unique_lock<mutex> &guard, bool flush_tail) {
	state.VerifyLock(guard);
	const bool retrying_planned_submit = state.planned_lazy_submit.bundle != nullptr;
	if (!retrying_planned_submit && state.pending_lazy_inputs.empty()) {
		return false;
	}
	EnsureStreamingExecutor(context, state, guard);
	if (!state.op || !state.op->executor || !state.op->executor->SupportsRefBundleInput()) {
		throw InvalidInputException("streaming UDF executor does not support ref-bundle input");
	}
	StreamingPendingLazyInput candidate;
	StreamingPendingLazyInput *pending = nullptr;
	if (retrying_planned_submit) {
		pending = &state.planned_lazy_submit;
	} else {
		auto plan = PlanStreamingLazySubmit(state, flush_tail);
		if (!plan) {
			return false;
		}
		candidate = TakeStreamingLazyInputBatch(state, plan, state.config.compute_batch_rows,
		                                        state.config.task_input_max_bytes);
		pending = &candidate;
	}
	if (!pending->bundle || pending->rows == 0) {
		return false;
	}
	idx_t submit_id = 0;
	// Lazy refs remain charged to their upstream output leases through handoff;
	// charging the same physical objects as retained task input would double count.
	if (!TrySubmitRefBundleRaw(context, *state.op, *pending->bundle, submit_id, 0)) {
		if (!retrying_planned_submit) {
			// Preserve the exact ref bundle associated with the admission waiter.
			state.planned_lazy_submit = std::move(candidate);
		}
		return false;
	}
	const auto submitted_rows = pending->rows;
	const auto submitted_bytes = pending->bytes;
	if (retrying_planned_submit) {
		state.planned_lazy_submit = StreamingPendingLazyInput();
	}
	StreamingInflightBatch inflight;
	inflight.submit_id = submit_id;
	inflight.total_rows = submitted_rows;
	inflight.bytes = submitted_bytes;
	auto inserted = state.inflight_batches.emplace(submit_id, inflight);
	if (!inserted.second) {
		throw InternalException("streaming UDF duplicate lazy submit_id %llu",
		                        static_cast<unsigned long long>(submit_id));
	}
	AtomicAddStreamingCounter(state.submitted_input_rows, submitted_rows);
	AtomicAddStreamingCounter(state.submitted_input_bytes, submitted_bytes);
	state.pending_rows = state.pending_rows >= submitted_rows ? state.pending_rows - submitted_rows : 0;
	state.pending_bytes = state.pending_bytes >= submitted_bytes ? state.pending_bytes - submitted_bytes : 0;
	state.inflight_rows += submitted_rows;
	state.inflight_bytes += submitted_bytes;
	state.max_active_batches =
	    MaxValue<idx_t>(state.max_active_batches, static_cast<idx_t>(state.inflight_batches.size()));
	state.max_outstanding_rows = MaxValue<idx_t>(state.max_outstanding_rows, state.inflight_rows);
	state.submitted_batches.fetch_add(1);
	StreamingUDFDebugState(state, "submit_lazy", true);
	return true;
}

static void DriveStreamingLazySubmits(ExecutionContext &context, StreamingUDFState &state,
                                      const unique_lock<mutex> &guard, bool flush_tail) {
	state.VerifyLock(guard);
	ThrowIfStreamingError(state);
	while (!StreamingOutputBackpressured(state)) {
		if (!TrySubmitStreamingLazyInput(context, state, guard, flush_tail)) {
			break;
		}
	}
}

static unique_ptr<DataChunk> BuildStreamingBatchFromPending(ExecutionContext &context, StreamingUDFState &state,
                                                            idx_t desired_rows, idx_t max_bytes,
                                                            bool allow_oversized_single_row) {
	if (state.pending_rows == 0 || state.pending_inputs.empty()) {
		return nullptr;
	}
	desired_rows = MinValue<idx_t>(desired_rows, state.pending_rows);
	if (desired_rows == 0) {
		return nullptr;
	}
	auto &first = state.pending_inputs.front();
	auto batch = make_uniq<DataChunk>();
	batch->Initialize(context.client, first.chunk->GetTypes(), desired_rows);

	idx_t remaining = desired_rows;
	idx_t accumulated_bytes = 0;
	while (remaining > 0 && !state.pending_inputs.empty()) {
		auto &entry = state.pending_inputs.front();
		auto &src = *entry.chunk;
		auto src_rows = src.size();
		auto offset = MinValue<idx_t>(entry.row_offset, src_rows);
		auto available_rows = src_rows - offset;
		if (available_rows == 0) {
			state.pending_inputs.pop_front();
			continue;
		}
		auto take_rows = MinValue<idx_t>(remaining, available_rows);
		if (max_bytes > 0) {
			const auto remaining_bytes = accumulated_bytes >= max_bytes ? idx_t(0) : max_bytes - accumulated_bytes;
			const auto candidate_bytes =
			    available_rows > 0
			        ? MaxValue<idx_t>(idx_t(1), SaturatingMultiply(entry.bytes, take_rows) / available_rows)
			        : entry.bytes;
			const auto byte_limited_rows = RowsWithinByteLimit(take_rows, candidate_bytes, remaining_bytes,
			                                                   allow_oversized_single_row && batch->size() == 0);
			if (byte_limited_rows == 0) {
				break;
			}
			take_rows = byte_limited_rows;
		}
		if (offset == 0 && take_rows == src_rows) {
			batch->Append(src, true);
		} else {
			SelectionVector take_sel(STANDARD_VECTOR_SIZE);
			for (idx_t i = 0; i < take_rows; i++) {
				take_sel.set_index(i, offset + i);
			}
			batch->Append(src, true, &take_sel, take_rows);
		}
		state.pending_rows -= take_rows;
		idx_t consumed_bytes = 0;
		if (take_rows == available_rows) {
			consumed_bytes = entry.bytes;
			state.pending_bytes = state.pending_bytes >= entry.bytes ? state.pending_bytes - entry.bytes : 0;
			state.pending_inputs.pop_front();
		} else {
			consumed_bytes =
			    available_rows > 0 ? SaturatingMultiply(entry.bytes, take_rows) / available_rows : entry.bytes;
			if (consumed_bytes == 0 && entry.bytes > 0) {
				consumed_bytes = 1;
			}
			consumed_bytes = MinValue<idx_t>(consumed_bytes, entry.bytes);
			entry.bytes -= consumed_bytes;
			entry.row_offset = offset + take_rows;
			state.pending_bytes = state.pending_bytes >= consumed_bytes ? state.pending_bytes - consumed_bytes : 0;
		}
		accumulated_bytes += consumed_bytes;
		remaining -= take_rows;
	}
	if (batch->size() == 0) {
		return nullptr;
	}
	return batch;
}

static void RestoreStreamingMaterializedEnvelopeToFront(StreamingUDFState &state,
                                                        StreamingPendingMaterializedEnvelope &&envelope) {
	for (idx_t idx = envelope.chunks.size(); idx > 0; idx--) {
		auto chunk = std::move(envelope.chunks[idx - 1]);
		if (!chunk || chunk->size() == 0) {
			continue;
		}
		StreamingPendingInputPiece piece;
		piece.bytes = EstimateStreamingChunkBytes(*chunk);
		piece.row_offset = 0;
		piece.chunk = std::move(chunk);
		state.pending_rows += piece.chunk->size();
		state.pending_bytes += piece.bytes;
		state.pending_inputs.push_front(std::move(piece));
	}
}

static StreamingPendingMaterializedEnvelope
TakeStreamingMaterializedEnvelope(ExecutionContext &context, StreamingUDFState &state, const StreamingSubmitPlan &plan,
                                  idx_t compute_batch_rows, idx_t max_bytes) {
	StreamingPendingMaterializedEnvelope envelope;
	if (state.pending_rows == 0 || state.pending_inputs.empty() || !plan) {
		return envelope;
	}

	idx_t max_rows = MinValue<idx_t>(plan.target_rows, state.pending_rows);
	if (max_rows == 0) {
		return envelope;
	}

	while (state.pending_rows > 0 && !state.pending_inputs.empty()) {
		if (envelope.rows >= max_rows) {
			break;
		}
		idx_t batch_max_bytes = 0;
		if (max_bytes > 0) {
			if (envelope.bytes >= max_bytes) {
				break;
			}
			batch_max_bytes = max_bytes - envelope.bytes;
		}
		auto remaining_rows = max_rows - envelope.rows;
		auto batch_rows = MinValue<idx_t>(STANDARD_VECTOR_SIZE, remaining_rows);
		auto batch = BuildStreamingBatchFromPending(context, state, batch_rows, batch_max_bytes, envelope.rows == 0);
		if (!batch) {
			break;
		}
		const auto rows = batch->size();
		const auto bytes = EstimateStreamingChunkBytes(*batch);
		if (rows == 0) {
			continue;
		}
		envelope.rows += rows;
		envelope.bytes += bytes;
		envelope.chunks.push_back(std::move(batch));
		if (max_bytes > 0 && envelope.bytes >= max_bytes) {
			break;
		}
	}
	if (!plan.allow_incomplete_compute_batch && compute_batch_rows > 0 && envelope.rows > 0) {
		const auto aligned_rows = envelope.rows - (envelope.rows % compute_batch_rows);
		if (aligned_rows < envelope.rows) {
			const bool byte_limited = max_bytes > 0 && envelope.rows < max_rows;
			if (aligned_rows == 0 && byte_limited) {
				// One complete compute batch cannot fit in the hard task-input
				// byte window. A partial task is the only progress-preserving split.
				return envelope;
			}
			if (aligned_rows == 0) {
				throw InternalException("streaming UDF work-unit plan produced an incomplete compute batch");
			}
			RestoreStreamingMaterializedEnvelopeToFront(state, std::move(envelope));
			StreamingSubmitPlan aligned_plan;
			aligned_plan.target_rows = aligned_rows;
			aligned_plan.allow_incomplete_compute_batch = true;
			return TakeStreamingMaterializedEnvelope(context, state, aligned_plan, compute_batch_rows, 0);
		}
	}
	return envelope;
}

static bool AcceptStreamingEventLocked(StreamingUDFState &state, UDFOutputEvent &&event, unique_lock<mutex> &guard,
                                       std::function<void()> &release_after_enqueue) {
	state.VerifyLock(guard);
	state.result_callbacks.fetch_add(1);
	if (event.kind == UDFOutputEventKind::ERROR) {
		SetStreamingErrorLocked(state, guard, event.error.empty() ? "streaming UDF async error" : event.error);
		return false;
	}
	if (event.kind == UDFOutputEventKind::FINISHED) {
		state.dispatcher_finished = true;
		state.dispatcher_finished_notifications.fetch_add(1);
		if (StreamingTerminalReady(state)) {
			PreventStreamingBlocking(state, guard);
			WakeAllStreamingSources(state, guard);
		}
		WakeStreamingUDFTasks(state, guard);
		return true;
	}
	if (event.submit_id == 0) {
		SetStreamingErrorLocked(state, guard, "streaming UDF output event received a result without submit_id");
		return false;
	}

	if (event.kind == UDFOutputEventKind::COMPLETE) {
		CompleteStreamingSubmitLocked(state, guard, event.submit_id, "COMPLETE");
		if (StreamingTerminalReady(state)) {
			PreventStreamingBlocking(state, guard);
			WakeAllStreamingSources(state, guard);
		} else if (StreamingHasReadyOutput(state)) {
			WakeAllStreamingSources(state, guard);
		}
		WakeStreamingUDFTasks(state, guard);
		return true;
	}

	if (event.kind != UDFOutputEventKind::DATA) {
		SetStreamingErrorLocked(state, guard, "streaming UDF received unknown output event kind");
		return false;
	}
	if (!event.ref_outputs) {
		SetStreamingErrorLocked(state, guard, "streaming UDF DATA event received null lazy/ref-bundle output");
		return false;
	}

	const auto submit_id = event.submit_id;
	const bool submit_complete = event.submit_complete;
	if (!EnqueueStreamingDataEventLocked(state, std::move(event), guard, release_after_enqueue)) {
		return false;
	}
	if (submit_complete) {
		CompleteStreamingSubmitLocked(state, guard, submit_id, "DATA completion");
		if (StreamingTerminalReady(state)) {
			PreventStreamingBlocking(state, guard);
			WakeAllStreamingSources(state, guard);
		} else if (StreamingHasReadyOutput(state)) {
			WakeAllStreamingSources(state, guard);
		}
		WakeStreamingUDFTasks(state, guard);
	}
	return true;
}

static void AcceptStreamingEvent(StreamingUDFState &state, UDFOutputEvent &&event) {
	auto guard = state.Lock();
	std::function<void()> release_after_enqueue;
	AcceptStreamingEventLocked(state, std::move(event), guard, release_after_enqueue);
	RefreshStreamingOutputCapacitySnapshotLocked(state, guard);
	guard.unlock();
	ReleaseUDFOutputLease(release_after_enqueue);
}

static bool DrainStreamingOutputEventsLocked(StreamingUDFState &state, unique_lock<mutex> &guard) {
	state.VerifyLock(guard);
	bool did_work = false;
	while (true) {
		UDFOutputEvent event;
		{
			lock_guard<mutex> event_guard(state.output_event_lock);
			if (state.pending_output_events.empty()) {
				break;
			}
			event = std::move(state.pending_output_events.front());
			state.pending_output_events.pop_front();
			auto queued = state.queued_output_events.load(std::memory_order_relaxed);
			state.queued_output_events.store(queued > 0 ? queued - 1 : 0, std::memory_order_relaxed);
		}
		std::function<void()> release_after_enqueue;
		AcceptStreamingEventLocked(state, std::move(event), guard, release_after_enqueue);
		RefreshStreamingOutputCapacitySnapshotLocked(state, guard);
		did_work = true;
		if (release_after_enqueue) {
			guard.unlock();
			ReleaseUDFOutputLease(release_after_enqueue);
			guard.lock();
		}
		if (state.has_error) {
			break;
		}
	}
	return did_work;
}

static void DriveStreamingSubmits(ExecutionContext &context, StreamingUDFState &state, const unique_lock<mutex> &guard,
                                  bool flush_tail) {
	state.VerifyLock(guard);
	ThrowIfStreamingError(state);
	DriveStreamingLazySubmits(context, state, guard, flush_tail);
	if (StreamingHasLazyInput(state)) {
		return;
	}
	while (state.pending_rows > 0 && !StreamingOutputBackpressured(state)) {
		EnsureStreamingExecutor(context, state, guard);
		const bool retrying_planned_submit = !state.planned_materialized_submit.chunks.empty();
		StreamingPendingMaterializedEnvelope candidate;
		StreamingPendingMaterializedEnvelope *envelope = nullptr;
		if (retrying_planned_submit) {
			envelope = &state.planned_materialized_submit;
		} else {
			auto plan = PlanStreamingMaterializedSubmit(state, flush_tail);
			if (!plan) {
				break;
			}
			candidate = TakeStreamingMaterializedEnvelope(context, state, plan, state.config.compute_batch_rows,
			                                              state.config.task_input_max_bytes);
			envelope = &candidate;
		}
		if (envelope->chunks.empty() || envelope->rows == 0) {
			break;
		}
		const auto rows = envelope->rows;
		const auto bytes = envelope->bytes;
		idx_t submit_id = 0;
		if (!TrySubmitMaterializedEnvelopeRaw(context, *state.op, envelope->chunks, bytes, submit_id)) {
			if (!retrying_planned_submit) {
				// The executor has registered admission for this exact byte
				// commitment. Keep the envelope intact instead of restoring and
				// repartitioning it while the waiter sleeps.
				state.pending_rows += rows;
				state.pending_bytes += bytes;
				state.planned_materialized_submit = std::move(candidate);
			}
			break;
		}
		if (retrying_planned_submit) {
			state.pending_rows = state.pending_rows >= rows ? state.pending_rows - rows : 0;
			state.pending_bytes = state.pending_bytes >= bytes ? state.pending_bytes - bytes : 0;
			state.planned_materialized_submit = StreamingPendingMaterializedEnvelope();
		}
		StreamingInflightBatch inflight;
		inflight.submit_id = submit_id;
		inflight.total_rows = rows;
		inflight.bytes = bytes;
		auto inserted = state.inflight_batches.emplace(submit_id, inflight);
		if (!inserted.second) {
			throw InternalException("streaming UDF duplicate submit_id %llu",
			                        static_cast<unsigned long long>(submit_id));
		}
		AtomicAddStreamingCounter(state.submitted_input_rows, rows);
		AtomicAddStreamingCounter(state.submitted_input_bytes, bytes);
		state.inflight_rows += rows;
		state.inflight_bytes += bytes;
		state.max_active_batches =
		    MaxValue<idx_t>(state.max_active_batches, static_cast<idx_t>(state.inflight_batches.size()));
		state.max_outstanding_rows = MaxValue<idx_t>(state.max_outstanding_rows, state.inflight_rows);
		state.submitted_batches.fetch_add(1);
		StreamingUDFDebugState(state, "submit_materialized_envelope", true);
	}
}

static void DrainStreamingPendingBeforeInputBlock(ExecutionContext &context, StreamingUDFState &state,
                                                  const unique_lock<mutex> &guard, idx_t incoming_bytes = 0) {
	state.VerifyLock(guard);
	if (!StreamingInputWouldBlock(state, incoming_bytes)) {
		return;
	}
	if (state.pending_rows == 0 || StreamingOutputBackpressured(state)) {
		return;
	}
	const auto pending_before = state.pending_rows;
	DriveStreamingSubmits(context, state, guard, true);
	if (state.pending_rows < pending_before) {
		StreamingUDFDebugState(state, "drain_partial_before_input_block", true);
	}
}

static void DrainStreamingLazyPendingBeforeInputBlock(ExecutionContext &context, StreamingUDFState &state,
                                                      const unique_lock<mutex> &guard, idx_t incoming_bytes) {
	state.VerifyLock(guard);
	if (!StreamingLazyInputWouldBlockBeforeAccept(state, incoming_bytes)) {
		return;
	}
	if (state.pending_rows == 0 || StreamingOutputBackpressured(state)) {
		return;
	}
	const auto pending_before = state.pending_rows;
	DriveStreamingSubmits(context, state, guard, true);
	if (state.pending_rows < pending_before) {
		StreamingUDFDebugState(state, "drain_lazy_before_input_block", true);
	}
}

static void FinishStreamingSubmissions(ClientContext &context, StreamingUDFState &state) {
	if (state.executor_finished_submitting || !state.op || !state.op->executor) {
		return;
	}
	state.executor_finished_submitting = true;
	state.op->finished_submitting = true;
	StreamingUDFDebugState(state, "executor_finished_submitting", true);
	state.op->executor->FinishedSubmitting(context);
}

static void DriveStreamingAfterSinkFinished(ExecutionContext &context, StreamingUDFState &state,
                                            unique_lock<mutex> &guard) {
	state.VerifyLock(guard);
	DrainStreamingOutputEventsLocked(state, guard);
	if (!state.sink_finished || state.source_finished || state.has_error) {
		return;
	}
	ThrowIfStreamingError(state);
	DriveStreamingSubmits(context, state, guard, true);
	ThrowIfStreamingError(state);
	if (state.pending_rows == 0) {
		FinishStreamingSubmissions(context.client, state);
	}
	if (StreamingTerminalReady(state)) {
		PreventStreamingBlocking(state, guard);
		WakeAllStreamingSources(state, guard);
	}
	WakeStreamingUDFTasks(state, guard);
}

PhysicalStreamingUDF::PhysicalStreamingUDF(PhysicalPlan &physical_plan, vector<LogicalType> types,
                                           TableFunction function_p, unique_ptr<FunctionData> bind_data_p,
                                           vector<ColumnIndex> column_ids_p, idx_t estimated_cardinality,
                                           vector<column_t> project_input_p)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::STREAMING_UDF, std::move(types), estimated_cardinality),
      function(std::move(function_p)), bind_data(std::move(bind_data_p)), column_ids(std::move(column_ids_p)),
      projected_input(std::move(project_input_p)) {
}

std::shared_ptr<StreamingUDFState> PhysicalStreamingUDF::GetStreamingState(ClientContext &) const {
	lock_guard<std::mutex> guard(streaming_state_lock);
	if (!streaming_state) {
		auto &udf_bind = bind_data->Cast<UDFFunctionData>();
		streaming_state = std::make_shared<StreamingUDFState>(udf_bind.payload, udf_bind.actor_handles);
		streaming_state->self = streaming_state;
	}
	return streaming_state;
}

void PhysicalStreamingUDF::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	sink_state.reset();
	if (children.size() != 1) {
		throw InternalException("PhysicalStreamingUDF requires exactly one child");
	}
	auto &state = meta_pipeline.GetState();
	state.SetPipelineSource(current, *this);
	auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this, MetaPipelineType::REGULAR, false);
	child_meta_pipeline.Build(children[0].get());
}

unique_ptr<GlobalSinkState> PhysicalStreamingUDF::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<StreamingUDFGlobalSinkState>(GetStreamingState(context));
}

unique_ptr<LocalSinkState> PhysicalStreamingUDF::GetLocalSinkState(ExecutionContext &) const {
	return make_uniq<StreamingUDFLocalSinkState>();
}

ProgressData PhysicalStreamingUDF::GetSinkProgress(ClientContext &, GlobalSinkState &gstate_p,
                                                   const ProgressData source_progress) const {
	auto &gstate = gstate_p.Cast<StreamingUDFGlobalSinkState>();
	auto &state = *gstate.state;
	state.upstream_progress_valid.store(false, std::memory_order_release);
	if (source_progress.IsValid() && source_progress.total > 0.0) {
		state.upstream_progress_done.store(source_progress.done, std::memory_order_relaxed);
		state.upstream_progress_total.store(source_progress.total, std::memory_order_relaxed);
		state.upstream_progress_valid.store(true, std::memory_order_release);
	}
	return source_progress;
}

SinkResultType PhysicalStreamingUDF::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &lstate = input.local_state.Cast<StreamingUDFLocalSinkState>();
	if (chunk.size() == 0) {
		lstate.consumed_blocked_input = false;
		return SinkResultType::NEED_MORE_INPUT;
	}
	auto &gstate = input.global_state.Cast<StreamingUDFGlobalSinkState>();
	auto &state = *gstate.state;
	state.sink_calls.fetch_add(1);
	auto incoming_rows = chunk.size();
	auto incoming_bytes = EstimateStreamingChunkBytes(chunk);

	auto guard = state.Lock();
	DrainStreamingOutputEventsLocked(state, guard);
	ThrowIfStreamingError(state);
	DriveStreamingSubmits(context, state, guard, false);
	if (lstate.consumed_blocked_input) {
		DrainStreamingPendingBeforeInputBlock(context, state, guard, 0);
		if (StreamingInputWouldBlock(state, 0)) {
			state.blocked_sinks.fetch_add(1);
			StreamingUDFDebugState(state, "sink_reblock_materialized", true);
			return BlockStreamingSink(state, guard, input.interrupt_state);
		}
		lstate.consumed_blocked_input = false;
		return SinkResultType::NEED_MORE_INPUT;
	}
	DrainStreamingPendingBeforeInputBlock(context, state, guard, incoming_bytes);
	if (StreamingInputWouldBlock(state, incoming_bytes)) {
		state.blocked_sinks.fetch_add(1);
		StreamingUDFDebugState(state, "sink_block_materialized_before_accept", true);
		return BlockStreamingSink(state, guard, input.interrupt_state);
	}
	state.reserved_rows += incoming_rows;
	state.reserved_bytes += incoming_bytes;
	guard.unlock();

	auto buffered = make_uniq<DataChunk>();
	try {
		buffered->Initialize(context.client, chunk.GetTypes(), incoming_rows);
		chunk.Copy(*buffered, 0);
	} catch (...) {
		auto cleanup_guard = state.Lock();
		state.reserved_rows = state.reserved_rows >= incoming_rows ? state.reserved_rows - incoming_rows : 0;
		state.reserved_bytes = state.reserved_bytes >= incoming_bytes ? state.reserved_bytes - incoming_bytes : 0;
		WakeStreamingUDFTasks(state, cleanup_guard);
		throw;
	}
	auto buffered_rows = buffered->size();
	auto buffered_bytes = EstimateStreamingChunkBytes(*buffered);

	guard = state.Lock();
	DrainStreamingOutputEventsLocked(state, guard);
	state.reserved_rows = state.reserved_rows >= incoming_rows ? state.reserved_rows - incoming_rows : 0;
	state.reserved_bytes = state.reserved_bytes >= incoming_bytes ? state.reserved_bytes - incoming_bytes : 0;

	StreamingPendingInputPiece piece;
	piece.bytes = buffered_bytes;
	piece.chunk = std::move(buffered);
	state.pending_rows += buffered_rows;
	state.pending_bytes += buffered_bytes;
	state.pending_inputs.push_back(std::move(piece));
	state.max_pending_rows = MaxValue<idx_t>(state.max_pending_rows, state.pending_rows + state.reserved_rows);
	AtomicAddStreamingCounter(state.accepted_input_rows, buffered_rows);
	AtomicAddStreamingCounter(state.accepted_input_bytes, buffered_bytes);
	ThrowIfStreamingError(state);
	DriveStreamingSubmits(context, state, guard, false);
	DrainStreamingOutputEventsLocked(state, guard);
	if (StreamingInputWouldBlock(state, 0)) {
		lstate.consumed_blocked_input = true;
		state.blocked_sinks.fetch_add(1);
		StreamingUDFDebugState(state, "sink_block_materialized", true);
		return BlockStreamingSink(state, guard, input.interrupt_state);
	}
	return SinkResultType::NEED_MORE_INPUT;
}

SinkResultType PhysicalStreamingUDF::SinkBatch(ExecutionContext &context, ExecutionBatch &batch,
                                               OperatorSinkInput &input) const {
	if (batch.kind == ExecutionBatchKind::MATERIALIZED_CHUNK) {
		if (!batch.materialized || batch.materialized->size() == 0) {
			return SinkResultType::NEED_MORE_INPUT;
		}
		return Sink(context, *batch.materialized, input);
	}
	if (batch.kind != ExecutionBatchKind::LAZY_DATA_CHUNK) {
		throw InvalidInputException("streaming UDF SinkBatch received unsupported input batch kind");
	}
	auto &lstate = input.local_state.Cast<StreamingUDFLocalSinkState>();
	if (!batch.lazy) {
		lstate.consumed_blocked_input = false;
		batch = ExecutionBatch();
		return SinkResultType::NEED_MORE_INPUT;
	}
	batch.lazy->RecomputeCardinality();
	if (batch.lazy->Empty()) {
		lstate.consumed_blocked_input = false;
		batch = ExecutionBatch();
		return SinkResultType::NEED_MORE_INPUT;
	}
	auto &gstate = input.global_state.Cast<StreamingUDFGlobalSinkState>();
	auto &state = *gstate.state;
	state.sink_calls.fetch_add(1);

	auto incoming_rows = batch.lazy->cardinality;
	auto incoming_bytes = batch.lazy->EstimatedBytes();

	auto guard = state.Lock();
	DrainStreamingOutputEventsLocked(state, guard);
	ThrowIfStreamingError(state);
	DriveStreamingSubmits(context, state, guard, false);
	if (lstate.consumed_blocked_input) {
		DrainStreamingPendingBeforeInputBlock(context, state, guard, 0);
		if (StreamingInputWouldBlock(state, 0)) {
			state.blocked_sinks.fetch_add(1);
			StreamingUDFDebugState(state, "sink_reblock_lazy", true);
			return BlockStreamingSink(state, guard, input.interrupt_state);
		}
		lstate.consumed_blocked_input = false;
	}
	DrainStreamingLazyPendingBeforeInputBlock(context, state, guard, incoming_bytes);
	if (StreamingLazyInputWouldBlockBeforeAccept(state, incoming_bytes)) {
		state.blocked_sinks.fetch_add(1);
		StreamingUDFDebugState(state, "sink_block_lazy_before_accept", true);
		return BlockStreamingSink(state, guard, input.interrupt_state);
	}

	StreamingPendingLazyInput pending;
	pending.rows = incoming_rows;
	pending.bytes = incoming_bytes;
	pending.bundle = std::move(batch.lazy);
	batch = ExecutionBatch();
	state.pending_rows += pending.rows;
	state.pending_bytes += pending.bytes;
	state.pending_lazy_inputs.push_back(std::move(pending));
	state.max_pending_rows = MaxValue<idx_t>(state.max_pending_rows, state.pending_rows + state.reserved_rows);
	AtomicAddStreamingCounter(state.accepted_input_rows, incoming_rows);
	AtomicAddStreamingCounter(state.accepted_input_bytes, incoming_bytes);

	ThrowIfStreamingError(state);
	DriveStreamingSubmits(context, state, guard, false);
	DrainStreamingOutputEventsLocked(state, guard);
	if (StreamingInputWouldBlock(state, 0)) {
		lstate.consumed_blocked_input = true;
		state.blocked_sinks.fetch_add(1);
		StreamingUDFDebugState(state, "sink_block_lazy", true);
		return BlockStreamingSink(state, guard, input.interrupt_state);
	}
	return SinkResultType::NEED_MORE_INPUT;
}

SinkFinalizeType PhysicalStreamingUDF::Finalize(Pipeline &pipeline, Event &, ClientContext &context,
                                                OperatorSinkFinalizeInput &input) const {
	auto &gstate = input.global_state.Cast<StreamingUDFGlobalSinkState>();
	auto &state = *gstate.state;
	ThreadContext thread_context(context);
	ExecutionContext execution_context(context, thread_context, &pipeline);
	auto guard = state.Lock();
	DrainStreamingOutputEventsLocked(state, guard);
	ThrowIfStreamingError(state);
	state.sink_finished = true;
	StreamingUDFDebugState(state, "finalize_sink", true);
	DriveStreamingAfterSinkFinished(execution_context, state, guard);
	if (StreamingTerminalReady(state)) {
		PreventStreamingBlocking(state, guard);
		WakeAllStreamingSources(state, guard);
		WakeStreamingUDFTasks(state, guard);
		return SinkFinalizeType::READY;
	}

	// Finalize remains the durable control task for tail submission, but it
	// must sleep until a real state transition. Waking the source here while an
	// empty source wakes Finalize creates a no-progress reschedule loop. Ready
	// output is the only reason to wake the source synchronously; executor and
	// output callbacks wake it for all later transitions.
	if (StreamingHasReadyOutput(state)) {
		WakeOneStreamingSource(state, guard);
	}
	StreamingUDFDebugState(state, "finalize_blocked", true);
	return BlockStreamingFinalize(state, guard, input.interrupt_state);
}

unique_ptr<GlobalSourceState> PhysicalStreamingUDF::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<StreamingUDFGlobalSourceState>(GetStreamingState(context));
}

unique_ptr<LocalSourceState> PhysicalStreamingUDF::GetLocalSourceState(ExecutionContext &, GlobalSourceState &) const {
	return make_uniq<StreamingUDFLocalSourceState>();
}

ProgressData PhysicalStreamingUDF::GetProgress(ClientContext &, GlobalSourceState &gstate_p) const {
	auto &gstate = gstate_p.Cast<StreamingUDFGlobalSourceState>();
	auto &state = *gstate.state;
	ProgressData progress;
	if (!state.upstream_progress_valid.load(std::memory_order_acquire)) {
		progress.SetInvalid();
		return progress;
	}

	const auto upstream_total = state.upstream_progress_total.load(std::memory_order_relaxed);
	const auto upstream_done = state.upstream_progress_done.load(std::memory_order_relaxed);
	if (upstream_total <= 0.0 || upstream_done < 0.0) {
		progress.SetInvalid();
		return progress;
	}
	const auto upstream_fraction = MinValue<double>(1.0, upstream_done / upstream_total);
	const auto accepted_rows = state.accepted_input_rows.load(std::memory_order_relaxed);
	const auto completed_rows = state.completed_input_rows.load(std::memory_order_relaxed);
	double completed_fraction = accepted_rows == 0
	                                ? (upstream_fraction >= 1.0 ? 1.0 : 0.0)
	                                : MinValue<double>(1.0, static_cast<double>(completed_rows) / accepted_rows);

	const auto produced_rows = state.produced_output_rows.load(std::memory_order_relaxed);
	const auto emitted_rows = state.emitted_output_rows.load(std::memory_order_relaxed);
	if (produced_rows > 0) {
		completed_fraction *= MinValue<double>(1.0, static_cast<double>(emitted_rows) / produced_rows);
	}

	progress.done = upstream_fraction * completed_fraction;
	progress.total = 1.0;
	return progress;
}

SourceResultType PhysicalStreamingUDF::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                                       OperatorSourceInput &input) const {
	ExecutionBatch batch;
	auto result = GetDataBatch(context, batch, input);
	if (result != SourceResultType::HAVE_MORE_OUTPUT) {
		chunk.SetCardinality(0);
		return result;
	}
	if (batch.kind == ExecutionBatchKind::MATERIALIZED_CHUNK) {
		if (batch.materialized) {
			chunk.Move(*batch.materialized);
		} else {
			chunk.SetCardinality(0);
		}
		return result;
	}
	if (batch.kind == ExecutionBatchKind::LAZY_DATA_CHUNK) {
		if (!batch.lazy) {
			chunk.SetCardinality(0);
			return result;
		}
		throw InvalidInputException(
		    "streaming UDF source produced lazy output for non-batch GetData. Planner must connect this edge "
		    "through a ref-aware source path.");
	}
	throw InternalException("streaming UDF source produced unsupported ExecutionBatch kind");
}

SourceResultType PhysicalStreamingUDF::GetDataBatch(ExecutionContext &context, ExecutionBatch &batch,
                                                    OperatorSourceInput &input) const {
	auto &gstate = input.global_state.Cast<StreamingUDFGlobalSourceState>();
	auto &state = *gstate.state;
	state.source_calls.fetch_add(1);

	auto drive_locked = [&](unique_lock<mutex> &guard) {
		DrainStreamingOutputEventsLocked(state, guard);
		DriveStreamingSubmits(context, state, guard, false);
		ThrowIfStreamingError(state);
		DriveStreamingAfterSinkFinished(context, state, guard);
		ThrowIfStreamingError(state);
	};

	auto finish_if_terminal = [&](unique_lock<mutex> &guard) -> bool {
		if (state.source_finished) {
			batch = ExecutionBatch();
			return true;
		}
		if (!StreamingTerminalReady(state)) {
			return false;
		}
		state.source_finished = true;
		RefreshStreamingOutputCapacitySnapshotLocked(state, guard);
		StreamingUDFDebugState(state, "source_finished", true);
		PreventStreamingBlocking(state, guard);
		WakeAllStreamingSources(state, guard);
		WakeStreamingUDFTasks(state, guard);
		batch = ExecutionBatch();
		return true;
	};

	auto emit_ready = [&](unique_lock<mutex> &guard) -> bool {
		if (state.ready_outputs.empty()) {
			return false;
		}
		auto ready = TakeStreamingReadyOutputLocked(state, guard);
		const auto ready_rows = UDFExecutionBatchSize(ready.batch);
		state.source_have_more.fetch_add(1);
		UDFExecutor *executor = state.op && state.op->executor ? state.op->executor.get() : nullptr;
		FlushDeferredStreamingOutputsLocked(state, guard);
		UpdateStreamingOutputBytesLocked(state, guard);
		DriveStreamingSubmits(context, state, guard, false);
		ThrowIfStreamingError(state);
		DriveStreamingAfterSinkFinished(context, state, guard);
		if (!state.ready_outputs.empty()) {
			WakeOneStreamingSource(state, guard);
		}
		StreamingUDFDebugState(state, "source_emit");
		if (StreamingTerminalReady(state)) {
			PreventStreamingBlocking(state, guard);
			WakeAllStreamingSources(state, guard);
		}
		WakeStreamingUDFTasks(state, guard);
		guard.unlock();
		if (executor) {
			state.notify_space_available.fetch_add(1);
			executor->NotifyOutputConsumerSpaceAvailable();
		}
		AtomicAddStreamingCounter(state.emitted_output_rows, ready_rows);
		AtomicAddStreamingCounter(state.emitted_output_bytes, ready.bytes);
		batch = std::move(ready.batch);
		return true;
	};

	UDFExecutor *notify_executor = nullptr;
	{
		auto guard = state.Lock();
		ThrowIfStreamingError(state);
		// The source can be the only task that runs after ready output frees
		// executor/result capacity. Keep pending lazy inputs moving even when
		// the sink is not finalized yet.
		drive_locked(guard);
		if (emit_ready(guard)) {
			return SourceResultType::HAVE_MORE_OUTPUT;
		}
		if (finish_if_terminal(guard)) {
			return SourceResultType::FINISHED;
		}
		notify_executor = state.op && state.op->executor ? state.op->executor.get() : nullptr;
		if (notify_executor) {
			state.notify_space_available.fetch_add(1);
			StreamingUDFDebugState(state, "source_empty_notify_capacity");
		} else {
			state.blocked_sources.fetch_add(1);
			state.source_blocked_empty.fetch_add(1);
			StreamingUDFDebugState(state, "source_blocked_empty", true);
			batch = ExecutionBatch();
			return BlockStreamingSource(state, guard, input.interrupt_state);
		}
	}

	notify_executor->NotifyOutputConsumerSpaceAvailable();

	auto guard = state.Lock();
	ThrowIfStreamingError(state);
	drive_locked(guard);
	if (emit_ready(guard)) {
		return SourceResultType::HAVE_MORE_OUTPUT;
	}
	if (StreamingTerminalReady(state)) {
		state.source_finished = true;
		RefreshStreamingOutputCapacitySnapshotLocked(state, guard);
		StreamingUDFDebugState(state, "source_finished", true);
		PreventStreamingBlocking(state, guard);
		WakeAllStreamingSources(state, guard);
		WakeStreamingUDFTasks(state, guard);
		batch = ExecutionBatch();
		return SourceResultType::FINISHED;
	}
	state.blocked_sources.fetch_add(1);
	state.source_blocked_empty.fetch_add(1);
	StreamingUDFDebugState(state, "source_blocked_empty", true);
	batch = ExecutionBatch();
	return BlockStreamingSource(state, guard, input.interrupt_state);
}

InsertionOrderPreservingMap<string> PhysicalStreamingUDF::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Name"] = function.name.empty() ? "udf" : function.name;
	result["streaming_breaker"] = "true";
	if (bind_data) {
		auto &udf_bind = bind_data->Cast<UDFFunctionData>();
		auto udf_name = GetStructStringField(udf_bind.payload, "udf_name");
		if (udf_name.first && !udf_name.second.empty()) {
			result["udf_name"] = udf_name.second;
		}
		auto call_mode = GetStructStringField(udf_bind.payload, "call_mode");
		if (call_mode.first && !call_mode.second.empty()) {
			result["call_mode"] = call_mode.second;
		}
		auto execution_backend = GetStructStringField(udf_bind.payload, "execution_backend");
		if (execution_backend.first && !execution_backend.second.empty()) {
			result["execution_backend"] = execution_backend.second;
		}
		auto row_preserving = GetStructBoolField(udf_bind.payload, "row_preserving");
		if (row_preserving.first) {
			result["row_preserving"] = row_preserving.second ? "true" : "false";
		}
		AppendUDFExecutionConfigParams(result, udf_bind.payload);
		if (IsRowPreservingPythonUDFLayoutPayload(udf_bind.payload)) {
			auto arg_count = GetStructIntField(udf_bind.payload, "scalar_arg_count");
			if (arg_count.first) {
				result["scalar_arg_count"] = std::to_string(arg_count.second);
			}
		}
		auto produce_ref_bundle = GetStructBoolField(udf_bind.payload, "produce_ref_bundle_output");
		if (produce_ref_bundle.first && produce_ref_bundle.second) {
			auto output_mode = GetStructStringField(udf_bind.payload, "streaming_output_mode");
			result["ref_bundle_output"] =
			    output_mode.first && !output_mode.second.empty() ? output_mode.second : "local_shm_ref_bundle";
		}
		auto produce_ray_block_stream = GetStructBoolField(udf_bind.payload, "produce_ray_block_stream");
		if (produce_ray_block_stream.first && produce_ray_block_stream.second) {
			result["ray_block_stream_output"] = "direct_block_metadata_pair";
		}
	}
	std::shared_ptr<StreamingUDFState> state;
	{
		lock_guard<std::mutex> guard(streaming_state_lock);
		state = streaming_state;
	}
	if (state) {
		result["udf_resolved_source_threads"] =
		    std::to_string(state->resolved_source_threads.load(std::memory_order_relaxed));
		result["udf_resolved_sink_threads"] =
		    std::to_string(state->resolved_sink_threads.load(std::memory_order_relaxed));
		result["udf_accepted_input_rows"] = std::to_string(state->accepted_input_rows.load(std::memory_order_relaxed));
		result["udf_accepted_input_bytes"] =
		    std::to_string(state->accepted_input_bytes.load(std::memory_order_relaxed));
		result["udf_submitted_input_rows"] =
		    std::to_string(state->submitted_input_rows.load(std::memory_order_relaxed));
		result["udf_submitted_input_bytes"] =
		    std::to_string(state->submitted_input_bytes.load(std::memory_order_relaxed));
		result["udf_completed_input_rows"] =
		    std::to_string(state->completed_input_rows.load(std::memory_order_relaxed));
		result["udf_completed_input_bytes"] =
		    std::to_string(state->completed_input_bytes.load(std::memory_order_relaxed));
		result["udf_produced_output_rows"] =
		    std::to_string(state->produced_output_rows.load(std::memory_order_relaxed));
		result["udf_produced_output_bytes"] =
		    std::to_string(state->produced_output_bytes.load(std::memory_order_relaxed));
		result["udf_emitted_output_rows"] = std::to_string(state->emitted_output_rows.load(std::memory_order_relaxed));
		result["udf_emitted_output_bytes"] =
		    std::to_string(state->emitted_output_bytes.load(std::memory_order_relaxed));
		result["udf_handoff_output_batches"] =
		    std::to_string(state->handoff_counters->outputs.load(std::memory_order_relaxed));
		result["udf_handoff_output_rows"] =
		    std::to_string(state->handoff_counters->rows.load(std::memory_order_relaxed));
		result["udf_handoff_output_bytes"] =
		    std::to_string(state->handoff_counters->bytes.load(std::memory_order_relaxed));
		result["udf_max_handoff_output_batches"] =
		    std::to_string(state->handoff_counters->max_outputs.load(std::memory_order_relaxed));
		result["udf_max_handoff_output_rows"] =
		    std::to_string(state->handoff_counters->max_rows.load(std::memory_order_relaxed));
		result["udf_max_handoff_output_bytes"] =
		    std::to_string(state->handoff_counters->max_bytes.load(std::memory_order_relaxed));
		{
			auto guard = state->Lock();
			result["udf_pending_input_rows"] = std::to_string(state->pending_rows);
			result["udf_pending_input_bytes"] = std::to_string(state->pending_bytes);
			result["udf_pending_input_batches"] = std::to_string(state->pending_inputs.size());
			result["udf_pending_lazy_batches"] = std::to_string(state->pending_lazy_inputs.size());
			result["udf_inflight_batches"] = std::to_string(state->inflight_batches.size());
			result["udf_inflight_rows"] = std::to_string(state->inflight_rows);
			result["udf_inflight_bytes"] = std::to_string(state->inflight_bytes);
			result["udf_ready_output_batches"] = std::to_string(state->ready_outputs.size());
			result["udf_ready_output_rows"] = std::to_string(state->ready_rows);
			result["udf_ready_output_bytes"] = std::to_string(state->ready_bytes);
			result["udf_deferred_output_batches"] = std::to_string(state->deferred_outputs.size());
			result["udf_deferred_output_rows"] = std::to_string(state->deferred_output_rows);
			result["udf_deferred_output_bytes"] = std::to_string(state->deferred_output_bytes);
			result["udf_blocked_source_tasks"] = std::to_string(state->blocked_source_tasks.size());
			result["udf_blocked_control_tasks"] = std::to_string(state->blocked_control_tasks.size());
			result["udf_sink_finished"] = state->sink_finished ? "1" : "0";
			result["udf_source_finished"] = state->source_finished ? "1" : "0";
			result["udf_dispatcher_finished"] = state->dispatcher_finished ? "1" : "0";
			result["udf_executor_finished_submitting"] = state->executor_finished_submitting ? "1" : "0";
			result["udf_streaming_terminal_ready"] = StreamingTerminalReady(*state) ? "1" : "0";
			result["udf_streaming_can_block"] = state->streaming_can_block ? "1" : "0";
			result["udf_max_pending_rows"] = std::to_string(state->max_pending_rows);
			result["udf_max_ready_observed_rows"] = std::to_string(state->max_ready_rows);
			result["udf_max_active_observed_batches"] = std::to_string(state->max_active_batches);
			result["udf_max_outstanding_observed_rows"] = std::to_string(state->max_outstanding_rows);
			AppendUDFExecutorStatsParams(result, state->op ? state->op->executor.get() : nullptr);
		}
	}
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

void PhysicalStreamingUDF::SerializeOperatorData(Serializer &serializer) const {
	FunctionSerializer::Serialize(serializer, function, bind_data.get());
	serializer.WriteProperty(200, "column_ids", column_ids);
	serializer.WriteProperty(201, "projected_input", projected_input);
	serializer.WritePropertyWithDefault(202, "ordinality_idx", ordinality_idx);
}

// ─── Serialize/deserialize for udf TableFunction ──────────────────────

static string UDFTableFunctionSerializationTestCorruption() {
	if (!DebugEnvFlagEnabled("VANE_ENABLE_UDF_TEST_HOOKS")) {
		return {};
	}
	const char *corruption = std::getenv("VANE_TEST_CORRUPT_UDF_PHYSICAL_PAYLOAD");
	return corruption ? string(corruption) : string();
}

static Value UDFTableFunctionSerializationPayload(const Value &payload) {
	auto corruption = UDFTableFunctionSerializationTestCorruption();
	if (corruption == "payload_version") {
		return ReplaceStructFields(payload, {{"payload_version", Value::BIGINT(999)}});
	}
	if (corruption == "logical_return_type") {
		return ReplaceStructFields(payload, {{"method_return_type", Value(LogicalType::VARCHAR)}});
	}
	return payload;
}

static LogicalType UDFTableFunctionSerializationReturnType(const LogicalType &return_type) {
	if (UDFTableFunctionSerializationTestCorruption() != "return_type") {
		return return_type;
	}
	return return_type == LogicalType::VARCHAR ? LogicalType::BIGINT : LogicalType::VARCHAR;
}

static void UDFTableFunctionSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data,
                                      const TableFunction &) {
	if (!bind_data) {
		serializer.WriteProperty<bool>(100, "has_bind_data", false);
		return;
	}
	auto &data = bind_data->Cast<UDFFunctionData>();
	serializer.WriteProperty<bool>(100, "has_bind_data", true);
	serializer.WriteProperty(101, "payload", UDFTableFunctionSerializationPayload(data.payload));
	serializer.WriteProperty(102, "return_type", UDFTableFunctionSerializationReturnType(data.return_type));
}

static unique_ptr<FunctionData> UDFTableFunctionDeserialize(Deserializer &deserializer, TableFunction &) {
	auto has_bind_data = deserializer.ReadProperty<bool>(100, "has_bind_data");
	if (!has_bind_data) {
		return nullptr;
	}
	auto payload = deserializer.ReadProperty<Value>(101, "payload");
	auto return_type = deserializer.ReadProperty<LogicalType>(102, "return_type");
	auto payload_return_type = udf_helpers::ResolvePayloadReturnType(payload);
	if (payload_return_type != return_type) {
		throw SerializationException("udf: serialized return type '%s' does not match payload return type '%s'",
		                             return_type.ToString(), payload_return_type.ToString());
	}
	return make_uniq<UDFFunctionData>(std::move(payload), std::move(return_type));
}

// ─── Public: create a TableFunction with INOUT callbacks ────────────────────
TableFunction MakeUDFTableFunction(Value payload, const vector<LogicalType> &return_types,
                                   const vector<string> &return_names) {
	// Create bind data
	LogicalType return_type;
	if (return_types.size() == 1) {
		return_type = return_types[0];
	} else {
		child_list_t<LogicalType> struct_children;
		for (idx_t i = 0; i < return_types.size(); i++) {
			auto name = i < return_names.size() ? return_names[i] : StringUtil::Format("c%d", i);
			struct_children.emplace_back(name, return_types[i]);
		}
		return_type = LogicalType::STRUCT(std::move(struct_children));
	}

	TableFunction tf("udf", {}, nullptr);
	tf.in_out_function = UDFInOutExecute;
	tf.in_out_function_final = UDFInOutFinal;
	tf.in_out_function_batch = UDFInOutExecuteBatch;
	tf.in_out_function_final_batch = UDFInOutFinalBatch;
	tf.init_global = UDFInitGlobal;
	tf.init_local = UDFInitLocal;
	tf.serialize = UDFTableFunctionSerialize;
	tf.deserialize = UDFTableFunctionDeserialize;
	tf.to_string = UDFTableFunctionToString;
	tf.dynamic_to_string = UDFTableFunctionDynamicToString;
	tf.function_info = make_shared_ptr<TableFunctionInfo>();
	return tf;
}

// ─── Registered version: includes bind function for conn.create_function() ──

//! Stores payload in function_info so the bind function can create UDFFunctionData.
struct UDFTableInfo : public TableFunctionInfo {
	explicit UDFTableInfo(Value payload_p, vector<LogicalType> output_types_p, vector<string> output_names_p)
	    : payload(std::move(payload_p)), output_types(std::move(output_types_p)),
	      output_names(std::move(output_names_p)) {
	}

	Value payload;
	vector<LogicalType> output_types;
	vector<string> output_names;
};

static unique_ptr<FunctionData> UDFRegisteredBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &return_names) {
	auto &info = input.info->Cast<UDFTableInfo>();
	return_types = info.output_types;
	return_names = info.output_names;

	LogicalType return_type;
	if (return_types.size() == 1) {
		return_type = return_types[0];
	} else {
		child_list_t<LogicalType> struct_children;
		for (idx_t i = 0; i < return_types.size(); i++) {
			struct_children.emplace_back(return_names[i], return_types[i]);
		}
		return_type = LogicalType::STRUCT(std::move(struct_children));
	}
	return make_uniq<UDFFunctionData>(info.payload, std::move(return_type));
}

TableFunction MakeUDFRegisteredTableFunction(string name, Value payload, vector<LogicalType> output_types,
                                             vector<string> output_names) {
	TableFunction tf(std::move(name), {LogicalType::TABLE}, nullptr, UDFRegisteredBind);
	tf.in_out_function = UDFInOutExecute;
	tf.in_out_function_final = UDFInOutFinal;
	tf.in_out_function_batch = UDFInOutExecuteBatch;
	tf.in_out_function_final_batch = UDFInOutFinalBatch;
	tf.init_global = UDFInitGlobal;
	tf.init_local = UDFInitLocal;
	tf.serialize = UDFTableFunctionSerialize;
	tf.deserialize = UDFTableFunctionDeserialize;
	tf.to_string = UDFTableFunctionToString;
	tf.dynamic_to_string = UDFTableFunctionDynamicToString;
	tf.function_info =
	    make_shared_ptr<UDFTableInfo>(std::move(payload), std::move(output_types), std::move(output_names));
	return tf;
}

TableFunction GetUDFBuiltinTableFunction() {
	// Minimal table function registered in the catalog at init time.
	// The INOUT callbacks are the same as the dynamically-created version.
	// This entry exists solely so BinaryDeserializer can look up "udf" as
	// a TableFunction when reconstructing serialized plans on remote workers.
	TableFunction tf("udf", {}, nullptr);
	tf.in_out_function = UDFInOutExecute;
	tf.in_out_function_final = UDFInOutFinal;
	tf.in_out_function_batch = UDFInOutExecuteBatch;
	tf.in_out_function_final_batch = UDFInOutFinalBatch;
	tf.init_global = UDFInitGlobal;
	tf.init_local = UDFInitLocal;
	tf.serialize = UDFTableFunctionSerialize;
	tf.deserialize = UDFTableFunctionDeserialize;
	tf.to_string = UDFTableFunctionToString;
	tf.dynamic_to_string = UDFTableFunctionDynamicToString;
	return tf;
}

} // namespace duckdb

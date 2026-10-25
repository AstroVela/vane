// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "parquet_mask.hpp"

#include "duckdb/common/vector_operations/unary_executor.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/expression_executor_state.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/storage/buffer_manager.hpp"

#include <cstring>

namespace duckdb {
namespace vane_execution {
namespace {

PhysicalTableScan *FindScan(PhysicalOperator &op, idx_t &key) {
	if (key >= op.types.size() || !op.types[key].IsIntegral()) {
		return nullptr;
	}
	if (op.type == PhysicalOperatorType::PROJECTION && op.children.size() == 1) {
		auto &expr = *op.Cast<PhysicalProjection>().select_list[key];
		if (expr.GetExpressionClass() != ExpressionClass::BOUND_REF) {
			return nullptr;
		}
		auto column = expr.Cast<BoundReferenceExpression>().index;
		auto &child = op.children[0].get();
		if (column >= child.types.size() || child.types[column] != op.types[key]) {
			return nullptr;
		}
		key = column;
		return FindScan(child, key);
	}
	if (op.type == PhysicalOperatorType::FILTER && op.children.size() == 1 && op.types == op.children[0].get().types) {
		return FindScan(op.children[0], key);
	}
	if (op.type != PhysicalOperatorType::TABLE_SCAN) {
		return nullptr;
	}
	auto &scan = op.Cast<PhysicalTableScan>();
	if ((scan.function.name != "read_parquet" && scan.function.name != "parquet_scan") ||
	    !scan.function.filter_pushdown || scan.extra_info.sample_options) {
		return nullptr;
	}
	if (!scan.projection_ids.empty()) {
		if (key >= scan.projection_ids.size()) {
			return nullptr;
		}
		key = scan.projection_ids[key];
	}
	if (key >= scan.column_ids.size() || scan.column_ids[key].HasChildren()) {
		return nullptr;
	}
	return &scan;
}

struct MaskData {
	MaskData(ClientContext &context, idx_t count)
	    : buffer(BufferManager::GetBufferManager(context).Allocate(MemoryTag::HASH_TABLE, count * sizeof(uint64_t))),
	      count(count) {
		std::memset(buffer.Ptr(), 0, count * sizeof(uint64_t));
	}

	BufferHandle buffer;
	const idx_t count;

	uint64_t *Words() {
		return reinterpret_cast<uint64_t *>(buffer.Ptr());
	}

	bool Contains(hash_t hash) const {
		auto words = reinterpret_cast<const uint64_t *>(buffer.Ptr());
		return (words[hash & (count - 1)] & (uint64_t(1) << ((hash >> 32) & 63))) != 0;
	}
};

struct MaskBindData : FunctionData {
	explicit MaskBindData(shared_ptr<MaskData> mask) : mask(std::move(mask)) {
	}
	shared_ptr<MaskData> mask;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<MaskBindData>(mask);
	}
	bool Equals(const FunctionData &other) const override {
		return mask == other.Cast<MaskBindData>().mask;
	}
};

struct MaskLocalState : FunctionLocalState {
	MaskLocalState() : hashes(LogicalType::HASH) {
	}
	Vector hashes;
};

unique_ptr<FunctionLocalState> InitMask(ExpressionState &, const BoundFunctionExpression &, FunctionData *) {
	return make_uniq<MaskLocalState>();
}

void ContainsMask(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &expr = state.expr.Cast<BoundFunctionExpression>();
	auto &mask = *expr.bind_info->Cast<MaskBindData>().mask;
	auto &local = ExecuteFunctionState::GetFunctionState(state)->Cast<MaskLocalState>();
	// SQL hash() also hashes NULL. The original join retains its
	// equality/NULL semantics; this is only a necessary membership condition.
	VectorOperations::Hash(args.data[0], local.hashes, args.size());
	UnaryExecutor::Execute<hash_t, bool>(local.hashes, result, args.size(),
	                                     [&](hash_t hash) { return mask.Contains(hash); });
}

struct MaskSinkState : GlobalSinkState {
	MaskSinkState(ClientContext &context, idx_t words) : mask(make_shared_ptr<MaskData>(context, words)) {
	}
	shared_ptr<MaskData> mask;
};

class PhysicalParquetMask : public PhysicalOperator {
public:
	PhysicalParquetMask(PhysicalPlan &plan, PhysicalOperator &probe, PhysicalOperator &domain, PhysicalTableScan &scan,
	                    idx_t column, idx_t words, LogicalType key_type)
	    : PhysicalOperator(plan, PhysicalOperatorType::EXTENSION, probe.types, probe.estimated_cardinality), scan(scan),
	      column(column), words(words), key_type(std::move(key_type)) {
		children.push_back(probe);
		children.push_back(domain);
		if (!scan.dynamic_filters) {
			scan.dynamic_filters = make_shared_ptr<DynamicTableFilterSet>();
		}
	}

	string GetName() const override {
		return "PARQUET_MASK_FILTER";
	}
	bool IsSink() const override {
		return true;
	}
	bool ParallelOperator() const override {
		return true;
	}
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override {
		scan.dynamic_filters->ClearFilters(*this);
		return make_uniq<MaskSinkState>(context, words);
	}
	SinkResultType Sink(ExecutionContext &, DataChunk &chunk, OperatorSinkInput &input) const override {
		auto &mask = *input.global_state.Cast<MaskSinkState>().mask;
		UnifiedVectorFormat word_data, mask_data;
		chunk.data[0].ToUnifiedFormat(chunk.size(), word_data);
		chunk.data[1].ToUnifiedFormat(chunk.size(), mask_data);
		auto word_values = UnifiedVectorFormat::GetData<uint64_t>(word_data);
		auto mask_values = UnifiedVectorFormat::GetData<uint64_t>(mask_data);
		for (idx_t i = 0; i < chunk.size(); i++) {
			auto word_index = word_data.sel->get_index(i);
			auto mask_index = mask_data.sel->get_index(i);
			if (!word_data.validity.RowIsValid(word_index) || !mask_data.validity.RowIsValid(mask_index) ||
			    word_values[word_index] >= words) {
				throw InvalidInputException("Parquet mask input requires non-null masks and bounded word indices");
			}
			mask.Words()[word_values[word_index]] |= mask_values[mask_index];
		}
		return SinkResultType::NEED_MORE_INPUT;
	}
	SinkFinalizeType Finalize(Pipeline &, Event &, ClientContext &, OperatorSinkFinalizeInput &input) const override {
		auto &state = input.global_state.Cast<MaskSinkState>();
		ScalarFunction function("vane_parquet_mask", {key_type}, LogicalType::BOOLEAN, ContainsMask);
		function.SetInitStateCallback(InitMask);
		function.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
		vector<unique_ptr<Expression>> arguments;
		arguments.push_back(make_uniq<BoundReferenceExpression>(key_type, 0));
		auto expression = make_uniq<BoundFunctionExpression>(LogicalType::BOOLEAN, std::move(function),
		                                                     std::move(arguments), make_uniq<MaskBindData>(state.mask));
		scan.dynamic_filters->PushFilter(*this, column, make_uniq<ExpressionFilter>(std::move(expression)));
		return SinkFinalizeType::READY;
	}
	OperatorResultType Execute(ExecutionContext &, DataChunk &input, DataChunk &output, GlobalOperatorState &,
	                           OperatorState &) const override {
		output.Reference(input);
		return OperatorResultType::NEED_MORE_INPUT;
	}
	void BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) override {
		op_state.reset();
		sink_state.reset();
		meta_pipeline.GetState().AddPipelineOperator(current, *this);
		auto &build = meta_pipeline.CreateChildMetaPipeline(current, *this, MetaPipelineType::JOIN_BUILD);
		build.Build(children[1]);
		// The admitted probe is a unary chain over one scan. The child build's
		// finalize event publishes its complete filter before that source starts.
		children[0].get().BuildPipelines(current, meta_pipeline);
	}
	vector<const_reference<PhysicalOperator>> GetSources() const override {
		return children[0].get().GetSources();
	}

private:
	PhysicalTableScan &scan;
	idx_t column;
	idx_t words;
	LogicalType key_type;
};

} // namespace

bool ValidParquetMaskSize(idx_t words) {
	return words >= 65536 && words <= 524288 && (words & (words - 1)) == 0;
}

bool CanFilterParquetKey(PhysicalOperator &probe, idx_t key) {
	return FindScan(probe, key) != nullptr;
}

PhysicalOperator &MakeParquetMask(PhysicalPlan &plan, PhysicalOperator &probe, PhysicalOperator &domain, idx_t key,
                                  idx_t words) {
	if (!ValidParquetMaskSize(words) || key >= probe.types.size() ||
	    domain.types != vector<LogicalType> {LogicalType::UBIGINT, LogicalType::UBIGINT}) {
		throw SerializationException("invalid Parquet mask shape");
	}
	auto key_type = probe.types[key];
	auto scan = FindScan(probe, key);
	if (!scan) {
		throw SerializationException("Parquet mask requires an unchanged integral scan key");
	}
	return plan.Make<PhysicalParquetMask>(probe, domain, *scan, key, words, std::move(key_type));
}

} // namespace vane_execution
} // namespace duckdb

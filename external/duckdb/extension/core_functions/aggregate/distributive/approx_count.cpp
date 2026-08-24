#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/hash.hpp"
#include "duckdb/common/types/hyperloglog.hpp"
#include "duckdb/common/type_visitor.hpp"
#include "core_functions/aggregate/distributive_functions.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "hyperloglog.hpp"

namespace duckdb {

// Algorithms from
// "New cardinality estimation algorithms for HyperLogLog sketches"
// Otmar Ertl, arXiv:1702.01284
namespace {

struct ApproxDistinctCountState {
	HyperLogLog hll;
};

struct ApproxCountDistinctFunction {
	template <class STATE>
	static void Initialize(STATE &state) {
		new (&state) STATE();
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE &target, AggregateInputData &) {
		target.hll.Merge(source.hll);
	}

	template <class T, class STATE>
	static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
		target = UnsafeNumericCast<T>(state.hll.Count());
	}

	static bool IgnoreNull() {
		return true;
	}
};

void ApproxCountDistinctSimpleUpdateFunction(Vector inputs[], AggregateInputData &, idx_t input_count, data_ptr_t state,
                                             idx_t count) {
	D_ASSERT(input_count == 1);
	auto &input = inputs[0];

	if (count > STANDARD_VECTOR_SIZE) {
		throw InternalException("ApproxCountDistinct - count must be at most vector size");
	}
	Vector hash_vec(LogicalType::HASH, count);
	VectorOperations::Hash(input, hash_vec, count);

	auto agg_state = reinterpret_cast<ApproxDistinctCountState *>(state);
	agg_state->hll.Update(input, hash_vec, count);
}

void ApproxCountDistinctUpdateFunction(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &state_vector,
                                       idx_t count) {
	D_ASSERT(input_count == 1);
	auto &input = inputs[0];
	UnifiedVectorFormat idata;
	input.ToUnifiedFormat(count, idata);

	if (count > STANDARD_VECTOR_SIZE) {
		throw InternalException("ApproxCountDistinct - count must be at most vector size");
	}
	Vector hash_vec(LogicalType::HASH, count);
	VectorOperations::Hash(input, hash_vec, count);

	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	const auto states = UnifiedVectorFormat::GetDataNoConst<ApproxDistinctCountState *>(sdata);

	UnifiedVectorFormat hdata;
	hash_vec.ToUnifiedFormat(count, hdata);
	const auto *hashes = UnifiedVectorFormat::GetData<hash_t>(hdata);
	for (idx_t i = 0; i < count; i++) {
		if (idata.validity.RowIsValid(idata.sel->get_index(i))) {
			auto agg_state = states[sdata.sel->get_index(i)];
			const auto hash = hashes[hdata.sel->get_index(i)];
			agg_state->hll.InsertElement(hash);
		}
	}
}

AggregateFunction GetApproxCountDistinctFunction(const LogicalType &input_type) {
	auto fun = AggregateFunction(
	    {input_type}, LogicalTypeId::BIGINT, AggregateFunction::StateSize<ApproxDistinctCountState>,
	    AggregateFunction::StateInitialize<ApproxDistinctCountState, ApproxCountDistinctFunction>,
	    ApproxCountDistinctUpdateFunction,
	    AggregateFunction::StateCombine<ApproxDistinctCountState, ApproxCountDistinctFunction>,
	    AggregateFunction::StateFinalize<ApproxDistinctCountState, int64_t, ApproxCountDistinctFunction>,
	    ApproxCountDistinctSimpleUpdateFunction);
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	return fun;
}

unique_ptr<FunctionData> BindApproxCountDistinct(ClientContext &, AggregateFunction &,
                                                 vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(arguments.size() == 1);
	if (TypeVisitor::Contains(arguments[0]->return_type, FileLogicalType::IsFile)) {
		throw BinderException("approx_count_distinct does not support FILE values");
	}
	return nullptr;
}

} // namespace

AggregateFunction ApproxCountDistinctFun::GetFunction() {
	auto function = GetApproxCountDistinctFunction(LogicalType::ANY);
	function.SetBindCallback(BindApproxCountDistinct);
	return function;
}

} // namespace duckdb

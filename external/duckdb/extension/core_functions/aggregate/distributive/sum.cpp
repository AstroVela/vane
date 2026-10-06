// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "core_functions/aggregate/distributive_functions.hpp"
#include "core_functions/aggregate/sum_helpers.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/bignum.hpp"
#include "duckdb/common/types/decimal.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/function/aggregate/hugeint_sum.hpp"

namespace duckdb {

namespace {

struct SumSetOperation {
	template <class STATE>
	static void Initialize(STATE &state) {
		state.Initialize();
	}
	template <class STATE>
	static void Combine(const STATE &source, STATE &target, AggregateInputData &) {
		target.Combine(source);
	}
	template <class STATE>
	static void AddValues(STATE &state, idx_t count) {
		state.isset = true;
	}
};

struct IntegerSumOperation : public BaseSumOperation<SumSetOperation, RegularAdd> {
	template <class T, class STATE>
	static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
		if (!state.isset) {
			finalize_data.ReturnNull();
		} else {
			target = Hugeint::Convert(state.value);
		}
	}
};

struct SumToHugeintOperation : public BaseSumOperation<SumSetOperation, AddToHugeint> {
	template <class T, class STATE>
	static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
		if (!state.isset) {
			finalize_data.ReturnNull();
		} else {
			target = state.value;
		}
	}
};

template <class ADD_OPERATOR>
struct DoubleSumOperation : public BaseSumOperation<SumSetOperation, ADD_OPERATOR> {
	template <class T, class STATE>
	static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
		if (!state.isset) {
			finalize_data.ReturnNull();
		} else {
			target = state.value;
		}
	}
};

using NumericSumOperation = DoubleSumOperation<RegularAdd>;
using KahanSumOperation = DoubleSumOperation<KahanAdd>;

struct HugeintSumOperation : public BaseSumOperation<SumSetOperation, HugeintAdd> {
	template <class T, class STATE>
	static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
		if (!state.isset) {
			finalize_data.ReturnNull();
		} else {
			target = state.value;
		}
	}
};

// Three unsigned limbs hold a signed 192-bit sum without signed arithmetic
// overflow. This covers idx_t (64-bit) rows of signed 128-bit input, regardless
// of the order in which distributed producers deliver their rows.
struct WideHugeintSumState {
	bool isset;
	uint64_t lower, middle, upper;

	void Initialize() {
		isset = false;
		lower = middle = upper = 0;
	}
	void Add(uint64_t low, uint64_t mid, uint64_t high) {
		auto old_lower = lower;
		lower += low;
		auto carry = uint64_t(lower < old_lower);
		auto old_middle = middle;
		middle += mid;
		upper += high + uint64_t(middle < old_middle);
		old_middle = middle;
		middle += carry;
		upper += uint64_t(middle < old_middle);
	}
	void Combine(const WideHugeintSumState &other) {
		isset = isset || other.isset;
		Add(other.lower, other.middle, other.upper);
	}
};

struct WideHugeintSumOperation {
	template <class STATE>
	static void Initialize(STATE &state) {
		state.Initialize();
	}
	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE &target, AggregateInputData &) {
		target.Combine(source);
	}
	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE &state, const INPUT_TYPE &value, AggregateUnaryInput &) {
		state.isset = true;
		state.Add(value.lower, uint64_t(value.upper), value.upper < 0 ? NumericLimits<uint64_t>::Maximum() : 0);
	}
	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE &state, const INPUT_TYPE &value, AggregateUnaryInput &input, idx_t count) {
		for (idx_t row = 0; row < count; row++) {
			Operation<INPUT_TYPE, STATE, OP>(state, value, input);
		}
	}
	static bool IgnoreNull() {
		return true;
	}
	template <class T, class STATE>
	static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
		if (!state.isset) {
			finalize_data.ReturnNull();
			return;
		}
		const auto sign_extension = state.middle >> 63 ? NumericLimits<uint64_t>::Maximum() : 0;
		if (state.upper != sign_extension) {
			throw OutOfRangeException("Overflow in HUGEINT sum");
		}
		target = hugeint_t(int64_t(state.middle), state.lower);
	}
};

void HugeintSumSerialize(Serializer &serializer, const optional_ptr<FunctionData>, const AggregateFunction &) {
	serializer.WriteProperty(1, "wide_accumulator", false);
}

void WideHugeintSumSerialize(Serializer &serializer, const optional_ptr<FunctionData>, const AggregateFunction &) {
	serializer.WriteProperty(1, "wide_accumulator", true);
}

unique_ptr<FunctionData> HugeintSumDeserialize(Deserializer &deserializer, AggregateFunction &function);

unique_ptr<FunctionData> SumNoOverflowBind(ClientContext &context, AggregateFunction &function,
                                           vector<unique_ptr<Expression>> &arguments) {
	throw BinderException("sum_no_overflow is for internal use only!");
}

void SumNoOverflowSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data,
                            const AggregateFunction &function) {
	return;
}

unique_ptr<FunctionData> SumNoOverflowDeserialize(Deserializer &deserializer, AggregateFunction &function) {
	function.SetReturnType(deserializer.Get<const LogicalType &>());
	return nullptr;
}

AggregateFunction GetSumAggregateNoOverflow(PhysicalType type) {
	switch (type) {
	case PhysicalType::INT32: {
		auto function = AggregateFunction::UnaryAggregate<SumState<int64_t>, int32_t, hugeint_t, IntegerSumOperation>(
		    LogicalType::INTEGER, LogicalType::HUGEINT);
		function.name = "sum_no_overflow";
		function.SetOrderDependent(AggregateOrderDependent::NOT_ORDER_DEPENDENT);
		function.SetBindCallback(SumNoOverflowBind);
		function.SetSerializeCallback(SumNoOverflowSerialize);
		function.SetDeserializeCallback(SumNoOverflowDeserialize);
		return function;
	}
	case PhysicalType::INT64: {
		auto function = AggregateFunction::UnaryAggregate<SumState<int64_t>, int64_t, hugeint_t, IntegerSumOperation>(
		    LogicalType::BIGINT, LogicalType::HUGEINT);
		function.name = "sum_no_overflow";
		function.SetOrderDependent(AggregateOrderDependent::NOT_ORDER_DEPENDENT);
		function.SetBindCallback(SumNoOverflowBind);
		function.SetSerializeCallback(SumNoOverflowSerialize);
		function.SetDeserializeCallback(SumNoOverflowDeserialize);
		return function;
	}
	default:
		throw BinderException("Unsupported internal type for sum_no_overflow");
	}
}

AggregateFunction GetSumAggregateNoOverflowDecimal() {
	AggregateFunction aggr({LogicalTypeId::DECIMAL}, LogicalTypeId::DECIMAL, nullptr, nullptr, nullptr, nullptr,
	                       nullptr, FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, SumNoOverflowBind);
	aggr.SetSerializeCallback(SumNoOverflowSerialize);
	aggr.SetDeserializeCallback(SumNoOverflowDeserialize);
	return aggr;
}

unique_ptr<BaseStatistics> SumPropagateStats(ClientContext &context, BoundAggregateExpression &expr,
                                             AggregateStatisticsInput &input) {
	if (input.node_stats && input.node_stats->has_max_cardinality) {
		auto &numeric_stats = input.child_stats[0];
		if (!NumericStats::HasMinMax(numeric_stats)) {
			return nullptr;
		}
		auto internal_type = numeric_stats.GetType().InternalType();
		hugeint_t max_negative;
		hugeint_t max_positive;
		switch (internal_type) {
		case PhysicalType::INT32:
			max_negative = NumericStats::Min(numeric_stats).GetValueUnsafe<int32_t>();
			max_positive = NumericStats::Max(numeric_stats).GetValueUnsafe<int32_t>();
			break;
		case PhysicalType::INT64:
			max_negative = NumericStats::Min(numeric_stats).GetValueUnsafe<int64_t>();
			max_positive = NumericStats::Max(numeric_stats).GetValueUnsafe<int64_t>();
			break;
		default:
			throw InternalException("Unsupported type for propagate sum stats");
		}
		auto max_sum_negative = max_negative * Hugeint::Convert(input.node_stats->max_cardinality);
		auto max_sum_positive = max_positive * Hugeint::Convert(input.node_stats->max_cardinality);
		if (max_sum_positive >= NumericLimits<int64_t>::Maximum() ||
		    max_sum_negative <= NumericLimits<int64_t>::Minimum()) {
			// sum can potentially exceed int64_t bounds: use hugeint sum
			return nullptr;
		}
		// total sum is guaranteed to fit in a single int64: use int64 sum instead of hugeint sum
		expr.function = GetSumAggregateNoOverflow(internal_type);
	}
	return nullptr;
}

AggregateFunction GetSumAggregate(PhysicalType type) {
	switch (type) {
	case PhysicalType::BOOL: {
		auto function = AggregateFunction::UnaryAggregate<SumState<int64_t>, bool, hugeint_t, IntegerSumOperation>(
		    LogicalType::BOOLEAN, LogicalType::HUGEINT);
		function.SetOrderDependent(AggregateOrderDependent::NOT_ORDER_DEPENDENT);
		return function;
	}
	case PhysicalType::INT16: {
		auto function = AggregateFunction::UnaryAggregate<SumState<int64_t>, int16_t, hugeint_t, IntegerSumOperation>(
		    LogicalType::SMALLINT, LogicalType::HUGEINT);
		function.SetOrderDependent(AggregateOrderDependent::NOT_ORDER_DEPENDENT);
		return function;
	}

	case PhysicalType::INT32: {
		auto function =
		    AggregateFunction::UnaryAggregate<SumState<hugeint_t>, int32_t, hugeint_t, SumToHugeintOperation>(
		        LogicalType::INTEGER, LogicalType::HUGEINT);
		function.SetStatisticsCallback(SumPropagateStats);
		function.SetOrderDependent(AggregateOrderDependent::NOT_ORDER_DEPENDENT);
		return function;
	}
	case PhysicalType::INT64: {
		auto function =
		    AggregateFunction::UnaryAggregate<SumState<hugeint_t>, int64_t, hugeint_t, SumToHugeintOperation>(
		        LogicalType::BIGINT, LogicalType::HUGEINT);
		function.SetStatisticsCallback(SumPropagateStats);
		function.SetOrderDependent(AggregateOrderDependent::NOT_ORDER_DEPENDENT);
		return function;
	}
	case PhysicalType::INT128: {
		auto function =
		    AggregateFunction::UnaryAggregate<SumState<hugeint_t>, hugeint_t, hugeint_t, HugeintSumOperation>(
		        LogicalType::HUGEINT, LogicalType::HUGEINT);
		function.SetOrderDependent(AggregateOrderDependent::NOT_ORDER_DEPENDENT);
		function.SetSerializeCallback(HugeintSumSerialize);
		function.SetDeserializeCallback(HugeintSumDeserialize);
		return function;
	}
	default:
		throw InternalException("Unimplemented sum aggregate");
	}
}

unique_ptr<FunctionData> HugeintSumDeserialize(Deserializer &deserializer, AggregateFunction &function) {
	const auto wide = deserializer.ReadProperty<bool>(1, "wide_accumulator");
	auto arguments = function.arguments;
	auto original_arguments = function.original_arguments;
	auto name = function.name;
	auto return_type = deserializer.Get<const LogicalType &>();
	if (arguments.size() != 1 ||
	    (wide && (arguments[0] != LogicalType::HUGEINT || return_type != LogicalType::HUGEINT))) {
		throw SerializationException("invalid HUGEINT sum accumulator signature");
	}
	function = wide ? WideHugeintSumFunction() : GetSumAggregate(arguments[0].InternalType());
	function.arguments = std::move(arguments);
	function.original_arguments = std::move(original_arguments);
	function.name = std::move(name);
	function.SetReturnType(return_type);
	return nullptr;
}

unique_ptr<FunctionData> BindDecimalSum(ClientContext &context, AggregateFunction &function,
                                        vector<unique_ptr<Expression>> &arguments) {
	auto decimal_type = arguments[0]->return_type;
	function = GetSumAggregate(decimal_type.InternalType());
	function.name = "sum";
	function.arguments[0] = decimal_type;
	function.SetReturnType(LogicalType::DECIMAL(Decimal::MAX_WIDTH_DECIMAL, DecimalType::GetScale(decimal_type)));
	function.SetOrderDependent(AggregateOrderDependent::NOT_ORDER_DEPENDENT);
	return nullptr;
}

struct BignumState {
	bool is_set;
	BignumIntermediate value;
};

struct BignumOperation {
	template <class STATE>
	static void Initialize(STATE &state) {
		state.is_set = false;
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE &state, const INPUT_TYPE &input, AggregateUnaryInput &unary_input,
	                              idx_t count) {
		for (idx_t i = 0; i < count; i++) {
			Operation<INPUT_TYPE, STATE, OP>(state, input, unary_input);
		}
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE &state, const INPUT_TYPE &input, AggregateUnaryInput &unary_input) {
		if (!state.is_set) {
			state.is_set = true;
			state.value.Initialize(unary_input.input.allocator);
		}
		BignumIntermediate rhs(input);
		state.value.AddInPlace(unary_input.input.allocator, rhs);
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE &target, AggregateInputData &input) {
		if (!source.is_set) {
			return;
		}
		if (!target.is_set) {
			target.value.Initialize(input.allocator);
			target.is_set = true;
		}
		target.value.AddInPlace(input.allocator, source.value);
	}

	template <class TARGET_TYPE, class STATE>
	static void Finalize(STATE &state, TARGET_TYPE &target, AggregateFinalizeData &finalize_data) {
		if (!state.is_set) {
			finalize_data.ReturnNull();
		} else {
			target = state.value.ToBignum(finalize_data.input.allocator);
		}
	}

	static bool IgnoreNull() {
		return true;
	}
};

} // namespace

AggregateFunction WideHugeintSumFunction() {
	auto function =
	    AggregateFunction::UnaryAggregate<WideHugeintSumState, hugeint_t, hugeint_t, WideHugeintSumOperation>(
	        LogicalType::HUGEINT, LogicalType::HUGEINT);
	function.name = "sum";
	function.SetOrderDependent(AggregateOrderDependent::NOT_ORDER_DEPENDENT);
	function.SetSerializeCallback(WideHugeintSumSerialize);
	function.SetDeserializeCallback(HugeintSumDeserialize);
	return function;
}

AggregateFunctionSet SumFun::GetFunctions() {
	AggregateFunctionSet sum;
	// decimal
	auto decimal =
	    AggregateFunction({LogicalTypeId::DECIMAL}, LogicalTypeId::DECIMAL, nullptr, nullptr, nullptr, nullptr, nullptr,
	                      FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, BindDecimalSum);
	decimal.SetSerializeCallback(HugeintSumSerialize);
	decimal.SetDeserializeCallback(HugeintSumDeserialize);
	sum.AddFunction(decimal);
	sum.AddFunction(GetSumAggregate(PhysicalType::BOOL));
	sum.AddFunction(GetSumAggregate(PhysicalType::INT16));
	sum.AddFunction(GetSumAggregate(PhysicalType::INT32));
	sum.AddFunction(GetSumAggregate(PhysicalType::INT64));
	sum.AddFunction(GetSumAggregate(PhysicalType::INT128));
	sum.AddFunction(AggregateFunction::UnaryAggregate<SumState<double>, double, double, NumericSumOperation>(
	    LogicalType::DOUBLE, LogicalType::DOUBLE));
	sum.AddFunction(AggregateFunction::UnaryAggregate<BignumState, bignum_t, bignum_t, BignumOperation>(
	    LogicalType::BIGNUM, LogicalType::BIGNUM));
	return sum;
}

AggregateFunction CountIfFun::GetFunction() {
	return GetSumAggregate(PhysicalType::BOOL);
}

AggregateFunctionSet SumNoOverflowFun::GetFunctions() {
	AggregateFunctionSet sum_no_overflow;
	sum_no_overflow.AddFunction(GetSumAggregateNoOverflow(PhysicalType::INT32));
	sum_no_overflow.AddFunction(GetSumAggregateNoOverflow(PhysicalType::INT64));
	sum_no_overflow.AddFunction(GetSumAggregateNoOverflowDecimal());
	return sum_no_overflow;
}

AggregateFunction KahanSumFun::GetFunction() {
	return AggregateFunction::UnaryAggregate<KahanSumState, double, double, KahanSumOperation>(LogicalType::DOUBLE,
	                                                                                           LogicalType::DOUBLE);
}

} // namespace duckdb

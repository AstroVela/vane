#include "duckdb/function/scalar/list_functions.hpp"
#include "duckdb/function/scalar/nested_functions.hpp"
#include "duckdb/common/type_visitor.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression_binder.hpp"
#include "duckdb/function/scalar/list/contains_or_position.hpp"

namespace duckdb {

static unique_ptr<FunctionData> ListSearchBind(ClientContext &, ScalarFunction &,
                                               vector<unique_ptr<Expression>> &arguments) {
	for (auto &argument : arguments) {
		if (TypeVisitor::Contains(argument->return_type, FileLogicalType::IsFile)) {
			throw BinderException("List search functions do not support FILE values");
		}
	}
	return nullptr;
}

template <class RETURN_TYPE, bool FIND_NULLS = false>
static void ListSearchFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	if (result.GetType().id() == LogicalTypeId::SQLNULL) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		ConstantVector::SetNull(result, true);
		return;
	}

	auto target_count = input.size();
	auto &input_list = input.data[0];
	auto &list_child = ListVector::GetEntry(input_list);
	auto &target = input.data[1];

	ListSearchOp<RETURN_TYPE, FIND_NULLS>(input_list, list_child, target, result, target_count);

	if (target_count == 1) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

ScalarFunction ListContainsFun::GetFunction() {
	return ScalarFunction({LogicalType::LIST(LogicalType::TEMPLATE("T")), LogicalType::TEMPLATE("T")},
	                      LogicalType::BOOLEAN, ListSearchFunction<bool>, ListSearchBind);
}

ScalarFunction ListPositionFun::GetFunction() {
	auto fun = ScalarFunction({LogicalType::LIST(LogicalType::TEMPLATE("T")), LogicalType::TEMPLATE("T")},
	                          LogicalType::INTEGER, ListSearchFunction<int32_t, true>, ListSearchBind);
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	return fun;
}

} // namespace duckdb

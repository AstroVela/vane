#include "core_functions/scalar/map_functions.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/parser/expression/bound_expression.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/type_visitor.hpp"
#include "duckdb/function/scalar/nested_functions.hpp"

namespace duckdb {

static void MapFromEntriesFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();

	MapUtil::ReinterpretMap(result, args.data[0], count);
	MapVector::MapConversionVerify(result, count);
	result.Verify(count);

	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static unique_ptr<FunctionData> MapFromEntriesBind(ClientContext &, ScalarFunction &,
                                                   vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(arguments.size() == 1);
	auto &entries_type = arguments[0]->return_type;
	if (entries_type.id() == LogicalTypeId::LIST || entries_type.id() == LogicalTypeId::ARRAY) {
		auto &entry_type = entries_type.id() == LogicalTypeId::LIST ? ListType::GetChildType(entries_type)
		                                                            : ArrayType::GetChildType(entries_type);
		if (entry_type.id() == LogicalTypeId::STRUCT && StructType::GetChildCount(entry_type) == 2 &&
		    TypeVisitor::Contains(StructType::GetChildType(entry_type, 0), FileLogicalType::IsFile)) {
			throw BinderException("map_from_entries does not support FILE keys");
		}
	}
	return nullptr;
}

ScalarFunction MapFromEntriesFun::GetFunction() {
	auto key_type = LogicalType::TEMPLATE("K");
	auto val_type = LogicalType::TEMPLATE("V");
	auto map_type = LogicalType::MAP(key_type, val_type);
	auto row_type = LogicalType::STRUCT({{"", key_type}, {"", val_type}});

	ScalarFunction fun({LogicalType::LIST(row_type)}, map_type, MapFromEntriesFunction, MapFromEntriesBind);
	fun.SetNullHandling(FunctionNullHandling::DEFAULT_NULL_HANDLING);

	fun.SetFallible();
	return fun;
}

} // namespace duckdb

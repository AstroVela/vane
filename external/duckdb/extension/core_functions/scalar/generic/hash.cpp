#include "core_functions/scalar/generic_functions.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/type_visitor.hpp"

namespace duckdb {

static void HashFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	args.Hash(result);
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static unique_ptr<FunctionData> HashBind(ClientContext &, ScalarFunction &, vector<unique_ptr<Expression>> &arguments) {
	for (auto &argument : arguments) {
		if (TypeVisitor::Contains(argument->return_type, FileLogicalType::IsFile)) {
			throw BinderException("hash does not support FILE values");
		}
	}
	return nullptr;
}

ScalarFunction HashFun::GetFunction() {
	auto hash_fun = ScalarFunction({LogicalType::ANY}, LogicalType::HASH, HashFunction, HashBind);
	hash_fun.varargs = LogicalType::ANY;
	hash_fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	return hash_fun;
}

} // namespace duckdb

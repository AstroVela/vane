#include "core_functions/scalar/generic_functions.hpp"

#include "duckdb/main/database.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/exception/parser_exception.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/planner/binder.hpp"

namespace duckdb {

namespace {
struct CurrentSettingBindData : public FunctionData {
	explicit CurrentSettingBindData(Value value_p) : value(std::move(value_p)) {
	}

	Value value;

public:
	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<CurrentSettingBindData>(value);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<CurrentSettingBindData>();
		return Value::NotDistinctFrom(value, other.value);
	}
};

void CurrentSettingFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<CurrentSettingBindData>();
	result.Reference(info.value);
}

unique_ptr<FunctionData> CurrentSettingBind(ScalarFunctionBindInput &input, ScalarFunction &bound_function,
                                            vector<unique_ptr<Expression>> &arguments) {
	auto &context = input.binder.context;
	auto &key_child = arguments[0];
	if (key_child->return_type.id() == LogicalTypeId::UNKNOWN) {
		throw ParameterNotResolvedException();
	}
	if (key_child->return_type.id() != LogicalTypeId::VARCHAR ||
	    key_child->return_type.id() != LogicalTypeId::VARCHAR || !key_child->IsFoldable()) {
		throw ParserException("Key name for current_setting needs to be a constant string");
	}
	Value key_val = ExpressionExecutor::EvaluateScalar(context, *key_child);
	D_ASSERT(key_val.type().id() == LogicalTypeId::VARCHAR);
	if (key_val.IsNull() || StringValue::Get(key_val).empty()) {
		throw ParserException("Key name for current_setting needs to be neither NULL nor empty");
	}

	auto key = StringUtil::Lower(StringValue::Get(key_val));
	Value val;
	if (!context.TryGetCurrentSetting(key, val)) {
		// Client-only queries in an explicit transaction may bind outside the
		// runner path, but the declared read capability must still avoid autoload.
		if (input.binder.IsBindingForRunner() || context.vane_runner_type == "ray") {
			auto message =
			    StringUtil::Format("Runner cannot capture client-context function current_setting for unavailable "
			                       "setting %s; load its extension on the client first",
			                       key);
			if (!context.transaction.IsAutoCommit()) {
				// A failed connection-state read must preserve the client transaction.
				throw BinderException(message);
			}
			throw NotImplementedException(message);
		}
		auto extension_name = Catalog::AutoloadExtensionByConfigName(context, key);
		// If autoloader didn't throw, the config is now available
		context.TryGetCurrentSetting(key, val);
	}

	bound_function.SetReturnType(val.type());
	return make_uniq<CurrentSettingBindData>(val);
}

void CurrentSettingSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data,
                             const ScalarFunction &) {
	serializer.WriteProperty(100, "value", bind_data->Cast<CurrentSettingBindData>().value);
}

unique_ptr<FunctionData> CurrentSettingDeserialize(Deserializer &deserializer, ScalarFunction &function) {
	auto value = deserializer.ReadProperty<Value>(100, "value");
	function.SetReturnType(value.type());
	return make_uniq<CurrentSettingBindData>(std::move(value));
}

} // namespace

ScalarFunction CurrentSettingFun::GetFunction() {
	auto fun = ScalarFunction({LogicalType::VARCHAR}, LogicalType::ANY, CurrentSettingFunction);
	fun.SetBindExtendedCallback(CurrentSettingBind);
	fun.SetSerializeCallback(CurrentSettingSerialize);
	fun.SetDeserializeCallback(CurrentSettingDeserialize);
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	fun.SetClientContextSnapshot();
	return fun;
}

} // namespace duckdb

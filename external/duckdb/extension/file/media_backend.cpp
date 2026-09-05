// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "media_backend.hpp"

#include "duckdb/common/error_data.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/planner/expression.hpp"

namespace duckdb {

static void ValidateMediaBackend(ClientContext &, SetScope, Value &value) {
	if (value.IsNull() || (value.GetValue<string>() != "python" && value.GetValue<string>() != "native")) {
		throw InvalidInputException("Media backend must be 'python' or 'native'");
	}
}

void MediaBackend::RegisterOption(DBConfig &config) {
	for (const auto &domain : {"image", "audio", "video"}) {
		config.AddExtensionOption(string(domain) + "_backend",
		                          "Execution backend selected when a query is bound: python or native",
		                          LogicalType::VARCHAR, Value("python"), ValidateMediaBackend);
	}
}

bool MediaBackend::UseNative(ClientContext &context, const string &domain) {
	Value value;
	if (!context.TryGetCurrentSetting(domain + "_backend", value)) {
		throw InternalException("The FILE extension has not registered %s_backend", domain);
	}
	ValidateMediaBackend(context, SetScope::SESSION, value);
	if (value.GetValue<string>() == "python") {
		return false;
	}
	if (!DatabaseInstance::GetDatabase(context).ExtensionIsLoaded(domain)) {
		throw BinderException("%s_backend='native' requires the %s extension to be loaded", domain, domain);
	}
	return true;
}

unique_ptr<Expression> MediaBackend::BindNative(FunctionBindExpressionInput &input, const string &domain,
                                                const string &function_name) {
	if (!UseNative(input.context, domain)) {
		return nullptr;
	}
	FunctionBinder binder(input.context);
	ErrorData error;
	auto result =
	    binder.BindScalarFunction(DEFAULT_SCHEMA, "native_" + function_name, std::move(input.children), error);
	if (!result) {
		error.Throw();
	}
	return result;
}

} // namespace duckdb

// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/scalar/udf_functions.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/exception.hpp"
#include "duckdb/common/shared_ptr.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/function/scalar_function.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace duckdb {

enum class UDFMode : uint8_t { SCALAR_MAP, RESULT_ONLY_BATCH, ROW_PRESERVING_BATCH };

inline std::pair<bool, string> UDFPayloadStringField(const Value &payload, const string &name) {
	if (payload.IsNull() || payload.type().id() != LogicalTypeId::STRUCT) {
		return std::make_pair(false, string());
	}
	auto &children = StructValue::GetChildren(payload);
	auto child_count = StructType::GetChildCount(payload.type());
	for (idx_t i = 0; i < child_count; i++) {
		if (StructType::GetChildName(payload.type(), i) != name || i >= children.size() || children[i].IsNull()) {
			continue;
		}
		if (children[i].type().id() != LogicalTypeId::VARCHAR) {
			return std::make_pair(false, string());
		}
		return std::make_pair(true, StringValue::Get(children[i]));
	}
	return std::make_pair(false, string());
}

inline UDFMode ClassifyUDFMode(const Value &payload) {
	auto call_mode = UDFPayloadStringField(payload, "call_mode");
	if (!call_mode.first || call_mode.second.empty()) {
		throw InvalidInputException("udf payload requires call_mode");
	}
	if (call_mode.second == "map") {
		return UDFMode::SCALAR_MAP;
	}
	if (call_mode.second == "map_batches_rows") {
		return UDFMode::ROW_PRESERVING_BATCH;
	}
	if (call_mode.second == "map_batches" || call_mode.second == "flat_map") {
		return UDFMode::RESULT_ONLY_BATCH;
	}
	throw NotImplementedException("unsupported udf expression call_mode: %s", call_mode.second);
}

inline bool UDFModePreservesRows(UDFMode mode) {
	return mode == UDFMode::SCALAR_MAP || mode == UDFMode::ROW_PRESERVING_BATCH;
}

inline string NormalizeRunnerType(string runner_type) {
	StringUtil::Trim(runner_type);
	std::transform(runner_type.begin(), runner_type.end(), runner_type.begin(),
	               [](unsigned char c) { return std::tolower(c); });
	if (!runner_type.empty() && runner_type != "local-fast" && runner_type != "local" && runner_type != "ray") {
		throw InvalidInputException("Invalid runner type '%s'. Please use 'local' or 'ray'.", runner_type);
	}
	return runner_type;
}

inline string ResolveRunnerTypeFromEnvironment() {
	const char *env = std::getenv("VANE_RUNNER");
	if (env) {
		auto runner_type = NormalizeRunnerType(string(env));
		if (runner_type.empty()) {
			return "ray";
		}
		return runner_type;
	}
	return "ray";
}

inline string ExpressionUDFExecutionBackendForRunner(const string &runner_type, bool use_actor) {
	auto normalized = NormalizeRunnerType(runner_type);
	if (normalized == "ray") {
		return use_actor ? "ray_actor" : "ray_task";
	}
	return use_actor ? "subprocess_actor" : "subprocess_task";
}

namespace udf_helpers {

LogicalType ResolvePayloadReturnType(const Value &payload);

} // namespace udf_helpers

struct RegisteredUDFFunctionInfo : public ScalarFunctionInfo {
	explicit RegisteredUDFFunctionInfo(Value payload_p, vector<string> input_names_p = {},
	                                   bool allow_named_arguments_p = false)
	    : payload(std::move(payload_p)), input_names(std::move(input_names_p)),
	      allow_named_arguments(allow_named_arguments_p) {
	}

	Value payload;
	//! SQL argument names in positional function-signature order.
	vector<string> input_names;
	//! Named arguments are deliberately opt-in because scalar Python UDFs do not expose a stable SQL name contract.
	bool allow_named_arguments;
};

struct UDFFunctionData : public FunctionData {
	UDFFunctionData(Value payload_p, LogicalType return_type_p, shared_ptr<void> actor_handles_p = nullptr);

	Value payload;
	LogicalType return_type;
	shared_ptr<void> actor_handles;

	unique_ptr<FunctionData> Copy() const override;
	bool Equals(const FunctionData &other) const override;
};

struct UDFFunction {
	static constexpr const char *Name = "udf";
	static constexpr const char *Parameters = "args..., payload";
	static constexpr const char *Description =
	    "Internal UDF placeholder. Planned into a UDF physical operator at execution time.";
	static constexpr const char *Example = "udf(x, payload)";
	static constexpr const char *Categories = "";

	static ScalarFunctionSet GetFunctions();
};

} // namespace duckdb

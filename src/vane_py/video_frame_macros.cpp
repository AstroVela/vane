// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/video_file_functions.hpp"
#include "video_frame_contract.hpp"
#include "duckdb/function/scalar_macro_function.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"

namespace duckdb {
namespace {

static unique_ptr<CreateMacroInfo> FrameMacro(VideoFrameOperation operation) {
	const bool by_index = operation == VideoFrameOperation::FRAME_BY_INDEX;
	const bool keyframes = operation == VideoFrameOperation::KEYFRAMES;
	const bool statistics = operation == VideoFrameOperation::SCAN_STATS;
	vector<string> names;
	vector<LogicalType> types;
	vector<Value> defaults;
	auto parameter = [&](string name, LogicalType type, Value default_value) {
		names.push_back(std::move(name));
		types.push_back(std::move(type));
		defaults.push_back(std::move(default_value));
	};
	// Let the scalar binder enforce the exact VIDEOFILE logical type.
	parameter("file", LogicalType::UNKNOWN, Value());
	if (by_index) {
		parameter("idx", LogicalType::BIGINT, Value());
	} else {
		parameter("start_time", LogicalType::DOUBLE, Value::DOUBLE(0));
		parameter("end_time", LogicalType::DOUBLE, Value(LogicalType::DOUBLE));
		if (!statistics) {
			parameter("width", LogicalType::BIGINT, Value(LogicalType::BIGINT));
			parameter("height", LogicalType::BIGINT, Value(LogicalType::BIGINT));
		}
		if (!keyframes) {
			parameter("is_key_frame", LogicalType::BOOLEAN, Value(LogicalType::BOOLEAN));
		}
		parameter("sample_interval_seconds", LogicalType::DOUBLE, Value(LogicalType::DOUBLE));
	}
	if (!statistics) {
		parameter("on_error", LogicalType::VARCHAR, Value("raise"));
	}
	parameter("max_input_bytes", LogicalType::BIGINT, Value::BIGINT(8 * 1024 * VideoFrameContract::MIB));
	parameter("max_decoded_frames", LogicalType::BIGINT, Value::BIGINT(1000000));
	parameter("max_pixels", LogicalType::BIGINT, Value::BIGINT(VideoFrameContract::MAX_PIXELS));
	if (!statistics) {
		parameter("max_output_bytes", LogicalType::BIGINT, Value::BIGINT(64 * VideoFrameContract::MIB));
		if (!by_index) {
			parameter("max_output_frames", LogicalType::BIGINT, Value::BIGINT(10000));
		}
	}
	if (statistics) {
		parameter("idx", LogicalType::BIGINT, Value(LogicalType::BIGINT));
	}
	parameter("index", LogicalType::BLOB, Value(LogicalType::BLOB));
	vector<unique_ptr<ParsedExpression>> arguments;
	auto column = [&](const char *name) {
		arguments.push_back(make_uniq<ColumnRefExpression>(name));
	};
	auto constant = [&](Value value) {
		arguments.push_back(make_uniq<ConstantExpression>(std::move(value)));
	};
	column("file");
	if (by_index) {
		constant(Value::DOUBLE(0));
		constant(Value(LogicalType::DOUBLE));
		constant(Value(LogicalType::BIGINT));
		constant(Value(LogicalType::BIGINT));
		constant(Value(LogicalType::BOOLEAN));
		constant(Value(LogicalType::DOUBLE));
	} else {
		for (auto name : {"start_time", "end_time"}) {
			column(name);
		}
		if (statistics) {
			constant(Value(LogicalType::BIGINT));
			constant(Value(LogicalType::BIGINT));
		} else {
			column("width");
			column("height");
		}
		if (keyframes) {
			constant(Value::BOOLEAN(true));
		} else {
			column("is_key_frame");
		}
		column("sample_interval_seconds");
	}
	if (statistics) {
		constant(Value("raise"));
	} else {
		column("on_error");
	}
	for (auto name : {"max_input_bytes", "max_decoded_frames", "max_pixels"}) {
		column(name);
	}
	if (statistics) {
		constant(Value::BIGINT(64 * VideoFrameContract::MIB));
	} else {
		column("max_output_bytes");
	}
	if (by_index) {
		constant(Value::BIGINT(1));
		column("idx");
	} else {
		if (statistics) {
			constant(Value::BIGINT(10000));
			column("idx");
		} else {
			column("max_output_frames");
			constant(Value(LogicalType::BIGINT));
		}
	}
	column("index");
	auto expression =
	    make_uniq<FunctionExpression>(string("_vane_") + VideoFrameContract::Name(operation), std::move(arguments));
	auto macro = make_uniq<ScalarMacroFunction>(std::move(expression));
	for (idx_t index = 0; index < names.size(); index++) {
		macro->parameters.push_back(make_uniq<ColumnRefExpression>(names[index]));
		macro->types.push_back(types[index]);
		if (index >= (by_index ? 2 : 1)) {
			macro->default_parameters.insert(
			    make_pair(names[index], make_uniq<ConstantExpression>(std::move(defaults[index]))));
		}
	}
	auto info = make_uniq<CreateMacroInfo>(CatalogType::MACRO_ENTRY);
	info->schema = DEFAULT_SCHEMA;
	info->name = VideoFrameContract::Name(operation);
	info->temporary = true;
	info->internal = true;
	info->macros.push_back(std::move(macro));
	return info;
}

static unique_ptr<CreateMacroInfo> IndexMacro() {
	vector<string> names {"file", "max_input_bytes", "max_decoded_frames", "max_pixels", "max_index_bytes"};
	vector<Value> defaults {Value(), Value::BIGINT(8 * 1024 * VideoFrameContract::MIB), Value::BIGINT(1000000),
	                        Value::BIGINT(VideoFrameContract::MAX_PIXELS), Value::BIGINT(64 * VideoFrameContract::MIB)};
	vector<unique_ptr<ParsedExpression>> arguments;
	for (auto &name : names) {
		arguments.push_back(make_uniq<ColumnRefExpression>(name));
	}
	auto macro =
	    make_uniq<ScalarMacroFunction>(make_uniq<FunctionExpression>("_vane_build_video_index", std::move(arguments)));
	for (idx_t i = 0; i < names.size(); i++) {
		macro->parameters.push_back(make_uniq<ColumnRefExpression>(names[i]));
		macro->types.push_back(i ? LogicalType::BIGINT : LogicalType::UNKNOWN);
		if (i) {
			macro->default_parameters.insert(make_pair(names[i], make_uniq<ConstantExpression>(defaults[i])));
		}
	}
	auto info = make_uniq<CreateMacroInfo>(CatalogType::MACRO_ENTRY);
	info->schema = DEFAULT_SCHEMA;
	info->name = "build_video_index";
	info->temporary = true;
	info->internal = true;
	info->macros.push_back(std::move(macro));
	return info;
}
} // namespace

vector<unique_ptr<CreateMacroInfo>> VideoFileFunctions::GetFrameMacros() {
	vector<unique_ptr<CreateMacroInfo>> result;
	for (auto operation : {VideoFrameOperation::FRAMES, VideoFrameOperation::KEYFRAMES,
	                       VideoFrameOperation::FRAME_BY_INDEX, VideoFrameOperation::SCAN_STATS}) {
		result.push_back(FrameMacro(operation));
	}
	result.push_back(IndexMacro());
	return result;
}
} // namespace duckdb

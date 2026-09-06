// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/video_file_functions.hpp"
#include "datasource_function.hpp"
#include "file_value.hpp"
#include "media_backend.hpp"
#include "vane_python/pybind11/gil_wrapper.hpp"
#include "vane_python/python_objects.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"

#include <cmath>

namespace duckdb {
namespace {

static constexpr uint64_t MIB = 1024 * 1024;

static Value VideoView(const Value &value) {
	if (value.IsNull()) {
		throw BinderException("read_video_frames path lists cannot contain NULL elements");
	}
	auto type = FileLogicalType::Create(FileMediaType::VIDEO);
	if (value.type().id() == LogicalTypeId::VARCHAR) {
		auto result = Value::STRUCT(type, {value, Value(LogicalType::VARCHAR), Value(LogicalType::BIGINT),
		                                   Value(LogicalType::BIGINT), Value(LogicalType::VARCHAR)});
		FileLogicalType::ValidateValue(result, "read_video_frames");
		return result;
	}
	if (!FileLogicalType::IsFile(value.type())) {
		throw BinderException("read_video_frames requires paths, FILE/VIDEOFILE values, or lists of those values");
	}
	auto media = FileLogicalType::GetMediaType(value.type());
	if (media != FileMediaType::UNKNOWN && media != FileMediaType::VIDEO) {
		throw BinderException("read_video_frames requires FILE or VIDEOFILE, not %s", value.type());
	}
	auto result = Value::STRUCT(type, StructValue::GetChildren(value));
	FileLogicalType::ValidateValue(result, "read_video_frames");
	return result;
}

static Value VideoViews(const Value &value) {
	vector<Value> files;
	uint64_t source_bytes = 0;
	auto append = [&](const Value &input) {
		if (files.size() == 100000) {
			throw BinderException("read_video_frames exceeds 100000 FILE views");
		}
		auto file = VideoView(input);
		auto reference = FileReference::FromValue(file, "read_video_frames");
		auto bytes = reference.url.size() + reference.content_type.size() + reference.checksum.size() + 512;
		if (bytes > 64 * MIB - source_bytes) {
			throw BinderException("read_video_frames source metadata exceeds 64 MiB");
		}
		source_bytes += bytes;
		files.push_back(std::move(file));
	};
	if (!value.IsNull()) {
		if (value.type().id() == LogicalTypeId::LIST) {
			for (auto &file : ListValue::GetChildren(value)) {
				append(file);
			}
		} else {
			append(value);
		}
	}
	return Value::LIST(FileLogicalType::Create(FileMediaType::VIDEO), std::move(files));
}

static Value Option(TableFunctionBindInput &input, const string &name, Value default_value) {
	auto entry = input.named_parameters.find(name);
	return entry == input.named_parameters.end() ? std::move(default_value) : entry->second;
}

static vector<Value> ScanParameters(TableFunctionBindInput &input) {
	vector<Value> p;
	p.push_back(VideoViews(input.inputs[0]));
	p.push_back(input.inputs[1]);
	p.push_back(input.inputs[2]);
	p.push_back(Option(input, "start_time", Value::DOUBLE(0)));
	p.push_back(Option(input, "end_time", Value(LogicalType::DOUBLE)));
	p.push_back(input.inputs.size() == 4 ? input.inputs[3]
	                                     : Option(input, "is_key_frame", Value(LogicalType::BOOLEAN)));
	p.push_back(Option(input, "sample_interval_seconds", Value(LogicalType::DOUBLE)));
	p.push_back(Option(input, "max_input_bytes", Value::BIGINT(8 * 1024 * MIB)));
	p.push_back(Option(input, "max_decoded_frames", Value::BIGINT(1000000)));
	p.push_back(Option(input, "max_pixels", Value::BIGINT(32 * MIB)));
	p.push_back(Option(input, "max_partition_bytes", Value::BIGINT(10 * MIB)));
	p.push_back(Option(input, "frame_limit", Value(LogicalType::BIGINT)));
	p.push_back(Option(input, "on_error", Value("raise")));
	p.push_back(Option(input, "read_task_count", Value(LogicalType::BIGINT)));
	const idx_t indices[] = {1, 2, 7, 8, 9, 10};
	const char *names[] = {"image_height",       "image_width", "max_input_bytes",
	                       "max_decoded_frames", "max_pixels",  "max_partition_bytes"};
	const uint64_t maxima[] = {100000, 100000, 16 * 1024 * MIB, 100000000, 32 * MIB, 256 * MIB};
	for (idx_t i = 0; i < 6; i++) {
		auto &value = p[indices[i]];
		if (value.IsNull() || value.GetValue<int64_t>() <= 0 || value.GetValue<uint64_t>() > maxima[i]) {
			throw BinderException("read_video_frames %s must be between 1 and %llu", names[i],
			                      static_cast<unsigned long long>(maxima[i]));
		}
	}
	auto pixels = p[1].GetValue<uint64_t>() * p[2].GetValue<uint64_t>();
	if (pixels > p[9].GetValue<uint64_t>()) {
		throw BinderException("read_video_frames output dimensions exceed max_pixels");
	}
	if (p[3].IsNull()) {
		throw BinderException("read_video_frames start_time cannot be NULL");
	}
	for (auto i : {3, 4, 6}) {
		if (!p[i].IsNull() && (!std::isfinite(p[i].GetValue<double>()) || p[i].GetValue<double>() < 0)) {
			throw BinderException("read_video_frames times must be finite and nonnegative");
		}
	}
	if ((!p[4].IsNull() && p[4].GetValue<double>() < p[3].GetValue<double>()) ||
	    (!p[6].IsNull() && p[6].GetValue<double>() == 0)) {
		throw BinderException("read_video_frames requires end_time >= start_time and a positive sampling interval");
	}
	if ((!p[11].IsNull() && p[11].GetValue<int64_t>() < 0) || (!p[13].IsNull() && p[13].GetValue<int64_t>() <= 0)) {
		throw BinderException("read_video_frames requires nonnegative frame_limit and positive read_task_count");
	}
	if (p[12].IsNull() || (p[12].GetValue<string>() != "raise" && p[12].GetValue<string>() != "skip")) {
		throw BinderException("read_video_frames on_error must be 'raise' or 'skip'");
	}
	uint64_t string_bytes = 0;
	for (auto &file : ListValue::GetChildren(p[0])) {
		auto reference = FileReference::FromValue(file, "read_video_frames");
		string_bytes = MaxValue<uint64_t>(string_bytes, 2 * reference.url.size() + reference.content_type.size() +
		                                                    reference.checksum.size());
	}
	if (pixels * 3 + string_bytes + 512 > p[10].GetValue<uint64_t>()) {
		throw BinderException("read_video_frames row exceeds max_partition_bytes");
	}
	return p;
}

static unique_ptr<TableRef> BindReadVideoFrames(ClientContext &context, TableFunctionBindInput &input) {
	const auto native = MediaBackend::UseNative(context, "video");
	auto parameters = ScanParameters(input);
	auto height = parameters[1].GetValue<uint32_t>();
	auto width = parameters[2].GetValue<uint32_t>();
	string scan_name = "native_read_video_frames";
	if (!native) {
		PythonGILWrapper gil;
		py::list values;
		for (auto &parameter : parameters) {
			values.append(PythonObject::FromValue(parameter, parameter.type(), context.GetClientProperties()));
		}
		auto source = py::module_::import("vane._read_video_frames").attr("_image_video_source")(*values);
		string source_id;
		parameters = SerializeDataSourceParameters(source, source_id);
		scan_name = "datasource_scan";
	}
	vector<unique_ptr<ParsedExpression>> arguments;
	for (auto &parameter : parameters) {
		arguments.push_back(make_uniq<ConstantExpression>(std::move(parameter)));
	}
	auto scan = make_uniq<TableFunctionRef>();
	scan->function = make_uniq<FunctionExpression>(scan_name, std::move(arguments));
	// Only engine-generated identifiers and validated numeric dimensions enter
	// this projection. User paths are ConstantExpressions, never SQL text.
	auto query = "SELECT file.url AS path, video_file(file(file.url, file.content_type, file.position, file.size, "
	             "file.checksum)) AS file, "
	             "frame_index, frame_time, frame_time_base_numerator, frame_time_base_denominator, "
	             "frame_pts, frame_dts, frame_duration, is_key_frame, "
	             "CAST(image(frame.data, frame.width, frame.height, frame.channels, frame.mode) AS IMAGE('RGB', " +
	             std::to_string(height) + ", " + std::to_string(width) + ")) AS data FROM video_scan";
	Parser parser(context.GetParserOptions());
	parser.ParseQuery(query);
	auto statement = unique_ptr_cast<SQLStatement, SelectStatement>(std::move(parser.statements[0]));
	statement->node->Cast<SelectNode>().from_table = std::move(scan);
	return make_uniq<SubqueryRef>(std::move(statement));
}

} // namespace

TableFunctionSet VideoFileFunctions::GetReadFunctions() {
	TableFunction function("read_video_frames", {LogicalType::ANY, LogicalType::BIGINT, LogicalType::BIGINT}, nullptr,
	                       nullptr);
	function.bind_replace = BindReadVideoFrames;
	for (auto name : {"start_time", "end_time", "sample_interval_seconds"}) {
		function.named_parameters[name] = LogicalType::DOUBLE;
	}
	for (auto name : {"max_input_bytes", "max_decoded_frames", "max_pixels", "max_partition_bytes", "frame_limit",
	                  "read_task_count"}) {
		function.named_parameters[name] = LogicalType::BIGINT;
	}
	function.named_parameters["is_key_frame"] = LogicalType::BOOLEAN;
	function.named_parameters["on_error"] = LogicalType::VARCHAR;
	TableFunctionSet result("read_video_frames");
	result.AddFunction(function);
	function.arguments.push_back(LogicalType::BOOLEAN);
	function.named_parameters.erase("is_key_frame");
	result.AddFunction(std::move(function));
	return result;
}

} // namespace duckdb

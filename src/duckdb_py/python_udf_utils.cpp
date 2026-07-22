// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
#include "duckdb_python/pybind11/gil_wrapper.hpp"
//                         DuckDB
//
// duckdb_python/python_udf_utils.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb_python/python_udf_utils.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/function/scalar/udf_functions.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb_python/pybind11/pybind_wrapper.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>

namespace duckdb {

namespace {

constexpr idx_t DEFAULT_UDF_TARGET_MAX_BATCH_BYTES = 134217728;
constexpr const char *UDF_TARGET_MAX_BATCH_BYTES_ENV = "VANE_UDF_TARGET_MAX_BATCH_BYTES";

static Value BuildShapeValue(const vector<idx_t> &shape) {
	vector<Value> shape_values;
	shape_values.reserve(shape.size());
	for (auto dim : shape) {
		shape_values.emplace_back(Value::BIGINT(NumericCast<int64_t>(dim)));
	}
	return Value::LIST(LogicalType::BIGINT, std::move(shape_values));
}

static Value BuildDuckDBOutputSchemaEntry(const string &name, const LogicalType &type) {
	child_list_t<Value> entry_children;
	entry_children.emplace_back("name", Value(name));
	entry_children.emplace_back("kind", Value("duckdb_type"));
	entry_children.emplace_back("type", Value(type.ToString()));
	entry_children.emplace_back("dtype", Value(LogicalType::VARCHAR));
	entry_children.emplace_back("shape", Value(LogicalType::LIST(LogicalType::BIGINT)));
	return Value::STRUCT(std::move(entry_children));
}

static Value BuildTensorOutputSchemaEntry(const string &name, const LogicalType &type) {
	child_list_t<Value> entry_children;
	entry_children.emplace_back("name", Value(name));
	entry_children.emplace_back("kind", Value("tensor"));
	entry_children.emplace_back("type", Value(LogicalType::VARCHAR));
	entry_children.emplace_back("dtype", Value(TensorType::GetChildType(type).ToString()));
	entry_children.emplace_back("shape", BuildShapeValue(TensorType::GetShape(type)));
	return Value::STRUCT(std::move(entry_children));
}

static Value BuildOutputSchemaValue(const vector<string> &output_names, const vector<LogicalType> &output_types) {
	D_ASSERT(output_names.size() == output_types.size());
	vector<Value> schema_entries;
	schema_entries.reserve(output_names.size());
	for (idx_t i = 0; i < output_names.size(); i++) {
		if (TensorType::IsTensor(output_types[i])) {
			schema_entries.push_back(BuildTensorOutputSchemaEntry(output_names[i], output_types[i]));
		} else {
			schema_entries.push_back(BuildDuckDBOutputSchemaEntry(output_names[i], output_types[i]));
		}
	}
	child_list_t<LogicalType> schema_children;
	schema_children.emplace_back("name", LogicalType::VARCHAR);
	schema_children.emplace_back("kind", LogicalType::VARCHAR);
	schema_children.emplace_back("type", LogicalType::VARCHAR);
	schema_children.emplace_back("dtype", LogicalType::VARCHAR);
	schema_children.emplace_back("shape", LogicalType::LIST(LogicalType::BIGINT));
	return Value::LIST(LogicalType::STRUCT(std::move(schema_children)), std::move(schema_entries));
}

static string PythonCallableDisplayName(const py::object &udf) {
	if (!py::hasattr(udf, "__qualname__")) {
		throw InvalidInputException("UDF callable must expose __qualname__");
	}
	auto name = py::cast<string>(py::str(py::getattr(udf, "__qualname__")));
	if (name.empty()) {
		throw InvalidInputException("UDF callable __qualname__ must be non-empty");
	}
	return name;
}

static std::pair<bool, idx_t> ParseOptionalPositiveIdx(const py::object &value, const char *label) {
	if (value.is_none()) {
		return std::make_pair(false, idx_t(0));
	}
	if (py::isinstance<py::bool_>(value)) {
		throw InvalidInputException("%s must be a positive integer", label);
	}
	auto parsed = py::cast<int64_t>(value);
	if (parsed <= 0) {
		throw InvalidInputException("%s must be a positive integer", label);
	}
	return std::make_pair(true, static_cast<idx_t>(parsed));
}

static idx_t ParsePositiveEnvIdx(const char *raw, const char *label) {
	if (!raw || raw[0] == '\0') {
		return idx_t(0);
	}
	errno = 0;
	char *end = nullptr;
	auto parsed = std::strtoll(raw, &end, 10);
	if (errno != 0 || end == raw || (end && *end != '\0') || parsed <= 0) {
		throw InvalidInputException("%s must be a positive integer", label);
	}
	return static_cast<idx_t>(parsed);
}

static idx_t ResolveTargetMaxBatchBytes(const std::pair<bool, idx_t> &target_max_batch_bytes_value) {
	if (target_max_batch_bytes_value.first) {
		return target_max_batch_bytes_value.second;
	}
	auto env_value = ParsePositiveEnvIdx(std::getenv(UDF_TARGET_MAX_BATCH_BYTES_ENV), UDF_TARGET_MAX_BATCH_BYTES_ENV);
	if (env_value > 0) {
		return env_value;
	}
	return DEFAULT_UDF_TARGET_MAX_BATCH_BYTES;
}

static std::pair<bool, double> ParseOptionalNonNegativeDouble(const py::object &value, const char *label) {
	if (value.is_none()) {
		return std::make_pair(false, 0.0);
	}
	if (py::isinstance<py::bool_>(value)) {
		throw InvalidInputException("%s must be a finite non-negative number", label);
	}
	double parsed;
	try {
		parsed = py::cast<double>(value);
	} catch (...) {
		throw InvalidInputException("%s must be a number", label);
	}
	if (!std::isfinite(parsed) || parsed < 0) {
		throw InvalidInputException("%s must be a finite non-negative number", label);
	}
	return std::make_pair(true, parsed);
}

static void ValidateExecutionBackend(const string &backend) {
	if (backend != "subprocess_task" && backend != "subprocess_actor" && backend != "ray_task" &&
	    backend != "ray_actor") {
		throw InvalidInputException(
		    "execution_backend must be one of: subprocess_task, subprocess_actor, ray_task, ray_actor");
	}
}

static bool IsActorExecutionBackend(const string &backend) {
	return backend == "subprocess_actor" || backend == "ray_actor";
}

static bool IsTaskExecutionBackend(const string &backend) {
	return backend == "subprocess_task" || backend == "ray_task";
}

static void ValidateUDFCallableShape(const py::object &udf, const string &execution_backend) {
	auto inspect_module = py::module_::import("inspect");
	const bool is_class = py::cast<bool>(inspect_module.attr("isclass")(udf));
	const bool is_function = py::cast<bool>(inspect_module.attr("isfunction")(udf));
	const bool is_method = py::cast<bool>(inspect_module.attr("ismethod")(udf));

	if (IsTaskExecutionBackend(execution_backend)) {
		if (is_class) {
			throw InvalidInputException("task UDF backends require a function, not a callable class");
		}
		if (!is_function && !is_method) {
			throw InvalidInputException("task UDF backends require a function");
		}
		return;
	}

	if (IsActorExecutionBackend(execution_backend) && !is_class) {
		throw InvalidInputException("actor UDF backends require a callable class");
	}
}

static idx_t ResolveActorNumber(const string &backend, const std::pair<bool, idx_t> &actor_number_value) {
	if (IsActorExecutionBackend(backend)) {
		if (!actor_number_value.first) {
			throw InvalidInputException("actor_number is required for execution_backend='%s'", backend);
		}
		return actor_number_value.second;
	}
	if (actor_number_value.first) {
		throw InvalidInputException(
		    "actor_number is only supported for execution_backend='subprocess_actor' or 'ray_actor'");
	}
	return idx_t(0);
}

static void ValidateStatefulActorContract(const string &backend, const Optional<py::object> &actor_number,
                                          bool stateful) {
	if (!stateful) {
		return;
	}
	if (!IsActorExecutionBackend(backend)) {
		throw InvalidInputException("stateful expression UDFs require an actor execution backend");
	}
	if (actor_number.is_none() || py::isinstance<py::bool_>(actor_number)) {
		throw InvalidInputException(
		    "actor_number must be exactly 1 for stateful vane.cls UDFs; multi-actor state semantics are not defined");
	}
	auto parsed = ParseOptionalPositiveIdx(actor_number, "actor_number");
	if (!parsed.first || parsed.second != 1) {
		throw InvalidInputException(
		    "actor_number must be exactly 1 for stateful vane.cls UDFs; multi-actor state semantics are not defined");
	}
}

static void ValidateActorGpusConfigured(const string &backend, const std::pair<bool, double> &gpus_value,
                                        const char *label) {
	if (IsActorExecutionBackend(backend) && !gpus_value.first) {
		throw InvalidInputException("%s is required for execution_backend='%s'", label, backend);
	}
}

static Value StringListValue(const vector<string> &strings) {
	vector<Value> values;
	values.reserve(strings.size());
	for (auto &item : strings) {
		values.emplace_back(Value(item));
	}
	return Value::LIST(LogicalType::VARCHAR, std::move(values));
}

static Value LogicalTypeStringListValue(const vector<LogicalType> &types) {
	vector<Value> values;
	values.reserve(types.size());
	for (auto &type : types) {
		values.emplace_back(Value(type.ToString()));
	}
	return Value::LIST(LogicalType::VARCHAR, std::move(values));
}

static Value PayloadWithAddedOrUpdatedFields(const Value &payload, child_list_t<Value> fields) {
	if (payload.IsNull() || payload.type().id() != LogicalTypeId::STRUCT) {
		throw InternalException("UDF payload must be a STRUCT");
	}

	auto &children = StructValue::GetChildren(payload);
	auto &payload_type = payload.type();
	vector<bool> consumed(fields.size(), false);
	child_list_t<Value> new_children;

	auto child_count = StructType::GetChildCount(payload_type);
	for (idx_t i = 0; i < child_count; i++) {
		auto child_name = StructType::GetChildName(payload_type, i);
		bool replaced = false;
		for (idx_t field_idx = 0; field_idx < fields.size(); field_idx++) {
			if (child_name != fields[field_idx].first) {
				continue;
			}
			new_children.emplace_back(child_name, fields[field_idx].second);
			consumed[field_idx] = true;
			replaced = true;
			break;
		}
		if (!replaced) {
			new_children.emplace_back(child_name, children[i]);
		}
	}

	for (idx_t field_idx = 0; field_idx < fields.size(); field_idx++) {
		if (!consumed[field_idx]) {
			new_children.emplace_back(fields[field_idx].first, fields[field_idx].second);
		}
	}

	return Value::STRUCT(std::move(new_children));
}
} // namespace

unique_ptr<Expression> LowerRegisteredExpressionUDF(FunctionBindExpressionInput &input) {
	if (!input.bind_data) {
		throw BinderException("registered expression UDF is missing bind payload");
	}
	auto &registered_data = input.bind_data->Cast<UDFFunctionData>();
	if (registered_data.payload.IsNull()) {
		throw BinderException("registered expression UDF payload cannot be NULL");
	}

	vector<unique_ptr<Expression>> children;
	children.reserve(input.children.size() + 1);
	for (auto &child : input.children) {
		children.push_back(std::move(child));
	}
	children.push_back(make_uniq<BoundConstantExpression>(registered_data.payload));

	FunctionBinder binder(input.context);
	ErrorData error;
	auto lowered = binder.BindScalarFunction(DEFAULT_SCHEMA, UDFFunction::Name, std::move(children), error);
	if (!lowered) {
		error.Throw();
	}
	return lowered;
}

Value BuildPythonUDFPayload(
    const string &name, const py::function &udf, const py::object &schema, const shared_ptr<DuckDBPyType> &return_type,
    const string &execution_backend, idx_t default_parallelism, const Optional<py::object> &cpus,
    const Optional<py::object> &gpus, const Optional<py::object> &memory_bytes, const Optional<py::object> &batch_size,
    const Optional<py::object> &output_batch_size, const Optional<py::object> &min_task_batch_size,
    const Optional<py::object> &preserve_compute_batch_boundaries, const Optional<py::object> &actor_number,
    const Optional<py::object> &target_max_batch_bytes, const Optional<py::object> &task_input_max_bytes,
    const Optional<py::object> &output_target_max_bytes, bool side_effects, bool flat_map) {
	PythonGILWrapper acquire;
	ValidateExecutionBackend(execution_backend);
	ValidateUDFCallableShape(udf, execution_backend);
	if (default_parallelism == 0) {
		throw InvalidInputException("default_parallelism must be a positive integer");
	}
	auto actor_number_value = ParseOptionalPositiveIdx(actor_number, "actor_number");
	const bool is_ray_actor = execution_backend == "ray_actor";
	const idx_t resolved_actor_number = ResolveActorNumber(execution_backend, actor_number_value);
	const idx_t actor_pool_size = is_ray_actor ? resolved_actor_number : idx_t(0);
	auto cpus_value = ParseOptionalNonNegativeDouble(cpus, "cpus");
	auto gpus_value = ParseOptionalNonNegativeDouble(gpus, "map_batches(gpus=...)");
	auto memory_bytes_value = ParseOptionalPositiveIdx(memory_bytes, "memory_bytes");
	const bool is_ray_backend = execution_backend == "ray_task" || execution_backend == "ray_actor";
	if (memory_bytes_value.first && !is_ray_backend) {
		throw InvalidInputException("memory_bytes requires a Ray UDF backend");
	}
	if (gpus_value.first && gpus_value.second > 0.0 && execution_backend != "ray_task" &&
	    execution_backend != "ray_actor") {
		throw InvalidInputException("GPU resources require a Ray UDF backend");
	}
	ValidateActorGpusConfigured(execution_backend, gpus_value, "map_batches(gpus=...)");
	auto batch_size_value = ParseOptionalPositiveIdx(batch_size, "batch_size");
	auto output_batch_size_value = ParseOptionalPositiveIdx(output_batch_size, "output_batch_size");
	auto min_task_batch_size_value = ParseOptionalPositiveIdx(min_task_batch_size, "min_task_batch_size");
	const bool preserve_compute_boundaries_value =
	    !preserve_compute_batch_boundaries.is_none() && py::cast<bool>(preserve_compute_batch_boundaries);
	if (min_task_batch_size_value.first) {
		if (!batch_size_value.first) {
			throw InvalidInputException("min_task_batch_size requires batch_size");
		}
		if (min_task_batch_size_value.second < batch_size_value.second) {
			throw InvalidInputException("min_task_batch_size must be at least batch_size");
		}
	}
	auto target_max_batch_bytes_value = ParseOptionalPositiveIdx(target_max_batch_bytes, "target_max_batch_bytes");
	auto task_input_max_bytes_value = ParseOptionalPositiveIdx(task_input_max_bytes, "task_input_max_bytes");
	auto output_target_max_bytes_value = ParseOptionalPositiveIdx(output_target_max_bytes, "output_target_max_bytes");
	const auto resolved_target_max_batch_bytes = ResolveTargetMaxBatchBytes(target_max_batch_bytes_value);
	const auto resolved_task_input_max_bytes =
	    task_input_max_bytes_value.first ? task_input_max_bytes_value.second : resolved_target_max_batch_bytes;
	const auto resolved_output_target_max_bytes =
	    output_target_max_bytes_value.first ? output_target_max_bytes_value.second : resolved_target_max_batch_bytes;

	auto pickle_module = py::module_::import("vane.pickle");
	auto dumps = pickle_module.attr("dumps");
	auto pickled_obj = dumps(udf);
	auto pickled_bytes = py::reinterpret_borrow<py::bytes>(pickled_obj);
	auto pickled_str = py::cast<string>(pickled_bytes);

	vector<string> output_names;
	vector<LogicalType> output_logical_types;
	if (!schema.is_none()) {
		if (!py::isinstance<py::dict>(schema)) {
			throw InvalidInputException("'schema' should be given as a Dict[str, DuckDBType]");
		}
		auto schema_dict = py::reinterpret_borrow<py::dict>(schema);
		for (auto &item : schema_dict) {
			auto name_obj = item.first;
			auto type_obj = item.second;
			auto type = py::cast<shared_ptr<DuckDBPyType>>(type_obj);
			output_names.push_back(std::string(py::str(name_obj)));
			output_logical_types.push_back(type->Type());
		}
	} else if (return_type) {
		output_names.emplace_back("value");
		output_logical_types.push_back(return_type->Type());
	} else {
		throw InvalidInputException("UDF requires schema or return_type");
	}

	child_list_t<Value> children;
	children.emplace_back("payload_version", Value::BIGINT(1));
	auto udf_display_name = PythonCallableDisplayName(udf);
	children.emplace_back("udf_name", Value(udf_display_name));
	children.emplace_back("call_mode", Value(flat_map ? "flat_map" : "map_batches"));
	children.emplace_back("execution_backend", Value(execution_backend));
	children.emplace_back("function_pickle", Value::BLOB_RAW(pickled_str));
	children.emplace_back("function_pickle_size_bytes", Value::BIGINT(NumericCast<int64_t>(pickled_str.size())));
	children.emplace_back("output_schema", BuildOutputSchemaValue(output_names, output_logical_types));
	children.emplace_back("ref_output_types", LogicalTypeStringListValue(output_logical_types));

	if (side_effects) {
		children.emplace_back("side_effects", Value::BOOLEAN(true));
	}
	if (IsActorExecutionBackend(execution_backend)) {
		children.emplace_back("actor_number", Value::BIGINT(static_cast<int64_t>(resolved_actor_number)));
	}
	if (is_ray_actor) {
		children.emplace_back("actor_pool_size", Value::BIGINT(static_cast<int64_t>(actor_pool_size)));
	}
	if (cpus_value.first) {
		children.emplace_back("cpus", Value::DOUBLE(cpus_value.second));
	}
	if (gpus_value.first) {
		children.emplace_back("gpus", Value::DOUBLE(gpus_value.second));
	}
	if (memory_bytes_value.first) {
		children.emplace_back("memory_bytes", Value::BIGINT(static_cast<int64_t>(memory_bytes_value.second)));
	}
	if (batch_size_value.first) {
		children.emplace_back("batch_size", Value::BIGINT(static_cast<int64_t>(batch_size_value.second)));
	}
	if (output_batch_size_value.first) {
		children.emplace_back("output_batch_size", Value::BIGINT(static_cast<int64_t>(output_batch_size_value.second)));
	}
	if (min_task_batch_size_value.first) {
		children.emplace_back("min_task_batch_size",
		                      Value::BIGINT(static_cast<int64_t>(min_task_batch_size_value.second)));
	}
	if (preserve_compute_boundaries_value) {
		children.emplace_back("preserve_compute_batch_boundaries", Value::BOOLEAN(true));
	}
	children.emplace_back("udf_target_max_batch_bytes",
	                      Value::BIGINT(static_cast<int64_t>(resolved_target_max_batch_bytes)));
	children.emplace_back("udf_task_input_max_bytes",
	                      Value::BIGINT(static_cast<int64_t>(resolved_task_input_max_bytes)));
	children.emplace_back("udf_output_target_max_bytes",
	                      Value::BIGINT(static_cast<int64_t>(resolved_output_target_max_bytes)));
	return Value::STRUCT(std::move(children));
}

Value BuildScalarUDFPayload(const string &name, const py::function &udf, const shared_ptr<DuckDBPyType> &return_type,
                            const string &execution_backend, idx_t default_parallelism,
                            const vector<LogicalType> &passthrough_types, const Optional<py::object> &cpus,
                            const Optional<py::object> &gpus, const Optional<py::object> &batch_size,
                            const Optional<py::object> &actor_number, bool side_effects) {
	PythonGILWrapper acquire;
	ValidateExecutionBackend(execution_backend);
	ValidateUDFCallableShape(udf, execution_backend);
	if (!return_type) {
		throw InvalidInputException("map requires return_type");
	}
	if (default_parallelism == 0) {
		throw InvalidInputException("default_parallelism must be a positive integer");
	}
	auto actor_number_value = ParseOptionalPositiveIdx(actor_number, "actor_number");
	const bool is_ray_actor = execution_backend == "ray_actor";
	const idx_t resolved_actor_number = ResolveActorNumber(execution_backend, actor_number_value);
	const idx_t actor_pool_size = is_ray_actor ? resolved_actor_number : idx_t(0);
	auto cpus_value = ParseOptionalNonNegativeDouble(cpus, "cpus");
	auto gpus_value = ParseOptionalNonNegativeDouble(gpus, "map(gpus=...)");
	if (gpus_value.first && gpus_value.second > 0.0 && execution_backend != "ray_task" &&
	    execution_backend != "ray_actor") {
		throw InvalidInputException("GPU resources require a Ray UDF backend");
	}
	ValidateActorGpusConfigured(execution_backend, gpus_value, "map(gpus=...)");
	auto batch_size_value = ParseOptionalPositiveIdx(batch_size, "batch_size");
	const auto resolved_target_max_batch_bytes = ResolveTargetMaxBatchBytes(std::make_pair(false, idx_t(0)));

	auto pickle_module = py::module_::import("vane.pickle");
	auto dumps = pickle_module.attr("dumps");
	auto pickled_obj = dumps(udf);
	auto pickled_bytes = py::reinterpret_borrow<py::bytes>(pickled_obj);
	auto pickled_str = py::cast<string>(pickled_bytes);

	vector<LogicalType> ref_output_logical_types = passthrough_types;
	ref_output_logical_types.push_back(return_type->Type());

	child_list_t<Value> children;
	children.emplace_back("payload_version", Value::BIGINT(1));
	children.emplace_back("udf_name", Value(name));
	children.emplace_back("call_mode", Value("map"));
	children.emplace_back("execution_backend", Value(execution_backend));
	children.emplace_back("function_pickle", Value::BLOB_RAW(pickled_str));
	children.emplace_back("function_pickle_size_bytes", Value::BIGINT(NumericCast<int64_t>(pickled_str.size())));
	children.emplace_back("method_return_type", Value(return_type->Type().ToString()));
	if (side_effects) {
		children.emplace_back("side_effects", Value::BOOLEAN(true));
	}
	if (IsActorExecutionBackend(execution_backend)) {
		children.emplace_back("actor_number", Value::BIGINT(static_cast<int64_t>(resolved_actor_number)));
	}
	if (is_ray_actor) {
		children.emplace_back("actor_pool_size", Value::BIGINT(static_cast<int64_t>(actor_pool_size)));
	}
	if (cpus_value.first) {
		children.emplace_back("cpus", Value::DOUBLE(cpus_value.second));
	}
	if (gpus_value.first) {
		children.emplace_back("gpus", Value::DOUBLE(gpus_value.second));
	}
	if (batch_size_value.first) {
		children.emplace_back("batch_size", Value::BIGINT(static_cast<int64_t>(batch_size_value.second)));
	}
	children.emplace_back("scalar_arg_count", Value::BIGINT(static_cast<int64_t>(passthrough_types.size())));
	children.emplace_back("udf_target_max_batch_bytes",
	                      Value::BIGINT(static_cast<int64_t>(resolved_target_max_batch_bytes)));
	children.emplace_back("udf_task_input_max_bytes",
	                      Value::BIGINT(static_cast<int64_t>(resolved_target_max_batch_bytes)));
	children.emplace_back("udf_output_target_max_bytes",
	                      Value::BIGINT(static_cast<int64_t>(resolved_target_max_batch_bytes)));
	vector<Value> ref_output_type_values;
	ref_output_type_values.reserve(ref_output_logical_types.size());
	for (auto &type : ref_output_logical_types) {
		ref_output_type_values.emplace_back(Value(type.ToString()));
	}
	children.emplace_back("ref_output_types", Value::LIST(LogicalType::VARCHAR, std::move(ref_output_type_values)));

	return Value::STRUCT(std::move(children));
}

Value BuildExpressionScalarUDFPayload(const string &name, const py::function &udf,
                                      const shared_ptr<DuckDBPyType> &return_type, const string &execution_backend,
                                      idx_t default_parallelism, idx_t scalar_arg_count) {
	vector<LogicalType> passthrough_types;
	auto payload = BuildScalarUDFPayload(name, udf, return_type, execution_backend, default_parallelism,
	                                     passthrough_types, py::none(), py::none(), py::none(), py::none(),
	                                     /*side_effects=*/false);

	child_list_t<Value> fields;
	fields.emplace_back("payload_version", Value::BIGINT(1));
	fields.emplace_back("expression_udf", Value::BOOLEAN(true));
	fields.emplace_back("method_return_type", Value(return_type->Type().ToString()));
	fields.emplace_back("scalar_arg_count", Value::BIGINT(NumericCast<int64_t>(scalar_arg_count)));
	return PayloadWithAddedOrUpdatedFields(payload, std::move(fields));
}

Value BuildExpressionMapBatchesUDFPayload(const string &name, const py::function &udf, const py::object &schema,
                                          const string &execution_backend, idx_t default_parallelism,
                                          const vector<string> &input_names, const Optional<py::object> &batch_size,
                                          bool row_preserving, const Optional<py::object> &gpus,
                                          const Optional<py::object> &actor_number, bool stateful) {
	ValidateStatefulActorContract(execution_backend, actor_number, stateful);
	auto payload =
	    BuildPythonUDFPayload(name, udf, schema, shared_ptr<DuckDBPyType>(), execution_backend, default_parallelism,
	                          py::none(), gpus, py::none(), batch_size, py::none(), py::none(), py::none(),
	                          actor_number, py::none(), py::none(), py::none(), /*side_effects=*/stateful,
	                          /*flat_map=*/false);
	auto gpus_value = ParseOptionalNonNegativeDouble(gpus, "map_batches(gpus=...)");
	const bool ray_backend = execution_backend == "ray_task" || execution_backend == "ray_actor";

	child_list_t<Value> fields;
	fields.emplace_back("payload_version", Value::BIGINT(1));
	fields.emplace_back("udf_name", Value(name));
	fields.emplace_back("expression_udf", Value::BOOLEAN(true));
	fields.emplace_back("input_names", StringListValue(input_names));
	fields.emplace_back("row_preserving", Value::BOOLEAN(row_preserving));
	fields.emplace_back("prebatched_input", Value::BOOLEAN(false));
	if (stateful) {
		fields.emplace_back("stateful", Value::BOOLEAN(true));
	}
	if (row_preserving) {
		fields.emplace_back("call_mode", Value("map_batches_rows"));
		fields.emplace_back("streaming_breaker", Value::BOOLEAN(true));
		fields.emplace_back("produce_ray_block_stream", Value::BOOLEAN(ray_backend));
		fields.emplace_back("produce_ref_bundle_output", Value::BOOLEAN(!ray_backend));
		fields.emplace_back("stream_output", Value::BOOLEAN(true));
		if (!ray_backend) {
			fields.emplace_back("streaming_output_mode", Value("local_shm_ref_bundle"));
		}
	} else {
		fields.emplace_back("streaming_breaker", Value::BOOLEAN(true));
		fields.emplace_back("produce_ray_block_stream", Value::BOOLEAN(ray_backend));
		fields.emplace_back("produce_ref_bundle_output", Value::BOOLEAN(!ray_backend));
		if (!ray_backend) {
			fields.emplace_back("streaming_output_mode", Value("local_shm_ref_bundle"));
		}
	}
	return PayloadWithAddedOrUpdatedFields(payload, std::move(fields));
}

Value AddAISQLPayloadMetadata(const Value &payload, const string &provider, const string &model,
                              const string &return_type, const Optional<py::object> &dimensions) {
	child_list_t<Value> fields;
	fields.emplace_back("ai_provider", Value(provider));
	fields.emplace_back("ai_model", Value(model));
	fields.emplace_back("ai_return_type", Value(return_type));
	if (!dimensions.is_none()) {
		auto dimensions_value = ParseOptionalPositiveIdx(dimensions, "dimensions");
		fields.emplace_back("ai_dimensions", Value::BIGINT(NumericCast<int64_t>(dimensions_value.second)));
	} else {
		fields.emplace_back("ai_dimensions", Value(LogicalType::BIGINT));
	}
	return PayloadWithAddedOrUpdatedFields(payload, std::move(fields));
}

} // namespace duckdb

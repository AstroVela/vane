// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_py/vllm_executor.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb_python/vllm_executor.hpp"

#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/arrow/arrow_appender.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/assert.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb_python/arrow/arrow_array_stream.hpp"
#include "duckdb_python/arrow/arrow_export_utils.hpp"
#include "duckdb_python/pybind11/gil_wrapper.hpp"
#include "duckdb_python/pybind11/pybind_wrapper.hpp"
#include "duckdb_python/pybind11/registered_py_object.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/python_objects.hpp"
#include "duckdb/execution/vllm_executor.hpp"
#include "duckdb/common/types/arrow_aux_data.hpp"
#include "duckdb/parallel/interrupt.hpp"

#include <cmath>

namespace duckdb {

namespace {

static py::object RequiredNormalizedOption(const py::dict &normalized, const char *name) {
	auto key = py::str(name);
	if (!normalized.contains(key)) {
		throw InvalidInputException("normalized vllm options are missing '%s'", name);
	}
	return py::reinterpret_borrow<py::object>(normalized[key]);
}

static idx_t NormalizedNonNegativeInteger(const py::dict &normalized, const char *name) {
	auto value = RequiredNormalizedOption(normalized, name);
	if (py::isinstance<py::bool_>(value) || !py::isinstance<py::int_>(value)) {
		throw InvalidInputException("vllm %s must be a non-boolean integer", name);
	}
	auto number = value.cast<int64_t>();
	if (number < 0) {
		throw InvalidInputException("vllm %s must be nonnegative", name);
	}
	return static_cast<idx_t>(number);
}

static double NormalizedUnitInterval(const py::dict &normalized, const char *name) {
	auto value = RequiredNormalizedOption(normalized, name);
	if (py::isinstance<py::bool_>(value) || (!py::isinstance<py::int_>(value) && !py::isinstance<py::float_>(value))) {
		throw InvalidInputException("vllm %s must be a non-boolean number", name);
	}
	auto number = value.cast<double>();
	if (!std::isfinite(number) || number < 0.0 || number > 1.0) {
		throw InvalidInputException("vllm %s must be finite and between 0 and 1", name);
	}
	return number;
}

static py::list ConvertToSingleBatch(vector<LogicalType> &types, vector<string> &names, DataChunk &input,
                                     ClientProperties &options, ClientContext &context) {
	ArrowSchema schema;
	ArrowConverter::ToArrowSchema(&schema, types, names, options);

	py::list single_batch;
	ArrowAppender appender(types, STANDARD_VECTOR_SIZE, options,
	                       ArrowTypeExtensionData::GetExtensionTypes(context, types));
	appender.Append(input, 0, input.size(), input.size());
	auto array = appender.Finalize();
	TransformDuckToArrowChunk(schema, array, single_batch);
	return single_batch;
}

static py::object ConvertDataChunkToPyArrowTable(DataChunk &input, ClientProperties &options, ClientContext &context) {
	auto types = input.GetTypes();
	vector<string> names;
	names.reserve(types.size());
	for (idx_t i = 0; i < types.size(); i++) {
		names.push_back(StringUtil::Format("c%d", i));
	}

	return pyarrow::ToArrowTable(types, names, ConvertToSingleBatch(types, names, input, options, context), options);
}

void AreExtensionsRegistered(const LogicalType &arrow_type, const LogicalType &duckdb_type) {
	if (arrow_type != duckdb_type) {
		if (arrow_type.id() == LogicalTypeId::BLOB && duckdb_type.id() == LogicalTypeId::UUID) {
			throw InvalidConfigurationException(
			    "Mismatch on return type from Arrow object (%s) and DuckDB (%s). It seems that you are using the UUID "
			    "arrow canonical extension, but the same is not yet registered. Make sure to register it first with "
			    "e.g., pa.register_extension_type(UUIDType()). ",
			    arrow_type.ToString(), duckdb_type.ToString());
		}
		if (!arrow_type.IsJSONType() && duckdb_type.IsJSONType()) {
			throw InvalidConfigurationException(
			    "Mismatch on return type from Arrow object (%s) and DuckDB (%s). It seems that you are using the JSON "
			    "arrow canonical extension, but the same is not yet registered. Make sure to register it first with "
			    "e.g., pa.register_extension_type(JSONType()). ",
			    arrow_type.ToString(), duckdb_type.ToString());
		}
	}
}

static unique_ptr<DataChunk> ConvertArrowTableToDataChunk(const py::object &table, ClientContext &context,
                                                          const vector<LogicalType> &expected_types) {
	auto ptr = table.ptr();
	D_ASSERT(py::gil_check());
	py::gil_scoped_release gil;

	auto stream_factory =
	    make_uniq<PythonTableArrowArrayStreamFactory>(ptr, context.GetClientProperties(), PyArrowObjectType::Table);
	auto stream_factory_produce = PythonTableArrowArrayStreamFactory::Produce;
	auto stream_factory_get_schema = PythonTableArrowArrayStreamFactory::GetSchema;

	vector<Value> children;
	children.reserve(3);
	children.push_back(Value::POINTER(CastPointerToValue(stream_factory.get())));
	children.push_back(Value::POINTER(CastPointerToValue(stream_factory_produce)));
	children.push_back(Value::POINTER(CastPointerToValue(stream_factory_get_schema)));

	named_parameter_map_t named_params;
	vector<LogicalType> input_types;
	vector<string> input_names;

	TableFunctionRef empty;
	TableFunction dummy_table_function;
	dummy_table_function.name = "ConvertArrowTableToDataChunk";
	TableFunctionBindInput bind_input(children, named_params, input_types, input_names, nullptr, nullptr,
	                                  dummy_table_function, empty);
	vector<LogicalType> return_types;
	vector<string> return_names;

	auto bind_data = ArrowTableFunction::ArrowScanBind(context, bind_input, return_types, return_names);

	if (!expected_types.empty()) {
		if (return_types.size() != expected_types.size()) {
			throw InvalidInputException("Arrow result column count %d does not match expected %d", return_types.size(),
			                            expected_types.size());
		}
		for (idx_t i = 0; i < return_types.size(); i++) {
			AreExtensionsRegistered(return_types[i], expected_types[i]);
		}
	}

	DataChunk result;
	result.Initialize(context, return_types, STANDARD_VECTOR_SIZE);
	result.SetCardinality(0);

	DataChunk scan_chunk;
	scan_chunk.Initialize(context, return_types, STANDARD_VECTOR_SIZE);

	vector<column_t> column_ids;
	column_ids.reserve(return_types.size());
	for (idx_t i = 0; i < return_types.size(); i++) {
		column_ids.push_back(i);
	}

	TableFunctionInitInput input(bind_data.get(), column_ids, vector<idx_t>(), nullptr);
	auto global_state = ArrowTableFunction::ArrowScanInitGlobal(context, input);
	auto local_state = ArrowTableFunction::ArrowScanInitLocalInternal(context, input, global_state.get());

	TableFunctionInput function_input(bind_data.get(), local_state.get(), global_state.get());
	while (true) {
		scan_chunk.Reset();
		ArrowTableFunction::ArrowScanFunction(context, function_input, scan_chunk);
		if (scan_chunk.size() == 0) {
			break;
		}
		scan_chunk.Flatten();
		result.Append(scan_chunk, true);
	}

	auto output = make_uniq<DataChunk>();
	output->Move(result);
	return output;
}

class VLLMPythonExecutor : public VLLMExecutor {
public:
	explicit VLLMPythonExecutor(py::object executor_p) : executor(make_uniq<RegisteredObject>(std::move(executor_p))) {
	}

	~VLLMPythonExecutor() override {
		if (executor) {
			PythonGILWrapper gil;
			executor.reset();
		}
	}

	void Submit(optional_ptr<const string> prefix, vector<string> prompts, DataChunk &rows,
	            ClientContext &context) override {
		PythonGILWrapper gil;
		auto options = context.GetClientProperties();

		if (expected_types.empty()) {
			expected_types = rows.GetTypes();
		}

		try {
			auto py_rows = ConvertDataChunkToPyArrowTable(rows, options, context);
			py::object py_prefix;
			if (prefix) {
				py_prefix = py::str(*prefix);
			} else {
				py_prefix = py::none();
			}
			executor->obj.attr("submit")(py_prefix, py::cast(prompts), py_rows);
		} catch (const py::error_already_set &ex) {
			throw InvalidInputException("vllm submit failed: %s", ex.what());
		}
	}

	std::pair<bool, VLLMResult> TakeReadyResult(ClientContext &context) override {
		PythonGILWrapper gil;
		try {
			auto output = executor->obj.attr("take_ready_result")();
			if (output.is_none()) {
				return std::make_pair(false, VLLMResult());
			}
			auto tuple = output.cast<py::tuple>();
			if (tuple.size() != 2) {
				throw InvalidInputException("vllm take_ready_result must return a (outputs, rows) tuple");
			}

			vector<string> outputs;
			vector<bool> outputs_validity;
			bool has_nulls = false;
			for (auto item : tuple[0]) {
				if (item.is_none()) {
					outputs.emplace_back();
					outputs_validity.push_back(false);
					has_nulls = true;
				} else {
					outputs.push_back(py::str(item));
					outputs_validity.push_back(true);
				}
			}
			if (!has_nulls) {
				outputs_validity.clear();
			}

			auto rows_chunk = ConvertArrowTableToDataChunk(tuple[1], context, expected_types);
			VLLMResult result;
			result.outputs = std::move(outputs);
			result.outputs_validity = std::move(outputs_validity);
			result.rows = std::move(rows_chunk);
			return std::make_pair(true, std::move(result));
		} catch (const py::error_already_set &ex) {
			throw InvalidInputException("vllm take_ready_result failed: %s", ex.what());
		}
	}

	void FinishedSubmitting(ClientContext &context) override {
		PythonGILWrapper gil;
		try {
			executor->obj.attr("finished_submitting")();
		} catch (const py::error_already_set &ex) {
			throw InvalidInputException("vllm finished_submitting failed: %s", ex.what());
		}
	}

	bool AllTasksFinished(ClientContext &context) override {
		PythonGILWrapper gil;
		try {
			auto finished = executor->obj.attr("all_tasks_finished")().cast<bool>();
			return finished;
		} catch (const py::error_already_set &ex) {
			throw InvalidInputException("vllm all_tasks_finished failed: %s", ex.what());
		}
	}

	VLLMWakeupRegistrationResult RegisterWakeup(InterruptState &interrupt_state) override {
		PythonGILWrapper gil;
		if (!py::hasattr(executor->obj, "register_wakeup_callback")) {
			return VLLMWakeupRegistrationResult::UNSUPPORTED;
		}
		try {
			auto callback = py::cpp_function([interrupt_state]() { interrupt_state.Callback(); });
			auto armed = executor->obj.attr("register_wakeup_callback")(std::move(callback));
			if (!py::isinstance<py::bool_>(armed)) {
				throw InvalidInputException("vllm register_wakeup_callback must return a bool");
			}
			return armed.cast<bool>() ? VLLMWakeupRegistrationResult::ARMED : VLLMWakeupRegistrationResult::READY;
		} catch (const py::error_already_set &ex) {
			throw InvalidInputException("vllm register_wakeup_callback failed: %s", ex.what());
		}
	}

	void WaitForResult(ClientContext &context) override {
		PythonGILWrapper gil;
		if (!executor) {
			return;
		}
		// Shutdown can clear the registered handle after the caller snapshots
		// the outer VLLMExecutor. Keep the Python executor alive for the full
		// blocking call, even if another finalizer shuts the bridge down.
		auto executor_ref = executor->obj;
		try {
			executor_ref.attr("wait_for_result")();
		} catch (const py::error_already_set &ex) {
			throw InvalidInputException("vllm wait_for_result failed: %s", ex.what());
		}
	}

	void Shutdown() override {
		PythonGILWrapper gil;
		if (!executor) {
			return;
		}
		try {
			executor->obj.attr("shutdown")();
			executor.reset();
		} catch (const py::error_already_set &ex) {
			throw InvalidInputException("vllm shutdown failed: %s", ex.what());
		}
	}

private:
	unique_ptr<RegisteredObject> executor;
	vector<LogicalType> expected_types;
};

static unique_ptr<VLLMExecutor> CreatePythonVLLMExecutor(ClientContext &context, const string &model,
                                                         const Value &options, VLLMConfig &config) {
	PythonGILWrapper gil;
	py::object options_obj;
	if (options.IsNull()) {
		options_obj = py::none();
	} else {
		options_obj = PythonObject::FromValue(options, options.type(), context.GetClientProperties());
	}

	py::object module;
	try {
		module = py::module_::import("vane.execution.vllm");
	} catch (const py::error_already_set &ex) {
		throw InvalidInputException("Failed to import vane.execution.vllm: %s", ex.what());
	}

	py::object normalized_obj;
	try {
		normalized_obj = module.attr("normalize_options")(options_obj);
	} catch (const py::error_already_set &ex) {
		throw InvalidInputException("Failed to normalize vllm options: %s", ex.what());
	}
	auto normalized = normalized_obj.cast<py::dict>();

	auto do_prefix_routing = RequiredNormalizedOption(normalized, "do_prefix_routing");
	if (!py::isinstance<py::bool_>(do_prefix_routing)) {
		throw InvalidInputException("vllm do_prefix_routing must be a bool");
	}
	config.do_prefix_routing = do_prefix_routing.cast<bool>();
	config.max_buffer_size = NormalizedNonNegativeInteger(normalized, "max_buffer_size");
	config.min_bucket_size = NormalizedNonNegativeInteger(normalized, "min_bucket_size");
	config.prefix_match_threshold = NormalizedUnitInterval(normalized, "prefix_match_threshold");
	config.load_balance_threshold = NormalizedNonNegativeInteger(normalized, "load_balance_threshold");
	config.inflight_limit = NormalizedNonNegativeInteger(normalized, "inflight_limit");

	auto batch_size_obj = RequiredNormalizedOption(normalized, "batch_size");
	if (batch_size_obj.is_none()) {
		config.batch_size = optional_idx();
	} else {
		if (py::isinstance<py::bool_>(batch_size_obj) || !py::isinstance<py::int_>(batch_size_obj)) {
			throw InvalidInputException("vllm batch_size must be a non-boolean integer or NULL");
		}
		auto batch_size = batch_size_obj.cast<int64_t>();
		if (batch_size < 1) {
			throw InvalidInputException("vllm batch_size must be at least 1 or NULL");
		}
		config.batch_size = optional_idx(static_cast<idx_t>(batch_size));
	}
	config.Validate();

	try {
		py::object executor_obj = module.attr("build_executor")(py::str(model), normalized_obj);
		return make_uniq<VLLMPythonExecutor>(std::move(executor_obj));
	} catch (const py::error_already_set &ex) {
		throw InvalidInputException("Failed to build vllm executor: %s", ex.what());
	}
}

} // namespace

void RegisterVLLMExecutorFactory() {
	SetVLLMExecutorFactory(CreatePythonVLLMExecutor);
}

} // namespace duckdb

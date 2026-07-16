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

namespace duckdb {

namespace {

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

	void WaitForResult(ClientContext &context) override {
		PythonGILWrapper gil;
		try {
			executor->obj.attr("wait_for_result")();
		} catch (const py::error_already_set &ex) {
			throw InvalidInputException("vllm wait_for_result failed: %s", ex.what());
		}
	}

	void Shutdown() override {
		if (!executor) {
			return;
		}
		PythonGILWrapper gil;
		try {
			executor->obj.attr("shutdown")();
			executor.reset();
		} catch (const py::error_already_set &ex) {
			executor.reset();
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
		module = py::module_::import("duckdb.execution.vllm");
	} catch (const py::error_already_set &ex) {
		throw InvalidInputException("Failed to import duckdb.execution.vllm: %s", ex.what());
	}

	py::object normalized_obj;
	try {
		normalized_obj = module.attr("normalize_options")(options_obj);
	} catch (const py::error_already_set &ex) {
		throw InvalidInputException("Failed to normalize vllm options: %s", ex.what());
	}
	auto normalized = normalized_obj.cast<py::dict>();

	config.do_prefix_routing = normalized["do_prefix_routing"].cast<bool>();
	config.max_buffer_size = normalized["max_buffer_size"].cast<idx_t>();
	config.min_bucket_size = normalized["min_bucket_size"].cast<idx_t>();
	config.prefix_match_threshold = normalized["prefix_match_threshold"].cast<double>();
	config.load_balance_threshold = normalized["load_balance_threshold"].cast<idx_t>();
	config.inflight_limit = normalized["inflight_limit"].cast<idx_t>();

	auto batch_size_obj = normalized["batch_size"];
	if (batch_size_obj.is_none()) {
		config.batch_size = optional_idx();
	} else {
		config.batch_size = optional_idx(batch_size_obj.cast<idx_t>());
	}

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

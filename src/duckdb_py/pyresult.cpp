// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#include "duckdb_python/pyrelation.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/pyresult.hpp"
#include "duckdb_python/python_objects.hpp"
#include "duckdb_python/numpy/numpy_type.hpp"
#include "duckdb_python/pybind11/gil_wrapper.hpp"

#include "duckdb_python/arrow/arrow_array_stream.hpp"
#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/arrow/arrow_util.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/arrow/result_arrow_wrapper.hpp"
#include "duckdb/common/types/date.hpp"
#include "duckdb/common/types/hugeint.hpp"
#include "duckdb/common/types/uhugeint.hpp"
#include "duckdb/common/types/time.hpp"
#include "duckdb/common/types/timestamp.hpp"
#include "duckdb/common/types/uuid.hpp"
#include "duckdb_python/numpy/array_wrapper.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/enums/stream_execution_result.hpp"
#include "duckdb_python/arrow/arrow_export_utils.hpp"
#include "duckdb/main/chunk_scan_state/query_result.hpp"
#include "duckdb/common/arrow/arrow_query_result.hpp"

using namespace pybind11::literals;

namespace duckdb {

DuckDBPyResult::DuckDBPyResult(unique_ptr<QueryResult> result_p)
    : DuckDBPyResult(MakeLocalPyResultSource(std::move(result_p))) {
}

DuckDBPyResult::DuckDBPyResult(unique_ptr<DuckDBPyResultSource> source_p) : source(std::move(source_p)) {
	if (!source) {
		throw InternalException("PyResult created without a result object");
	}
}

DuckDBPyResult::~DuckDBPyResult() {
	try {
		D_ASSERT(py::gil_check());
		py::gil_scoped_release gil;
		source.reset();
		current_chunk.reset();
	} catch (...) { // NOLINT
	}
}

const DuckDBPyResultMetadata &DuckDBPyResult::Metadata() const {
	if (!source) {
		throw InvalidInputException("result closed");
	}
	return source->Metadata();
}

ClientProperties DuckDBPyResult::GetClientProperties() {
	return Metadata().client_properties;
}

const vector<string> &DuckDBPyResult::GetNames() {
	return Metadata().names;
}

const vector<LogicalType> &DuckDBPyResult::GetTypes() {
	return Metadata().types;
}

unique_ptr<DataChunk> DuckDBPyResult::FetchChunk() {
	return FetchNext();
}

unique_ptr<DataChunk> DuckDBPyResult::FetchNext(bool raw) {
	if (!source) {
		throw InvalidInputException("result closed");
	}
	row_consumption_started = true;
	auto chunk = source->FetchChunk(raw);
	if (!chunk || chunk->size() == 0) {
		result_closed = true;
	}
	return chunk;
}

Optional<py::tuple> DuckDBPyResult::Fetchone() {
	if (!source) {
		throw InvalidInputException("result closed");
	}
	if (!current_chunk || chunk_offset >= current_chunk->size()) {
		py::gil_scoped_release release;
		current_chunk = FetchNext();
		chunk_offset = 0;
	}

	if (!current_chunk || current_chunk->size() == 0) {
		return py::none();
	}
	auto &metadata = Metadata();
	py::tuple res(metadata.types.size());

	for (idx_t col_idx = 0; col_idx < metadata.types.size(); col_idx++) {
		auto val = current_chunk->data[col_idx].GetValue(chunk_offset);
		if (val.IsNull()) {
			res[col_idx] = py::none();
			continue;
		}
		res[col_idx] = PythonObject::FromValue(val, metadata.types[col_idx], metadata.client_properties);
	}
	chunk_offset++;
	return res;
}

py::list DuckDBPyResult::Fetchmany(idx_t size) {
	py::list res;
	for (idx_t i = 0; i < size; i++) {
		auto fres = Fetchone();
		if (fres.is_none()) {
			break;
		}
		res.append(fres);
	}
	return res;
}

py::list DuckDBPyResult::Fetchall() {
	py::list res;
	while (true) {
		auto fres = Fetchone();
		if (fres.is_none()) {
			break;
		}
		res.append(fres);
	}
	return res;
}

py::dict DuckDBPyResult::FetchNumpy() {
	return FetchNumpyInternal();
}

void DuckDBPyResult::FillNumpy(py::dict &res, idx_t col_idx, NumpyResultConversion &conversion, const char *name) {
	if (Metadata().types[col_idx].id() == LogicalTypeId::ENUM) {
		auto &import_cache = *DuckDBPyConnection::ImportCache();
		auto pandas_categorical = import_cache.pandas.Categorical();
		auto categorical_dtype = import_cache.pandas.CategoricalDtype();
		if (!pandas_categorical || !categorical_dtype) {
			throw InvalidInputException("'pandas' is required for this operation but it was not installed");
		}

		// first we (might) need to create the categorical type
		if (categories_type.find(col_idx) == categories_type.end()) {
			// Equivalent to: pandas.CategoricalDtype(['a', 'b'], ordered=True)
			categories_type[col_idx] = categorical_dtype(categories[col_idx], true);
		}
		// Equivalent to: pandas.Categorical.from_codes(codes=[0, 1, 0, 1], dtype=dtype)
		res[name] = pandas_categorical.attr("from_codes")(conversion.ToArray(col_idx),
		                                                  py::arg("dtype") = categories_type[col_idx]);
		if (!conversion.ToPandas()) {
			res[name] = res[name].attr("to_numpy")();
		}
	} else {
		res[name] = conversion.ToArray(col_idx);
	}
}

static void InsertCategory(const vector<LogicalType> &types, unordered_map<idx_t, py::list> &categories) {
	for (idx_t col_idx = 0; col_idx < types.size(); col_idx++) {
		auto &type = types[col_idx];
		if (type.id() == LogicalTypeId::ENUM) {
			// It's an ENUM type, in addition to converting the codes we must convert the categories
			if (categories.find(col_idx) == categories.end()) {
				auto &categories_list = EnumType::GetValuesInsertOrder(type);
				auto categories_size = EnumType::GetSize(type);
				for (idx_t i = 0; i < categories_size; i++) {
					categories[col_idx].append(py::cast(categories_list.GetValue(i).ToString()));
				}
			}
		}
	}
}

unique_ptr<NumpyResultConversion> DuckDBPyResult::InitializeNumpyConversion(bool pandas) {
	if (!source) {
		throw InvalidInputException("result closed");
	}

	idx_t initial_capacity = STANDARD_VECTOR_SIZE * 2ULL;
	auto known_row_count = source->KnownRowCount();
	if (known_row_count.IsValid()) {
		initial_capacity = known_row_count.GetIndex();
	}

	auto &metadata = Metadata();
	auto conversion =
	    make_uniq<NumpyResultConversion>(metadata.types, initial_capacity, metadata.client_properties, pandas);
	return conversion;
}

py::dict DuckDBPyResult::FetchNumpyInternal(bool stream, idx_t vectors_per_chunk,
                                            unique_ptr<NumpyResultConversion> conversion_p) {
	if (!source) {
		throw InvalidInputException("result closed");
	}
	if (!conversion_p) {
		conversion_p = InitializeNumpyConversion();
	}
	auto &conversion = *conversion_p;

	if (!stream) {
		vectors_per_chunk = NumericLimits<idx_t>::Maximum();
	}

	idx_t count_vec = 0;
	if (current_chunk && chunk_offset < current_chunk->size() && count_vec < vectors_per_chunk) {
		current_chunk->Slice(chunk_offset, current_chunk->size() - chunk_offset);
		conversion.Append(*current_chunk);
		current_chunk.reset();
		chunk_offset = 0;
		count_vec++;
	}
	for (; count_vec < vectors_per_chunk; count_vec++) {
		unique_ptr<DataChunk> chunk;
		{
			D_ASSERT(py::gil_check());
			py::gil_scoped_release release;
			chunk = FetchNext(true);
		}
		if (!chunk || chunk->size() == 0) {
			break;
		}
		conversion.Append(*chunk);
	}
	InsertCategory(Metadata().types, categories);

	// now that we have materialized the result in contiguous arrays, construct the actual NumPy arrays or categorical
	// types
	py::dict res;
	auto &metadata = Metadata();
	auto names = metadata.names;
	QueryResult::DeduplicateColumns(names);
	for (idx_t col_idx = 0; col_idx < metadata.names.size(); col_idx++) {
		auto &name = names[col_idx];
		FillNumpy(res, col_idx, conversion, name.c_str());
	}
	return res;
}

static void ReplaceDFColumn(PandasDataFrame &df, const char *col_name, idx_t idx, const py::handle &new_value) {
	df.attr("drop")("columns"_a = col_name, "inplace"_a = true);
	df.attr("insert")(idx, col_name, new_value, "allow_duplicates"_a = false);
}

// TODO: unify these with an enum/flag to indicate which conversions to do
void DuckDBPyResult::ConvertDateTimeTypes(PandasDataFrame &df, bool date_as_object) const {
	auto names = df.attr("columns").cast<vector<string>>();
	auto &metadata = Metadata();

	for (idx_t i = 0; i < metadata.types.size(); i++) {
		auto column = df.attr("__getitem__")(names[i].c_str());
		if (metadata.types[i] == LogicalType::TIMESTAMP_TZ) {
			// first localize to UTC then convert to timezone_config
			auto utc_local = column.attr("dt").attr("tz_localize")("UTC");
			auto new_value = utc_local.attr("dt").attr("tz_convert")(metadata.client_properties.time_zone);
			// We need to create the column anew because the exact dt changed to a new timezone
			ReplaceDFColumn(df, names[i].c_str(), i, new_value);
		} else if (date_as_object && metadata.types[i] == LogicalType::DATE) {
			// Convert through numpy datetime64[D] to avoid pandas accessor lifetime issues on pandas>=3
			auto numpy_dates = column.attr("to_numpy")("dtype"_a = "datetime64[D]");
			auto new_value = numpy_dates.attr("astype")("object");
			ReplaceDFColumn(df, names[i].c_str(), i, new_value);
		}
	}
}

static py::object ConvertNumpyDtype(py::handle numpy_array) {
	D_ASSERT(py::gil_check());
	auto &import_cache = *DuckDBPyConnection::ImportCache();

	auto dtype = numpy_array.attr("dtype");
	if (!py::isinstance(numpy_array, import_cache.numpy.ma.masked_array())) {
		return dtype;
	}

	auto numpy_type = ConvertNumpyType(dtype);
	switch (numpy_type.type) {
	case NumpyNullableType::BOOL: {
		return import_cache.pandas.BooleanDtype()();
	}
	case NumpyNullableType::UINT_8: {
		return import_cache.pandas.UInt8Dtype()();
	}
	case NumpyNullableType::UINT_16: {
		return import_cache.pandas.UInt16Dtype()();
	}
	case NumpyNullableType::UINT_32: {
		return import_cache.pandas.UInt32Dtype()();
	}
	case NumpyNullableType::UINT_64: {
		return import_cache.pandas.UInt64Dtype()();
	}
	case NumpyNullableType::INT_8: {
		return import_cache.pandas.Int8Dtype()();
	}
	case NumpyNullableType::INT_16: {
		return import_cache.pandas.Int16Dtype()();
	}
	case NumpyNullableType::INT_32: {
		return import_cache.pandas.Int32Dtype()();
	}
	case NumpyNullableType::INT_64: {
		return import_cache.pandas.Int64Dtype()();
	}
	case NumpyNullableType::FLOAT_32:
	case NumpyNullableType::FLOAT_64:
	case NumpyNullableType::FLOAT_16: // there is no pandas.Float16Dtype
	default:
		return dtype;
	}
}

PandasDataFrame DuckDBPyResult::FrameFromNumpy(bool date_as_object, const py::handle &o) {
	D_ASSERT(py::gil_check());
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	auto pandas = import_cache.pandas();
	if (!pandas) {
		throw InvalidInputException("'pandas' is required for this operation but it was not installed");
	}

	py::object items = o.attr("items")();
	for (const py::handle &item : items) {
		// Each item is a tuple of (key, value)
		auto key_value = py::cast<py::tuple>(item);
		py::handle key = key_value[0];   // Access the first element (key)
		py::handle value = key_value[1]; // Access the second element (value)

		auto dtype = ConvertNumpyDtype(value);
		if (py::isinstance(value, import_cache.numpy.ma.masked_array())) {
			// o[key] = pd.Series(value.filled(pd.NA), dtype=dtype)
			auto series = pandas.attr("Series")(value.attr("data"), py::arg("dtype") = dtype);
			series.attr("__setitem__")(value.attr("mask"), import_cache.pandas.NA());
			o.attr("__setitem__")(key, series);
		}
	}

	PandasDataFrame df = py::cast<PandasDataFrame>(pandas.attr("DataFrame").attr("from_dict")(o));
	// Convert TZ and (optionally) Date types
	ConvertDateTimeTypes(df, date_as_object);

	auto names = df.attr("columns").cast<vector<string>>();
	D_ASSERT(Metadata().types.size() == names.size());
	return df;
}

PandasDataFrame DuckDBPyResult::FetchDF(bool date_as_object) {
	auto conversion = InitializeNumpyConversion(true);
	return FrameFromNumpy(date_as_object, FetchNumpyInternal(false, 1, std::move(conversion)));
}

PandasDataFrame DuckDBPyResult::FetchDFChunk(idx_t num_of_vectors, bool date_as_object) {
	auto conversion = InitializeNumpyConversion(true);
	return FrameFromNumpy(date_as_object, FetchNumpyInternal(true, num_of_vectors, std::move(conversion)));
}

py::dict DuckDBPyResult::FetchPyTorch() {
	auto result_dict = FetchNumpyInternal();
	auto from_numpy = py::module::import("torch").attr("from_numpy");
	for (auto &item : result_dict) {
		result_dict[item.first] = from_numpy(item.second);
	}
	return result_dict;
}

py::dict DuckDBPyResult::FetchTF() {
	auto result_dict = FetchNumpyInternal();
	auto convert_to_tensor = py::module::import("tensorflow").attr("convert_to_tensor");
	for (auto &item : result_dict) {
		result_dict[item.first] = convert_to_tensor(item.second);
	}
	return result_dict;
}

duckdb::pyarrow::Table DuckDBPyResult::FetchArrowTable(idx_t rows_per_batch, bool to_polars) {
	if (!source) {
		throw InvalidInputException("There is no query result");
	}
	auto names = Metadata().names;
	if (to_polars) {
		QueryResult::DeduplicateColumns(names);
	}

	auto reader = FetchRecordBatchReader(rows_per_batch);
	py::object arrow_table = reader.attr("read_all")();
	if (to_polars) {
		arrow_table = arrow_table.attr("rename_columns")(names);
	}
	return py::cast<duckdb::pyarrow::Table>(arrow_table);
}

ArrowArrayStream DuckDBPyResult::FetchArrowArrayStream(idx_t rows_per_batch) {
	if (!source) {
		throw InvalidInputException("There is no query result");
	}
	if (row_consumption_started || current_chunk) {
		throw InvalidInputException("Cannot switch a partially consumed row result to an Arrow stream");
	}
	auto stream = source->TakeArrowStream(rows_per_batch);
	source.reset();
	return stream;
}

duckdb::pyarrow::RecordBatchReader DuckDBPyResult::FetchRecordBatchReader(idx_t rows_per_batch) {
	if (!source) {
		throw InvalidInputException("There is no query result");
	}
	PythonGILWrapper acquire;
	auto pyarrow_lib_module = py::module::import("pyarrow").attr("lib");
	auto record_batch_reader_func = pyarrow_lib_module.attr("RecordBatchReader").attr("_import_from_c");
	auto stream = FetchArrowArrayStream(rows_per_batch);
	try {
		py::object record_batch_reader = record_batch_reader_func((uint64_t)&stream); // NOLINT
		return py::cast<duckdb::pyarrow::RecordBatchReader>(record_batch_reader);
	} catch (...) {
		if (stream.release) {
			stream.release(&stream);
		}
		throw;
	}
}

// Destructor for capsules that own a heap-allocated ArrowArrayStream (slow path).
static void ArrowArrayStreamPyCapsuleDestructor(PyObject *object) {
	auto data = PyCapsule_GetPointer(object, "arrow_array_stream");
	if (!data) {
		return;
	}
	auto stream = reinterpret_cast<ArrowArrayStream *>(data);
	if (stream->release) {
		stream->release(stream);
	}
	delete stream;
}

py::object DuckDBPyResult::FetchArrowCapsule(idx_t rows_per_batch) {
	auto stream_p = FetchArrowArrayStream(rows_per_batch);
	auto stream = new ArrowArrayStream();
	*stream = stream_p;
	return py::capsule(stream, "arrow_array_stream", ArrowArrayStreamPyCapsuleDestructor);
}

py::list DuckDBPyResult::GetDescription(const vector<string> &names, const vector<LogicalType> &types) {
	py::list desc;

	for (idx_t col_idx = 0; col_idx < names.size(); col_idx++) {
		auto py_name = py::str(names[col_idx]);
		auto py_type = DuckDBPyType(types[col_idx]);
		desc.append(py::make_tuple(py_name, py_type, py::none(), py::none(), py::none(), py::none(), py::none()));
	}
	return desc;
}

void DuckDBPyResult::Close() {
	current_chunk.reset();
	chunk_offset = 0;
	if (source) {
		source->Close();
		source.reset();
	}
}

bool DuckDBPyResult::IsClosed() const {
	return result_closed;
}

} // namespace duckdb

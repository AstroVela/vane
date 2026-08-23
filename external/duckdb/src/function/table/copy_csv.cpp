#include "duckdb/common/bind_helpers.hpp"
#include "duckdb/common/csv_writer.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/multi_file/multi_file_function.hpp"
#include "duckdb/common/multi_file/multi_file_reader.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/serializer/write_stream.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/string_type.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/operator/csv_scanner/csv_multi_file_info.hpp"
#include "duckdb/execution/operator/csv_scanner/sniffer/csv_sniffer.hpp"
#include "duckdb/function/copy_function.hpp"
#include "duckdb/function/scalar/string_functions.hpp"
#include "duckdb/function/table/read_csv.hpp"
#include "duckdb/parser/expression/bound_expression.hpp"
#include "duckdb/parser/expression/cast_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/parsed_data/copy_info.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_binder.hpp"

namespace duckdb {

void AreOptionsEqual(char str_1, char str_2, const string &name_str_1, const string &name_str_2) {
	if (str_1 == '\0' || str_2 == '\0') {
		return;
	}
	if (str_1 == str_2) {
		throw BinderException("%s must not appear in the %s specification and vice versa", name_str_1, name_str_2);
	}
}

void SubstringDetection(char str_1, string &str_2, const string &name_str_1, const string &name_str_2) {
	if (str_1 == '\0' || str_2.empty()) {
		return;
	}
	if (str_2.find(str_1) != string::npos) {
		throw BinderException("%s must not appear in the %s specification and vice versa", name_str_1, name_str_2);
	}
}

void StringDetection(const string &str_1, const string &str_2, const string &name_str_1, const string &name_str_2) {
	if (str_1.empty() || str_2.empty()) {
		return;
	}
	if (str_2.find(str_1) != string::npos) {
		throw BinderException("%s must not appear in the %s specification and vice versa", name_str_1, name_str_2);
	}
}

//===--------------------------------------------------------------------===//
// Bind
//===--------------------------------------------------------------------===//

void BaseCSVData::Finalize() {
	auto delimiter_string = options.dialect_options.state_machine_options.delimiter.GetValue();

	// quote and delimiter must not be substrings of each other
	SubstringDetection(options.dialect_options.state_machine_options.quote.GetValue(), delimiter_string, "QUOTE",
	                   "DELIMITER");

	// escape and delimiter must not be substrings of each other
	SubstringDetection(options.dialect_options.state_machine_options.escape.GetValue(), delimiter_string, "ESCAPE",
	                   "DELIMITER");

	// escape and quote must not be substrings of each other (but can be the same)
	if (options.dialect_options.state_machine_options.quote != options.dialect_options.state_machine_options.escape) {
		AreOptionsEqual(options.dialect_options.state_machine_options.quote.GetValue(),
		                options.dialect_options.state_machine_options.escape.GetValue(), "QUOTE", "ESCAPE");
	}

	// comment and quote must not be substrings of each other
	AreOptionsEqual(options.dialect_options.state_machine_options.comment.GetValue(),
	                options.dialect_options.state_machine_options.quote.GetValue(), "COMMENT", "QUOTE");

	// delimiter and comment must not be substrings of each other
	SubstringDetection(options.dialect_options.state_machine_options.comment.GetValue(), delimiter_string, "COMMENT",
	                   "DELIMITER");

	// quote and delimiter must not be substrings of each other
	SubstringDetection(options.thousands_separator, options.decimal_separator, "THOUSANDS", "DECIMAL_SEPARATOR");

	// null string and delimiter must not be substrings of each other
	for (auto &null_str : options.null_str) {
		if (!null_str.empty()) {
			StringDetection(options.dialect_options.state_machine_options.delimiter.GetValue(), null_str, "DELIMITER",
			                "NULL");

			// quote and nullstr must not be substrings of each other
			SubstringDetection(options.dialect_options.state_machine_options.quote.GetValue(), null_str, "QUOTE",
			                   "NULL");

			// Validate the nullstr against the escape character
			const char escape = options.dialect_options.state_machine_options.escape.GetValue();
			// Allow nullstr to be escape character + some non-special character, e.g., "\N" (MySQL default).
			// In this case, only unquoted occurrences of the nullstr will be recognized as null values.
			if (options.dialect_options.state_machine_options.strict_mode == false && null_str.size() == 2 &&
			    null_str[0] == escape && null_str[1] != '\0') {
				continue;
			}
			SubstringDetection(escape, null_str, "ESCAPE", "NULL");
		}
	}

	if (!options.prefix.empty() || !options.suffix.empty()) {
		if (options.prefix.empty() || options.suffix.empty()) {
			throw BinderException("COPY ... (FORMAT CSV) must have both PREFIX and SUFFIX, or none at all");
		}
		if (options.dialect_options.header.GetValue()) {
			throw BinderException("COPY ... (FORMAT CSV)'s HEADER cannot be combined with PREFIX/SUFFIX");
		}
	}
}

unique_ptr<FunctionData> WriteCSVData::Copy() const {
	auto result = make_uniq<WriteCSVData>(options.name_list);
	result->column_ids = column_ids;
	result->options = options;
	result->filename_col_idx = filename_col_idx;
	result->hive_partition_col_idx = hive_partition_col_idx;
	result->flush_size = flush_size;
	result->sql_types = sql_types;
	// Bound cast expressions can contain connection-owned function state. Copies
	// intentionally rebuild them from sql_types in the execution context.
	return std::move(result);
}

static vector<unique_ptr<Expression>> CreateCastExpressions(WriteCSVData &bind_data, ClientContext &context,
                                                            const vector<string> &names,
                                                            const vector<LogicalType> &sql_types) {
	auto &options = bind_data.options;
	auto &formats = options.write_date_format;

	bool has_dateformat = !formats[LogicalTypeId::DATE].IsNull();
	bool has_timestampformat = !formats[LogicalTypeId::TIMESTAMP].IsNull();

	// Create a binder
	auto binder = Binder::CreateBinder(context);

	auto &bind_context = binder->bind_context;
	auto table_index = binder->GenerateTableIndex();
	bind_context.AddGenericBinding(table_index, "copy_csv", names, sql_types);

	// Create the ParsedExpressions (cast, strftime, etc..)
	vector<unique_ptr<ParsedExpression>> unbound_expressions;
	for (idx_t i = 0; i < sql_types.size(); i++) {
		auto &type = sql_types[i];
		auto &name = names[i];

		bool is_timestamp = type.id() == LogicalTypeId::TIMESTAMP || type.id() == LogicalTypeId::TIMESTAMP_TZ;
		if (has_dateformat && type.id() == LogicalTypeId::DATE) {
			// strftime(<name>, 'format')
			vector<unique_ptr<ParsedExpression>> children;
			children.push_back(make_uniq<BoundExpression>(make_uniq<BoundReferenceExpression>(name, type, i)));
			children.push_back(make_uniq<ConstantExpression>(formats[LogicalTypeId::DATE]));
			auto func = make_uniq_base<ParsedExpression, FunctionExpression>("strftime", std::move(children));
			unbound_expressions.push_back(std::move(func));
		} else if (has_timestampformat && is_timestamp) {
			// strftime(<name>, 'format')
			vector<unique_ptr<ParsedExpression>> children;
			children.push_back(make_uniq<BoundExpression>(make_uniq<BoundReferenceExpression>(name, type, i)));
			children.push_back(make_uniq<ConstantExpression>(formats[LogicalTypeId::TIMESTAMP]));
			auto func = make_uniq_base<ParsedExpression, FunctionExpression>("strftime", std::move(children));
			unbound_expressions.push_back(std::move(func));
		} else {
			// CAST <name> AS VARCHAR
			auto column = make_uniq<BoundExpression>(make_uniq<BoundReferenceExpression>(name, type, i));
			auto expr = make_uniq_base<ParsedExpression, CastExpression>(LogicalType::VARCHAR, std::move(column));
			unbound_expressions.push_back(std::move(expr));
		}
	}

	// Create an ExpressionBinder, bind the Expressions
	vector<unique_ptr<Expression>> expressions;
	ExpressionBinder expression_binder(*binder, context);
	expression_binder.target_type = LogicalType::VARCHAR;
	for (auto &expr : unbound_expressions) {
		expressions.push_back(expression_binder.Bind(expr));
	}

	return expressions;
}

static unique_ptr<FunctionData> WriteCSVBind(ClientContext &context, CopyFunctionBindInput &input,
                                             const vector<string> &names, const vector<LogicalType> &sql_types) {
	auto bind_data = make_uniq<WriteCSVData>(names);

	// check all the options in the copy info
	for (auto &option : input.info.options) {
		auto loption = StringUtil::Lower(option.first);
		auto &set = option.second;
		bind_data->options.SetWriteOption(loption, ConvertVectorToValue(set));
	}
	// verify the parsed options
	if (bind_data->options.force_quote.empty()) {
		// no FORCE_QUOTE specified: initialize to false
		bind_data->options.force_quote.resize(names.size(), false);
	}
	bind_data->Finalize();

	switch (bind_data->options.compression) {
	case FileCompressionType::GZIP:
		if (!IsFileCompressed(input.file_extension, FileCompressionType::GZIP)) {
			input.file_extension += CompressionExtensionFromType(FileCompressionType::GZIP);
		}
		break;
	case FileCompressionType::ZSTD:
		if (!IsFileCompressed(input.file_extension, FileCompressionType::ZSTD)) {
			input.file_extension += CompressionExtensionFromType(FileCompressionType::ZSTD);
		}
		break;
	default:
		break;
	}

	auto expressions = CreateCastExpressions(*bind_data, context, names, sql_types);
	bind_data->cast_expressions = std::move(expressions);
	bind_data->sql_types = sql_types;

	return std::move(bind_data);
}

static void WriteCSVSerialize(Serializer &serializer, const FunctionData &bind_data_p, const CopyFunction &function) {
	auto &bind_data = bind_data_p.Cast<WriteCSVData>();
	if (bind_data.options.name_list.size() != bind_data.sql_types.size()) {
		throw SerializationException("CSV COPY bind data has %llu names but %llu types",
		                             bind_data.options.name_list.size(), bind_data.sql_types.size());
	}
	serializer.WriteProperty(100, "options", SerializedCSVReaderOptions(bind_data.options, MultiFileOptions()));
	serializer.WriteProperty(101, "sql_types", bind_data.sql_types);
	serializer.WriteProperty(102, "flush_size", bind_data.flush_size);
}

static unique_ptr<FunctionData> WriteCSVDeserialize(Deserializer &deserializer, CopyFunction &function) {
	auto serialized_options = deserializer.ReadProperty<SerializedCSVReaderOptions>(100, "options");
	auto sql_types = deserializer.ReadProperty<vector<LogicalType>>(101, "sql_types");
	auto flush_size = deserializer.ReadProperty<idx_t>(102, "flush_size");
	if (serialized_options.options.name_list.size() != sql_types.size()) {
		throw SerializationException("CSV COPY bind data has %llu names but %llu types",
		                             serialized_options.options.name_list.size(), sql_types.size());
	}
	if (flush_size == 0) {
		throw SerializationException("CSV COPY bind data has a zero flush size");
	}

	auto result = make_uniq<WriteCSVData>(serialized_options.options.name_list);
	result->options = std::move(serialized_options.options);
	result->sql_types = std::move(sql_types);
	result->flush_size = flush_size;
	result->Finalize();
	return std::move(result);
}

static void CSVListCopyOptions(ClientContext &context, CopyOptionsInput &input) {
	auto &copy_options = input.options;
	copy_options["auto_detect"] = CopyOption(LogicalType::BOOLEAN, CopyOptionMode::READ_ONLY);
	copy_options["sample_size"] = CopyOption(LogicalType::BIGINT, CopyOptionMode::READ_ONLY);
	copy_options["skip"] = CopyOption(LogicalType::BIGINT, CopyOptionMode::READ_ONLY);
	copy_options["max_line_size"] = CopyOption(LogicalType::BIGINT, CopyOptionMode::READ_ONLY);
	copy_options["maximum_line_size"] = CopyOption(LogicalType::BIGINT, CopyOptionMode::READ_ONLY);
	copy_options["ignore_errors"] = CopyOption(LogicalType::BOOLEAN, CopyOptionMode::READ_ONLY);
	copy_options["buffer_size"] = CopyOption(LogicalType::BIGINT, CopyOptionMode::READ_ONLY);
	copy_options["decimal_separator"] = CopyOption(LogicalType::VARCHAR, CopyOptionMode::READ_ONLY);
	copy_options["null_padding"] = CopyOption(LogicalType::BOOLEAN, CopyOptionMode::READ_ONLY);
	copy_options["parallel"] = CopyOption(LogicalType::BOOLEAN, CopyOptionMode::READ_ONLY);
	copy_options["allow_quoted_nulls"] = CopyOption(LogicalType::BOOLEAN, CopyOptionMode::READ_ONLY);
	copy_options["store_rejects"] = CopyOption(LogicalType::BOOLEAN, CopyOptionMode::READ_ONLY);
	copy_options["force_not_null"] = CopyOption(LogicalType::ANY, CopyOptionMode::READ_ONLY);
	copy_options["rejects_table"] = CopyOption(LogicalType::VARCHAR, CopyOptionMode::READ_ONLY);
	copy_options["rejects_scan"] = CopyOption(LogicalType::VARCHAR, CopyOptionMode::READ_ONLY);
	copy_options["rejects_limit"] = CopyOption(LogicalType::BIGINT, CopyOptionMode::READ_ONLY);
	copy_options["encoding"] = CopyOption(LogicalType::VARCHAR, CopyOptionMode::READ_ONLY);
	copy_options["thousands"] = CopyOption(LogicalType::VARCHAR, CopyOptionMode::READ_ONLY);

	copy_options["force_quote"] = CopyOption(LogicalType::ANY, CopyOptionMode::WRITE_ONLY);
	copy_options["prefix"] = CopyOption(LogicalType::VARCHAR, CopyOptionMode::WRITE_ONLY);
	copy_options["suffix"] = CopyOption(LogicalType::VARCHAR, CopyOptionMode::WRITE_ONLY);

	copy_options["new_line"] = CopyOption(LogicalType::VARCHAR, CopyOptionMode::READ_WRITE);
	copy_options["date_format"] = CopyOption(LogicalType::VARCHAR, CopyOptionMode::READ_WRITE);
	copy_options["dateformat"] = CopyOption(LogicalType::VARCHAR, CopyOptionMode::READ_WRITE);
	copy_options["timestamp_format"] = CopyOption(LogicalType::VARCHAR, CopyOptionMode::READ_WRITE);
	copy_options["timestampformat"] = CopyOption(LogicalType::VARCHAR, CopyOptionMode::READ_WRITE);
	copy_options["quote"] = CopyOption(LogicalType::VARCHAR, CopyOptionMode::READ_WRITE);
	copy_options["comment"] = CopyOption(LogicalType::VARCHAR, CopyOptionMode::READ_WRITE);
	copy_options["delim"] = CopyOption(LogicalType::VARCHAR, CopyOptionMode::READ_WRITE);
	copy_options["delimiter"] = CopyOption(LogicalType::VARCHAR, CopyOptionMode::READ_WRITE);
	copy_options["sep"] = CopyOption(LogicalType::VARCHAR, CopyOptionMode::READ_WRITE);
	copy_options["separator"] = CopyOption(LogicalType::VARCHAR, CopyOptionMode::READ_WRITE);
	copy_options["escape"] = CopyOption(LogicalType::VARCHAR, CopyOptionMode::READ_WRITE);
	copy_options["header"] = CopyOption(LogicalType::BOOLEAN, CopyOptionMode::READ_WRITE);
	copy_options["nullstr"] = CopyOption(LogicalType::ANY, CopyOptionMode::READ_WRITE);
	copy_options["null"] = CopyOption(LogicalType::ANY, CopyOptionMode::READ_WRITE);
	copy_options["compression"] = CopyOption(LogicalType::VARCHAR, CopyOptionMode::READ_WRITE);
	copy_options["strict_mode"] = CopyOption(LogicalType::BOOLEAN, CopyOptionMode::READ_WRITE);
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
struct LocalWriteCSVData : public LocalFunctionData {
public:
	LocalWriteCSVData(ClientContext &context, vector<unique_ptr<Expression>> expressions_p, const idx_t &flush_size)
	    : expressions(std::move(expressions_p)), executor(context, expressions),
	      writer_local_state(context, flush_size) {
	}

public:
	//! Own the expressions used by the executor. Deserialized writer binds defer
	//! binding these expressions until this real execution context exists.
	vector<unique_ptr<Expression>> expressions;
	//! Used to execute the expressions that transform input -> string
	ExpressionExecutor executor;
	//! A chunk with VARCHAR columns to cast intermediates into
	DataChunk cast_chunk;
	//! Local state for the CSV writer
	CSVWriterState writer_local_state;
};

struct GlobalWriteCSVData : public GlobalFunctionData {
	GlobalWriteCSVData(CSVReaderOptions &options, FileSystem &fs_p, const string &file_path_p,
	                   FileCompressionType compression)
	    : writer(options, fs_p, file_path_p, compression) {
	}

	idx_t FileSize() {
		return writer.FileSize();
	}

	unique_ptr<CSVWriterState> GetLocalState(ClientContext &context, const idx_t flush_size) {
		{
			lock_guard<mutex> guard(local_state_lock);
			if (!local_states.empty()) {
				auto result = std::move(local_states.back());
				local_states.pop_back();
				return result;
			}
		}
		auto result = make_uniq<CSVWriterState>(context, flush_size);
		result->require_manual_flush = true;
		return result;
	}

	void StoreLocalState(unique_ptr<CSVWriterState> lstate) {
		lock_guard<mutex> guard(local_state_lock);
		lstate->Reset();
		local_states.push_back(std::move(lstate));
	}

	void AddRows(idx_t count) {
		rows_written += count;
	}

	void SetWrittenStatistics(CopyFunctionFileStatistics &statistics) {
		lock_guard<mutex> guard(statistics_lock);
		written_statistics = statistics;
	}

	void FinalizeWrittenStatistics(idx_t file_size) {
		lock_guard<mutex> guard(statistics_lock);
		if (!written_statistics) {
			return;
		}
		written_statistics->row_count = rows_written.load();
		written_statistics->file_size_bytes = file_size;
	}

	CSVWriter writer;

private:
	mutex local_state_lock;
	vector<unique_ptr<CSVWriterState>> local_states;
	atomic<idx_t> rows_written {0};
	mutex statistics_lock;
	optional_ptr<CopyFunctionFileStatistics> written_statistics;
};

static unique_ptr<LocalFunctionData> WriteCSVInitializeLocal(ExecutionContext &context, FunctionData &bind_data) {
	auto &csv_data = bind_data.Cast<WriteCSVData>();
	if (csv_data.sql_types.size() != csv_data.options.name_list.size()) {
		throw InternalException("CSV writer has %llu columns but %llu input types", csv_data.options.name_list.size(),
		                        csv_data.sql_types.size());
	}
	vector<unique_ptr<Expression>> expressions;
	{
		lock_guard<mutex> guard(csv_data.cast_expressions_lock);
		if (csv_data.cast_expressions.empty()) {
			csv_data.cast_expressions =
			    CreateCastExpressions(csv_data, context.client, csv_data.options.name_list, csv_data.sql_types);
		}
		if (csv_data.cast_expressions.size() != csv_data.options.name_list.size()) {
			throw InternalException("CSV writer has %llu columns but %llu cast expressions",
			                        csv_data.options.name_list.size(), csv_data.cast_expressions.size());
		}
		expressions.reserve(csv_data.cast_expressions.size());
		for (const auto &expression : csv_data.cast_expressions) {
			if (!expression) {
				throw InternalException("CSV writer local initialization received a null cast expression");
			}
			expressions.push_back(expression->Copy());
		}
	}
	auto local_data = make_uniq<LocalWriteCSVData>(context.client, std::move(expressions), csv_data.flush_size);

	// create the chunk with VARCHAR types
	vector<LogicalType> types;
	types.resize(csv_data.options.name_list.size(), LogicalType::VARCHAR);

	local_data->cast_chunk.Initialize(Allocator::Get(context.client), types);
	return std::move(local_data);
}

static unique_ptr<GlobalFunctionData> WriteCSVInitializeGlobal(ClientContext &context, FunctionData &bind_data,
                                                               const string &file_path) {
	auto &csv_data = bind_data.Cast<WriteCSVData>();
	auto &options = csv_data.options;
	auto global_data =
	    make_uniq<GlobalWriteCSVData>(options, FileSystem::GetFileSystem(context), file_path, options.compression);

	global_data->writer.Initialize();

	return std::move(global_data);
}

static void WriteCSVChunkInternal(CSVWriter &writer, CSVWriterState &writer_local_state, DataChunk &cast_chunk,
                                  DataChunk &input, ExpressionExecutor &executor) {
	// first cast the columns of the chunk to varchar
	cast_chunk.Reset();
	cast_chunk.SetCardinality(input);

	executor.Execute(input, cast_chunk);

	cast_chunk.Flatten();

	writer.WriteChunk(cast_chunk, writer_local_state);
}

static void WriteCSVSink(ExecutionContext &context, FunctionData &bind_data, GlobalFunctionData &gstate,
                         LocalFunctionData &lstate, DataChunk &input) {
	auto &local_data = lstate.Cast<LocalWriteCSVData>();
	auto &global_state = gstate.Cast<GlobalWriteCSVData>();

	WriteCSVChunkInternal(global_state.writer, local_data.writer_local_state, local_data.cast_chunk, input,
	                      local_data.executor);
	global_state.AddRows(input.size());
}

//===--------------------------------------------------------------------===//
// Combine
//===--------------------------------------------------------------------===//
static void WriteCSVCombine(ExecutionContext &context, FunctionData &bind_data, GlobalFunctionData &gstate,
                            LocalFunctionData &lstate) {
	auto &local_data = lstate.Cast<LocalWriteCSVData>();
	auto &global_state = gstate.Cast<GlobalWriteCSVData>();
	global_state.writer.Flush(local_data.writer_local_state);
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//
void WriteCSVFinalize(ClientContext &context, FunctionData &bind_data, GlobalFunctionData &gstate) {
	auto &global_state = gstate.Cast<GlobalWriteCSVData>();
	auto &csv_data = bind_data.Cast<WriteCSVData>();
	auto &options = csv_data.options;

	if (!options.suffix.empty()) {
		global_state.writer.WriteRawString(options.suffix);
	} else if (global_state.writer.WrittenAnything()) {
		global_state.writer.WriteRawString(global_state.writer.writer_options.newline);
	}
	const auto file_size = global_state.writer.CloseAndGetFileSize();
	global_state.FinalizeWrittenStatistics(file_size);
}

static void WriteCSVGetWrittenStatistics(ClientContext &context, FunctionData &bind_data, GlobalFunctionData &gstate,
                                         CopyFunctionFileStatistics &statistics) {
	gstate.Cast<GlobalWriteCSVData>().SetWrittenStatistics(statistics);
}

//===--------------------------------------------------------------------===//
// Execution Mode
//===--------------------------------------------------------------------===//
CopyFunctionExecutionMode WriteCSVExecutionMode(bool preserve_insertion_order, bool supports_batch_index) {
	if (!preserve_insertion_order) {
		return CopyFunctionExecutionMode::PARALLEL_COPY_TO_FILE;
	}
	if (supports_batch_index) {
		return CopyFunctionExecutionMode::BATCH_COPY_TO_FILE;
	}
	return CopyFunctionExecutionMode::REGULAR_COPY_TO_FILE;
}
//===--------------------------------------------------------------------===//
// Prepare Batch
//===--------------------------------------------------------------------===//
struct WriteCSVBatchData : public PreparedBatchData {
	explicit WriteCSVBatchData(unique_ptr<CSVWriterState> writer_state) : writer_local_state(std::move(writer_state)) {
	}

	//! The thread-local buffer to write data into
	unique_ptr<CSVWriterState> writer_local_state;
};

unique_ptr<PreparedBatchData> WriteCSVPrepareBatch(ClientContext &context, FunctionData &bind_data,
                                                   GlobalFunctionData &gstate,
                                                   unique_ptr<ColumnDataCollection> collection) {
	auto &csv_data = bind_data.Cast<WriteCSVData>();

	// create the cast chunk with VARCHAR types
	vector<LogicalType> types;
	types.resize(csv_data.options.name_list.size(), LogicalType::VARCHAR);
	DataChunk cast_chunk;
	cast_chunk.Initialize(Allocator::Get(context), types);

	auto &original_types = collection->Types();
	auto expressions = CreateCastExpressions(csv_data, context, csv_data.options.name_list, original_types);
	ExpressionExecutor executor(context, expressions);
	auto &global_state = gstate.Cast<GlobalWriteCSVData>();

	// write CSV chunks to the batch data
	auto local_writer_state = global_state.GetLocalState(context, NextPowerOfTwo(collection->SizeInBytes()));
	auto batch = make_uniq<WriteCSVBatchData>(std::move(local_writer_state));
	const auto row_count = collection->Count();
	for (auto &chunk : collection->Chunks()) {
		WriteCSVChunkInternal(global_state.writer, *batch->writer_local_state, cast_chunk, chunk, executor);
	}
	global_state.AddRows(row_count);
	return std::move(batch);
}

//===--------------------------------------------------------------------===//
// Flush Batch
//===--------------------------------------------------------------------===//
void WriteCSVFlushBatch(ClientContext &context, FunctionData &bind_data, GlobalFunctionData &gstate,
                        PreparedBatchData &batch) {
	auto &csv_batch = batch.Cast<WriteCSVBatchData>();
	auto &global_state = gstate.Cast<GlobalWriteCSVData>();
	global_state.writer.Flush(*csv_batch.writer_local_state);
	global_state.StoreLocalState(std::move(csv_batch.writer_local_state));
}

//===--------------------------------------------------------------------===//
// File rotation
//===--------------------------------------------------------------------===//
bool WriteCSVRotateFiles(FunctionData &, const optional_idx &file_size_bytes) {
	return file_size_bytes.IsValid();
}

bool WriteCSVRotateNextFile(GlobalFunctionData &gstate, FunctionData &, const optional_idx &file_size_bytes) {
	auto &global_state = gstate.Cast<GlobalWriteCSVData>();
	return file_size_bytes.IsValid() && global_state.FileSize() > file_size_bytes.GetIndex();
}

void CSVCopyFunction::RegisterFunction(BuiltinFunctions &set) {
	CopyFunction info("csv");
	info.copy_to_bind = WriteCSVBind;
	info.copy_options = CSVListCopyOptions;
	info.copy_to_initialize_local = WriteCSVInitializeLocal;
	info.copy_to_initialize_global = WriteCSVInitializeGlobal;
	info.copy_to_get_written_statistics = WriteCSVGetWrittenStatistics;
	info.copy_to_sink = WriteCSVSink;
	info.copy_to_combine = WriteCSVCombine;
	info.copy_to_finalize = WriteCSVFinalize;
	info.execution_mode = WriteCSVExecutionMode;
	info.prepare_batch = WriteCSVPrepareBatch;
	info.flush_batch = WriteCSVFlushBatch;
	info.rotate_files = WriteCSVRotateFiles;
	info.rotate_next_file = WriteCSVRotateNextFile;
	info.serialize = WriteCSVSerialize;
	info.deserialize = WriteCSVDeserialize;

	info.copy_from_bind = MultiFileFunction<CSVMultiFileInfo>::MultiFileBindCopy;
	for (auto &function : ReadCSVTableFunction::GetFunctions()) {
		if (function.name == "read_csv" && function.arguments[0] == LogicalType::VARCHAR) {
			info.copy_from_function = std::move(function);
			break;
		}
	}
	if (info.copy_from_function.name.empty()) {
		throw InternalException("Could not construct the read_csv COPY FROM function");
	}

	info.extension = "csv";

	set.AddFunction(info);
}

} // namespace duckdb

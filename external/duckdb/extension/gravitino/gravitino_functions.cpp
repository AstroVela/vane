// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "gravitino_catalog.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/function/distributed_table_function.hpp"
#include "duckdb/function/pragma_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"

namespace duckdb {
using namespace duckdb_yyjson; // NOLINT

static string Arg(const vector<Value> &args, idx_t index) {
	if (args[index].IsNull()) {
		throw InvalidInputException("Gravitino arguments cannot be NULL");
	}
	return args[index].GetValue<string>();
}
enum class MetadataKind { CATALOG, SCHEMAS, SCHEMA, FILESETS, FILESET };
struct MetadataInfo : public TableFunctionInfo {
	explicit MetadataInfo(MetadataKind kind) : kind(kind) {
	}
	MetadataKind kind;
};
struct MetadataData : public TableFunctionData {
	vector<vector<Value>> rows;
	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<MetadataData>(*this);
	}
	bool Equals(const FunctionData &other) const override {
		return rows == other.Cast<MetadataData>().rows;
	}
};
struct MetadataState : public GlobalTableFunctionState {
	idx_t position = 0;
};
static unique_ptr<FunctionData> BindMetadata(ClientContext &context, TableFunctionBindInput &input,
                                             vector<LogicalType> &types, vector<string> &names) {
	auto &catalog = GravitinoCatalog::Get(context, Arg(input.inputs, 0));
	auto kind = static_cast<MetadataInfo &>(*input.info).kind;
	auto result = make_uniq<MetadataData>();
	types = {LogicalType::VARCHAR, LogicalType::VARCHAR};
	names = {"name", "metadata"};
	string suffix;
	const char *field = "catalog";
	string name = catalog.client.config.catalog;
	if (kind != MetadataKind::CATALOG) {
		suffix = "/schemas";
		field = "schema";
	}
	if (kind == MetadataKind::SCHEMA || kind == MetadataKind::FILESETS || kind == MetadataKind::FILESET) {
		name = Arg(input.inputs, 1);
		suffix += "/" + GravitinoClient::Encode(name);
	}
	if (kind == MetadataKind::FILESETS || kind == MetadataKind::FILESET) {
		suffix += "/filesets";
		field = "fileset";
	}
	if (kind == MetadataKind::FILESET) {
		name = Arg(input.inputs, 2);
		suffix += "/" + GravitinoClient::Encode(name);
	}
	if (kind == MetadataKind::SCHEMAS || kind == MetadataKind::FILESETS) {
		for (const auto &entry : catalog.client.List(context, suffix)) {
			result->rows.push_back({Value(entry), Value(LogicalType::VARCHAR)});
		}
	} else {
		auto response = catalog.client.Get(context, suffix);
		auto metadata = yyjson_obj_get(response.Root(), field);
		if (!yyjson_is_obj(metadata) || GravitinoJson::String(metadata, "name") != name) {
			throw IOException("Gravitino returned metadata for an unexpected resource");
		}
		result->rows.push_back({Value(name), Value(GravitinoJson::Dump(metadata))});
	}
	return std::move(result);
}
static unique_ptr<GlobalTableFunctionState> InitMetadata(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<MetadataState>();
}
static void ReadMetadata(ClientContext &, TableFunctionInput &input, DataChunk &output) {
	auto &data = input.bind_data->Cast<MetadataData>();
	auto &state = input.global_state->Cast<MetadataState>();
	idx_t count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, data.rows.size() - state.position);
	for (idx_t row = 0; row < count; row++) {
		output.SetValue(0, row, data.rows[state.position + row][0]);
		output.SetValue(1, row, data.rows[state.position + row][1]);
	}
	state.position += count;
	output.SetCardinality(count);
}
static void SerializeMetadata(Serializer &serializer, const optional_ptr<FunctionData> data, const TableFunction &) {
	serializer.WriteProperty(100, "rows", data->Cast<MetadataData>().rows);
}
static unique_ptr<FunctionData> DeserializeMetadata(Deserializer &deserializer, TableFunction &) {
	auto data = make_uniq<MetadataData>();
	data->rows = deserializer.ReadProperty<vector<vector<Value>>>(100, "rows");
	if (data->rows.size() > 4096) {
		throw SerializationException("Gravitino metadata exceeds the identifier limit");
	}
	for (const auto &row : data->rows) {
		if (row.size() != 2 || row[0].type() != LogicalType::VARCHAR || row[0].IsNull() ||
		    row[1].type() != LogicalType::VARCHAR) {
			throw SerializationException("Invalid Gravitino metadata row");
		}
	}
	return std::move(data);
}

// Resolve on the coordinator, then use file's existing scan/split protocol.
// Workers need storage access, never a Gravitino token or a live attachment.
static unique_ptr<TableRef> BindFiles(ClientContext &context, TableFunctionBindInput &input) {
	auto &catalog = GravitinoCatalog::Get(context, Arg(input.inputs, 0));
	auto relative = Arg(input.inputs, 3);
	vector<unique_ptr<ParsedExpression>> arguments;
	if (FileSystem::HasGlob(relative)) {
		vector<Value> paths;
		for (const auto &file : catalog.client.Glob(context, Arg(input.inputs, 1), Arg(input.inputs, 2), relative)) {
			paths.emplace_back(file.path);
		}
		arguments.push_back(make_uniq<ConstantExpression>(Value::LIST(LogicalType::VARCHAR, std::move(paths))));
	} else {
		auto path = catalog.client.Resolve(context, Arg(input.inputs, 1), Arg(input.inputs, 2), relative);
		arguments.push_back(make_uniq<ConstantExpression>(Value(path)));
	}
	// Roots and expanded matches are concrete paths. Do not reinterpret their
	// metacharacters when the file scan is initialized or replayed on workers.
	auto glob = make_uniq<ConstantExpression>(Value::BOOLEAN(false));
	glob->SetAlias("glob");
	arguments.push_back(std::move(glob));
	for (const auto &parameter : input.named_parameters) {
		auto argument = make_uniq<ConstantExpression>(parameter.second);
		argument->SetAlias(parameter.first);
		arguments.push_back(std::move(argument));
	}
	auto result = make_uniq<TableFunctionRef>();
	result->function = make_uniq<FunctionExpression>("list_files", std::move(arguments));
	return std::move(result);
}

static void CreateFileset(ClientContext &context, const FunctionParameters &parameters) {
	auto &catalog = GravitinoCatalog::Get(context, Arg(parameters.values, 0));
	catalog.RequireMutation(context);
	auto schema = Arg(parameters.values, 1);
	auto body = Arg(parameters.values, 2);
	GravitinoJson request(body);
	auto root = request.Root();
	GravitinoClient::Identifier(GravitinoJson::String(root, "name"));
	auto type = GravitinoJson::String(root, "type");
	if (type != "MANAGED" && type != "EXTERNAL") {
		throw InvalidInputException("Fileset type must be MANAGED or EXTERNAL");
	}
	GravitinoJson::Properties(yyjson_obj_get(root, "properties"));
	if (yyjson_obj_get(root, "comment")) {
		GravitinoJson::String(root, "comment");
	}
	auto location = yyjson_obj_get(root, "storageLocation");
	auto locations = yyjson_obj_get(root, "storageLocations");
	if (locations) {
		GravitinoJson::Properties(locations);
		yyjson_obj_iter iter = yyjson_obj_iter_with(locations);
		while (auto key = yyjson_obj_iter_next(&iter)) {
			if (!yyjson_get_len(key) || !yyjson_get_len(yyjson_obj_iter_get_val(key))) {
				throw InvalidInputException("Fileset location names and values cannot be empty");
			}
		}
	}
	if (location && (!yyjson_is_str(location) || !yyjson_get_len(location))) {
		throw InvalidInputException("storageLocation must be a nonempty string when supplied");
	}
	if (location && locations) {
		throw InvalidInputException("Specify storageLocation or storageLocations, not both");
	}
	if (type == "EXTERNAL" && !location && (!locations || yyjson_obj_size(locations) == 0)) {
		throw InvalidInputException("An EXTERNAL Fileset requires a storage location");
	}
	catalog.client.Request(context, "POST", "/schemas/" + GravitinoClient::Encode(schema) + "/filesets", body);
}
static void AlterFileset(ClientContext &context, const FunctionParameters &parameters) {
	auto &catalog = GravitinoCatalog::Get(context, Arg(parameters.values, 0));
	catalog.RequireMutation(context);
	auto body = Arg(parameters.values, 3);
	GravitinoClient::ValidateChanges(body, GravitinoResource::FILESET);
	catalog.client.Request(context, "PUT",
	                       GravitinoClient::FilesetPath(Arg(parameters.values, 1), Arg(parameters.values, 2)), body);
}
static void DropFileset(ClientContext &context, const FunctionParameters &parameters) {
	auto &catalog = GravitinoCatalog::Get(context, Arg(parameters.values, 0));
	catalog.RequireMutation(context);
	catalog.client.Request(context, "DELETE",
	                       GravitinoClient::FilesetPath(Arg(parameters.values, 1), Arg(parameters.values, 2)));
}
static void AlterSchema(ClientContext &context, const FunctionParameters &parameters) {
	auto &catalog = GravitinoCatalog::Get(context, Arg(parameters.values, 0));
	catalog.RequireMutation(context);
	auto body = Arg(parameters.values, 2);
	GravitinoClient::ValidateChanges(body, GravitinoResource::SCHEMA);
	catalog.client.Request(context, "PUT", "/schemas/" + GravitinoClient::Encode(Arg(parameters.values, 1)), body);
}
static void AlterCatalog(ClientContext &context, const FunctionParameters &parameters) {
	auto &catalog = GravitinoCatalog::Get(context, Arg(parameters.values, 0));
	catalog.RequireMutation(context);
	auto body = Arg(parameters.values, 1);
	GravitinoClient::ValidateChanges(body, GravitinoResource::CATALOG);
	catalog.client.Request(context, "PUT", "", body);
}
void RegisterGravitinoFunctions(ExtensionLoader &loader) {
	for (auto kind : {MetadataKind::CATALOG, MetadataKind::SCHEMAS, MetadataKind::SCHEMA, MetadataKind::FILESETS,
	                  MetadataKind::FILESET}) {
		const char *name = kind == MetadataKind::CATALOG    ? "gravitino_catalog"
		                   : kind == MetadataKind::SCHEMAS  ? "gravitino_schemas"
		                   : kind == MetadataKind::SCHEMA   ? "gravitino_schema"
		                   : kind == MetadataKind::FILESETS ? "gravitino_filesets"
		                                                    : "gravitino_fileset";
		idx_t argc = kind == MetadataKind::FILESET                                      ? 3
		             : (kind == MetadataKind::SCHEMA || kind == MetadataKind::FILESETS) ? 2
		                                                                                : 1;
		TableFunction function(name, vector<LogicalType>(argc, LogicalType::VARCHAR), ReadMetadata, BindMetadata,
		                       InitMetadata);
		function.function_info = make_shared_ptr<MetadataInfo>(kind);
		function.SetSerializeCallback(SerializeMetadata);
		function.SetDeserializeCallback(DeserializeMetadata);
		function.SetDistributedScanCallbacks(MakeDistributedSingletonSourceCallbacks());
		loader.RegisterFunction(std::move(function));
	}
	TableFunction files("gravitino_files", vector<LogicalType>(4, LogicalType::VARCHAR), nullptr, nullptr);
	files.bind_replace = BindFiles;
	files.named_parameters["recursive"] = LogicalType::BOOLEAN;
	loader.RegisterFunction(std::move(files));
	loader.RegisterFunction(PragmaFunction::PragmaCall("gravitino_create_fileset", CreateFileset,
	                                                   vector<LogicalType>(3, LogicalType::VARCHAR)));
	loader.RegisterFunction(PragmaFunction::PragmaCall("gravitino_alter_fileset", AlterFileset,
	                                                   vector<LogicalType>(4, LogicalType::VARCHAR)));
	loader.RegisterFunction(PragmaFunction::PragmaCall("gravitino_drop_fileset", DropFileset,
	                                                   vector<LogicalType>(3, LogicalType::VARCHAR)));
	loader.RegisterFunction(PragmaFunction::PragmaCall("gravitino_alter_schema", AlterSchema,
	                                                   vector<LogicalType>(3, LogicalType::VARCHAR)));
	loader.RegisterFunction(PragmaFunction::PragmaCall("gravitino_alter_catalog", AlterCatalog,
	                                                   vector<LogicalType>(2, LogicalType::VARCHAR)));
}
} // namespace duckdb

// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "gravitino_catalog.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/function/distributed_table_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {
using namespace duckdb_yyjson; // NOLINT

enum class DiscoveryKind { CATALOGS, CATALOG, SCHEMAS, SCHEMA, TABLES, TABLE };
struct DiscoveryInfo : public TableFunctionInfo {
	explicit DiscoveryInfo(DiscoveryKind kind) : kind(kind) {
	}
	DiscoveryKind kind;
};
struct DiscoveryData : public TableFunctionData {
	vector<vector<Value>> rows;
	bool SupportStatementCache() const override {
		return false;
	}
	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<DiscoveryData>(*this);
	}
	bool Equals(const FunctionData &other) const override {
		return rows == other.Cast<DiscoveryData>().rows;
	}
};
struct DiscoveryState : public GlobalTableFunctionState {
	idx_t position = 0;
};

static string DiscoveryArg(const vector<Value> &args, idx_t index) {
	if (args[index].IsNull()) {
		throw InvalidInputException("Gravitino discovery arguments cannot be NULL");
	}
	auto value = args[index].GetValue<string>();
	if (index) {
		GravitinoClient::Identifier(value);
	} else if (value.empty() || value.find('\0') != string::npos) {
		throw InvalidInputException("Gravitino discovery requires an attachment alias");
	}
	return value;
}
static string ResponseString(yyjson_val *object, const char *key) {
	auto value = yyjson_obj_get(object, key);
	if (!yyjson_is_str(value)) {
		throw IOException("Gravitino discovery field '%s' must be a string", key);
	}
	return string(yyjson_get_str(value), yyjson_get_len(value));
}
static void ResponseName(const string &name) {
	try {
		GravitinoClient::Identifier(name);
	} catch (const InvalidInputException &) {
		throw IOException("Gravitino returned an invalid discovery identifier");
	}
}
static yyjson_val *Resource(GravitinoJson &response, const char *kind, const string &name) {
	auto value = yyjson_obj_get(response.Root(), kind);
	if (!yyjson_is_obj(value) || ResponseString(value, "name") != name) {
		throw IOException("Gravitino returned metadata for an unexpected discovery resource");
	}
	return value;
}
static Value NamespaceValue(const vector<string> &names) {
	vector<Value> parts;
	for (const auto &name : names) {
		parts.emplace_back(name);
	}
	return Value::LIST(LogicalType::VARCHAR, std::move(parts));
}
static bool SensitiveProperty(const string &key) {
	string normalized;
	for (auto c : StringUtil::Lower(key)) {
		if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
			normalized += c;
		}
	}
	for (const auto *part : {"password", "secret", "token", "credential", "accesskey", "privatekey"}) {
		if (normalized.find(part) != string::npos) {
			return true;
		}
	}
	return normalized == "authorization";
}
static yyjson_mut_val *CopyMetadata(yyjson_mut_doc *doc, yyjson_val *value, idx_t depth = 0, bool properties = false) {
	if (depth > 64) {
		throw IOException("Gravitino metadata exceeds the discovery nesting limit");
	}
	yyjson_mut_val *result;
	if (yyjson_is_obj(value)) {
		result = yyjson_mut_obj(doc);
		size_t index, count;
		yyjson_val *key, *child;
		yyjson_obj_foreach(value, index, count, key, child) {
			string name(yyjson_get_str(key), yyjson_get_len(key));
			auto copied = properties && SensitiveProperty(name)
			                  ? yyjson_mut_str(doc, "[REDACTED]")
			                  : CopyMetadata(doc, child, depth + 1, name == "properties");
			auto copied_key = yyjson_mut_strncpy(doc, name.data(), name.size());
			if (!copied || !copied_key || !yyjson_mut_obj_add(result, copied_key, copied)) {
				throw OutOfMemoryException("Could not copy Gravitino discovery metadata");
			}
		}
	} else if (yyjson_is_arr(value)) {
		result = yyjson_mut_arr(doc);
		size_t index, count;
		yyjson_val *child;
		yyjson_arr_foreach(value, index, count, child) {
			if (!yyjson_mut_arr_append(result, CopyMetadata(doc, child, depth + 1))) {
				throw OutOfMemoryException("Could not copy Gravitino discovery metadata");
			}
		}
	} else {
		result = yyjson_val_mut_copy(doc, value);
	}
	if (!result) {
		throw OutOfMemoryException("Could not copy Gravitino discovery metadata");
	}
	return result;
}
static string DiscoveryMetadata(yyjson_val *value) {
	auto properties = yyjson_obj_get(value, "properties");
	if (properties && !yyjson_is_null(properties)) {
		try {
			GravitinoJson::Properties(properties);
		} catch (const InvalidInputException &) {
			throw IOException("Gravitino discovery properties must be a string-to-string object");
		}
	}
	std::unique_ptr<yyjson_mut_doc, decltype(&yyjson_mut_doc_free)> doc(yyjson_mut_doc_new(nullptr),
	                                                                    yyjson_mut_doc_free);
	if (!doc) {
		throw OutOfMemoryException("Could not allocate Gravitino discovery metadata");
	}
	// Redact before rows can enter a serialized plan or leave the coordinator.
	yyjson_mut_doc_set_root(doc.get(), CopyMetadata(doc.get(), value));
	size_t length;
	std::unique_ptr<char, decltype(&free)> text(yyjson_mut_write(doc.get(), 0, &length), free);
	if (!text) {
		throw OutOfMemoryException("Could not serialize Gravitino discovery metadata");
	}
	return string(text.get(), length);
}
static void CompleteList(yyjson_val *root) {
	for (const auto *field : {"nextPageToken", "nextToken", "continuationToken", "next_page_token"}) {
		auto value = yyjson_obj_get(root, field);
		if (value && !yyjson_is_null(value) && (!yyjson_is_str(value) || yyjson_get_len(value))) {
			throw IOException("Gravitino discovery pagination is unsupported; refusing an incomplete list");
		}
	}
	auto more = yyjson_obj_get(root, "hasMore");
	if (more && (!yyjson_is_bool(more) || yyjson_get_bool(more))) {
		throw IOException("Gravitino discovery pagination is unsupported; refusing an incomplete list");
	}
}
static void DiscoveryList(DiscoveryData &data, GravitinoJson &response, const vector<string> &parent,
                          const char *kind) {
	CompleteList(response.Root());
	auto entries = yyjson_obj_get(response.Root(), "identifiers");
	if (!yyjson_is_arr(entries) || yyjson_arr_size(entries) > 4096) {
		throw IOException("Gravitino discovery list must contain at most 4096 identifiers");
	}
	unordered_set<string> seen;
	size_t index, count;
	yyjson_val *entry;
	yyjson_arr_foreach(entries, index, count, entry) {
		auto name = ResponseString(entry, "name");
		auto parts = yyjson_obj_get(entry, "namespace");
		if (!yyjson_is_arr(parts) || yyjson_arr_size(parts) != parent.size()) {
			throw IOException("Gravitino returned an unexpected discovery namespace");
		}
		for (idx_t part = 0; part < parent.size(); part++) {
			auto value = yyjson_arr_get(parts, part);
			if (!yyjson_is_str(value) || string(yyjson_get_str(value), yyjson_get_len(value)) != parent[part]) {
				throw IOException("Gravitino returned an unexpected discovery namespace");
			}
		}
		ResponseName(name);
		if (!seen.insert(name).second) {
			throw IOException("Gravitino returned duplicate discovery identifiers");
		}
		data.rows.push_back({NamespaceValue(parent), Value(name), Value(kind), Value(LogicalType::VARCHAR),
		                     Value(LogicalType::VARCHAR), Value(LogicalType::VARCHAR)});
	}
}
static unique_ptr<FunctionData> BindDiscovery(ClientContext &context, TableFunctionBindInput &input,
                                              vector<LogicalType> &types, vector<string> &names) {
	auto &client = GravitinoCatalog::GetMetalake(context, DiscoveryArg(input.inputs, 0)).client;
	auto kind = static_cast<DiscoveryInfo &>(*input.info).kind;
	auto result = make_uniq<DiscoveryData>();
	types = {LogicalType::LIST(LogicalType::VARCHAR),
	         LogicalType::VARCHAR,
	         LogicalType::VARCHAR,
	         LogicalType::VARCHAR,
	         LogicalType::VARCHAR,
	         LogicalType::VARCHAR};
	names = {"namespace", "name", "kind", "catalog_type", "provider", "metadata"};
	vector<string> parent {client.config.metalake};
	if (kind == DiscoveryKind::CATALOGS) {
		auto response = client.GetDiscovery(context, "/catalogs?details=false");
		DiscoveryList(*result, response, parent, "catalog");
		return std::move(result);
	}
	auto catalog = DiscoveryArg(input.inputs, 1);
	auto path = "/catalogs/" + GravitinoClient::Encode(catalog);
	const bool schemas = kind == DiscoveryKind::SCHEMAS || kind == DiscoveryKind::SCHEMA;
	const bool tables = kind == DiscoveryKind::TABLES || kind == DiscoveryKind::TABLE;
	string schema, name = catalog;
	if (kind == DiscoveryKind::SCHEMA || tables) {
		schema = DiscoveryArg(input.inputs, 2);
	}
	if (kind == DiscoveryKind::TABLE) {
		name = DiscoveryArg(input.inputs, 3);
	}
	if (kind == DiscoveryKind::SCHEMAS) {
		parent.push_back(catalog);
		auto response = client.GetDiscovery(context, path + "/schemas");
		DiscoveryList(*result, response, parent, "schema");
		return std::move(result);
	}
	auto catalog_response = client.GetDiscovery(context, path);
	auto catalog_metadata = Resource(catalog_response, "catalog", catalog);
	auto catalog_type = ResponseString(catalog_metadata, "type");
	auto provider = ResponseString(catalog_metadata, "provider");
	if (catalog_type.empty() || provider.empty()) {
		throw IOException("Gravitino catalog type and provider must be nonempty");
	}
	if (tables && !StringUtil::CIEquals(catalog_type, "RELATIONAL")) {
		throw NotImplementedException("Table discovery requires a Gravitino RELATIONAL catalog");
	}
	if (kind == DiscoveryKind::CATALOG) {
		result->rows.push_back({NamespaceValue(parent), Value(catalog), Value("catalog"), Value(catalog_type),
		                        Value(provider), Value(DiscoveryMetadata(catalog_metadata))});
		return std::move(result);
	}
	parent.push_back(catalog);
	path += "/schemas/" + GravitinoClient::Encode(schema);
	if (tables) {
		parent.push_back(schema);
		path += "/tables";
	}
	if (kind == DiscoveryKind::TABLES) {
		auto response = client.GetDiscovery(context, path);
		DiscoveryList(*result, response, parent, "table");
		return std::move(result);
	}
	if (kind == DiscoveryKind::TABLE) {
		path += "/" + GravitinoClient::Encode(name);
	} else {
		name = schema;
	}
	auto response = client.GetDiscovery(context, path);
	const char *resource_kind = schemas ? "schema" : "table";
	auto metadata = Resource(response, resource_kind, name);
	result->rows.push_back({NamespaceValue(parent), Value(name), Value(resource_kind), Value(catalog_type),
	                        Value(provider), Value(DiscoveryMetadata(metadata))});
	return std::move(result);
}
static unique_ptr<GlobalTableFunctionState> InitDiscovery(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<DiscoveryState>();
}
static void ReadDiscovery(ClientContext &, TableFunctionInput &input, DataChunk &output) {
	auto &data = input.bind_data->Cast<DiscoveryData>();
	auto &state = input.global_state->Cast<DiscoveryState>();
	auto count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, data.rows.size() - state.position);
	for (idx_t row = 0; row < count; row++) {
		for (idx_t column = 0; column < 6; column++) {
			output.SetValue(column, row, data.rows[state.position + row][column]);
		}
	}
	state.position += count;
	output.SetCardinality(count);
}
static void SerializeDiscovery(Serializer &serializer, const optional_ptr<FunctionData> data, const TableFunction &) {
	serializer.WriteProperty(100, "rows", data->Cast<DiscoveryData>().rows);
}
static unique_ptr<FunctionData> DeserializeDiscovery(Deserializer &deserializer, TableFunction &) {
	auto result = make_uniq<DiscoveryData>();
	result->rows = deserializer.ReadProperty<vector<vector<Value>>>(100, "rows");
	if (result->rows.size() > 4096) {
		throw SerializationException("Gravitino discovery exceeds the identifier limit");
	}
	for (const auto &row : result->rows) {
		if (row.size() != 6 || row[0].IsNull() || row[0].type() != LogicalType::LIST(LogicalType::VARCHAR)) {
			throw SerializationException("Invalid Gravitino discovery row");
		}
		for (idx_t column = 1; column < row.size(); column++) {
			if (row[column].type() != LogicalType::VARCHAR || (column < 3 && row[column].IsNull())) {
				throw SerializationException("Invalid Gravitino discovery field");
			}
		}
		auto &parts = ListValue::GetChildren(row[0]);
		if (parts.empty() || parts.size() > 3) {
			throw SerializationException("Invalid Gravitino discovery namespace");
		}
		for (const auto &part : parts) {
			if (part.IsNull()) {
				throw SerializationException("Invalid Gravitino discovery namespace component");
			}
			ResponseName(part.GetValue<string>());
		}
		ResponseName(row[1].GetValue<string>());
		auto kind = row[2].GetValue<string>();
		if ((parts.size() == 1 && kind != "catalog") || (parts.size() == 2 && kind != "schema") ||
		    (parts.size() == 3 && kind != "table")) {
			throw SerializationException("Invalid Gravitino discovery resource kind");
		}
	}
	return std::move(result);
}
void RegisterGravitinoDiscovery(ExtensionLoader &loader) {
	const vector<string> names {"gravitino_metalake_catalogs", "gravitino_metalake_catalog",
	                            "gravitino_metalake_schemas",  "gravitino_metalake_schema",
	                            "gravitino_metalake_tables",   "gravitino_metalake_table"};
	const vector<idx_t> arguments {1, 2, 2, 3, 3, 4};
	for (idx_t index = 0; index < names.size(); index++) {
		TableFunction function(names[index], vector<LogicalType>(arguments[index], LogicalType::VARCHAR), ReadDiscovery,
		                       BindDiscovery, InitDiscovery);
		function.function_info = make_shared_ptr<DiscoveryInfo>(static_cast<DiscoveryKind>(index));
		function.SetSerializeCallback(SerializeDiscovery);
		function.SetDeserializeCallback(DeserializeDiscovery);
		function.SetDistributedScanCallbacks(MakeDistributedSingletonSourceCallbacks());
		loader.RegisterFunction(std::move(function));
	}
}
} // namespace duckdb

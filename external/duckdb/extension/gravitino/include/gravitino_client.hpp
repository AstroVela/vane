// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once
#include "duckdb/common/common.hpp"
#include "duckdb/common/types/value.hpp"
#include "yyjson.hpp"
#include <memory>

namespace duckdb {
class ClientContext;
using GravitinoJsonValue = duckdb_yyjson::yyjson_val;
struct GravitinoJsonDeleter {
	void operator()(duckdb_yyjson::yyjson_doc *doc) const;
};
class GravitinoJson {
public:
	explicit GravitinoJson(const string &text);
	GravitinoJsonValue *Root() const;
	static string String(GravitinoJsonValue *value, const char *field);
	static string Dump(GravitinoJsonValue *value);
	static string Quote(const string &value);
	static void Properties(GravitinoJsonValue *value);

private:
	std::unique_ptr<duckdb_yyjson::yyjson_doc, GravitinoJsonDeleter> doc;
};
struct GravitinoConfig {
	string endpoint;
	string metalake;
	string catalog;
	string token;
	string location_name;
	idx_t timeout_ms = 30000;
	idx_t max_response_bytes = 2 * 1024 * 1024;
	void Validate();
};
struct GravitinoResponse {
	long status;
	string body;
};
class GravitinoClient {
public:
	explicit GravitinoClient(GravitinoConfig config);
	const GravitinoConfig config;
	GravitinoResponse Request(ClientContext &context, const string &method, const string &suffix,
	                          const string &body = "", long allowed_error_status = 0) const;
	GravitinoJson Get(ClientContext &context, const string &suffix = "") const;
	vector<string> List(ClientContext &context, const string &suffix) const;
	string Resolve(ClientContext &context, const string &schema, const string &fileset, const string &path) const;
	static string Encode(const string &name);
	static void Identifier(const string &name);
	static string FilesetPath(const string &schema, const string &fileset);
	static void ValidateChanges(const string &json, bool allow_rename);
};
} // namespace duckdb

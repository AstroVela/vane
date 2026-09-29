// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "gravitino_client.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include <curl/curl.h>
#include <exception>
#include <mutex>

namespace duckdb {
using namespace duckdb_yyjson; // NOLINT

void GravitinoJsonDeleter::operator()(yyjson_doc *doc) const {
	yyjson_doc_free(doc);
}
GravitinoJson::GravitinoJson(const string &text) : doc(yyjson_read(text.data(), text.size(), 0)) {
	if (!doc || !yyjson_is_obj(Root())) {
		throw InvalidInputException("Gravitino requires a JSON object");
	}
}
GravitinoJsonValue *GravitinoJson::Root() const {
	return yyjson_doc_get_root(doc.get());
}
string GravitinoJson::String(GravitinoJsonValue *value, const char *field) {
	auto item = yyjson_obj_get(value, field);
	if (!yyjson_is_str(item)) {
		throw InvalidInputException("Gravitino JSON field '%s' must be a string", field);
	}
	return string(yyjson_get_str(item), yyjson_get_len(item));
}
string GravitinoJson::Dump(GravitinoJsonValue *value) {
	if (!value) {
		throw InvalidInputException("Gravitino response is missing a required field");
	}
	size_t size;
	auto data = yyjson_val_write(value, 0, &size);
	if (!data) {
		throw OutOfMemoryException("Could not serialize Gravitino metadata");
	}
	std::unique_ptr<char, decltype(&free)> owner(data, free);
	return string(data, size);
}
string GravitinoJson::Quote(const string &value) {
	auto doc = yyjson_mut_doc_new(nullptr);
	if (!doc) {
		throw OutOfMemoryException("Could not allocate Gravitino JSON");
	}
	std::unique_ptr<yyjson_mut_doc, decltype(&yyjson_mut_doc_free)> owner(doc, yyjson_mut_doc_free);
	yyjson_mut_doc_set_root(doc, yyjson_mut_strn(doc, value.data(), value.size()));
	size_t size;
	auto data = yyjson_mut_write(doc, 0, &size);
	if (!data) {
		throw OutOfMemoryException("Could not serialize Gravitino JSON");
	}
	std::unique_ptr<char, decltype(&free)> text(data, free);
	return string(data, size);
}
void GravitinoJson::Properties(GravitinoJsonValue *value) {
	if (!yyjson_is_obj(value)) {
		throw InvalidInputException("Gravitino properties must be a string-to-string JSON object");
	}
	yyjson_obj_iter iter = yyjson_obj_iter_with(value);
	while (auto key = yyjson_obj_iter_next(&iter)) {
		if (!yyjson_is_str(yyjson_obj_iter_get_val(key))) {
			throw InvalidInputException("Gravitino properties must have string values");
		}
	}
}

static void InitializeCurl() {
	static std::once_flag initialized;
	std::call_once(initialized, [] {
		if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
			throw IOException("Could not initialize Gravitino HTTP transport");
		}
	});
}
void GravitinoConfig::Validate() {
	InitializeCurl();
	GravitinoClient::Identifier(metalake);
	GravitinoClient::Identifier(catalog);
	if (endpoint.empty() || endpoint.size() > 8192 || endpoint.find('\0') != string::npos || token.size() > 16384 ||
	    token.find_first_of("\r\n") != string::npos || token.find('\0') != string::npos) {
		throw InvalidInputException("Invalid Gravitino endpoint or bearer token");
	}
	std::unique_ptr<CURLU, decltype(&curl_url_cleanup)> url(curl_url(), curl_url_cleanup);
	if (!url || curl_url_set(url.get(), CURLUPART_URL, endpoint.c_str(), 0) != CURLUE_OK) {
		throw InvalidInputException("Gravitino endpoint must be an absolute HTTP(S) URL");
	}
	char *part = nullptr;
	if (curl_url_get(url.get(), CURLUPART_SCHEME, &part, 0) != CURLUE_OK) {
		throw InvalidInputException("Missing Gravitino endpoint protocol");
	}
	string scheme(part);
	curl_free(part);
	if (scheme != "http" && scheme != "https") {
		throw InvalidInputException("Gravitino endpoint must use HTTP or HTTPS");
	}
	for (auto kind : {CURLUPART_USER, CURLUPART_PASSWORD, CURLUPART_QUERY, CURLUPART_FRAGMENT}) {
		part = nullptr;
		if (curl_url_get(url.get(), kind, &part, 0) == CURLUE_OK) {
			curl_free(part);
			throw InvalidInputException("Gravitino endpoint cannot include credentials, a query, or a fragment");
		}
	}
	while (!endpoint.empty() && endpoint.back() == '/') {
		endpoint.pop_back();
	}
	if (!timeout_ms || timeout_ms > 300000 || !max_response_bytes || max_response_bytes > 16 * 1024 * 1024) {
		throw InvalidInputException("Gravitino timeout_ms must be 1..300000 and max_response_bytes 1..16777216");
	}
}
GravitinoClient::GravitinoClient(GravitinoConfig config_p) : config(std::move(config_p)) {
}
void GravitinoClient::Identifier(const string &name) {
	if (name.empty() || name.size() > 1024 || name == "." || name == ".." ||
	    name.find_first_of("/\\") != string::npos || name.find('\0') != string::npos) {
		throw InvalidInputException("Gravitino names must be nonempty path components of at most 1024 bytes");
	}
}
string GravitinoClient::Encode(const string &name) {
	Identifier(name);
	static constexpr const char *HEX = "0123456789ABCDEF";
	string result;
	for (unsigned char c : name) {
		if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' ||
		    c == '.' || c == '~') {
			result += char(c);
		} else {
			result += '%';
			result += HEX[c >> 4];
			result += HEX[c & 15];
		}
	}
	return result;
}
struct GravitinoTransfer {
	GravitinoTransfer(ClientContext &context, idx_t limit) : context(context), limit(limit) {
	}
	ClientContext &context;
	idx_t limit;
	string body;
	idx_t header_bytes = 0;
	bool oversized = false;
	bool compressed = false;
	std::exception_ptr error;
};
static size_t Receive(char *data, size_t size, size_t count, void *opaque) noexcept {
	auto &state = *static_cast<GravitinoTransfer *>(opaque);
	try {
		if (size && count > state.limit / size) {
			state.oversized = true;
			return 0;
		}
		auto bytes = size * count;
		if (bytes > state.limit - state.body.size()) {
			state.oversized = true;
			return 0;
		}
		state.body.append(data, bytes);
		return bytes;
	} catch (...) {
		state.error = std::current_exception();
		return 0;
	}
}
static size_t ReceiveHeader(char *data, size_t size, size_t count, void *opaque) noexcept {
	auto &state = *static_cast<GravitinoTransfer *>(opaque);
	try {
		if (size && count > 65536 / size) {
			state.oversized = true;
			return 0;
		}
		auto bytes = size * count;
		if (bytes > 65536 - state.header_bytes) {
			state.oversized = true;
			return 0;
		}
		state.header_bytes += bytes;
		string line(data, bytes);
		if (StringUtil::CIStartsWith(line, "content-encoding:")) {
			auto encoding = StringUtil::Lower(line.substr(17));
			StringUtil::Trim(encoding);
			if (!encoding.empty() && encoding != "identity") {
				state.compressed = true;
				return 0;
			}
		}
		return bytes;
	} catch (...) {
		state.error = std::current_exception();
		return 0;
	}
}
static int Progress(void *opaque, curl_off_t, curl_off_t, curl_off_t, curl_off_t) noexcept {
	return static_cast<GravitinoTransfer *>(opaque)->context.IsInterrupted() ? 1 : 0;
}
template <class T>
static void SetCurlOption(CURL *curl, CURLoption option, T value) {
	if (curl_easy_setopt(curl, option, value) != CURLE_OK) {
		throw IOException("Could not configure Gravitino HTTP transport");
	}
}
GravitinoResponse GravitinoClient::Request(ClientContext &context, const string &method, const string &suffix,
                                           const string &body, long allowed_error_status) const {
	if (!DBConfig::GetConfig(context).options.enable_external_access) {
		throw PermissionException("Gravitino requires enable_external_access");
	}
	if (context.IsInterrupted()) {
		throw InterruptException();
	}
	if (body.size() > config.max_response_bytes) {
		throw InvalidInputException("Gravitino metadata request exceeds max_response_bytes");
	}
	InitializeCurl();
	std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(), curl_easy_cleanup);
	if (!curl) {
		throw OutOfMemoryException("Could not allocate Gravitino HTTP client");
	}
	string url =
	    config.endpoint + "/api/metalakes/" + Encode(config.metalake) + "/catalogs/" + Encode(config.catalog) + suffix;
	GravitinoTransfer transfer {context, config.max_response_bytes};
	curl_slist *headers = nullptr;
	std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> header_owner(nullptr, curl_slist_free_all);
	for (const auto &header : vector<string> {"Accept: application/vnd.gravitino.v1+json",
	                                          "Content-Type: application/json", "Accept-Encoding: identity"}) {
		auto next = curl_slist_append(headers, header.c_str());
		if (!next) {
			throw OutOfMemoryException("Could not allocate Gravitino HTTP headers");
		}
		headers = next;
		header_owner.release();
		header_owner.reset(headers);
	}
	if (!config.token.empty()) {
		auto next = curl_slist_append(headers, ("Authorization: Bearer " + config.token).c_str());
		if (!next) {
			throw OutOfMemoryException("Could not allocate Gravitino authentication header");
		}
		header_owner.release();
		header_owner.reset(next);
		headers = next;
	}
	SetCurlOption(curl.get(), CURLOPT_URL, url.c_str());
	SetCurlOption(curl.get(), CURLOPT_CUSTOMREQUEST, method.c_str());
	SetCurlOption(curl.get(), CURLOPT_HTTPHEADER, headers);
	SetCurlOption(curl.get(), CURLOPT_FOLLOWLOCATION, 0L);
	SetCurlOption(curl.get(), CURLOPT_NOSIGNAL, 1L);
	SetCurlOption(curl.get(), CURLOPT_TIMEOUT_MS, long(config.timeout_ms));
	SetCurlOption(curl.get(), CURLOPT_CONNECTTIMEOUT_MS, long(config.timeout_ms));
	SetCurlOption(curl.get(), CURLOPT_HTTP_CONTENT_DECODING, 0L);
	SetCurlOption(curl.get(), CURLOPT_WRITEFUNCTION, Receive);
	SetCurlOption(curl.get(), CURLOPT_WRITEDATA, &transfer);
	SetCurlOption(curl.get(), CURLOPT_HEADERFUNCTION, ReceiveHeader);
	SetCurlOption(curl.get(), CURLOPT_HEADERDATA, &transfer);
	SetCurlOption(curl.get(), CURLOPT_NOPROGRESS, 0L);
	SetCurlOption(curl.get(), CURLOPT_XFERINFOFUNCTION, Progress);
	SetCurlOption(curl.get(), CURLOPT_XFERINFODATA, &transfer);
	if (!body.empty()) {
		SetCurlOption(curl.get(), CURLOPT_POSTFIELDS, body.data());
		SetCurlOption(curl.get(), CURLOPT_POSTFIELDSIZE_LARGE, curl_off_t(body.size()));
	}
	auto result = curl_easy_perform(curl.get());
	if (transfer.error) {
		std::rethrow_exception(transfer.error);
	}
	if (context.IsInterrupted()) {
		throw InterruptException();
	}
	if (transfer.oversized || transfer.compressed || result != CURLE_OK) {
		const auto detail = transfer.oversized    ? "response exceeds configured bounds"
		                    : transfer.compressed ? "compressed responses are unsupported"
		                                          : curl_easy_strerror(result);
		throw IOException("Gravitino request failed: %s%s", detail,
		                  method == "GET" ? "" : "; mutation outcome is unknown; no retry was performed");
	}
	long status = 0;
	curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &status);
	if (allowed_error_status && status == allowed_error_status) {
		return {status, std::move(transfer.body)};
	}
	if (status < 200 || status >= 300) {
		throw IOException("Gravitino %s request returned HTTP %d%s", method, status,
		                  method != "GET" && status >= 500 ? "; mutation outcome is unknown" : "");
	}
	auto response = [&]() -> GravitinoJson {
		try {
			return GravitinoJson(transfer.body);
		} catch (const InvalidInputException &) {
			throw IOException("Gravitino returned invalid JSON%s",
			                  method == "GET" ? "" : "; mutation outcome is unknown");
		}
	}();
	auto code = yyjson_obj_get(response.Root(), "code");
	if (!yyjson_is_int(code) || yyjson_get_sint(code) != 0) {
		throw IOException("Gravitino response did not confirm success%s",
		                  method == "GET" ? "" : "; mutation outcome is unknown");
	}
	if (method == "DELETE") {
		auto dropped = yyjson_obj_get(response.Root(), "dropped");
		if (!yyjson_is_bool(dropped)) {
			throw IOException("Gravitino drop response is missing its outcome; mutation outcome is unknown");
		}
		if (!yyjson_get_bool(dropped) && allowed_error_status != 404) {
			throw CatalogException("Gravitino resource was not found; nothing was dropped");
		}
	}
	return {status, std::move(transfer.body)};
}
GravitinoJson GravitinoClient::Get(ClientContext &context, const string &suffix) const {
	return GravitinoJson(Request(context, "GET", suffix).body);
}
vector<string> GravitinoClient::List(ClientContext &context, const string &suffix) const {
	auto response = Get(context, suffix);
	auto values = yyjson_obj_get(response.Root(), "identifiers");
	if (!yyjson_is_arr(values) || yyjson_arr_size(values) > 4096) {
		throw IOException("Gravitino identifiers must be an array of at most 4096 entries");
	}
	vector<string> result;
	size_t i, count;
	yyjson_val *value;
	yyjson_arr_foreach(values, i, count, value) {
		auto name = GravitinoJson::String(value, "name");
		Identifier(name);
		result.push_back(std::move(name));
	}
	return result;
}
string GravitinoClient::FilesetPath(const string &schema, const string &fileset) {
	return "/schemas/" + Encode(schema) + "/filesets/" + Encode(fileset);
}
string GravitinoClient::Resolve(ClientContext &context, const string &schema, const string &fileset,
                                const string &path) const {
	if (path.size() > 8192 || path.find('\0') != string::npos || path.find('\\') != string::npos ||
	    (!path.empty() && path.front() == '/')) {
		throw InvalidInputException("Fileset paths must be relative paths without NUL or backslashes");
	}
	for (const auto &part : StringUtil::Split(path, '/')) {
		if (part == ".." || part == "." || part.find('%') != string::npos) {
			throw InvalidInputException("Fileset paths cannot contain dot segments or percent escapes");
		}
	}
	auto response = Get(context, FilesetPath(schema, fileset));
	auto data = yyjson_obj_get(response.Root(), "fileset");
	auto locations = yyjson_obj_get(data, "storageLocations");
	string location;
	if (!yyjson_is_obj(locations) || !yyjson_obj_size(locations)) {
		throw InvalidInputException("Gravitino Fileset must provide storageLocations");
	}
	auto location_name = config.location_name;
	if (location_name.empty()) {
		auto properties = yyjson_obj_get(data, "properties");
		auto default_name = yyjson_obj_get(properties, "default-location-name");
		// Gravitino 1.3 represents its unnamed location under the reserved
		// key 'unknown'. Never choose an arbitrary location from the map.
		location_name = default_name ? GravitinoJson::String(properties, "default-location-name") : "unknown";
	}
	location = GravitinoJson::String(locations, location_name.c_str());
	if (StringUtil::StartsWith(location, "s3a://")) {
		location.replace(0, 6, "s3://");
	} else if (StringUtil::StartsWith(location, "file:/")) {
		if (StringUtil::StartsWith(location, "file:///")) {
			location.erase(0, 7);
		} else if (!StringUtil::StartsWith(location, "file://")) {
			location.erase(0, 5);
		} else {
			throw NotImplementedException("Fileset file URIs with a host are unsupported");
		}
	}
	if (location.empty() || location.find('\0') != string::npos || location.find_first_of("%?#\\") != string::npos ||
	    location.find("{{") != string::npos) {
		throw InvalidInputException("Fileset storage location is empty or contains unresolved/unsupported URI syntax");
	}
	if (location.front() != '/' && !StringUtil::StartsWith(location, "s3://")) {
		throw NotImplementedException("Gravitino Fileset reads currently support absolute local paths and S3");
	}
	if (!path.empty()) {
		if (location.back() != '/') {
			location += '/';
		}
		location += path;
	}
	return location;
}
void GravitinoClient::ValidateChanges(const string &json, bool allow_rename) {
	GravitinoJson data(json);
	auto updates = yyjson_obj_get(data.Root(), "updates");
	if (!yyjson_is_arr(updates) || !yyjson_arr_size(updates) || yyjson_arr_size(updates) > 128) {
		throw InvalidInputException("Gravitino updates must contain 1..128 changes");
	}
	size_t i, count;
	yyjson_val *update;
	yyjson_arr_foreach(updates, i, count, update) {
		auto type = GravitinoJson::String(update, "@type");
		if (type == "rename" && allow_rename) {
			Identifier(GravitinoJson::String(update, "newName"));
		} else if (type == "updateComment") {
			GravitinoJson::String(update, "newComment");
		} else if (type == "setProperty") {
			GravitinoJson::String(update, "property");
			GravitinoJson::String(update, "value");
		} else if (type == "removeProperty") {
			GravitinoJson::String(update, "property");
		} else {
			throw NotImplementedException("Unsupported Gravitino metadata change: %s", type);
		}
	}
}
} // namespace duckdb

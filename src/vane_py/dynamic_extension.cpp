// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/dynamic_extension.hpp"

#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/local_file_system.hpp"
#include "duckdb/main/extension.hpp"
#include "duckdb/main/extension_helper.hpp"
#include "duckdb/main/extension_manager.hpp"
#include "vane_python/pyconnection/pyconnection.hpp"

namespace duckdb {

static ClientContext &DynamicExtensionContext(const shared_ptr<DuckDBPyConnection> &connection) {
	if (!connection) {
		throw InvalidInputException("A connection is required for dynamic extension loading");
	}
	auto &native_connection = connection->con.GetConnection();
	if (!native_connection.context) {
		throw ConnectionException("Connection already closed!");
	}
	return *native_connection.context;
}

static py::dict InspectDynamicExtension(const string &path) {
	LocalFileSystem file_system;
	auto handle = file_system.OpenFile(path, FileFlags::FILE_FLAGS_READ);
	auto metadata = ExtensionHelper::ParseExtensionMetaData(*handle);
	auto compatibility_error = metadata.GetInvalidMetadataError();
	if (!metadata.AppearsValid()) {
		throw InvalidInputException(compatibility_error);
	}

	py::dict result;
	result["canonical_name"] = ExtensionHelper::GetExtensionName(path);
	result["abi_type"] = string(EnumUtil::ToChars(metadata.abi_type));
	result["platform"] = metadata.platform;
	result["duckdb_version"] = metadata.duckdb_version;
	result["duckdb_capi_version"] = metadata.duckdb_capi_version;
	result["extension_version"] = metadata.extension_version;
	result["compatibility_error"] = compatibility_error;
	return result;
}

static py::object LoadedDynamicExtension(const string &extension, const shared_ptr<DuckDBPyConnection> &connection) {
	auto &context = DynamicExtensionContext(connection);
	auto canonical_name = ExtensionHelper::GetExtensionName(extension);
	auto &manager = ExtensionManager::Get(context);
	auto extension_info = manager.GetExtensionInfo(canonical_name);
	if (!extension_info) {
		return py::none();
	}

	lock_guard<mutex> guard(extension_info->lock);
	if (!extension_info->is_loaded) {
		return py::none();
	}
	if (!extension_info->install_info) {
		throw InternalException("Loaded extension '%s' has no install provenance", canonical_name);
	}

	py::dict result;
	result["canonical_name"] = canonical_name;
	result["full_path"] = extension_info->install_info->full_path;
	result["install_mode"] = string(EnumUtil::ToChars(extension_info->install_info->mode));
	result["extension_version"] = extension_info->install_info->version;
	return std::move(result);
}

static py::dict LoadDynamicExtension(const string &path, const shared_ptr<DuckDBPyConnection> &connection) {
	auto &context = DynamicExtensionContext(connection);
	ExtensionHelper::LoadExternalExtension(context, path);
	auto loaded = LoadedDynamicExtension(path, connection);
	if (loaded.is_none()) {
		throw InternalException("DuckDB returned from loading '%s' without loaded extension state", path);
	}
	return loaded.cast<py::dict>();
}

void InitializeDynamicExtensionBindings(py::module_ &module) {
	module.def("_dynamic_extension_canonical_name", &ExtensionHelper::GetExtensionName, py::arg("extension"));
	module.def(
	    "_dynamic_extension_directory",
	    [](const shared_ptr<DuckDBPyConnection> &connection) {
		    return ExtensionHelper::ExtensionDirectory(DynamicExtensionContext(connection));
	    },
	    py::kw_only(), py::arg("connection").none(false));
	module.def("_inspect_dynamic_extension", &InspectDynamicExtension, py::arg("path"));
	module.def("_load_dynamic_extension", &LoadDynamicExtension, py::arg("path"), py::kw_only(),
	           py::arg("connection").none(false));
	module.def("_loaded_dynamic_extension", &LoadedDynamicExtension, py::arg("extension"), py::kw_only(),
	           py::arg("connection").none(false));
}

} // namespace duckdb

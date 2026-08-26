// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/dynamic_extension.hpp"

#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/local_file_system.hpp"
#include "duckdb/common/windows_util.hpp"
#include "duckdb/main/extension.hpp"
#include "duckdb/main/extension_helper.hpp"
#include "duckdb/main/extension_manager.hpp"
#include "vane_python/pyconnection/pyconnection.hpp"

#ifdef DUCKDB_WINDOWS
#include <aclapi.h>
#endif

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

static void SecureDynamicExtensionCacheDirectory(const string &path) {
#ifdef DUCKDB_WINDOWS
	auto windows_path = WindowsUtil::UTF8ToUnicode(path.c_str());
	HANDLE process_token = nullptr;
	if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &process_token)) {
		throw IOException("Could not open process token for extension cache directory '%s' (Windows error %u)", path,
		                  GetLastError());
	}
	DWORD token_user_size = 0;
	GetTokenInformation(process_token, TokenUser, nullptr, 0, &token_user_size);
	auto status = GetLastError();
	if (status != ERROR_INSUFFICIENT_BUFFER || token_user_size == 0) {
		CloseHandle(process_token);
		throw IOException("Could not size process identity for extension cache directory '%s' (Windows error %u)", path,
		                  status);
	}
	vector<uint8_t> token_user_buffer(token_user_size);
	if (!GetTokenInformation(process_token, TokenUser, token_user_buffer.data(), token_user_size, &token_user_size)) {
		status = GetLastError();
		CloseHandle(process_token);
		throw IOException("Could not read process identity for extension cache directory '%s' (Windows error %u)", path,
		                  status);
	}
	CloseHandle(process_token);
	auto token_user = reinterpret_cast<TOKEN_USER *>(token_user_buffer.data());

	EXPLICIT_ACCESSW user_access {};
	user_access.grfAccessPermissions = GENERIC_ALL;
	user_access.grfAccessMode = SET_ACCESS;
	user_access.grfInheritance = SUB_CONTAINERS_AND_OBJECTS_INHERIT;
	user_access.Trustee.TrusteeForm = TRUSTEE_IS_SID;
	user_access.Trustee.TrusteeType = TRUSTEE_IS_USER;
	user_access.Trustee.ptstrName = static_cast<LPWSTR>(token_user->User.Sid);

	PACL private_acl = nullptr;
	status = SetEntriesInAclW(1, &user_access, nullptr, &private_acl);
	if (status != ERROR_SUCCESS || !private_acl) {
		throw IOException("Could not create private ACL for extension cache directory '%s' (Windows error %u)", path,
		                  status);
	}

	status = SetNamedSecurityInfoW(const_cast<LPWSTR>(windows_path.c_str()), SE_FILE_OBJECT,
	                               DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION, nullptr, nullptr,
	                               private_acl, nullptr);
	LocalFree(private_acl);
	if (status != ERROR_SUCCESS) {
		throw IOException("Could not secure extension cache directory '%s' (Windows error %u)", path, status);
	}
#else
	throw NotImplementedException("Windows DACLs are unavailable on this platform for '%s'", path);
#endif
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
	module.def("_secure_dynamic_extension_cache_directory", &SecureDynamicExtensionCacheDirectory, py::arg("path"));
	module.def("_inspect_dynamic_extension", &InspectDynamicExtension, py::arg("path"));
	module.def("_load_dynamic_extension", &LoadDynamicExtension, py::arg("path"), py::kw_only(),
	           py::arg("connection").none(false));
	module.def("_loaded_dynamic_extension", &LoadedDynamicExtension, py::arg("extension"), py::kw_only(),
	           py::arg("connection").none(false));
}

} // namespace duckdb

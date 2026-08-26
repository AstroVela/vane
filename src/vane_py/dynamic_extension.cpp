// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/dynamic_extension.hpp"

#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/local_file_system.hpp"
#include "duckdb/common/windows_util.hpp"
#include "duckdb/main/database.hpp"
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

#ifdef DUCKDB_WINDOWS
static vector<uint8_t> DynamicExtensionProcessUserSid(const string &path) {
	HANDLE process_token = nullptr;
	if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &process_token)) {
		throw IOException("Could not open process token for extension cache path '%s' (Windows error %u)", path,
		                  GetLastError());
	}
	DWORD token_user_size = 0;
	GetTokenInformation(process_token, TokenUser, nullptr, 0, &token_user_size);
	auto status = GetLastError();
	if (status != ERROR_INSUFFICIENT_BUFFER || token_user_size == 0) {
		CloseHandle(process_token);
		throw IOException("Could not size process identity for extension cache path '%s' (Windows error %u)", path,
		                  status);
	}
	vector<uint8_t> token_user_buffer(token_user_size);
	if (!GetTokenInformation(process_token, TokenUser, token_user_buffer.data(), token_user_size, &token_user_size)) {
		status = GetLastError();
		CloseHandle(process_token);
		throw IOException("Could not read process identity for extension cache path '%s' (Windows error %u)", path,
		                  status);
	}
	CloseHandle(process_token);
	return token_user_buffer;
}

static vector<uint8_t> DynamicExtensionWellKnownSid(WELL_KNOWN_SID_TYPE type, const string &path) {
	vector<uint8_t> sid_buffer(SECURITY_MAX_SID_SIZE);
	DWORD sid_size = NumericCast<DWORD>(sid_buffer.size());
	if (!CreateWellKnownSid(type, nullptr, sid_buffer.data(), &sid_size)) {
		throw IOException("Could not create a trusted identity for extension cache path '%s' (Windows error %u)", path,
		                  GetLastError());
	}
	sid_buffer.resize(sid_size);
	return sid_buffer;
}

static vector<uint8_t> DynamicExtensionAccountSid(const wchar_t *account_name, const string &path) {
	DWORD sid_size = 0;
	DWORD domain_name_size = 0;
	SID_NAME_USE sid_type;
	LookupAccountNameW(nullptr, account_name, nullptr, &sid_size, nullptr, &domain_name_size, &sid_type);
	auto status = GetLastError();
	if (status != ERROR_INSUFFICIENT_BUFFER || sid_size == 0 || domain_name_size == 0) {
		throw IOException("Could not size a trusted account identity for extension cache path '%s' (Windows error %u)",
		                  path, status);
	}
	vector<uint8_t> sid_buffer(sid_size);
	vector<wchar_t> domain_name_buffer(domain_name_size);
	if (!LookupAccountNameW(nullptr, account_name, sid_buffer.data(), &sid_size, domain_name_buffer.data(),
	                        &domain_name_size, &sid_type)) {
		throw IOException("Could not read a trusted account identity for extension cache path '%s' (Windows error %u)",
		                  path, GetLastError());
	}
	sid_buffer.resize(sid_size);
	return sid_buffer;
}

static bool DynamicExtensionSidEquals(PSID candidate, const vector<uint8_t> &trusted_sid) {
	return candidate && IsValidSid(candidate) && EqualSid(candidate, const_cast<uint8_t *>(trusted_sid.data()));
}

static bool DynamicExtensionCacheTrustedPrincipal(PSID candidate, PSID process_user_sid,
                                                  const vector<uint8_t> &local_system_sid,
                                                  const vector<uint8_t> &administrators_sid,
                                                  const vector<uint8_t> &trusted_installer_sid) {
	if ((candidate && process_user_sid && IsValidSid(candidate) && IsValidSid(process_user_sid) &&
	     EqualSid(candidate, process_user_sid)) ||
	    DynamicExtensionSidEquals(candidate, local_system_sid) ||
	    DynamicExtensionSidEquals(candidate, administrators_sid) ||
	    DynamicExtensionSidEquals(candidate, trusted_installer_sid)) {
		return true;
	}
	return false;
}

static bool DynamicExtensionCacheTrustedSid(PSID candidate, PSID owner, PSID process_user_sid,
                                            const vector<uint8_t> &local_system_sid,
                                            const vector<uint8_t> &administrators_sid,
                                            const vector<uint8_t> &trusted_installer_sid,
                                            const vector<uint8_t> &creator_owner_sid,
                                            const vector<uint8_t> &creator_owner_rights_sid) {
	if (DynamicExtensionCacheTrustedPrincipal(candidate, process_user_sid, local_system_sid, administrators_sid,
	                                          trusted_installer_sid)) {
		return true;
	}
	return owner && (DynamicExtensionSidEquals(candidate, creator_owner_sid) ||
	                 DynamicExtensionSidEquals(candidate, creator_owner_rights_sid));
}
#endif

static void SecureDynamicExtensionCachePath(const string &path, bool directory) {
#ifdef DUCKDB_WINDOWS
	auto windows_path = WindowsUtil::UTF8ToUnicode(path.c_str());
	auto attributes = GetFileAttributesW(windows_path.c_str());
	if (attributes == INVALID_FILE_ATTRIBUTES) {
		throw IOException("Could not inspect extension cache path '%s' (Windows error %u)", path, GetLastError());
	}
	bool path_is_directory = (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
	if (path_is_directory != directory || (attributes & FILE_ATTRIBUTE_REPARSE_POINT)) {
		throw IOException("Extension cache path has an unsafe file type: '%s'", path);
	}
	auto token_user_buffer = DynamicExtensionProcessUserSid(path);
	auto token_user = reinterpret_cast<TOKEN_USER *>(token_user_buffer.data());
	auto local_system_sid = DynamicExtensionWellKnownSid(WinLocalSystemSid, path);
	auto administrators_sid = DynamicExtensionWellKnownSid(WinBuiltinAdministratorsSid, path);
	auto trusted_installer_sid = DynamicExtensionAccountSid(L"NT SERVICE\\TrustedInstaller", path);

	PSID owner = nullptr;
	PSECURITY_DESCRIPTOR owner_security_descriptor = nullptr;
	auto status =
	    GetNamedSecurityInfoW(const_cast<LPWSTR>(windows_path.c_str()), SE_FILE_OBJECT, OWNER_SECURITY_INFORMATION,
	                          &owner, nullptr, nullptr, nullptr, &owner_security_descriptor);
	if (status != ERROR_SUCCESS) {
		if (owner_security_descriptor) {
			LocalFree(owner_security_descriptor);
		}
		throw IOException("Could not read the owner of extension cache path '%s' (Windows error %u)", path, status);
	}
	auto owner_trusted = DynamicExtensionCacheTrustedPrincipal(owner, token_user->User.Sid, local_system_sid,
	                                                           administrators_sid, trusted_installer_sid);
	LocalFree(owner_security_descriptor);
	if (!owner_trusted) {
		throw IOException("Extension cache path has an untrusted owner: '%s'", path);
	}

	EXPLICIT_ACCESSW user_access {};
	user_access.grfAccessPermissions = GENERIC_ALL;
	user_access.grfAccessMode = SET_ACCESS;
	user_access.grfInheritance = directory ? SUB_CONTAINERS_AND_OBJECTS_INHERIT : NO_INHERITANCE;
	user_access.Trustee.TrusteeForm = TRUSTEE_IS_SID;
	user_access.Trustee.TrusteeType = TRUSTEE_IS_USER;
	user_access.Trustee.ptstrName = static_cast<LPWSTR>(token_user->User.Sid);

	PACL private_acl = nullptr;
	status = SetEntriesInAclW(1, &user_access, nullptr, &private_acl);
	if (status != ERROR_SUCCESS || !private_acl) {
		if (private_acl) {
			LocalFree(private_acl);
		}
		throw IOException("Could not create private ACL for extension cache path '%s' (Windows error %u)", path,
		                  status);
	}

	status = SetNamedSecurityInfoW(const_cast<LPWSTR>(windows_path.c_str()), SE_FILE_OBJECT,
	                               DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION, nullptr, nullptr,
	                               private_acl, nullptr);
	LocalFree(private_acl);
	if (status != ERROR_SUCCESS) {
		throw IOException("Could not secure extension cache path '%s' (Windows error %u)", path, status);
	}
#else
	(void)directory;
	throw NotImplementedException("Windows DACLs are unavailable for extension cache path '%s'", path);
#endif
}

static bool DynamicExtensionCachePathIsReplaceable(const string &path) {
#ifdef DUCKDB_WINDOWS
	auto windows_path = WindowsUtil::UTF8ToUnicode(path.c_str());
	auto attributes = GetFileAttributesW(windows_path.c_str());
	if (attributes == INVALID_FILE_ATTRIBUTES) {
		throw IOException("Could not inspect extension cache ancestor '%s' (Windows error %u)", path, GetLastError());
	}
	if (!(attributes & FILE_ATTRIBUTE_DIRECTORY) || (attributes & FILE_ATTRIBUTE_REPARSE_POINT)) {
		return true;
	}

	auto process_user_buffer = DynamicExtensionProcessUserSid(path);
	auto process_user = reinterpret_cast<TOKEN_USER *>(process_user_buffer.data());
	auto local_system_sid = DynamicExtensionWellKnownSid(WinLocalSystemSid, path);
	auto administrators_sid = DynamicExtensionWellKnownSid(WinBuiltinAdministratorsSid, path);
	// Windows system-volume roots can be owned by the Windows Modules Installer service.
	auto trusted_installer_sid = DynamicExtensionAccountSid(L"NT SERVICE\\TrustedInstaller", path);
	auto creator_owner_sid = DynamicExtensionWellKnownSid(WinCreatorOwnerSid, path);
	auto creator_owner_rights_sid = DynamicExtensionWellKnownSid(WinCreatorOwnerRightsSid, path);

	PSID owner = nullptr;
	PACL dacl = nullptr;
	PSECURITY_DESCRIPTOR security_descriptor = nullptr;
	auto status = GetNamedSecurityInfoW(const_cast<LPWSTR>(windows_path.c_str()), SE_FILE_OBJECT,
	                                    OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION, &owner, nullptr, &dacl,
	                                    nullptr, &security_descriptor);
	if (status != ERROR_SUCCESS) {
		throw IOException("Could not read security information for extension cache ancestor '%s' (Windows error %u)",
		                  path, status);
	}

	bool replaceable = !dacl;
	auto owner_trusted = DynamicExtensionCacheTrustedPrincipal(owner, process_user->User.Sid, local_system_sid,
	                                                           administrators_sid, trusted_installer_sid);
	if (!owner_trusted) {
		replaceable = true;
	}

	PEXPLICIT_ACCESSW entries = nullptr;
	ULONG entry_count = 0;
	if (!replaceable) {
		status = GetExplicitEntriesFromAclW(dacl, &entry_count, &entries);
		if (status != ERROR_SUCCESS) {
			LocalFree(security_descriptor);
			throw IOException("Could not inspect permissions for extension cache ancestor '%s' (Windows error %u)",
			                  path, status);
		}
	}

	GENERIC_MAPPING file_mapping {FILE_GENERIC_READ, FILE_GENERIC_WRITE, FILE_GENERIC_EXECUTE, FILE_ALL_ACCESS};
	constexpr ACCESS_MASK replacement_rights = DELETE | FILE_DELETE_CHILD | WRITE_DAC | WRITE_OWNER;
	for (ULONG index = 0; !replaceable && index < entry_count; index++) {
		auto &entry = entries[index];
		if ((entry.grfAccessMode != GRANT_ACCESS && entry.grfAccessMode != SET_ACCESS) ||
		    (entry.grfInheritance & INHERIT_ONLY_ACE)) {
			continue;
		}
		auto permissions = entry.grfAccessPermissions;
		MapGenericMask(&permissions, &file_mapping);
		if (!(permissions & replacement_rights)) {
			continue;
		}

		PSID trustee_sid = nullptr;
		if (entry.Trustee.TrusteeForm == TRUSTEE_IS_SID) {
			trustee_sid = entry.Trustee.ptstrName;
		} else if (entry.Trustee.TrusteeForm == TRUSTEE_IS_OBJECTS_AND_SID && entry.Trustee.ptstrName) {
			auto objects_and_sid = reinterpret_cast<OBJECTS_AND_SID *>(entry.Trustee.ptstrName);
			trustee_sid = objects_and_sid->pSid;
		}
		if (!DynamicExtensionCacheTrustedSid(trustee_sid, owner_trusted ? owner : nullptr, process_user->User.Sid,
		                                     local_system_sid, administrators_sid, trusted_installer_sid,
		                                     creator_owner_sid, creator_owner_rights_sid)) {
			replaceable = true;
		}
	}

	if (entries) {
		LocalFree(entries);
	}
	LocalFree(security_descriptor);
	return replaceable;
#else
	throw NotImplementedException("Windows DACLs are unavailable for extension cache ancestor '%s'", path);
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
		    // Vane creates every missing cache component with private permissions; DuckDB's ExtensionDirectory helper
		    // would create the whole path first using the process umask.
		    auto extension_directories =
		        ExtensionHelper::GetExtensionDirectoryPath(DynamicExtensionContext(connection));
		    if (extension_directories.empty()) {
			    throw InternalException("DuckDB returned no configured extension directories");
		    }
		    return extension_directories[0];
	    },
	    py::kw_only(), py::arg("connection").none(false));
	module.def("_dynamic_extension_platform", &DuckDB::Platform);
	module.def("_dynamic_extension_compatibility_version", &ExtensionHelper::GetVersionDirectoryName);
	module.def("_secure_dynamic_extension_cache_path", &SecureDynamicExtensionCachePath, py::arg("path"), py::kw_only(),
	           py::arg("directory"));
	module.def("_dynamic_extension_cache_path_is_replaceable", &DynamicExtensionCachePathIsReplaceable,
	           py::arg("path"));
	module.def("_inspect_dynamic_extension", &InspectDynamicExtension, py::arg("path"));
	module.def("_load_dynamic_extension", &LoadDynamicExtension, py::arg("path"), py::kw_only(),
	           py::arg("connection").none(false));
	module.def("_loaded_dynamic_extension", &LoadedDynamicExtension, py::arg("extension"), py::kw_only(),
	           py::arg("connection").none(false));
}

} // namespace duckdb

// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "file_list_function.hpp"

#include "file_mime_type.hpp"
#include "file_resolver.hpp"
#include "file_value.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/enums/file_glob_options.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/multi_file/multi_file_list.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/timestamp.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/function/distributed_table_function.hpp"
#include "duckdb/main/client_context.hpp"

#ifdef _WIN32
#include "duckdb/common/windows_util.hpp"
#endif

#include <algorithm>

#ifndef _WIN32
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace duckdb {

namespace {

static constexpr idx_t DISTRIBUTED_FILE_LIST_PROTOCOL_VERSION = 1;
static constexpr idx_t DISTRIBUTED_FILE_LIST_PAYLOAD_VERSION = 1;
static constexpr idx_t DISTRIBUTED_FILE_LIST_SPLIT_CODEC_VERSION = 1;
static constexpr const char *DISTRIBUTED_FILE_LIST_SPLIT_CODEC = "vane.file-list-rows";

struct ListedFile {
	FileStatValue stat;
	Value file;

	bool operator==(const ListedFile &other) const {
		return stat.url == other.stat.url && stat.has_object_size == other.stat.has_object_size &&
		       (!stat.has_object_size || stat.object_size == other.stat.object_size) &&
		       stat.has_last_modified == other.stat.has_last_modified &&
		       (!stat.has_last_modified || stat.last_modified == other.stat.last_modified) &&
		       stat.has_version == other.stat.has_version &&
		       (!stat.has_version || stat.version == other.stat.version) && stat.has_etag == other.stat.has_etag &&
		       (!stat.has_etag || stat.etag == other.stat.etag) &&
		       stat.has_content_type == other.stat.has_content_type &&
		       (!stat.has_content_type || stat.content_type == other.stat.content_type) && file == other.file;
	}
};

struct FileListBindData : public TableFunctionData {
	vector<string> paths;
	bool recursive = false;
	bool coordinator_files_materialized = false;
	bool distributed_worker = false;
	bool distributed_splits_applied = false;
	vector<ListedFile> files;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<FileListBindData>(*this);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto other = dynamic_cast<const FileListBindData *>(&other_p);
		return other && paths == other->paths && recursive == other->recursive &&
		       coordinator_files_materialized == other->coordinator_files_materialized &&
		       distributed_worker == other->distributed_worker &&
		       distributed_splits_applied == other->distributed_splits_applied && files == other->files &&
		       column_ids == other->column_ids;
	}
};

struct FileListGlobalState : public GlobalTableFunctionState {
	vector<ListedFile> files;
	idx_t offset = 0;
};

static void ValidatePath(const string &path, const string &function_name) {
	if (path.empty()) {
		throw InvalidInputException("%s() path cannot be empty", function_name);
	}
	if (path.find('\0') != string::npos) {
		throw InvalidInputException("%s() path cannot contain NUL bytes", function_name);
	}
}

static void ValidateFileStat(const FileStatValue &stat, const string &source) {
	ValidatePath(stat.url, source);
	if (stat.has_last_modified && !Timestamp::IsFinite(stat.last_modified)) {
		throw InvalidInputException("%s contains a non-finite last_modified value", source);
	}
	if ((stat.has_version && stat.version.empty()) || (stat.has_etag && stat.etag.empty()) ||
	    (stat.has_content_type && stat.content_type.empty())) {
		throw InvalidInputException("%s contains an empty optional metadata value", source);
	}
}

static void SerializeFileStat(Serializer &serializer, const FileStatValue &stat) {
	ValidateFileStat(stat, "list_files() metadata");
	serializer.WriteProperty(1, "url", stat.url);
	serializer.WriteProperty(2, "has_object_size", stat.has_object_size);
	serializer.WriteProperty(3, "object_size", stat.has_object_size ? stat.object_size : 0);
	serializer.WriteProperty(4, "has_last_modified", stat.has_last_modified);
	serializer.WriteProperty(5, "last_modified_micros", stat.has_last_modified ? stat.last_modified.value : 0);
	serializer.WriteProperty(6, "has_version", stat.has_version);
	serializer.WriteProperty(7, "version", stat.has_version ? stat.version : string());
	serializer.WriteProperty(8, "has_etag", stat.has_etag);
	serializer.WriteProperty(9, "etag", stat.has_etag ? stat.etag : string());
	serializer.WriteProperty(10, "has_content_type", stat.has_content_type);
	serializer.WriteProperty(11, "content_type", stat.has_content_type ? stat.content_type : string());
}

static FileStatValue DeserializeFileStat(Deserializer &deserializer) {
	FileStatValue result;
	result.url = deserializer.ReadProperty<string>(1, "url");
	result.has_object_size = deserializer.ReadProperty<bool>(2, "has_object_size");
	auto object_size = deserializer.ReadProperty<uint64_t>(3, "object_size");
	if (result.has_object_size) {
		result.object_size = object_size;
	}
	result.has_last_modified = deserializer.ReadProperty<bool>(4, "has_last_modified");
	auto last_modified_micros = deserializer.ReadProperty<int64_t>(5, "last_modified_micros");
	if (result.has_last_modified) {
		result.last_modified = timestamp_t(last_modified_micros);
	}
	result.has_version = deserializer.ReadProperty<bool>(6, "has_version");
	auto version = deserializer.ReadProperty<string>(7, "version");
	if (result.has_version) {
		result.version = std::move(version);
	}
	result.has_etag = deserializer.ReadProperty<bool>(8, "has_etag");
	auto etag = deserializer.ReadProperty<string>(9, "etag");
	if (result.has_etag) {
		result.etag = std::move(etag);
	}
	result.has_content_type = deserializer.ReadProperty<bool>(10, "has_content_type");
	auto content_type = deserializer.ReadProperty<string>(11, "content_type");
	if (result.has_content_type) {
		result.content_type = std::move(content_type);
	}
	ValidateFileStat(result, "distributed list_files() metadata");
	return result;
}

static bool IsNativePath(const string &path) {
	auto scheme = path.find("://");
	return scheme == string::npos || StringUtil::CIStartsWith(path, "file://");
}

static optional_idx FindURISuffix(const string &path) {
	auto scheme = path.find("://");
	if (scheme == string::npos ||
	    (!StringUtil::CIStartsWith(path, "http://") && !StringUtil::CIStartsWith(path, "https://"))) {
		return optional_idx();
	}
	// HTTP(S) reserves '?' and '#' for query and fragment components.
	// Connector URLs such as memory:// and s3:// retain them as path syntax.
	auto suffix = path.find_first_of("?#", scheme + 3);
	return suffix == string::npos ? optional_idx() : optional_idx(suffix);
}

static idx_t GlobPathBegin(const string &path) {
	auto scheme = path.find("://");
	if (scheme == string::npos) {
		return 0;
	}
	if (StringUtil::CIStartsWith(path, "http://") || StringUtil::CIStartsWith(path, "https://")) {
		auto authority_end = path.find('/', scheme + 3);
		return authority_end == string::npos ? path.size() : authority_end;
	}
	// Connector URLs commonly use everything after :// as their key rather
	// than an RFC authority, so glob metacharacters there remain significant.
	return scheme + 3;
}

static optional_idx FindPathGlob(const string &path) {
	auto begin = GlobPathBegin(path);
	auto suffix = FindURISuffix(path);
	auto end = suffix.IsValid() ? suffix.GetIndex() : path.size();
	if (end <= begin) {
		return optional_idx();
	}
	for (idx_t index = begin; index < end; index++) {
		switch (path[index]) {
		case '*':
		case '?':
		case '[':
			return optional_idx(index);
		default:
			break;
		}
	}
	return optional_idx();
}

static bool HasPathGlob(const string &path) {
	return FindPathGlob(path).IsValid();
}

static string EscapeLiteralGlobPath(const string &path) {
	auto begin = GlobPathBegin(path);
	auto suffix = FindURISuffix(path);
	auto end = suffix.IsValid() ? suffix.GetIndex() : path.size();
	if (end <= begin) {
		return path;
	}
	string result = path.substr(0, begin);
	for (idx_t index = begin; index < end; index++) {
		switch (path[index]) {
		case '*':
			result += "[*]";
			break;
		case '?':
			result += "[?]";
			break;
		case '[':
			result += "[[]";
			break;
		default:
			result += path[index];
			break;
		}
	}
	result += path.substr(end);
	return result;
}

static string AppendPathComponent(FileSystem &file_system, const string &path, const string &component) {
	auto uri_suffix = FindURISuffix(path);
	auto base = uri_suffix.IsValid() ? path.substr(0, uri_suffix.GetIndex()) : path;
	auto suffix = uri_suffix.IsValid() ? path.substr(uri_suffix.GetIndex()) : string();
	auto separator = file_system.PathSeparator(base);
	if (StringUtil::EndsWith(base, separator)) {
		return base + component + suffix;
	}
	return file_system.JoinPath(base, component) + suffix;
}

static string DiscoveryPathPrefix(FileSystem &file_system, const string &locator) {
	auto path_end = FindURISuffix(locator);
	auto end = path_end.IsValid() ? path_end.GetIndex() : locator.size();
	auto glob = FindPathGlob(locator);
	auto separator = file_system.PathSeparator(locator);
	if (separator.empty()) {
		return string();
	}
	if (glob.IsValid()) {
		auto separator_offset = locator.rfind(separator, glob.GetIndex());
		if (separator_offset == string::npos) {
			return string();
		}
		return locator.substr(0, separator_offset + separator.size());
	}
	auto prefix = locator.substr(0, end);
	if (!StringUtil::EndsWith(prefix, separator)) {
		prefix += separator;
	}
	return prefix;
}

static string TrimLeadingURLSeparators(const string &path) {
	idx_t offset = 0;
	while (offset < path.size() && path[offset] == '/') {
		offset++;
	}
	return path.substr(offset);
}

// Filesystem adapters can erase the caller's URI spelling while expanding a
// glob (for example, file:// to a native path or memory:// to memory:///). Only
// restore that spelling when the returned key is still below the caller's
// stable, non-glob prefix; otherwise the provider result remains authoritative.
static string NormalizeDiscoveredPath(FileSystem &file_system, const string &locator, const string &discovered_path) {
	auto scheme_separator = locator.find("://");
	if (scheme_separator == string::npos) {
		return discovered_path;
	}
	auto discovered_scheme_separator = discovered_path.find("://");
	if (discovered_scheme_separator != string::npos &&
	    !StringUtil::CIEquals(locator.substr(0, scheme_separator),
	                          discovered_path.substr(0, discovered_scheme_separator))) {
		return discovered_path;
	}

	auto prefix = DiscoveryPathPrefix(file_system, locator);
	if (prefix.empty()) {
		return discovered_path;
	}
	string prefix_key;
	string discovered_key;
	if (IsNativePath(locator)) {
		prefix_key = file_system.ConvertSeparators(file_system.ExpandPath(prefix));
		discovered_key = file_system.ConvertSeparators(file_system.ExpandPath(discovered_path));
	} else {
		prefix_key = TrimLeadingURLSeparators(prefix.substr(scheme_separator + 3));
		discovered_key = TrimLeadingURLSeparators(discovered_scheme_separator == string::npos
		                                              ? discovered_path
		                                              : discovered_path.substr(discovered_scheme_separator + 3));
	}
	if (prefix_key.empty()) {
		return discovered_path;
	}

#ifdef _WIN32
	auto matches_prefix = IsNativePath(locator) ? StringUtil::CIStartsWith(discovered_key, prefix_key)
	                                            : StringUtil::StartsWith(discovered_key, prefix_key);
#else
	auto matches_prefix = StringUtil::StartsWith(discovered_key, prefix_key);
#endif
	if (!matches_prefix) {
		return discovered_path;
	}
	auto suffix = discovered_key.substr(prefix_key.size());
#ifdef _WIN32
	if (IsNativePath(locator)) {
		suffix = StringUtil::Replace(suffix, "\\", "/");
	}
#endif
	return prefix + suffix;
}

static void VerifyNativeDirectoryAccessible(FileSystem &file_system, const string &path) {
#ifndef _WIN32
	auto local_path = file_system.ExpandPath(path);
	if (access(local_path.c_str(), R_OK | X_OK) != 0) {
		throw IOException("list_files() path '%s' exists but is not accessible", path);
	}
#else
	(void)file_system;
	(void)path;
#endif
}

static bool IsNativeSymbolicLink(FileSystem &file_system, const string &path) {
	auto local_path = file_system.ExpandPath(path);
#ifndef _WIN32
	struct stat status;
	return lstat(local_path.c_str(), &status) == 0 && S_ISLNK(status.st_mode);
#else
	auto unicode_path = WindowsUtil::UTF8ToUnicode(local_path.c_str());
	auto attributes = GetFileAttributesW(unicode_path.c_str());
	return attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
#endif
}

static string ResolveListedPath(FileSystem &file_system, const string &directory, const string &path) {
	if (path.find("://") != string::npos) {
		return path;
	}
	if (directory.find("://") != string::npos) {
		// FileSystem::ListFiles reports direct child names. Registered adapters
		// may instead return a complete protocol-stripped path, so retain only
		// its final component and preserve the caller's directory locator.
		auto separator = file_system.PathSeparator(directory);
		if (separator.empty()) {
			return directory + path;
		}
		auto child_path = path;
		while (child_path.size() > separator.size() && StringUtil::EndsWith(child_path, separator)) {
			child_path.resize(child_path.size() - separator.size());
		}
		if (child_path.empty() || child_path == separator) {
			return directory;
		}
		auto separator_offset = child_path.rfind(separator);
		auto child_name =
		    separator_offset == string::npos ? child_path : child_path.substr(separator_offset + separator.size());
		return AppendPathComponent(file_system, directory, child_name);
	}
	if (file_system.IsPathAbsolute(path)) {
		return path;
	}
	auto separator = file_system.PathSeparator(directory);
	auto has_trailing_separator = StringUtil::EndsWith(directory, separator);
#ifdef _WIN32
	has_trailing_separator = has_trailing_separator || StringUtil::EndsWith(directory, separator == "/" ? "\\" : "/");
#endif
	if (has_trailing_separator) {
		return directory + path;
	}
	return file_system.JoinPath(directory, path);
}

static bool TryListConcreteDirectory(ClientContext &context, FileSystem &file_system, const string &path,
                                     bool recursive, vector<OpenFileInfo> &files) {
	files.clear();
	vector<string> directories {path};
	unordered_set<string> scheduled_directories {path};
	try {
		for (idx_t index = 0; index < directories.size(); index++) {
			if (context.IsInterrupted()) {
				throw InterruptException();
			}
			auto directory = directories[index];
			auto native_directory = IsNativePath(directory);
			if (native_directory) {
				VerifyNativeDirectoryAccessible(file_system, directory);
			}
			auto listable = file_system.ListFiles(
			    directory,
			    [&](OpenFileInfo &info) {
				    if (context.IsInterrupted()) {
					    throw InterruptException();
				    }
				    info.path = NormalizeDiscoveredPath(file_system, directory,
				                                        ResolveListedPath(file_system, directory, info.path));
				    if (FileSystem::IsDirectory(info)) {
					    if (recursive && (!native_directory || !IsNativeSymbolicLink(file_system, info.path)) &&
					        scheduled_directories.insert(info.path).second) {
						    directories.push_back(info.path);
					    }
					    return;
				    }
				    files.push_back(std::move(info));
			    },
			    nullptr);
			if (!listable) {
				throw IOException("list_files() path '%s' exists but is not accessible", directory);
			}
		}
	} catch (const NotImplementedException &) {
		files.clear();
		return false;
	}
	std::sort(files.begin(), files.end());
	return true;
}

static bool TryMetadataValue(const OpenFileInfo &info, const string &name, const LogicalType &type, Value &result) {
	if (!info.extended_info) {
		return false;
	}
	auto entry = info.extended_info->options.find(name);
	if (entry == info.extended_info->options.end() || entry->second.IsNull()) {
		return false;
	}
	string error;
	return entry->second.DefaultTryCastAs(type, result, &error, true);
}

static bool TryMetadataString(const OpenFileInfo &info, const string &name, string &result) {
	if (!info.extended_info) {
		return false;
	}
	auto entry = info.extended_info->options.find(name);
	if (entry == info.extended_info->options.end() || entry->second.IsNull() ||
	    entry->second.type().id() == LogicalTypeId::BLOB) {
		return false;
	}
	Value value;
	if (!TryMetadataValue(info, name, LogicalType::VARCHAR, value)) {
		return false;
	}
	result = value.GetValue<string>();
	return !result.empty();
}

static ListedFile MakeListedFile(FileStatValue stat) {
	FileReference file;
	file.url = stat.url;
	if (stat.has_content_type) {
		file.has_content_type = true;
		file.content_type = stat.content_type;
	}
	if (stat.has_object_size && stat.object_size <= NumericLimits<int64_t>::Maximum()) {
		file.has_range = true;
		file.position = 0;
		file.size = NumericCast<int64_t>(stat.object_size);
	}
	return {std::move(stat), file.ToValue()};
}

static ListedFile MakeListedFile(const OpenFileInfo &info) {
	FileStatValue stat;
	stat.url = info.path;

	Value value;
	if (TryMetadataValue(info, "file_size", LogicalType::UBIGINT, value) ||
	    TryMetadataValue(info, "object_size", LogicalType::UBIGINT, value)) {
		stat.has_object_size = true;
		stat.object_size = value.GetValue<uint64_t>();
	}
	if (TryMetadataValue(info, "last_modified", LogicalType::TIMESTAMP, value)) {
		auto timestamp = value.GetValue<timestamp_t>();
		if (Timestamp::IsFinite(timestamp)) {
			stat.has_last_modified = true;
			stat.last_modified = timestamp;
		}
	}
	stat.has_version =
	    TryMetadataString(info, "version", stat.version) || TryMetadataString(info, "version_id", stat.version);
	stat.has_etag = TryMetadataString(info, "etag", stat.etag);
	stat.has_content_type = TryMetadataString(info, "content_type", stat.content_type) ||
	                        TryMetadataString(info, "mime_type", stat.content_type) ||
	                        FileMimeType::FromPath(stat.url, stat.content_type);
	return MakeListedFile(std::move(stat));
}

static vector<OpenFileInfo> ExpandGlob(FileSystem &file_system, const string &pattern, const string &path,
                                       bool explicit_glob, bool directory_semantics) {
	vector<OpenFileInfo> files;
	try {
		auto file_list = file_system.Glob(pattern, FileGlobOptions::ALLOW_EMPTY, nullptr);
		if (!file_list) {
			throw IOException("list_files() filesystem '%s' returned no file list for path '%s'", file_system.GetName(),
			                  path);
		}
		files = file_list->GetAllFiles();
		for (auto &file : files) {
			file.path = NormalizeDiscoveredPath(file_system, path, file.path);
		}
	} catch (const NotImplementedException &) {
		throw NotImplementedException("list_files() filesystem '%s' does not support %s listing for path '%s'",
		                              file_system.GetName(), explicit_glob ? "glob" : "directory", path);
	}
	if (!directory_semantics && files.size() == 1 && files[0].path == pattern) {
		// A supported glob can return an exact key whose name contains wildcard
		// characters. Only classify an unchanged result as fallback when that key
		// does not exist.
		bool exact_file = false;
		try {
			exact_file = file_system.FileExists(pattern);
		} catch (const NotImplementedException &) {
		}
		if (!exact_file) {
			throw NotImplementedException("list_files() filesystem '%s' does not support %s listing for path '%s'",
			                              file_system.GetName(), explicit_glob ? "glob" : "directory", path);
		}
	}
	std::sort(files.begin(), files.end());
	return files;
}

static void VerifyEmptyDirectoryListable(FileSystem &file_system, const string &path) {
	if (IsNativePath(path)) {
		VerifyNativeDirectoryAccessible(file_system, path);
	}
	bool listable;
	try {
		listable = file_system.ListFiles(
		    path, [](OpenFileInfo &) {}, nullptr);
	} catch (const NotImplementedException &) {
		// Some directory-capable filesystems expose globbing without a separate
		// listing probe. The successful empty glob is authoritative for them.
		return;
	}
	if (!listable) {
		throw IOException("list_files() path '%s' exists but is not accessible", path);
	}
}

static bool IsListedDirectory(FileSystem &file_system, const OpenFileInfo &info) {
	if (info.extended_info) {
		// Object stores attach object metadata without a type and may report every
		// prefix as a directory. Only path-only listings need a separate probe.
		return FileSystem::IsDirectory(info);
	}
	try {
		return file_system.DirectoryExists(info.path);
	} catch (const NotImplementedException &) {
		return false;
	}
}

static vector<string> FileSearchPathCandidates(ClientContext &context, FileSystem &file_system, const string &path) {
	vector<string> result;
	if (path.find("://") != string::npos) {
		return result;
	}
	auto expanded_path = file_system.ExpandPath(path);
	if (file_system.IsPathAbsolute(expanded_path)) {
		return result;
	}

	Value value;
	if (!context.TryGetCurrentSetting("file_search_path", value) || value.IsNull()) {
		return result;
	}
	for (const auto &search_path : StringUtil::Split(value.ToString(), ',')) {
		if (!search_path.empty()) {
			result.push_back(file_system.JoinPath(search_path, path));
		}
	}
	return result;
}

static vector<ListedFile> StatConcreteFiles(ClientContext &context, const vector<string> &paths) {
	vector<ListedFile> result;
	result.reserve(paths.size());
	for (const auto &path : paths) {
		if (context.IsInterrupted()) {
			throw InterruptException();
		}
		FileReference file;
		file.url = path;
		result.push_back(MakeListedFile(ResolvedFile::Open(context, file)->Stat()));
	}
	std::sort(result.begin(), result.end(),
	          [](const ListedFile &left, const ListedFile &right) { return left.stat.url < right.stat.url; });
	return result;
}

static vector<ListedFile> DiscoverPath(ClientContext &context, const string &path, bool recursive) {
	ValidatePath(path, "list_files");
	if (context.IsInterrupted()) {
		throw InterruptException();
	}
	auto &file_system = FileSystem::GetFileSystem(context);
	auto explicit_glob = HasPathGlob(path);
	auto directory_semantics = file_system.HasDirectorySemantics(path);

	bool file_exists = false;
	bool file_exists_known = false;
	if (!explicit_glob || directory_semantics) {
		try {
			file_exists = file_system.FileExists(path);
			file_exists_known = true;
		} catch (const NotImplementedException &) {
		}
	}
	if (file_exists) {
		return StatConcreteFiles(context, {path});
	}

	bool directory_exists = false;
	bool directory_exists_known = false;
	if (!explicit_glob || directory_semantics) {
		try {
			directory_exists = file_system.DirectoryExists(path);
			directory_exists_known = true;
		} catch (const NotImplementedException &) {
		}
	}
	auto concrete_path_known_missing = !explicit_glob && !file_exists && file_exists_known && !directory_exists &&
	                                   (!directory_semantics || directory_exists_known);
	auto try_discover_search_path = [&](vector<ListedFile> &result) {
		vector<string> literal_files;
		vector<string> literal_directories;
		for (const auto &candidate : FileSearchPathCandidates(context, file_system, path)) {
			if (context.IsInterrupted()) {
				throw InterruptException();
			}
			try {
				if (file_system.FileExists(candidate)) {
					literal_files.push_back(candidate);
					continue;
				}
			} catch (const NotImplementedException &) {
			}
			if (!file_system.HasDirectorySemantics(candidate)) {
				continue;
			}
			try {
				if (file_system.DirectoryExists(candidate)) {
					literal_directories.push_back(candidate);
				}
			} catch (const NotImplementedException &) {
			}
		}
		if (!literal_files.empty()) {
			result = StatConcreteFiles(context, literal_files);
			return true;
		}
		if (literal_directories.empty()) {
			return false;
		}
		for (const auto &directory : literal_directories) {
			auto discovered = DiscoverPath(context, directory, recursive);
			for (auto &file : discovered) {
				result.push_back(std::move(file));
			}
		}
		std::sort(result.begin(), result.end(),
		          [](const ListedFile &left, const ListedFile &right) { return left.stat.url < right.stat.url; });
		return true;
	};

	// Match direct-path precedence for literal glob-named entries resolved
	// through file_search_path. If no exact candidate exists, the path retains
	// its explicit-glob meaning below.
	if (explicit_glob && directory_semantics && !directory_exists) {
		vector<ListedFile> result;
		if (try_discover_search_path(result)) {
			return result;
		}
	}

	// A context-aware literal glob resolves concrete files through
	// file_search_path without duplicating the filesystem's lookup rules.
	if (!explicit_glob && !directory_exists) {
		vector<OpenFileInfo> literal_matches;
		bool literal_glob_implemented = true;
		try {
			literal_matches = ExpandGlob(file_system, path, path, false, directory_semantics);
		} catch (const NotImplementedException &) {
			literal_glob_implemented = false;
			// The filesystem can still support directory globbing below.
		}
		vector<string> literal_files;
		for (const auto &info : literal_matches) {
			if (!IsListedDirectory(file_system, info)) {
				literal_files.push_back(info.path);
			}
		}
		if (!literal_files.empty()) {
			return StatConcreteFiles(context, literal_files);
		}
		vector<ListedFile> result;
		if (try_discover_search_path(result)) {
			return result;
		}
		if (concrete_path_known_missing && !literal_glob_implemented) {
			throw IOException("list_files() path '%s' does not exist or is not listable", path);
		}
	}

	auto treat_as_glob = explicit_glob && !directory_exists;
	auto discovery_path = path;
	// DirectoryExists established a literal path. Escape its glob metacharacters
	// before adding the wildcard that enumerates the directory.
	auto pattern_base = directory_exists ? EscapeLiteralGlobPath(discovery_path) : discovery_path;
	auto pattern = treat_as_glob ? path : AppendPathComponent(file_system, pattern_base, recursive ? "**" : "*");
	if (recursive && !treat_as_glob && directory_semantics) {
		// A trailing wildcard lets DuckDB's crawl visit only concrete
		// directories while retaining symbolic links to regular files.
		pattern = AppendPathComponent(file_system, pattern, "*");
	}
	vector<OpenFileInfo> discovered;
	auto listed_directly = directory_exists && directory_semantics &&
	                       TryListConcreteDirectory(context, file_system, discovery_path, recursive, discovered);
	if (!listed_directly) {
		try {
			discovered = ExpandGlob(file_system, pattern, path, treat_as_glob, directory_semantics);
		} catch (const NotImplementedException &) {
			if (treat_as_glob) {
				// A provider without glob support can still expose an exact object
				// whose key contains glob metacharacters. Preserve glob behavior when
				// it is implemented, but recover the literal key before reporting the
				// provider limitation.
				try {
					if (file_system.FileExists(path)) {
						return StatConcreteFiles(context, {path});
					}
				} catch (const NotImplementedException &) {
				}
			}
			if (concrete_path_known_missing) {
				throw IOException("list_files() path '%s' does not exist or is not listable", path);
			}
			throw;
		}
	}
	if (discovered.empty() && !treat_as_glob) {
		if (!directory_exists) {
			throw IOException("list_files() path '%s' does not exist or is not listable", path);
		}
		if (directory_semantics && !listed_directly) {
			VerifyEmptyDirectoryListable(file_system, discovery_path);
		}
	}

	vector<ListedFile> result;
	result.reserve(discovered.size());
	for (const auto &info : discovered) {
		if (context.IsInterrupted()) {
			throw InterruptException();
		}
		if (!IsListedDirectory(file_system, info)) {
			result.push_back(MakeListedFile(info));
		}
	}
	return result;
}

static string EncodeDistributedFileListRows(const vector<ListedFile> &files) {
	MemoryStream stream(Allocator::DefaultAllocator());
	BinarySerializer serializer(stream);
	serializer.Begin();
	serializer.WriteProperty(1, "payload_version", DISTRIBUTED_FILE_LIST_PAYLOAD_VERSION);
	serializer.WriteList(2, "files", files.size(), [&](Serializer::List &list, idx_t index) {
		list.WriteObject([&](Serializer &object) { SerializeFileStat(object, files[index].stat); });
	});
	serializer.End();
	return string(reinterpret_cast<const char *>(stream.GetData()), stream.GetPosition());
}

static vector<ListedFile> DecodeDistributedFileListRows(const string &payload) {
	if (payload.empty()) {
		throw InvalidInputException("Distributed list_files() split payload is empty");
	}
	auto data = reinterpret_cast<data_ptr_t>(const_cast<char *>(payload.data()));
	MemoryStream stream(data, payload.size());
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto version = deserializer.ReadProperty<idx_t>(1, "payload_version");
	if (version != DISTRIBUTED_FILE_LIST_PAYLOAD_VERSION) {
		throw InvalidInputException("Unsupported distributed list_files() payload version %llu", version);
	}
	vector<ListedFile> result;
	deserializer.ReadList(2, "files", [&](Deserializer::List &list, idx_t) {
		list.ReadObject([&](Deserializer &object) { result.push_back(MakeListedFile(DeserializeFileStat(object))); });
	});
	deserializer.End();
	if (stream.GetPosition() != payload.size()) {
		throw InvalidInputException("Distributed list_files() split payload has trailing bytes");
	}
	return result;
}

static vector<ListedFile> DiscoverFileListRows(ClientContext &context, const FileListBindData &bind_data) {
	vector<ListedFile> result;
	for (const auto &path : bind_data.paths) {
		auto discovered = DiscoverPath(context, path, bind_data.recursive);
		for (auto &file : discovered) {
			result.push_back(std::move(file));
		}
	}
	return result;
}

static vector<DistributedScanSplit>
PlanDistributedFileListSplits(const TableFunctionDistributedScanPlanningInput &input) {
	if (!input.bind_data) {
		throw InvalidInputException("Distributed list_files() planning requires bind data");
	}
	auto &bind_data = input.bind_data->Cast<FileListBindData>();
	vector<ListedFile> files;
	if (bind_data.distributed_worker) {
		if (!bind_data.distributed_splits_applied) {
			throw InvalidInputException("Distributed list_files() worker bind has no applied split assignment");
		}
		files = bind_data.files;
	} else if (bind_data.coordinator_files_materialized) {
		files = bind_data.files;
	} else {
		if (!input.client_context) {
			throw InvalidInputException("Distributed list_files() planning requires a coordinator ClientContext");
		}
		files = DiscoverFileListRows(*input.client_context.get_mutable(), bind_data);
	}
	if (files.empty()) {
		return {};
	}
	DistributedScanSplit split;
	split.split_id = "0";
	split.payload = EncodeDistributedFileListRows(files);
	split.estimated_cardinality = optional_idx(files.size());
	split.estimated_bytes = optional_idx(NumericCast<idx_t>(split.payload.size()));
	split.Validate();
	return {std::move(split)};
}

static unique_ptr<FunctionData> CreateDistributedFileListWorkerBind(const TableFunctionDistributedScanInput &input) {
	if (!input.bind_data) {
		throw InvalidInputException("Distributed list_files() worker bind requires coordinator bind data");
	}
	auto &source = input.bind_data->Cast<FileListBindData>();
	auto result = make_uniq<FileListBindData>();
	result->recursive = source.recursive;
	result->distributed_worker = true;
	result->column_ids = source.column_ids;
	return std::move(result);
}

static void ApplyDistributedFileListSplits(optional_ptr<FunctionData> worker_bind_data,
                                           const vector<DistributedScanSplit> &splits) {
	if (!worker_bind_data) {
		throw InvalidInputException("Distributed list_files() splits require worker bind data");
	}
	auto &worker = worker_bind_data->Cast<FileListBindData>();
	if (!worker.distributed_worker || worker.coordinator_files_materialized || !worker.paths.empty()) {
		throw InvalidInputException("Distributed list_files() splits require a detached worker bind");
	}

	if (splits.empty()) {
		worker.files.clear();
		worker.distributed_splits_applied = true;
		return;
	}
	if (splits.size() != 1 || splits[0].split_id != "0") {
		throw InvalidInputException("Distributed list_files() requires exactly one canonical split");
	}
	splits[0].Validate();
	worker.files = DecodeDistributedFileListRows(splits[0].payload);
	worker.distributed_splits_applied = true;
}

static TableFunctionDistributedScanCallbacks DistributedFileListScanCallbacks() {
	TableFunctionDistributedScanCallbacks callbacks;
	callbacks.protocol_version = DISTRIBUTED_FILE_LIST_PROTOCOL_VERSION;
	callbacks.split_codec = {DISTRIBUTED_FILE_LIST_SPLIT_CODEC, DISTRIBUTED_FILE_LIST_SPLIT_CODEC_VERSION};
	callbacks.plan_splits = PlanDistributedFileListSplits;
	callbacks.create_worker_bind = CreateDistributedFileListWorkerBind;
	callbacks.apply_splits = ApplyDistributedFileListSplits;
	return callbacks;
}

static vector<string> BindPaths(const Value &value) {
	if (value.IsNull()) {
		throw BinderException("list_files() path cannot be NULL");
	}
	vector<string> result;
	if (value.type().id() == LogicalTypeId::VARCHAR) {
		result.push_back(value.GetValue<string>());
	} else {
		D_ASSERT(value.type().id() == LogicalTypeId::LIST);
		for (const auto &path : ListValue::GetChildren(value)) {
			if (path.IsNull()) {
				throw BinderException("list_files() path list cannot contain NULL");
			}
			result.push_back(path.GetValue<string>());
		}
	}
	for (const auto &path : result) {
		ValidatePath(path, "list_files");
	}
	return result;
}

static unique_ptr<FunctionData> FileListBind(ClientContext &, TableFunctionBindInput &input,
                                             vector<LogicalType> &return_types, vector<string> &names) {
	auto result = make_uniq<FileListBindData>();
	result->paths = BindPaths(input.inputs[0]);
	if (input.inputs.size() == 2) {
		if (input.inputs[1].IsNull()) {
			throw BinderException("list_files() recursive cannot be NULL");
		}
		result->recursive = input.inputs[1].GetValue<bool>();
	} else {
		auto recursive = input.named_parameters.find("recursive");
		if (recursive != input.named_parameters.end()) {
			if (recursive->second.IsNull()) {
				throw BinderException("list_files() recursive cannot be NULL");
			}
			result->recursive = recursive->second.GetValue<bool>();
		}
	}

	names = {"url", "object_size", "last_modified", "version", "etag", "content_type", "file"};
	return_types = {LogicalType::VARCHAR, LogicalType::UBIGINT, LogicalType::TIMESTAMP,   LogicalType::VARCHAR,
	                LogicalType::VARCHAR, LogicalType::VARCHAR, FileLogicalType::Create()};
	return std::move(result);
}

static unique_ptr<GlobalTableFunctionState> FileListInit(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<FileListBindData>();
	auto result = make_uniq<FileListGlobalState>();
	if (bind_data.distributed_worker) {
		if (!bind_data.distributed_splits_applied) {
			throw InvalidInputException("Distributed list_files() worker bind has no applied split assignment");
		}
		result->files = bind_data.files;
	} else if (bind_data.coordinator_files_materialized) {
		result->files = bind_data.files;
	} else {
		result->files = DiscoverFileListRows(context, bind_data);
	}
	return std::move(result);
}

static Value OptionalStringValue(bool has_value, const string &value) {
	return has_value ? Value(value) : Value(LogicalType::VARCHAR);
}

static void FileListScan(ClientContext &, TableFunctionInput &input, DataChunk &output) {
	auto &state = input.global_state->Cast<FileListGlobalState>();
	auto count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, state.files.size() - state.offset);
	for (idx_t row = 0; row < count; row++) {
		auto &listed = state.files[state.offset + row];
		auto &stat = listed.stat;
		output.SetValue(0, row, Value(stat.url));
		output.SetValue(1, row, stat.has_object_size ? Value::UBIGINT(stat.object_size) : Value(LogicalType::UBIGINT));
		output.SetValue(2, row,
		                stat.has_last_modified ? Value::TIMESTAMP(stat.last_modified) : Value(LogicalType::TIMESTAMP));
		output.SetValue(3, row, OptionalStringValue(stat.has_version, stat.version));
		output.SetValue(4, row, OptionalStringValue(stat.has_etag, stat.etag));
		output.SetValue(5, row, OptionalStringValue(stat.has_content_type, stat.content_type));
		output.SetValue(6, row, listed.file);
	}
	state.offset += count;
	output.SetCardinality(count);
}

static void FileListSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data_p,
                              const TableFunction &) {
	if (!bind_data_p) {
		throw SerializationException("list_files() requires bind data");
	}
	auto &bind_data = bind_data_p->Cast<FileListBindData>();
	auto coordinator_files_materialized = bind_data.coordinator_files_materialized;
	auto files = bind_data.files;
	if (!bind_data.distributed_worker && !coordinator_files_materialized) {
		auto context = serializer.GetSerializationData().TryGet<ClientContext>();
		if (context) {
			files = DiscoverFileListRows(*context, bind_data);
			coordinator_files_materialized = true;
		}
	}
	if ((!bind_data.distributed_worker &&
	     (bind_data.distributed_splits_applied || (!coordinator_files_materialized && !files.empty()))) ||
	    (bind_data.distributed_worker && (coordinator_files_materialized || !bind_data.paths.empty())) ||
	    (bind_data.distributed_worker && !bind_data.distributed_splits_applied && !files.empty())) {
		throw SerializationException("list_files() bind contains invalid distributed state");
	}
	serializer.WriteProperty(101, "paths", bind_data.paths);
	serializer.WriteProperty(102, "recursive", bind_data.recursive);
	serializer.WriteProperty(103, "coordinator_files_materialized", coordinator_files_materialized);
	serializer.WriteProperty(104, "distributed_worker", bind_data.distributed_worker);
	serializer.WriteProperty(105, "distributed_splits_applied", bind_data.distributed_splits_applied);
	serializer.WriteList(106, "files", files.size(), [&](Serializer::List &list, idx_t index) {
		list.WriteObject([&](Serializer &object) { SerializeFileStat(object, files[index].stat); });
	});
}

static unique_ptr<FunctionData> FileListDeserialize(Deserializer &deserializer, TableFunction &) {
	auto result = make_uniq<FileListBindData>();
	result->paths = deserializer.ReadProperty<vector<string>>(101, "paths");
	result->recursive = deserializer.ReadProperty<bool>(102, "recursive");
	result->coordinator_files_materialized = deserializer.ReadProperty<bool>(103, "coordinator_files_materialized");
	result->distributed_worker = deserializer.ReadProperty<bool>(104, "distributed_worker");
	result->distributed_splits_applied = deserializer.ReadProperty<bool>(105, "distributed_splits_applied");
	deserializer.ReadList(106, "files", [&](Deserializer::List &files, idx_t) {
		files.ReadObject(
		    [&](Deserializer &object) { result->files.push_back(MakeListedFile(DeserializeFileStat(object))); });
	});
	for (const auto &path : result->paths) {
		ValidatePath(path, "list_files");
	}
	if ((!result->distributed_worker &&
	     (result->distributed_splits_applied || (!result->coordinator_files_materialized && !result->files.empty()))) ||
	    (result->distributed_worker && (result->coordinator_files_materialized || !result->paths.empty())) ||
	    (result->distributed_worker && !result->distributed_splits_applied && !result->files.empty())) {
		throw SerializationException("list_files() bind contains invalid distributed state");
	}
	return std::move(result);
}

static TableFunction MakeFileListFunction(vector<LogicalType> arguments, bool named_recursive) {
	TableFunction result("list_files", std::move(arguments), FileListScan, FileListBind, FileListInit);
	if (named_recursive) {
		result.named_parameters["recursive"] = LogicalType::BOOLEAN;
	}
	result.serialize = FileListSerialize;
	result.deserialize = FileListDeserialize;
	result.SetDistributedScanCallbacks(DistributedFileListScanCallbacks());
	return result;
}

} // namespace

vector<TableFunction> FileListFunction::GetFunctions() {
	vector<TableFunction> result;
	result.push_back(MakeFileListFunction({LogicalType::VARCHAR}, true));
	result.push_back(MakeFileListFunction({LogicalType::VARCHAR, LogicalType::BOOLEAN}, false));
	result.push_back(MakeFileListFunction({LogicalType::LIST(LogicalType::VARCHAR)}, true));
	return result;
}

} // namespace duckdb

// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#include "vane_python/pyfilesystem.hpp"

#include "duckdb/common/string_util.hpp"
#include "vane_python/pybind11/pybind_wrapper.hpp"
#include "vane_python/pybind11/gil_wrapper.hpp"

namespace duckdb {

PythonFileHandle::PythonFileHandle(FileSystem &file_system, const string &path, const py::object &handle,
                                   FileOpenFlags flags)
    : FileHandle(file_system, path, flags), handle(handle) {
}
PythonFileHandle::~PythonFileHandle() {
	try {
		PythonGILWrapper gil;
		handle.dec_ref();
		handle.release();
	} catch (...) { // NOLINT
	}
}

const py::object &PythonFileHandle::GetHandle(const FileHandle &handle) {
	return handle.Cast<PythonFileHandle>().handle;
}

void PythonFileHandle::Close() {
	PythonGILWrapper gil;
	handle.attr("close")();
}

PythonFilesystem::~PythonFilesystem() {
	try {
		PythonGILWrapper gil;
		filesystem.dec_ref();
		filesystem.release();
	} catch (...) { // NOLINT
	}
}

string PythonFilesystem::DecodeFlags(FileOpenFlags flags) {
	// see https://stackoverflow.com/a/58925279 for truth table of python file modes
	bool read = flags.OpenForReading();
	bool write = flags.OpenForWriting();
	bool append = flags.OpenForAppending();
	bool truncate = flags.OverwriteExistingFile();

	string flags_s;
	if (read && write && truncate) {
		flags_s = "w+";
	} else if (read && write && append) {
		flags_s = "a+";
	} else if (read && write) {
		flags_s = "r+";
	} else if (read) {
		flags_s = "r";
	} else if (write) {
		flags_s = "w";
	} else if (append) {
		flags_s = "a";
	} else {
		throw InvalidInputException("%s: unsupported file flags", GetName());
	}

	flags_s.insert(1, "b"); // always read in binary mode

	return flags_s;
}

unique_ptr<FileHandle> PythonFilesystem::OpenFile(const string &path, FileOpenFlags flags,
                                                  optional_ptr<FileOpener> opener) {
	if (flags.OpenNonBlocking()) {
		throw NotImplementedException("Nonblocking opens are not supported by registered Python filesystems");
	}
	PythonGILWrapper gil;

	if (flags.Compression() != FileCompressionType::UNCOMPRESSED) {
		throw IOException("Compression not supported");
	}
	// maybe this can be implemented in a better way?
	if (flags.ReturnNullIfNotExists()) {
		if (!FileExists(path)) {
			return nullptr;
		}
	}

	// TODO: lock support?

	string flags_s = DecodeFlags(flags);

	const auto &handle = filesystem.attr("open")(path, py::str(flags_s));
	return make_uniq<PythonFileHandle>(*this, path, handle, flags);
}

int64_t PythonFilesystem::Write(FileHandle &handle, void *buffer, int64_t nr_bytes) {
	PythonGILWrapper gil;

	const auto &write = PythonFileHandle::GetHandle(handle).attr("write");

	auto data = py::bytes(std::string(const_char_ptr_cast(buffer), nr_bytes));

	return py::int_(write(data));
}
void PythonFilesystem::Write(FileHandle &handle, void *buffer, int64_t nr_bytes, idx_t location) {
	PythonGILWrapper gil;
	auto &py_handle = PythonFileHandle::GetHandle(handle);
	py_handle.attr("seek")(location);
	auto data = py::bytes(std::string(const_char_ptr_cast(buffer), nr_bytes));
	py_handle.attr("write")(data);
}

int64_t PythonFilesystem::Read(FileHandle &handle, void *buffer, int64_t nr_bytes) {
	PythonGILWrapper gil;

	const auto &read = PythonFileHandle::GetHandle(handle).attr("read");

	string data = py::bytes(read(nr_bytes));

	memcpy(buffer, data.c_str(), data.size());

	return data.size();
}

void PythonFilesystem::Read(duckdb::FileHandle &handle, void *buffer, int64_t nr_bytes, uint64_t location) {
	PythonGILWrapper gil;
	auto &py_handle = PythonFileHandle::GetHandle(handle);
	py_handle.attr("seek")(location);
	string data = py::bytes(py_handle.attr("read")(nr_bytes));
	memcpy(buffer, data.c_str(), data.size());
}
bool PythonFilesystem::FileExists(const string &filename, optional_ptr<FileOpener> opener) {
	return Exists(filename, "isfile");
}
bool PythonFilesystem::Exists(const string &filename, const char *func_name) const {
	PythonGILWrapper gil;

	try {
		return py::bool_(filesystem.attr(func_name)(filename));
	} catch (const py::error_already_set &error) {
		if (!error.matches(PyExc_NotImplementedError)) {
			throw;
		}
		throw NotImplementedException("Registered Python filesystem '%s' does not implement %s()", GetName(),
		                              func_name);
	}
}

static string StablePathKeyPrefix(const string &path, bool glob_pattern) {
	auto glob = glob_pattern ? path.find_first_of("*?[") : string::npos;
	if (glob == string::npos) {
		return path;
	}
	auto separator = path.rfind('/', glob);
	return separator == string::npos ? string() : path.substr(0, separator + 1);
}

static bool IsSameOrChildPathKey(const string &path, const string &prefix) {
	if (prefix.empty() || !StringUtil::StartsWith(path, prefix)) {
		return false;
	}
	return path.size() == prefix.size() || StringUtil::EndsWith(prefix, "/") || path[prefix.size()] == '/';
}

// Registered fsspec protocol identifiers can contain underscores even though
// RFC URL schemes cannot.
static optional_idx FindLeadingProtocolSeparator(const string &path) {
	auto separator = path.find("://");
	if (separator == string::npos || separator == 0 || !StringUtil::CharacterIsAlpha(path[0])) {
		return optional_idx();
	}
	for (idx_t index = 1; index < separator; index++) {
		auto character = path[index];
		if (!StringUtil::CharacterIsAlphaNumeric(character) && character != '+' && character != '-' &&
		    character != '.' && character != '_') {
			return optional_idx();
		}
	}
	return optional_idx(separator);
}

static string URLAuthority(const string &path, idx_t scheme_separator) {
	auto authority_begin = scheme_separator + 3;
	auto authority_end = path.find_first_of("/?#", authority_begin);
	return path.substr(authority_begin, authority_end == string::npos ? string::npos : authority_end - authority_begin);
}

static bool HasCallerIdentityPrefix(const string &path) {
	auto scheme_separator = FindLeadingProtocolSeparator(path);
	if (!scheme_separator.IsValid()) {
		return true;
	}
	auto offset = scheme_separator.GetIndex() + 3;
	while (offset < path.size() && path[offset] == '/') {
		offset++;
	}
	return offset < path.size();
}

// fsspec unstrip_protocol() always selects the provider's first protocol and
// can omit an authority owned by the configured instance. Compare paths in the
// provider's stripped key space, then restore only the caller's stable locator
// prefix. Results outside that prefix retain the provider representation.
string PythonFilesystem::RestoreCallerPath(const string &locator, const string &returned_path,
                                           const string &fallback_path, bool glob_pattern) const {
	D_ASSERT(py::gil_check());
	auto returned_scheme_separator = FindLeadingProtocolSeparator(returned_path);
	if (returned_scheme_separator.IsValid()) {
		auto returned_protocol = returned_path.substr(0, returned_scheme_separator.GetIndex());
		bool supported_protocol = false;
		for (const auto &protocol : protocols) {
			if (StringUtil::CIEquals(protocol, returned_protocol)) {
				supported_protocol = true;
				break;
			}
		}
		if (!supported_protocol) {
			return fallback_path;
		}

		auto locator_scheme_separator = FindLeadingProtocolSeparator(locator);
		if (locator_scheme_separator.IsValid()) {
			auto locator_authority = URLAuthority(locator, locator_scheme_separator.GetIndex());
			auto returned_authority = URLAuthority(returned_path, returned_scheme_separator.GetIndex());
			if (!locator_authority.empty() && !returned_authority.empty() && locator_authority != returned_authority) {
				return fallback_path;
			}
		}
	}

	auto strip_protocol = filesystem.attr("_strip_protocol");
	string locator_key = py::str(strip_protocol(py::str(locator)));
	string returned_key = py::str(strip_protocol(py::str(returned_path)));
	if (locator_key.empty()) {
		return fallback_path;
	}
	auto stable_prefix = StablePathKeyPrefix(locator_key, glob_pattern);
	if (!IsSameOrChildPathKey(returned_key, stable_prefix)) {
		return fallback_path;
	}

	auto caller_prefix = locator;
	if (stable_prefix.size() < locator_key.size()) {
		auto pattern_suffix = locator_key.substr(stable_prefix.size());
		if (!StringUtil::EndsWith(locator, pattern_suffix)) {
			return fallback_path;
		}
		caller_prefix.resize(locator.size() - pattern_suffix.size());
	}
	if (stable_prefix == "/" && !HasCallerIdentityPrefix(caller_prefix)) {
		return fallback_path;
	}
	auto returned_suffix = returned_key.substr(stable_prefix.size());
	if (StringUtil::EndsWith(caller_prefix, "/") && StringUtil::StartsWith(returned_suffix, "/")) {
		returned_suffix.erase(0, 1);
	}
	auto restored_path = caller_prefix + returned_suffix;
	if (py::str(strip_protocol(py::str(restored_path))).cast<string>() != returned_key) {
		return fallback_path;
	}
	return restored_path;
}

vector<OpenFileInfo> PythonFilesystem::Glob(const string &path, FileOpener *opener) {
	PythonGILWrapper gil;

	if (path.empty()) {
		return {path};
	}
	try {
		auto returner = py::list(filesystem.attr("glob")(path));

		vector<OpenFileInfo> results;
		auto unstrip_protocol = filesystem.attr("unstrip_protocol");
		for (auto item : returner) {
			string returned_path = py::str(item);
			string fallback_path = returned_path;
			if (!FindLeadingProtocolSeparator(returned_path).IsValid()) {
				fallback_path = py::str(unstrip_protocol(py::str(item)));
			}
			results.emplace_back(RestoreCallerPath(path, returned_path, fallback_path, true));
		}
		return results;
	} catch (const py::error_already_set &error) {
		if (!error.matches(PyExc_NotImplementedError)) {
			throw;
		}
		throw NotImplementedException("Registered Python filesystem '%s' does not implement glob()", GetName());
	}
}
string PythonFilesystem::PathSeparator(const string &path) {
	return "/";
}
int64_t PythonFilesystem::GetFileSize(FileHandle &handle) {
	D_ASSERT(!py::gil_check());
	// TODO: this value should be cached on the PythonFileHandle
	PythonGILWrapper gil;

	return py::int_(filesystem.attr("size")(handle.path));
}
void PythonFilesystem::Seek(duckdb::FileHandle &handle, uint64_t location) {
	D_ASSERT(!py::gil_check());
	PythonGILWrapper gil;

	auto seek = PythonFileHandle::GetHandle(handle).attr("seek");
	seek(location);
	if (PyErr_Occurred()) {
		PyErr_PrintEx(1);
		throw InvalidInputException("Python exception occurred!");
	}
}
bool PythonFilesystem::CanHandleFile(const string &fpath) {
	for (const auto &protocol : protocols) {
		if (StringUtil::StartsWith(fpath, protocol + "://")) {
			return true;
		}
	}
	return false;
}
void PythonFilesystem::MoveFile(const string &source, const string &dest, optional_ptr<FileOpener> opener) {
	D_ASSERT(!py::gil_check());
	PythonGILWrapper gil;

	auto move = filesystem.attr("mv");
	move(py::str(source), py::str(dest));
}
void PythonFilesystem::RemoveFile(const string &filename, optional_ptr<FileOpener> opener) {
	D_ASSERT(!py::gil_check());
	PythonGILWrapper gil;

	auto remove = filesystem.attr("rm");
	remove(py::str(filename));
}
timestamp_t PythonFilesystem::GetLastModifiedTime(FileHandle &handle) {
	D_ASSERT(!py::gil_check());
	// TODO: this value should be cached on the PythonFileHandle
	PythonGILWrapper gil;

	auto last_mod = filesystem.attr("modified")(handle.path);

	return Timestamp::FromEpochSeconds(py::int_(last_mod.attr("timestamp")()));
}
void PythonFilesystem::FileSync(FileHandle &handle) {
	D_ASSERT(!py::gil_check());
	PythonGILWrapper gil;

	PythonFileHandle::GetHandle(handle).attr("flush")();
}
bool PythonFilesystem::DirectoryExists(const string &directory, optional_ptr<FileOpener> opener) {
	return Exists(directory, "isdir");
}
void PythonFilesystem::RemoveDirectory(const string &directory, optional_ptr<FileOpener> opener) {
	D_ASSERT(!py::gil_check());
	PythonGILWrapper gil;

	filesystem.attr("rm")(directory, py::arg("recursive") = true);
}
void PythonFilesystem::CreateDirectory(const string &directory, optional_ptr<FileOpener> opener) {
	D_ASSERT(!py::gil_check());
	PythonGILWrapper gil;

	filesystem.attr("mkdir")(py::str(directory));
}
bool PythonFilesystem::ListFiles(const string &directory, const std::function<void(const string &, bool)> &callback,
                                 FileOpener *opener) {
	D_ASSERT(!py::gil_check());
	PythonGILWrapper gil;

	try {
		for (auto item : filesystem.attr("ls")(py::str(directory), py::arg("detail") = true)) {
			bool is_dir = py::cast<std::string>(item["type"]) == "directory";
			string returned_path = py::str(item["name"]);
			callback(RestoreCallerPath(directory, returned_path, returned_path, false), is_dir);
		}
	} catch (const py::error_already_set &error) {
		if (!error.matches(PyExc_NotImplementedError)) {
			throw;
		}
		throw NotImplementedException("Registered Python filesystem '%s' does not implement ls()", GetName());
	}

	// The return value reports whether listing succeeded, not whether the
	// directory contained entries.
	return true;
}
void PythonFilesystem::Truncate(FileHandle &handle, int64_t new_size) {
	D_ASSERT(!py::gil_check());
	PythonGILWrapper gil;

	filesystem.attr("touch")(handle.path, py::arg("truncate") = true);
}
bool PythonFilesystem::IsPipe(const string &filename, optional_ptr<FileOpener> opener) {
	return false;
}
idx_t PythonFilesystem::SeekPosition(FileHandle &handle) {
	D_ASSERT(!py::gil_check());
	PythonGILWrapper gil;

	return py::int_(PythonFileHandle::GetHandle(handle).attr("tell")());
}
} // namespace duckdb

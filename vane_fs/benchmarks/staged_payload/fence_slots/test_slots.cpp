// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>

using namespace vane_fs;
enum class Fault { None, Interrupted, Short, Zero, WriteError, SyncError };
static Fault fault = Fault::None;
static bool armed = false, injected = false;
static int reserve_calls = 0, fence_syncs = 0;
static void Require(bool value, const char *message) {
	if (!value)
		throw std::runtime_error(message);
}
static bool Fence(int fd) {
	char path[4096];
	auto name = "/proc/self/fd/" + std::to_string(fd);
	auto length = readlink(name.c_str(), path, sizeof(path));
	std::string value = length > 0 ? std::string(path, size_t(length)) : "";
	const std::string suffix = ".stage-fence";
	return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}
extern "C" ssize_t __real_pwrite(int, const void *, size_t, off_t);
extern "C" int __real_fdatasync(int);
extern "C" ssize_t __wrap_pwrite(int fd, const void *data, size_t bytes, off_t offset) {
	if (armed && Fence(fd) && offset >= 4096 && offset < 8192) {
		++reserve_calls;
		if (!injected && fault != Fault::None && fault != Fault::SyncError) {
			injected = true;
			if (fault == Fault::Short)
				return __real_pwrite(fd, data, std::min<size_t>(73, bytes), offset);
			if (fault == Fault::Zero)
				return 0;
			errno = fault == Fault::Interrupted ? EINTR : EIO;
			return -1;
		}
	}
	return __real_pwrite(fd, data, bytes, offset);
}
extern "C" int __wrap_fdatasync(int fd) {
	if (armed && Fence(fd)) {
		++fence_syncs;
		if (fault == Fault::SyncError) {
			injected = true;
			errno = EIO;
			return -1;
		}
	}
	return __real_fdatasync(fd);
}
static void Initial(const std::string &path, Fault kind) {
	fault = kind;
	armed = true;
	injected = false;
	reserve_calls = fence_syncs = 0;
	bool failed = false;
	try {
		Workspace workspace(path, 100, Durability::Fsync);
		Require(std::filesystem::file_size(path + ".stage-fence") == 8192, "Initial fence has wrong size");
		std::ifstream input(path + ".stage-fence", std::ios::binary);
		input.seekg(4096);
		std::string second(4096, 'x');
		input.read(&second[0], second.size());
		Require(bool(input) && second == std::string(4096, '\0'), "Unused slot is not invalid zero padding");
		Require(fence_syncs == 1, "Materialization added a fence barrier");
		Require(reserve_calls == ((kind == Fault::Short || kind == Fault::Interrupted) ? 2 : 1),
		        "Materialization did not handle short write/EINTR exactly");
		armed = false;
		auto session = workspace.Checkout();
		session->WriteFile("/file", std::string(1024 * 1024, 'x'));
		workspace.Sync();
		Require(session->Read("/file") == std::string(1024 * 1024, 'x'), "Initial publication content differs");
		workspace.Close();
	} catch (const Error &error) {
		Require(error.code == ErrorCode::Storage, "Initialization failure has wrong error code");
		failed = true;
	}
	armed = false;
	Require(failed == (kind == Fault::Zero || kind == Fault::WriteError || kind == Fault::SyncError),
	        "Initialization fault was lost or a retry failed");
	Require(kind == Fault::None || injected, "Fault did not reach materialization/initial sync");
}
static void Legacy(const std::string &path) {
	{
		Workspace workspace(path);
		workspace.Checkout()->WriteFile("/stable", "stable");
		workspace.Close();
	}
	// The second slot is still invalid zero padding. Removing it constructs
	// the old valid one-slot CLEAN layout without changing its first record.
	auto fd = open((path + ".stage-fence").c_str(), O_RDWR | O_CLOEXEC);
	Require(fd >= 0 && ftruncate(fd, 4096) == 0 && fdatasync(fd) == 0 && close(fd) == 0, "Prepare legacy fence");
	{
		Workspace workspace(path, 100, Durability::Fsync);
		Require(std::filesystem::file_size(path + ".stage-fence") == 4096, "Existing one-slot fence was rewritten");
		auto session = workspace.Checkout();
		Require(session->Read("/stable") == "stable", "Legacy inline content differs");
		session->WriteFile("/new", std::string(1024 * 1024, 'n'));
		workspace.Sync();
		workspace.Close();
	}
	Workspace reopened(path);
	auto read = reopened.Checkout();
	Require(read->Read("/stable") == "stable" && read->Read("/new") == std::string(1024 * 1024, 'n'),
	        "Legacy/new publication differs after close");
	reopened.Close();
}
int main(int argc, char **argv) {
	try {
		Require(argc == 2, "Expected owned directory");
		for (auto kind :
		     {Fault::None, Fault::Interrupted, Fault::Short, Fault::Zero, Fault::WriteError, Fault::SyncError})
			Initial(std::string(argv[1]) + "/initial-" + std::to_string(int(kind)) + ".sqlite", kind);
		Legacy(std::string(argv[1]) + "/legacy.sqlite");
		std::cout
		    << "PASS invalid padding, one initial barrier, EINTR/short retry, zero/EIO/sync failure and legacy fence\n";
		return 0;
	} catch (const std::exception &error) {
		armed = false;
		std::cerr << error.what() << '\n';
		return 1;
	}
}

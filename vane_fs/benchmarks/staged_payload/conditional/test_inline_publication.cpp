// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "staging.hpp"
#include "vane_fs/workspace.hpp"
#include <filesystem>
#include <iostream>
#include <thread>

using namespace vane_fs;
static void Require(bool condition, const char *message) {
	if (!condition)
		throw std::runtime_error(message);
}
template <class F>
static void Busy(F operation) {
	try {
		operation();
	} catch (const Error &error) {
		Require(error.code == ErrorCode::Busy, error.what());
		return;
	}
	throw std::runtime_error("Expected publication contention before mutation");
}
static void Gate(const std::string &path) {
	Workspace workspace(path, 50, Durability::Fsync);
	auto session = workspace.Checkout();
	session->WriteFile("/file", "");
	workspace.AcquireMount("main");
	auto file = session->OpenFile("/file");
	workspace.Sync();
	Staging observer(path, 50);
	observer.Initialize(workspace.Id(), false);
	observer.Ready();
	{
		Staging::Guard held(observer);
		// CLEAN inline operations must not need the publication gate.
		session->WriteFile("/inline", std::string(4096, 'i'));
		session->MakeDirectory("/dir");
		session->Rename("/dir", "/moved");
		session->RemoveDirectory("/moved");
		Busy([&] { workspace.Sync(); });
		Busy([&] { session->WriteInode(file.inode, "s", 0, false, true); });
		Busy([&] { session->WriteFile("/large", std::string(262144, 'l')); });
		// This unaligned range can produce 64 padded blocks despite having
		// fewer than 256 KiB of application bytes. It must acquire the gate.
		Busy([&] { session->Write("/file", std::string(253954, 'b'), 4095); });
		Busy([&] { session->WriteInode(file.inode, std::string(253954, 'b'), 4095); });
		Require(session->Read("/file").empty(), "Rejected external write mutated the file");
	}
	session->Write("/file", std::string(1024 * 1024, 'p'));
	{
		Staging::Guard held(observer);
		Require(observer.Pending(), "Fixture did not open an external batch");
		// Inline overwrites and namespace changes also proceed during OPEN;
		// their WAL records are covered by the same subsequent FULL barrier.
		session->WriteInode(file.inode, std::string(4096, 'q'), 0);
		session->Rename("/inline", "/renamed");
		session->MakeDirectory("/pending-dir");
	}
	workspace.Sync();
	{
		Staging::Guard held(observer);
		Require(!observer.Pending(), "FULL barrier did not publish the batch");
	}
	Require(session->Read("/file") == std::string(4096, 'q') + std::string(1024 * 1024 - 4096, 'p'),
	        "Inline overwrite of pending external data differs");
	session->CloseFile(file.inode);
	workspace.ReleaseMount("main");
	workspace.Close();
	Workspace reopened(path);
	Require(reopened.Checkout()->Read("/renamed") == std::string(4096, 'i'), "Barrier lost namespace changes");
}
static void Concurrent(const std::string &path) {
	Workspace external(path, 5000, Durability::Fsync), inline_writer(path, 5000, Durability::Fsync),
	    barrier(path, 5000, Durability::Fsync);
	auto a = external.Checkout(), b = inline_writer.Checkout();
	a->WriteFile("/large", "");
	b->WriteFile("/inline", std::string(4096, 'i'));
	barrier.Sync();
	std::exception_ptr errors[3];
	std::thread writers[3] = {std::thread([&] {
		                          try {
			                          for (int i = 0; i < 96; ++i)
				                          a->Write("/large", std::string(262144, char(i % 26 + 'a')),
				                                   int64_t(i) * 262144);
		                          } catch (...) {
			                          errors[0] = std::current_exception();
		                          }
	                          }),
	                          std::thread([&] {
		                          try {
			                          for (int i = 0; i < 256; ++i) {
				                          b->Write("/inline", std::string(4096, char(i % 26 + 'A')));
				                          b->MakeDirectory("/dir");
				                          b->Rename("/dir", "/moved");
				                          b->RemoveDirectory("/moved");
			                          }
		                          } catch (...) {
			                          errors[1] = std::current_exception();
		                          }
	                          }),
	                          std::thread([&] {
		                          try {
			                          for (int i = 0; i < 64; ++i)
				                          barrier.Sync();
		                          } catch (...) {
			                          errors[2] = std::current_exception();
		                          }
	                          })};
	for (auto &writer : writers)
		writer.join();
	for (const auto &error : errors)
		if (error)
			std::rethrow_exception(error);
	barrier.Sync();
	for (int i = 0; i < 96; ++i)
		Require(b->Read("/large", int64_t(i) * 262144, 262144) == std::string(262144, char(i % 26 + 'a')),
		        "Concurrent external extent differs");
	Require(a->Read("/inline") == std::string(4096, char(255 % 26 + 'A')), "Concurrent inline write differs");
	external.Close();
	inline_writer.Close();
	barrier.Close();
	Workspace reopened(path);
	Require(reopened.Checkout()->Read("/inline") == std::string(4096, char(255 % 26 + 'A')), "Reopen lost inline data");
}
int main(int argc, char **argv) {
	if (argc != 2)
		return 2;
	try {
		auto root = std::filesystem::path(argv[1]);
		Require(std::filesystem::create_directory(root), "Test directory must not exist");
		struct Cleanup {
			std::filesystem::path path;
			~Cleanup() {
				std::filesystem::remove_all(path);
			}
		} cleanup {root};
		Gate((root / "gate.sqlite").string());
		Concurrent((root / "concurrent.sqlite").string());
		std::cout << "Inline publication, unaligned extents, FULL gates and concurrent barriers passed\n";
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}

// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

using namespace vane_fs;
static void Require(bool value, const char *message) {
	if (!value)
		throw std::runtime_error(message);
}
static std::string Pattern(size_t size) {
	std::string data(size, '\0');
	for (size_t i = 0; i < size; ++i)
		data[i] = char(1 + (i * 17 + i / 4096 * 13) % 251);
	return data;
}
static void Overlay(std::string &expected, const std::string &data, size_t offset) {
	if (expected.size() < offset + data.size())
		expected.resize(offset + data.size(), '\0');
	expected.replace(offset, data.size(), data);
}
int main(int argc, char **argv) {
	try {
		Require(argc == 2, "Expected owned database prefix");
		constexpr size_t MIB = 1048576;
		for (auto durability : {Durability::Strict, Durability::Fsync}) {
			auto label = durability == Durability::Strict ? "strict" : "fsync";
			Workspace workspace(std::string(argv[1]) + "-" + label, 5000, durability);
			auto session = workspace.Checkout();
			for (int kind = 0; kind < 6; ++kind) {
				auto path = "/case-" + std::to_string(kind);
				std::string expected;
				session->WriteFile(path, "");
				if (kind == 2 || kind == 4) {
					auto prefix = std::string(4096, 'p');
					session->Write(path, prefix, 0);
					Overlay(expected, prefix, 0);
				}
				if (kind == 3 || kind == 4) {
					auto tail = std::string(4096, 't');
					session->Write(path, tail, MIB);
					Overlay(expected, tail, MIB);
				}
				size_t offset = kind == 1 || kind == 2 || kind == 4 ? 48 : 0;
				auto input = Pattern(kind == 5 ? 2 * MIB : MIB + (kind == 0 ? 0 : 73));
				if (kind == 5)
					for (size_t block = 3; block < input.size() / 4096; block += 7)
						input.replace(block * 4096, 4096, 4096, '\0');
				session->Write(path, input, int64_t(offset));
				Overlay(expected, input, offset);
				input.assign(input.size(), 'x');
				workspace.Sync();
				Require(session->Read(path) == expected, "Input storage escaped the synchronous write");
				auto snapshot = workspace.Snapshot();
				session->Write(path, "later", 4001);
				workspace.Sync();
				workspace.CollectGarbage();
				auto frozen = workspace.OpenSnapshot(snapshot);
				Require(frozen->Read(path) == expected, "Borrowed boundaries or GC changed the snapshot");
				frozen->Close();
				workspace.DropSnapshot(snapshot);
				workspace.CollectGarbage();
				Overlay(expected, "later", 4001);
				Require(session->Read(path) == expected, "Boundary update or gather fallback differs");
				std::cout << "PASS " << label << " case " << kind << '\n';
			}
			workspace.Close();
		}
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}

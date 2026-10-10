// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include <iostream>

int main(int argc, char **argv) {
	if (argc != 3 || (std::string(argv[2]) != "strict" && std::string(argv[2]) != "fsync"))
		return 2;
	try {
		using namespace vane_fs;
		const auto mode = std::string(argv[2]) == "fsync" ? Durability::Fsync : Durability::Strict;
		Workspace workspace(argv[1], 5000, mode);
		auto live = workspace.Checkout();
		std::string original;
		for (int repeat = 0; repeat < 1100; ++repeat)
			for (int value = 0; value < 251; ++value)
				original.push_back(static_cast<char>(value));
		live->MakeDirectory("/dir");
		live->WriteFile("/dir/file", original);
		live->WriteFile("/sparse", "");
		live->Write("/sparse", "end", 1024 * 1024 - 3);
		auto frozen = workspace.Snapshot();
		auto child = workspace.Fork("main", "child");
		workspace.Checkout(child.id)->WriteFile("/child-only", "child data");
		workspace.Merge(workspace.PreviewMerge(child.id, "main"));
		live->Write("/dir/file", "changed", 4095);
		workspace.Sync();
		workspace.Close();
		std::cout << "{\"state\":{\"snapshot\":\"" << frozen << "\"},\"durability\":\"" << argv[2] << "\"}\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}

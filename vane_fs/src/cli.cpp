// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include <iostream>

int main(int argc, char **argv) {
	if (argc < 3) {
		std::cerr << "Usage: vane-fs DATABASE init|branches|snapshot [BRANCH]|fork SOURCE NAME|recover|gc\n";
		return argc == 2 && std::string(argv[1]) == "--help" ? 0 : 2;
	}
	try {
		std::string command = argv[2];
		if (!((argc == 3 && (command == "init" || command == "branches" || command == "recover" || command == "gc")) ||
		      ((argc == 3 || argc == 4) && command == "snapshot") || (argc == 5 && command == "fork"))) {
			throw std::runtime_error("Invalid command or arguments; use --help");
		}
		vane_fs::Workspace workspace(argv[1]);
		if (command == "init")
			std::cout << workspace.Id() << '\n';
		else if (command == "branches") {
			for (const auto &branch : workspace.ListBranches())
				std::cout << branch.id << '\t' << branch.name << '\t' << branch.state << '\n';
		} else if (command == "snapshot")
			std::cout << workspace.Snapshot(argc == 4 ? argv[3] : "main") << '\n';
		else if (command == "fork")
			std::cout << workspace.Fork(argv[3], argv[4]).id << '\n';
		else if (command == "recover") {
			auto result = workspace.RecoverOwners();
			std::cout << "owners=" << result.owners << " pins=" << result.pins << " mounts=" << result.mounts << '\n';
		} else {
			auto result = workspace.CollectGarbage();
			std::cout << "versions=" << result.versions << " payloads=" << result.payloads
			          << " snapshots=" << result.snapshots << '\n';
		}
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "VaneFS: " << error.what() << '\n';
		return 1;
	}
}

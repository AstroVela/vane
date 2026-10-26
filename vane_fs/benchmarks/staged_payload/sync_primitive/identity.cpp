// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <sqlite3.h>

int main() {
	std::cout << sqlite3_libversion() << '\n' << sqlite3_sourceid() << '\n';
	for (int i = 0; sqlite3_compileoption_get(i); ++i) {
		std::cout << sqlite3_compileoption_get(i) << '\n';
	}
}

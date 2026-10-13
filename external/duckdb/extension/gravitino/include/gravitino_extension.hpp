// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once
#include "duckdb/main/extension.hpp"

namespace duckdb {
class GravitinoExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override;
	std::string Name() override;
	std::string Version() const override;
};
void RegisterGravitinoCatalog(ExtensionLoader &loader);
void RegisterGravitinoFunctions(ExtensionLoader &loader);
void RegisterGravitinoFileSystem(ExtensionLoader &loader);
} // namespace duckdb

// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
#pragma once
#include "duckdb.hpp"
namespace duckdb {
class AudioExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override;
	std::string Name() override;
	std::string Version() const override;
};
} // namespace duckdb

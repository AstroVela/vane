// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "file_extension.hpp"

#include "file_functions.hpp"

#include "duckdb/function/scalar/nested_functions.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

static void LoadInternal(ExtensionLoader &loader) {
	loader.RegisterType(FileLogicalType::TYPE_NAME, FileLogicalType::Create());

	for (auto &function : FileFunctions::GetFunctions()) {
		loader.RegisterFunction(std::move(function));
	}

	auto file_key_extract = GetKeyExtractFunction();
	file_key_extract.arguments[0] = FileLogicalType::Create();
	loader.RegisterFunction(std::move(file_key_extract));
}

void FileExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string FileExtension::Name() {
	return "file";
}

std::string FileExtension::Version() const {
#ifdef EXT_VERSION_FILE
	return EXT_VERSION_FILE;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(file, loader) {
	duckdb::LoadInternal(loader);
}
}

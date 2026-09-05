// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "file_extension.hpp"

#include "file_functions.hpp"
#include "file_list_function.hpp"
#include "file_metadata_functions.hpp"
#include "media_backend.hpp"

#include "duckdb/function/scalar/nested_functions.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/config.hpp"

namespace duckdb {

static void LoadInternal(ExtensionLoader &loader) {
	MediaBackend::RegisterOption(DBConfig::GetConfig(loader.GetDatabaseInstance()));
	for (auto media_type : FileLogicalType::MEDIA_TYPES) {
		loader.RegisterType(FileLogicalType::GetTypeName(media_type), FileLogicalType::Create(media_type));
	}
	loader.RegisterType(ImageLogicalType::TYPE_NAME, ImageLogicalType::Create());

	for (auto &function : FileFunctions::GetFunctions()) {
		loader.RegisterFunction(std::move(function));
	}
	for (auto &function : FileMetadataFunctions::GetFunctions()) {
		loader.RegisterFunction(std::move(function));
	}
	for (auto &function : FileListFunction::GetFunctions()) {
		loader.RegisterFunction(std::move(function));
	}

	for (auto media_type : FileLogicalType::MEDIA_TYPES) {
		auto file_key_extract = GetKeyExtractFunction();
		file_key_extract.arguments[0] = FileLogicalType::Create(media_type);
		loader.RegisterFunction(std::move(file_key_extract));
	}
	auto image_key_extract = GetKeyExtractFunction();
	image_key_extract.arguments[0] = ImageLogicalType::Create();
	loader.RegisterFunction(std::move(image_key_extract));
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

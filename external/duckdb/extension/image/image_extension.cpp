// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
#include "image_extension.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "media_reader.hpp"
namespace duckdb {
static void LoadInternal(ExtensionLoader &loader) {
	RegisterMediaImages(loader);
}
void ImageExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string ImageExtension::Name() {
	return "image";
}
std::string ImageExtension::Version() const {
	return "1";
}
} // namespace duckdb
extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(image, loader) {
	duckdb::LoadInternal(loader);
}
}

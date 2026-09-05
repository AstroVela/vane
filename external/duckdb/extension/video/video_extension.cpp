// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
#include "video_extension.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "media_reader.hpp"
namespace duckdb {
static void LoadInternal(ExtensionLoader &loader) {
	RegisterMediaVideo(loader);
}
void VideoExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string VideoExtension::Name() {
	return "video";
}
std::string VideoExtension::Version() const {
	return "1";
}
} // namespace duckdb
extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(video, loader) {
	duckdb::LoadInternal(loader);
}
}

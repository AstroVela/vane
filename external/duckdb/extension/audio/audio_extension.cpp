// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
#include "audio_extension.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "media_reader.hpp"
namespace duckdb {
static void LoadInternal(ExtensionLoader &loader) {
	RegisterMediaAudio(loader);
}
void AudioExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string AudioExtension::Name() {
	return "audio";
}
std::string AudioExtension::Version() const {
	return "1";
}
} // namespace duckdb
extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(audio, loader) {
	duckdb::LoadInternal(loader);
}
}

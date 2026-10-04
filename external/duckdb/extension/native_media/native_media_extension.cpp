// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
#include "native_media_extension.hpp"
#include "image_codec.hpp"
#include "media_reader.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
namespace duckdb {
static void LoadInternal(ExtensionLoader &loader) {
	RegisterMediaAudio(loader);
	RegisterMediaImages(loader);
	RegisterImagePixelFunctions(loader);
	RegisterImageComputeFunctions(loader);
	RegisterMediaVideo(loader);
}
void NativeMediaExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string NativeMediaExtension::Name() {
	return "native_media";
}
std::string NativeMediaExtension::Version() const {
#ifdef EXT_VERSION_NATIVE_MEDIA
	return EXT_VERSION_NATIVE_MEDIA;
#else
	return "";
#endif
}
} // namespace duckdb
extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(native_media, loader) {
	duckdb::LoadInternal(loader);
}
}

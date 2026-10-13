// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "gravitino_extension.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {
void GravitinoExtension::Load(ExtensionLoader &loader) {
	RegisterGravitinoCatalog(loader);
	RegisterGravitinoFunctions(loader);
	RegisterGravitinoFileSystem(loader);
}
std::string GravitinoExtension::Name() {
	return "gravitino";
}
std::string GravitinoExtension::Version() const {
#ifdef EXT_VERSION_GRAVITINO
	return EXT_VERSION_GRAVITINO;
#else
	return "";
#endif
}
} // namespace duckdb
extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(gravitino, loader) {
	duckdb::GravitinoExtension extension;
	extension.Load(loader);
}
}

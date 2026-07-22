// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/import_cache/modules/vane_module.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb_python/import_cache/python_import_cache_item.hpp"

//! Note: This class is generated using scripts.
//! If you need to add a new object to the cache you must:
//! 1. adjust scripts/imports.py
//! 2. run python scripts/generate_import_cache_json.py
//! 3. run python scripts/generate_import_cache_cpp.py
//! 4. run pre-commit to fix formatting errors

namespace duckdb {

struct VanePolarsioCacheItem : public PythonImportCacheItem {

public:
	static constexpr const char *Name = "vane.polars_io";

public:
	VanePolarsioCacheItem() : PythonImportCacheItem("vane.polars_io"), duckdb_source("duckdb_source", this) {
	}
	~VanePolarsioCacheItem() override {
	}

	PythonImportCacheItem duckdb_source;

protected:
	bool IsRequired() const override final {
		return false;
	}
};

struct VaneFilesystemCacheItem : public PythonImportCacheItem {

public:
	static constexpr const char *Name = "vane.filesystem";

public:
	VaneFilesystemCacheItem()
	    : PythonImportCacheItem("vane.filesystem"), ModifiedMemoryFileSystem("ModifiedMemoryFileSystem", this) {
	}
	~VaneFilesystemCacheItem() override {
	}

	PythonImportCacheItem ModifiedMemoryFileSystem;

protected:
	bool IsRequired() const override final {
		return false;
	}
};

struct VaneCacheItem : public PythonImportCacheItem {

public:
	static constexpr const char *Name = "vane";

public:
	VaneCacheItem() : PythonImportCacheItem("vane"), filesystem(), Value("Value", this), polars_io() {
	}
	~VaneCacheItem() override {
	}

	VaneFilesystemCacheItem filesystem;
	PythonImportCacheItem Value;
	VanePolarsioCacheItem polars_io;
};

} // namespace duckdb

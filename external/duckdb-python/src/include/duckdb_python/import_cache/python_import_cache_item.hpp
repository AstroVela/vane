//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/import_cache/python_import_cache_item.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb_python/nb/casters.hpp"
#include "duckdb.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {

struct PythonImportCache;

struct PythonImportCacheItem {
public:
	PythonImportCacheItem(const string &name, optional_ptr<PythonImportCacheItem> parent)
	    : name(name), is_module(false), load_succeeded(false), parent(parent), object(nullptr) {
	}
	PythonImportCacheItem(const string &name)
	    : name(name), is_module(true), load_succeeded(false), parent(nullptr), object(nullptr) {
	}

	virtual ~PythonImportCacheItem() {
	}

public:
	bool LoadSucceeded() const;
	bool IsLoaded() const;
	nb::handle operator()(bool load = true);
	nb::handle Load(PythonImportCache &cache, nb::handle source, bool load);

protected:
	virtual bool IsRequired() const {
		return true;
	}

private:
	nb::handle AddCache(PythonImportCache &cache, nb::object object);
	void LoadAttribute(PythonImportCache &cache, nb::handle source);
	void LoadModule(PythonImportCache &cache);

private:
	//! The name of the item
	string name;
	//! Whether the item is a module
	bool is_module;
	//! Whether or not we attempted to load the item
	bool load_succeeded;
	//! The parent of this item (either a module or an attribute)
	optional_ptr<PythonImportCacheItem> parent;
	//! The stored item
	nb::handle object;
};

} // namespace duckdb

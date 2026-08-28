// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/pair.hpp"
#include "parquet_types.h"

namespace duckdb {

class FileSystem;
struct CopyFunctionFileStatistics;

struct ArrowParquetWriterOptions {
	duckdb_parquet::CompressionCodec::type codec;
	idx_t row_group_size;
	idx_t dictionary_page_size_limit;
	bool disable_dictionary;
	bool enable_bloom_filters;
	double bloom_filter_false_positive_ratio;
	int64_t compression_level;
	vector<pair<string, string>> key_value_metadata;
};

//! Owns the Arrow C Data objects imported for one local COPY stream.
class ArrowParquetLocalState {
public:
	ArrowParquetLocalState();
	~ArrowParquetLocalState();

	void ImportSchema(ArrowSchema &schema, const vector<string> &names);
	void ImportRecordBatch(ArrowArray &array);

private:
	struct Impl;
	unique_ptr<Impl> impl;

	friend class ArrowParquetWriter;
};

//! Writes imported Arrow RecordBatches directly with parquet::arrow::FileWriter.
class ArrowParquetWriter {
public:
	ArrowParquetWriter(FileSystem &fs, const string &file_path, ArrowParquetLocalState &local_state,
	                   const ArrowParquetWriterOptions &options);
	~ArrowParquetWriter();

	void Write(ArrowParquetLocalState &local_state, idx_t offset, idx_t cardinality);
	void SetWrittenStatistics(CopyFunctionFileStatistics &statistics);
	void Finalize();
	idx_t FileSize() const;
	idx_t NumberOfRowGroups() const;

private:
	struct Impl;
	unique_ptr<Impl> impl;
};

} // namespace duckdb

#include "parquet_column_schema.hpp"
#include "parquet_reader.hpp"
#include "duckdb/common/operator/cast_operators.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/type_visitor.hpp"
#include "duckdb/common/unordered_set.hpp"

namespace duckdb {

constexpr const char *ParquetFileTypeMetadata::KEY;

static char ParquetFileTypeMetadataKindToChar(ParquetFileTypeMetadataKind kind) {
	switch (kind) {
	case ParquetFileTypeMetadataKind::FILE:
		return 'F';
	case ParquetFileTypeMetadataKind::UNION:
		return 'U';
	default:
		throw InternalException("Unknown Parquet FILE metadata kind");
	}
}

string ParquetFileTypeMetadata::Serialize(const vector<ParquetFileTypeMetadataEntry> &entries) {
	string result = "1";
	for (auto &entry : entries) {
		result += ';';
		result += ParquetFileTypeMetadataKindToChar(entry.kind);
		result += ':';
		for (idx_t path_index = 0; path_index < entry.path.size(); path_index++) {
			if (path_index > 0) {
				result += '.';
			}
			result += to_string(entry.path[path_index]);
		}
	}
	return result;
}

[[noreturn]] static void ThrowInvalidParquetFileTypeMetadata(const string &file_path, const string &message) {
	throw InvalidInputException("Failed to read Parquet file \"%s\": invalid %s metadata: %s", file_path,
	                            ParquetFileTypeMetadata::KEY, message);
}

vector<ParquetFileTypeMetadataEntry> ParquetFileTypeMetadata::Deserialize(const string &metadata,
                                                                          const string &file_path) {
	auto parts = StringUtil::Split(metadata, ';');
	if (parts.empty() || parts[0] != "1") {
		ThrowInvalidParquetFileTypeMetadata(file_path, "unsupported format version");
	}

	vector<ParquetFileTypeMetadataEntry> result;
	unordered_set<string> seen_entries;
	for (idx_t part_index = 1; part_index < parts.size(); part_index++) {
		auto &part = parts[part_index];
		if (part.size() < 3 || part[1] != ':') {
			ThrowInvalidParquetFileTypeMetadata(file_path, "malformed path entry");
		}

		ParquetFileTypeMetadataEntry entry;
		switch (part[0]) {
		case 'F':
			entry.kind = ParquetFileTypeMetadataKind::FILE;
			break;
		case 'U':
			entry.kind = ParquetFileTypeMetadataKind::UNION;
			break;
		default:
			ThrowInvalidParquetFileTypeMetadata(file_path, "unknown path kind");
		}

		for (auto &path_part : StringUtil::Split(part.substr(2), '.')) {
			idx_t child_index;
			if (!TryCast::Operation<string_t, idx_t>(string_t(path_part), child_index, true)) {
				ThrowInvalidParquetFileTypeMetadata(file_path, "invalid child index");
			}
			entry.path.push_back(child_index);
		}
		if (entry.path.empty()) {
			ThrowInvalidParquetFileTypeMetadata(file_path, "empty path");
		}
		if (!seen_entries.insert(part).second) {
			ThrowInvalidParquetFileTypeMetadata(file_path, "duplicate path");
		}
		result.push_back(std::move(entry));
	}
	if (result.empty() || Serialize(result) != metadata) {
		ThrowInvalidParquetFileTypeMetadata(file_path, "non-canonical encoding");
	}
	return result;
}

static ParquetColumnSchema &ResolveParquetFileTypePath(ParquetColumnSchema &root, const vector<idx_t> &path,
                                                       const string &file_path) {
	auto node = &root;
	for (auto child_index : path) {
		if (child_index >= node->children.size()) {
			ThrowInvalidParquetFileTypeMetadata(file_path, "path is outside the Parquet schema");
		}
		node = &node->children[child_index];
	}
	return *node;
}

static LogicalType ParquetFileStorageType() {
	auto result = FileLogicalType::Create();
	result.SetAlias(string());
	return result;
}

static bool RefreshParquetFileTypes(ParquetColumnSchema &schema, const string &file_path) {
	if (FileLogicalType::IsFile(schema.type)) {
		return true;
	}

	bool contains_file = false;
	for (auto &child : schema.children) {
		contains_file = RefreshParquetFileTypes(child, file_path) || contains_file;
	}
	if (!contains_file) {
		return false;
	}

	switch (schema.type.id()) {
	case LogicalTypeId::STRUCT: {
		child_list_t<LogicalType> children;
		for (auto &child : schema.children) {
			children.emplace_back(child.name, child.type);
		}
		schema.type = LogicalType::STRUCT(std::move(children));
		break;
	}
	case LogicalTypeId::UNION: {
		if (schema.children.size() < 2 || !schema.children[0].name.empty() ||
		    schema.children[0].type != LogicalType::UTINYINT) {
			ThrowInvalidParquetFileTypeMetadata(file_path, "UNION path has incompatible storage");
		}
		child_list_t<LogicalType> members;
		for (idx_t child_index = 1; child_index < schema.children.size(); child_index++) {
			auto &child = schema.children[child_index];
			members.emplace_back(child.name, child.type);
		}
		schema.type = LogicalType::UNION(std::move(members));
		break;
	}
	case LogicalTypeId::LIST:
		if (schema.children.size() != 1) {
			ThrowInvalidParquetFileTypeMetadata(file_path, "LIST path has incompatible storage");
		}
		schema.type = LogicalType::LIST(schema.children[0].type);
		break;
	case LogicalTypeId::MAP: {
		if (schema.children.size() != 1 || schema.children[0].type.id() != LogicalTypeId::STRUCT ||
		    StructType::GetChildCount(schema.children[0].type) != 2) {
			ThrowInvalidParquetFileTypeMetadata(file_path, "MAP path has incompatible storage");
		}
		auto &entry_type = schema.children[0].type;
		schema.type =
		    LogicalType::MAP(StructType::GetChildType(entry_type, 0), StructType::GetChildType(entry_type, 1));
		break;
	}
	default:
		ThrowInvalidParquetFileTypeMetadata(file_path, "FILE path has an unsupported parent type");
	}
	return true;
}

void ParquetFileTypeMetadata::Apply(ParquetColumnSchema &root, const vector<ParquetFileTypeMetadataEntry> &entries,
                                    const string &file_path) {
	for (auto &entry : entries) {
		if (entry.kind != ParquetFileTypeMetadataKind::FILE) {
			continue;
		}
		auto &node = ResolveParquetFileTypePath(root, entry.path, file_path);
		if (node.type != ParquetFileStorageType() || node.children.size() != FileLogicalType::FIELD_COUNT) {
			ThrowInvalidParquetFileTypeMetadata(file_path, "FILE path has incompatible storage");
		}
		node.type = FileLogicalType::Create();
	}
	RefreshParquetFileTypes(root, file_path);

	for (idx_t entry_index = entries.size(); entry_index > 0; entry_index--) {
		auto &entry = entries[entry_index - 1];
		if (entry.kind != ParquetFileTypeMetadataKind::UNION) {
			continue;
		}
		auto &node = ResolveParquetFileTypePath(root, entry.path, file_path);
		if (node.type.id() != LogicalTypeId::STRUCT || node.children.size() < 2 || !node.children[0].name.empty() ||
		    node.children[0].type != LogicalType::UTINYINT) {
			ThrowInvalidParquetFileTypeMetadata(file_path, "UNION path has incompatible storage");
		}
		child_list_t<LogicalType> members;
		for (idx_t child_index = 1; child_index < node.children.size(); child_index++) {
			auto &child = node.children[child_index];
			members.emplace_back(child.name, child.type);
		}
		node.type = LogicalType::UNION(std::move(members));
		if (!TypeVisitor::Contains(node.type, FileLogicalType::IsFile)) {
			ThrowInvalidParquetFileTypeMetadata(file_path, "UNION path does not contain FILE");
		}
	}
	RefreshParquetFileTypes(root, file_path);
}

void ParquetColumnSchema::SetSchemaIndex(idx_t schema_idx) {
	D_ASSERT(!schema_index.IsValid());
	schema_index = schema_idx;
}

//! Writer constructors

ParquetColumnSchema ParquetColumnSchema::FromLogicalType(const string &name, const LogicalType &type, idx_t max_define,
                                                         idx_t max_repeat, idx_t column_index,
                                                         duckdb_parquet::FieldRepetitionType::type repetition_type,
                                                         bool allow_geometry, ParquetColumnSchemaType schema_type) {
	ParquetColumnSchema res;
	res.name = name;
	res.max_define = max_define;
	res.max_repeat = max_repeat;
	res.column_index = column_index;
	res.repetition_type = repetition_type;
	res.schema_type = schema_type;
	res.type = type;
	res.allow_geometry = allow_geometry;
	return res;
}

//! Reader constructors

ParquetColumnSchema ParquetColumnSchema::FromSchemaElement(const duckdb_parquet::SchemaElement &element,
                                                           idx_t max_define, idx_t max_repeat, idx_t schema_index,
                                                           idx_t column_index, ParquetColumnSchemaType schema_type,
                                                           const ParquetOptions &options) {
	ParquetColumnSchema res;
	res.name = element.name;
	res.max_define = max_define;
	res.max_repeat = max_repeat;
	res.schema_index = schema_index;
	res.column_index = column_index;
	res.schema_type = schema_type;
	res.type = ParquetReader::DeriveLogicalType(element, options, res);
	return res;
}

ParquetColumnSchema ParquetColumnSchema::FromParentSchema(ParquetColumnSchema parent, LogicalType result_type,
                                                          ParquetColumnSchemaType schema_type) {
	ParquetColumnSchema res;
	res.name = parent.name;
	res.max_define = parent.max_define;
	res.max_repeat = parent.max_repeat;
	D_ASSERT(parent.schema_index.IsValid());
	res.schema_index = parent.schema_index;
	res.column_index = parent.column_index;
	res.schema_type = schema_type;
	res.type = result_type;
	res.children.push_back(std::move(parent));
	return res;
}

ParquetColumnSchema ParquetColumnSchema::FromChildSchemas(const string &name, const LogicalType &type, idx_t max_define,
                                                          idx_t max_repeat, idx_t schema_index, idx_t column_index,
                                                          vector<ParquetColumnSchema> &&children,
                                                          ParquetColumnSchemaType schema_type) {
	ParquetColumnSchema res;
	res.name = name;
	res.max_define = max_define;
	res.max_repeat = max_repeat;
	res.schema_index = schema_index;
	res.column_index = column_index;
	res.schema_type = schema_type;
	res.type = type;
	res.children = std::move(children);
	return res;
}

ParquetColumnSchema ParquetColumnSchema::FileRowNumber() {
	ParquetColumnSchema res;
	res.name = "file_row_number";
	res.max_define = 0;
	res.max_repeat = 0;
	res.schema_index = 0;
	res.column_index = 0;
	res.schema_type = ParquetColumnSchemaType::FILE_ROW_NUMBER;
	res.type = LogicalType::BIGINT, res.repetition_type = duckdb_parquet::FieldRepetitionType::type::OPTIONAL;
	return res;
}

unique_ptr<BaseStatistics> ParquetColumnSchema::Stats(const FileMetaData &file_meta_data,
                                                      const ParquetOptions &parquet_options, idx_t row_group_idx_p,
                                                      const vector<ColumnChunk> &columns) const {
	if (schema_type == ParquetColumnSchemaType::EXPRESSION) {
		return nullptr;
	}
	if (schema_type == ParquetColumnSchemaType::FILE_ROW_NUMBER) {
		auto &row_groups = file_meta_data.row_groups;
		D_ASSERT(row_group_idx_p < row_groups.size());
		if (row_groups[row_group_idx_p].num_rows == 0) {
			return NumericStats::CreateEmpty(type).ToUnique();
		}

		idx_t row_group_offset_min = 0;
		for (idx_t i = 0; i < row_group_idx_p; i++) {
			row_group_offset_min += row_groups[i].num_rows;
		}

		auto stats = NumericStats::CreateUnknown(type);
		NumericStats::SetMin(stats, Value::BIGINT(UnsafeNumericCast<int64_t>(row_group_offset_min)));
		NumericStats::SetMax(stats, Value::BIGINT(UnsafeNumericCast<int64_t>(
		                                row_group_offset_min + row_groups[row_group_idx_p].num_rows - 1)));
		stats.Set(StatsInfo::CANNOT_HAVE_NULL_VALUES);
		return stats.ToUnique();
	}
	return ParquetStatisticsUtils::TransformColumnStatistics(*this, columns, parquet_options.can_have_nan);
}

} // namespace duckdb

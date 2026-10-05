// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "fragment_plan.hpp"
#include "file_snapshot.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/multi_file/multi_file_reader.hpp"
#include "duckdb/common/multi_file/multi_file_states.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/operator/helper/physical_set.hpp"
#include "duckdb/execution/operator/scan/physical_empty_result.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/client_config.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/parsed_expression_iterator.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/planner.hpp"

#include <set>

namespace duckdb {
namespace vane_execution {
namespace {

constexpr idx_t FORMAT_VERSION = 1;
constexpr idx_t MAX_PARTITIONS = 2147483647;
constexpr const char *PARQUET_CAPABILITY = "vane.parquet-scan:1";
constexpr const char *PARQUET_CODEC = "vane.parquet-file:1";
constexpr const char *FROZEN_PARQUET_CODEC = "vane.parquet-snapshot:1";

void CheckPartitions(idx_t partitions) {
	if (partitions == 0 || partitions > MAX_PARTITIONS) {
		throw InvalidInputException("fragment partition count must be between 1 and 2147483647");
	}
}

void CheckTypes(const vector<LogicalType> &types) {
	for (auto &type : types) {
		switch (type.id()) {
		case LogicalTypeId::BOOLEAN:
		case LogicalTypeId::TINYINT:
		case LogicalTypeId::SMALLINT:
		case LogicalTypeId::INTEGER:
		case LogicalTypeId::BIGINT:
		case LogicalTypeId::UTINYINT:
		case LogicalTypeId::USMALLINT:
		case LogicalTypeId::UINTEGER:
		case LogicalTypeId::UBIGINT:
		case LogicalTypeId::FLOAT:
		case LogicalTypeId::DOUBLE:
		case LogicalTypeId::VARCHAR:
		case LogicalTypeId::SQLNULL:
			break;
		default:
			throw NotImplementedException("fragment compiler does not support type %s", type.ToString());
		}
	}
}

string Encode(const std::function<void(Serializer &)> &write) {
	MemoryStream stream;
	SerializationOptions options;
	options.serialization_compatibility = SerializationCompatibility::Latest();
	options.serialize_default_values = true;
	BinarySerializer serializer(stream, options);
	serializer.Begin();
	write(serializer);
	serializer.End();
	return string(reinterpret_cast<const char *>(stream.GetData()), stream.GetPosition());
}

void Decode(const string &payload, const std::function<void(BinaryDeserializer &)> &read) {
	if (payload.empty()) {
		throw SerializationException("empty native fragment payload");
	}
	MemoryStream stream(reinterpret_cast<data_ptr_t>(const_cast<char *>(payload.data())), payload.size());
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	read(deserializer);
	deserializer.End();
	if (stream.GetPosition() != payload.size()) {
		throw SerializationException("native fragment payload has trailing bytes");
	}
}

void WriteIdentity(Serializer &serializer, const string &kind) {
	serializer.WriteProperty(1, "format", "vane." + kind);
	serializer.WriteProperty(2, "version", FORMAT_VERSION);
	serializer.WriteProperty(3, "engine", EngineIdentity());
}

void ReadIdentity(Deserializer &deserializer, const string &kind) {
	if (deserializer.ReadProperty<string>(1, "format") != "vane." + kind ||
	    deserializer.ReadProperty<idx_t>(2, "version") != FORMAT_VERSION) {
		throw SerializationException("unsupported native %s format", kind);
	}
	if (deserializer.ReadProperty<string>(3, "engine") != EngineIdentity()) {
		throw SerializationException("native fragment engine identity mismatch");
	}
}

} // namespace

string EngineIdentity() {
	return string(DuckDB::SourceID()) + ":fragment:" + VANE_FRAGMENT_BUILD_ID;
}

string SerializeSchema(const vector<LogicalType> &types) {
	return Encode([&](Serializer &serializer) {
		WriteIdentity(serializer, "schema");
		serializer.WriteProperty(10, "types", types);
	});
}

vector<LogicalType> DeserializeSchema(const string &payload) {
	vector<LogicalType> types;
	Decode(payload, [&](BinaryDeserializer &deserializer) {
		ReadIdentity(deserializer, "schema");
		types = deserializer.ReadProperty<vector<LogicalType>>(10, "types");
	});
	CheckTypes(types);
	return types;
}

void SourceSpec::Serialize(Serializer &serializer) const {
	serializer.WriteProperty(1, "source_id", source_id);
	serializer.WriteProperty(2, "function", function_name);
	serializer.WriteProperty(3, "capability", capability);
	serializer.WriteProperty(4, "codec", codec);
	serializer.WriteProperty(5, "splits", splits);
	serializer.WriteProperty(6, "requires_snapshot", requires_snapshot);
}

SourceSpec SourceSpec::Deserialize(Deserializer &deserializer) {
	SourceSpec result;
	result.source_id = deserializer.ReadProperty<string>(1, "source_id");
	result.function_name = deserializer.ReadProperty<string>(2, "function");
	result.capability = deserializer.ReadProperty<string>(3, "capability");
	result.codec = deserializer.ReadProperty<string>(4, "codec");
	result.splits = deserializer.ReadProperty<vector<DistributedScanSplit>>(5, "splits");
	result.requires_snapshot = deserializer.ReadProperty<bool>(6, "requires_snapshot");
	if (result.source_id.empty() || result.capability.empty() || result.codec.empty()) {
		throw SerializationException("empty fragment source identity");
	}
	std::set<string> ids;
	for (auto &split : result.splits) {
		split.Validate();
		if (!ids.insert(split.split_id).second) {
			throw SerializationException("duplicate fragment split id");
		}
	}
	return result;
}

void PlanNode::Serialize(Serializer &serializer) const {
	serializer.WriteProperty(1, "input_port", input_port);
	serializer.WriteProperty(2, "types", types);
	serializer.WriteProperty(3, "operator", native_operator);
	serializer.WriteProperty(4, "source_id", source_id);
	serializer.WriteProperty(5, "children", children);
}

PlanNode PlanNode::Deserialize(Deserializer &deserializer) {
	PlanNode result;
	result.input_port = deserializer.ReadProperty<string>(1, "input_port");
	result.types = deserializer.ReadProperty<vector<LogicalType>>(2, "types");
	result.native_operator = deserializer.ReadProperty<string>(3, "operator");
	result.source_id = deserializer.ReadProperty<string>(4, "source_id");
	result.children = deserializer.ReadProperty<vector<PlanNode>>(5, "children");
	CheckTypes(result.types);
	if (!result.input_port.empty()) {
		if (!result.native_operator.empty() || !result.source_id.empty() || !result.children.empty()) {
			throw SerializationException("fragment input port contains operator state");
		}
	} else if (result.native_operator.empty()) {
		throw SerializationException("fragment node has no native operator");
	}
	return result;
}

string FragmentSpec::Serialize() const {
	return Encode([&](Serializer &serializer) {
		WriteIdentity(serializer, "fragment");
		serializer.WriteProperty(10, "fragment_id", fragment_id);
		serializer.WriteProperty(11, "partitions", partition_count);
		serializer.WriteProperty(12, "names", names);
		serializer.WriteProperty(13, "root", root);
		serializer.WriteProperty(14, "sources", sources);
		serializer.WriteProperty(15, "source_dependencies", source_dependencies);
	});
}

FragmentSpec FragmentSpec::Deserialize(const string &payload) {
	FragmentSpec result;
	Decode(payload, [&](BinaryDeserializer &deserializer) {
		ReadIdentity(deserializer, "fragment");
		result.fragment_id = deserializer.ReadProperty<string>(10, "fragment_id");
		result.partition_count = deserializer.ReadProperty<idx_t>(11, "partitions");
		result.names = deserializer.ReadProperty<vector<string>>(12, "names");
		result.root = deserializer.ReadProperty<PlanNode>(13, "root");
		result.sources = deserializer.ReadProperty<vector<SourceSpec>>(14, "sources");
		result.source_dependencies = deserializer.ReadProperty<vector<SourceSpec>>(15, "source_dependencies");
	});
	CheckPartitions(result.partition_count);
	if (result.fragment_id.empty() || result.names.size() != result.root.types.size()) {
		throw SerializationException("invalid fragment identity or result names");
	}
	return result;
}

namespace {

bool IsParquetScan(const string &name) {
	return name == "read_parquet" || name == "parquet_scan";
}

bool SupportedScan(const string &name) {
	return name == "range" || name == "generate_series" || IsParquetScan(name);
}

void CheckScanColumns(const string &name, const vector<ColumnIndex> &column_ids) {
	if (!IsParquetScan(name)) {
		return;
	}
	for (auto &column : column_ids) {
		if (column.GetPrimaryIndex() == MultiFileReader::COLUMN_IDENTIFIER_FILE_INDEX) {
			// A task's file list has different indices from the bound query's list.
			// Match the virtual column ID so physical columns named file_index work.
			throw NotImplementedException("fragment compiler does not support Parquet virtual file_index");
		}
	}
}

void CheckFunctionOrigin(CatalogEntry &entry) {
	switch (entry.type) {
	case CatalogType::SCALAR_FUNCTION_ENTRY:
	case CatalogType::TABLE_FUNCTION_ENTRY:
	case CatalogType::AGGREGATE_FUNCTION_ENTRY:
	case CatalogType::MACRO_ENTRY:
	case CatalogType::TABLE_MACRO_ENTRY:
		if (!entry.internal) {
			throw NotImplementedException("fragment compiler requires a built-in function: %s", entry.name);
		}
		break;
	default:
		break;
	}
}

void CheckParsedExpression(ClientContext &context, ParsedExpression &expression) {
	if (expression.GetExpressionClass() == ExpressionClass::FUNCTION) {
		auto &function = expression.Cast<FunctionExpression>();
		static const std::set<string> scalar_functions = {"+",      "-",    "*",           "/",        "//",
		                                                  "%",      "||",   "abs",         "lower",    "upper",
		                                                  "length", "hash", "starts_with", "contains", "list_value"};
		const auto name = StringUtil::Lower(function.function_name);
		const bool scan = SupportedScan(name);
		if (!scan && !scalar_functions.count(name)) {
			throw NotImplementedException("fragment compiler does not support function %s", name);
		}
		auto &entry =
		    Catalog::GetEntry(context, scan ? CatalogType::TABLE_FUNCTION_ENTRY : CatalogType::SCALAR_FUNCTION_ENTRY,
		                      function.catalog, function.schema, name);
		CheckFunctionOrigin(entry);
	}
	// A subquery expression can hide functions from ordinary expression traversal.
	// Table subqueries are visited by EnumerateQueryNodeChildren instead.
	if (expression.GetExpressionClass() == ExpressionClass::SUBQUERY ||
	    expression.GetExpressionClass() == ExpressionClass::PARAMETER) {
		throw NotImplementedException("fragment compiler does not support subquery expressions or parameters");
	}
	ParsedExpressionIterator::EnumerateChildren(
	    expression, [&](ParsedExpression &child) { CheckParsedExpression(context, child); });
}

SourceSpec ParquetSource(const string &id, const string &name, FunctionData *bind_data);

class LogicalValidator : public LogicalOperatorVisitor {
public:
	vector<SourceSpec> source_dependencies;
	bool frozen_files = false;

	void VisitOperator(LogicalOperator &op) override {
		switch (op.type) {
		case LogicalOperatorType::LOGICAL_GET: {
			auto &get = op.Cast<LogicalGet>();
			if (!SupportedScan(get.function.name) || !get.children.empty()) {
				throw NotImplementedException("fragment compiler does not support scan %s", get.function.name);
			}
			// Validate before filter pushdown can remove a filter-only virtual column.
			CheckScanColumns(get.function.name, get.GetColumnIds());
			if (frozen_files && IsParquetScan(get.function.name)) {
				for (auto &column : get.GetColumnIds()) {
					if (column.GetPrimaryIndex() == MultiFileReader::COLUMN_IDENTIFIER_FILENAME) {
						throw NotImplementedException("FTE snapshots do not support the virtual filename column");
					}
				}
			}
			if (IsParquetScan(get.function.name)) {
				// Statistics and file pruning depend on the bound file set even if
				// optimization later removes every executable scan.
				source_dependencies.push_back(ParquetSource("dependency" + std::to_string(source_dependencies.size()),
				                                            get.function.name, get.bind_data.get()));
			}
			break;
		}
		case LogicalOperatorType::LOGICAL_PROJECTION:
		case LogicalOperatorType::LOGICAL_FILTER:
		case LogicalOperatorType::LOGICAL_DUMMY_SCAN:
		case LogicalOperatorType::LOGICAL_EXPRESSION_GET:
		case LogicalOperatorType::LOGICAL_EMPTY_RESULT:
		case LogicalOperatorType::LOGICAL_CHUNK_GET:
			break;
		default:
			throw NotImplementedException("fragment compiler does not support logical operator %s", op.GetName());
		}
		LogicalOperatorVisitor::VisitOperator(op);
	}

	void VisitExpression(unique_ptr<Expression> *expression) override {
		// Query-stable functions (e.g. CURRENT_TIMESTAMP) also change across tasks
		// and retries. Reject them before optional constant folding can hide them.
		if (!(*expression)->IsConsistent()) {
			throw NotImplementedException(
			    "fragment compiler requires expressions consistent across tasks and attempts");
		}
		LogicalOperatorVisitor::VisitExpression(expression);
	}
};

void CheckOperator(PhysicalOperator &op, idx_t children) {
	CheckTypes(op.types);
	switch (op.type) {
	case PhysicalOperatorType::PROJECTION:
	case PhysicalOperatorType::FILTER:
	case PhysicalOperatorType::EXPRESSION_SCAN:
		if (children == 1) {
			return;
		}
		break;
	case PhysicalOperatorType::DUMMY_SCAN:
	case PhysicalOperatorType::EMPTY_RESULT:
	case PhysicalOperatorType::COLUMN_DATA_SCAN:
		if (children == 0) {
			return;
		}
		break;
	case PhysicalOperatorType::TABLE_SCAN: {
		auto &scan = op.Cast<PhysicalTableScan>();
		if (children == 0 && SupportedScan(scan.function.name)) {
			CheckScanColumns(scan.function.name, scan.column_ids);
			return;
		}
		break;
	}
	default:
		break;
	}
	throw NotImplementedException("fragment compiler does not support physical operator %s with %llu children",
	                              op.GetName(), children);
}

TableFunctionDistributedScanInput ScanInput(PhysicalTableScan &scan) {
	return TableFunctionDistributedScanInput(scan.bind_data.get(), scan.parameters, scan.column_ids,
	                                         scan.projection_ids, scan.table_filters.get(), scan.estimated_cardinality);
}

MultiFileBindData &ParquetBind(FunctionData *bind_data) {
	auto bind = dynamic_cast<MultiFileBindData *>(bind_data);
	if (!bind || !bind->file_list) {
		throw SerializationException("Parquet fragment requires a native multi-file bind");
	}
	return *bind;
}

string EncodeFile(const OpenFileInfo &file) {
	return Encode([&](Serializer &serializer) {
		serializer.WriteProperty(1, "path", file.path);
		unordered_map<string, Value> options;
		if (file.extended_info) {
			options = file.extended_info->options;
		}
		serializer.WriteProperty(2, "options", options);
	});
}

SourceSpec ParquetSource(const string &id, const string &name, FunctionData *bind_data) {
	SourceSpec source;
	source.source_id = id;
	source.function_name = name;
	source.capability = PARQUET_CAPABILITY;
	source.codec = PARQUET_CODEC;
	source.requires_snapshot = true;
	for (auto &file : ParquetBind(bind_data).file_list->GetAllFiles()) {
		DistributedScanSplit split;
		split.split_id = "file" + std::to_string(source.splits.size());
		split.payload = EncodeFile(file);
		source.splits.push_back(std::move(split));
	}
	return source;
}

OpenFileInfo DecodeFile(const string &payload) {
	OpenFileInfo file;
	Decode(payload, [&](BinaryDeserializer &deserializer) {
		file.path = deserializer.ReadProperty<string>(1, "path");
		auto options = deserializer.ReadProperty<unordered_map<string, Value>>(2, "options");
		if (!options.empty()) {
			file.extended_info = make_shared_ptr<ExtendedOpenFileInfo>();
			file.extended_info->options = std::move(options);
		}
	});
	if (file.path.empty()) {
		throw SerializationException("empty Parquet fragment file path");
	}
	return file;
}

PlanNode EncodeNode(ClientContext &context, PhysicalOperator &op, FragmentSpec &fragment, idx_t partitions) {
	CheckOperator(op, op.children.size());
	PlanNode node;
	node.types = op.types;
	if (op.type == PhysicalOperatorType::TABLE_SCAN) {
		auto &scan = op.Cast<PhysicalTableScan>();
		if (!scan.function.HasSerializationCallbacks()) {
			throw NotImplementedException("scan %s has no portable split contract", scan.function.name);
		}
		SourceSpec source;
		const auto source_id = "source" + std::to_string(fragment.sources.size());
		source.source_id = source_id;
		source.function_name = scan.function.name;
		if (IsParquetScan(scan.function.name)) {
			source = ParquetSource(source_id, scan.function.name, scan.bind_data.get());
			auto &bind = ParquetBind(scan.bind_data.get());
			// Copy drops cached coordinator readers; workers receive files only
			// through their explicit assignments, never by re-expanding a glob.
			auto worker_bind = bind.Copy();
			worker_bind->Cast<MultiFileBindData>().file_list =
			    make_shared_ptr<SimpleMultiFileList>(vector<OpenFileInfo>());
			scan.bind_data = std::move(worker_bind);
		} else {
			if (!scan.function.HasDistributedScanCallbacks()) {
				throw NotImplementedException("scan %s has no portable split contract", scan.function.name);
			}
			auto &callbacks = scan.function.GetDistributedScanCallbacks();
			callbacks.Validate(scan.function);
			auto input = ScanInput(scan);
			source.capability = callbacks.GetCapability().CanonicalIdentity();
			source.codec = callbacks.split_codec.CanonicalIdentity();
			source.splits = callbacks.plan_splits(TableFunctionDistributedScanPlanningInput(
			    input, partitions, FileSystem::GetFileSystem(context), &context));
			// Only the independently owned worker bind crosses the process boundary.
			scan.bind_data = callbacks.create_worker_bind(input);
		}
		std::set<string> ids;
		for (auto &split : source.splits) {
			split.Validate();
			if (!ids.insert(split.split_id).second) {
				throw InvalidInputException("scan produced duplicate split ids");
			}
		}
		node.source_id = source.source_id;
		fragment.sources.push_back(std::move(source));
	}
	for (auto &child : op.children) {
		node.children.push_back(EncodeNode(context, child.get(), fragment, partitions));
	}
	// Serialize each native node separately so an input is a real port in the
	// fragment codec, never a fake empty scan hidden in executable plan bytes.
	node.native_operator = Encode([&](Serializer &serializer) { op.SerializeNode(serializer); });
	return node;
}

FragmentSpec InputFragment(const string &id, idx_t partitions, const vector<string> &names,
                           const vector<LogicalType> &types) {
	FragmentSpec fragment;
	fragment.fragment_id = id;
	fragment.partition_count = partitions;
	fragment.names = names;
	fragment.root.input_port = "in";
	fragment.root.types = types;
	return fragment;
}

string HashExpression(const vector<LogicalType> &types, const vector<idx_t> &columns) {
	vector<unique_ptr<Expression>> expressions;
	std::set<idx_t> seen;
	for (auto column : columns) {
		if (column >= types.size() || !seen.insert(column).second) {
			throw InvalidInputException("HASH columns must be distinct valid result-column positions");
		}
		expressions.push_back(make_uniq<BoundReferenceExpression>(types[column], column));
	}
	return Encode([&](Serializer &serializer) {
		WriteIdentity(serializer, "hash");
		serializer.WriteProperty(10, "input_types", types);
		serializer.WriteProperty(11, "expressions", expressions);
	});
}

} // namespace

namespace {
void FilePatterns(ParsedExpression &expression, vector<string> &patterns, bool allow_list = true) {
	if (expression.GetExpressionClass() == ExpressionClass::CONSTANT) {
		auto &value = expression.Cast<ConstantExpression>().value;
		if (!value.IsNull() && value.type().id() == LogicalTypeId::VARCHAR) {
			patterns.push_back(value.GetValue<string>());
			return;
		}
	} else if (allow_list && expression.GetExpressionClass() == ExpressionClass::FUNCTION) {
		auto &function = expression.Cast<FunctionExpression>();
		if (StringUtil::Lower(function.function_name) == "list_value") {
			for (auto &child : function.children) {
				FilePatterns(*child, patterns, false);
			}
			return;
		}
	}
	throw NotImplementedException("FTE file inputs require literal absolute paths or lists of literal paths");
}

void StageFileArguments(ClientContext &context, SelectStatement &select, const string &directory, idx_t budget,
                        unordered_map<string, FrozenFile> &frozen, idx_t &used) {
	ParsedExpressionIterator::EnumerateQueryNodeChildren(
	    *select.node, [](unique_ptr<ParsedExpression> &) {},
	    [&](TableRef &ref) {
		    if (ref.type != TableReferenceType::TABLE_FUNCTION) {
			    return;
		    }
		    auto &expression = *ref.Cast<TableFunctionRef>().function;
		    if (expression.GetExpressionClass() != ExpressionClass::FUNCTION) {
			    return;
		    }
		    auto &function = expression.Cast<FunctionExpression>();
		    if (!IsParquetScan(StringUtil::Lower(function.function_name))) {
			    return;
		    }
		    if (function.children.empty()) {
			    throw InvalidInputException("Parquet scan requires file paths");
		    }
		    vector<string> patterns;
		    FilePatterns(*function.children[0], patterns);
		    if (patterns.empty() || patterns.size() > 4096) {
			    throw InvalidInputException("FTE source pattern count exceeds limit");
		    }
		    vector<Value> paths;
		    vector<OpenFileInfo> files;
		    auto &fs = FileSystem::GetFileSystem(context);
		    for (auto &pattern : patterns) {
			    if (!fs.IsPathAbsolute(pattern)) {
				    throw NotImplementedException("FTE snapshots require absolute local paths");
			    }
			    for (auto &file : fs.GlobFiles(pattern)) {
				    if (paths.size() >= 4096) {
					    throw InvalidInputException("FTE source file reference count exceeds limit");
				    }
				    auto target = FrozenFilePath(context, file.path, directory);
				    auto existing = frozen.find(target);
				    if (existing == frozen.end()) {
					    if (frozen.size() >= 4096) {
						    throw InvalidInputException("FTE source file count exceeds limit");
					    }
					    auto snapshot = FreezeFile(context, file.path, target, budget - used);
					    used += snapshot.bytes;
					    existing = frozen.emplace(target, std::move(snapshot)).first;
				    }
				    paths.push_back(Value(existing->second.path));
				    files.emplace_back(existing->second.path);
			    }
		    }
		    function.children[0] = make_uniq<ConstantExpression>(Value::LIST(LogicalType::VARCHAR, std::move(paths)));
		    MultiFileReader::SetFileList(ref.Cast<TableFunctionRef>(), std::move(files));
	    });
}

void MarkFrozenSources(vector<SourceSpec> &sources, const unordered_map<string, FrozenFile> &frozen) {
	for (auto &source : sources) {
		if (!IsParquetScan(source.function_name)) {
			continue;
		}
		for (auto &split : source.splits) {
			auto file = DecodeFile(split.payload);
			auto found = frozen.find(file.path);
			if (found == frozen.end()) {
				throw InternalException("bound file was not frozen before compilation");
			}
			if (!file.extended_info) {
				file.extended_info = make_shared_ptr<ExtendedOpenFileInfo>();
			}
			file.extended_info->options["vane_snapshot_sha256"] = Value(found->second.sha256);
			file.extended_info->options["vane_snapshot_bytes"] = Value::UBIGINT(found->second.bytes);
			split.payload = EncodeFile(file);
		}
		source.codec = FROZEN_PARQUET_CODEC;
		source.requires_snapshot = false;
	}
}
} // namespace

FragmentGraph Compile(ClientContext &context, const string &sql, const string &query_id, idx_t partitions,
                      const vector<idx_t> &hash_columns, const string &snapshot_directory, idx_t source_budget,
                      idx_t *source_bytes) {
	CheckPartitions(partitions);
	auto trimmed_id = query_id;
	StringUtil::Trim(trimmed_id);
	if (trimmed_id.empty()) {
		throw InvalidInputException("fragment query id must be non-empty");
	}
	Parser parser(context.GetParserOptions());
	parser.ParseQuery(sql);
	if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
		throw NotImplementedException("fragment compiler requires exactly one SELECT statement");
	}
	FragmentGraph graph;
	graph.query_id = query_id;
	unordered_map<string, FrozenFile> frozen;
	idx_t used = 0;
	context.RunFunctionInTransaction([&]() {
		auto &select = parser.statements[0]->Cast<SelectStatement>();
		ParsedExpressionIterator::EnumerateQueryNodeChildren(
		    *select.node,
		    [&](unique_ptr<ParsedExpression> &expression) { CheckParsedExpression(context, *expression); },
		    [](TableRef &ref) {
			    if (ref.type != TableReferenceType::EMPTY_FROM && ref.type != TableReferenceType::TABLE_FUNCTION &&
			        ref.type != TableReferenceType::SUBQUERY && ref.type != TableReferenceType::EXPRESSION_LIST) {
				    throw NotImplementedException("fragment compiler requires explicit range or Parquet sources");
			    }
		    });
		if (!snapshot_directory.empty()) {
			StageFileArguments(context, select, snapshot_directory, source_budget, frozen, used);
		}
		Planner planner(context);
		// Column references can become SQL value functions during binding. Check
		// the resolved entry before macro expansion or table-argument evaluation;
		// child binders inherit this callback, while real columns need no lookup.
		planner.binder->SetCatalogLookupCallback(CheckFunctionOrigin);
		planner.CreatePlan(std::move(parser.statements[0]));
		if (!planner.plan || !planner.properties.IsReadOnly() || planner.properties.parameter_count) {
			throw NotImplementedException("fragment compiler requires a bound read-only query without parameters");
		}
		LogicalValidator validator;
		validator.frozen_files = !snapshot_directory.empty();
		validator.VisitOperator(*planner.plan);
		CheckTypes(planner.types);
		if (ClientConfig::GetConfig(context).enable_optimizer && planner.plan->RequireOptimizer()) {
			Optimizer optimizer(*planner.binder, context);
			planner.plan = optimizer.Optimize(std::move(planner.plan));
		}
		PhysicalPlanGenerator physical_planner(context);
		auto physical = physical_planner.Plan(std::move(planner.plan));
		FragmentSpec source;
		source.fragment_id = "fragment0";
		source.names = planner.names;
		source.source_dependencies = std::move(validator.source_dependencies);
		source.root = EncodeNode(context, physical->Root(), source, partitions);
		// Constants execute once. Parallelism comes only from independently
		// assignable scan splits, not from duplicating a complete local query.
		source.partition_count = source.sources.empty() ? 1 : partitions;
		graph.fragments.push_back(std::move(source));
		if (!hash_columns.empty()) {
			ExchangeSpec hash;
			hash.exchange_id = "exchange0";
			hash.producer = "fragment0";
			hash.consumer = "fragment1";
			hash.distribution = "hash";
			hash.partitioning = HashExpression(planner.types, hash_columns);
			graph.exchanges.push_back(std::move(hash));
			graph.fragments.push_back(InputFragment("fragment1", partitions, planner.names, planner.types));
		}
		if (graph.fragments.back().partition_count != 1) {
			ExchangeSpec gather;
			gather.exchange_id = "exchange" + std::to_string(graph.exchanges.size());
			gather.producer = graph.fragments.back().fragment_id;
			gather.consumer = "fragment" + std::to_string(graph.fragments.size());
			gather.distribution = "gather";
			graph.fragments.push_back(InputFragment(gather.consumer, 1, planner.names, planner.types));
			graph.exchanges.push_back(std::move(gather));
		}
	});
	if (!snapshot_directory.empty()) {
		for (auto &fragment : graph.fragments) {
			MarkFrozenSources(fragment.sources, frozen);
			MarkFrozenSources(fragment.source_dependencies, frozen);
		}
	}
	if (source_bytes) {
		*source_bytes = used;
	}
	return graph;
}

namespace {

PhysicalOperator &LoadNode(ClientContext &context, PhysicalPlan &plan, const PlanNode &node, const InputFactory &inputs,
                           const unordered_map<string, const SourceSpec *> &sources,
                           const SourceAssignments &assignments, std::set<string> &used_sources,
                           std::set<string> &used_inputs) {
	if (!node.input_port.empty()) {
		if (!inputs || !used_inputs.insert(node.input_port).second) {
			throw InvalidInputException("fragment input %s must be bound exactly once", node.input_port);
		}
		auto &input = inputs(plan, node.input_port, node.types);
		if (input.types != node.types) {
			throw InvalidInputException("fragment input %s schema mismatch", node.input_port);
		}
		return input;
	}
	unique_ptr<PhysicalOperator> op;
	bound_parameter_map_t parameters;
	Decode(node.native_operator, [&](BinaryDeserializer &deserializer) {
		deserializer.Set<ClientContext &>(context);
		deserializer.Set<DatabaseInstance &>(DatabaseInstance::GetDatabase(context));
		deserializer.Set<bound_parameter_map_t &>(parameters);
		op = PhysicalOperator::Deserialize(deserializer, plan);
	});
	if (!op->children.empty() || op->types != node.types) {
		throw SerializationException("fragment node contains unexpected children or schema");
	}
	CheckOperator(*op, node.children.size());
	if (op->type == PhysicalOperatorType::TABLE_SCAN) {
		auto source = sources.find(node.source_id);
		auto assignment = assignments.find(node.source_id);
		if (source == sources.end() || assignment == assignments.end() || !used_sources.insert(node.source_id).second) {
			throw InvalidInputException("fragment source %s must have one explicit split assignment", node.source_id);
		}
		auto &spec = *source->second;
		auto &scan = op->Cast<PhysicalTableScan>();
		const bool parquet = IsParquetScan(scan.function.name);
		string capability;
		string codec;
		if (parquet) {
			capability = PARQUET_CAPABILITY;
			codec = spec.codec == FROZEN_PARQUET_CODEC ? FROZEN_PARQUET_CODEC : PARQUET_CODEC;
		} else {
			auto &callbacks = scan.function.GetDistributedScanCallbacks();
			callbacks.Validate(scan.function);
			capability = callbacks.GetCapability().CanonicalIdentity();
			codec = callbacks.split_codec.CanonicalIdentity();
		}
		if (scan.function.name != spec.function_name || capability != spec.capability || codec != spec.codec) {
			throw SerializationException("fragment source capability or codec mismatch");
		}
		if (spec.requires_snapshot != (parquet && codec == PARQUET_CODEC)) {
			throw SerializationException("fragment source snapshot requirement mismatch");
		}
		unordered_map<string, const DistributedScanSplit *> known;
		for (auto &split : spec.splits) {
			known.emplace(split.split_id, &split);
		}
		vector<DistributedScanSplit> selected;
		std::set<string> selected_ids;
		for (auto &id : assignment->second) {
			auto found = known.find(id);
			if (found == known.end() || !selected_ids.insert(id).second) {
				throw InvalidInputException("unknown or duplicate assigned split %s", id);
			}
			selected.push_back(*found->second);
		}
		if (parquet) {
			vector<OpenFileInfo> files;
			for (auto &split : selected) {
				files.push_back(DecodeFile(split.payload));
			}
			scan.extra_info.total_files = files.size();
			scan.extra_info.filtered_files = files.size();
			ParquetBind(scan.bind_data.get()).file_list = make_shared_ptr<SimpleMultiFileList>(std::move(files));
		} else {
			scan.function.GetDistributedScanCallbacks().apply_splits(scan.bind_data.get(), selected);
		}
		scan.distributed_scan_splits_applied = true;
		scan.distributed_scan_empty = selected.empty();
	} else if (!node.source_id.empty()) {
		throw SerializationException("non-scan fragment node has a source id");
	}
	for (auto &child : node.children) {
		op->children.push_back(LoadNode(context, plan, child, inputs, sources, assignments, used_sources, used_inputs));
	}
	auto &result = *op;
	plan.TakeOwnership(std::move(op));
	return result;
}

} // namespace

unique_ptr<PhysicalPlan> Load(ClientContext &context, const FragmentSpec &fragment, const InputFactory &inputs,
                              const SourceAssignments &assignments) {
	unordered_map<string, const SourceSpec *> sources;
	for (auto &source : fragment.sources) {
		if (!sources.emplace(source.source_id, &source).second) {
			throw SerializationException("duplicate fragment source id");
		}
	}
	if (assignments.size() != sources.size()) {
		throw InvalidInputException("all fragment sources need an explicit assignment, including empty assignments");
	}
	std::set<string> used_sources;
	std::set<string> used_inputs;
	auto plan = make_uniq<PhysicalPlan>(Allocator::Get(context));
	context.RunFunctionInTransaction([&]() {
		plan->SetRoot(LoadNode(context, *plan, fragment.root, inputs, sources, assignments, used_sources, used_inputs));
	});
	if (used_sources.size() != sources.size()) {
		throw SerializationException("fragment contains unused source descriptions");
	}
	return plan;
}

vector<idx_t> HashPartitions(ClientContext &context, const string &partitioning, DataChunk &chunk, idx_t partitions) {
	CheckPartitions(partitions);
	vector<unique_ptr<Expression>> expressions;
	vector<LogicalType> types;
	Decode(partitioning, [&](BinaryDeserializer &deserializer) {
		ReadIdentity(deserializer, "hash");
		deserializer.Set<ClientContext &>(context);
		deserializer.Set<DatabaseInstance &>(DatabaseInstance::GetDatabase(context));
		types = deserializer.ReadProperty<vector<LogicalType>>(10, "input_types");
		expressions = deserializer.ReadProperty<vector<unique_ptr<Expression>>>(11, "expressions");
	});
	if (types != chunk.GetTypes() || expressions.empty()) {
		throw InvalidInputException("HASH partitioning requires matching input schema and non-empty keys");
	}
	vector<LogicalType> key_types;
	for (auto &expression : expressions) {
		if (expression->GetExpressionClass() != ExpressionClass::BOUND_REF) {
			throw SerializationException("unsupported native HASH expression");
		}
		auto &ref = expression->Cast<BoundReferenceExpression>();
		if (ref.index >= types.size() || ref.return_type != types[ref.index]) {
			throw SerializationException("native HASH reference does not match input schema");
		}
		key_types.push_back(ref.return_type);
	}
	ExpressionExecutor executor(context, expressions);
	DataChunk keys;
	keys.Initialize(Allocator::Get(context), key_types);
	executor.Execute(chunk, keys);
	Vector hashes(LogicalType::HASH);
	keys.Hash(hashes);
	UnifiedVectorFormat data;
	hashes.ToUnifiedFormat(chunk.size(), data);
	auto values = UnifiedVectorFormat::GetData<hash_t>(data);
	vector<idx_t> result;
	for (idx_t i = 0; i < chunk.size(); i++) {
		result.push_back(values[data.sel->get_index(i)] % partitions);
	}
	return result;
}

namespace {

const vector<string> &ConnectionSettings() {
	static const vector<string> names = {"integer_division",
	                                     "ieee_floating_point_ops",
	                                     "old_implicit_casting",
	                                     "default_collation",
	                                     "default_order",
	                                     "default_null_order",
	                                     "preserve_identifier_case",
	                                     "max_expression_depth",
	                                     "disabled_optimizers",
	                                     "TimeZone",
	                                     "Calendar"};
	return names;
}

vector<Value> ReadConnectionSettings(ClientContext &context) {
	vector<Value> result;
	for (auto &name : ConnectionSettings()) {
		Value value;
		if (!context.TryGetCurrentSetting(name, value)) {
			throw NotImplementedException("submission connection profile requires setting %s", name);
		}
		if (name == "default_collation" && !value.ToString().empty()) {
			throw NotImplementedException("submission profile does not support custom collations");
		}
		// Defaults and explicitly set values can use different aliases (e.g.
		// ASCENDING vs ASC). These callbacks only normalize their input Value.
		SettingCallbackInfo info(context, SetScope::SESSION);
		if (name == "default_order") {
			DefaultOrderSetting::OnSet(info, value);
		} else if (name == "default_null_order") {
			DefaultNullOrderSetting::OnSet(info, value);
		}
		result.push_back(std::move(value));
	}
	return result;
}

void SetConnectionSetting(ClientContext &context, const string &name, const Value &value) {
	auto &config = DBConfig::GetConfig(context);
	config.CheckLock(name);
	auto option = DBConfig::GetOptionByName(name);
	if (!option) {
		ExtensionOption extension;
		if (!config.TryGetExtensionOption(name, extension)) {
			throw NotImplementedException("worker lacks connection setting %s", name);
		}
		PhysicalSet::SetExtensionVariable(context, extension, name, SetScope::SESSION, value);
		return;
	}
	// Custom settings can have only a global setter even when their declared
	// scope is GLOBAL_DEFAULT. Never mutate a shared database during prepare.
	if (option->scope == SettingScopeTarget::GLOBAL_ONLY || (!option->default_value && !option->set_local)) {
		Value current;
		if (!context.TryGetCurrentSetting(name, current) || current != value) {
			throw InvalidInputException("worker database setting %s does not match submission", name);
		}
		return;
	}
	auto converted = value.CastAs(context, DBConfig::ParseLogicalType(option->parameter_type));
	if (option->default_value) {
		if (option->set_callback) {
			SettingCallbackInfo info(context, SetScope::SESSION);
			option->set_callback(info, converted);
		}
		PhysicalSet::SetGenericVariable(context, option->setting_idx.GetIndex(), SetScope::SESSION,
		                                std::move(converted));
	} else if (option->set_local) {
		option->set_local(context, converted);
	} else {
		throw NotImplementedException("worker cannot restore session setting %s", name);
	}
}

string FileStamp(ClientContext &context, const OpenFileInfo &file) {
	auto &fs = FileSystem::GetFileSystem(context);
	if (!fs.IsPathAbsolute(file.path)) {
		throw NotImplementedException("submission file sources require absolute local paths");
	}
	auto handle = fs.OpenFile(file, FileFlags::FILE_FLAGS_READ);
	if (!handle->file_system.IsLocalFileSystem()) {
		throw NotImplementedException("submission file sources require regular local files visible to workers");
	}
	auto stats = handle->Stats();
	if (stats.file_type != FileType::FILE_TYPE_REGULAR) {
		throw NotImplementedException("submission file sources require regular local files visible to workers");
	}
	// timestamp_t retains microseconds; the native nanosecond fraction preserves
	// the remaining precision from the same file-handle stat operation.
	auto fraction = stats.extended_file_info.find("mtime_nsec");
	if (fraction == stats.extended_file_info.end()) {
		throw NotImplementedException("submission file sources require precise local modification times");
	}
	auto mtime_nsec = fraction->second.GetValue<int64_t>();
	if (mtime_nsec < 0 || mtime_nsec >= 1000000000) {
		throw IOException("invalid local file modification time fraction");
	}
	return Encode([&](Serializer &serializer) {
		serializer.WriteProperty(1, "path", file.path);
		serializer.WriteProperty(2, "size", stats.file_size);
		serializer.WriteProperty(3, "mtime", stats.last_modification_time.value);
		serializer.WriteProperty(4, "mtime_nsec", mtime_nsec);
	});
}

} // namespace

string CaptureConnection(ClientContext &context) {
	auto values = ReadConnectionSettings(context);
	return Encode([&](Serializer &serializer) {
		WriteIdentity(serializer, "connection");
		serializer.WriteProperty(10, "settings", ConnectionSettings());
		serializer.WriteProperty(11, "values", values);
		serializer.WriteProperty(12, "optimizer", ClientConfig::GetConfig(context).enable_optimizer);
	});
}

void ApplyConnection(ClientContext &context, const string &snapshot) {
	vector<string> names;
	vector<Value> values;
	bool optimizer = true;
	Decode(snapshot, [&](BinaryDeserializer &deserializer) {
		ReadIdentity(deserializer, "connection");
		names = deserializer.ReadProperty<vector<string>>(10, "settings");
		values = deserializer.ReadProperty<vector<Value>>(11, "values");
		optimizer = deserializer.ReadProperty<bool>(12, "optimizer");
	});
	if (names != ConnectionSettings() || values.size() != names.size()) {
		throw SerializationException("unsupported submission connection profile");
	}
	for (idx_t i = 0; i < names.size(); i++) {
		SetConnectionSetting(context, names[i], values[i]);
	}
	ClientConfig::GetConfig(context).enable_optimizer = optimizer;
	if (CaptureConnection(context) != snapshot) {
		throw InvalidInputException("worker connection does not reproduce submission settings");
	}
}

string CaptureSources(ClientContext &context, const FragmentSpec &fragment, bool require_replay) {
	vector<string> identities;
	vector<string> dependencies;
	vector<string> versions;
	std::set<string> stamped_files;
	auto capture_versions = [&](const SourceSpec &source) {
		if (source.codec == FROZEN_PARQUET_CODEC) {
			if (source.requires_snapshot || !IsParquetScan(source.function_name) ||
			    source.capability != PARQUET_CAPABILITY) {
				throw SerializationException("invalid frozen source capability");
			}
			for (auto &split : source.splits) {
				auto file = DecodeFile(split.payload);
				if (!file.extended_info || !file.extended_info->options.count("vane_snapshot_sha256") ||
				    !file.extended_info->options.count("vane_snapshot_bytes")) {
					throw SerializationException("frozen source has no content identity");
				}
				auto &options = file.extended_info->options;
				auto stamp = FileStamp(context, file);
				auto hash = FileFingerprint(context, file);
				auto handle = FileSystem::GetFileSystem(context).OpenFile(file, FileFlags::FILE_FLAGS_READ);
				if (hash != options.at("vane_snapshot_sha256").GetValue<string>() ||
				    idx_t(handle->GetFileSize()) != options.at("vane_snapshot_bytes").GetValue<idx_t>()) {
					throw InvalidInputException("immutable FTE source snapshot changed");
				}
				if (stamped_files.insert(file.path).second) {
					versions.push_back(stamp + hash);
				}
			}
			return;
		}
		if (source.requires_snapshot) {
			if (!IsParquetScan(source.function_name) || require_replay) {
				throw NotImplementedException(
				    "FTE requires an immutable source version; ordinary Parquet files are not replayable");
			}
			for (auto &split : source.splits) {
				auto file = DecodeFile(split.payload);
				if (stamped_files.insert(file.path).second) {
					versions.push_back(FileStamp(context, file));
				}
			}
		}
	};
	for (auto &source : fragment.sources) {
		identities.push_back(Encode([&](Serializer &serializer) { source.Serialize(serializer); }));
		capture_versions(source);
	}
	for (auto &source : fragment.source_dependencies) {
		if (!IsParquetScan(source.function_name) || source.capability != PARQUET_CAPABILITY ||
		    (source.codec != PARQUET_CODEC && source.codec != FROZEN_PARQUET_CODEC) ||
		    source.requires_snapshot != (source.codec == PARQUET_CODEC)) {
			throw SerializationException("unsupported fragment source dependency");
		}
		dependencies.push_back(Encode([&](Serializer &serializer) { source.Serialize(serializer); }));
		capture_versions(source);
	}
	return Encode([&](Serializer &serializer) {
		WriteIdentity(serializer, "sources");
		serializer.WriteProperty(10, "fragment", fragment.fragment_id);
		serializer.WriteProperty(11, "sources", identities);
		serializer.WriteProperty(12, "file_versions", versions);
		serializer.WriteProperty(13, "source_dependencies", dependencies);
	});
}

void ValidateSources(ClientContext &context, const FragmentSpec &fragment, const string &snapshot,
                     bool require_replay) {
	// Decode the envelope before touching a filesystem or comparing source state.
	Decode(snapshot, [&](BinaryDeserializer &deserializer) {
		ReadIdentity(deserializer, "sources");
		deserializer.ReadProperty<string>(10, "fragment");
		deserializer.ReadProperty<vector<string>>(11, "sources");
		deserializer.ReadProperty<vector<string>>(12, "file_versions");
		deserializer.ReadProperty<vector<string>>(13, "source_dependencies");
	});
	if (CaptureSources(context, fragment, require_replay) != snapshot) {
		throw InvalidInputException("submission source snapshot changed or does not match fragment");
	}
}

vector<std::pair<string, string>> ScanCapabilities(ClientContext &context) {
	std::set<std::pair<string, string>> capabilities;
	context.RunFunctionInTransaction([&]() {
		for (auto &name : vector<string> {"range", "generate_series", "read_parquet"}) {
			auto &entry = Catalog::GetEntry<TableFunctionCatalogEntry>(context, "", "", name);
			if (!entry.internal) {
				continue;
			}
			for (auto &function : entry.functions.functions) {
				if (!function.HasSerializationCallbacks()) {
					continue;
				}
				if (IsParquetScan(name)) {
					capabilities.emplace(PARQUET_CAPABILITY, PARQUET_CODEC);
					capabilities.emplace(PARQUET_CAPABILITY, FROZEN_PARQUET_CODEC);
				} else if (function.HasDistributedScanCallbacks()) {
					auto &callbacks = function.GetDistributedScanCallbacks();
					callbacks.Validate(function);
					capabilities.emplace(callbacks.GetCapability().CanonicalIdentity(),
					                     callbacks.split_codec.CanonicalIdentity());
				}
			}
		}
	});
	return vector<std::pair<string, string>>(capabilities.begin(), capabilities.end());
}

} // namespace vane_execution
} // namespace duckdb

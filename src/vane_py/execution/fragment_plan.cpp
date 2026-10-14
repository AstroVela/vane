// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "fragment_plan.hpp"
#include "correlated_partition.hpp"
#include "file_snapshot.hpp"
#include "exchange_types.hpp"
#include "parquet_split.hpp"
#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_perfecthash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_ungrouped_aggregate.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/execution/operator/join/physical_piecewise_merge_join.hpp"
#include "duckdb/execution/physical_plan_inputs.hpp"
#include "duckdb/execution/dynamic_filter_serialization.hpp"
#include "duckdb/execution/operator/join/physical_delim_join.hpp"
#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/parser/expression/subquery_expression.hpp"
#include "duckdb/execution/operator/helper/physical_limit.hpp"
#include "duckdb/execution/operator/helper/physical_streaming_limit.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/order/physical_top_n.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/function/lambda_functions.hpp"
#include "duckdb/function/aggregate/wide_integer.hpp"
#include "duckdb/function/aggregate/wide_interval.hpp"
#include "duckdb/function/scalar/generic_common.hpp"
#include "duckdb/execution/distributed/pipeline_node/aggregate.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"

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

#include <algorithm>
#include <set>

namespace duckdb {
namespace vane_execution {
namespace {

constexpr idx_t FORMAT_VERSION = 2;
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
		CheckExchangeType(type);
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
	// Exported aggregate states require the same native ABI as well as code.
	return string(DuckDB::SourceID()) + ":" + DuckDB::Platform() + ":fragment:" + VANE_FRAGMENT_BUILD_ID;
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
	serializer.WriteProperty(6, "scoped", scoped);
}

PlanNode PlanNode::Deserialize(Deserializer &deserializer) {
	PlanNode result;
	result.input_port = deserializer.ReadProperty<string>(1, "input_port");
	result.types = deserializer.ReadProperty<vector<LogicalType>>(2, "types");
	result.native_operator = deserializer.ReadProperty<string>(3, "operator");
	result.source_id = deserializer.ReadProperty<string>(4, "source_id");
	result.children = deserializer.ReadProperty<vector<PlanNode>>(5, "children");
	result.scoped = deserializer.ReadProperty<bool>(6, "scoped");
	CheckTypes(result.types);
	if (!result.input_port.empty()) {
		if (!result.native_operator.empty() || !result.source_id.empty() || !result.children.empty() || result.scoped) {
			throw SerializationException("fragment input port contains operator state");
		}
	} else if (result.native_operator.empty()) {
		throw SerializationException("fragment node has no native operator");
	}
	if (result.scoped) {
		if (!result.source_id.empty()) {
			throw SerializationException("scoped fragment cannot contain an implicit source");
		}
		std::set<string> ports;
		for (auto &child : result.children) {
			if (child.input_port.empty() || !ports.insert(child.input_port).second) {
				throw SerializationException("scoped fragment requires distinct explicit input ports");
			}
		}
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
	case CatalogType::TABLE_ENTRY:
	case CatalogType::VIEW_ENTRY:
		throw NotImplementedException(
		    "fragment compiler requires explicit range or Parquet sources, or CTE references");
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

void CheckScalarFunction(const ScalarFunction &function, bool bound = false) {
	// Binder's scalar-subquery cardinality guard uses error(), marked volatile
	// to defer its exception until execution. Explicit SQL calls are still checked.
	if (function.GetStability() != FunctionStability::CONSISTENT && (!bound || function.name != "error")) {
		throw NotImplementedException("fragment compiler requires expressions consistent across tasks and attempts: %s",
		                              function.name);
	}
	// CONSISTENT alone does not promise portability. For example current_setting
	// binds a value but rebinds it at the worker, whose execution settings differ.
	if (function.IsClientContextRead()) {
		throw NotImplementedException("fragment compiler does not support unfrozen connection context reads: %s",
		                              function.name);
	}
}

void CheckParsedExpression(ClientContext &context, ParsedExpression &expression) {
	if (expression.GetExpressionClass() == ExpressionClass::FUNCTION) {
		auto &function = expression.Cast<FunctionExpression>();
		static const std::set<string> aggregates = {"count", "count_star", "sum", "avg", "min", "max"};
		const auto name = StringUtil::Lower(function.function_name);
		const bool scan = SupportedScan(name);
		const bool aggregate = aggregates.count(name);
		auto entry = Catalog::GetEntry(context, function.catalog, function.schema,
		                               EntryLookupInfo(scan        ? CatalogType::TABLE_FUNCTION_ENTRY
		                                               : aggregate ? CatalogType::AGGREGATE_FUNCTION_ENTRY
		                                                           : CatalogType::SCALAR_FUNCTION_ENTRY,
		                                               name),
		                               OnEntryNotFound::RETURN_NULL);
		if (!entry) {
			throw NotImplementedException("fragment compiler does not support function %s", name);
		}
		CheckFunctionOrigin(*entry);
		if (!scan && !aggregate) {
			if (entry->type != CatalogType::SCALAR_FUNCTION_ENTRY) {
				throw NotImplementedException("fragment compiler does not support function %s", name);
			}
			// Check before binding can evaluate table arguments. Function names do
			// not define portability; native binding and serialization do.
			// Overloads with different arities can have different stability (age).
			// Unpacked COLUMNS can change the arity during binding.
			bool unpacked = false;
			for (auto &child : function.children) {
				unpacked |= child->GetExpressionType() == ExpressionType::OPERATOR_UNPACK;
			}
			vector<reference<ScalarFunction>> candidates;
			bool has_portable_candidate = false;
			for (auto &overload : entry->Cast<ScalarFunctionCatalogEntry>().functions.functions) {
				const auto arguments = function.children.size();
				if (unpacked || arguments == overload.arguments.size() ||
				    (overload.HasVarArgs() && arguments > overload.arguments.size())) {
					candidates.push_back(overload);
					has_portable_candidate |=
					    overload.GetStability() == FunctionStability::CONSISTENT && !overload.IsClientContextRead();
				}
			}
			for (auto &candidate : candidates) {
				auto &overload = candidate.get();
				// Timezone overloads have the same arity but different time dependencies.
				// Defer query-stable candidates to the selected-overload callback, which
				// runs before any binding-time evaluation. Side effects/context reads
				// are still rejected here, before source staging.
				if (has_portable_candidate && overload.GetStability() == FunctionStability::CONSISTENT_WITHIN_QUERY &&
				    !overload.IsClientContextRead()) {
					continue;
				}
				CheckScalarFunction(overload);
			}
		}
	}
	if (expression.GetExpressionClass() == ExpressionClass::PARAMETER) {
		throw NotImplementedException("fragment compiler does not support parameters");
	}
}

// DuckDB's ordinary expression visitor only visits an IN/ANY comparison
// child, not its query. Walk both to validate/freeze every source before bind.
void VisitParsedQuery(QueryNode &query, const std::function<void(ParsedExpression &)> &expressions,
                      const std::function<void(TableRef &)> &tables) {
	std::function<void(ParsedExpression &)> visit = [&](ParsedExpression &expression) {
		expressions(expression);
		if (expression.GetExpressionClass() == ExpressionClass::SUBQUERY) {
			VisitParsedQuery(*expression.Cast<SubqueryExpression>().subquery->node, expressions, tables);
		}
		ParsedExpressionIterator::EnumerateChildren(expression, [&](ParsedExpression &child) { visit(child); });
	};
	ParsedExpressionIterator::EnumerateQueryNodeChildren(
	    query, [&](unique_ptr<ParsedExpression> &expression) { visit(*expression); }, tables);
}

SourceSpec ParquetSource(const string &id, const string &name, FunctionData *bind_data,
                         ClientContext *context = nullptr, idx_t partitions = 1);

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
				auto &filename_idx = get.bind_data->Cast<MultiFileBindData>().reader_bind.filename_idx;
				for (auto &column : get.GetColumnIds()) {
					auto column_id = column.GetPrimaryIndex();
					// Explicit filename options add an ordinary-index generated column.
					// Use its bound identity so real columns with the same name remain supported.
					if (column_id == MultiFileReader::COLUMN_IDENTIFIER_FILENAME ||
					    (filename_idx.IsValid() && column_id == filename_idx.GetIndex())) {
						throw NotImplementedException("FTE snapshots do not support generated filename columns");
					}
				}
			}
			if (IsParquetScan(get.function.name)) {
				// Late materialization introduces a second scan joined on file_index,
				// which is not stable after assigning files to fragment partitions.
				// Change only this bound scan's capability, not the shared catalog or
				// session optimizer settings; native TopN can still be distributed.
				get.function.late_materialization = false;
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
		case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
		case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
		case LogicalOperatorType::LOGICAL_ANY_JOIN:
		case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		case LogicalOperatorType::LOGICAL_DELIM_GET:
		case LogicalOperatorType::LOGICAL_CTE_REF:
		case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
		case LogicalOperatorType::LOGICAL_DISTINCT:
		case LogicalOperatorType::LOGICAL_ORDER_BY:
		case LogicalOperatorType::LOGICAL_TOP_N:
		case LogicalOperatorType::LOGICAL_LIMIT:
			break;
		default:
			throw NotImplementedException("fragment compiler does not support logical operator %s", op.GetName());
		}
		LogicalOperatorVisitor::VisitOperator(op);
	}

	void VisitExpression(unique_ptr<Expression> *expression) override {
		// Query-stable functions (e.g. CURRENT_TIMESTAMP) also change across tasks
		// and retries. Inspect each function before optional constant folding.
		if ((*expression)->GetExpressionClass() == ExpressionClass::BOUND_FUNCTION) {
			auto &function = (*expression)->Cast<BoundFunctionExpression>();
			CheckScalarFunction(function.function, true);
			// Lambda bodies live in bind data, outside ordinary expression children.
			// Recurse here as well so nested list functions cannot hide a context read.
			if (function.function.HasBindLambdaCallback()) {
				if (!function.bind_info) {
					throw InternalException("bound lambda function has no bind data");
				}
				auto &body = function.bind_info->Cast<ListLambdaBindData>().lambda_expr;
				if (body) {
					VisitExpression(&body);
				}
			}
		}
		LogicalOperatorVisitor::VisitExpression(expression);
	}
};

void CheckOperator(PhysicalOperator &op, idx_t children, bool scoped = false) {
	CheckTypes(op.types);
	switch (op.type) {
	case PhysicalOperatorType::PROJECTION:
	case PhysicalOperatorType::FILTER:
	case PhysicalOperatorType::EXPRESSION_SCAN:
	case PhysicalOperatorType::HASH_GROUP_BY:
	case PhysicalOperatorType::PERFECT_HASH_GROUP_BY:
	case PhysicalOperatorType::UNGROUPED_AGGREGATE:
	case PhysicalOperatorType::ORDER_BY:
	case PhysicalOperatorType::TOP_N:
	case PhysicalOperatorType::STREAMING_LIMIT:
	case PhysicalOperatorType::LIMIT:
		if (children == 1 || (scoped && children == 0 && op.type == PhysicalOperatorType::HASH_GROUP_BY)) {
			return;
		}
		break;
	case PhysicalOperatorType::HASH_JOIN:
		if (children == 2) {
			auto &join = op.Cast<PhysicalHashJoin>();
			if (join.join_type == JoinType::INNER || join.join_type == JoinType::LEFT ||
			    join.join_type == JoinType::RIGHT || join.join_type == JoinType::OUTER ||
			    join.join_type == JoinType::SEMI || join.join_type == JoinType::ANTI ||
			    join.join_type == JoinType::RIGHT_SEMI || join.join_type == JoinType::RIGHT_ANTI ||
			    join.join_type == JoinType::SINGLE || join.join_type == JoinType::MARK) {
				return;
			}
		}
		break;
	case PhysicalOperatorType::CROSS_PRODUCT:
	case PhysicalOperatorType::NESTED_LOOP_JOIN:
	case PhysicalOperatorType::PIECEWISE_MERGE_JOIN:
	case PhysicalOperatorType::IE_JOIN:
	case PhysicalOperatorType::BLOCKWISE_NL_JOIN:
	case PhysicalOperatorType::CTE:
		if (children == 2) {
			return;
		}
		break;
	case PhysicalOperatorType::LEFT_DELIM_JOIN:
	case PhysicalOperatorType::RIGHT_DELIM_JOIN:
		if (children == 1) {
			return;
		}
		break;
	case PhysicalOperatorType::DELIM_SCAN:
	case PhysicalOperatorType::CTE_SCAN:
		if (scoped && children == 0) {
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

SourceSpec ParquetSource(const string &id, const string &name, FunctionData *bind_data, ClientContext *context,
                         idx_t partitions) {
	SourceSpec source;
	source.source_id = id;
	source.function_name = name;
	source.capability = PARQUET_CAPABILITY;
	source.codec = PARQUET_CODEC;
	source.requires_snapshot = true;
	auto &bind = ParquetBind(bind_data);
	auto files =
	    context && partitions > 1 ? PlanParquetRowGroups(*context, bind, partitions) : bind.file_list->GetAllFiles();
	for (auto &file : files) {
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

PlanNode EncodeNode(ClientContext &context, PhysicalOperator &op, FragmentSpec &fragment, idx_t partitions,
                    DynamicTableFilterSerializationState &filters) {
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
			source = ParquetSource(source_id, scan.function.name, scan.bind_data.get(), &context, partitions);
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
		node.children.push_back(EncodeNode(context, child.get(), fragment, partitions, filters));
	}
	// Serialize each native node separately so an input is a real port in the
	// fragment codec, never a fake empty scan hidden in executable plan bytes.
	node.native_operator = Encode([&](Serializer &serializer) {
		serializer.GetSerializationData().SetCustom(filters);
		op.SerializeNode(serializer);
	});
	return node;
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
// A subtree keeps its native operators together until a distribution boundary.
// Incoming edges are attached when the owning fragment receives its identity.
struct DistributedSubtree {
	FragmentSpec fragment;
	vector<ExchangeSpec> incoming;
	std::set<const DynamicTableFilterSet *> local_filters;
};

class AnalyticalPlanner {
public:
	AnalyticalPlanner(ClientContext &context, PhysicalPlan &plan, FragmentGraph &graph, idx_t partitions)
	    : context(context), plan(plan), graph(graph), partitions(partitions) {
	}

	string Finish(DistributedSubtree subtree, const vector<string> &names = {}) {
		auto &fragment = subtree.fragment;
		fragment.fragment_id = "fragment" + std::to_string(graph.fragments.size());
		fragment.names = names;
		if (fragment.names.empty()) {
			for (idx_t col = 0; col < fragment.root.types.size(); col++) {
				fragment.names.push_back("c" + std::to_string(col));
			}
		}
		for (auto &edge : subtree.incoming) {
			edge.consumer = fragment.fragment_id;
			graph.exchanges.push_back(std::move(edge));
		}
		auto id = fragment.fragment_id;
		graph.fragments.push_back(std::move(fragment));
		return id;
	}

	DistributedSubtree Exchange(DistributedSubtree child, const string &distribution, idx_t count,
	                            const vector<idx_t> &keys = {}) {
		ExchangeSpec edge;
		edge.exchange_id = "exchange" + std::to_string(next_exchange++);
		edge.consumer_port = edge.exchange_id;
		edge.distribution = distribution;
		auto types = child.fragment.root.types;
		if (distribution == "hash") {
			edge.partitioning = HashExpression(types, keys);
		}
		edge.producer = Finish(std::move(child));
		DistributedSubtree result;
		result.fragment.partition_count = count;
		result.fragment.root.input_port = edge.consumer_port;
		result.fragment.root.types = std::move(types);
		result.incoming.push_back(std::move(edge));
		return result;
	}

	DistributedSubtree Build(PhysicalOperator &op) {
		CheckOperator(op, op.children.size());
		switch (op.type) {
		case PhysicalOperatorType::CTE:
		case PhysicalOperatorType::LEFT_DELIM_JOIN:
		case PhysicalOperatorType::RIGHT_DELIM_JOIN:
		case PhysicalOperatorType::CROSS_PRODUCT:
		case PhysicalOperatorType::NESTED_LOOP_JOIN:
		case PhysicalOperatorType::PIECEWISE_MERGE_JOIN:
		case PhysicalOperatorType::IE_JOIN:
		case PhysicalOperatorType::BLOCKWISE_NL_JOIN:
			return Scoped(op);
		default:
			break;
		}
		if (op.children.empty()) {
			DistributedSubtree result;
			result.fragment.root = EncodeNode(context, op, result.fragment, partitions, filter_serialization);
			result.fragment.partition_count = result.fragment.sources.empty() ? 1 : partitions;
			if (op.type == PhysicalOperatorType::TABLE_SCAN) {
				auto &scan = op.Cast<PhysicalTableScan>();
				if (scan.dynamic_filters) {
					result.local_filters.insert(scan.dynamic_filters.get());
				}
			}
			return result;
		}
		if (op.type == PhysicalOperatorType::HASH_JOIN) {
			auto &join = op.Cast<PhysicalHashJoin>();
			// MARK requires global NULL/empty-build knowledge. SINGLE must see
			// all matching rows before checking scalar-subquery cardinality.
			// Delimiter elimination can leave correlated metadata on an ordinary
			// join, including when its outer input is empty. Keep its native scope.
			if (join.join_type == JoinType::MARK || join.join_type == JoinType::SINGLE || !join.delim_types.empty()) {
				return Scoped(op);
			}
			return Join(join);
		}
		auto child = Build(op.children[0]);
		switch (op.type) {
		case PhysicalOperatorType::HASH_GROUP_BY: {
			auto &aggregate = op.Cast<PhysicalHashAggregate>();
			if (aggregate.grouping_sets.size() != 1 || !aggregate.grouped_aggregate_data.grouping_functions.empty()) {
				throw NotImplementedException("distributed grouping sets are not supported");
			}
			return Aggregate(op, std::move(child), aggregate.grouped_aggregate_data.groups,
			                 InputAggregates(aggregate.grouped_aggregate_data.aggregates, &aggregate.filter_indexes));
		}
		case PhysicalOperatorType::PERFECT_HASH_GROUP_BY: {
			auto &aggregate = op.Cast<PhysicalPerfectHashAggregate>();
			return Aggregate(op, std::move(child), aggregate.groups,
			                 InputAggregates(aggregate.aggregates, &aggregate.filter_indexes));
		}
		case PhysicalOperatorType::UNGROUPED_AGGREGATE: {
			vector<unique_ptr<Expression>> groups;
			return Aggregate(op, std::move(child), groups,
			                 InputAggregates(op.Cast<PhysicalUngroupedAggregate>().aggregates));
		}
		case PhysicalOperatorType::TOP_N: {
			auto &top = op.Cast<PhysicalTopN>();
			top.dynamic_filter.reset();
			if (child.fragment.partition_count > 1) {
				if (top.limit > NumericLimits<idx_t>::Maximum() - top.offset) {
					throw InvalidInputException("TopN limit plus offset overflows");
				}
				vector<BoundOrderByNode> orders;
				for (auto &order : top.orders) {
					orders.push_back(order.Copy());
				}
				auto &partial = plan.Make<PhysicalTopN>(op.types, std::move(orders), top.limit + top.offset, 0, nullptr,
				                                        op.estimated_cardinality);
				Wrap(partial, child);
			}
			break;
		}
		default:
			break;
		}
		if (op.type == PhysicalOperatorType::ORDER_BY || op.type == PhysicalOperatorType::TOP_N ||
		    op.type == PhysicalOperatorType::LIMIT || op.type == PhysicalOperatorType::STREAMING_LIMIT) {
			if (child.fragment.partition_count > 1) {
				child = Exchange(std::move(child), "gather", 1);
			}
		}
		if (op.type == PhysicalOperatorType::LIMIT) {
			auto &limit = op.Cast<PhysicalLimit>();
			auto &streaming = plan.Make<PhysicalStreamingLimit>(
			    op.types, std::move(limit.limit_val), std::move(limit.offset_val), op.estimated_cardinality, false);
			Wrap(streaming, child);
		} else {
			if (op.type == PhysicalOperatorType::STREAMING_LIMIT) {
				op.Cast<PhysicalStreamingLimit>().parallel = false;
			}
			Wrap(op, child);
		}
		return child;
	}

private:
	ClientContext &context;
	PhysicalPlan &plan;
	FragmentGraph &graph;
	idx_t partitions;
	idx_t next_exchange = 0;
	// Nodes have independent payloads, but join and scan references must share
	// an ID namespace. Each loaded fragment gets a fresh set of filter objects.
	DynamicTableFilterSerializationState filter_serialization;

	bool HasScopedState(const PhysicalOperator &op) {
		if (op.type == PhysicalOperatorType::DELIM_SCAN || op.type == PhysicalOperatorType::CTE_SCAN ||
		    op.type == PhysicalOperatorType::COLUMN_DATA_SCAN) {
			return true;
		}
		for (auto &child : op.GetChildren()) {
			if (HasScopedState(child.get())) {
				return true;
			}
		}
		return false;
	}

	DistributedSubtree Scoped(PhysicalOperator &root) {
		DistributedSubtree result;
		CorrelatedPartition partitioning;
		const bool partitioned = partitions > 1 && PlanCorrelatedPartition(root, partitioning);
		result.fragment.partition_count = partitioned ? partitions : 1;
		auto &node = result.fragment.root;
		node.types = root.types;
		node.scoped = true;
		PhysicalPlanInputBindings bindings;
		std::set<const PhysicalOperator *> visited;
		std::function<void(PhysicalOperator &)> visit = [&](PhysicalOperator &op) {
			if (!visited.insert(&op).second) {
				return;
			}
			CheckOperator(op, op.children.size(), true);
			// Keep delimiter caches and CTE consumers with their owner. Every
			// independent input still goes through the distributed planner and
			// explicit source assignments; no hidden scan is serialized here.
			auto partition_input = partitioning.inputs.find(&op);
			const bool external_input =
			    partitioned
			        ? partition_input != partitioning.inputs.end()
			        : !HasScopedState(op) && (op.type == PhysicalOperatorType::TABLE_SCAN || !op.children.empty());
			if (&op != &root && external_input) {
				vector<idx_t> keys;
				string distribution = "gather";
				if (partitioned) {
					distribution = partition_input->second.IsValid() ? "hash" : "broadcast";
					if (partition_input->second.IsValid()) {
						keys.push_back(partition_input->second.GetIndex());
					}
				}
				auto input = Exchange(Build(op), distribution, result.fragment.partition_count, keys);
				bindings.names.emplace(&op, input.fragment.root.input_port);
				node.children.push_back(std::move(input.fragment.root));
				for (auto &edge : input.incoming) {
					result.incoming.push_back(std::move(edge));
				}
				return;
			}
			if (op.type == PhysicalOperatorType::HASH_JOIN) {
				auto &join = op.Cast<PhysicalHashJoin>();
				join.filter_pushdown.reset();
				join.join_stats.clear();
			} else if (op.type == PhysicalOperatorType::PIECEWISE_MERGE_JOIN ||
			           op.type == PhysicalOperatorType::IE_JOIN) {
				static_cast<PhysicalRangeJoin &>(op).filter_pushdown.reset();
			} else if (op.type == PhysicalOperatorType::TOP_N) {
				op.Cast<PhysicalTopN>().dynamic_filter.reset();
			}
			vector<unique_ptr<Expression>> *aggregates = nullptr;
			if (op.type == PhysicalOperatorType::HASH_GROUP_BY) {
				aggregates = &op.Cast<PhysicalHashAggregate>().grouped_aggregate_data.aggregates;
			} else if (op.type == PhysicalOperatorType::PERFECT_HASH_GROUP_BY) {
				aggregates = &op.Cast<PhysicalPerfectHashAggregate>().aggregates;
			} else if (op.type == PhysicalOperatorType::UNGROUPED_AGGREGATE) {
				aggregates = &op.Cast<PhysicalUngroupedAggregate>().aggregates;
			}
			if (aggregates) {
				for (auto &aggregate : *aggregates) {
					WidenAggregate(aggregate->Cast<BoundAggregateExpression>());
				}
			}
			for (auto &child : op.GetChildren()) {
				visit(const_cast<PhysicalOperator &>(child.get()));
			}
		};
		visit(root);
		node.native_operator = Encode([&](Serializer &serializer) {
			serializer.GetSerializationData().SetCustom(bindings);
			serializer.GetSerializationData().SetCustom(filter_serialization);
			root.Serialize(serializer);
		});
		return result;
	}

	PlanNode Node(PhysicalOperator &op) {
		PlanNode node;
		node.types = op.types;
		node.native_operator = Encode([&](Serializer &serializer) {
			serializer.GetSerializationData().SetCustom(filter_serialization);
			op.SerializeNode(serializer);
		});
		return node;
	}
	void Wrap(PhysicalOperator &op, DistributedSubtree &child) {
		auto node = Node(op);
		node.children.push_back(std::move(child.fragment.root));
		child.fragment.root = std::move(node);
	}
	vector<unique_ptr<Expression>> Copy(const vector<unique_ptr<Expression>> &expressions) {
		vector<unique_ptr<Expression>> result;
		for (auto &expression : expressions) {
			result.push_back(expression->Copy());
		}
		return result;
	}
	vector<unique_ptr<Expression>> InputAggregates(const vector<unique_ptr<Expression>> &expressions,
	                                               const unordered_map<Expression *, size_t> *filters = nullptr) {
		vector<unique_ptr<Expression>> result;
		for (auto &expression : expressions) {
			auto &original = expression->Cast<BoundAggregateExpression>();
			// Physical planning wraps ordered aggregates and clears order_bys.
			// Restore the logical arguments/order keys before deciding whether a
			// partial result can be merged, just as native serialization does.
			auto aggregate = FunctionBinder::UnbindSortedAggregate(original);
			if (filters && original.filter) {
				aggregate->filter->Cast<BoundReferenceExpression>().index = filters->at(original.filter.get());
			}
			result.push_back(std::move(aggregate));
		}
		return result;
	}
	vector<idx_t> ProjectKeys(DistributedSubtree &child, const vector<unique_ptr<Expression>> &keys) {
		auto types = child.fragment.root.types;
		vector<unique_ptr<Expression>> expressions;
		vector<idx_t> columns;
		for (idx_t col = 0; col < types.size(); col++) {
			expressions.push_back(make_uniq<BoundReferenceExpression>(types[col], col));
		}
		for (auto &key : keys) {
			columns.push_back(types.size());
			types.push_back(key->return_type);
			expressions.push_back(key->Copy());
		}
		auto &projection = plan.Make<PhysicalProjection>(types, std::move(expressions), 0);
		Wrap(projection, child);
		return columns;
	}
	bool WidenAggregate(BoundAggregateExpression &aggregate) {
		// A sorted wrapper owns a different state and bind payload. Its order is
		// already fixed; never replace only its function while retaining that state.
		if (aggregate.function.HasStateDestructorCallback() || aggregate.children.size() != 1) {
			return false;
		}
		auto &name = aggregate.function.name;
		if (name == "avg" && aggregate.children[0]->return_type.id() == LogicalTypeId::INTERVAL) {
			aggregate.function = WideIntervalAvgFunction();
			return true;
		}
		if (aggregate.children[0]->return_type.InternalType() != PhysicalType::INT128) {
			return false;
		}
		if (name == "sum" || name == "sum_no_overflow") {
			aggregate.function = WideIntegerSumFunction(aggregate.children[0]->return_type);
			return true;
		}
		if (name == "avg") {
			aggregate.function = WideIntegerAvgFunction(aggregate.children[0]->return_type);
			return true;
		}
		return false;
	}
	PhysicalOperator &MakeAggregate(vector<unique_ptr<Expression>> groups, vector<unique_ptr<Expression>> aggregates) {
		vector<LogicalType> types;
		for (auto &group : groups) {
			types.push_back(group->return_type);
		}
		for (auto &aggregate : aggregates) {
			types.push_back(aggregate->return_type);
		}
		if (groups.empty()) {
			return plan.Make<PhysicalUngroupedAggregate>(types, std::move(aggregates), 1,
			                                             TupleDataValidityType::CAN_HAVE_NULL_VALUES);
		}
		return plan.Make<PhysicalHashAggregate>(context, types, std::move(aggregates), std::move(groups), 0);
	}
	DistributedSubtree Aggregate(PhysicalOperator &op, DistributedSubtree child,
	                             const vector<unique_ptr<Expression>> &groups,
	                             const vector<unique_ptr<Expression>> &aggregates) {
		bool split = !aggregates.empty();
		bool wide_aggregate = false;
		for (auto &expression : aggregates) {
			auto &aggregate = expression->Cast<BoundAggregateExpression>();
			auto name = aggregate.function.name;
			// Binder uses FIRST to retain an interval/collated group value, and
			// ARG_MIN/MAX for collated MIN/MAX. Ordered DISTINCT ON becomes
			// ARG_MIN_NULL over a native sort key, retaining NULL payloads.
			// Parsed SQL still admits only the public aggregate subset. Keep
			// these native rewrites as complete groups.
			const bool internal_rewrite =
			    name == "first" || name == "arg_min" || name == "arg_max" || name == "arg_min_null";
			if (name != "count" && name != "count_star" && name != "sum" && name != "sum_no_overflow" &&
			    name != "avg" && name != "min" && name != "max" && !internal_rewrite) {
				throw NotImplementedException("unsupported distributed aggregate %s", name);
			}
			split = split && !aggregate.IsDistinct() && !aggregate.order_bys && !internal_rewrite &&
			        aggregate.function.HasStateCombineCallback() &&
			        (!aggregate.function.HasBindCallback() || name == "sum_no_overflow") &&
			        !aggregate.function.HasStateDestructorCallback();
			// MIN/MAX(x, n) return the n extreme values as a list. Applying
			// unary MIN/MAX to partial lists compares whole lists, not elements.
			if ((name == "min" || name == "max") && aggregate.children.size() != 1) {
				split = false;
			}
			// Wide native state also handles whole groups arriving in a different
			// order. Finalization, rather than a partial subtotal, checks range.
			wide_aggregate = WidenAggregate(aggregate) || wide_aggregate;
		}
		if (child.fragment.partition_count == 1) {
			if (wide_aggregate) {
				auto &aggregate = MakeAggregate(Copy(groups), Copy(aggregates));
				Wrap(aggregate, child);
			} else {
				Wrap(op, child);
			}
			return child;
		}
		if (!split) {
			// These overloads need complete input for each group; repartition
			// original rows and let the native aggregate enforce SQL semantics.
			auto keys = ProjectKeys(child, groups);
			child =
			    Exchange(std::move(child), groups.empty() ? "gather" : "hash", groups.empty() ? 1 : partitions, keys);
			auto &final = MakeAggregate(Copy(groups), Copy(aggregates));
			Wrap(final, child);
			return child;
		}
		vector<unique_ptr<Expression>> partials, finals, output;
		vector<unique_ptr<Expression>> final_groups;
		vector<idx_t> keys;
		for (idx_t i = 0; i < groups.size(); i++) {
			keys.push_back(i);
			final_groups.push_back(make_uniq<BoundReferenceExpression>(groups[i]->return_type, i));
			output.push_back(make_uniq<BoundReferenceExpression>(groups[i]->return_type, i));
		}
		for (auto &expression : aggregates) {
			auto &aggregate = expression->Cast<BoundAggregateExpression>();
			auto partial =
			    ExportAggregateFunction::Bind(unique_ptr_cast<Expression, BoundAggregateExpression>(aggregate.Copy()));
			auto column = groups.size() + partials.size();
			finals.push_back(distributed::BindMergeAggregate(aggregate, partial->return_type, column));
			partials.push_back(std::move(partial));
			output.push_back(make_uniq<BoundReferenceExpression>(aggregate.return_type, column));
		}
		auto &partial = MakeAggregate(Copy(groups), std::move(partials));
		Wrap(partial, child);
		child = Exchange(std::move(child), groups.empty() ? "gather" : "hash", groups.empty() ? 1 : partitions, keys);
		auto &final = MakeAggregate(std::move(final_groups), std::move(finals));
		Wrap(final, child);
		auto &projection = plan.Make<PhysicalProjection>(op.types, std::move(output), op.estimated_cardinality);
		Wrap(projection, child);
		return child;
	}
	DistributedSubtree Join(PhysicalHashJoin &join) {
		auto left = Build(join.children[0]);
		auto right = Build(join.children[1]);
		vector<unique_ptr<Expression>> left_keys, right_keys;
		for (auto &condition : join.conditions) {
			if (condition.comparison == ExpressionType::COMPARE_EQUAL ||
			    condition.comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
				if (condition.left->return_type != condition.right->return_type) {
					throw NotImplementedException("hash join keys require identical types");
				}
				left_keys.push_back(condition.left->Copy());
				right_keys.push_back(condition.right->Copy());
			}
		}
		if (left_keys.empty() || !join.delim_types.empty()) {
			throw NotImplementedException("distributed hash join requires uncorrelated equality keys");
		}
		const bool broadcast = join.children[1].get().estimated_cardinality <= 32 &&
		                       (join.join_type == JoinType::INNER || join.join_type == JoinType::LEFT ||
		                        join.join_type == JoinType::SEMI || join.join_type == JoinType::ANTI);
		if (broadcast) {
			// Every probe partition sees the complete build input. Preserve its
			// existing distribution and fuse the probe subtree with the join;
			// shuffling it by the join key adds no correctness guarantee here.
			auto count = left.fragment.partition_count;
			right = Exchange(std::move(right), count == 1 ? "gather" : "broadcast", count);
		} else {
			auto lk = ProjectKeys(left, left_keys);
			left = Exchange(std::move(left), "hash", partitions, lk);
			auto rk = ProjectKeys(right, right_keys);
			right = Exchange(std::move(right), "hash", partitions, rk);
		}
		// Broadcast supplies the complete build input before the local probe
		// scan starts. An exchange clears local_filters: filters must not cross
		// a fragment boundary or use only one hash partition's build values.
		if (join.filter_pushdown) {
			auto &probes = join.filter_pushdown->probe_info;
			probes.erase(std::remove_if(probes.begin(), probes.end(),
			                            [&](const JoinFilterPushdownFilter &probe) {
				                            return !left.local_filters.count(probe.dynamic_filters.get());
			                            }),
			             probes.end());
			if (probes.empty()) {
				join.filter_pushdown.reset();
			}
		}
		join.join_stats.clear();
		auto node = Node(join);
		node.children.push_back(std::move(left.fragment.root));
		node.children.push_back(std::move(right.fragment.root));
		left.fragment.root = std::move(node);
		for (auto &edge : right.incoming) {
			left.incoming.push_back(std::move(edge));
		}
		return left;
	}
};

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
	struct FileScan {
		TableFunctionRef &ref;
		vector<OpenFileInfo> files;
	};
	vector<FileScan> scans;
	auto &fs = FileSystem::GetFileSystem(context);
	// Expand every scan before writing: the snapshot directory can be inside a
	// recursive input glob. Keep repeated references in their original order.
	VisitParsedQuery(
	    *select.node, [](ParsedExpression &) {},
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
		    vector<OpenFileInfo> files;
		    for (auto &pattern : patterns) {
			    if (context.IsInterrupted()) {
				    throw InterruptException();
			    }
			    if (!fs.IsPathAbsolute(pattern)) {
				    throw NotImplementedException("FTE snapshots require absolute local paths");
			    }
			    for (auto &file : fs.GlobFiles(pattern)) {
				    if (files.size() >= 4096) {
					    throw InvalidInputException("FTE source file reference count exceeds limit");
				    }
				    files.push_back(std::move(file));
			    }
		    }
		    scans.push_back({ref.Cast<TableFunctionRef>(), std::move(files)});
	    });
	for (auto &scan : scans) {
		vector<Value> paths;
		vector<OpenFileInfo> files;
		for (auto &file : scan.files) {
			if (context.IsInterrupted()) {
				throw InterruptException();
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
		auto &function = scan.ref.function->Cast<FunctionExpression>();
		function.children[0] = make_uniq<ConstantExpression>(Value::LIST(LogicalType::VARCHAR, std::move(paths)));
		MultiFileReader::SetFileList(scan.ref, std::move(files));
	}
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
			file.extended_info->options["vane_snapshot_identity"] = Value(found->second.identity);
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
		VisitParsedQuery(
		    *select.node, [&](ParsedExpression &expression) { CheckParsedExpression(context, expression); },
		    [](TableRef &ref) {
			    if (ref.type != TableReferenceType::EMPTY_FROM && ref.type != TableReferenceType::TABLE_FUNCTION &&
			        ref.type != TableReferenceType::SUBQUERY && ref.type != TableReferenceType::EXPRESSION_LIST &&
			        ref.type != TableReferenceType::JOIN && ref.type != TableReferenceType::BASE_TABLE) {
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
		planner.binder->SetScalarFunctionCallback(
		    [](const ScalarFunction &function) { CheckScalarFunction(function, true); });
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
		AnalyticalPlanner distributed(context, *physical, graph, partitions);
		auto root = distributed.Build(physical->Root());
		if (!hash_columns.empty()) {
			root = distributed.Exchange(std::move(root), "hash", partitions, hash_columns);
		}
		if (root.fragment.partition_count != 1) {
			root = distributed.Exchange(std::move(root), "gather", 1);
		}
		root.fragment.source_dependencies = std::move(validator.source_dependencies);
		distributed.Finish(std::move(root), planner.names);
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
                           std::set<string> &used_inputs, DynamicTableFilterSerializationState &filters) {
	if (!node.input_port.empty()) {
		if (!inputs || !used_inputs.insert(node.input_port).second) {
			throw InvalidInputException("fragment input %s must be bound exactly once", node.input_port);
		}
		auto input = inputs(plan, node.input_port, node.types);
		if (!input || input->types != node.types) {
			throw InvalidInputException("fragment input %s schema mismatch", node.input_port);
		}
		auto &result = *input;
		plan.TakeOwnership(std::move(input));
		return result;
	}
	unique_ptr<PhysicalOperator> op;
	bound_parameter_map_t parameters;
	PhysicalPlanInputBindings bindings;
	std::set<const PhysicalOperator *> bound_inputs;
	std::set<string> scoped_ports;
	if (node.scoped) {
		bindings.bind = [&](PhysicalPlan &owner, const string &port,
		                    const vector<LogicalType> &types) -> unique_ptr<PhysicalOperator> {
			auto declared = std::find_if(node.children.begin(), node.children.end(),
			                             [&](const PlanNode &child) { return child.input_port == port; });
			if (!inputs || declared == node.children.end() || declared->types != types ||
			    !used_inputs.insert(port).second) {
				throw SerializationException("scoped fragment input %s has no unique matching declaration", port);
			}
			auto input = inputs(owner, port, types);
			if (input) {
				bound_inputs.insert(input.get());
			}
			scoped_ports.insert(port);
			return input;
		};
	}
	Decode(node.native_operator, [&](BinaryDeserializer &deserializer) {
		deserializer.Set<ClientContext &>(context);
		deserializer.Set<DatabaseInstance &>(DatabaseInstance::GetDatabase(context));
		deserializer.Set<bound_parameter_map_t &>(parameters);
		deserializer.GetSerializationData().SetCustom(filters);
		if (node.scoped) {
			deserializer.GetSerializationData().SetCustom(bindings);
		}
		op = PhysicalOperator::Deserialize(deserializer, plan);
	});
	if (node.scoped) {
		if (op->types != node.types || scoped_ports.size() != node.children.size()) {
			throw SerializationException("scoped fragment has mismatched output or unused inputs");
		}
		std::set<const PhysicalOperator *> checked;
		std::function<void(const PhysicalOperator &)> check = [&](const PhysicalOperator &entry) {
			if (bound_inputs.count(&entry) || !checked.insert(&entry).second) {
				return;
			}
			if (entry.type == PhysicalOperatorType::TABLE_SCAN) {
				throw SerializationException("scoped fragment contains an implicit scan without source assignments");
			}
			CheckOperator(const_cast<PhysicalOperator &>(entry), entry.children.size(), true);
			if (entry.type == PhysicalOperatorType::CTE_SCAN && !entry.Cast<PhysicalColumnDataScan>().collection) {
				throw SerializationException("scoped fragment CTE scan has no owning materialization");
			}
			for (auto &child : entry.GetChildren()) {
				check(child.get());
			}
		};
		check(*op);
		auto &result = *op;
		plan.TakeOwnership(std::move(op));
		return result;
	}
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
		op->children.push_back(
		    LoadNode(context, plan, child, inputs, sources, assignments, used_sources, used_inputs, filters));
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
	DynamicTableFilterSerializationState filters;
	auto plan = make_uniq<PhysicalPlan>(Allocator::Get(context));
	context.RunFunctionInTransaction([&]() {
		plan->SetRoot(
		    LoadNode(context, *plan, fragment.root, inputs, sources, assignments, used_sources, used_inputs, filters));
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
	                                     "scalar_subquery_error_on_multiple_rows",
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

string FileStamp(const OpenFileInfo &file, const string &identity) {
	return Encode([&](Serializer &serializer) {
		serializer.WriteProperty(1, "path", file.path);
		serializer.WriteProperty(2, "identity", identity);
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
				    !file.extended_info->options.count("vane_snapshot_bytes") ||
				    !file.extended_info->options.count("vane_snapshot_identity")) {
					throw SerializationException("frozen source has no content identity");
				}
				auto &options = file.extended_info->options;
				auto identity = FileIdentity(context, file);
				if (identity != options.at("vane_snapshot_identity").GetValue<string>()) {
					throw InvalidInputException("immutable FTE source snapshot changed");
				}
				if (stamped_files.insert(file.path).second) {
					versions.push_back(FileStamp(file, identity) +
					                   options.at("vane_snapshot_sha256").GetValue<string>());
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
					versions.push_back(FileStamp(file, FileIdentity(context, file)));
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

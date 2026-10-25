// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "fragment_plan.hpp"
#include "correlated_partition.hpp"
#include "parquet_mask.hpp"
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
#include "duckdb/execution/operator/set/physical_cte.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/parser/expression/subquery_expression.hpp"
#include "duckdb/execution/operator/helper/physical_limit.hpp"
#include "duckdb/execution/operator/helper/physical_streaming_limit.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/order/physical_top_n.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/function/lambda_functions.hpp"
#include "duckdb/function/aggregate/wide_integer.hpp"
#include "duckdb/function/aggregate/wide_interval.hpp"
#include "duckdb/function/scalar/generic_common.hpp"
#include "duckdb/function/scalar/compressed_materialization_utils.hpp"
#include "duckdb/common/types/hugeint.hpp"
#include "duckdb/execution/distributed/pipeline_node/aggregate.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"

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
#include "duckdb/common/vector_operations/vector_operations.hpp"
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
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
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
	if (!split_partitions.empty()) {
		serializer.WriteProperty(7, "split_partitions", split_partitions);
	}
}

SourceSpec SourceSpec::Deserialize(Deserializer &deserializer) {
	SourceSpec result;
	result.source_id = deserializer.ReadProperty<string>(1, "source_id");
	result.function_name = deserializer.ReadProperty<string>(2, "function");
	result.capability = deserializer.ReadProperty<string>(3, "capability");
	result.codec = deserializer.ReadProperty<string>(4, "codec");
	result.splits = deserializer.ReadProperty<vector<DistributedScanSplit>>(5, "splits");
	result.requires_snapshot = deserializer.ReadProperty<bool>(6, "requires_snapshot");
	result.split_partitions = deserializer.ReadPropertyWithDefault<vector<idx_t>>(7, "split_partitions");
	if (result.source_id.empty() || result.capability.empty() || result.codec.empty()) {
		throw SerializationException("empty fragment source identity");
	}
	if (!result.split_partitions.empty() &&
	    (result.split_partitions.size() != result.splits.size() ||
	     (result.function_name != "read_parquet" && result.function_name != "parquet_scan"))) {
		throw SerializationException("split partition assignment must cover every source split");
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
	if (parquet_mask_words) {
		serializer.WriteProperty(7, "parquet_mask_words", parquet_mask_words);
		serializer.WriteProperty(8, "parquet_mask_key", parquet_mask_key);
	}
}

PlanNode PlanNode::Deserialize(Deserializer &deserializer) {
	PlanNode result;
	result.input_port = deserializer.ReadProperty<string>(1, "input_port");
	result.types = deserializer.ReadProperty<vector<LogicalType>>(2, "types");
	result.native_operator = deserializer.ReadProperty<string>(3, "operator");
	result.source_id = deserializer.ReadProperty<string>(4, "source_id");
	result.children = deserializer.ReadProperty<vector<PlanNode>>(5, "children");
	result.scoped = deserializer.ReadProperty<bool>(6, "scoped");
	result.parquet_mask_words = deserializer.ReadPropertyWithDefault<idx_t>(7, "parquet_mask_words");
	result.parquet_mask_key = deserializer.ReadPropertyWithDefault<idx_t>(8, "parquet_mask_key");
	CheckTypes(result.types);
	if (result.parquet_mask_words) {
		if (!ValidParquetMaskSize(result.parquet_mask_words) || !result.input_port.empty() ||
		    !result.native_operator.empty() || !result.source_id.empty() || result.scoped ||
		    result.children.size() != 2 || result.children[0].types != result.types ||
		    result.children[1].types != vector<LogicalType> {LogicalType::UBIGINT, LogicalType::UBIGINT} ||
		    result.parquet_mask_key >= result.types.size() || !result.types[result.parquet_mask_key].IsIntegral()) {
			throw SerializationException("invalid Parquet mask fragment node");
		}
		return result;
	}
	if (result.parquet_mask_key) {
		throw SerializationException("Parquet mask key has no bounded mask");
	}
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
	for (auto &source : result.sources) {
		for (auto partition : source.split_partitions) {
			if (partition >= result.partition_count) {
				throw SerializationException("source split partition is out of range");
			}
		}
	}
	for (auto &source : result.source_dependencies) {
		if (!source.split_partitions.empty()) {
			throw SerializationException("source dependency cannot declare partition assignments");
		}
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
	explicit LogicalValidator(idx_t partitions) : partitions(partitions) {
	}

	idx_t partitions;
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
				if (partitions > 1) {
					// Cost the remaining files when the optimizer requests statistics,
					// after file pruning, rather than opening every bound footer here.
					get.function.cardinality = [](ClientContext &context, const FunctionData *data) {
						return make_uniq<NodeStatistics>(
						    ParquetScanCardinality(context, data->Cast<MultiFileBindData>()));
					};
				}
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
	unique_ptr<FunctionData> coordinator_bind;
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
			coordinator_bind = std::move(scan.bind_data);
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
			auto worker_bind = callbacks.create_worker_bind(input);
			coordinator_bind = std::move(scan.bind_data);
			scan.bind_data = std::move(worker_bind);
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
	if (coordinator_bind) {
		// A correlated key filter can compile this scan again. Preserve its
		// coordinator files/splits; only the serialized bind is worker-only.
		op.Cast<PhysicalTableScan>().bind_data = std::move(coordinator_bind);
	}
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
	// Exchanges clear this lineage. A broadcast join preserves the probe scan
	// and its exact split assignments, including when it duplicates probe rows.
	optional_ptr<const PhysicalOperator> local_scan;
	// Ordered columns with the exact native representation used by HASH.
	// An empty vector means that the output distribution is not proven.
	vector<idx_t> hash_keys;
};

double EstimatedExchangeBytes(const PhysicalOperator &op) {
	double width = 0;
	for (auto &type : op.types) {
		auto physical = type.InternalType();
		if (physical == PhysicalType::LIST || physical == PhysicalType::STRUCT || physical == PhysicalType::ARRAY) {
			// No length statistics are available for nested values here. Keep
			// these inputs on the small-row fallback rather than guessing.
			return NumericLimits<double>::Maximum();
		}
		// Include validity and a conservative string payload estimate. This is
		// a planning estimate; execution still enforces the operator budget.
		width += 1 + (physical == PhysicalType::VARCHAR ? 64 : GetTypeIdSize(physical));
	}
	return double(op.estimated_cardinality) * width;
}

bool BroadcastJoin(const PhysicalHashJoin &join, idx_t probe_partitions, idx_t broadcast_limit) {
	if (join.join_type != JoinType::INNER && join.join_type != JoinType::LEFT && join.join_type != JoinType::SEMI &&
	    join.join_type != JoinType::ANTI && join.join_type != JoinType::MARK) {
		return false;
	}
	auto &build = join.children[1].get();
	if (build.estimated_cardinality <= 32) {
		return true;
	}
	const auto build_bytes = EstimatedExchangeBytes(build);
	// Broadcasting sends N copies of the build instead of shuffling both
	// sides once. Bound each replica and require its extra traffic to fit in
	// the probe traffic saved. Fusing the probe also preserves scan filters.
	const auto probe_bytes = EstimatedExchangeBytes(join.children[0]);
	const auto extra_bytes = build_bytes * double(probe_partitions - 1);
	if (build_bytes <= 8 * 1024 * 1024) {
		return extra_bytes <= probe_bytes;
	}
	// Larger replicas need declared worker memory and a wider traffic margin.
	// Unknown nested widths cannot justify this additional replication.
	return build_bytes <= broadcast_limit && probe_bytes != NumericLimits<double>::Maximum() &&
	       extra_bytes * 4 < probe_bytes;
}

class AnalyticalPlanner {
public:
	AnalyticalPlanner(ClientContext &context, PhysicalPlan &plan, FragmentGraph &graph, idx_t partitions,
	                  idx_t worker_operator_bytes)
	    : context(context), plan(plan), graph(graph), partitions(partitions),
	      broadcast_limit(
	          MaxValue<idx_t>(8 * 1024 * 1024, MinValue<idx_t>(64 * 1024 * 1024, worker_operator_bytes / 16))) {
		// Cost 128 bytes per word for grouping, probe tables and encoded masks.
		// This bounds domain rows, not total allocation: native operator memory
		// and spill budgets still apply, including thread-local aggregate states.
		const auto word_capacity = broadcast_limit / 128;
		while (max_bitmap_words <= word_capacity / 2) {
			max_bitmap_words *= 2;
		}
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
		output_partitioning.emplace(id, std::make_pair(graph.fragments.size(), std::move(subtree.hash_keys)));
		graph.fragments.push_back(std::move(fragment));
		return id;
	}

	DistributedSubtree Exchange(DistributedSubtree child, const string &distribution, idx_t count,
	                            const vector<idx_t> &keys = {}) {
		auto types = child.fragment.root.types;
		auto producer = Finish(std::move(child));
		return ExchangeFrom(producer, std::move(types), distribution, count, keys);
	}

	DistributedSubtree ExchangeFrom(const string &producer, vector<LogicalType> types, const string &distribution,
	                                idx_t count, const vector<idx_t> &keys = {}) {
		ExchangeSpec edge;
		edge.exchange_id = "exchange" + std::to_string(next_exchange++);
		edge.consumer_port = edge.exchange_id;
		edge.distribution = distribution;
		if (distribution == "hash") {
			auto found = output_partitioning.find(producer);
			if (found != output_partitioning.end() && !keys.empty() && found->second.second == keys &&
			    graph.fragments[found->second.first].partition_count == count &&
			    graph.fragments[found->second.first].root.types == types) {
				edge.distribution = "passthrough";
			} else {
				edge.partitioning = HashExpression(types, keys);
			}
		}
		edge.producer = producer;
		DistributedSubtree result;
		result.fragment.partition_count = count;
		result.fragment.root.input_port = edge.consumer_port;
		result.fragment.root.types = std::move(types);
		if (distribution == "hash") {
			result.hash_keys = keys;
		}
		result.incoming.push_back(std::move(edge));
		return result;
	}

	DistributedSubtree Build(PhysicalOperator &op) {
		auto result = BuildOperator(op);
		auto found = probe_filters.find(&op);
		if (found != probe_filters.end()) {
			for (auto &filter : found->second) {
				auto count = result.fragment.partition_count;
				auto domain = ExchangeFrom(filter.domain_producer, filter.domain->types,
				                           count == 1 ? "gather" : "broadcast", count);
				result = SemiFilter(op, std::move(result), filter.column, *filter.domain, std::move(domain),
				                    filter.keys_only ? 0 : filter.build_column, filter.bitmap_words);
			}
		}
		return result;
	}

	DistributedSubtree BuildOperator(PhysicalOperator &op) {
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
				result.local_scan = &op;
				auto &scan = op.Cast<PhysicalTableScan>();
				if (scan.dynamic_filters) {
					result.local_filters.insert(scan.dynamic_filters.get());
				}
			}
			return result;
		}
		if (op.type == PhysicalOperatorType::HASH_JOIN) {
			auto &join = op.Cast<PhysicalHashJoin>();
			// A complete broadcast gives each MARK probe partition the global
			// NULL/empty-build knowledge. Require the base broadcast budget so
			// preserving an early probe filter cannot later select a hash join.
			// SINGLE must see all matches before checking scalar cardinality.
			// Delimiter elimination can leave correlated metadata on an ordinary
			// join, including when its outer input is empty. Keep its native scope.
			if (join.join_type == JoinType::SINGLE || !join.delim_types.empty() ||
			    (join.join_type == JoinType::MARK && !BroadcastJoin(join, partitions, 8 * 1024 * 1024))) {
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
	idx_t broadcast_limit;
	idx_t max_bitmap_words = 65536;
	idx_t next_exchange = 0;
	unordered_map<string, std::pair<idx_t, vector<idx_t>>> output_partitioning;
	// Nodes have independent payloads, but join and scan references must share
	// an ID namespace. Each loaded fragment gets a fresh set of filter objects.
	DynamicTableFilterSerializationState filter_serialization;
	struct EarlyProbeFilter {
		PhysicalOperator *input = nullptr;
		idx_t column = 0;
		PhysicalOperator *build = nullptr;
		idx_t build_column = 0;
		string producer;
		PhysicalOperator *domain = nullptr;
		string domain_producer;
		bool keys_only = false;
		idx_t bitmap_words = 0;
	};
	unordered_map<const PhysicalOperator *, vector<EarlyProbeFilter>> probe_filters;
	unordered_map<const PhysicalOperator *, string> shared_join_builds;

	bool HasFilteredAggregate(const PhysicalOperator &op) {
		if (op.type == PhysicalOperatorType::FILTER) {
			auto child = &op.children[0].get();
			while (child->type == PhysicalOperatorType::PROJECTION) {
				child = &child->children[0].get();
			}
			if (child->type == PhysicalOperatorType::HASH_GROUP_BY ||
			    child->type == PhysicalOperatorType::PERFECT_HASH_GROUP_BY) {
				return true;
			}
		}
		if (op.type != PhysicalOperatorType::PROJECTION && op.type != PhysicalOperatorType::FILTER &&
		    op.type != PhysicalOperatorType::HASH_JOIN) {
			return false;
		}
		for (auto &child : op.children) {
			if (HasFilteredAggregate(child.get())) {
				return true;
			}
		}
		return false;
	}

	bool HasFilteredBuildRoot(const PhysicalOperator &root) {
		auto op = &root;
		while (op->type == PhysicalOperatorType::PROJECTION) {
			op = &op->children[0].get();
		}
		if (op->type == PhysicalOperatorType::FILTER) {
			return true;
		}
		if (op->type == PhysicalOperatorType::TABLE_SCAN) {
			auto &scan = op->Cast<PhysicalTableScan>();
			return scan.table_filters && !scan.table_filters->filters.empty();
		}
		return false;
	}

	EarlyProbeFilter PlanBuildFilter(PhysicalHashJoin &join) {
		EarlyProbeFilter result;
		auto &target = join.children[1].get();
		auto target_bytes = EstimatedExchangeBytes(target);
		// Keep the two consumers in independent hash-join inputs. A fused
		// broadcast parent could make its probe-side seed wait for this build.
		// The probe can gather below an earlier join, so exclude fusion even
		// with one probe partition rather than costing only the requested count.
		if (partitions == 1 || join.join_type != JoinType::INNER || !join.delim_types.empty() ||
		    BroadcastJoin(join, 1, broadcast_limit) || probe_filters.count(&target) ||
		    target_bytes == NumericLimits<double>::Maximum()) {
			return result;
		}
		for (auto &condition : join.conditions) {
			if ((condition.comparison != ExpressionType::COMPARE_EQUAL &&
			     condition.comparison != ExpressionType::COMPARE_NOT_DISTINCT_FROM) ||
			    condition.left->GetExpressionClass() != ExpressionClass::BOUND_REF ||
			    condition.right->GetExpressionClass() != ExpressionClass::BOUND_REF ||
			    !condition.left->return_type.IsIntegral() ||
			    condition.left->return_type != condition.right->return_type) {
				continue;
			}
			auto try_build = [&](PhysicalHashJoin &probe, idx_t column) {
				auto &build = probe.children[1].get();
				if (column >= build.types.size() || build.types[column] != condition.left->return_type ||
				    !HasFilteredBuildRoot(build) || HasScopedState(build) || !build.estimated_cardinality ||
				    EstimatedExchangeBytes(build) > 8 * 1024 * 1024 ||
				    !BroadcastJoin(probe, partitions, 8 * 1024 * 1024) ||
				    double(build.estimated_cardinality) * 8 >= double(target.estimated_cardinality) ||
				    EstimatedExchangeBytes(build) * partitions >= target_bytes) {
					return false;
				}
				result.input = &target;
				result.column = condition.right->Cast<BoundReferenceExpression>().index;
				result.build = &build;
				result.build_column = column;
				return true;
			};
			auto column = condition.left->Cast<BoundReferenceExpression>().index;
			auto op = &join.children[0].get();
			while (column < op->types.size() && op->types[column] == condition.left->return_type) {
				if (op->type == PhysicalOperatorType::PROJECTION) {
					auto &expression = *op->Cast<PhysicalProjection>().select_list[column];
					if (expression.GetExpressionClass() != ExpressionClass::BOUND_REF) {
						break;
					}
					column = expression.Cast<BoundReferenceExpression>().index;
					op = &op->children[0].get();
				} else if (op->type == PhysicalOperatorType::FILTER) {
					op = &op->children[0].get();
				} else if (op->type == PhysicalOperatorType::HASH_JOIN) {
					auto &probe = op->Cast<PhysicalHashJoin>();
					if (probe.join_type != JoinType::INNER || !probe.delim_types.empty()) {
						break;
					}
					if (column < probe.lhs_output_columns.col_idxs.size()) {
						column = probe.lhs_output_columns.col_idxs[column];
						// The optimizer can retain either copy of an equality key.
						// INNER matches prove that the surviving probe key belongs
						// to the complete build domain even when that copy is omitted.
						for (auto &key : probe.conditions) {
							if ((key.comparison == ExpressionType::COMPARE_EQUAL ||
							     key.comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) &&
							    key.left->GetExpressionClass() == ExpressionClass::BOUND_REF &&
							    key.right->GetExpressionClass() == ExpressionClass::BOUND_REF &&
							    key.left->return_type == condition.left->return_type &&
							    key.right->return_type == condition.left->return_type &&
							    key.left->Cast<BoundReferenceExpression>().index == column &&
							    try_build(probe, key.right->Cast<BoundReferenceExpression>().index)) {
								return result;
							}
						}
						op = &probe.children[0].get();
						continue;
					}
					auto index = probe.rhs_output_columns.col_idxs[column - probe.lhs_output_columns.col_idxs.size()];
					if (index < probe.conditions.size()) {
						auto &key = *probe.conditions[index].right;
						if (key.GetExpressionClass() != ExpressionClass::BOUND_REF) {
							break;
						}
						column = key.Cast<BoundReferenceExpression>().index;
					} else {
						column = probe.payload_columns.col_idxs[index - probe.conditions.size()];
					}
					// These output values are a subset of this complete small build.
					// Reuse it as a necessary-condition filter for the other input;
					// the original multi-key/residual join still decides matches.
					if (try_build(probe, column)) {
						return result;
					}
					op = &probe.children[1].get();
				} else {
					break;
				}
			}
		}
		return result;
	}

	EarlyProbeFilter PlanProbeFilter(PhysicalHashJoin &join) {
		EarlyProbeFilter result;
		auto &build = join.children[1].get();
		if (partitions == 1 || (join.join_type != JoinType::INNER && join.join_type != JoinType::SEMI) ||
		    !build.estimated_cardinality) {
			return result;
		}
		// Native scan pushdown cannot cross an integer compression projection.
		// A filter at the exchange boundary still sees the exact join-key type;
		// recognize a filtered build even when no native scan target was found.
		if ((!join.filter_pushdown || !join.filter_pushdown->build_side_has_filter) && !HasFilteredBuildRoot(build)) {
			return result;
		}
		// Find existing early filters before considering larger broadcasts. Their
		// savings can invalidate an ancestor or descendant join's traffic estimate.
		const bool broadcast = BroadcastJoin(join, partitions, 8 * 1024 * 1024);
		if (broadcast && EstimatedExchangeBytes(build) > 8 * 1024 * 1024) {
			return result;
		}
		if (!broadcast) {
			// Reuse the original producer directly for its hash join. Computed
			// or repeated build keys still use the ordinary projection path.
			std::set<idx_t> keys;
			for (auto &condition : join.conditions) {
				if (condition.comparison != ExpressionType::COMPARE_EQUAL &&
				    condition.comparison != ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
					continue;
				}
				if (condition.right->GetExpressionClass() != ExpressionClass::BOUND_REF ||
				    !keys.insert(condition.right->Cast<BoundReferenceExpression>().index).second) {
					return result;
				}
			}
		}
		for (auto &condition : join.conditions) {
			if ((condition.comparison != ExpressionType::COMPARE_EQUAL &&
			     condition.comparison != ExpressionType::COMPARE_NOT_DISTINCT_FROM) ||
			    condition.left->GetExpressionClass() != ExpressionClass::BOUND_REF ||
			    condition.right->GetExpressionClass() != ExpressionClass::BOUND_REF ||
			    !condition.left->return_type.IsIntegral() ||
			    condition.left->return_type != condition.right->return_type) {
				continue;
			}
			// Move only a necessary condition through unchanged projections,
			// filters and the probe side of inner/semi joins. Stop at cardinality
			// boundaries, computed keys, preserved outer rows and shared scopes.
			EarlyProbeFilter target;
			auto column = condition.left->Cast<BoundReferenceExpression>().index;
			auto op = &join.children[0].get();
			if (!broadcast) {
				target.input = op;
				target.column = column;
			}
			while (column < op->types.size() && op->types[column] == condition.left->return_type) {
				if (op->type == PhysicalOperatorType::PROJECTION) {
					auto &expression = *op->Cast<PhysicalProjection>().select_list[column];
					if (expression.GetExpressionClass() != ExpressionClass::BOUND_REF) {
						break;
					}
					column = expression.Cast<BoundReferenceExpression>().index;
					op = &op->children[0].get();
				} else if (op->type == PhysicalOperatorType::FILTER) {
					op = &op->children[0].get();
				} else if (op->type == PhysicalOperatorType::HASH_JOIN) {
					auto &probe = op->Cast<PhysicalHashJoin>();
					if ((probe.join_type != JoinType::INNER && probe.join_type != JoinType::SEMI) ||
					    !probe.delim_types.empty() || column >= probe.lhs_output_columns.col_idxs.size()) {
						break;
					}
					column = probe.lhs_output_columns.col_idxs[column];
					op = &probe.children[0].get();
					if (!BroadcastJoin(probe, partitions, 8 * 1024 * 1024) && column < op->types.size() &&
					    op->types[column] == condition.left->return_type) {
						target.input = op;
						target.column = column;
					}
				} else {
					break;
				}
			}
			if (!target.input) {
				continue;
			}
			auto key_bytes =
			    double(build.estimated_cardinality) * (1 + GetTypeIdSize(condition.right->return_type.InternalType()));
			auto probe_bytes = EstimatedExchangeBytes(*target.input);
			target.keys_only = !broadcast;
			if (!broadcast && key_bytes > 8 * 1024 * 1024 &&
			    (HasFilteredAggregate(build) || key_bytes > MaxValue<idx_t>(16 * 1024 * 1024, broadcast_limit))) {
				// An ancestor's early filter already reduces this exchange. Its
				// original cardinality cannot fund another blocking domain.
				if (probe_filters.count(target.input)) {
					continue;
				}
				// A filtered build can be overestimated. Bound the domain by its
				// word count instead of broadcasting every key. Each word retains
				// 64 hash positions; collisions only retain extra probes.
				target.bitmap_words = 65536;
				if (target.input->estimated_cardinality <= 8 * target.bitmap_words) {
					continue;
				}
				// Size each filter to its own probe and traffic savings. A larger
				// worker budget must not disable a useful smaller key domain.
				while (target.bitmap_words < max_bitmap_words &&
				       16 * target.bitmap_words < target.input->estimated_cardinality &&
				       EstimatedExchangeBytes(build) + double(target.bitmap_words) * 18 * partitions < probe_bytes) {
					target.bitmap_words *= 2;
				}
				key_bytes = target.bitmap_words * 18;
			} else {
				// Key replicas can use the same bounded per-query allowance as a
				// full-build broadcast. Charge all extra edges below as before.
				if (!broadcast && key_bytes > MaxValue<idx_t>(16 * 1024 * 1024, broadcast_limit)) {
					continue;
				}
				if (double(build.estimated_cardinality) * (broadcast ? 8 : 4) >=
				    double(target.input->estimated_cardinality)) {
					// A single-key join's output also estimates the probes retained
					// by its key domain. This does not hold for just one condition of
					// a multi-key/residual join or a filter moved across another join.
					if (broadcast || join.conditions.size() != 1 || target.input != &join.children[0].get() ||
					    !join.estimated_cardinality ||
					    double(join.estimated_cardinality) * 4 >= double(target.input->estimated_cardinality)) {
						continue;
					}
					// Charge the extra build/key edges only against traffic estimated
					// to disappear, rather than the entire unfiltered probe exchange.
					probe_bytes *= 1 - double(join.estimated_cardinality) / double(target.input->estimated_cardinality);
				}
			}
			// Key-only replicas carry no payload. Allow a larger estimate and
			// fourfold selectivity, while still charging every extra build/key
			// edge below. Complete-build broadcasts keep their smaller limits.
			auto traffic = broadcast ? EstimatedExchangeBytes(build) * partitions
			                         : EstimatedExchangeBytes(build) + key_bytes * partitions;
			// A streaming build adds a new complete-build barrier and bitmap
			// hashing work. Require the same fourfold margin as full broadcasts.
			// A filtered aggregate already blocks before producing its keys.
			const auto margin = target.bitmap_words && !HasFilteredAggregate(build) ? 4 : 1;
			if (traffic * margin < probe_bytes) {
				target.build = &build;
				target.build_column = condition.right->Cast<BoundReferenceExpression>().index;
				return target;
			}
		}
		return result;
	}

	unique_ptr<Expression> FilterFunction(const string &name, vector<unique_ptr<Expression>> arguments) {
		FunctionBinder binder(context);
		ErrorData error;
		auto result = binder.BindScalarFunction(DEFAULT_SCHEMA, name, std::move(arguments), error);
		if (error.HasError()) {
			throw InternalException("cannot bind join filter %s: %s", name, error.Message());
		}
		return result;
	}
	unique_ptr<Expression> FilterHash(const LogicalType &type, idx_t column) {
		vector<unique_ptr<Expression>> arguments;
		arguments.push_back(make_uniq<BoundReferenceExpression>(type, column));
		return FilterFunction("hash", std::move(arguments));
	}
	unique_ptr<Expression> FilterWord(idx_t column, idx_t words) {
		vector<unique_ptr<Expression>> arguments;
		arguments.push_back(make_uniq<BoundReferenceExpression>(LogicalType::UBIGINT, column));
		arguments.push_back(make_uniq<BoundConstantExpression>(Value::UBIGINT(words - 1)));
		return FilterFunction("&", std::move(arguments));
	}
	unique_ptr<Expression> FilterBit(idx_t column) {
		vector<unique_ptr<Expression>> arguments;
		arguments.push_back(make_uniq<BoundReferenceExpression>(LogicalType::UBIGINT, column));
		arguments.push_back(make_uniq<BoundConstantExpression>(Value::UBIGINT(32)));
		auto upper = FilterFunction(">>", std::move(arguments));
		arguments.clear();
		arguments.push_back(std::move(upper));
		arguments.push_back(make_uniq<BoundConstantExpression>(Value::UBIGINT(63)));
		auto bit = FilterFunction("&", std::move(arguments));
		arguments.clear();
		arguments.push_back(make_uniq<BoundConstantExpression>(Value::UBIGINT(1)));
		arguments.push_back(std::move(bit));
		return FilterFunction("<<", std::move(arguments));
	}
	vector<unique_ptr<Expression>> FilterMaskAggregate() {
		auto &entry =
		    Catalog::GetEntry<AggregateFunctionCatalogEntry>(context, SYSTEM_CATALOG, DEFAULT_SCHEMA, "bit_or");
		FunctionBinder binder(context);
		ErrorData error;
		vector<unique_ptr<Expression>> arguments;
		arguments.push_back(make_uniq<BoundReferenceExpression>(LogicalType::UBIGINT, 1));
		auto function = binder.BindFunction("bit_or", entry.functions, arguments, error);
		if (error.HasError()) {
			throw InternalException("cannot bind join filter mask aggregate: %s", error.Message());
		}
		vector<unique_ptr<Expression>> aggregates;
		aggregates.push_back(binder.BindAggregateFunction(entry.functions.GetFunctionByOffset(function.GetIndex()),
		                                                  std::move(arguments)));
		return aggregates;
	}
	std::pair<DistributedSubtree, PhysicalOperator *> BitmapDomain(DistributedSubtree domain, PhysicalOperator &build,
	                                                               idx_t key, idx_t words) {
		vector<unique_ptr<Expression>> expressions;
		expressions.push_back(FilterHash(build.types[key], key));
		auto &hash = plan.Make<PhysicalProjection>(vector<LogicalType> {LogicalType::UBIGINT}, std::move(expressions),
		                                           build.estimated_cardinality);
		Wrap(hash, domain);
		expressions.clear();
		expressions.push_back(FilterWord(0, words));
		expressions.push_back(FilterBit(0));
		auto &projection =
		    plan.Make<PhysicalProjection>(vector<LogicalType> {LogicalType::UBIGINT, LogicalType::UBIGINT},
		                                  std::move(expressions), build.estimated_cardinality);
		Wrap(projection, domain);
		vector<unique_ptr<Expression>> groups;
		groups.push_back(make_uniq<BoundReferenceExpression>(LogicalType::UBIGINT, 0));
		auto &partial = MakeAggregate(Copy(groups), FilterMaskAggregate());
		Wrap(partial, domain);
		domain = Exchange(std::move(domain), "gather", 1);
		// One globally merged row per word prevents the filtering join below
		// from multiplying probe rows or missing bits from another partition.
		auto &complete = MakeAggregate(std::move(groups), FilterMaskAggregate());
		Wrap(complete, domain);
		return std::make_pair(std::move(domain), &complete);
	}

	void BuildKeyDomain(EarlyProbeFilter &filter) {
		auto &build = *filter.build;
		auto domain = ExchangeFrom(filter.producer, build.types, "hash", partitions, {filter.build_column});
		if (filter.bitmap_words) {
			auto bitmap = BitmapDomain(std::move(domain), build, filter.build_column, filter.bitmap_words);
			filter.domain = bitmap.second;
			filter.domain_producer = Finish(std::move(bitmap.first));
			return;
		}
		vector<unique_ptr<Expression>> expressions;
		expressions.push_back(
		    make_uniq<BoundReferenceExpression>(build.types[filter.build_column], filter.build_column));
		auto type = expressions[0]->return_type;
		auto &projection = plan.Make<PhysicalProjection>(vector<LogicalType> {type}, std::move(expressions),
		                                                 build.estimated_cardinality);
		projection.children.push_back(build);
		Wrap(projection, domain);
		filter.domain = &projection;
		filter.domain_producer = Finish(std::move(domain));
	}

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

	PhysicalTableScan &KeyScan(PhysicalTableScan &scan, idx_t column) {
		const auto key = scan.projection_ids.empty() ? column : scan.projection_ids[column];
		vector<ColumnIndex> columns {scan.column_ids[key]};
		unique_ptr<TableFilterSet> filters;
		if (scan.table_filters) {
			filters = make_uniq<TableFilterSet>();
			for (auto &entry : scan.table_filters->filters) {
				idx_t position = 0;
				if (entry.first != key) {
					position = columns.size();
					columns.push_back(scan.column_ids[entry.first]);
				}
				filters->filters.emplace(position, entry.second->Copy());
			}
		}
		ExtraOperatorInfo info;
		info.file_filters = scan.extra_info.file_filters;
		info.total_files = scan.extra_info.total_files;
		info.filtered_files = scan.extra_info.filtered_files;
		info.scan_node_id = scan.extra_info.scan_node_id;
		info.scan_group_id = scan.extra_info.scan_group_id;
		// Keep static predicates and their input columns, but do not share an
		// original join's runtime filters with this independently bound scan.
		return plan
		    .Make<PhysicalTableScan>(vector<LogicalType> {scan.types[column]}, scan.function, scan.bind_data->Copy(),
		                             scan.returned_types, std::move(columns), vector<idx_t> {0}, scan.names,
		                             std::move(filters), scan.estimated_cardinality, std::move(info), scan.parameters,
		                             scan.virtual_columns)
		    .Cast<PhysicalTableScan>();
	}

	DistributedSubtree FilterCorrelatedInput(PhysicalOperator &op, DistributedSubtree input, idx_t key,
	                                         const CorrelatedKeyFilter &filter, const string &owner_producer) {
		auto count = input.fragment.partition_count;
		auto keys = filter.scan;

		bool rescan = keys.source && EstimatedExchangeBytes(*keys.source) <= broadcast_limit &&
		              double(keys.source->estimated_cardinality) * 8 < double(op.estimated_cardinality) &&
		              double(keys.source->estimated_cardinality) * count < double(op.estimated_cardinality);
		if (!rescan && keys.source && keys.source == filter.owner.source && broadcast_limit > 8 * 1024 * 1024) {
			// A filtered Parquet owner can be badly overestimated. Its direct
			// delimiter join estimates the surviving fact rows; use that estimate
			// only when this scan is the complete owner, not a wider superset.
			auto key_bytes =
			    double(keys.source->estimated_cardinality) * (1 + GetTypeIdSize(op.types[key].InternalType()));
			auto probe_bytes = EstimatedExchangeBytes(op);
			auto matches = double(filter.estimated_matching_rows);
			rescan = key_bytes <= broadcast_limit && matches > 0 && matches * 4 < double(op.estimated_cardinality) &&
			         probe_bytes != NumericLimits<double>::Maximum() &&
			         key_bytes * count < probe_bytes * (1 - matches / double(op.estimated_cardinality));
			if (!rescan && key_bytes > broadcast_limit && matches > 0 &&
			    matches * 4 < double(op.estimated_cardinality) && op.types[key].IsIntegral() &&
			    input.incoming.empty() && input.fragment.sources.size() == 1 && input.local_scan &&
			    !probe_filters.count(&op) && CanFilterParquetKey(op, key) &&
			    probe_bytes != NumericLimits<double>::Maximum()) {
				auto &source = keys.source->Cast<PhysicalTableScan>();
				if (!source.extra_info.sample_options) {
					// The proof above uses this scan's complete owner and a single
					// correlation key. Bound its domain instead of trusting the key
					// cardinality estimate, and read no unrelated owner payload.
					auto saved = probe_bytes * (1 - matches / double(op.estimated_cardinality));
					idx_t words = 65536;
					while (words < max_bitmap_words && 16 * words < op.estimated_cardinality &&
					       key_bytes + double(words * 2) * 18 * (partitions + count) < saved) {
						words *= 2;
					}
					// Charge the extra key scan, partial-domain gather and complete
					// domain replicas against traffic estimated to disappear.
					if (op.estimated_cardinality > 8 * words &&
					    key_bytes + double(words) * 18 * (partitions + count) < saved) {
						auto &scan = KeyScan(source, keys.column);
						auto bitmap = BitmapDomain(Build(scan), scan, 0, words);
						auto domain = Exchange(std::move(bitmap.first), count == 1 ? "gather" : "broadcast", count);
						return ParquetBitmapFilter(op, std::move(input), key, std::move(domain), words);
					}
				}
			}
		}
		DistributedSubtree domain;
		if (rescan) {
			// Project bounded keys before their exchange. A repeated independent
			// Parquet scan cannot introduce a dependency on the delimiter consumer.
			domain = Build(*keys.source);
		} else {
			keys = filter.owner;
			auto &owner = *keys.source;
			auto key_bytes =
			    double(owner.estimated_cardinality) * (1 + GetTypeIdSize(owner.types[keys.column].InternalType()));
			if (owner_producer.empty() || !owner.estimated_cardinality || key_bytes > broadcast_limit ||
			    double(owner.estimated_cardinality) * 8 >= double(op.estimated_cardinality) ||
			    EstimatedExchangeBytes(owner) + key_bytes * count >= EstimatedExchangeBytes(op)) {
				return input;
			}
			// Both delimiter variants consume their complete owner before any
			// DELIM_SCAN pipeline. The other branch projects/broadcasts keys into
			// the SEMI build, so both branches drain the shared producer first.
			// Reuse its output; never duplicate the owner's joins or aggregates.
			domain = ExchangeFrom(owner_producer, owner.types, "hash", partitions, {keys.column});
		}
		vector<unique_ptr<Expression>> expressions;
		expressions.push_back(make_uniq<BoundReferenceExpression>(op.types[key], keys.column));
		auto &projection = plan.Make<PhysicalProjection>(vector<LogicalType> {op.types[key]}, std::move(expressions),
		                                                 keys.source->estimated_cardinality);
		projection.children.push_back(*keys.source);
		Wrap(projection, domain);
		domain = Exchange(std::move(domain), count == 1 ? "gather" : "broadcast", count);
		return SemiFilter(op, std::move(input), key, projection, std::move(domain), 0);
	}

	DistributedSubtree ParquetBitmapFilter(PhysicalOperator &op, DistributedSubtree input, idx_t key,
	                                       DistributedSubtree domain, idx_t words) {
		PlanNode node;
		node.types = op.types;
		node.parquet_mask_words = words;
		node.parquet_mask_key = key;
		node.children.push_back(std::move(input.fragment.root));
		node.children.push_back(std::move(domain.fragment.root));
		input.fragment.root = std::move(node);
		for (auto &edge : domain.incoming) {
			input.incoming.push_back(std::move(edge));
		}
		return input;
	}

	DistributedSubtree BitmapFilter(PhysicalOperator &op, DistributedSubtree input, idx_t key, PhysicalOperator &build,
	                                DistributedSubtree domain, idx_t words) {
		if (input.incoming.empty() && input.fragment.sources.size() == 1 && CanFilterParquetKey(op, key)) {
			// This source-only unary probe can consume the complete domain
			// before scan initialization, using the same build dependency as
			// the word join below. Do not move it across incoming exchanges or
			// another already-installed filter's native fragment root.
			return ParquetBitmapFilter(op, std::move(input), key, std::move(domain), words);
		}
		const auto columns = op.types.size();
		vector<LogicalType> types = op.types;
		vector<unique_ptr<Expression>> expressions;
		for (idx_t i = 0; i < columns; i++) {
			expressions.push_back(make_uniq<BoundReferenceExpression>(op.types[i], i));
		}
		expressions.push_back(FilterHash(op.types[key], key));
		types.push_back(LogicalType::UBIGINT);
		auto &hash = plan.Make<PhysicalProjection>(types, std::move(expressions), op.estimated_cardinality);
		Wrap(hash, input);

		vector<JoinCondition> conditions;
		JoinCondition condition;
		condition.left = FilterWord(columns, words);
		condition.right = make_uniq<BoundReferenceExpression>(LogicalType::UBIGINT, 0);
		condition.comparison = ExpressionType::COMPARE_EQUAL;
		conditions.push_back(std::move(condition));
		types.push_back(LogicalType::UBIGINT);
		LogicalComparisonJoin logical(JoinType::INNER);
		logical.types = types;
		auto &join =
		    plan.Make<PhysicalHashJoin>(logical, hash, build, std::move(conditions), JoinType::INNER, vector<idx_t> {},
		                                vector<idx_t> {1}, vector<LogicalType> {}, op.estimated_cardinality, nullptr);
		auto node = Node(join);
		node.children.push_back(std::move(input.fragment.root));
		node.children.push_back(std::move(domain.fragment.root));
		input.fragment.root = std::move(node);
		for (auto &edge : domain.incoming) {
			input.incoming.push_back(std::move(edge));
		}

		// Compute the bit only after matching a populated word. Sparse domains
		// can discard most probes without evaluating or carrying another column.
		vector<unique_ptr<Expression>> arguments;
		arguments.push_back(FilterBit(columns));
		arguments.push_back(make_uniq<BoundReferenceExpression>(LogicalType::UBIGINT, columns + 1));
		expressions.clear();
		expressions.push_back(make_uniq<BoundComparisonExpression>(
		    ExpressionType::COMPARE_NOTEQUAL, FilterFunction("&", std::move(arguments)),
		    make_uniq<BoundConstantExpression>(Value::UBIGINT(0))));
		auto &filter = plan.Make<PhysicalFilter>(types, std::move(expressions), op.estimated_cardinality);
		Wrap(filter, input);
		expressions.clear();
		for (idx_t i = 0; i < columns; i++) {
			expressions.push_back(make_uniq<BoundReferenceExpression>(op.types[i], i));
		}
		auto &output = plan.Make<PhysicalProjection>(op.types, std::move(expressions), op.estimated_cardinality);
		Wrap(output, input);
		return input;
	}

	DistributedSubtree SemiFilter(PhysicalOperator &op, DistributedSubtree input, idx_t key, PhysicalOperator &build,
	                              DistributedSubtree domain, idx_t build_key, idx_t bitmap_words = 0) {
		if (bitmap_words) {
			return BitmapFilter(op, std::move(input), key, build, std::move(domain), bitmap_words);
		}
		// NULL-safe equality is only a necessary condition; the original
		// join still decides matches. SEMI preserves probe duplicates.
		vector<JoinCondition> conditions;
		JoinCondition condition;
		condition.left = make_uniq<BoundReferenceExpression>(op.types[key], key);
		condition.right = make_uniq<BoundReferenceExpression>(build.types[build_key], build_key);
		condition.comparison = ExpressionType::COMPARE_NOT_DISTINCT_FROM;
		conditions.push_back(std::move(condition));
		LogicalComparisonJoin logical(JoinType::SEMI);
		logical.types = op.types;
		auto &join = plan.Make<PhysicalHashJoin>(logical, op, build, std::move(conditions), JoinType::SEMI,
		                                         op.estimated_cardinality);
		auto node = Node(join);
		node.children.push_back(std::move(input.fragment.root));
		node.children.push_back(std::move(domain.fragment.root));
		input.fragment.root = std::move(node);
		for (auto &edge : domain.incoming) {
			input.incoming.push_back(std::move(edge));
		}
		return input;
	}

	unordered_map<const PhysicalOperator *, EarlyProbeFilter> PlanCTEFilters(PhysicalOperator &root) {
		unordered_map<const PhysicalOperator *, EarlyProbeFilter> filters;
		if (partitions == 1 || root.type != PhysicalOperatorType::CTE) {
			return filters;
		}
		auto &cte = root.Cast<PhysicalCTE>();
		auto &owner = root.children[0].get();
		auto bytes = EstimatedExchangeBytes(owner);
		if (HasScopedState(owner) || !owner.estimated_cardinality || bytes > 8 * 1024 * 1024) {
			return filters;
		}
		bool independent = true;
		std::set<const PhysicalOperator *> visited;
		std::function<void(PhysicalOperator &)> visit = [&](PhysicalOperator &op) {
			// Do not transfer a filter across nested ownership or a shared DAG
			// input that another consumer may need without this restriction.
			if (!visited.insert(&op).second || op.type == PhysicalOperatorType::CTE ||
			    op.type == PhysicalOperatorType::LEFT_DELIM_JOIN || op.type == PhysicalOperatorType::RIGHT_DELIM_JOIN) {
				independent = false;
				return;
			}
			if (op.type == PhysicalOperatorType::HASH_JOIN) {
				auto &join = op.Cast<PhysicalHashJoin>();
				auto &probe = join.children[0].get();
				auto &build = join.children[1].get();
				auto probe_bytes = EstimatedExchangeBytes(probe);
				if ((join.join_type == JoinType::INNER || join.join_type == JoinType::SEMI) &&
				    join.delim_types.empty() && join.conditions.size() == 1 &&
				    build.type == PhysicalOperatorType::CTE_SCAN &&
				    build.Cast<PhysicalColumnDataScan>().cte_index == cte.table_index && build.types == owner.types &&
				    !HasScopedState(probe) &&
				    double(owner.estimated_cardinality) * 8 < double(probe.estimated_cardinality) &&
				    probe_bytes != NumericLimits<double>::Maximum() && bytes * partitions < probe_bytes) {
					auto &condition = join.conditions[0];
					if ((condition.comparison == ExpressionType::COMPARE_EQUAL ||
					     condition.comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) &&
					    condition.left->GetExpressionClass() == ExpressionClass::BOUND_REF &&
					    condition.right->GetExpressionClass() == ExpressionClass::BOUND_REF &&
					    condition.left->return_type.IsIntegral() &&
					    condition.left->return_type == condition.right->return_type) {
						EarlyProbeFilter filter;
						filter.column = condition.left->Cast<BoundReferenceExpression>().index;
						filter.build = &owner;
						filter.build_column = condition.right->Cast<BoundReferenceExpression>().index;
						filters.emplace(&probe, filter);
					}
				}
			}
			for (auto &child : op.GetChildren()) {
				visit(const_cast<PhysicalOperator &>(child.get()));
			}
		};
		visit(root.children[1]);
		if (!independent) {
			filters.clear();
		}
		return filters;
	}

	DistributedSubtree Scoped(PhysicalOperator &root) {
		DistributedSubtree result;
		CorrelatedPartition partitioning;
		auto cte_filters = PlanCTEFilters(root);
		const bool partitioned = partitions > 1 && PlanCorrelatedPartition(root, partitioning);
		result.fragment.partition_count = partitioned ? partitions : 1;
		if (partitioned && !partitioning.output_keys.empty()) {
			result.hash_keys = {partitioning.output_keys[0]};
		}
		auto &node = result.fragment.root;
		node.types = root.types;
		node.scoped = true;
		PhysicalPlanInputBindings bindings;
		string owner_producer;
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
				auto child = Build(op);
				auto filter = partitioning.filters.find(&op);
				if (partitioned && filter != partitioning.filters.end()) {
					child = FilterCorrelatedInput(op, std::move(child), partition_input->second.GetIndex(),
					                              filter->second, owner_producer);
				}
				auto cte_filter = cte_filters.find(&op);
				if (cte_filter != cte_filters.end() && !owner_producer.empty()) {
					// The native CTE materializes its owner before its consumers.
					// Each extra branch also drains that owner into a SEMI build
					// before probing, so bounded fanout cannot depend on fact rows.
					auto &key_filter = cte_filter->second;
					auto count = child.fragment.partition_count;
					auto domain = ExchangeFrom(owner_producer, key_filter.build->types,
					                           count == 1 ? "gather" : "broadcast", count);
					child = SemiFilter(op, std::move(child), key_filter.column, *key_filter.build, std::move(domain),
					                   key_filter.build_column);
				}
				auto input = Exchange(std::move(child), distribution, result.fragment.partition_count, keys);
				if ((partitioned || root.type == PhysicalOperatorType::CTE) && &op == &root.children[0].get()) {
					owner_producer = input.incoming[0].producer;
				}
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
		// Casts and compression can change HASH even when they are injective.
		// Retain routing only through unchanged references and row filters.
		if (op.type == PhysicalOperatorType::PROJECTION) {
			vector<idx_t> output_keys;
			auto &expressions = op.Cast<PhysicalProjection>().select_list;
			for (auto key : child.hash_keys) {
				for (idx_t i = 0; i < expressions.size(); i++) {
					auto &expression = *expressions[i];
					if (expression.GetExpressionClass() == ExpressionClass::BOUND_REF &&
					    expression.Cast<BoundReferenceExpression>().index == key &&
					    expression.return_type == child.fragment.root.types[key]) {
						output_keys.push_back(i);
						break;
					}
				}
			}
			if (output_keys.size() != child.hash_keys.size()) {
				output_keys.clear();
			}
			child.hash_keys = std::move(output_keys);
		} else if (op.type != PhysicalOperatorType::FILTER) {
			child.hash_keys.clear();
		}
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
		std::set<idx_t> referenced;
		for (idx_t col = 0; col < types.size(); col++) {
			expressions.push_back(make_uniq<BoundReferenceExpression>(types[col], col));
		}
		for (auto &key : keys) {
			if (key->GetExpressionClass() == ExpressionClass::BOUND_REF) {
				auto column = key->Cast<BoundReferenceExpression>().index;
				if (referenced.insert(column).second) {
					// The key is already in the payload. Reuse its position instead
					// of copying and transporting a second column of the same values.
					columns.push_back(column);
					continue;
				}
			}
			// Repeated keys still occupy separate hash positions: the opposite
			// join input can use different columns for those conditions.
			columns.push_back(types.size());
			types.push_back(key->return_type);
			expressions.push_back(key->Copy());
		}
		if (types.size() != child.fragment.root.types.size()) {
			auto &projection = plan.Make<PhysicalProjection>(types, std::move(expressions), 0);
			Wrap(projection, child);
		}
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
	optional_idx GroupKeyReference(const Expression &expression, vector<const BoundFunctionExpression *> &transforms) {
		if (expression.GetExpressionType() == ExpressionType::BOUND_REF) {
			return expression.Cast<BoundReferenceExpression>().index;
		}
		if (expression.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
			return optional_idx();
		}
		auto &function = expression.Cast<BoundFunctionExpression>();
		if (function.children.size() != 2 ||
		    function.children[1]->GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) {
			return optional_idx();
		}
		auto &input = function.children[0]->return_type;
		auto &output = function.return_type;
		if (!input.IsIntegral() || !output.IsIntegral()) {
			return optional_idx();
		}
		auto input_width = GetTypeIdSize(input.InternalType());
		auto output_width = GetTypeIdSize(output.InternalType());
		const bool compress = output.IsUnsigned() && output_width <= 8 && input_width > output_width &&
		                      function.function == CMIntegralCompressFun::GetFunction(input, output);
		const bool decompress = input.IsUnsigned() && input_width <= 8 && input_width < output_width &&
		                        function.function == CMIntegralDecompressFun::GetFunction(input, output);
		if (!compress && !decompress) {
			return optional_idx();
		}
		transforms.push_back(&function);
		return GroupKeyReference(*function.children[0], transforms);
	}
	bool InjectiveGroupKey(pair<Value, Value> bounds, const vector<const BoundFunctionExpression *> &transforms) {
		// No nonempty split: there are no groups to merge. Otherwise validate
		// every internal compression against file bounds, not its name or the
		// optimizer's estimates. Explicit narrowing can wrap and merge keys.
		if (transforms.empty() || (bounds.first.IsNull() && bounds.second.IsNull())) {
			return true;
		}
		if (!bounds.first.DefaultTryCastAs(LogicalType::HUGEINT) ||
		    !bounds.second.DefaultTryCastAs(LogicalType::HUGEINT)) {
			return false;
		}
		auto lower = bounds.first.GetValue<hugeint_t>();
		auto upper = bounds.second.GetValue<hugeint_t>();
		for (auto i = transforms.size(); i > 0; i--) {
			auto &function = *transforms[i - 1];
			auto offset = function.children[1]->Cast<BoundConstantExpression>().value;
			if (offset.IsNull() || !offset.DefaultTryCastAs(LogicalType::HUGEINT)) {
				return false;
			}
			auto base = offset.GetValue<hugeint_t>();
			if (GetTypeIdSize(function.children[0]->return_type.InternalType()) >
			    GetTypeIdSize(function.return_type.InternalType())) {
				if (lower < base || !Hugeint::TrySubtractInPlace(lower, base) ||
				    !Hugeint::TrySubtractInPlace(upper, base)) {
					return false;
				}
			} else if (!Hugeint::TryAddInPlace(lower, base) || !Hugeint::TryAddInPlace(upper, base)) {
				return false;
			}
			// Checked endpoint casts prove the entire monotone range fits. Do
			// not evaluate an overflowing native compression as part of planning.
			if (!Value::HUGEINT(lower).DefaultTryCastAs(function.return_type) ||
			    !Value::HUGEINT(upper).DefaultTryCastAs(function.return_type)) {
				return false;
			}
		}
		return true;
	}
	bool CompleteParquetGroups(PhysicalOperator &input, DistributedSubtree &child,
	                           const vector<unique_ptr<Expression>> &groups) {
		if (!child.local_scan || child.fragment.sources.size() != 1) {
			return false;
		}
		for (auto &group : groups) {
			if (group->GetExpressionType() != ExpressionType::BOUND_REF || !group->return_type.IsIntegral()) {
				continue;
			}
			auto column = group->Cast<BoundReferenceExpression>().index;
			auto op = &input;
			vector<const BoundFunctionExpression *> transforms;
			while (!op->children.empty()) {
				if (op->type == PhysicalOperatorType::PROJECTION) {
					auto &expressions = op->Cast<PhysicalProjection>().select_list;
					if (column >= expressions.size()) {
						break;
					}
					auto reference = GroupKeyReference(*expressions[column], transforms);
					if (!reference.IsValid()) {
						break;
					}
					column = reference.GetIndex();
				} else if (op->type == PhysicalOperatorType::HASH_JOIN) {
					auto &join = op->Cast<PhysicalHashJoin>();
					if ((join.join_type != JoinType::INNER && join.join_type != JoinType::LEFT &&
					     join.join_type != JoinType::SEMI && join.join_type != JoinType::ANTI &&
					     join.join_type != JoinType::MARK) ||
					    column >= join.lhs_output_columns.col_idxs.size()) {
						break;
					}
					column = join.lhs_output_columns.col_idxs[column];
				} else if (op->type != PhysicalOperatorType::FILTER) {
					break;
				}
				op = &op->children[0].get();
			}
			if (op != child.local_scan.get() || column >= op->types.size() || !op->types[column].IsIntegral()) {
				continue;
			}
			auto &scan = op->Cast<PhysicalTableScan>();
			if (!IsParquetScan(scan.function.name)) {
				continue;
			}
			auto position = scan.projection_ids.empty() ? column : scan.projection_ids[column];
			auto &index = scan.column_ids[position];
			if (index.HasChildren()) {
				continue;
			}
			vector<OpenFileInfo> splits;
			for (auto &split : child.fragment.sources[0].splits) {
				splits.push_back(DecodeFile(split.payload));
			}
			pair<Value, Value> bounds;
			vector<idx_t> assignment;
			if (PlanParquetGroupPartitions(context, ParquetBind(scan.bind_data.get()), splits, index.GetPrimaryIndex(),
			                               child.fragment.partition_count, assignment, &bounds) &&
			    (assignment.empty() || !scan.extra_info.sample_options) &&
			    InjectiveGroupKey(std::move(bounds), transforms)) {
				child.fragment.sources[0].split_partitions = std::move(assignment);
				return true;
			}
		}
		return false;
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
		if (child.fragment.partition_count == 1 || CompleteParquetGroups(op.children[0], child, groups)) {
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
		if (partial.type == PhysicalOperatorType::HASH_GROUP_BY) {
			partial.Cast<PhysicalHashAggregate>().SetPartialStateOutput();
		}
		Wrap(partial, child);
		child = Exchange(std::move(child), groups.empty() ? "gather" : "hash", groups.empty() ? 1 : partitions, keys);
		auto &final = MakeAggregate(std::move(final_groups), std::move(finals));
		Wrap(final, child);
		auto &projection = plan.Make<PhysicalProjection>(op.types, std::move(output), op.estimated_cardinality);
		Wrap(projection, child);
		return child;
	}
	DistributedSubtree Join(PhysicalHashJoin &join) {
		auto filter = PlanProbeFilter(join);
		// Keep existing probe filters first, and never make both join inputs
		// depend on a new filter supplied by the opposite input.
		auto build_filter = filter.input ? EarlyProbeFilter() : PlanBuildFilter(join);
		if (build_filter.input) {
			auto found = shared_join_builds.find(build_filter.build);
			if (found == shared_join_builds.end()) {
				auto producer = Finish(Build(*build_filter.build));
				found = shared_join_builds.emplace(build_filter.build, std::move(producer)).first;
			}
			build_filter.domain = build_filter.build;
			build_filter.domain_producer = found->second;
			probe_filters[build_filter.input].push_back(build_filter);
		}
		string build_producer;
		auto shared_build = shared_join_builds.find(&join.children[1].get());
		if (shared_build != shared_join_builds.end()) {
			build_producer = shared_build->second;
		}
		// Do not trade away a filter moved before another join, or cost a larger
		// broadcast against probes already reduced by an ancestor's early filter.
		const bool preserve_filter =
		    (filter.input && filter.input != &join.children[0].get()) || probe_filters.count(&join.children[0].get());
		const auto join_broadcast_limit = preserve_filter ? idx_t(8 * 1024 * 1024) : broadcast_limit;
		if (filter.keys_only && BroadcastJoin(join, partitions, join_broadcast_limit)) {
			// A direct key filter only reduces this join's own hash input. Fusing
			// the probe with a full-build broadcast removes that exchange entirely.
			filter = EarlyProbeFilter();
		}
		DistributedSubtree left, right;
		if (filter.input) {
			// Both the original join and the early SEMI build drain this shared
			// producer before probing. Never rescan or recompute the build side.
			if (build_producer.empty()) {
				build_producer = Finish(Build(join.children[1]));
			}
			filter.producer = build_producer;
			filter.domain = filter.build;
			filter.domain_producer = filter.producer;
			if (filter.keys_only) {
				BuildKeyDomain(filter);
			}
			probe_filters[filter.input].push_back(filter);
			left = Build(join.children[0]);
			auto &filters = probe_filters.at(filter.input);
			filters.pop_back();
			if (filters.empty()) {
				probe_filters.erase(filter.input);
			}
		} else {
			left = Build(join.children[0]);
			if (build_producer.empty()) {
				right = Build(join.children[1]);
			}
		}
		if (build_filter.input) {
			auto &filters = probe_filters.at(build_filter.input);
			filters.pop_back();
			if (filters.empty()) {
				probe_filters.erase(build_filter.input);
			}
		}
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
		const bool broadcast = BroadcastJoin(join, left.fragment.partition_count, join_broadcast_limit);
		if (broadcast) {
			// Every probe partition sees the complete build input. Preserve its
			// existing distribution and fuse the probe subtree with the join;
			// shuffling it by the join key adds no correctness guarantee here.
			auto count = left.fragment.partition_count;
			right = build_producer.empty() ? Exchange(std::move(right), count == 1 ? "gather" : "broadcast", count)
			                               : ExchangeFrom(build_producer, join.children[1].get().types,
			                                              count == 1 ? "gather" : "broadcast", count);
		} else {
			auto lk = ProjectKeys(left, left_keys);
			left = Exchange(std::move(left), "hash", partitions, lk);
			if (!build_producer.empty()) {
				vector<idx_t> rk;
				for (auto &key : right_keys) {
					rk.push_back(key->Cast<BoundReferenceExpression>().index);
				}
				right = ExchangeFrom(build_producer, join.children[1].get().types, "hash", partitions, rk);
			} else {
				auto rk = ProjectKeys(right, right_keys);
				right = Exchange(std::move(right), "hash", partitions, rk);
			}
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
		vector<idx_t> output_keys;
		if (join.join_type == JoinType::INNER || join.join_type == JoinType::SEMI) {
			for (auto key : left.hash_keys) {
				auto &columns = join.lhs_output_columns.col_idxs;
				auto found = std::find(columns.begin(), columns.end(), key);
				if (found == columns.end()) {
					output_keys.clear();
					break;
				}
				output_keys.push_back(idx_t(found - columns.begin()));
			}
		}
		left.hash_keys = std::move(output_keys);
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
                        unordered_map<string, FrozenFile> &frozen, idx_t &used, SourceSnapshotCache *cache) {
	struct FileScan {
		TableFunctionRef &ref;
		vector<OpenFileInfo> files;
	};
	vector<FileScan> scans;
	unordered_map<string, string> references;
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
			auto existing = references.find(target);
			if (existing == references.end()) {
				if (frozen.size() >= 4096) {
					throw InvalidInputException("FTE source file count exceeds limit");
				}
				auto snapshot = cache ? cache->Freeze(context, file.path, target, budget - used)
				                      : FreezeFile(context, file.path, target, budget - used);
				used += snapshot.bytes;
				existing = references.emplace(target, snapshot.path).first;
				frozen.emplace(existing->second, std::move(snapshot));
			}
			paths.push_back(Value(existing->second));
			files.emplace_back(existing->second);
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
                      idx_t *source_bytes, SourceSnapshotCache *source_cache, idx_t worker_operator_bytes) {
	CheckPartitions(partitions);
	if (worker_operator_bytes > idx_t(NumericLimits<int64_t>::Maximum())) {
		throw InvalidInputException("worker operator planning budget exceeds the supported range");
	}
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
			StageFileArguments(context, select, snapshot_directory, source_budget, frozen, used, source_cache);
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
		LogicalValidator validator(partitions);
		validator.frozen_files = !snapshot_directory.empty();
		validator.VisitOperator(*planner.plan);
		CheckTypes(planner.types);
		if (ClientConfig::GetConfig(context).enable_optimizer && planner.plan->RequireOptimizer()) {
			Optimizer optimizer(*planner.binder, context);
			planner.plan = optimizer.Optimize(std::move(planner.plan));
		}
		PhysicalPlanGenerator physical_planner(context);
		auto physical = physical_planner.Plan(std::move(planner.plan));
		AnalyticalPlanner distributed(context, *physical, graph, partitions, worker_operator_bytes);
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
	if (node.parquet_mask_words) {
		auto &probe =
		    LoadNode(context, plan, node.children[0], inputs, sources, assignments, used_sources, used_inputs, filters);
		auto &domain =
		    LoadNode(context, plan, node.children[1], inputs, sources, assignments, used_sources, used_inputs, filters);
		return MakeParquetMask(plan, probe, domain, node.parquet_mask_key, node.parquet_mask_words);
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
		optional_idx owner;
		for (auto &id : assignment->second) {
			auto found = known.find(id);
			if (found == known.end() || !selected_ids.insert(id).second) {
				throw InvalidInputException("unknown or duplicate assigned split %s", id);
			}
			if (!spec.split_partitions.empty()) {
				auto partition = spec.split_partitions[idx_t(found->second - spec.splits.data())];
				if (!parquet || (owner.IsValid() && owner.GetIndex() != partition)) {
					throw InvalidInputException("owned Parquet splits must belong to one complete partition");
				}
				owner = partition;
			}
			selected.push_back(*found->second);
		}
		if (owner.IsValid() && idx_t(std::count(spec.split_partitions.begin(), spec.split_partitions.end(),
		                                        owner.GetIndex())) != selected.size()) {
			throw InvalidInputException("owned Parquet splits must cover the complete partition");
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

vector<idx_t> BindHashColumns(ClientContext &context, const string &partitioning,
                              const vector<LogicalType> &input_types, idx_t partitions) {
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
	if (types != input_types || expressions.empty()) {
		throw InvalidInputException("HASH partitioning requires matching input schema and non-empty keys");
	}
	vector<idx_t> columns;
	for (auto &expression : expressions) {
		if (expression->GetExpressionClass() != ExpressionClass::BOUND_REF) {
			throw SerializationException("unsupported native HASH expression");
		}
		auto &ref = expression->Cast<BoundReferenceExpression>();
		if (ref.index >= types.size() || ref.return_type != types[ref.index]) {
			throw SerializationException("native HASH reference does not match input schema");
		}
		columns.push_back(ref.index);
	}
	return columns;
}

vector<idx_t> HashPartitions(const vector<idx_t> &columns, DataChunk &chunk, idx_t partitions, Vector &hashes) {
	// Preparation has checked the schema and bound references. Hash the same
	// vectors in the same order without rebuilding an executor for each batch.
	D_ASSERT(!columns.empty() && partitions > 0);
	VectorOperations::Hash(chunk.data[columns[0]], hashes, chunk.size());
	for (idx_t i = 1; i < columns.size(); i++) {
		VectorOperations::CombineHash(hashes, chunk.data[columns[i]], chunk.size());
	}
	UnifiedVectorFormat data;
	hashes.ToUnifiedFormat(chunk.size(), data);
	auto values = UnifiedVectorFormat::GetData<hash_t>(data);
	vector<idx_t> result;
	result.reserve(chunk.size());
	for (idx_t i = 0; i < chunk.size(); i++) {
		result.push_back(values[data.sel->get_index(i)] % partitions);
	}
	return result;
}

vector<idx_t> HashPartitions(ClientContext &context, const string &partitioning, DataChunk &chunk, idx_t partitions) {
	auto columns = BindHashColumns(context, partitioning, chunk.GetTypes(), partitions);
	Vector hashes(LogicalType::HASH);
	return HashPartitions(columns, chunk, partitions, hashes);
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

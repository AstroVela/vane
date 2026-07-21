// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/main/relation.hpp"
#include "duckdb/common/printer.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/main/relation/aggregate_relation.hpp"
#include "duckdb/main/relation/cross_product_relation.hpp"
#include "duckdb/main/relation/distinct_relation.hpp"
#include "duckdb/main/relation/explain_relation.hpp"
#include "duckdb/main/relation/filter_relation.hpp"
#include "duckdb/main/relation/insert_relation.hpp"
#include "duckdb/main/relation/limit_relation.hpp"
#include "duckdb/main/relation/repartition_relation.hpp"
#include "duckdb/main/relation/local_exchange_relation.hpp"
#include "duckdb/main/relation/order_relation.hpp"
#include "duckdb/main/relation/projection_relation.hpp"
#include "duckdb/main/relation/setop_relation.hpp"
#include "duckdb/main/relation/subquery_relation.hpp"
#include "duckdb/main/relation/table_function_relation.hpp"
#include "duckdb/main/relation/create_table_relation.hpp"
#include "duckdb/main/relation/create_view_relation.hpp"
#include "duckdb/main/relation/write_csv_relation.hpp"
#include "duckdb/main/relation/write_parquet_relation.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression_binder/relation_binder.hpp"
#include "duckdb/planner/expression_binder/where_binder.hpp"
#include "duckdb/planner/table_binding.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/expression/conjunction_expression.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/main/relation/join_relation.hpp"
#include "duckdb/main/relation/value_relation.hpp"
#include "duckdb/parser/statement/explain_statement.hpp"

namespace duckdb {

shared_ptr<Relation> Relation::Project(const string &select_list) {
	return Project(select_list, vector<string>());
}

shared_ptr<Relation> Relation::Project(const string &expression, const string &alias) {
	return Project(expression, vector<string>({alias}));
}

shared_ptr<Relation> Relation::Project(const string &select_list, const vector<string> &aliases) {
	auto expressions = Parser::ParseExpressionList(select_list, context->GetContext()->GetParserOptions());
	return make_shared_ptr<ProjectionRelation>(shared_from_this(), std::move(expressions), aliases);
}

shared_ptr<Relation> Relation::Project(const vector<string> &expressions) {
	vector<string> aliases;
	return Project(expressions, aliases);
}

shared_ptr<Relation> Relation::Project(vector<unique_ptr<ParsedExpression>> expressions,
                                       const vector<string> &aliases) {
	return make_shared_ptr<ProjectionRelation>(shared_from_this(), std::move(expressions), aliases);
}

static vector<unique_ptr<ParsedExpression>> StringListToExpressionList(const ClientContext &context,
                                                                       const vector<string> &expressions) {
	if (expressions.empty()) {
		throw ParserException("Zero expressions provided");
	}
	vector<unique_ptr<ParsedExpression>> result_list;
	for (auto &expr : expressions) {
		auto expression_list = Parser::ParseExpressionList(expr, context.GetParserOptions());
		if (expression_list.size() != 1) {
			throw ParserException("Expected a single expression in the expression list");
		}
		result_list.push_back(std::move(expression_list[0]));
	}
	return result_list;
}

shared_ptr<Relation> Relation::Project(const vector<string> &expressions, const vector<string> &aliases) {
	auto result_list = StringListToExpressionList(*context->GetContext(), expressions);
	return make_shared_ptr<ProjectionRelation>(shared_from_this(), std::move(result_list), aliases);
}

shared_ptr<Relation> Relation::Filter(const string &expression) {
	auto expression_list = Parser::ParseExpressionList(expression, context->GetContext()->GetParserOptions());
	if (expression_list.size() != 1) {
		throw ParserException("Expected a single expression as filter condition");
	}
	return Filter(std::move(expression_list[0]));
}

shared_ptr<Relation> Relation::Filter(unique_ptr<ParsedExpression> expression) {
	return make_shared_ptr<FilterRelation>(shared_from_this(), std::move(expression));
}

shared_ptr<Relation> Relation::Filter(const vector<string> &expressions) {
	// if there are multiple expressions, we AND them together
	auto expression_list = StringListToExpressionList(*context->GetContext(), expressions);
	D_ASSERT(!expression_list.empty());

	auto expr = std::move(expression_list[0]);
	for (idx_t i = 1; i < expression_list.size(); i++) {
		expr = make_uniq<ConjunctionExpression>(ExpressionType::CONJUNCTION_AND, std::move(expr),
		                                        std::move(expression_list[i]));
	}
	return make_shared_ptr<FilterRelation>(shared_from_this(), std::move(expr));
}

shared_ptr<Relation> Relation::Limit(int64_t limit, int64_t offset) {
	return make_shared_ptr<LimitRelation>(shared_from_this(), limit, offset);
}

shared_ptr<Relation> Relation::Repartition(idx_t num_partitions, vector<unique_ptr<ParsedExpression>> partition_by) {
	if (num_partitions > 0 && false) {
		throw InvalidInputException("num_partitions must be greater than zero");
	}
	return make_shared_ptr<RepartitionRelation>(shared_from_this(), num_partitions, std::move(partition_by));
}

shared_ptr<Relation> Relation::Repartition(idx_t num_partitions, const vector<string> &partition_by) {
	vector<unique_ptr<ParsedExpression>> expressions;
	if (!partition_by.empty()) {
		expressions = StringListToExpressionList(*context->GetContext(), partition_by);
	}
	return Repartition(num_partitions, std::move(expressions));
}

shared_ptr<Relation> Relation::LocalExchange(idx_t num_partitions) {
	return make_shared_ptr<LocalExchangeRelation>(shared_from_this(), num_partitions);
}

shared_ptr<Relation> Relation::Order(const string &expression) {
	auto order_list = Parser::ParseOrderList(expression, context->GetContext()->GetParserOptions());
	return Order(std::move(order_list));
}

shared_ptr<Relation> Relation::Order(vector<OrderByNode> order_list) {
	return make_shared_ptr<OrderRelation>(shared_from_this(), std::move(order_list));
}

shared_ptr<Relation> Relation::Order(const vector<string> &expressions) {
	if (expressions.empty()) {
		throw ParserException("Zero ORDER BY expressions provided");
	}
	vector<OrderByNode> order_list;
	for (auto &expression : expressions) {
		auto inner_list = Parser::ParseOrderList(expression, context->GetContext()->GetParserOptions());
		if (inner_list.size() != 1) {
			throw ParserException("Expected a single ORDER BY expression in the expression list");
		}
		order_list.push_back(std::move(inner_list[0]));
	}
	return Order(std::move(order_list));
}

shared_ptr<Relation> Relation::Join(const shared_ptr<Relation> &other, const string &condition, JoinType type,
                                    JoinRefType ref_type) {
	auto expression_list = Parser::ParseExpressionList(condition, context->GetContext()->GetParserOptions());
	D_ASSERT(!expression_list.empty());
	return Join(other, std::move(expression_list), type, ref_type);
}

shared_ptr<Relation> Relation::Join(const shared_ptr<Relation> &other,
                                    vector<unique_ptr<ParsedExpression>> expression_list, JoinType type,
                                    JoinRefType ref_type) {
	if (expression_list.size() > 1 || expression_list[0]->GetExpressionType() == ExpressionType::COLUMN_REF) {
		// multiple columns or single column ref: the condition is a USING list
		vector<string> using_columns;
		for (auto &expr : expression_list) {
			if (expr->GetExpressionType() != ExpressionType::COLUMN_REF) {
				throw ParserException("Expected a single expression as join condition");
			}
			auto &colref = expr->Cast<ColumnRefExpression>();
			if (colref.IsQualified()) {
				throw ParserException("Expected unqualified column for column in USING clause");
			}
			using_columns.push_back(colref.column_names[0]);
		}
		return make_shared_ptr<JoinRelation>(shared_from_this(), other, std::move(using_columns), type, ref_type);
	} else {
		// single expression that is not a column reference: use the expression as a join condition
		return make_shared_ptr<JoinRelation>(shared_from_this(), other, std::move(expression_list[0]), type, ref_type);
	}
}

shared_ptr<Relation> Relation::CrossProduct(const shared_ptr<Relation> &other, JoinRefType join_ref_type) {
	return make_shared_ptr<CrossProductRelation>(shared_from_this(), other, join_ref_type);
}

shared_ptr<Relation> Relation::Union(const shared_ptr<Relation> &other) {
	return make_shared_ptr<SetOpRelation>(shared_from_this(), other, SetOperationType::UNION, true);
}

shared_ptr<Relation> Relation::Except(const shared_ptr<Relation> &other) {
	return make_shared_ptr<SetOpRelation>(shared_from_this(), other, SetOperationType::EXCEPT, true);
}

shared_ptr<Relation> Relation::Intersect(const shared_ptr<Relation> &other) {
	return make_shared_ptr<SetOpRelation>(shared_from_this(), other, SetOperationType::INTERSECT, true);
}

shared_ptr<Relation> Relation::Distinct() {
	return make_shared_ptr<DistinctRelation>(shared_from_this());
}

shared_ptr<Relation> Relation::Alias(const string &alias) {
	return make_shared_ptr<SubqueryRelation>(shared_from_this(), alias);
}

shared_ptr<Relation> Relation::Aggregate(const string &aggregate_list) {
	auto expression_list = Parser::ParseExpressionList(aggregate_list, context->GetContext()->GetParserOptions());
	return make_shared_ptr<AggregateRelation>(shared_from_this(), std::move(expression_list));
}

shared_ptr<Relation> Relation::Aggregate(vector<unique_ptr<ParsedExpression>> expressions) {
	return make_shared_ptr<AggregateRelation>(shared_from_this(), std::move(expressions));
}

shared_ptr<Relation> Relation::Aggregate(const string &aggregate_list, const string &group_list) {
	auto expression_list = Parser::ParseExpressionList(aggregate_list, context->GetContext()->GetParserOptions());
	auto groups = Parser::ParseGroupByList(group_list, context->GetContext()->GetParserOptions());
	return make_shared_ptr<AggregateRelation>(shared_from_this(), std::move(expression_list), std::move(groups));
}

shared_ptr<Relation> Relation::Aggregate(const vector<string> &aggregates) {
	auto aggregate_list = StringListToExpressionList(*context->GetContext(), aggregates);
	return make_shared_ptr<AggregateRelation>(shared_from_this(), std::move(aggregate_list));
}

shared_ptr<Relation> Relation::Aggregate(const vector<string> &aggregates, const vector<string> &groups) {
	auto aggregate_list = StringUtil::Join(aggregates, ", ");
	auto group_list = StringUtil::Join(groups, ", ");
	return this->Aggregate(aggregate_list, group_list);
}

shared_ptr<Relation> Relation::Aggregate(vector<unique_ptr<ParsedExpression>> expressions, const string &group_list) {
	auto groups = Parser::ParseGroupByList(group_list, context->GetContext()->GetParserOptions());
	return make_shared_ptr<AggregateRelation>(shared_from_this(), std::move(expressions), std::move(groups));
}

string Relation::GetAlias() {
	return alias;
}

unique_ptr<TableRef> Relation::GetTableRef() {
	auto select = make_uniq<SelectStatement>();
	select->node = GetQueryNode();
	return make_uniq<SubqueryRef>(std::move(select), GetAlias());
}

unique_ptr<QueryResult> Relation::Execute() {
	return context->GetContext()->Execute(shared_from_this());
}

unique_ptr<QueryResult> Relation::ExecuteOrThrow() {
	auto res = Execute();
	D_ASSERT(res);
	if (res->HasError()) {
		res->ThrowError();
	}
	return res;
}

BoundStatement Relation::Bind(Binder &binder) {
	SelectStatement stmt;
	stmt.node = GetQueryNode();
	return binder.Bind(stmt.Cast<SQLStatement>());
}

bool Relation::CanMapColumnBindings(Relation &relation) {
	if (relation.InheritsColumnBindings()) {
		auto child = relation.ChildRelation();
		if (child) {
			return CanMapColumnBindings(*child);
		}
	}
	switch (relation.type) {
	case RelationType::JOIN_RELATION: {
		auto &join = relation.Cast<JoinRelation>();
		if (!join.using_columns.empty() || !join.condition || join.join_ref_type != JoinRefType::REGULAR ||
		    join.join_type != JoinType::INNER) {
			return false;
		}
		return CanMapColumnBindings(*join.left) && CanMapColumnBindings(*join.right);
	}
	case RelationType::CROSS_PRODUCT_RELATION: {
		auto &cross = relation.Cast<CrossProductRelation>();
		return CanMapColumnBindings(*cross.left) && CanMapColumnBindings(*cross.right);
	}
	default:
		return true;
	}
}

struct RelationBindingGroup {
	string alias;
	idx_t column_count;
};

static void CollectBindingGroups(Relation &relation, vector<RelationBindingGroup> &groups) {
	if (relation.InheritsColumnBindings()) {
		auto child = relation.ChildRelation();
		if (child) {
			CollectBindingGroups(*child, groups);
			return;
		}
	}
	switch (relation.type) {
	case RelationType::JOIN_RELATION: {
		auto &join = relation.Cast<JoinRelation>();
		if (join.join_type == JoinType::SEMI || join.join_type == JoinType::ANTI) {
			CollectBindingGroups(*join.left, groups);
			return;
		}
		if (join.join_type == JoinType::RIGHT_SEMI || join.join_type == JoinType::RIGHT_ANTI) {
			CollectBindingGroups(*join.right, groups);
			return;
		}
		CollectBindingGroups(*join.left, groups);
		CollectBindingGroups(*join.right, groups);
		return;
	}
	case RelationType::CROSS_PRODUCT_RELATION: {
		auto &cross = relation.Cast<CrossProductRelation>();
		CollectBindingGroups(*cross.left, groups);
		CollectBindingGroups(*cross.right, groups);
		return;
	}
	default:
		groups.push_back({relation.GetAlias(), relation.Columns().size()});
		return;
	}
}

class MappedRelationBinding : public Binding {
public:
	MappedRelationBinding(string alias, vector<string> names, vector<LogicalType> types,
	                      vector<ColumnBinding> column_bindings_p)
	    : Binding(BindingType::BASE, BindingAlias(std::move(alias)), std::move(types), std::move(names),
	              column_bindings_p.empty() ? DConstants::INVALID_INDEX : column_bindings_p[0].table_index),
	      column_bindings(std::move(column_bindings_p)) {
		D_ASSERT(column_bindings.size() == this->names.size());
	}

	BindResult Bind(ColumnRefExpression &colref, idx_t depth) override {
		column_t column_index;
		if (!TryGetBindingIndex(colref.GetColumnName(), column_index)) {
			return BindResult(ColumnNotFoundError(colref.GetColumnName()));
		}
		if (colref.GetAlias().empty()) {
			colref.SetAlias(names[column_index]);
		}
		return BindResult(make_uniq<BoundColumnRefExpression>(colref.GetName(), types[column_index],
		                                                      column_bindings[column_index], depth));
	}

private:
	vector<ColumnBinding> column_bindings;
};

shared_ptr<Binder> Relation::CreateBinderForBoundRelation(Binder &binder, Relation &relation,
                                                          const BoundStatement &bound_relation) {
	auto bindings = bound_relation.plan->GetColumnBindings();
	D_ASSERT(bindings.size() == bound_relation.names.size());
	D_ASSERT(bindings.size() == bound_relation.types.size());

	struct BindingColumns {
		idx_t table_index;
		vector<string> names;
		vector<LogicalType> types;
		vector<ColumnBinding> bindings;
	};
	vector<BindingColumns> binding_columns;
	unordered_map<idx_t, idx_t> binding_positions;
	for (idx_t column_idx = 0; column_idx < bindings.size(); column_idx++) {
		const auto &binding = bindings[column_idx];
		auto entry = binding_positions.find(binding.table_index);
		if (entry == binding_positions.end()) {
			binding_positions.emplace(binding.table_index, binding_columns.size());
			binding_columns.push_back({binding.table_index, {}, {}, {}});
			entry = binding_positions.find(binding.table_index);
		}
		auto &columns = binding_columns[entry->second];
		if (columns.names.size() <= binding.column_index) {
			columns.names.resize(binding.column_index + 1);
			columns.types.resize(binding.column_index + 1);
			columns.bindings.resize(binding.column_index + 1);
		}
		columns.names[binding.column_index] = bound_relation.names[column_idx];
		columns.types[binding.column_index] = bound_relation.types[column_idx];
		columns.bindings[binding.column_index] = binding;
	}

	for (auto &columns : binding_columns) {
		for (idx_t column_idx = 0; column_idx < columns.names.size(); column_idx++) {
			if (columns.names[column_idx].empty() ||
			    columns.bindings[column_idx].table_index == DConstants::INVALID_INDEX) {
				throw InternalException("Failed to build relation bindings: missing column at index %llu", column_idx);
			}
		}
	}

	vector<RelationBindingGroup> relation_groups;
	CollectBindingGroups(relation, relation_groups);
	case_insensitive_map_t<bool> used_aliases;
	auto child_binder = Binder::CreateBinder(binder.context, binder.shared_from_this());
	auto add_binding = [&](string alias, vector<string> names, vector<LogicalType> types,
	                       vector<ColumnBinding> bindings) {
		QueryResult::DeduplicateColumns(names);
		if (alias.empty() || used_aliases.find(alias) != used_aliases.end()) {
			alias = StringUtil::Format("__relation_%llu", bindings[0].table_index);
		}
		while (used_aliases.find(alias) != used_aliases.end()) {
			alias += "_";
		}
		used_aliases.emplace(alias, true);
		child_binder->bind_context.AddBinding(make_uniq<MappedRelationBinding>(std::move(alias), std::move(names),
		                                                                       std::move(types), std::move(bindings)));
	};

	bool mapped_relation_groups = relation_groups.size() == binding_columns.size();
	if (mapped_relation_groups) {
		for (idx_t group_idx = 0; group_idx < relation_groups.size(); group_idx++) {
			if (relation_groups[group_idx].column_count != binding_columns[group_idx].names.size()) {
				mapped_relation_groups = false;
				break;
			}
		}
	}
	if (mapped_relation_groups) {
		for (idx_t group_idx = 0; group_idx < relation_groups.size(); group_idx++) {
			auto &columns = binding_columns[group_idx];
			add_binding(relation_groups[group_idx].alias, std::move(columns.names), std::move(columns.types),
			            std::move(columns.bindings));
		}
		return child_binder;
	}

	if (binding_columns.size() == 1 && relation_groups.size() > 1) {
		auto &columns = binding_columns[0];
		idx_t total_columns = 0;
		for (auto &group : relation_groups) {
			total_columns += group.column_count;
		}
		if (total_columns == columns.names.size()) {
			idx_t offset = 0;
			for (auto &group : relation_groups) {
				vector<string> names(columns.names.begin() + offset,
				                     columns.names.begin() + offset + group.column_count);
				vector<LogicalType> types(columns.types.begin() + offset,
				                          columns.types.begin() + offset + group.column_count);
				vector<ColumnBinding> bindings(columns.bindings.begin() + offset,
				                               columns.bindings.begin() + offset + group.column_count);
				add_binding(group.alias, std::move(names), std::move(types), std::move(bindings));
				offset += group.column_count;
			}
			return child_binder;
		}
	}

	for (auto &columns : binding_columns) {
		auto alias = binding_columns.size() == 1 ? relation.GetAlias() : string();
		add_binding(std::move(alias), std::move(columns.names), std::move(columns.types), std::move(columns.bindings));
	}
	return child_binder;
}

unique_ptr<Expression> Relation::BindExpressionOnBoundRelation(Binder &binder, Relation &relation,
                                                               BoundStatement &bound_relation,
                                                               unique_ptr<ParsedExpression> expression,
                                                               const string &operation) {
	auto child_binder = CreateBinderForBoundRelation(binder, relation, bound_relation);
	unique_ptr<Expression> bound_expression;
	if (operation == "filter") {
		child_binder->BindWhereStarExpression(expression);
		ExpressionBinder::QualifyColumnNames(*child_binder, expression);
		WhereBinder where_binder(*child_binder, binder.context);
		bound_expression = where_binder.Bind(expression);
	} else {
		ExpressionBinder::QualifyColumnNames(*child_binder, expression);
		RelationBinder relation_binder(*child_binder, binder.context, operation);
		bound_expression = relation_binder.Bind(expression);
	}
	if (!bound_expression) {
		throw InternalException("Failed to bind %s expression", operation);
	}
	child_binder->PlanSubqueries(bound_expression, bound_relation.plan);
	binder.MoveCorrelatedExpressionsFrom(*child_binder);
	return bound_expression;
}

BoundStatement Relation::BindSelectNodeOnChild(Binder &binder, Relation &child, BoundStatement child_bound,
                                               unique_ptr<SelectNode> select_node) {
	auto child_binder = CreateBinderForBoundRelation(binder, child, child_bound);
	auto result = child_binder->BindSelectNode(*select_node, std::move(child_bound));
	binder.MoveCorrelatedExpressionsFrom(*child_binder);
	return result;
}

shared_ptr<Relation> Relation::InsertRel(const string &schema_name, const string &table_name) {
	return InsertRel(INVALID_CATALOG, schema_name, table_name);
}

shared_ptr<Relation> Relation::InsertRel(const string &catalog_name, const string &schema_name,
                                         const string &table_name) {
	return make_shared_ptr<InsertRelation>(shared_from_this(), catalog_name, schema_name, table_name);
}

void Relation::Insert(const string &table_name) {
	Insert(INVALID_SCHEMA, table_name);
}

void Relation::Insert(const string &schema_name, const string &table_name) {
	Insert(INVALID_CATALOG, schema_name, table_name);
}

void Relation::Insert(const string &catalog_name, const string &schema_name, const string &table_name) {
	auto insert = InsertRel(catalog_name, schema_name, table_name);
	auto res = insert->Execute();
	if (res->HasError()) {
		const string prepended_message = "Failed to insert into table '" + table_name + "': ";
		res->ThrowError(prepended_message);
	}
}

void Relation::Insert(const vector<vector<Value>> &values) {
	throw InvalidInputException("INSERT with values can only be used on base tables!");
}

void Relation::Insert(vector<vector<unique_ptr<ParsedExpression>>> &&expressions) {
	(void)std::move(expressions);
	throw InvalidInputException("INSERT with expressions can only be used on base tables!");
}

shared_ptr<Relation> Relation::CreateRel(const string &schema_name, const string &table_name, bool temporary,
                                         OnCreateConflict on_conflict) {
	return CreateRel(INVALID_CATALOG, schema_name, table_name, temporary, on_conflict);
}

shared_ptr<Relation> Relation::CreateRel(const string &catalog_name, const string &schema_name,
                                         const string &table_name, bool temporary, OnCreateConflict on_conflict) {
	return make_shared_ptr<CreateTableRelation>(shared_from_this(), catalog_name, schema_name, table_name, temporary,
	                                            on_conflict);
}

void Relation::Create(const string &table_name, bool temporary, OnCreateConflict on_conflict) {
	Create(INVALID_CATALOG, INVALID_SCHEMA, table_name, temporary, on_conflict);
}

void Relation::Create(const string &schema_name, const string &table_name, bool temporary,
                      OnCreateConflict on_conflict) {
	Create(INVALID_CATALOG, schema_name, table_name, temporary, on_conflict);
}

void Relation::Create(const string &catalog_name, const string &schema_name, const string &table_name, bool temporary,
                      OnCreateConflict on_conflict) {
	if (table_name.empty()) {
		throw ParserException("Empty table name not supported");
	}
	auto create = CreateRel(catalog_name, schema_name, table_name, temporary, on_conflict);
	auto res = create->Execute();
	if (res->HasError()) {
		const string prepended_message = "Failed to create table '" + table_name + "': ";
		res->ThrowError(prepended_message);
	}
}

shared_ptr<Relation> Relation::WriteCSVRel(const string &csv_file, case_insensitive_map_t<vector<Value>> options) {
	return make_shared_ptr<duckdb::WriteCSVRelation>(shared_from_this(), csv_file, std::move(options));
}

void Relation::WriteCSV(const string &csv_file, case_insensitive_map_t<vector<Value>> options) {
	auto write_csv = WriteCSVRel(csv_file, std::move(options));
	auto res = write_csv->Execute();
	if (res->HasError()) {
		const string prepended_message = "Failed to write '" + csv_file + "': ";
		res->ThrowError(prepended_message);
	}
}

shared_ptr<Relation> Relation::WriteParquetRel(const string &parquet_file,
                                               case_insensitive_map_t<vector<Value>> options) {
	auto write_parquet =
	    make_shared_ptr<duckdb::WriteParquetRelation>(shared_from_this(), parquet_file, std::move(options));
	return std::move(write_parquet);
}

void Relation::WriteParquet(const string &parquet_file, case_insensitive_map_t<vector<Value>> options) {
	auto write_parquet = WriteParquetRel(parquet_file, std::move(options));
	auto res = write_parquet->Execute();
	if (res->HasError()) {
		const string prepended_message = "Failed to write '" + parquet_file + "': ";
		res->ThrowError(prepended_message);
	}
}

shared_ptr<Relation> Relation::CreateView(const string &name, bool replace, bool temporary) {
	return CreateView(INVALID_SCHEMA, name, replace, temporary);
}

shared_ptr<Relation> Relation::CreateView(const string &schema_name, const string &name, bool replace, bool temporary) {
	auto view = make_shared_ptr<CreateViewRelation>(shared_from_this(), schema_name, name, replace, temporary);
	auto res = view->Execute();
	if (res->HasError()) {
		const string prepended_message = "Failed to create view '" + name + "': ";
		res->ThrowError(prepended_message);
	}
	return shared_from_this();
}

unique_ptr<QueryResult> Relation::Query(const string &sql) const {
	return context->GetContext()->Query(sql, false);
}

unique_ptr<QueryResult> Relation::Query(const string &name, const string &sql) {
	bool replace = true;
	bool temp = IsReadOnly();
	CreateView(name, replace, temp);
	return Query(sql);
}

unique_ptr<QueryResult> Relation::Explain(ExplainType type, ExplainFormat format) {
	auto explain = make_shared_ptr<ExplainRelation>(shared_from_this(), type, format);
	return explain->Execute();
}

void Relation::TryBindRelation(vector<ColumnDefinition> &columns) {
	context->TryBindRelation(*this, columns);
}

void Relation::Update(const string &update, const string &condition) {
	throw InvalidInputException("UPDATE can only be used on base tables!");
}

void Relation::Update(vector<string>, // NOLINT: unused variable / copied on every invocation ...
                      vector<unique_ptr<ParsedExpression>> &&update, // NOLINT: unused variable
                      unique_ptr<ParsedExpression> condition) {      // NOLINT: unused variable
	(void)std::move(update);
	(void)std::move(condition);
	throw InvalidInputException("UPDATE can only be used on base tables!");
}

void Relation::Delete(const string &condition) {
	throw InvalidInputException("DELETE can only be used on base tables!");
}

shared_ptr<Relation> Relation::TableFunction(const std::string &fname, const vector<Value> &values,
                                             const named_parameter_map_t &named_parameters) {
	return make_shared_ptr<TableFunctionRelation>(context->GetContext(), fname, values, named_parameters,
	                                              shared_from_this());
}

shared_ptr<Relation> Relation::TableFunction(const std::string &fname, const vector<Value> &values) {
	return make_shared_ptr<TableFunctionRelation>(context->GetContext(), fname, values, shared_from_this());
}

string Relation::ToString() {
	string str;
	str += "---------------------\n";
	str += "--- Relation Tree ---\n";
	str += "---------------------\n";
	str += ToString(0);
	str += "\n\n";
	str += "---------------------\n";
	str += "-- Result Columns  --\n";
	str += "---------------------\n";
	auto &cols = Columns();
	for (idx_t i = 0; i < cols.size(); i++) {
		str += "- " + cols[i].Name() + " (" + cols[i].Type().ToString() + ")\n";
	}
	return str;
}

// LCOV_EXCL_START
string Relation::GetQuery() {
	return GetQueryNode()->ToString();
}

void Relation::Head(idx_t limit) {
	auto limit_node = Limit(NumericCast<int64_t>(limit));
	limit_node->Execute()->Print();
}
// LCOV_EXCL_STOP

void Relation::Print() {
	Printer::Print(ToString());
}

string Relation::RenderWhitespace(idx_t depth) {
	return string(depth * 2, ' ');
}

void Relation::AddExternalDependency(shared_ptr<ExternalDependency> dependency) {
	external_dependencies.push_back(std::move(dependency));
}

vector<shared_ptr<ExternalDependency>> Relation::GetAllDependencies() {
	vector<shared_ptr<ExternalDependency>> all_dependencies;
	Relation *cur = this;
	while (cur) {
		for (auto &dep : cur->external_dependencies) {
			all_dependencies.push_back(dep);
		}
		cur = cur->ChildRelation();
	}
	return all_dependencies;
}

} // namespace duckdb

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
#include "duckdb/parser/tableref/bound_ref_wrapper.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/query_node/recursive_cte_node.hpp"
#include "duckdb/parser/query_node/set_operation_node.hpp"
#include "duckdb/parser/expression/conjunction_expression.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/expression/subquery_expression.hpp"
#include "duckdb/parser/parsed_expression_iterator.hpp"
#include "duckdb/parser/tableref/list.hpp"
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
	if (!CanSerializeToQueryNode()) {
		throw NotImplementedException("Cannot create a table reference for a relation that cannot be faithfully "
		                              "represented as a SQL query node; conversion would discard the exchange or lose "
		                              "relation bindings");
	}
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
	if (!CanSerializeToQueryNode()) {
		throw NotImplementedException(
		    "Cannot bind a relation that cannot be faithfully represented as a SQL query node; "
		    "conversion would discard the exchange or lose relation bindings");
	}
	SelectStatement stmt;
	stmt.node = GetQueryNode();
	return binder.Bind(stmt.Cast<SQLStatement>());
}

BoundStatement Relation::BindAsInput(Binder &binder) {
	return Bind(binder);
}

bool Relation::RequiresDirectRelationBinding(Relation &child) {
	if (!child.CanSerializeToQueryNode()) {
		return child.CanBindAsInput();
	}
	return child.InheritsColumnBindings() || ExposesMultiSourceBindings(child);
}

bool Relation::ExposesMultiSourceBindings(Relation &child) {
	auto child_ptr = &child;
	while (child_ptr->InheritsColumnBindings()) {
		child_ptr = child_ptr->ChildRelation();
		D_ASSERT(child_ptr);
	}
	return child_ptr->type == RelationType::JOIN_RELATION || child_ptr->type == RelationType::CROSS_PRODUCT_RELATION;
}

bool Relation::RequiresSQLMultiSourceBinding(Relation &child) {
	auto child_ptr = &child;
	// Filters can be merged into the same SELECT without changing relational
	// ordering. Other inheriting relations (LIMIT, DISTINCT, ORDER, etc.) are
	// semantic boundaries and must remain separate plan nodes.
	while (child_ptr->type == RelationType::FILTER_RELATION) {
		child_ptr = child_ptr->ChildRelation();
	}
	return child_ptr->type == RelationType::JOIN_RELATION || child_ptr->type == RelationType::CROSS_PRODUCT_RELATION;
}

namespace {

class SerializedExpressionScopeChecker {
public:
	explicit SerializedExpressionScopeChecker(Relation &child) : child_alias(child.GetAlias()) {
		for (auto &column : child.Columns()) {
			serialized_columns.push_back(column.Name());
		}
		auto source = &child;
		while (source->InheritsColumnBindings()) {
			source = source->ChildRelation();
			D_ASSERT(source);
		}
		CollectRelationBindings(*source, hidden_bindings);
	}

	bool Check(const ParsedExpression &expression) {
		VisitExpression(expression);
		return serializable;
	}

private:
	struct ScopedBinding {
		string catalog;
		string schema;
		string alias;
		vector<string> columns;
	};

	static bool Matches(const string &left, const string &right) {
		return StringUtil::CIEquals(left, right);
	}

	static bool HasColumn(const ScopedBinding &binding, const string &column_name) {
		return std::any_of(binding.columns.begin(), binding.columns.end(),
		                   [&](const string &column) { return Matches(column, column_name); });
	}

	static bool HasAlias(const vector<ScopedBinding> &scope, const string &alias) {
		return std::any_of(scope.begin(), scope.end(),
		                   [&](const ScopedBinding &binding) { return Matches(binding.alias, alias); });
	}

	void CollectRelationBindings(Relation &relation, vector<ScopedBinding> &bindings) {
		switch (relation.type) {
		case RelationType::JOIN_RELATION: {
			auto &join = relation.Cast<JoinRelation>();
			CollectRelationBindings(*join.left, bindings);
			CollectRelationBindings(*join.right, bindings);
			return;
		}
		case RelationType::CROSS_PRODUCT_RELATION: {
			auto &cross_product = relation.Cast<CrossProductRelation>();
			CollectRelationBindings(*cross_product.left, bindings);
			CollectRelationBindings(*cross_product.right, bindings);
			return;
		}
		default:
			break;
		}
		ScopedBinding binding;
		binding.alias = relation.GetAlias();
		for (auto &column : relation.Columns()) {
			binding.columns.push_back(column.Name());
		}
		if (relation.type == RelationType::TABLE_RELATION) {
			binding.columns.emplace_back("rowid");
		} else if (relation.type == RelationType::TABLE_FUNCTION_RELATION) {
			for (auto &column_name : relation.Cast<TableFunctionRelation>().GetVirtualColumnNames()) {
				binding.columns.push_back(column_name);
			}
		}
		bindings.push_back(std::move(binding));
	}

	bool QualifierIsVisible(const string &qualifier) const {
		if (StringUtil::CIEquals(qualifier, child_alias)) {
			return true;
		}
		for (auto scope = query_scopes.rbegin(); scope != query_scopes.rend(); scope++) {
			if (HasAlias(*scope, qualifier)) {
				return true;
			}
		}
		return false;
	}

	bool BindingMatchesHiddenReference(const ScopedBinding &binding, const ColumnRefExpression &column_ref) const {
		auto &names = column_ref.column_names;
		D_ASSERT(names.size() >= 2);
		// The binder tries table.column, schema.table.column, and
		// catalog.schema.table.column before interpreting the leading name as a
		// struct column. Any remaining names are struct fields.
		idx_t max_table_position = MinValue<idx_t>(2, names.size() - 2);
		for (idx_t table_position = 0; table_position <= max_table_position; table_position++) {
			if (Matches(binding.alias, names[table_position]) && HasColumn(binding, names[table_position + 1])) {
				return true;
			}
		}
		return false;
	}

	bool ScopeBindsReference(const vector<ScopedBinding> &scope, const ColumnRefExpression &column_ref) const {
		auto &names = column_ref.column_names;
		D_ASSERT(names.size() >= 2);
		for (auto &binding : scope) {
			// table.column followed by zero or more struct fields
			if (Matches(binding.alias, names[0]) && HasColumn(binding, names[1])) {
				return true;
			}
			// catalog.schema.table.column or schema.table.column
			if (names.size() == 3 && Matches(binding.alias, names[1]) && HasColumn(binding, names[2]) &&
			    ((!binding.catalog.empty() && Matches(binding.catalog, names[0])) ||
			     (!binding.schema.empty() && Matches(binding.schema, names[0])))) {
				return true;
			}
			if (names.size() >= 4 && Matches(binding.catalog, names[0]) && Matches(binding.schema, names[1]) &&
			    Matches(binding.alias, names[2]) && HasColumn(binding, names[3])) {
				return true;
			}
		}
		// If the leading name is a column in this scope, the binder interprets
		// the remaining names as struct fields before trying an outer scope.
		return std::any_of(scope.begin(), scope.end(),
		                   [&](const ScopedBinding &binding) { return HasColumn(binding, names[0]); });
	}

	bool ReferencesHiddenBinding(const ColumnRefExpression &column_ref) const {
		if (!column_ref.IsQualified()) {
			auto &column_name = column_ref.GetColumnName();
			bool hidden_column =
			    std::any_of(hidden_bindings.begin(), hidden_bindings.end(),
			                [&](const ScopedBinding &binding) { return HasColumn(binding, column_name); });
			bool serialized_column = std::any_of(serialized_columns.begin(), serialized_columns.end(),
			                                     [&](const string &column) { return Matches(column, column_name); });
			if (!hidden_column || serialized_column) {
				return false;
			}
			for (auto scope = query_scopes.rbegin(); scope != query_scopes.rend(); scope++) {
				if (std::any_of(scope->begin(), scope->end(),
				                [&](const ScopedBinding &binding) { return HasColumn(binding, column_name); })) {
					return false;
				}
			}
			return true;
		}
		auto &names = column_ref.column_names;
		if (!child_alias.empty() && Matches(child_alias, names[0]) &&
		    std::any_of(serialized_columns.begin(), serialized_columns.end(),
		                [&](const string &column) { return Matches(column, names[1]); })) {
			// A single-source unary boundary is emitted with the child's alias, so
			// its materialized output columns remain qualified by that alias. Hidden
			// virtual columns (e.g., rowid) are intentionally absent here.
			return false;
		}
		bool matches_hidden_binding =
		    std::any_of(hidden_bindings.begin(), hidden_bindings.end(), [&](const ScopedBinding &binding) {
			    return BindingMatchesHiddenReference(binding, column_ref);
		    });
		if (!matches_hidden_binding) {
			return false;
		}
		for (auto scope = query_scopes.rbegin(); scope != query_scopes.rend(); scope++) {
			if (ScopeBindsReference(*scope, column_ref)) {
				return false;
			}
		}
		return true;
	}

	bool HiddenQualifier(const string &qualifier) const {
		return std::any_of(hidden_bindings.begin(), hidden_bindings.end(),
		                   [&](const ScopedBinding &binding) { return Matches(binding.alias, qualifier); });
	}

	void VisitTableRefQuery(QueryNode &node) {
		// A FROM item's alias belongs to the containing SELECT and is not visible
		// inside the query that defines that item. Drop only the current SELECT's
		// scope while descending, retaining any genuinely correlated outer scopes.
		D_ASSERT(!query_scopes.empty());
		auto current_scope = std::move(query_scopes.back());
		query_scopes.pop_back();
		VisitQueryNode(node);
		query_scopes.push_back(std::move(current_scope));
	}

	void VisitTableRefExpression(const ParsedExpression &expression) {
		D_ASSERT(!query_scopes.empty());
		auto current_scope = std::move(query_scopes.back());
		query_scopes.pop_back();
		VisitExpression(expression);
		query_scopes.push_back(std::move(current_scope));
	}

	void VisitExpression(const ParsedExpression &expression) {
		if (!serializable) {
			return;
		}
		switch (expression.GetExpressionClass()) {
		case ExpressionClass::COLUMN_REF: {
			auto &column_ref = expression.Cast<ColumnRefExpression>();
			if (ReferencesHiddenBinding(column_ref)) {
				serializable = false;
			}
			return;
		}
		case ExpressionClass::STAR: {
			auto &star = expression.Cast<StarExpression>();
			auto references_hidden_table = [&](const QualifiedColumnName &column) {
				if (!column.IsQualified()) {
					return false;
				}
				ColumnRefExpression column_ref(column.column, column.table);
				return ReferencesHiddenBinding(column_ref);
			};
			if ((!star.relation_name.empty() && HiddenQualifier(star.relation_name) &&
			     !QualifierIsVisible(star.relation_name)) ||
			    std::any_of(star.exclude_list.begin(), star.exclude_list.end(), references_hidden_table) ||
			    std::any_of(star.rename_list.begin(), star.rename_list.end(),
			                [&](const auto &entry) { return references_hidden_table(entry.first); })) {
				serializable = false;
				return;
			}
			break;
		}
		case ExpressionClass::SUBQUERY: {
			auto &subquery = expression.Cast<SubqueryExpression>();
			if (subquery.child) {
				VisitExpression(*subquery.child);
			}
			if (!subquery.subquery || !subquery.subquery->node) {
				serializable = false;
				return;
			}
			VisitQueryNode(*subquery.subquery->node);
			return;
		}
		default:
			break;
		}
		ParsedExpressionIterator::EnumerateChildren(expression,
		                                            [&](const ParsedExpression &child) { VisitExpression(child); });
	}

	bool CollectQueryOutputColumns(const QueryNode &node, vector<string> &columns) const {
		switch (node.type) {
		case QueryNodeType::SELECT_NODE: {
			auto &select = node.Cast<SelectNode>();
			for (auto &expression : select.select_list) {
				if (expression->GetExpressionClass() == ExpressionClass::STAR) {
					return false;
				}
				columns.push_back(expression->GetName());
			}
			return true;
		}
		case QueryNodeType::SET_OPERATION_NODE: {
			auto &set_operation = node.Cast<SetOperationNode>();
			if (set_operation.children.empty()) {
				return false;
			}
			return CollectQueryOutputColumns(*set_operation.children[0], columns);
		}
		case QueryNodeType::RECURSIVE_CTE_NODE:
			return CollectQueryOutputColumns(*node.Cast<RecursiveCTENode>().left, columns);
		default:
			return false;
		}
	}

	void CollectTableBindings(const TableRef &ref, vector<ScopedBinding> &bindings) {
		if (!ref.alias.empty()) {
			ScopedBinding binding;
			binding.alias = ref.alias;
			if (ref.type == TableReferenceType::SUBQUERY) {
				CollectQueryOutputColumns(*ref.Cast<SubqueryRef>().subquery->node, binding.columns);
			}
			for (idx_t i = 0; i < ref.column_name_alias.size(); i++) {
				if (i < binding.columns.size()) {
					binding.columns[i] = ref.column_name_alias[i];
				} else {
					binding.columns.push_back(ref.column_name_alias[i]);
				}
			}
			bindings.push_back(std::move(binding));
			return;
		}
		switch (ref.type) {
		case TableReferenceType::BASE_TABLE: {
			auto &base_table = ref.Cast<BaseTableRef>();
			ScopedBinding binding;
			binding.catalog = base_table.catalog_name;
			binding.schema = base_table.schema_name;
			binding.alias = base_table.table_name;
			binding.columns = ref.column_name_alias;
			bindings.push_back(std::move(binding));
			break;
		}
		case TableReferenceType::JOIN: {
			auto &join = ref.Cast<JoinRef>();
			CollectTableBindings(*join.left, bindings);
			CollectTableBindings(*join.right, bindings);
			break;
		}
		case TableReferenceType::TABLE_FUNCTION: {
			auto &table_function = ref.Cast<TableFunctionRef>();
			if (table_function.function && table_function.function->GetExpressionClass() == ExpressionClass::FUNCTION) {
				ScopedBinding binding;
				binding.alias = table_function.function->Cast<FunctionExpression>().function_name;
				binding.columns = ref.column_name_alias;
				bindings.push_back(std::move(binding));
			}
			break;
		}
		case TableReferenceType::PIVOT:
			// PIVOT/UNPIVOT creates a new output binding. Its source aliases are
			// available to the pivot expressions, but not to the containing SELECT.
			break;
		default:
			break;
		}
	}

	void VisitTableRef(TableRef &ref) {
		if (!serializable) {
			return;
		}
		switch (ref.type) {
		case TableReferenceType::EXPRESSION_LIST: {
			auto &expression_list = ref.Cast<ExpressionListRef>();
			for (auto &row : expression_list.values) {
				for (auto &expression : row) {
					VisitTableRefExpression(*expression);
				}
			}
			break;
		}
		case TableReferenceType::JOIN: {
			auto &join = ref.Cast<JoinRef>();
			VisitTableRef(*join.left);
			VisitTableRef(*join.right);
			if (join.condition) {
				VisitExpression(*join.condition);
			}
			for (auto &expression : join.duplicate_eliminated_columns) {
				VisitExpression(*expression);
			}
			break;
		}
		case TableReferenceType::PIVOT: {
			auto &pivot = ref.Cast<PivotRef>();
			VisitTableRef(*pivot.source);
			for (auto &aggregate : pivot.aggregates) {
				VisitTableRefExpression(*aggregate);
			}
			for (auto &column : pivot.pivots) {
				for (auto &expression : column.pivot_expressions) {
					VisitTableRefExpression(*expression);
				}
				for (auto &entry : column.entries) {
					if (entry.expr) {
						VisitTableRefExpression(*entry.expr);
					}
				}
				if (column.subquery) {
					VisitTableRefQuery(*column.subquery);
				}
			}
			break;
		}
		case TableReferenceType::SUBQUERY: {
			auto &subquery = ref.Cast<SubqueryRef>();
			VisitTableRefQuery(*subquery.subquery->node);
			break;
		}
		case TableReferenceType::TABLE_FUNCTION: {
			auto &table_function = ref.Cast<TableFunctionRef>();
			VisitTableRefExpression(*table_function.function);
			if (table_function.subquery) {
				VisitTableRefQuery(*table_function.subquery->node);
			}
			break;
		}
		case TableReferenceType::SHOW_REF: {
			auto &show = ref.Cast<ShowRef>();
			if (show.query) {
				VisitTableRefQuery(*show.query);
			}
			break;
		}
		case TableReferenceType::BASE_TABLE:
		case TableReferenceType::EMPTY_FROM:
		case TableReferenceType::COLUMN_DATA:
		case TableReferenceType::DELIM_GET:
			break;
		default:
			serializable = false;
			break;
		}
	}

	void VisitQueryNode(QueryNode &node) {
		if (!serializable) {
			return;
		}
		for (auto &entry : node.cte_map.map) {
			VisitQueryNode(*entry.second->query->node);
		}
		switch (node.type) {
		case QueryNodeType::SELECT_NODE: {
			auto &select = node.Cast<SelectNode>();
			vector<ScopedBinding> bindings;
			if (select.from_table) {
				CollectTableBindings(*select.from_table, bindings);
			}
			query_scopes.push_back(std::move(bindings));
			for (auto &expression : select.select_list) {
				VisitExpression(*expression);
			}
			for (auto &expression : select.groups.group_expressions) {
				VisitExpression(*expression);
			}
			if (select.where_clause) {
				VisitExpression(*select.where_clause);
			}
			if (select.having) {
				VisitExpression(*select.having);
			}
			if (select.qualify) {
				VisitExpression(*select.qualify);
			}
			if (select.from_table) {
				VisitTableRef(*select.from_table);
			}
			ParsedExpressionIterator::EnumerateQueryNodeModifiers(
			    node, [&](unique_ptr<ParsedExpression> &expression) { VisitExpression(*expression); });
			query_scopes.pop_back();
			break;
		}
		case QueryNodeType::SET_OPERATION_NODE: {
			auto &set_operation = node.Cast<SetOperationNode>();
			for (auto &child : set_operation.children) {
				VisitQueryNode(*child);
			}
			ParsedExpressionIterator::EnumerateQueryNodeModifiers(
			    node, [&](unique_ptr<ParsedExpression> &expression) { VisitExpression(*expression); });
			break;
		}
		case QueryNodeType::RECURSIVE_CTE_NODE: {
			auto &recursive_cte = node.Cast<RecursiveCTENode>();
			VisitQueryNode(*recursive_cte.left);
			VisitQueryNode(*recursive_cte.right);
			for (auto &target : recursive_cte.key_targets) {
				VisitExpression(*target);
			}
			ParsedExpressionIterator::EnumerateQueryNodeModifiers(
			    node, [&](unique_ptr<ParsedExpression> &expression) { VisitExpression(*expression); });
			break;
		}
		default:
			serializable = false;
			break;
		}
	}

private:
	string child_alias;
	vector<string> serialized_columns;
	vector<ScopedBinding> hidden_bindings;
	vector<vector<ScopedBinding>> query_scopes;
	bool serializable = true;
};

} // namespace

bool Relation::CanSerializeExpressionOnChild(Relation &child, const ParsedExpression &expression) {
	if (!child.CanSerializeToQueryNode()) {
		return false;
	}
	if (!child.InheritsColumnBindings() || RequiresSQLMultiSourceBinding(child)) {
		return true;
	}

	// An inheriting semantic boundary is emitted as a subquery. Check references
	// against the columns and aliases that survive that boundary, including
	// aliases introduced by nested query scopes.
	return SerializedExpressionScopeChecker(child).Check(expression);
}

unique_ptr<TableRef> Relation::BindRelationInput(Binder &binder, Relation &child) {
	// BoundRefWrapper lets the normal SELECT binder consume a non-SQL child
	// without converting its logical plan back into a QueryNode. Keep the
	// child's full BindContext for binding-preserving relations and native sources.
	auto child_binder = Binder::CreateBinder(binder.context, binder.shared_from_this());
	auto child_bound = child.BindAsInput(*child_binder);
	binder.MoveCorrelatedExpressionsFrom(*child_binder);

	bool multi_source = child.type == RelationType::JOIN_RELATION || child.type == RelationType::CROSS_PRODUCT_RELATION;
	bool native_source =
	    child.type == RelationType::TABLE_RELATION || child.type == RelationType::TABLE_FUNCTION_RELATION;
	bool preserves_bindings = child.InheritsColumnBindings() || multi_source || native_source;
	if (preserves_bindings) {
		// The bind context is authoritative for the columns that survive an
		// operator. JoinRef binding retains both inputs in BoundStatement metadata
		// even when SEMI/ANTI joins remove one side from the output.
		child_bound.names.clear();
		child_bound.types.clear();
		child_binder->bind_context.GetTypesAndNames(child_bound.names, child_bound.types);
	} else {
		// Projections, aggregates, and explicit aliases establish a new scope.
		// Rebase that scope on the bound plan's single output table index.
		auto input_binder = Binder::CreateBinder(binder.context, binder.shared_from_this());
		auto names = BindContext::AliasColumnNames(child.GetAlias(), child_bound.names, {});
		input_binder->bind_context.AddGenericBinding(child_bound.plan->GetRootIndex(), child.GetAlias(), names,
		                                             child_bound.types);
		child_binder = std::move(input_binder);
	}
	return make_uniq<BoundRefWrapper>(std::move(child_bound), std::move(child_binder));
}

BoundStatement Relation::BindSelectNodeOnChild(Binder &binder, Relation &child, unique_ptr<SelectNode> select_node) {
	select_node->from_table = BindRelationInput(binder, child);
	SelectStatement stmt;
	stmt.node = std::move(select_node);
	return binder.Bind(stmt.Cast<SQLStatement>());
}

unique_ptr<LogicalOperator> Relation::PlanRelationFilter(Binder &binder, unique_ptr<Expression> condition,
                                                         unique_ptr<LogicalOperator> child) {
	return binder.PlanFilter(std::move(condition), std::move(child));
}

void Relation::ExpandRelationFilter(Binder &binder, unique_ptr<ParsedExpression> &condition) {
	binder.BindWhereStarExpression(condition);
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
	if (!CanSerializeToQueryNode()) {
		return string();
	}
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

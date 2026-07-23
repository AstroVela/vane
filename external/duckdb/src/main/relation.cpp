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
#include "duckdb/planner/expression/bound_subquery_expression.hpp"
#include "duckdb/planner/expression_binder/relation_binder.hpp"
#include "duckdb/planner/expression_binder/select_binder.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/query_node/bound_select_node.hpp"
#include "duckdb/parser/tableref/bound_ref_wrapper.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/expression/conjunction_expression.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/expression/subquery_expression.hpp"
#include "duckdb/parser/parsed_expression_iterator.hpp"
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
	unique_ptr<TableRef> result;
	auto client_context = context->GetContext();
	client_context->RunFunctionInTransaction([&]() {
		auto binder = Binder::CreateBinder(*client_context);
		if (!CanSerializeToQueryNodeInternal(*binder)) {
			throw NotImplementedException(
			    "Cannot create a table reference for a relation that cannot be faithfully "
			    "represented as a SQL query node; conversion would discard the exchange or lose "
			    "relation bindings");
		}
		result = GetTableRefInternal();
	});
	return result;
}

unique_ptr<TableRef> Relation::GetTableRefInternal() {
	auto select = make_uniq<SelectStatement>();
	select->node = GetQueryNode();
	return make_uniq<SubqueryRef>(std::move(select), GetAlias());
}

unique_ptr<TableRef> Relation::GetTableRefForSerialization(Relation &relation) {
	return relation.GetTableRefInternal();
}

bool Relation::CanSerializeToQueryNode() {
	bool result = false;
	auto client_context = context->GetContext();
	client_context->RunFunctionInTransaction([&]() {
		auto binder = Binder::CreateBinder(*client_context);
		result = CanSerializeToQueryNodeInternal(*binder);
	});
	return result;
}

unique_ptr<QueryNode> Relation::TryGetSerializableQueryNode() {
	unique_ptr<QueryNode> result;
	auto client_context = context->GetContext();
	client_context->RunFunctionInTransaction([&]() {
		auto binder = Binder::CreateBinder(*client_context);
		result = TryGetSerializableQueryNode(*binder);
	});
	return result;
}

unique_ptr<QueryNode> Relation::TryGetSerializableQueryNode(Binder &binder) {
	if (!CanSerializeToQueryNodeInternal(binder)) {
		return nullptr;
	}
	return GetQueryNode();
}

bool Relation::CanBindAsInput() {
	bool result = false;
	auto client_context = context->GetContext();
	client_context->RunFunctionInTransaction([&]() {
		auto binder = Binder::CreateBinder(*client_context);
		result = CanBindAsInputInternal(*binder);
	});
	return result;
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
	auto query_node = TryGetSerializableQueryNode(binder);
	if (!query_node) {
		throw NotImplementedException(
		    "Cannot bind a relation that cannot be faithfully represented as a SQL query node; "
		    "conversion would discard the exchange or lose relation bindings");
	}
	SelectStatement stmt;
	stmt.node = std::move(query_node);
	return binder.Bind(stmt.Cast<SQLStatement>());
}

BoundStatement Relation::BindAsInput(Binder &binder) {
	return Bind(binder);
}

bool Relation::RequiresDirectRelationBinding(Binder &binder, Relation &child) {
	if (!child.CanSerializeToQueryNodeInternal(binder)) {
		return child.CanBindAsInputInternal(binder);
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
	SerializedExpressionScopeChecker(Binder &binder, Relation &child, BoundRefWrapper &bound_child)
	    : child_alias(child.GetAlias()), serialization_parent_binder(binder), bound_child_binder(*bound_child.binder) {
		for (auto &column : child.Columns()) {
			serialized_columns.push_back(column.Name());
			serialized_types.push_back(column.Type());
		}
		serialized_columns = BindContext::AliasColumnNames(child_alias, serialized_columns, {});
		auto source = &child;
		while (source->InheritsColumnBindings()) {
			source = source->ChildRelation();
			D_ASSERT(source);
		}
		CollectRelationBindings(*source, hidden_bindings);
	}

	bool Check(const ParsedExpression &expression) {
		VisitExpression(expression);
		return serializable && BoundExpressionBindingsMatch(expression);
	}

private:
	struct ResolvedCorrelation {
		ColumnBinding binding;
		idx_t depth;

		bool operator==(const ResolvedCorrelation &other) const {
			return binding == other.binding && depth == other.depth;
		}
	};

	struct ScopedBinding {
		string alias;
		vector<string> columns;
	};

	static void AddCorrelation(vector<ResolvedCorrelation> &correlations, ColumnBinding binding, idx_t depth) {
		ResolvedCorrelation correlation {binding, depth};
		if (std::find(correlations.begin(), correlations.end(), correlation) == correlations.end()) {
			correlations.push_back(std::move(correlation));
		}
	}

	static void CollectColumnBindings(const Expression &expression, vector<ColumnBinding> &bindings) {
		ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
		    expression, [&](const BoundColumnRefExpression &column_ref) {
			    if (std::find(bindings.begin(), bindings.end(), column_ref.binding) == bindings.end()) {
				    bindings.push_back(column_ref.binding);
			    }
		    });
	}

	static void InitializeSelectNode(Binder &binder, BoundSelectNode &node) {
		node.projection_index = binder.GenerateTableIndex();
		node.group_index = binder.GenerateTableIndex();
		node.aggregate_index = binder.GenerateTableIndex();
		node.groupings_index = binder.GenerateTableIndex();
		node.window_index = binder.GenerateTableIndex();
		node.prune_index = binder.GenerateTableIndex();
	}

	static bool CollectSyntheticBinding(const BoundColumnRefExpression &column_ref, const BoundSelectNode &node,
	                                    vector<ResolvedCorrelation> &bindings) {
		auto &binding = column_ref.binding;
		if (binding.table_index == node.aggregate_index) {
			if (binding.column_index >= node.aggregates.size()) {
				throw InternalException("Aggregate binding is out of range while checking relation serialization");
			}
			CollectBoundExpressionBindings(*node.aggregates[binding.column_index], node, bindings);
			return true;
		}
		if (binding.table_index == node.window_index) {
			if (binding.column_index >= node.windows.size()) {
				throw InternalException("Window binding is out of range while checking relation serialization");
			}
			CollectBoundExpressionBindings(*node.windows[binding.column_index], node, bindings);
			return true;
		}
		if (binding.table_index == node.group_index) {
			if (binding.column_index >= node.groups.group_expressions.size()) {
				throw InternalException("Group binding is out of range while checking relation serialization");
			}
			CollectBoundExpressionBindings(*node.groups.group_expressions[binding.column_index], node, bindings);
			return true;
		}
		if (binding.table_index == node.groupings_index) {
			for (auto &group : node.groups.group_expressions) {
				CollectBoundExpressionBindings(*group, node, bindings);
			}
			return true;
		}
		for (auto &entry : node.unnests) {
			auto &unnest = entry.second;
			if (binding.table_index != unnest.index) {
				continue;
			}
			if (binding.column_index >= unnest.expressions.size()) {
				throw InternalException("Unnest binding is out of range while checking relation serialization");
			}
			CollectBoundExpressionBindings(*unnest.expressions[binding.column_index], node, bindings);
			return true;
		}
		return false;
	}

	static void CollectBoundExpressionBindings(const Expression &expression, const BoundSelectNode &node,
	                                           vector<ResolvedCorrelation> &bindings) {
		if (expression.GetExpressionClass() == ExpressionClass::BOUND_SUBQUERY) {
			auto &subquery = expression.Cast<BoundSubqueryExpression>();
			for (auto &correlation : subquery.binder->correlated_columns) {
				AddCorrelation(bindings, correlation.binding, correlation.depth);
			}
		}
		if (expression.type == ExpressionType::BOUND_COLUMN_REF) {
			auto &column_ref = expression.Cast<BoundColumnRefExpression>();
			if (!CollectSyntheticBinding(column_ref, node, bindings)) {
				AddCorrelation(bindings, column_ref.binding, column_ref.depth);
			}
			return;
		}
		ExpressionIterator::EnumerateChildren(
		    expression, [&](const Expression &child) { CollectBoundExpressionBindings(child, node, bindings); });
	}

	static bool TryBindExpression(Binder &binder, const ParsedExpression &expression,
	                              vector<ResolvedCorrelation> &bindings, unique_ptr<ParsedExpression> &bound_copy) {
		bound_copy = expression.Copy();
		BoundSelectNode node;
		InitializeSelectNode(binder, node);
		BoundGroupInformation group_info;
		try {
			SelectBinder expression_binder(binder, binder.context, node, group_info);
			auto bound_expression = expression_binder.Bind(bound_copy);
			CollectBoundExpressionBindings(*bound_expression, node, bindings);
			return true;
		} catch (Exception &ex) {
			ErrorData error(ex);
			if (error.Type() != ExceptionType::BINDER) {
				throw;
			}
			return false;
		}
	}

	bool NormalizeSerializedBindings(const vector<ResolvedCorrelation> &bindings,
	                                 vector<ResolvedCorrelation> &normalized) const {
		for (auto &binding : bindings) {
			if (binding.binding.table_index != serialized_table_index) {
				AddCorrelation(normalized, binding.binding, binding.depth);
				continue;
			}
			if (binding.binding.column_index >= serialized_output_origins.size()) {
				return false;
			}
			for (auto &origin : serialized_output_origins[binding.binding.column_index]) {
				AddCorrelation(normalized, origin, binding.depth);
			}
		}
		return true;
	}

	static bool BindingsMatch(const vector<ResolvedCorrelation> &left, const vector<ResolvedCorrelation> &right) {
		if (left.size() != right.size()) {
			return false;
		}
		return std::all_of(left.begin(), left.end(), [&](const auto &binding) {
			return std::find(right.begin(), right.end(), binding) != right.end();
		});
	}

	bool BoundExpressionBindingsMatch(const ParsedExpression &expression) {
		if (!InitializeSubqueryScopes()) {
			return false;
		}

		vector<ResolvedCorrelation> direct_bindings;
		unique_ptr<ParsedExpression> direct_copy;
		auto direct_bound = TryBindExpression(bound_child_binder, expression, direct_bindings, direct_copy);
		vector<ResolvedCorrelation> serialized_bindings;
		unique_ptr<ParsedExpression> serialized_copy;
		auto serialized_bound =
		    TryBindExpression(*serialized_scope_binder, expression, serialized_bindings, serialized_copy);

		if (!direct_bound) {
			// Binding can expand a macro before rejecting an expression class that is
			// not valid in a synthetic SELECT. Inspect that expanded tree so hidden
			// qualifiers introduced by the macro are still detected.
			VisitExpression(*direct_copy);
			if (serialized_bound) {
				return false;
			}
			return serializable;
		}
		if (!serialized_bound) {
			return false;
		}

		vector<ResolvedCorrelation> normalized_bindings;
		if (!NormalizeSerializedBindings(serialized_bindings, normalized_bindings)) {
			return false;
		}
		return BindingsMatch(direct_bindings, normalized_bindings);
	}

	bool InitializeSubqueryScopes() {
		if (serialized_scope_binder) {
			return serializable;
		}
		serialized_scope_binder =
		    Binder::CreateBinder(serialization_parent_binder.context, &serialization_parent_binder);
		serialized_table_index = serialized_scope_binder->GenerateTableIndex();
		serialized_scope_binder->bind_context.AddGenericBinding(serialized_table_index, child_alias, serialized_columns,
		                                                        serialized_types);
		ResolveSerializedOutputOrigins();
		return serializable;
	}

	void ResolveSerializedOutputOrigins() {
		vector<unique_ptr<ParsedExpression>> output_columns;
		StarExpression star;
		bound_child_binder.bind_context.GenerateAllColumnExpressions(star, output_columns);
		if (output_columns.size() != serialized_columns.size()) {
			serializable = false;
			return;
		}

		RelationBinder relation_binder(bound_child_binder, bound_child_binder.context, "relation serialization");
		serialized_output_origins.reserve(output_columns.size());
		for (auto &column : output_columns) {
			auto bound_column = relation_binder.Bind(column);
			vector<ColumnBinding> origins;
			CollectColumnBindings(*bound_column, origins);
			if (origins.empty()) {
				serializable = false;
				return;
			}
			serialized_output_origins.push_back(std::move(origins));
		}
	}

	bool TryBindSubquery(const QueryNode &node, Binder &parent, vector<ResolvedCorrelation> &correlations) const {
		try {
			auto query = node.Copy();
			// Correlated lookup walks the active expression binders rather than
			// Binder parent contexts alone.
			RelationBinder outer_binder(parent, parent.context, "relation serialization");
			auto query_binder = Binder::CreateBinder(parent.context, &parent);
			query_binder->Bind(*query);
			for (auto &correlation : query_binder->correlated_columns) {
				AddCorrelation(correlations, correlation.binding, correlation.depth);
			}
			return true;
		} catch (Exception &ex) {
			ErrorData error(ex);
			if (error.Type() != ExceptionType::BINDER) {
				throw;
			}
			return false;
		}
	}

	bool SubqueryBindingsMatch(const QueryNode &node) {
		if (!InitializeSubqueryScopes()) {
			return false;
		}
		vector<ResolvedCorrelation> direct_correlations;
		if (!TryBindSubquery(node, bound_child_binder, direct_correlations)) {
			return false;
		}
		vector<ResolvedCorrelation> serialized_correlations;
		if (!TryBindSubquery(node, *serialized_scope_binder, serialized_correlations)) {
			return false;
		}

		vector<ResolvedCorrelation> normalized_correlations;
		for (auto &correlation : serialized_correlations) {
			if (correlation.binding.table_index != serialized_table_index) {
				AddCorrelation(normalized_correlations, correlation.binding, correlation.depth);
				continue;
			}
			if (correlation.binding.column_index >= serialized_output_origins.size()) {
				return false;
			}
			for (auto &origin : serialized_output_origins[correlation.binding.column_index]) {
				AddCorrelation(normalized_correlations, origin, correlation.depth);
			}
		}
		if (direct_correlations.size() != normalized_correlations.size()) {
			return false;
		}
		return std::all_of(direct_correlations.begin(), direct_correlations.end(), [&](const auto &correlation) {
			return std::find(normalized_correlations.begin(), normalized_correlations.end(), correlation) !=
			       normalized_correlations.end();
		});
	}

	static bool Matches(const string &left, const string &right) {
		return StringUtil::CIEquals(left, right);
	}

	static bool HasColumn(const ScopedBinding &binding, const string &column_name) {
		return std::any_of(binding.columns.begin(), binding.columns.end(),
		                   [&](const string &column) { return Matches(column, column_name); });
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
		binding.columns = BindContext::AliasColumnNames(binding.alias, binding.columns, {});
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
		return StringUtil::CIEquals(qualifier, child_alias);
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
		return true;
	}

	bool HiddenQualifier(const string &qualifier) const {
		return std::any_of(hidden_bindings.begin(), hidden_bindings.end(),
		                   [&](const ScopedBinding &binding) { return Matches(binding.alias, qualifier); });
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
			if (!SubqueryBindingsMatch(*subquery.subquery->node)) {
				serializable = false;
			}
			return;
		}
		default:
			break;
		}
		ParsedExpressionIterator::EnumerateChildren(expression,
		                                            [&](const ParsedExpression &child) { VisitExpression(child); });
	}

private:
	string child_alias;
	vector<string> serialized_columns;
	vector<LogicalType> serialized_types;
	vector<ScopedBinding> hidden_bindings;
	Binder &serialization_parent_binder;
	Binder &bound_child_binder;
	shared_ptr<Binder> serialized_scope_binder;
	idx_t serialized_table_index;
	vector<vector<ColumnBinding>> serialized_output_origins;
	bool serializable = true;
};

} // namespace

bool Relation::CanSerializeExpressionOnBoundChild(Binder &binder, Relation &child, TableRef &bound_child,
                                                  const ParsedExpression &expression) {
	if (!child.InheritsColumnBindings() || RequiresSQLMultiSourceBinding(child)) {
		return true;
	}
	if (bound_child.type != TableReferenceType::BOUND_TABLE_REF) {
		throw InternalException("Expected a bound relation input while checking expression serialization");
	}
	auto &bound_ref = bound_child.Cast<BoundRefWrapper>();
	if (!bound_ref.binder) {
		throw InternalException("Bound relation input is missing its binder");
	}

	// An inheriting semantic boundary is emitted as a subquery. Check references
	// against the columns and aliases that survive that boundary. Nested query
	// scopes are resolved by the Binder in both the direct and serialized inputs,
	// then compared by their underlying column bindings.
	return SerializedExpressionScopeChecker(binder, child, bound_ref).Check(expression);
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

unique_ptr<SelectNode> Relation::WrapQueryNode(unique_ptr<QueryNode> query_node, const string &alias,
                                               const vector<ColumnDefinition> &columns) {
	auto statement = make_uniq<SelectStatement>();
	statement->node = std::move(query_node);
	auto result = make_uniq<SelectNode>();
	result->from_table = make_uniq<SubqueryRef>(std::move(statement), alias);
	vector<string> output_names;
	output_names.reserve(columns.size());
	for (auto &column : columns) {
		output_names.push_back(column.Name());
	}
	auto input_names = BindContext::AliasColumnNames(alias, output_names, {});
	for (idx_t column_idx = 0; column_idx < output_names.size(); column_idx++) {
		unique_ptr<ColumnRefExpression> column_ref;
		if (alias.empty()) {
			column_ref = make_uniq<ColumnRefExpression>(input_names[column_idx]);
		} else {
			column_ref = make_uniq<ColumnRefExpression>(input_names[column_idx], alias);
		}
		column_ref->SetAlias(output_names[column_idx]);
		result->select_list.push_back(std::move(column_ref));
	}
	if (result->select_list.empty()) {
		result->select_list.push_back(make_uniq<StarExpression>());
	}
	return result;
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
	auto query_node = TryGetSerializableQueryNode();
	if (!query_node) {
		return string();
	}
	return query_node->ToString();
}

string Relation::GetQuery(Binder &binder) {
	auto query_node = TryGetSerializableQueryNode(binder);
	if (!query_node) {
		return string();
	}
	return query_node->ToString();
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

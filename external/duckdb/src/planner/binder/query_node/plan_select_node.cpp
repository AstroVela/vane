#include "duckdb/planner/binder.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/planner/bound_result_modifier.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/list.hpp"
#include "duckdb/planner/operator/logical_dummy_scan.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"
#include "duckdb/planner/query_node/bound_select_node.hpp"

namespace duckdb {

static void InlineOrderSelectList(unique_ptr<Expression> &expression, idx_t projection_index,
                                  const vector<unique_ptr<Expression>> &select_list) {
	if (expression->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &column_ref = expression->Cast<BoundColumnRefExpression>();
		if (column_ref.binding.table_index == projection_index) {
			if (column_ref.binding.column_index >= select_list.size()) {
				throw InternalException(
				    "ORDER BY select-list reference is out of range while preserving input bindings");
			}
			expression = select_list[column_ref.binding.column_index]->Copy();
			return;
		}
	}
	ExpressionIterator::EnumerateChildren(*expression, [&](unique_ptr<Expression> &child) {
		InlineOrderSelectList(child, projection_index, select_list);
	});
}

static bool TryGetOrderSelectListIndex(const Expression &expression, idx_t projection_index, idx_t &column_index) {
	if (expression.type != ExpressionType::BOUND_COLUMN_REF) {
		return false;
	}
	auto &column_ref = expression.Cast<BoundColumnRefExpression>();
	if (column_ref.binding.table_index != projection_index) {
		return false;
	}
	column_index = column_ref.binding.column_index;
	return true;
}

static void RewriteOrderForInputBindings(BoundSelectNode &statement) {
	if (statement.modifiers.empty()) {
		return;
	}
	if (statement.modifiers.size() != 1 || statement.modifiers[0]->type != ResultModifierType::ORDER_MODIFIER) {
		throw InternalException("Expected only an ORDER BY modifier while preserving relation input bindings");
	}
	auto &bound_order = statement.modifiers[0]->Cast<BoundOrderModifier>();
	unordered_set<idx_t> seen_projection_columns;
	vector<BoundOrderByNode> rewritten_orders;
	rewritten_orders.reserve(bound_order.orders.size());
	for (auto &order : bound_order.orders) {
		idx_t projection_column;
		if (TryGetOrderSelectListIndex(*order.expression, statement.projection_index, projection_column) &&
		    !seen_projection_columns.insert(projection_column).second) {
			// The SELECT binder shares identical ORDER BY expressions through one
			// projection slot. A repeated sort key is redundant; dropping it keeps
			// volatile expressions at one evaluation per input row.
			continue;
		}
		InlineOrderSelectList(order.expression, statement.projection_index, statement.select_list);
		rewritten_orders.push_back(std::move(order));
	}
	bound_order.orders = std::move(rewritten_orders);
}

unique_ptr<LogicalOperator> Binder::PlanFilter(unique_ptr<Expression> condition, unique_ptr<LogicalOperator> root) {
	PlanSubqueries(condition, root);
	auto filter = make_uniq<LogicalFilter>(std::move(condition));
	filter->AddChild(std::move(root));
	return std::move(filter);
}

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundSelectNode &statement, SelectPlanMode mode) {
	D_ASSERT(statement.from_table.plan);
	auto root = std::move(statement.from_table.plan);

	// plan the sample clause
	if (statement.sample_options) {
		root = make_uniq<LogicalSample>(std::move(statement.sample_options), std::move(root));
	}

	if (statement.where_clause) {
		root = PlanFilter(std::move(statement.where_clause), std::move(root));
	}

	if (!statement.aggregates.empty() || !statement.groups.group_expressions.empty() || statement.having) {
		if (!statement.groups.group_expressions.empty()) {
			// visit the groups
			for (auto &group : statement.groups.group_expressions) {
				PlanSubqueries(group, root);
			}
		}
		// now visit all aggregate expressions
		for (auto &expr : statement.aggregates) {
			PlanSubqueries(expr, root);
		}
		// finally create the aggregate node with the group_index and aggregate_index as obtained from the binder
		auto aggregate = make_uniq<LogicalAggregate>(statement.group_index, statement.aggregate_index,
		                                             std::move(statement.aggregates));
		aggregate->groups = std::move(statement.groups.group_expressions);
		aggregate->groupings_index = statement.groupings_index;
		aggregate->grouping_sets = std::move(statement.groups.grouping_sets);
		aggregate->grouping_functions = std::move(statement.grouping_functions);

		aggregate->AddChild(std::move(root));
		root = std::move(aggregate);
	} else if (!statement.groups.grouping_sets.empty()) {
		// edge case: we have grouping sets but no groups or aggregates
		// this can only happen if we have e.g. select 1 from tbl group by ();
		// just output a dummy scan
		root = make_uniq_base<LogicalOperator, LogicalDummyScan>(statement.group_index);
	}

	if (statement.having) {
		PlanSubqueries(statement.having, root);
		auto having = make_uniq<LogicalFilter>(std::move(statement.having));

		having->AddChild(std::move(root));
		root = std::move(having);
	}

	if (!statement.windows.empty()) {
		auto win = make_uniq<LogicalWindow>(statement.window_index);
		win->expressions = std::move(statement.windows);
		// visit the window expressions
		for (auto &expr : win->expressions) {
			PlanSubqueries(expr, root);
		}
		D_ASSERT(!win->expressions.empty());
		win->AddChild(std::move(root));
		root = std::move(win);
	}

	if (statement.qualify) {
		PlanSubqueries(statement.qualify, root);
		auto qualify = make_uniq<LogicalFilter>(std::move(statement.qualify));

		qualify->AddChild(std::move(root));
		root = std::move(qualify);
	}

	for (idx_t i = statement.unnests.size(); i > 0; i--) {
		auto unnest_level = i - 1;
		auto entry = statement.unnests.find(unnest_level);
		if (entry == statement.unnests.end()) {
			throw InternalException("unnests specified at level %d but none were found", unnest_level);
		}
		auto &unnest_node = entry->second;
		auto unnest = make_uniq<LogicalUnnest>(unnest_node.index);
		unnest->expressions = std::move(unnest_node.expressions);
		// visit the unnest expressions
		for (auto &expr : unnest->expressions) {
			PlanSubqueries(expr, root);
		}
		D_ASSERT(!unnest->expressions.empty());
		unnest->AddChild(std::move(root));
		root = std::move(unnest);
	}

	for (auto &expr : statement.select_list) {
		PlanSubqueries(expr, root);
	}

	if (mode == SelectPlanMode::PRESERVE_INPUT_BINDINGS_FOR_ORDER) {
		// ORDER BY-only expressions have been bound as hidden select-list entries.
		// Inline those structured expressions before the SELECT projection is
		// created, then plan the modifier directly on the input. Window, UNNEST,
		// and subquery operators planned above remain in the tree, while the
		// input's table bindings stay visible to subsequent Relation operators.
		RewriteOrderForInputBindings(statement);
		return VisitQueryNode(statement, std::move(root));
	}

	auto proj = make_uniq<LogicalProjection>(statement.projection_index, std::move(statement.select_list));
	auto &projection = *proj;
	proj->AddChild(std::move(root));
	root = std::move(proj);

	// finish the plan by handling the elements of the QueryNode
	root = VisitQueryNode(statement, std::move(root));

	// add a prune node if necessary
	if (statement.need_prune) {
		D_ASSERT(root);
		vector<unique_ptr<Expression>> prune_expressions;
		for (idx_t i = 0; i < statement.column_count; i++) {
			prune_expressions.push_back(make_uniq<BoundColumnRefExpression>(
			    projection.expressions[i]->return_type, ColumnBinding(statement.projection_index, i)));
		}
		auto prune = make_uniq<LogicalProjection>(statement.prune_index, std::move(prune_expressions));
		prune->AddChild(std::move(root));
		root = std::move(prune);
	}
	return root;
}

} // namespace duckdb

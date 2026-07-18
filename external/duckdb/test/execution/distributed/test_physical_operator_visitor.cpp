// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/physical_operator_visitor.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"

using namespace duckdb;

TEST_CASE("PhysicalOperatorVisitor: visits operators and expressions", "[execution]") {
	Allocator allocator;
	LogicalType int_type = LogicalType::INTEGER;
	vector<LogicalType> types = {int_type};

	TableFunction function;
	unique_ptr<FunctionData> bind_data;
	vector<LogicalType> return_types;
	vector<ColumnIndex> column_ids;
	vector<idx_t> projection_ids;
	vector<string> names;
	unique_ptr<TableFilterSet> table_filters;
	idx_t estimated_cardinality = 0;
	ExtraOperatorInfo extra_info;
	vector<Value> params;
	virtual_column_map_t virtual_map;

	auto plan_ptr = std::make_shared<PhysicalPlan>(allocator);
	auto &table_scan2 = plan_ptr->Make<PhysicalTableScan>(
	    types, function, std::move(bind_data), return_types, column_ids, projection_ids, names,
	    std::move(table_filters), estimated_cardinality, std::move(extra_info), params, virtual_map);
	vector<unique_ptr<Expression>> filter_select_list2;
	filter_select_list2.push_back(duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::INTEGER(1)));
	auto &filter2 = plan_ptr->Make<PhysicalFilter>(types, std::move(filter_select_list2), estimated_cardinality);
	filter2.children.emplace_back(table_scan2);
	vector<unique_ptr<Expression>> select_list2;
	select_list2.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(int_type, 0));
	auto &projection2 = plan_ptr->Make<PhysicalProjection>(types, std::move(select_list2), estimated_cardinality);
	projection2.children.emplace_back(filter2);
	plan_ptr->SetRoot(projection2);

	struct CountingVisitor : public PhysicalOperatorVisitor {
		int ops = 0;
		int exprs = 0;
		void VisitOperator(PhysicalOperator &op) override {
			ops++;
			PhysicalOperatorVisitor::VisitOperator(op);
		}
		unique_ptr<Expression> VisitReplace(BoundConstantExpression &expr, unique_ptr<Expression> *expr_ptr) override {
			exprs++;
			return nullptr;
		}
		unique_ptr<Expression> VisitReplace(BoundReferenceExpression &expr, unique_ptr<Expression> *expr_ptr) override {
			exprs++;
			return nullptr;
		}
	} visitor;

	visitor.VisitOperator(plan_ptr->Root());
	REQUIRE(visitor.ops == 3);
	REQUIRE(visitor.exprs == 2);
}

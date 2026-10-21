// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "correlated_partition.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_perfecthash_aggregate.hpp"
#include "duckdb/execution/operator/join/physical_delim_join.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

#include <set>

namespace duckdb {
namespace vane_execution {
namespace {

bool IsDelim(const PhysicalOperator &op) {
	return op.type == PhysicalOperatorType::LEFT_DELIM_JOIN || op.type == PhysicalOperatorType::RIGHT_DELIM_JOIN;
}

// A nested delimiter can be its own fragment only if all its cached scans have
// owners inside that subtree. In particular, never detach a reference to an
// enclosing delimiter or CTE just because it sits below another delimiter.
bool Independent(const PhysicalOperator &root) {
	std::set<const PhysicalOperator *> visited, owned, scans;
	bool valid = true;
	std::function<void(const PhysicalOperator &)> visit = [&](const PhysicalOperator &op) {
		if (!visited.insert(&op).second) {
			return;
		}
		if (IsDelim(op)) {
			auto &delim = static_cast<const PhysicalDelimJoin &>(op);
			for (auto &scan : delim.delim_scans) {
				owned.insert(&scan.get());
			}
			if (op.type == PhysicalOperatorType::LEFT_DELIM_JOIN) {
				owned.insert(&delim.join.children[0].get());
			}
		}
		if (op.type == PhysicalOperatorType::CTE || op.type == PhysicalOperatorType::CTE_SCAN) {
			valid = false;
		}
		if (op.type == PhysicalOperatorType::DELIM_SCAN || op.type == PhysicalOperatorType::COLUMN_DATA_SCAN) {
			scans.insert(&op);
		}
		for (auto &child : op.GetChildren()) {
			visit(child.get());
		}
	};
	visit(root);
	for (auto scan : scans) {
		valid = valid && owned.count(scan);
	}
	return valid;
}

using Keys = vector<unique_ptr<Expression>>;

unique_ptr<Expression> CopyKey(const Expression &expression, const Keys &input) {
	if (expression.GetExpressionClass() == ExpressionClass::BOUND_REF) {
		auto index = expression.Cast<BoundReferenceExpression>().index;
		return index < input.size() && input[index] ? input[index]->Copy() : nullptr;
	}
	if (expression.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
		return nullptr;
	}
	auto &function = expression.Cast<BoundFunctionExpression>();
	const bool compress = StringUtil::StartsWith(function.function.name, "__internal_compress_integral_");
	const bool decompress = StringUtil::StartsWith(function.function.name, "__internal_decompress_integral_");
	if ((!compress && !decompress) || function.children.size() != 2 ||
	    function.children[1]->GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) {
		return nullptr;
	}
	// These functions are inserted by compressed materialization, whose range
	// proof makes them injective. Their SQL binder rejects direct calls.
	auto child = CopyKey(*function.children[0], input);
	if (!child) {
		return nullptr;
	}
	if (decompress && child->GetExpressionClass() == ExpressionClass::BOUND_FUNCTION) {
		auto &inner = child->Cast<BoundFunctionExpression>();
		if (StringUtil::StartsWith(inner.function.name, "__internal_compress_integral_") &&
		    inner.children.size() == 2 && inner.children[0]->return_type == expression.return_type &&
		    inner.children[1]->Equals(*function.children[1])) {
			return inner.children[0]->Copy();
		}
	}
	auto result = expression.Copy();
	result->Cast<BoundFunctionExpression>().children[0] = std::move(child);
	return result;
}

struct Flow {
	explicit Flow(idx_t columns) : keys(columns) {
	}
	bool valid = true;
	bool partitioned = false;
	const PhysicalOperator *boundary = nullptr;
	Keys keys;
};

// Trace an unchanged key through inner joins to a scan containing every value
// the delimiter owner can produce. Do not cross null-extending joins, casts,
// aggregates, or another shared-state scope. A scan is cheap to duplicate and
// does not introduce a dependency cycle back to the delimiter being filtered.
void FindKeySource(PhysicalOperator &op, idx_t column, const LogicalType &type, CorrelatedKeySource &result) {
	if (column >= op.types.size() || op.types[column] != type) {
		return;
	}
	if (op.type == PhysicalOperatorType::TABLE_SCAN) {
		// Repeated reads of bound Parquet files have the same key domain.
		// Arbitrary table functions need not be deterministic across scans.
		auto &name = op.Cast<PhysicalTableScan>().function.name;
		if (name != "read_parquet" && name != "parquet_scan") {
			return;
		}
		if (op.estimated_cardinality &&
		    (!result.source || op.estimated_cardinality < result.source->estimated_cardinality)) {
			result.source = &op;
			result.column = column;
		}
		return;
	}
	if (op.type == PhysicalOperatorType::FILTER) {
		FindKeySource(op.children[0], column, type, result);
		return;
	}
	if (op.type == PhysicalOperatorType::PROJECTION) {
		auto &expression = *op.Cast<PhysicalProjection>().select_list[column];
		if (expression.GetExpressionClass() == ExpressionClass::BOUND_REF) {
			FindKeySource(op.children[0], expression.Cast<BoundReferenceExpression>().index, type, result);
		}
		return;
	}
	if (op.type != PhysicalOperatorType::HASH_JOIN) {
		return;
	}
	auto &join = op.Cast<PhysicalHashJoin>();
	if (join.join_type != JoinType::INNER) {
		return;
	}
	idx_t side, input_column;
	if (column < join.lhs_output_columns.col_idxs.size()) {
		side = 0;
		input_column = join.lhs_output_columns.col_idxs[column];
	} else {
		side = 1;
		auto index = join.rhs_output_columns.col_idxs[column - join.lhs_output_columns.col_idxs.size()];
		if (index < join.conditions.size()) {
			auto &key = *join.conditions[index].right;
			if (key.GetExpressionClass() != ExpressionClass::BOUND_REF) {
				return;
			}
			input_column = key.Cast<BoundReferenceExpression>().index;
		} else {
			input_column = join.payload_columns.col_idxs[index - join.conditions.size()];
		}
	}
	FindKeySource(join.children[side], input_column, type, result);
	for (auto &condition : join.conditions) {
		if (condition.comparison != ExpressionType::COMPARE_EQUAL &&
		    condition.comparison != ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
			continue;
		}
		auto &key = side == 0 ? *condition.left : *condition.right;
		auto &other = side == 0 ? *condition.right : *condition.left;
		if (key.GetExpressionClass() == ExpressionClass::BOUND_REF && key.return_type == type &&
		    key.Cast<BoundReferenceExpression>().index == input_column &&
		    other.GetExpressionClass() == ExpressionClass::BOUND_REF) {
			FindKeySource(join.children[1 - side], other.Cast<BoundReferenceExpression>().index, type, result);
		}
	}
}

class Locality {
public:
	Locality(PhysicalDelimJoin &root, idx_t group, idx_t owner_column, const LogicalType &key_type)
	    : root(root), group(group), owner_column(owner_column), key_type(key_type) {
	}

	CorrelatedPartition result;
	double saved_broadcast_rows = 0;

	bool Analyze() {
		auto &owner = root.children[0].get();
		if (owner_column >= owner.types.size() || owner.types[owner_column] != key_type || !Independent(owner)) {
			return false;
		}
		result.inputs.emplace(&owner, optional_idx(owner_column));
		auto output = Visit(root.join);
		if (output.valid && output.partitioned) {
			for (idx_t i = 0; i < output.keys.size(); i++) {
				auto &key = output.keys[i];
				if (key && key->GetExpressionClass() == ExpressionClass::BOUND_REF &&
				    key->return_type == root.types[i]) {
					result.output_keys.push_back(i);
				}
			}
			CorrelatedKeyFilter filter;
			FindKeySource(owner, owner_column, key_type, filter.scan);
			filter.owner.source = &owner;
			filter.owner.column = owner_column;
			for (auto &input : filterable_inputs) {
				filter.estimated_matching_rows = input.second;
				result.filters.emplace(input.first, filter);
			}
		}
		return output.valid && output.partitioned;
	}

private:
	PhysicalDelimJoin &root;
	idx_t group;
	idx_t owner_column;
	const LogicalType &key_type;
	unordered_map<const PhysicalOperator *, idx_t> filterable_inputs;

	Flow Seed(idx_t columns, idx_t column) {
		Flow result(columns);
		result.partitioned = true;
		result.valid = column < columns;
		if (result.valid) {
			result.keys[column] = make_uniq<BoundReferenceExpression>(key_type, 0);
		}
		return result;
	}

	// Only hash an independent input when equality identifies the original key,
	// in the same native type. Other complete inputs can safely be broadcast.
	void MatchInput(Flow &input, const Expression &input_key, const Flow &other, const Expression &other_key,
	                bool filterable, idx_t estimated_matching_rows) {
		if (!input.boundary || input.partitioned || !other.partitioned ||
		    input_key.GetExpressionClass() != ExpressionClass::BOUND_REF || input_key.return_type != key_type) {
			return;
		}
		auto key = CopyKey(other_key, other.keys);
		if (!key || key->GetExpressionClass() != ExpressionClass::BOUND_REF || key->return_type != key_type) {
			return;
		}
		auto index = input_key.Cast<BoundReferenceExpression>().index;
		if (index >= input.keys.size()) {
			return;
		}
		result.inputs[input.boundary] = optional_idx(index);
		saved_broadcast_rows += double(input.boundary->estimated_cardinality);
		input.partitioned = true;
		input.keys[index] = std::move(key);
		if (filterable) {
			filterable_inputs.emplace(input.boundary, estimated_matching_rows);
		}
	}

	bool IsOwnedScan(const PhysicalOperator &op) const {
		for (auto &scan : root.delim_scans) {
			if (&scan.get() == &op) {
				return true;
			}
		}
		return false;
	}

	Flow Join(PhysicalHashJoin &join) {
		if (join.children.size() != 2) {
			Flow output(join.types.size());
			output.valid = false;
			return output;
		}
		auto left = Visit(join.children[0]);
		auto right = Visit(join.children[1]);
		Flow output(join.types.size());
		if (!left.valid || !right.valid) {
			output.valid = false;
			return output;
		}
		bool colocated = false;
		// A residual or second key can reduce the join without making this
		// one-key filter selective. Keep those estimates out of its costing.
		const auto estimated_matching_rows = join.conditions.size() == 1 ? join.estimated_cardinality : 0;
		for (auto &condition : join.conditions) {
			if (condition.comparison != ExpressionType::COMPARE_EQUAL &&
			    condition.comparison != ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
				continue;
			}
			MatchInput(left, *condition.left, right, *condition.right,
			           join.join_type == JoinType::INNER && IsOwnedScan(join.children[1]), estimated_matching_rows);
			MatchInput(right, *condition.right, left, *condition.left,
			           join.join_type == JoinType::INNER && IsOwnedScan(join.children[0]), estimated_matching_rows);
			auto lk = CopyKey(*condition.left, left.keys);
			auto rk = CopyKey(*condition.right, right.keys);
			colocated = colocated || (lk && rk && lk->Equals(*rk));
		}
		if ((!left.partitioned && !right.partitioned) || (left.partitioned && right.partitioned && !colocated)) {
			output.valid = false;
			return output;
		}
		// Retaining a replicated side would duplicate its unmatched rows. MARK
		// additionally needs NULL/empty-build summaries; keep it globally scoped.
		const auto type = join.join_type;
		if (type == JoinType::MARK || type == JoinType::OUTER ||
		    (!left.partitioned && type != JoinType::INNER && type != JoinType::RIGHT && type != JoinType::RIGHT_SEMI &&
		     type != JoinType::RIGHT_ANTI) ||
		    (!right.partitioned && type != JoinType::INNER && type != JoinType::LEFT && type != JoinType::SINGLE &&
		     type != JoinType::SEMI && type != JoinType::ANTI)) {
			output.valid = false;
			return output;
		}
		output.partitioned = true;
		idx_t column = 0;
		if (type != JoinType::RIGHT_SEMI && type != JoinType::RIGHT_ANTI) {
			for (auto index : join.lhs_output_columns.col_idxs) {
				if (column >= output.keys.size() || index >= left.keys.size()) {
					output.valid = false;
					return output;
				}
				if (type != JoinType::RIGHT && left.keys[index]) {
					output.keys[column] = left.keys[index]->Copy();
				}
				column++;
			}
		}
		if (type != JoinType::SEMI && type != JoinType::ANTI) {
			for (auto index : join.rhs_output_columns.col_idxs) {
				if (column >= output.keys.size()) {
					output.valid = false;
					return output;
				}
				if (type != JoinType::LEFT && type != JoinType::SINGLE) {
					if (index < join.conditions.size()) {
						output.keys[column] = CopyKey(*join.conditions[index].right, right.keys);
					} else {
						auto payload = index - join.conditions.size();
						if (payload >= join.payload_columns.col_idxs.size()) {
							output.valid = false;
							return output;
						}
						auto source = join.payload_columns.col_idxs[payload];
						if (source >= right.keys.size()) {
							output.valid = false;
							return output;
						}
						if (right.keys[source]) {
							output.keys[column] = right.keys[source]->Copy();
						}
					}
				}
				column++;
			}
		}
		output.valid = column == output.keys.size();
		return output;
	}

	Flow Visit(PhysicalOperator &op) {
		const auto cached_side = root.type == PhysicalOperatorType::LEFT_DELIM_JOIN ? 0 : 1;
		if (&op == &root.join.children[cached_side].get()) {
			return Seed(op.types.size(), owner_column);
		}
		for (auto &scan : root.delim_scans) {
			if (&op == &scan.get()) {
				return Seed(op.types.size(), group);
			}
		}
		Flow output(op.types.size());
		if (Independent(op)) {
			// One serialized input has one routing policy. Reject shared DAG
			// boundaries rather than letting a later use replace that policy.
			output.valid = result.inputs.emplace(&op, optional_idx()).second;
			output.boundary = &op;
			return output;
		}
		if (op.type == PhysicalOperatorType::HASH_JOIN) {
			return Join(op.Cast<PhysicalHashJoin>());
		}
		if (op.children.size() != 1) {
			output.valid = false;
			return output;
		}
		auto child = Visit(op.children[0]);
		if (!child.valid || !child.partitioned) {
			output.valid = false;
			return output;
		}
		output.partitioned = true;
		if (op.type == PhysicalOperatorType::FILTER) {
			output.keys = std::move(child.keys);
			output.valid = output.keys.size() == op.types.size();
			return output;
		}
		if (op.type == PhysicalOperatorType::PROJECTION) {
			auto &expressions = op.Cast<PhysicalProjection>().select_list;
			if (expressions.size() != output.keys.size()) {
				output.valid = false;
				return output;
			}
			for (idx_t i = 0; i < expressions.size(); i++) {
				output.keys[i] = CopyKey(*expressions[i], child.keys);
			}
			return output;
		}
		const vector<unique_ptr<Expression>> *groups = nullptr;
		const vector<GroupingSet> *sets = nullptr;
		if (op.type == PhysicalOperatorType::HASH_GROUP_BY) {
			auto &aggregate = op.Cast<PhysicalHashAggregate>();
			groups = &aggregate.grouped_aggregate_data.groups;
			sets = &aggregate.grouping_sets;
		} else if (op.type == PhysicalOperatorType::PERFECT_HASH_GROUP_BY) {
			groups = &op.Cast<PhysicalPerfectHashAggregate>().groups;
		}
		if (!groups || groups->size() > output.keys.size()) {
			output.valid = false;
			return output;
		}
		bool has_key = false;
		for (idx_t i = 0; i < groups->size(); i++) {
			output.keys[i] = CopyKey(*(*groups)[i], child.keys);
			has_key = has_key || bool(output.keys[i]);
		}
		output.valid = has_key;
		if (sets) {
			for (auto &set : *sets) {
				bool complete_group = false;
				for (auto index : set) {
					complete_group = complete_group || (index < groups->size() && bool(output.keys[index]));
				}
				output.valid = output.valid && complete_group;
			}
			// An omitted grouping column becomes NULL. It no longer identifies
			// the input partition even if another grouping column keeps that
			// grouping set local. Never propagate its original key lineage.
			for (idx_t i = 0; i < groups->size(); i++) {
				for (auto &set : *sets) {
					if (!set.count(i)) {
						output.keys[i].reset();
					}
				}
			}
		}
		return output;
	}
};

} // namespace

bool PlanCorrelatedPartition(PhysicalOperator &root, CorrelatedPartition &result) {
	if (!IsDelim(root) || !Independent(root)) {
		return false;
	}
	auto &delim = static_cast<PhysicalDelimJoin &>(root);
	if (delim.children.size() != 1 || delim.join.type != PhysicalOperatorType::HASH_JOIN ||
	    delim.join.children.size() != 2 || delim.delim_scans.empty() ||
	    !delim.distinct.grouped_aggregate_data.aggregates.empty()) {
		return false;
	}
	auto &groups = delim.distinct.grouped_aggregate_data.groups;
	bool found = false;
	double best = -1;
	for (idx_t i = 0; i < groups.size(); i++) {
		auto &key = *groups[i];
		// Integral keys have identical equality/hash semantics in every native
		// operator. Collated strings and other representations need their own proof.
		if (key.GetExpressionClass() != ExpressionClass::BOUND_REF || !key.return_type.IsIntegral()) {
			continue;
		}
		auto column = key.Cast<BoundReferenceExpression>().index;
		Locality proof(delim, i, column, key.return_type);
		if (proof.Analyze() && proof.saved_broadcast_rows > best) {
			best = proof.saved_broadcast_rows;
			result = std::move(proof.result);
			found = true;
		}
	}
	return found;
}

} // namespace vane_execution
} // namespace duckdb

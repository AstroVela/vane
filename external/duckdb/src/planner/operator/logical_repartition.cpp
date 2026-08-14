// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/operator/logical_repartition.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb/planner/operator/logical_repartition.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/limits.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"

namespace duckdb {

namespace {

//! Values in a serialized logical plan. These are deliberately independent of
//! RepartitionSpec::Type so changing the in-memory enum cannot reinterpret a plan.
enum class RepartitionWireType : uint8_t { HASH = 1, RANDOM = 2, INTO_PARTITIONS = 3, RANGE = 4 };

static size_t ReadPartitionCount(uint64_t count) {
	if (count > static_cast<uint64_t>(NumericLimits<size_t>::Maximum())) {
		throw SerializationException("Logical repartition partition count is too large for this platform");
	}
	return static_cast<size_t>(count);
}

static void ValidatePartitionExpressions(const vector<unique_ptr<Expression>> &expressions) {
	for (auto &expression : expressions) {
		if (!expression) {
			throw SerializationException("Logical repartition contains a null partition expression");
		}
	}
}

} // namespace

LogicalRepartition::LogicalRepartition(std::shared_ptr<RepartitionSpec> repartition_spec_p)
    : LogicalOperator(LogicalOperatorType::LOGICAL_REPARTITION), repartition_spec(std::move(repartition_spec_p)) {
}

vector<ColumnBinding> LogicalRepartition::GetColumnBindings() {
	return children[0]->GetColumnBindings();
}

idx_t LogicalRepartition::EstimateCardinality(ClientContext &context) {
	return children[0]->EstimateCardinality(context);
}

void LogicalRepartition::ResolveTypes() {
	types = children[0]->types;
}

void LogicalRepartition::Serialize(Serializer &serializer) const {
	LogicalOperator::Serialize(serializer);
	if (!repartition_spec) {
		throw SerializationException("Logical repartition is missing its repartition specification");
	}

	RepartitionWireType wire_type;
	uint64_t num_partitions;
	switch (repartition_spec->type()) {
	case RepartitionSpec::Type::Hash: {
		wire_type = RepartitionWireType::HASH;
		auto *hash_spec = dynamic_cast<HashRepartitionSpec *>(repartition_spec.get());
		if (!hash_spec || !hash_spec->config()) {
			throw SerializationException("Logical hash repartition has an invalid specification");
		}
		auto &config = *hash_spec->config();
		if (expressions.empty() || config.by.empty() || expressions.size() != config.by.size()) {
			throw SerializationException("Logical hash repartition has inconsistent partition expressions");
		}
		ValidatePartitionExpressions(expressions);
		for (idx_t expression_idx = 0; expression_idx < config.by.size(); expression_idx++) {
			auto &config_expression = config.by[expression_idx];
			if (!config_expression) {
				throw SerializationException("Logical hash repartition contains a null partition expression");
			}
			if (!config_expression->Equals(*expressions[expression_idx])) {
				throw SerializationException("Logical hash repartition has inconsistent partition expressions");
			}
		}
		num_partitions = static_cast<uint64_t>(config.num_partitions);
		break;
	}
	case RepartitionSpec::Type::Random: {
		wire_type = RepartitionWireType::RANDOM;
		auto *random_spec = dynamic_cast<RandomRepartitionSpec *>(repartition_spec.get());
		if (!random_spec || !random_spec->config()) {
			throw SerializationException("Logical random repartition has an invalid specification");
		}
		if (!expressions.empty()) {
			throw SerializationException("Logical random repartition must not contain partition expressions");
		}
		num_partitions = static_cast<uint64_t>(random_spec->config()->num_partitions);
		break;
	}
	case RepartitionSpec::Type::IntoPartitions: {
		wire_type = RepartitionWireType::INTO_PARTITIONS;
		auto *into_spec = dynamic_cast<IntoPartitionsRepartitionSpec *>(repartition_spec.get());
		if (!into_spec || !into_spec->config()) {
			throw SerializationException("Logical into-partitions repartition has an invalid specification");
		}
		if (!expressions.empty()) {
			throw SerializationException("Logical into-partitions repartition must not contain partition expressions");
		}
		if (into_spec->config()->num_partitions == 0) {
			throw SerializationException("Logical into-partitions repartition requires a positive partition count");
		}
		num_partitions = static_cast<uint64_t>(into_spec->config()->num_partitions);
		break;
	}
	case RepartitionSpec::Type::Range:
		throw SerializationException("Range repartition cannot be serialized in a logical plan");
	default:
		throw SerializationException("Logical repartition has an unknown in-memory repartition type");
	}

	serializer.WriteProperty<uint8_t>(200, "repartition_type", static_cast<uint8_t>(wire_type));
	serializer.WriteProperty<uint64_t>(201, "num_partitions", num_partitions);
	serializer.WriteProperty<vector<unique_ptr<Expression>>>(202, "partition_by", expressions);
}

unique_ptr<LogicalOperator> LogicalRepartition::Deserialize(Deserializer &deserializer) {
	auto wire_type = static_cast<RepartitionWireType>(deserializer.ReadProperty<uint8_t>(200, "repartition_type"));
	auto num_partitions = ReadPartitionCount(deserializer.ReadProperty<uint64_t>(201, "num_partitions"));
	auto partition_by = deserializer.ReadProperty<vector<unique_ptr<Expression>>>(202, "partition_by");
	ValidatePartitionExpressions(partition_by);

	std::shared_ptr<RepartitionSpec> spec;
	switch (wire_type) {
	case RepartitionWireType::HASH: {
		if (partition_by.empty()) {
			throw SerializationException("Logical hash repartition requires at least one partition expression");
		}
		vector<ExprRef> expression_refs;
		expression_refs.reserve(partition_by.size());
		for (auto &expression : partition_by) {
			auto copy = expression->Copy();
			expression_refs.emplace_back(copy.release());
		}
		spec = RepartitionSpec::create_hash(num_partitions, std::move(expression_refs));
		break;
	}
	case RepartitionWireType::RANDOM:
		if (!partition_by.empty()) {
			throw SerializationException("Logical random repartition must not contain partition expressions");
		}
		spec = RepartitionSpec::create_random(num_partitions);
		break;
	case RepartitionWireType::INTO_PARTITIONS:
		if (!partition_by.empty()) {
			throw SerializationException("Logical into-partitions repartition must not contain partition expressions");
		}
		if (num_partitions == 0) {
			throw SerializationException("Logical into-partitions repartition requires a positive partition count");
		}
		spec = RepartitionSpec::create_into_partitions(num_partitions);
		break;
	case RepartitionWireType::RANGE:
		throw SerializationException("Range repartition cannot be deserialized from a logical plan");
	default:
		throw SerializationException("Logical repartition contains an unknown wire repartition type");
	}

	auto result = make_uniq<LogicalRepartition>(std::move(spec));
	result->expressions = std::move(partition_by);
	return std::move(result);
}

} // namespace duckdb

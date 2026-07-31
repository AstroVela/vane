// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/operator/exchange/repartition.hpp"

#include "duckdb/common/exception.hpp"

#include <algorithm>
#include <cstring>

namespace duckdb {

namespace {

void ValidateRangeOrders(const vector<BoundOrderByNode> &orders) {
	if (orders.empty()) {
		throw InvalidInputException("range repartition requires at least one order expression");
	}
	for (const auto &order : orders) {
		if (!order.expression) {
			throw InvalidInputException("range repartition order expression cannot be null");
		}
		if (order.type != OrderType::ASCENDING && order.type != OrderType::DESCENDING) {
			throw InvalidInputException("range repartition requires explicit ASC or DESC order");
		}
		if (order.null_order != OrderByNullType::NULLS_FIRST && order.null_order != OrderByNullType::NULLS_LAST) {
			throw InvalidInputException("range repartition requires explicit NULLS FIRST or NULLS LAST order");
		}
	}
}

bool EncodedSortKeyLess(const string &left, const string &right) {
	const auto min_size = std::min(left.size(), right.size());
	const auto comparison = std::memcmp(left.data(), right.data(), min_size);
	if (comparison != 0) {
		return comparison < 0;
	}
	return left.size() < right.size();
}

vector<BoundOrderByNode> CopyRangeOrders(const vector<BoundOrderByNode> &orders) {
	vector<BoundOrderByNode> result;
	result.reserve(orders.size());
	for (const auto &order : orders) {
		result.push_back(order.Copy());
	}
	return result;
}

std::vector<ExprRef> CopyRangeExpressions(const vector<BoundOrderByNode> &orders) {
	std::vector<ExprRef> result;
	result.reserve(orders.size());
	for (const auto &order : orders) {
		result.emplace_back(order.expression->Copy().release());
	}
	return result;
}

} // namespace

// HashRepartitionConfig 实现
HashRepartitionConfig::HashRepartitionConfig(size_t num_partitions, std::vector<ExprRef> by)
    : num_partitions(num_partitions), by(std::move(by)) {
}

std::vector<std::string> HashRepartitionConfig::multiline_display() const {
	std::vector<std::string> result;
	result.push_back("Num partitions = " + (num_partitions ? std::to_string(num_partitions) : "None"));

	std::string by_str = "By = ";
	for (size_t i = 0; i < by.size(); ++i) {
		if (i > 0)
			by_str += ", ";
		by_str += "(expr)"; // placeholder for expression display
	}
	result.push_back(by_str);

	return result;
}

std::shared_ptr<HashRepartitionConfig> HashRepartitionConfig::create(size_t num_partitions, std::vector<ExprRef> by) {
	return std::make_shared<HashRepartitionConfig>(num_partitions, std::move(by));
}

// RandomShuffleConfig 实现
RandomShuffleConfig::RandomShuffleConfig(size_t num_partitions) : num_partitions(num_partitions) {
}

std::vector<std::string> RandomShuffleConfig::multiline_display() const {
	return {"Num partitions = " + (num_partitions ? std::to_string(num_partitions) : "None")};
}

std::shared_ptr<RandomShuffleConfig> RandomShuffleConfig::create(size_t num_partitions) {
	return std::make_shared<RandomShuffleConfig>(num_partitions);
}

// RangeRepartitionConfig 实现
RangeRepartitionConfig::RangeRepartitionConfig(size_t num_partitions, vector<BoundOrderByNode> orders,
                                               vector<string> boundaries)
    : num_partitions_(num_partitions), orders_(std::move(orders)), boundaries_(std::move(boundaries)) {
	ValidateRangeOrders(orders_);
	ValidateBoundaries(num_partitions_, boundaries_);
}

std::vector<std::string> RangeRepartitionConfig::multiline_display() const {
	std::vector<std::string> result;
	result.push_back("Num partitions = " + (num_partitions_ ? std::to_string(num_partitions_) : "None"));

	std::string by = "By = ";
	for (size_t i = 0; i < orders_.size(); ++i) {
		if (i > 0)
			by += ", ";
		by += orders_[i].ToString();
	}
	result.push_back(std::move(by));
	result.push_back("Boundaries = " + std::to_string(boundaries_.size()));

	return result;
}

std::shared_ptr<const RangeRepartitionConfig>
RangeRepartitionConfig::create(size_t num_partitions, vector<BoundOrderByNode> orders, vector<string> boundaries) {
	return std::make_shared<const RangeRepartitionConfig>(num_partitions, std::move(orders), std::move(boundaries));
}

void RangeRepartitionConfig::Validate(const vector<BoundOrderByNode> &orders, size_t num_partitions,
                                      const vector<string> &boundaries) {
	ValidateRangeOrders(orders);
	if (num_partitions == 0) {
		throw InvalidInputException("range repartition requires at least one partition");
	}
	ValidateBoundaries(num_partitions, boundaries);
}

void RangeRepartitionConfig::ValidateBoundaries(size_t num_partitions, const vector<string> &boundaries) {
	if (num_partitions && boundaries.size() >= num_partitions) {
		throw InvalidInputException("range repartition requires fewer boundary keys than partitions");
	}
	for (size_t index = 1; index < boundaries.size(); index++) {
		if (EncodedSortKeyLess(boundaries[index], boundaries[index - 1])) {
			throw InvalidInputException("range repartition boundary keys must be sorted");
		}
	}
}

vector<BoundOrderByNode> RangeRepartitionConfig::CopyOrders() const {
	return CopyRangeOrders(orders_);
}

// IntoPartitionsConfig 实现
IntoPartitionsConfig::IntoPartitionsConfig(size_t num_partitions) : num_partitions(num_partitions) {
}

std::vector<std::string> IntoPartitionsConfig::multiline_display() const {
	return {"Num partitions = " + std::to_string(num_partitions)};
}

std::shared_ptr<IntoPartitionsConfig> IntoPartitionsConfig::create(size_t num_partitions) {
	return std::make_shared<IntoPartitionsConfig>(num_partitions);
}

// RepartitionSpec 工厂方法
std::shared_ptr<RepartitionSpec> RepartitionSpec::create_hash(size_t num_partitions, std::vector<ExprRef> by) {
	auto config = HashRepartitionConfig::create(num_partitions, std::move(by));
	return std::make_shared<HashRepartitionSpec>(config);
}

std::shared_ptr<RepartitionSpec> RepartitionSpec::create_random(size_t num_partitions) {
	auto config = RandomShuffleConfig::create(num_partitions);
	return std::make_shared<RandomRepartitionSpec>(config);
}

std::shared_ptr<RepartitionSpec> RepartitionSpec::create_into_partitions(size_t num_partitions) {
	auto config = IntoPartitionsConfig::create(num_partitions);
	return std::make_shared<IntoPartitionsRepartitionSpec>(config);
}

std::shared_ptr<RepartitionSpec> RepartitionSpec::create_range(size_t num_partitions, vector<BoundOrderByNode> orders,
                                                               vector<string> boundaries) {
	auto config = RangeRepartitionConfig::create(num_partitions, std::move(orders), std::move(boundaries));
	return std::make_shared<RangeRepartitionSpec>(std::move(config));
}

// HashRepartitionSpec 实现
HashRepartitionSpec::HashRepartitionSpec(std::shared_ptr<HashRepartitionConfig> config) : config_(std::move(config)) {
}

std::vector<std::string> HashRepartitionSpec::multiline_display() const {
	return config_->multiline_display();
}

ClusteringSpecRef HashRepartitionSpec::to_clustering_spec(size_t upstream_num_partitions) const {
	size_t actual_num_partitions = config_->num_partitions ? config_->num_partitions : upstream_num_partitions;
	auto clustering_config = HashClusteringConfig(actual_num_partitions, config_->by);
	return ClusteringSpec::from_hash_config(clustering_config);
}

// 其他 RepartitionSpec 派生类的实现类似...
// 为简洁起见，这里省略详细实现

// RandomRepartitionSpec 实现
RandomRepartitionSpec::RandomRepartitionSpec(std::shared_ptr<RandomShuffleConfig> config) : config_(std::move(config)) {
}

std::vector<std::string> RandomRepartitionSpec::multiline_display() const {
	return config_->multiline_display();
}

ClusteringSpecRef RandomRepartitionSpec::to_clustering_spec(size_t upstream_num_partitions) const {
	size_t actual = config_->num_partitions ? config_->num_partitions : upstream_num_partitions;
	auto cfg = RandomClusteringConfig(actual);
	return ClusteringSpec::from_random_config(cfg);
}

// IntoPartitionsRepartitionSpec 实现
IntoPartitionsRepartitionSpec::IntoPartitionsRepartitionSpec(std::shared_ptr<IntoPartitionsConfig> config)
    : config_(std::move(config)) {
}

std::vector<std::string> IntoPartitionsRepartitionSpec::multiline_display() const {
	return config_->multiline_display();
}

ClusteringSpecRef IntoPartitionsRepartitionSpec::to_clustering_spec(size_t /*upstream_num_partitions*/) const {
	auto cfg = UnknownClusteringConfig(config_->num_partitions);
	return ClusteringSpec::from_unknown_config(cfg);
}

// RangeRepartitionSpec 实现
RangeRepartitionSpec::RangeRepartitionSpec(std::shared_ptr<const RangeRepartitionConfig> config)
    : config_(std::move(config)) {
	if (!config_) {
		throw InvalidInputException("range repartition requires a non-null configuration");
	}
}

std::vector<std::string> RangeRepartitionSpec::multiline_display() const {
	return config_->multiline_display();
}

ClusteringSpecRef RangeRepartitionSpec::to_clustering_spec(size_t upstream_num_partitions) const {
	size_t actual = config_->num_partitions() ? config_->num_partitions() : upstream_num_partitions;
	RangeRepartitionConfig::Validate(config_->orders(), actual, config_->boundaries());
	return ClusteringSpec::from_range_config(RangeClusteringConfig(actual, config_->CopyOrders()));
}

// RangeClusteringConfig 实现
RangeClusteringConfig::RangeClusteringConfig(size_t num_partitions, vector<BoundOrderByNode> orders)
    : num_partitions_(num_partitions), orders_(std::move(orders)) {
	ValidateRangeOrders(orders_);
	if (num_partitions_ == 0) {
		throw InvalidInputException("range clustering requires at least one partition");
	}
}

RangeClusteringConfig::RangeClusteringConfig(const RangeClusteringConfig &other)
    : num_partitions_(other.num_partitions_), orders_(CopyRangeOrders(other.orders_)) {
}

RangeClusteringConfig &RangeClusteringConfig::operator=(const RangeClusteringConfig &other) {
	if (this != &other) {
		auto orders = CopyRangeOrders(other.orders_);
		num_partitions_ = other.num_partitions_;
		orders_ = std::move(orders);
	}
	return *this;
}

std::vector<std::string> RangeClusteringConfig::multiline_display() const {
	std::vector<std::string> result;
	result.push_back("Num partitions = " + std::to_string(num_partitions_));

	std::string by = "By = ";
	for (size_t i = 0; i < orders_.size(); ++i) {
		if (i > 0)
			by += ", ";
		by += orders_[i].ToString();
	}
	result.push_back(std::move(by));

	return result;
}

std::vector<ExprRef> RangeClusteringConfig::partition_by() const {
	return CopyRangeExpressions(orders_);
}

// HashClusteringConfig 实现
HashClusteringConfig::HashClusteringConfig(size_t num_partitions, std::vector<ExprRef> by)
    : num_partitions(num_partitions), by(std::move(by)) {
}

// HashClusteringSpec 实现
HashClusteringSpec::HashClusteringSpec(const HashClusteringConfig &config) : config_(config) {
}

size_t HashClusteringSpec::num_partitions() const {
	return config_.num_partitions;
}

std::vector<ExprRef> HashClusteringSpec::partition_by() const {
	return config_.by;
}

std::vector<std::string> HashClusteringSpec::multiline_display() const {
	return config_.multiline_display();
}

// RandomClusteringSpec 实现
RandomClusteringSpec::RandomClusteringSpec(const RandomClusteringConfig &config) : config_(config) {
}

size_t RandomClusteringSpec::num_partitions() const {
	return config_.num_partitions;
}

std::vector<ExprRef> RandomClusteringSpec::partition_by() const {
	return {};
}

std::vector<std::string> RandomClusteringSpec::multiline_display() const {
	return config_.multiline_display();
}

// UnknownClusteringSpec 实现
UnknownClusteringSpec::UnknownClusteringSpec(const UnknownClusteringConfig &config) : config_(config) {
}

size_t UnknownClusteringSpec::num_partitions() const {
	return config_.num_partitions;
}

std::vector<ExprRef> UnknownClusteringSpec::partition_by() const {
	return {};
}

std::vector<std::string> UnknownClusteringSpec::multiline_display() const {
	return config_.multiline_display();
}

std::vector<std::string> HashClusteringConfig::multiline_display() const {
	std::vector<std::string> result;
	result.push_back("Num partitions = " + std::to_string(num_partitions));

	std::string by_str = "By = ";
	for (size_t i = 0; i < by.size(); ++i) {
		if (i > 0)
			by_str += ", ";
		by_str += "(expr)"; // placeholder for expression display
	}
	result.push_back(by_str);

	return result;
}

// RandomClusteringConfig 实现
RandomClusteringConfig::RandomClusteringConfig(size_t num_partitions) : num_partitions(num_partitions) {
}

std::vector<std::string> RandomClusteringConfig::multiline_display() const {
	return {"Num partitions = " + std::to_string(num_partitions)};
}

// UnknownClusteringConfig 实现
UnknownClusteringConfig::UnknownClusteringConfig(size_t num_partitions) : num_partitions(num_partitions) {
}

std::vector<std::string> UnknownClusteringConfig::multiline_display() const {
	return {"Num partitions = " + std::to_string(num_partitions)};
}

// ClusteringSpec 静态工厂方法
// The ClusteringSpec factory helpers are now implemented inline in repartition.hpp.

// RangeClusteringSpec 实现
RangeClusteringSpec::RangeClusteringSpec(RangeClusteringConfig config) : config_(std::move(config)) {
}

size_t RangeClusteringSpec::num_partitions() const {
	return config_.num_partitions();
}

std::vector<ExprRef> RangeClusteringSpec::partition_by() const {
	return config_.partition_by();
}

std::vector<std::string> RangeClusteringSpec::multiline_display() const {
	return config_.multiline_display();
}

} // namespace duckdb

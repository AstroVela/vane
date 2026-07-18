// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/operator/exchange/repartition.hpp"

namespace duckdb {

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
RangeRepartitionConfig::RangeRepartitionConfig(size_t num_partitions, std::shared_ptr<RecordBatch> boundaries,
                                               std::vector<std::shared_ptr<class BoundExpr>> by,
                                               std::vector<bool> descending)
    : num_partitions(num_partitions), boundaries(std::move(boundaries)), by(std::move(by)),
      descending(std::move(descending)) {
}

std::vector<std::string> RangeRepartitionConfig::multiline_display() const {
	std::vector<std::string> result;
	result.push_back("Num partitions = " + (num_partitions ? std::to_string(num_partitions) : "None"));

	std::string pairs = "By = ";
	for (size_t i = 0; i < by.size(); ++i) {
		if (i > 0)
			pairs += ", ";
		pairs += "(expr, " + (descending[i] ? std::string("descending") : std::string("ascending")) + ")";
	}
	result.push_back(pairs);

	return result;
}

std::shared_ptr<RangeRepartitionConfig> RangeRepartitionConfig::create(size_t num_partitions,
                                                                       std::shared_ptr<RecordBatch> boundaries,
                                                                       std::vector<std::shared_ptr<class BoundExpr>> by,
                                                                       std::vector<bool> descending) {
	return std::make_shared<RangeRepartitionConfig>(num_partitions, std::move(boundaries), std::move(by),
	                                                std::move(descending));
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

std::shared_ptr<RepartitionSpec> RepartitionSpec::create_range(size_t num_partitions,
                                                               std::shared_ptr<RecordBatch> boundaries,
                                                               std::vector<std::shared_ptr<class BoundExpr>> by,
                                                               std::vector<bool> descending) {
	auto config =
	    RangeRepartitionConfig::create(num_partitions, std::move(boundaries), std::move(by), std::move(descending));
	return std::make_shared<RangeRepartitionSpec>(config);
}

// HashRepartitionSpec 实现
HashRepartitionSpec::HashRepartitionSpec(std::shared_ptr<HashRepartitionConfig> config) : config_(std::move(config)) {
}

std::vector<ExprRef> HashRepartitionSpec::repartition_by() const {
	return config_->by;
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

std::vector<ExprRef> RandomRepartitionSpec::repartition_by() const {
	return {};
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

std::vector<ExprRef> IntoPartitionsRepartitionSpec::repartition_by() const {
	return {};
}

std::vector<std::string> IntoPartitionsRepartitionSpec::multiline_display() const {
	return config_->multiline_display();
}

ClusteringSpecRef IntoPartitionsRepartitionSpec::to_clustering_spec(size_t /*upstream_num_partitions*/) const {
	auto cfg = UnknownClusteringConfig(config_->num_partitions);
	return ClusteringSpec::from_unknown_config(cfg);
}

// RangeRepartitionSpec 实现
RangeRepartitionSpec::RangeRepartitionSpec(std::shared_ptr<RangeRepartitionConfig> config)
    : config_(std::move(config)) {
}

std::vector<ExprRef> RangeRepartitionSpec::repartition_by() const {
	// Convert BoundExpr list to ExprRef placeholders (real conversion would be more complex)
	std::vector<ExprRef> res;
	for (auto &b : config_->by) {
		res.push_back(nullptr);
	}
	return res;
}

std::vector<std::string> RangeRepartitionSpec::multiline_display() const {
	return config_->multiline_display();
}

ClusteringSpecRef RangeRepartitionSpec::to_clustering_spec(size_t upstream_num_partitions) const {
	size_t actual = config_->num_partitions ? config_->num_partitions : upstream_num_partitions;
	auto cfg = RangeClusteringConfig(actual, repartition_by(), std::vector<bool> {});
	return ClusteringSpec::from_range_config(cfg);
}

// RangeClusteringConfig 实现
RangeClusteringConfig::RangeClusteringConfig(size_t num_partitions, std::vector<ExprRef> by,
                                             std::vector<bool> descending)
    : num_partitions(num_partitions), by(std::move(by)), descending(std::move(descending)) {
}

std::vector<std::string> RangeClusteringConfig::multiline_display() const {
	std::vector<std::string> result;
	result.push_back("Num partitions = " + std::to_string(num_partitions));

	std::string pairs = "By = ";
	for (size_t i = 0; i < by.size(); ++i) {
		if (i > 0)
			pairs += ", ";
		// We don't depend on Expression internals here; print a placeholder
		pairs += "(expr, " + (descending[i] ? std::string("descending") : std::string("ascending")) + ")";
	}
	result.push_back(pairs);

	return result;
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
RangeClusteringSpec::RangeClusteringSpec(const RangeClusteringConfig &config) : config_(config) {
}

size_t RangeClusteringSpec::num_partitions() const {
	return config_.num_partitions;
}

std::vector<ExprRef> RangeClusteringSpec::partition_by() const {
	return config_.by;
}

std::vector<std::string> RangeClusteringSpec::multiline_display() const {
	return config_.multiline_display();
}

// 其他 ClusteringSpec 派生类的实现类似...
// 为简洁起见，这里省略详细实现

// 聚类规范转换函数
ClusteringSpecRef translate_clustering_spec(ClusteringSpecRef input_clustering_spec,
                                            const std::vector<ExprRef> &projection) {
	// 简化实现，实际需要根据输入规范类型和投影表达式进行复杂转换
	// 这里返回一个未知聚类规范作为示例
	return ClusteringSpec::unknown_with_num_partitions(input_clustering_spec->num_partitions());
}

} // namespace duckdb

// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <cstddef>

namespace duckdb {

// 前置声明
class Expression;
class RecordBatch;
class ClusteringSpec;

using ExprRef = std::shared_ptr<Expression>;
using ClusteringSpecRef = std::shared_ptr<ClusteringSpec>;

// Forward-declare clustering config types so ClusteringSpec can reference them
class RangeClusteringConfig;
class HashClusteringConfig;
class RandomClusteringConfig;
class UnknownClusteringConfig;

// Forward-declare clustering spec derived classes so ClusteringSpec default
// implementations can reference them later (definitions come below).
class RangeClusteringSpec;
class HashClusteringSpec;
class RandomClusteringSpec;
class UnknownClusteringSpec;

// 基础配置类
class BaseConfig {
public:
	virtual ~BaseConfig() = default;
	virtual std::vector<std::string> multiline_display() const = 0;
};

// 哈希重新分区配置
class HashRepartitionConfig : public BaseConfig {
public:
	size_t num_partitions; // 0 means auto
	std::vector<ExprRef> by;

	HashRepartitionConfig(size_t num_partitions, std::vector<ExprRef> by);
	std::vector<std::string> multiline_display() const override;

	static std::shared_ptr<HashRepartitionConfig> create(size_t num_partitions, std::vector<ExprRef> by);
};

// 随机洗牌配置
class RandomShuffleConfig : public BaseConfig {
public:
	size_t num_partitions; // 0 means auto

	RandomShuffleConfig(size_t num_partitions);
	std::vector<std::string> multiline_display() const override;

	static std::shared_ptr<RandomShuffleConfig> create(size_t num_partitions);
};

// 范围重新分区配置
class RangeRepartitionConfig : public BaseConfig {
public:
	size_t num_partitions; // 0 means auto
	std::shared_ptr<RecordBatch> boundaries;
	std::vector<std::shared_ptr<class BoundExpr>> by;
	std::vector<bool> descending;

	RangeRepartitionConfig(size_t num_partitions, std::shared_ptr<RecordBatch> boundaries,
	                       std::vector<std::shared_ptr<class BoundExpr>> by, std::vector<bool> descending);
	std::vector<std::string> multiline_display() const override;

	static std::shared_ptr<RangeRepartitionConfig> create(size_t num_partitions,
	                                                      std::shared_ptr<RecordBatch> boundaries,
	                                                      std::vector<std::shared_ptr<class BoundExpr>> by,
	                                                      std::vector<bool> descending);
};

// 分区数配置
class IntoPartitionsConfig : public BaseConfig {
public:
	size_t num_partitions;

	IntoPartitionsConfig(size_t num_partitions);
	std::vector<std::string> multiline_display() const override;

	static std::shared_ptr<IntoPartitionsConfig> create(size_t num_partitions);
};

// 重新分区规范枚举的类层次结构
class RepartitionSpec {
public:
	enum class Type { Hash, Random, IntoPartitions, Range };

	virtual ~RepartitionSpec() = default;

	virtual Type type() const = 0;
	virtual std::string var_name() const = 0;
	virtual std::vector<ExprRef> repartition_by() const = 0;
	virtual std::vector<std::string> multiline_display() const = 0;
	virtual ClusteringSpecRef to_clustering_spec(size_t upstream_num_partitions) const = 0;

	// 静态工厂方法
	static std::shared_ptr<RepartitionSpec> create_hash(size_t num_partitions, std::vector<ExprRef> by);
	static std::shared_ptr<RepartitionSpec> create_random(size_t num_partitions);
	static std::shared_ptr<RepartitionSpec> create_into_partitions(size_t num_partitions);
	static std::shared_ptr<RepartitionSpec> create_range(size_t num_partitions, std::shared_ptr<RecordBatch> boundaries,
	                                                     std::vector<std::shared_ptr<class BoundExpr>> by,
	                                                     std::vector<bool> descending);
};

// 聚类规范（ClusteringSpec）基础类
class ClusteringSpec {
public:
	enum class Type { Range, Hash, Random, Unknown };

	virtual ~ClusteringSpec() = default;

	virtual Type type() const = 0;

	// Default implementations are provided below after the derived classes
	virtual std::string var_name() const;
	virtual size_t num_partitions() const;
	virtual std::vector<ExprRef> partition_by() const;
	virtual std::vector<std::string> multiline_display() const;

	static ClusteringSpecRef unknown();
	static ClusteringSpecRef unknown_with_num_partitions(size_t num_partitions);
	static ClusteringSpecRef from_range_config(const RangeClusteringConfig &config);
	static ClusteringSpecRef from_hash_config(const HashClusteringConfig &config);
	static ClusteringSpecRef from_random_config(const RandomClusteringConfig &config);
	static ClusteringSpecRef from_unknown_config(const UnknownClusteringConfig &config);
};

// Inline static factory helpers for operator-level ClusteringSpec (mirror Rust impl)
// (Factory helpers and default method implementations are defined after the
// derived clustering spec classes so all types are complete.)

// Forward-declare the clustering config types at namespace scope (they are defined below)
class RangeClusteringConfig;
class HashClusteringConfig;
class RandomClusteringConfig;
class UnknownClusteringConfig;

// 具体的重新分区规范实现
class HashRepartitionSpec : public RepartitionSpec {
private:
	std::shared_ptr<HashRepartitionConfig> config_;

public:
	HashRepartitionSpec(std::shared_ptr<HashRepartitionConfig> config);

	Type type() const override {
		return Type::Hash;
	}
	std::string var_name() const override {
		return "Hash";
	}
	std::vector<ExprRef> repartition_by() const override;
	std::vector<std::string> multiline_display() const override;
	ClusteringSpecRef to_clustering_spec(size_t upstream_num_partitions) const override;

	const std::shared_ptr<HashRepartitionConfig> &config() const {
		return config_;
	}
};

class RandomRepartitionSpec : public RepartitionSpec {
private:
	std::shared_ptr<RandomShuffleConfig> config_;

public:
	RandomRepartitionSpec(std::shared_ptr<RandomShuffleConfig> config);

	Type type() const override {
		return Type::Random;
	}
	std::string var_name() const override {
		return "Random";
	}
	std::vector<ExprRef> repartition_by() const override;
	std::vector<std::string> multiline_display() const override;
	ClusteringSpecRef to_clustering_spec(size_t upstream_num_partitions) const override;

	const std::shared_ptr<RandomShuffleConfig> &config() const {
		return config_;
	}
};

class IntoPartitionsRepartitionSpec : public RepartitionSpec {
private:
	std::shared_ptr<IntoPartitionsConfig> config_;

public:
	IntoPartitionsRepartitionSpec(std::shared_ptr<IntoPartitionsConfig> config);

	Type type() const override {
		return Type::IntoPartitions;
	}
	std::string var_name() const override {
		return "IntoPartitions";
	}
	std::vector<ExprRef> repartition_by() const override;
	std::vector<std::string> multiline_display() const override;
	ClusteringSpecRef to_clustering_spec(size_t upstream_num_partitions) const override;

	const std::shared_ptr<IntoPartitionsConfig> &config() const {
		return config_;
	}
};

class RangeRepartitionSpec : public RepartitionSpec {
private:
	std::shared_ptr<RangeRepartitionConfig> config_;

public:
	RangeRepartitionSpec(std::shared_ptr<RangeRepartitionConfig> config);

	Type type() const override {
		return Type::Range;
	}
	std::string var_name() const override {
		return "Range";
	}
	std::vector<ExprRef> repartition_by() const override;
	std::vector<std::string> multiline_display() const override;
	ClusteringSpecRef to_clustering_spec(size_t upstream_num_partitions) const override;

	const std::shared_ptr<RangeRepartitionConfig> &config() const {
		return config_;
	}
};

class RangeClusteringConfig : public BaseConfig {
public:
	size_t num_partitions;
	std::vector<ExprRef> by;
	std::vector<bool> descending;

	RangeClusteringConfig(size_t num_partitions, std::vector<ExprRef> by, std::vector<bool> descending);
	std::vector<std::string> multiline_display() const override;
};

class HashClusteringConfig : public BaseConfig {
public:
	size_t num_partitions;
	std::vector<ExprRef> by;

	HashClusteringConfig(size_t num_partitions, std::vector<ExprRef> by);
	std::vector<std::string> multiline_display() const override;
};

class RandomClusteringConfig : public BaseConfig {
public:
	size_t num_partitions;

	RandomClusteringConfig(size_t num_partitions);
	std::vector<std::string> multiline_display() const override;
};

class UnknownClusteringConfig : public BaseConfig {
public:
	size_t num_partitions;

	UnknownClusteringConfig(size_t num_partitions);
	std::vector<std::string> multiline_display() const override;
};

// 具体的聚类规范实现
class RangeClusteringSpec : public ClusteringSpec {
private:
	RangeClusteringConfig config_;

public:
	RangeClusteringSpec(const RangeClusteringConfig &config);

	Type type() const override {
		return Type::Range;
	}
	std::string var_name() const override {
		return "Range";
	}
	size_t num_partitions() const override;
	std::vector<ExprRef> partition_by() const override;
	std::vector<std::string> multiline_display() const override;

	const RangeClusteringConfig &config() const {
		return config_;
	}
};

class HashClusteringSpec : public ClusteringSpec {
private:
	HashClusteringConfig config_;

public:
	HashClusteringSpec(const HashClusteringConfig &config);

	Type type() const override {
		return Type::Hash;
	}
	std::string var_name() const override {
		return "Hash";
	}
	size_t num_partitions() const override;
	std::vector<ExprRef> partition_by() const override;
	std::vector<std::string> multiline_display() const override;
};

class RandomClusteringSpec : public ClusteringSpec {
private:
	RandomClusteringConfig config_;

public:
	RandomClusteringSpec(const RandomClusteringConfig &config);

	Type type() const override {
		return Type::Random;
	}
	std::string var_name() const override {
		return "Random";
	}
	size_t num_partitions() const override;
	std::vector<ExprRef> partition_by() const override;
	std::vector<std::string> multiline_display() const override;
};

class UnknownClusteringSpec : public ClusteringSpec {
private:
	UnknownClusteringConfig config_;

public:
	UnknownClusteringSpec(const UnknownClusteringConfig &config);

	Type type() const override {
		return Type::Unknown;
	}
	std::string var_name() const override {
		return "Unknown";
	}
	size_t num_partitions() const override;
	std::vector<ExprRef> partition_by() const override;
	std::vector<std::string> multiline_display() const override;
};

// 聚类规范转换函数
ClusteringSpecRef translate_clustering_spec(ClusteringSpecRef input_clustering_spec,
                                            const std::vector<ExprRef> &projection);

// Default implementations for ClusteringSpec that mirror the Rust `impl`.
inline std::string ClusteringSpec::var_name() const {
	switch (type()) {
	case Type::Range:
		return "Range";
	case Type::Hash:
		return "Hash";
	case Type::Random:
		return "Random";
	case Type::Unknown:
	default:
		return "Unknown";
	}
}

inline size_t ClusteringSpec::num_partitions() const {
	switch (type()) {
	case Type::Range:
		return static_cast<const RangeClusteringSpec *>(this)->num_partitions();
	case Type::Hash:
		return static_cast<const HashClusteringSpec *>(this)->num_partitions();
	case Type::Random:
		return static_cast<const RandomClusteringSpec *>(this)->num_partitions();
	case Type::Unknown:
	default:
		return static_cast<const UnknownClusteringSpec *>(this)->num_partitions();
	}
}

inline std::vector<ExprRef> ClusteringSpec::partition_by() const {
	switch (type()) {
	case Type::Range:
		return static_cast<const RangeClusteringSpec *>(this)->partition_by();
	case Type::Hash:
		return static_cast<const HashClusteringSpec *>(this)->partition_by();
	case Type::Random:
	case Type::Unknown:
	default:
		return {};
	}
}

inline std::vector<std::string> ClusteringSpec::multiline_display() const {
	switch (type()) {
	case Type::Range:
		return static_cast<const RangeClusteringSpec *>(this)->multiline_display();
	case Type::Hash:
		return static_cast<const HashClusteringSpec *>(this)->multiline_display();
	case Type::Random:
		return static_cast<const RandomClusteringSpec *>(this)->multiline_display();
	case Type::Unknown:
	default:
		return static_cast<const UnknownClusteringSpec *>(this)->multiline_display();
	}
}

// Inline static factory helpers for operator-level ClusteringSpec (mirror Rust impl)
inline ClusteringSpecRef ClusteringSpec::unknown() {
	return std::make_shared<UnknownClusteringSpec>(UnknownClusteringConfig(0));
}

inline ClusteringSpecRef ClusteringSpec::unknown_with_num_partitions(size_t num_partitions) {
	return std::make_shared<UnknownClusteringSpec>(UnknownClusteringConfig(num_partitions));
}

inline ClusteringSpecRef ClusteringSpec::from_range_config(const RangeClusteringConfig &config) {
	return std::make_shared<RangeClusteringSpec>(config);
}

inline ClusteringSpecRef ClusteringSpec::from_hash_config(const HashClusteringConfig &config) {
	return std::make_shared<HashClusteringSpec>(config);
}

inline ClusteringSpecRef ClusteringSpec::from_random_config(const RandomClusteringConfig &config) {
	return std::make_shared<RandomClusteringSpec>(config);
}

inline ClusteringSpecRef ClusteringSpec::from_unknown_config(const UnknownClusteringConfig &config) {
	return std::make_shared<UnknownClusteringSpec>(config);
}

} // namespace duckdb

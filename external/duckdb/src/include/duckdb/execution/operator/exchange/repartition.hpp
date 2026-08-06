// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/planner/bound_result_modifier.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace duckdb {

class Expression;
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

class BaseConfig {
public:
	virtual ~BaseConfig() = default;
	virtual std::vector<std::string> multiline_display() const = 0;
};

class HashRepartitionConfig : public BaseConfig {
public:
	size_t num_partitions; // 0 means auto
	std::vector<ExprRef> by;

	HashRepartitionConfig(size_t num_partitions, std::vector<ExprRef> by);
	std::vector<std::string> multiline_display() const override;

	static std::shared_ptr<HashRepartitionConfig> create(size_t num_partitions, std::vector<ExprRef> by);
};

class RandomShuffleConfig : public BaseConfig {
public:
	size_t num_partitions; // 0 means auto

	RandomShuffleConfig(size_t num_partitions);
	std::vector<std::string> multiline_display() const override;

	static std::shared_ptr<RandomShuffleConfig> create(size_t num_partitions);
};

class RangeRepartitionConfig : public BaseConfig {
public:
	RangeRepartitionConfig(size_t num_partitions, vector<BoundOrderByNode> orders, vector<string> boundaries);
	std::vector<std::string> multiline_display() const override;

	static std::shared_ptr<const RangeRepartitionConfig> create(size_t num_partitions, vector<BoundOrderByNode> orders,
	                                                            vector<string> boundaries);
	static void Validate(const vector<BoundOrderByNode> &orders, size_t num_partitions,
	                     const vector<string> &boundaries);
	static void ValidateBoundaries(size_t num_partitions, const vector<string> &boundaries);

	size_t num_partitions() const {
		return num_partitions_;
	}
	const vector<BoundOrderByNode> &orders() const {
		return orders_;
	}
	const vector<string> &boundaries() const {
		return boundaries_;
	}
	vector<BoundOrderByNode> CopyOrders() const;

private:
	size_t num_partitions_; // 0 means auto
	vector<BoundOrderByNode> orders_;
	vector<string> boundaries_;
};

class IntoPartitionsConfig : public BaseConfig {
public:
	size_t num_partitions;

	IntoPartitionsConfig(size_t num_partitions);
	std::vector<std::string> multiline_display() const override;

	static std::shared_ptr<IntoPartitionsConfig> create(size_t num_partitions);
};

class RepartitionSpec {
public:
	enum class Type { Hash, Random, IntoPartitions, Range };

	virtual ~RepartitionSpec() = default;

	virtual Type type() const = 0;
	virtual std::string var_name() const = 0;
	virtual std::vector<std::string> multiline_display() const = 0;
	virtual ClusteringSpecRef to_clustering_spec(size_t upstream_num_partitions) const = 0;

	static std::shared_ptr<RepartitionSpec> create_hash(size_t num_partitions, std::vector<ExprRef> by);
	static std::shared_ptr<RepartitionSpec> create_random(size_t num_partitions);
	static std::shared_ptr<RepartitionSpec> create_into_partitions(size_t num_partitions);
	static std::shared_ptr<RepartitionSpec> create_range(size_t num_partitions, vector<BoundOrderByNode> orders,
	                                                     vector<string> boundaries);
};

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
	static ClusteringSpecRef from_range_config(RangeClusteringConfig config);
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
	std::vector<std::string> multiline_display() const override;
	ClusteringSpecRef to_clustering_spec(size_t upstream_num_partitions) const override;

	const std::shared_ptr<IntoPartitionsConfig> &config() const {
		return config_;
	}
};

class RangeRepartitionSpec : public RepartitionSpec {
private:
	std::shared_ptr<const RangeRepartitionConfig> config_;

public:
	explicit RangeRepartitionSpec(std::shared_ptr<const RangeRepartitionConfig> config);

	Type type() const override {
		return Type::Range;
	}
	std::string var_name() const override {
		return "Range";
	}
	std::vector<std::string> multiline_display() const override;
	ClusteringSpecRef to_clustering_spec(size_t upstream_num_partitions) const override;

	const RangeRepartitionConfig &config() const {
		return *config_;
	}
};

class RangeClusteringConfig : public BaseConfig {
public:
	RangeClusteringConfig(size_t num_partitions, vector<BoundOrderByNode> orders);
	RangeClusteringConfig(const RangeClusteringConfig &other);
	RangeClusteringConfig &operator=(const RangeClusteringConfig &other);
	RangeClusteringConfig(RangeClusteringConfig &&other) noexcept = default;
	RangeClusteringConfig &operator=(RangeClusteringConfig &&other) noexcept = default;
	std::vector<std::string> multiline_display() const override;

	size_t num_partitions() const {
		return num_partitions_;
	}
	const vector<BoundOrderByNode> &orders() const {
		return orders_;
	}
	std::vector<ExprRef> partition_by() const;

private:
	size_t num_partitions_;
	vector<BoundOrderByNode> orders_;
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

class RangeClusteringSpec : public ClusteringSpec {
private:
	RangeClusteringConfig config_;

public:
	explicit RangeClusteringSpec(RangeClusteringConfig config);

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

inline ClusteringSpecRef ClusteringSpec::from_range_config(RangeClusteringConfig config) {
	return std::make_shared<RangeClusteringSpec>(std::move(config));
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

// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <new>
#include <type_traits>
#include <utility>

namespace duckdb {
namespace distributed {

struct NullOpt {};

static const NullOpt nullopt = NullOpt();

template <typename T>
class Optional {
public:
	Optional() : initialized_(false) {
	}

	Optional(NullOpt) : initialized_(false) {
	}

	Optional(const T &value) : initialized_(false) {
		Construct(value);
	}

	Optional(T &&value) : initialized_(false) {
		Construct(std::move(value));
	}

	Optional(const Optional &other) : initialized_(false) {
		if (other.initialized_) {
			Construct(*other.Ptr());
		}
	}

	Optional(Optional &&other) noexcept : initialized_(false) {
		if (other.initialized_) {
			Construct(std::move(*other.Ptr()));
			other.reset();
		}
	}

	~Optional() {
		reset();
	}

	Optional &operator=(NullOpt) {
		reset();
		return *this;
	}

	Optional &operator=(const Optional &other) {
		if (this == &other) {
			return *this;
		}
		if (other.initialized_) {
			if (initialized_) {
				*Ptr() = *other.Ptr();
			} else {
				Construct(*other.Ptr());
			}
		} else {
			reset();
		}
		return *this;
	}

	Optional &operator=(Optional &&other) noexcept {
		if (this == &other) {
			return *this;
		}
		if (other.initialized_) {
			if (initialized_) {
				*Ptr() = std::move(*other.Ptr());
			} else {
				Construct(std::move(*other.Ptr()));
			}
			other.reset();
		} else {
			reset();
		}
		return *this;
	}

	Optional &operator=(const T &value) {
		if (initialized_) {
			*Ptr() = value;
		} else {
			Construct(value);
		}
		return *this;
	}

	Optional &operator=(T &&value) {
		if (initialized_) {
			*Ptr() = std::move(value);
		} else {
			Construct(std::move(value));
		}
		return *this;
	}

	bool has_value() const {
		return initialized_;
	}

	explicit operator bool() const {
		return initialized_;
	}

	T &value() {
		return *Ptr();
	}

	const T &value() const {
		return *Ptr();
	}

	T value_or(const T &fallback) const {
		return initialized_ ? *Ptr() : fallback;
	}

	T &operator*() {
		return *Ptr();
	}

	const T &operator*() const {
		return *Ptr();
	}

	T *operator->() {
		return Ptr();
	}

	const T *operator->() const {
		return Ptr();
	}

	void reset() {
		if (initialized_) {
			Ptr()->~T();
			initialized_ = false;
		}
	}

	template <typename... Args>
	void emplace(Args &&...args) {
		reset();
		Construct(std::forward<Args>(args)...);
	}

private:
	template <typename... Args>
	void Construct(Args &&...args) {
		new (&storage_) T(std::forward<Args>(args)...);
		initialized_ = true;
	}

	T *Ptr() {
		return reinterpret_cast<T *>(&storage_);
	}

	const T *Ptr() const {
		return reinterpret_cast<const T *>(&storage_);
	}

	typename std::aligned_storage<sizeof(T), alignof(T)>::type storage_;
	bool initialized_;
};

} // namespace distributed
} // namespace duckdb

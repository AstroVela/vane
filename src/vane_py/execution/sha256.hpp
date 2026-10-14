// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/common/exception.hpp"

#include <openssl/evp.h>
#include <memory>

namespace duckdb {
namespace vane_execution {

// OpenSSL is already a required native dependency. EVP selects its optimized
// implementation while preserving the SHA-256 identities used by snapshots,
// materialized objects and individual frame checksums.
class SHA256State {
public:
	SHA256State() : state(EVP_MD_CTX_new(), EVP_MD_CTX_free) {
		if (!state || EVP_DigestInit_ex(state.get(), EVP_sha256(), nullptr) != 1) {
			throw IOException("cannot initialize exchange SHA-256");
		}
	}

	void AddBytes(const_data_ptr_t data, idx_t size) {
		if (finished || EVP_DigestUpdate(state.get(), data, size) != 1) {
			throw IOException("cannot update exchange SHA-256");
		}
	}

	void FinishHex(char *output) {
		unsigned char digest[EVP_MAX_MD_SIZE];
		unsigned int size = 0;
		if (finished || EVP_DigestFinal_ex(state.get(), digest, &size) != 1 || size != 32) {
			throw IOException("cannot finalize exchange SHA-256");
		}
		finished = true;
		static constexpr char hex[] = "0123456789abcdef";
		for (idx_t i = 0; i < size; i++) {
			output[2 * i] = hex[digest[i] >> 4];
			output[2 * i + 1] = hex[digest[i] & 15];
		}
	}

private:
	std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> state;
	bool finished = false;
};

} // namespace vane_execution
} // namespace duckdb

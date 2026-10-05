// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "direct_exchange.hpp"

namespace duckdb {
namespace vane_execution {

struct MaterializedObject {
	idx_t bytes = 0;
	idx_t rows = 0;
	idx_t frames = 0;
	string sha256;
};

struct MaterializedStatus {
	bool done = false;
	string error;
	MaterializedObject object;
};

// Independent native I/O between a bounded task buffer and a private object.
// Sealing a writer is not a commit. The coordinator selects the visible attempt.
// Close never removes files; their owner is the query store, not this worker.
class MaterializedIO {
public:
	MaterializedIO(bool write, const string &path, shared_ptr<DirectChannel> channel, const string &identity,
	               idx_t max_bytes, idx_t staging_bytes, MaterializedObject expected = {});
	~MaterializedIO();
	void Cancel(const string &reason);
	void Close();
	MaterializedStatus Status() const;
	static idx_t StagingBytes(idx_t frame_bytes);
	static void Verify(const string &path, const MaterializedObject &expected, const string &schema);

private:
	struct Impl;
	unique_ptr<Impl> impl;
};

} // namespace vane_execution
} // namespace duckdb

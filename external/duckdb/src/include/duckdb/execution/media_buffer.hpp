// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/allocator.hpp"
#include "duckdb/execution/media_resource.hpp"

namespace duckdb {
struct LogicalType;
class Vector;
class DataChunk;

//! Allocate primitive pixel storage only after admission. References/slices
//! retain the allocation and its lease; copies belong to the receiving domain.
//! The allocator is retained until after the actual allocation has been freed.
DUCKDB_API unique_ptr<Vector>
AllocateMediaVector(const LogicalType &type, idx_t count, MediaResourceLease lease,
                    shared_ptr<Allocator> allocator = Allocator::DefaultAllocatorReference(),
                    shared_ptr<MediaFlowTicket> flow = nullptr, shared_ptr<void> lifetime = nullptr);

//! Fixed arrays whose primitive child owns both its local lease and the parent
//! working-set lifetime. No dense default-capacity pixel cache is allocated.
DUCKDB_API void AllocateMediaArray(Vector &array, idx_t rows, MediaResourceLease lease,
                                   shared_ptr<void> lifetime = nullptr);

//! Install a fresh list child without retaining it in a reusable VectorCache.
//! The caller must not append beyond count or treat this as admission for copies.
DUCKDB_API void AllocateMediaList(Vector &list, idx_t rows, idx_t count, MediaResourceLease lease,
                                  shared_ptr<MediaFlowTicket> flow = nullptr);

//! Payload reservation for a fresh bounded receiver copy: primitive slots, fixed
//! primitive arrays, structs and out-of-line strings. Variable-sized nested
//! collections require a separate capacity contract and are rejected.
//! Excludes validity masks, allocator slack, bookkeeping and later task copies.
DUCKDB_API idx_t EstimateMediaFlowInputBytes(const DataChunk &chunk);
//! Measure an actual row slice without allocating the receiving payload.
DUCKDB_API idx_t EstimateMediaFlowInputBytes(const DataChunk &chunk, idx_t offset, idx_t rows);
//! Largest prefix that fits, measured using actual selected values.
DUCKDB_API idx_t MediaFlowRowsWithinByteLimit(const DataChunk &chunk, idx_t offset, idx_t rows, idx_t bytes);
} // namespace duckdb

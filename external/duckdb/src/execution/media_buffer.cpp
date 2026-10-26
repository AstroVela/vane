// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/media_buffer.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types/vector_buffer.hpp"

namespace duckdb {
namespace {
class MediaAllocationBuffer : public VectorBuffer {
public:
	MediaAllocationBuffer(shared_ptr<Allocator> allocator_p, MediaResourceLease lease_p)
	    : VectorBuffer(VectorBufferType::OPAQUE_BUFFER), allocator(std::move(allocator_p)), lease(std::move(lease_p)) {
		data = allocator->Allocate(lease.Budget().bytes);
	}
	~MediaAllocationBuffer() override {
		// Derived members die before the base's data. Free explicitly before
		// releasing the lease can wake another allocating producer.
		data.Reset();
	}

private:
	shared_ptr<Allocator> allocator;
	MediaResourceLease lease;
};
} // namespace

unique_ptr<Vector> AllocateMediaVector(const LogicalType &type, idx_t count, MediaResourceLease lease,
                                       shared_ptr<Allocator> allocator) {
	const auto budget = lease.Budget();
	if (!allocator || !lease || budget.decoders != 0 || !type.IsNumeric() || count == 0) {
		throw InvalidInputException("Media pixel allocation requires a byte-only lease and nonempty numeric storage");
	}
	const auto width = GetTypeIdSize(type.InternalType());
	if (budget.bytes / width != count || budget.bytes % width != 0) {
		throw InvalidInputException("Media pixel allocation must exactly match its byte reservation");
	}
	auto owner = make_buffer<MediaAllocationBuffer>(std::move(allocator), std::move(lease));
	auto result = make_uniq<Vector>(type, owner->GetData());
	// This fresh primitive vector has no pre-existing Arrow auxiliary to replace.
	// Its owned auxiliary follows ordinary Reference, slice and nested children.
	result->SetAuxiliary(std::move(owner));
	return result;
}

void AllocateMediaList(Vector &list, idx_t rows, idx_t count, MediaResourceLease lease) {
	auto child = AllocateMediaVector(ListType::GetChildType(list.GetType()), count, std::move(lease));
	Vector fresh(list.GetType(), rows);
	fresh.SetAuxiliary(make_buffer<VectorListBuffer>(std::move(child), count));
	list.Reference(fresh);
}
} // namespace duckdb

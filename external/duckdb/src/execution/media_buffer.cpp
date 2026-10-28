// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/media_buffer.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types/vector_buffer.hpp"

#include <limits>

namespace duckdb {
namespace {
class MediaAllocationBuffer : public VectorBuffer {
public:
	MediaAllocationBuffer(shared_ptr<Allocator> allocator_p, idx_t bytes, MediaResourceLease lease_p,
	                      shared_ptr<MediaFlowTicket> flow_p)
	    : VectorBuffer(VectorBufferType::OPAQUE_BUFFER), allocator(std::move(allocator_p)), flow(std::move(flow_p)),
	      lease(std::move(lease_p)) {
		data = allocator->Allocate(bytes);
	}
	~MediaAllocationBuffer() override {
		// Derived members die before the base's data. Free explicitly before
		// releasing the lease can wake another allocating producer.
		data.Reset();
	}

private:
	shared_ptr<Allocator> allocator;
	shared_ptr<MediaFlowTicket> flow;
	MediaResourceLease lease;
};

idx_t AddInputBytes(idx_t left, idx_t right) {
	if (right > std::numeric_limits<idx_t>::max() - left) {
		throw OutOfMemoryException("Native media UDF input payload size overflow");
	}
	return left + right;
}

idx_t MultiplyInputBytes(idx_t count, idx_t width) {
	if (width && count > std::numeric_limits<idx_t>::max() / width) {
		throw OutOfMemoryException("Native media UDF input payload size overflow");
	}
	return count * width;
}

idx_t MediaFlowVectorInputBytes(const Vector &vector, idx_t rows) {
	const auto &type = vector.GetType();
	const auto physical = type.InternalType();
	if (physical == PhysicalType::ARRAY) {
		const auto &child = ArrayType::GetChildType(type);
		if (child.IsNested() || child.InternalType() == PhysicalType::VARCHAR) {
			throw NotImplementedException("Native media flow receiver requires fixed arrays of primitive values");
		}
		return MultiplyInputBytes(MultiplyInputBytes(rows, ArrayType::GetSize(type)),
		                          GetTypeIdSize(child.InternalType()));
	}
	if (physical == PhysicalType::LIST) {
		throw NotImplementedException("Native media flow receiver does not support variable-sized nested input; "
		                              "cast images to IMAGE(mode, height, width) before the UDF");
	}
	if (physical == PhysicalType::STRUCT) {
		idx_t bytes = 0;
		// Struct slicing has already applied its selection to the children.
		// Applying the parent's dictionary again can read a different row and
		// under-reserve nonuniform strings. Follow RecursiveToUnifiedFormat's
		// child traversal, letting each leaf resolve its own selection.
		for (auto &entry : StructVector::GetEntries(vector)) {
			bytes = AddInputBytes(bytes, MediaFlowVectorInputBytes(*entry, rows));
		}
		return bytes;
	}
	auto bytes = MultiplyInputBytes(rows, GetTypeIdSize(physical));
	if (physical == PhysicalType::VARCHAR) {
		Vector view(type, nullptr);
		view.Reference(vector);
		UnifiedVectorFormat format;
		view.ToUnifiedFormat(rows, format);
		auto strings = UnifiedVectorFormat::GetData<string_t>(format);
		for (idx_t row = 0; row < rows; row++) {
			auto index = format.sel->get_index(row);
			if (format.validity.RowIsValid(index) && !strings[index].IsInlined()) {
				bytes = AddInputBytes(bytes, strings[index].GetSize());
			}
		}
	}
	return bytes;
}
} // namespace

unique_ptr<Vector> AllocateMediaVector(const LogicalType &type, idx_t count, MediaResourceLease lease,
                                       shared_ptr<Allocator> allocator, shared_ptr<MediaFlowTicket> flow) {
	const auto budget = lease.Budget();
	if (!allocator || (!lease && !flow) || budget.decoders != 0 || !type.IsNumeric() || count == 0) {
		throw InvalidInputException("Media pixel allocation requires admission and nonempty numeric storage");
	}
	const auto width = GetTypeIdSize(type.InternalType());
	if (count > std::numeric_limits<idx_t>::max() / width ||
	    (lease && (budget.bytes / width != count || budget.bytes % width != 0))) {
		throw InvalidInputException("Media pixel allocation must exactly match its byte reservation");
	}
	auto owner =
	    make_buffer<MediaAllocationBuffer>(std::move(allocator), count * width, std::move(lease), std::move(flow));
	auto result = make_uniq<Vector>(type, owner->GetData());
	// This fresh primitive vector has no pre-existing Arrow auxiliary to replace.
	// Its owned auxiliary follows ordinary Reference, slice and nested children.
	result->SetAuxiliary(std::move(owner));
	return result;
}

void AllocateMediaList(Vector &list, idx_t rows, idx_t count, MediaResourceLease lease,
                       shared_ptr<MediaFlowTicket> flow) {
	auto child = AllocateMediaVector(ListType::GetChildType(list.GetType()), count, std::move(lease),
	                                 Allocator::DefaultAllocatorReference(), std::move(flow));
	Vector fresh(list.GetType(), rows);
	fresh.SetAuxiliary(make_buffer<VectorListBuffer>(std::move(child), count));
	list.Reference(fresh);
}

idx_t EstimateMediaFlowInputBytes(const DataChunk &chunk) {
	if (!chunk.size()) {
		return 0;
	}
	idx_t bytes = 0;
	for (auto &column : chunk.data) {
		bytes = AddInputBytes(bytes, MediaFlowVectorInputBytes(column, chunk.size()));
	}
	return bytes;
}
} // namespace duckdb

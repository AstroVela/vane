// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/allocator.hpp"
#include "duckdb/common/exception.hpp"

namespace duckdb {

bool ArrayVector::UsesDeferredStorage(const LogicalType &type) {
	if (ImageLogicalType::IsFixedShape(type)) {
		return true;
	}
	if (!TensorType::IsFixedShapeTensor(type)) {
		return false;
	}
	auto &child = ArrayType::GetChildType(type);
	if (child.HasAlias()) {
		return false;
	}
	switch (child.id()) {
	case LogicalTypeId::BOOLEAN:
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
		return true;
	default:
		return false;
	}
}

void ArrayVector::Reserve(Vector &output, idx_t count) {
	if (!UsesDeferredStorage(output.GetType()) || count == 0) {
		return;
	}
	D_ASSERT(output.GetVectorType() == VectorType::FLAT_VECTOR ||
	         output.GetVectorType() == VectorType::CONSTANT_VECTOR);
	auto &buffer = output.GetAuxiliary()->Cast<VectorArrayBuffer>();
	auto width = buffer.GetArraySize();
	auto element_size = GetTypeIdSize(ArrayType::GetChildType(output.GetType()).InternalType());
	if (count > DConstants::MAX_VECTOR_SIZE / width / element_size) {
		throw OutOfMemoryException("Dense array buffer exceeds the maximum vector size");
	}
	auto &validity = output.GetVectorType() == VectorType::CONSTANT_VECTOR ? ConstantVector::Validity(output)
	                                                                       : FlatVector::Validity(output);
	if (count > validity.Capacity()) {
		validity.Resize(count);
	}
	auto current = buffer.GetChildSize();
	auto required = count * width;
	if (required <= current) {
		return;
	}
	auto &elements = buffer.GetChild();
	elements.Flatten(current);
	auto element_buffer = elements.GetBuffer();
	// Logical element count and allocated capacity are distinct. Only reuse an
	// owned, unshared allocation at its original address: slices and Arrow
	// imports may reference another buffer or a smaller window within it.
	const bool owned = element_buffer && element_buffer->GetData() == FlatVector::GetDataUnsafe<uint8_t>(elements) &&
	                   element_buffer.use_count() == 2; // elements plus this local reference
	auto capacity = owned ? element_buffer->GetDataSize() / (width * element_size) : 0;
	if (count <= capacity) {
		FlatVector::Validity(elements).Resize(capacity * width);
		memset(FlatVector::GetDataUnsafe<uint8_t>(elements) + current * element_size, 0,
		       (required - current) * element_size);
		buffer.SetSize(count);
		return;
	}
	// Amortize row-at-a-time writers without reserving a full vector up front.
	// Bound geometric slack by the same limit as the requested element span.
	auto allocated_rows = MaxValue(count, MinValue(capacity * 2, DConstants::MAX_VECTOR_SIZE / width / element_size));
	auto allocated_elements = allocated_rows * width;
	auto stored_allocator = element_buffer ? element_buffer->GetAllocator() : nullptr;
	auto allocation = stored_allocator ? stored_allocator->Allocate(allocated_elements * element_size)
	                                   : Allocator::DefaultAllocator().Allocate(allocated_elements * element_size);
	Vector grown(elements.GetType(), idx_t(0));
	if (current) {
		memcpy(allocation.get(), FlatVector::GetDataUnsafe<uint8_t>(elements), current * element_size);
	}
	// NULL rows and not-yet-written selected rows must not expose heap bytes.
	memset(allocation.get() + current * element_size, 0, (allocated_elements - current) * element_size);
	grown.GetBuffer()->SetData(std::move(allocation));
	FlatVector::SetData(grown, grown.GetBuffer()->GetData());
	FlatVector::Validity(grown).Resize(allocated_elements);
	FlatVector::Validity(grown).SliceInPlace(FlatVector::Validity(elements), 0, 0, current);
	elements.Reference(grown);
	buffer.SetSize(count);
}

void ArrayVector::SetNullElements(Vector &output, idx_t row) {
	Reserve(output, row + 1);
	auto width = ArrayType::GetSize(output.GetType());
	auto element_size = GetTypeIdSize(ArrayType::GetChildType(output.GetType()).InternalType());
	auto &elements = ArrayVector::GetEntry(output);
	memset(FlatVector::GetDataUnsafe<uint8_t>(elements) + row * width * element_size, 0, width * element_size);
	ValidityMask invalid(width);
	invalid.SetAllInvalid(width);
	FlatVector::Validity(elements).SliceInPlace(invalid, row * width, 0, width);
}

void ArrayVector::CopyRows(const Vector &source, Vector &target, const SelectionVector &sel, idx_t source_offset,
                           idx_t target_offset, idx_t count) {
	Reserve(target, target_offset + count);
	auto width = ArrayType::GetSize(target.GetType());
	auto element_size = GetTypeIdSize(ArrayType::GetChildType(target.GetType()).InternalType());
	D_ASSERT(width == ArrayType::GetSize(source.GetType()));
	const bool constant = source.GetVectorType() == VectorType::CONSTANT_VECTOR;
	const auto &validity = FlatVector::Validity(target);
	idx_t source_rows = 0;
	for (idx_t row = 0; row < count; row++) {
		if (validity.RowIsValid(target_offset + row)) {
			source_rows = MaxValue(source_rows, (constant ? 0 : idx_t(sel.get_index(source_offset + row))) + 1);
		}
	}
	Vector source_elements(ArrayVector::GetEntry(source).GetType(), nullptr);
	source_elements.Reference(ArrayVector::GetEntry(source));
	if (source_rows) {
		source_elements.Flatten(source_rows * width);
	}
	auto &target_elements = ArrayVector::GetEntry(target);
	auto target_data = FlatVector::GetDataUnsafe<uint8_t>(target_elements);
	for (idx_t row = 0; row < count; row++) {
		auto target_row = target_offset + row;
		if (!validity.RowIsValid(target_row)) {
			SetNullElements(target, target_row);
			continue;
		}
		auto source_row = constant ? 0 : idx_t(sel.get_index(source_offset + row));
		memmove(target_data + target_row * width * element_size,
		        FlatVector::GetDataUnsafe<uint8_t>(source_elements) + source_row * width * element_size,
		        width * element_size);
		if (!FlatVector::Validity(source_elements).AllValid() || !FlatVector::Validity(target_elements).AllValid()) {
			// An implicit all-valid mask has no offsettable data buffer.
			auto validity_offset = FlatVector::Validity(source_elements).AllValid() ? 0 : source_row * width;
			FlatVector::Validity(target_elements)
			    .SliceInPlace(FlatVector::Validity(source_elements), target_row * width, validity_offset, width);
		}
	}
}

} // namespace duckdb

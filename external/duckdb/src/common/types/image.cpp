// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/common/types/image.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/extra_type_info.hpp"
#include "duckdb/common/type_visitor.hpp"

namespace duckdb {

static unique_ptr<ExtensionTypeInfo> ImageTypeModifiers(const string &mode) {
	auto info = make_uniq<ExtensionTypeInfo>();
	LogicalTypeModifier modifier {Value(mode)};
	modifier.label = "'" + mode + "'";
	info->modifiers.push_back(std::move(modifier));
	return info;
}

LogicalType ImageLogicalType::Create() {
	auto result = LogicalType::STRUCT({{"data", LogicalType::LIST(LogicalType::UTINYINT)},
	                                   {"channel", LogicalType::USMALLINT},
	                                   {"height", LogicalType::UINTEGER},
	                                   {"width", LogicalType::UINTEGER},
	                                   {"mode", LogicalType::UTINYINT}});
	result.SetAlias(TYPE_NAME);
	return result;
}

LogicalType ImageLogicalType::Create(const string &mode) {
	ChannelsForMode(mode);
	auto result = Create();
	result.SetExtensionInfo(ImageTypeModifiers(mode));
	return result;
}

LogicalType ImageLogicalType::Create(const string &mode, uint32_t height, uint32_t width) {
	auto channels = ChannelsForMode(mode);
	if (!width || !height || uint64_t(width) * height > uint64_t(NumericLimits<int32_t>::Maximum()) / channels) {
		throw InvalidInputException("Fixed-shape IMAGE requires positive dimensions and at most 2147483647 pixel values");
	}
	// IMAGE, like TENSOR, permits fixed pixel arrays beyond SQL ARRAY's 100k limit.
	auto info = make_shared_ptr<ArrayTypeInfo>(LogicalType::UTINYINT, idx_t(width) * height * channels);
	auto result = LogicalType(LogicalTypeId::ARRAY, std::move(info));
	result.SetAlias(TYPE_NAME);
	auto modifiers = ImageTypeModifiers(mode);
	modifiers->modifiers.emplace_back(Value::UINTEGER(height));
	modifiers->modifiers.emplace_back(Value::UINTEGER(width));
	result.SetExtensionInfo(std::move(modifiers));
	return result;
}

bool ImageLogicalType::IsImage(const LogicalType &type) {
	if (!type.HasAlias() || type.GetAlias() != TYPE_NAME || !type.AuxInfo()) {
		return false;
	}
	if (type.id() == LogicalTypeId::ARRAY) {
		return type.AuxInfo()->type == ExtraTypeInfoType::ARRAY_TYPE_INFO &&
		       ArrayType::GetChildType(type) == LogicalType::UTINYINT && type.HasExtensionInfo();
	}
	if (type.id() != LogicalTypeId::STRUCT || type.AuxInfo()->type != ExtraTypeInfoType::STRUCT_TYPE_INFO) {
		return false;
	}
	auto &fields = StructType::GetChildTypes(type);
	return fields.size() == FIELD_COUNT && fields[DATA].first == "data" &&
	       fields[DATA].second == LogicalType::LIST(LogicalType::UTINYINT) && fields[CHANNELS].first == "channel" &&
	       fields[CHANNELS].second == LogicalType::USMALLINT && fields[HEIGHT].first == "height" &&
	       fields[HEIGHT].second == LogicalType::UINTEGER && fields[WIDTH].first == "width" &&
	       fields[WIDTH].second == LogicalType::UINTEGER && fields[MODE].first == "mode" &&
	       fields[MODE].second == LogicalType::UTINYINT;
}

bool ImageLogicalType::IsFixedShape(const LogicalType &type) {
	return IsImage(type) && type.id() == LogicalTypeId::ARRAY;
}

string ImageLogicalType::GetMode(const LogicalType &type) {
	if (!IsImage(type)) {
		throw InvalidInputException("Expected IMAGE type, got %s", type);
	}
	if (!type.HasExtensionInfo()) {
		return string();
	}
	auto &modifiers = type.GetExtensionInfo()->modifiers;
	auto count = IsFixedShape(type) ? 3 : 1;
	if (modifiers.size() != idx_t(count) || modifiers[0].value.IsNull() ||
	    modifiers[0].value.type() != LogicalType::VARCHAR) {
		throw InvalidInputException("Malformed IMAGE type metadata");
	}
	auto mode = modifiers[0].value.GetValue<string>();
	ChannelsForMode(mode);
	return mode;
}

static uint32_t ImageDimension(const LogicalType &type, idx_t index) {
	if (!ImageLogicalType::IsFixedShape(type)) {
		throw InvalidInputException("IMAGE dimensions require a fixed-shape type");
	}
	ImageLogicalType::GetMode(type);
	auto &value = type.GetExtensionInfo()->modifiers[index].value;
	if (value.IsNull() || value.type() != LogicalType::UINTEGER || !value.GetValue<uint32_t>()) {
		throw InvalidInputException("Malformed fixed-shape IMAGE dimensions");
	}
	return value.GetValue<uint32_t>();
}

uint32_t ImageLogicalType::GetHeight(const LogicalType &type) {
	return ImageDimension(type, 1);
}

uint32_t ImageLogicalType::GetWidth(const LogicalType &type) {
	return ImageDimension(type, 2);
}

LogicalType ImageLogicalType::CommonType(const LogicalType &left, const LogicalType &right) {
	if (left == right) {
		return left;
	}
	auto mode = GetMode(left);
	return !mode.empty() && mode == GetMode(right) ? Create(mode) : Create();
}

bool ImageLogicalType::CanWiden(const LogicalType &source, const LogicalType &target) {
	return source == target || (!IsFixedShape(target) &&
	                           (GetMode(target).empty() || GetMode(source) == GetMode(target)));
}

uint8_t ImageLogicalType::ChannelsForMode(const string &mode) {
	return ModeCode(mode);
}

uint8_t ImageLogicalType::ModeCode(const string &mode) {
	if (mode == "L") {
		return 1;
	}
	if (mode == "LA") {
		return 2;
	}
	if (mode == "RGB") {
		return 3;
	}
	if (mode == "RGBA") {
		return 4;
	}
	throw InvalidInputException("IMAGE mode must be one of L, LA, RGB, or RGBA, got '%s'", mode);
}

string ImageLogicalType::ModeName(uint8_t mode) {
	switch (mode) {
	case 1:
		return "L";
	case 2:
		return "LA";
	case 3:
		return "RGB";
	case 4:
		return "RGBA";
	default:
		throw InvalidInputException("IMAGE contains an unsupported mode code %d", mode);
	}
}

void ImageLogicalType::ValidateShape(const LogicalType &type, uint32_t width, uint32_t height, const string &mode,
                                     const string &boundary) {
	auto required_mode = GetMode(type);
	if ((!required_mode.empty() && required_mode != mode) ||
	    (IsFixedShape(type) && (GetWidth(type) != width || GetHeight(type) != height))) {
		throw InvalidInputException("%s() IMAGE layout %s %dx%d does not match %s", boundary, mode, height, width, type);
	}
	if (IsFixedShape(type) && uint64_t(width) * height * ChannelsForMode(mode) != ArrayType::GetSize(type)) {
		throw InvalidInputException("%s() received malformed fixed-shape IMAGE storage", boundary);
	}
}

void ImageLogicalType::ValidateFields(idx_t size, uint32_t width, uint32_t height, uint16_t channels,
                                      const string &mode, const string &boundary) {
	if (!width || !height) {
		throw InvalidInputException("%s() IMAGE width and height must be positive", boundary);
	}
	if (channels != ChannelsForMode(mode)) {
		throw InvalidInputException("%s() IMAGE mode %s requires %d channels, got %d", boundary, mode,
		                            ChannelsForMode(mode), channels);
	}
	if (uint64_t(width) * height > NumericLimits<idx_t>::Maximum() / channels) {
		throw InvalidInputException("%s() IMAGE dimensions exceed addressable storage", boundary);
	}
	auto expected = idx_t(width) * height * channels;
	if (size != expected) {
		throw InvalidInputException("%s() IMAGE data has %d pixel values, expected %d for %dx%d %s", boundary, size,
		                            expected, width, height, mode);
	}
}

ImageLayout ImageVector::Layout(const Value &value) {
	if (ImageLogicalType::IsFixedShape(value.type())) {
		auto mode = ImageLogicalType::GetMode(value.type());
		return {ImageLogicalType::GetWidth(value.type()), ImageLogicalType::GetHeight(value.type()),
		        ImageLogicalType::ChannelsForMode(mode), ImageLogicalType::ModeCode(mode)};
	}
	auto &fields = StructValue::GetChildren(value);
	for (auto &field : fields) {
		if (field.IsNull()) {
			throw InvalidInputException("Non-NULL IMAGE values cannot contain NULL fields");
		}
	}
	return {fields[ImageLogicalType::WIDTH].GetValue<uint32_t>(), fields[ImageLogicalType::HEIGHT].GetValue<uint32_t>(),
	        fields[ImageLogicalType::CHANNELS].GetValue<uint16_t>(), fields[ImageLogicalType::MODE].GetValue<uint8_t>()};
}

const vector<Value> &ImageVector::Pixels(const Value &value) {
	return ImageLogicalType::IsFixedShape(value.type())
	           ? ArrayValue::GetChildren(value)
	           : ListValue::GetChildren(StructValue::GetChildren(value)[ImageLogicalType::DATA]);
}

void ImageLogicalType::ValidateValue(const Value &value, const string &boundary) {
	if (value.IsNull() || !TypeVisitor::Contains(value.type(), IsImage)) {
		return;
	}
	if (IsImage(value.type())) {
		auto layout = ImageVector::Layout(value);
		auto &pixels = ImageVector::Pixels(value);
		auto mode = ModeName(layout.mode);
		ValidateFields(pixels.size(), layout.width, layout.height, layout.channels, mode, boundary);
		ValidateShape(value.type(), layout.width, layout.height, mode, boundary);
		for (auto &pixel : pixels) {
			if (pixel.IsNull() || pixel.type() != LogicalType::UTINYINT) {
				throw InvalidInputException("%s() IMAGE pixels must be non-NULL UInt8 values", boundary);
			}
		}
		return;
	}
	const vector<Value> *children;
	switch (value.type().InternalType()) {
	case PhysicalType::STRUCT:
		children = &StructValue::GetChildren(value);
		break;
	case PhysicalType::LIST:
		children = &ListValue::GetChildren(value);
		break;
	case PhysicalType::ARRAY:
		children = &ArrayValue::GetChildren(value);
		break;
	default:
		throw InternalException("IMAGE value is nested in unsupported physical type %s", value.type());
	}
	for (auto &child : *children) {
		ValidateValue(child, boundary);
	}
}

ImageLayout ImageVector::Layout(Vector &input, idx_t row) {
	if (ImageLogicalType::IsFixedShape(input.GetType())) {
		auto &type = input.GetType();
		auto mode = ImageLogicalType::GetMode(type);
		return {ImageLogicalType::GetWidth(type), ImageLogicalType::GetHeight(type),
		        ImageLogicalType::ChannelsForMode(mode), ImageLogicalType::ModeCode(mode)};
	}
	auto &fields = StructVector::GetEntries(input);
	for (auto &field : fields) {
		if (FlatVector::IsNull(*field, row)) {
			throw InvalidInputException("Non-NULL IMAGE values cannot contain NULL fields");
		}
	}
	return {FlatVector::GetData<uint32_t>(*fields[ImageLogicalType::WIDTH])[row],
	        FlatVector::GetData<uint32_t>(*fields[ImageLogicalType::HEIGHT])[row],
	        FlatVector::GetData<uint16_t>(*fields[ImageLogicalType::CHANNELS])[row],
	        FlatVector::GetData<uint8_t>(*fields[ImageLogicalType::MODE])[row]};
}

static pair<Vector *, list_entry_t> ImagePixelRange(Vector &input, idx_t row) {
	if (ImageLogicalType::IsFixedShape(input.GetType())) {
		auto size = ArrayType::GetSize(input.GetType());
		return {&ArrayVector::GetEntry(input), list_entry_t(row * size, size)};
	}
	auto &data = *StructVector::GetEntries(input)[ImageLogicalType::DATA];
	return {&ListVector::GetEntry(data), FlatVector::GetData<list_entry_t>(data)[row]};
}

const_data_ptr_t ImageVector::Pixels(Vector &input, idx_t row) {
	auto range = ImagePixelRange(input, row);
	return FlatVector::GetData<uint8_t>(*range.first) + range.second.offset;
}

void ImageVector::Flatten(Vector &input, idx_t count) {
	input.Flatten(count);
	if (!ImageLogicalType::IsImage(input.GetType())) {
		return;
	}
	idx_t pixel_count;
	if (ImageLogicalType::IsFixedShape(input.GetType())) {
		pixel_count = count * ArrayType::GetSize(input.GetType());
		ArrayVector::GetEntry(input).Flatten(pixel_count);
	} else {
		auto &fields = StructVector::GetEntries(input);
		for (auto &field : fields) {
			field->Flatten(count);
		}
		pixel_count = ListVector::GetListSize(*fields[ImageLogicalType::DATA]);
		ListVector::GetEntry(*fields[ImageLogicalType::DATA]).Flatten(pixel_count);
	}
}

void ImageVector::ValidateRows(Vector &input, const vector<idx_t> &rows, const string &boundary) {
	if (rows.empty() || !TypeVisitor::Contains(input.GetType(), ImageLogicalType::IsImage)) {
		return;
	}
	idx_t count = 0;
	for (auto row : rows) {
		count = MaxValue(count, row + 1);
	}
	input.Flatten(count);
	if (!ImageLogicalType::IsImage(input.GetType())) {
		vector<idx_t> active;
		for (auto row : rows) {
			if (!FlatVector::IsNull(input, row)) {
				active.push_back(row);
			}
		}
		if (active.empty()) {
			return;
		}
		switch (input.GetType().id()) {
		case LogicalTypeId::STRUCT:
			for (auto &child : StructVector::GetEntries(input)) {
				ValidateRows(*child, active, boundary);
			}
			break;
		case LogicalTypeId::LIST:
		case LogicalTypeId::MAP: {
			auto entries = FlatVector::GetData<list_entry_t>(input);
			auto child_count = ListVector::GetListSize(input);
			vector<bool> selected(child_count, false);
			for (auto row : active) {
				auto entry = entries[row];
				if (entry.offset > child_count || entry.length > child_count - entry.offset) {
					throw InvalidInputException("%s: invalid LIST offsets around IMAGE", boundary);
				}
				for (idx_t child = entry.offset; child < entry.offset + entry.length; child++) {
					selected[child] = true;
				}
			}
			vector<idx_t> child_rows;
			for (idx_t child = 0; child < child_count; child++) {
				if (selected[child]) {
					child_rows.push_back(child);
				}
			}
			ValidateRows(ListVector::GetEntry(input), child_rows, boundary);
			break;
		}
		case LogicalTypeId::ARRAY: {
			auto width = ArrayType::GetSize(input.GetType());
			vector<idx_t> child_rows;
			for (auto row : active) {
				for (idx_t child = row * width; child < (row + 1) * width; child++) {
					child_rows.push_back(child);
				}
			}
			ValidateRows(ArrayVector::GetEntry(input), child_rows, boundary);
			break;
		}
		case LogicalTypeId::UNION: {
			vector<vector<idx_t>> member_rows(UnionType::GetMemberCount(input.GetType()));
			for (auto row : active) {
				union_tag_t tag;
				if (!UnionVector::TryGetTag(input, row, tag) || tag >= member_rows.size()) {
					throw InvalidInputException("%s: invalid UNION tag around IMAGE", boundary);
				}
				member_rows[tag].push_back(row);
			}
			for (idx_t member = 0; member < member_rows.size(); member++) {
				ValidateRows(UnionVector::GetMember(input, member), member_rows[member], boundary);
			}
			break;
		}
		default:
			throw InternalException("IMAGE nested in unsupported type %s", input.GetType());
		}
		return;
	}
	Flatten(input, count);
	auto pixel_count = ImageLogicalType::IsFixedShape(input.GetType())
	                       ? count * ArrayType::GetSize(input.GetType())
	                       : ListVector::GetListSize(*StructVector::GetEntries(input)[ImageLogicalType::DATA]);

	for (auto row : rows) {
		if (FlatVector::IsNull(input, row)) {
			continue;
		}
		auto layout = Layout(input, row);
		auto range = ImagePixelRange(input, row);
		if (range.second.offset > pixel_count || range.second.length > pixel_count - range.second.offset) {
			throw InvalidInputException("%s() IMAGE contains invalid pixel offsets", boundary);
		}
		auto mode = ImageLogicalType::ModeName(layout.mode);
		ImageLogicalType::ValidateFields(range.second.length, layout.width, layout.height, layout.channels, mode, boundary);
		ImageLogicalType::ValidateShape(input.GetType(), layout.width, layout.height, mode, boundary);
		auto &validity = FlatVector::Validity(*range.first);
		if (!validity.AllValid()) {
			for (idx_t i = range.second.offset; i < range.second.offset + range.second.length; i++) {
				if (!validity.RowIsValid(i)) {
					throw InvalidInputException("%s() IMAGE pixels cannot contain NULL", boundary);
				}
			}
		}
	}
}

data_ptr_t ImageVector::Allocate(Vector &output, idx_t row, uint32_t width, uint32_t height, const string &mode) {
	auto channels = ImageLogicalType::ChannelsForMode(mode);
	ImageLogicalType::ValidateShape(output.GetType(), width, height, mode, "IMAGE output");
	auto size = idx_t(width) * height * channels;
	ImageLogicalType::ValidateFields(size, width, height, channels, mode, "IMAGE output");
	if (!ImageLogicalType::IsFixedShape(output.GetType())) {
		auto &fields = StructVector::GetEntries(output);
		auto &data = *fields[ImageLogicalType::DATA];
		auto offset = ListVector::GetListSize(data);
		if (size > NumericLimits<idx_t>::Maximum() - offset) {
			throw OutOfMemoryException("IMAGE pixel vector exceeds addressable storage");
		}
		ListVector::Reserve(data, offset + size);
		ListVector::SetListSize(data, offset + size);
		FlatVector::GetData<list_entry_t>(data)[row] = list_entry_t(offset, size);
		FlatVector::GetData<uint16_t>(*fields[ImageLogicalType::CHANNELS])[row] = channels;
		FlatVector::GetData<uint32_t>(*fields[ImageLogicalType::HEIGHT])[row] = height;
		FlatVector::GetData<uint32_t>(*fields[ImageLogicalType::WIDTH])[row] = width;
		FlatVector::GetData<uint8_t>(*fields[ImageLogicalType::MODE])[row] = ImageLogicalType::ModeCode(mode);
		for (auto &field : fields) {
			FlatVector::SetNull(*field, row, false);
		}
	}
	FlatVector::SetNull(output, row, false);
	auto range = ImagePixelRange(output, row);
	auto &validity = FlatVector::Validity(*range.first);
	if (!validity.AllValid()) {
		for (idx_t i = range.second.offset; i < range.second.offset + size; i++) {
			validity.SetValid(i);
		}
	}
	return FlatVector::GetData<uint8_t>(*range.first) + range.second.offset;
}

Value ImageVector::FromPixels(vector<Value> pixels, uint32_t width, uint32_t height, const string &mode,
                              const LogicalType &type) {
	ImageLogicalType::ValidateFields(pixels.size(), width, height, ImageLogicalType::ChannelsForMode(mode), mode, "IMAGE");
	ImageLogicalType::ValidateShape(type, width, height, mode, "IMAGE");
	if (ImageLogicalType::IsFixedShape(type)) {
		auto result = Value::ARRAY(LogicalType::UTINYINT, std::move(pixels));
		result.Reinterpret(type);
		return result;
	}
	return Value::STRUCT(type, {Value::LIST(LogicalType::UTINYINT, std::move(pixels)),
	                            Value::USMALLINT(ImageLogicalType::ChannelsForMode(mode)), Value::UINTEGER(height),
	                            Value::UINTEGER(width), Value::UTINYINT(ImageLogicalType::ModeCode(mode))});
}

} // namespace duckdb

// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/common/types/image.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/extra_type_info.hpp"
#include "duckdb/common/type_visitor.hpp"

#include <cmath>

namespace duckdb {

static unique_ptr<ExtensionTypeInfo> ImageTypeModifiers(const string &mode) {
	auto info = make_uniq<ExtensionTypeInfo>();
	LogicalTypeModifier modifier {Value(mode)};
	modifier.label = "'" + mode + "'";
	info->modifiers.push_back(std::move(modifier));
	return info;
}

static LogicalType DynamicImageType(const LogicalType &pixel) {
	auto result = LogicalType::STRUCT({{"data", LogicalType::LIST(pixel)},
	                                   {"channel", LogicalType::USMALLINT},
	                                   {"height", LogicalType::UINTEGER},
	                                   {"width", LogicalType::UINTEGER},
	                                   {"mode", LogicalType::UTINYINT}});
	result.SetAlias(ImageLogicalType::TYPE_NAME);
	return result;
}

LogicalType ImageLogicalType::Create() {
	return DynamicImageType(LogicalType::FLOAT);
}

LogicalType ImageLogicalType::Create(const string &mode) {
	auto result = DynamicImageType(PixelType(mode));
	result.SetExtensionInfo(ImageTypeModifiers(mode));
	return result;
}

LogicalType ImageLogicalType::Create(const string &mode, uint32_t height, uint32_t width) {
	auto channels = ChannelsForMode(mode);
	if (!width || !height || uint64_t(width) * height > uint64_t(NumericLimits<int32_t>::Maximum()) / channels) {
		throw InvalidInputException(
		    "Fixed-shape IMAGE requires positive dimensions and at most 2147483647 pixel values");
	}
	// IMAGE, like TENSOR, permits fixed pixel arrays beyond SQL ARRAY's 100k limit.
	auto info = make_shared_ptr<ArrayTypeInfo>(PixelType(mode), idx_t(width) * height * channels);
	auto result = LogicalType(LogicalTypeId::ARRAY, std::move(info));
	result.SetAlias(TYPE_NAME);
	auto modifiers = ImageTypeModifiers(mode);
	modifiers->modifiers.emplace_back(Value::UINTEGER(height));
	modifiers->modifiers.emplace_back(Value::UINTEGER(width));
	result.SetExtensionInfo(std::move(modifiers));
	return result;
}

static bool IsPixelType(const LogicalType &type) {
	return type == LogicalType::UTINYINT || type == LogicalType::USMALLINT || type == LogicalType::FLOAT;
}

bool ImageLogicalType::IsImage(const LogicalType &type) {
	if (!type.HasAlias() || type.GetAlias() != TYPE_NAME || !type.AuxInfo()) {
		return false;
	}
	if (type.id() == LogicalTypeId::ARRAY) {
		return type.AuxInfo()->type == ExtraTypeInfoType::ARRAY_TYPE_INFO &&
		       IsPixelType(ArrayType::GetChildType(type)) && type.HasExtensionInfo();
	}
	if (type.id() != LogicalTypeId::STRUCT || type.AuxInfo()->type != ExtraTypeInfoType::STRUCT_TYPE_INFO) {
		return false;
	}
	auto &fields = StructType::GetChildTypes(type);
	return fields.size() == FIELD_COUNT && fields[DATA].first == "data" &&
	       fields[DATA].second.id() == LogicalTypeId::LIST &&
	       IsPixelType(ListType::GetChildType(fields[DATA].second)) && fields[CHANNELS].first == "channel" &&
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
		if (StorageType(type) != LogicalType::FLOAT) {
			throw InvalidInputException("Generic IMAGE requires Float32 storage");
		}
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
	if (StorageType(type) != PixelType(mode)) {
		throw InvalidInputException("IMAGE pixel storage does not match its declared mode");
	}
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
	return source == target ||
	       (!IsFixedShape(target) && (GetMode(target).empty() || GetMode(source) == GetMode(target)));
}

static const char *IMAGE_MODES[] = {"L", "LA", "RGB", "RGBA", "L16", "LA16", "RGB16", "RGBA16", "RGB32F", "RGBA32F"};

uint8_t ImageLogicalType::ChannelsForMode(const string &mode) {
	auto code = ModeCode(mode);
	return code <= 8 ? uint8_t((code - 1) % 4 + 1) : uint8_t(code - 6);
}

LogicalType ImageLogicalType::PixelType(const string &mode) {
	auto code = ModeCode(mode);
	return code <= 4 ? LogicalType::UTINYINT : code <= 8 ? LogicalType::USMALLINT : LogicalType::FLOAT;
}

LogicalType ImageLogicalType::StorageType(const LogicalType &type) {
	if (!IsImage(type)) {
		throw InvalidInputException("Expected IMAGE type, got %s", type);
	}
	return type.id() == LogicalTypeId::ARRAY ? ArrayType::GetChildType(type)
	                                         : ListType::GetChildType(StructType::GetChildType(type, DATA));
}

idx_t ImageLogicalType::ElementSize(const string &mode) {
	return GetTypeIdSize(PixelType(mode).InternalType());
}

uint8_t ImageLogicalType::ModeCode(const string &mode) {
	for (uint8_t i = 0; i < 10; i++) {
		if (mode == IMAGE_MODES[i]) {
			return i + 1;
		}
	}
	throw InvalidInputException(
	    "IMAGE mode must be L, LA, RGB, RGBA, L16, LA16, RGB16, RGBA16, RGB32F, or RGBA32F, got '%s'", mode);
}

string ImageLogicalType::ModeName(uint8_t mode) {
	if (!mode || mode > 10) {
		throw InvalidInputException("IMAGE contains an unsupported mode code %d", mode);
	}
	return IMAGE_MODES[mode - 1];
}

void ImageLogicalType::ValidateShape(const LogicalType &type, uint32_t width, uint32_t height, const string &mode,
                                     const string &boundary) {
	auto required_mode = GetMode(type);
	if ((!required_mode.empty() && required_mode != mode) ||
	    (IsFixedShape(type) && (GetWidth(type) != width || GetHeight(type) != height))) {
		throw InvalidInputException("%s() IMAGE layout %s %dx%d does not match %s", boundary, mode, height, width,
		                            type);
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
	if (uint64_t(width) * height > NumericLimits<idx_t>::Maximum() / channels / sizeof(float)) {
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
	        fields[ImageLogicalType::CHANNELS].GetValue<uint16_t>(),
	        fields[ImageLogicalType::MODE].GetValue<uint8_t>()};
}

const vector<Value> &ImageVector::Pixels(const Value &value) {
	return ImageLogicalType::IsFixedShape(value.type())
	           ? ArrayValue::GetChildren(value)
	           : ListValue::GetChildren(StructValue::GetChildren(value)[ImageLogicalType::DATA]);
}

const Value &ImageVector::PixelValues(const Value &value) {
	return ImageLogicalType::IsFixedShape(value.type()) ? value
	                                                    : StructValue::GetChildren(value)[ImageLogicalType::DATA];
}

template <class T>
static T ReadPixel(const_data_ptr_t source, idx_t index) {
	T value;
	memcpy(&value, source + index * sizeof(T), sizeof(T));
	return value;
}

void ImageVector::ValidatePixels(const_data_ptr_t source, const LogicalType &storage_type, idx_t count,
                                 const string &mode, const string &boundary) {
	auto pixel_type = ImageLogicalType::PixelType(mode);
	if (storage_type != LogicalType::FLOAT) {
		if (pixel_type != storage_type) {
			throw InvalidInputException("%s IMAGE pixel dtype does not match its mode", boundary);
		}
		return;
	}
	auto maximum = pixel_type == LogicalType::UTINYINT ? 255.0f : 65535.0f;
	for (idx_t i = 0; i < count; i++) {
		auto value = ReadPixel<float>(source, i);
		if (!std::isfinite(value) ||
		    (pixel_type != LogicalType::FLOAT && (value < 0 || value > maximum || std::floor(value) != value))) {
			throw InvalidInputException("%s IMAGE pixels must be finite and exactly representable in mode %s", boundary,
			                            mode);
		}
	}
}

void ImageVector::CopyPixels(const_data_ptr_t source, const LogicalType &source_type, data_ptr_t target,
                             const LogicalType &target_type, idx_t count) {
	if (source_type == target_type) {
		memcpy(target, source, count * GetTypeIdSize(source_type.InternalType()));
		return;
	}
	for (idx_t i = 0; i < count; i++) {
		float value = source_type == LogicalType::UTINYINT    ? source[i]
		              : source_type == LogicalType::USMALLINT ? float(ReadPixel<uint16_t>(source, i))
		                                                      : ReadPixel<float>(source, i);
		if (target_type == LogicalType::FLOAT) {
			memcpy(target + i * sizeof(float), &value, sizeof(float));
		} else {
			auto maximum = target_type == LogicalType::UTINYINT ? 255.0f : 65535.0f;
			if (!std::isfinite(value) || value < 0 || value > maximum || std::floor(value) != value) {
				throw InvalidInputException("IMAGE storage conversion would change pixel values");
			}
			if (target_type == LogicalType::UTINYINT) {
				target[i] = uint8_t(value);
			} else {
				auto pixel = uint16_t(value);
				memcpy(target + i * sizeof(pixel), &pixel, sizeof(pixel));
			}
		}
	}
}

void ImageVector::CopyPixels(const Value &value, data_ptr_t target) {
	auto mode = ImageLogicalType::ModeName(Layout(value).mode);
	auto pixel_type = ImageLogicalType::PixelType(mode);
	if (auto bytes = ByteSequenceValue::TryGet(PixelValues(value))) {
		CopyPixels(const_data_ptr_cast(bytes->data()), ImageLogicalType::StorageType(value.type()), target, pixel_type,
		           Layout(value).Size());
		return;
	}
	idx_t i = 0;
	for (auto &pixel : Pixels(value)) {
		auto number = pixel.GetValue<float>();
		CopyPixels(const_data_ptr_cast(&number), LogicalType::FLOAT, target + i++ * ImageLogicalType::ElementSize(mode),
		           pixel_type, 1);
	}
}

void ImageVector::WritePixels(Vector &output, idx_t row, uint32_t width, uint32_t height, const string &mode,
                              const_data_ptr_t source) {
	auto pixel_type = ImageLogicalType::PixelType(mode);
	auto count = idx_t(width) * height * ImageLogicalType::ChannelsForMode(mode);
	ImageLogicalType::ValidateFields(count, width, height, ImageLogicalType::ChannelsForMode(mode), mode,
	                                 "IMAGE output");
	ValidatePixels(source, pixel_type, count, mode, "IMAGE output");
	auto target = Allocate(output, row, width, height, mode);
	CopyPixels(source, pixel_type, target, ImageLogicalType::StorageType(output.GetType()), count);
}

Value ImageVector::GetValue(const Vector &input, idx_t row) {
	SelectionVector selection(1);
	selection.set_index(0, row);
	Vector selected(input, selection, 1);
	ValidateRows(selected, {0}, "IMAGE scalar extraction");
	if (FlatVector::IsNull(selected, 0)) {
		return Value(input.GetType());
	}
	auto layout = Layout(selected, 0);
	return FromPixels(Pixels(selected, 0), layout.Size(), layout.width, layout.height,
	                  ImageLogicalType::ModeName(layout.mode), input.GetType(), true);
}

void ImageLogicalType::ValidateValue(const Value &value, const string &boundary) {
	if (value.IsNull() || !TypeVisitor::Contains(value.type(), IsImage)) {
		return;
	}
	if (IsImage(value.type())) {
		auto layout = ImageVector::Layout(value);
		auto mode = ModeName(layout.mode);
		if (auto bytes = ByteSequenceValue::TryGet(ImageVector::PixelValues(value))) {
			auto storage = StorageType(value.type());
			ValidateFields(bytes->size() / GetTypeIdSize(storage.InternalType()), layout.width, layout.height,
			               layout.channels, mode, boundary);
			ImageVector::ValidatePixels(const_data_ptr_cast(bytes->data()), storage, layout.Size(), mode, boundary);
			ValidateShape(value.type(), layout.width, layout.height, mode, boundary);
			return;
		}
		auto &pixels = ImageVector::Pixels(value);
		ValidateFields(pixels.size(), layout.width, layout.height, layout.channels, mode, boundary);
		ValidateShape(value.type(), layout.width, layout.height, mode, boundary);
		for (auto &pixel : pixels) {
			if (pixel.IsNull() || pixel.type() != StorageType(value.type())) {
				throw InvalidInputException("%s() IMAGE pixels must be non-NULL values of its storage dtype", boundary);
			}
			auto number = pixel.GetValue<float>();
			ImageVector::ValidatePixels(const_data_ptr_cast(&number), LogicalType::FLOAT, 1, mode, boundary);
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
	return FlatVector::GetData(*range.first) +
	       range.second.offset * GetTypeIdSize(range.first->GetType().InternalType());
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
		ImageLogicalType::ValidateFields(range.second.length, layout.width, layout.height, layout.channels, mode,
		                                 boundary);
		ImageLogicalType::ValidateShape(input.GetType(), layout.width, layout.height, mode, boundary);
		ValidatePixels(Pixels(input, row), ImageLogicalType::StorageType(input.GetType()), layout.Size(), mode,
		               boundary);
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
	ArrayVector::Reserve(output, row + 1);
	if (!ImageLogicalType::IsFixedShape(output.GetType())) {
		auto &fields = StructVector::GetEntries(output);
		auto &data = *fields[ImageLogicalType::DATA];
		auto offset = ListVector::GetListSize(data);
		if (offset > NumericLimits<idx_t>::Maximum() / sizeof(float) ||
		    size > NumericLimits<idx_t>::Maximum() / sizeof(float) - offset) {
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
	return FlatVector::GetData(*range.first) +
	       range.second.offset * GetTypeIdSize(range.first->GetType().InternalType());
}

Value ImageVector::FromPixels(vector<Value> pixels, uint32_t width, uint32_t height, const string &mode,
                              const LogicalType &type) {
	ImageLogicalType::ValidateFields(pixels.size(), width, height, ImageLogicalType::ChannelsForMode(mode), mode,
	                                 "IMAGE");
	ImageLogicalType::ValidateShape(type, width, height, mode, "IMAGE");
	auto storage = ImageLogicalType::StorageType(type);
	for (auto &pixel : pixels) {
		if (pixel.IsNull() || pixel.type() != ImageLogicalType::PixelType(mode)) {
			throw InvalidInputException("IMAGE pixels must be non-NULL values of the mode pixel dtype");
		}
		auto number = pixel.GetValue<float>();
		ValidatePixels(const_data_ptr_cast(&number), LogicalType::FLOAT, 1, mode, "IMAGE");
		if (storage == LogicalType::FLOAT) {
			pixel = Value::FLOAT(number);
		}
	}
	if (ImageLogicalType::IsFixedShape(type)) {
		auto result = Value::ARRAY(ImageLogicalType::StorageType(type), std::move(pixels));
		result.Reinterpret(type);
		return result;
	}
	return Value::STRUCT(type, {Value::LIST(ImageLogicalType::StorageType(type), std::move(pixels)),
	                            Value::USMALLINT(ImageLogicalType::ChannelsForMode(mode)), Value::UINTEGER(height),
	                            Value::UINTEGER(width), Value::UTINYINT(ImageLogicalType::ModeCode(mode))});
}

Value ImageVector::FromPixels(const_data_ptr_t pixels, idx_t size, uint32_t width, uint32_t height, const string &mode,
                              const LogicalType &type, bool storage) {
	ImageLogicalType::ValidateFields(size, width, height, ImageLogicalType::ChannelsForMode(mode), mode, "IMAGE");
	ImageLogicalType::ValidateShape(type, width, height, mode, "IMAGE");
	auto pixel_type = storage ? ImageLogicalType::StorageType(type) : ImageLogicalType::PixelType(mode);
	auto target_type = ImageLogicalType::StorageType(type);
	ValidatePixels(pixels, pixel_type, size, mode, "IMAGE");
	string converted;
	if (pixel_type != target_type) {
		converted.resize(size * GetTypeIdSize(target_type.InternalType()));
		CopyPixels(pixels, pixel_type, data_ptr_cast(converted.data()), target_type, size);
		pixels = const_data_ptr_cast(converted.data());
	}
	size *= GetTypeIdSize(target_type.InternalType());
	if (ImageLogicalType::IsFixedShape(type)) {
		return ByteSequenceValue::Create(type, pixels, size);
	}
	return Value::STRUCT(type, {ByteSequenceValue::Create(LogicalType::LIST(target_type), pixels, size),
	                            Value::USMALLINT(ImageLogicalType::ChannelsForMode(mode)), Value::UINTEGER(height),
	                            Value::UINTEGER(width), Value::UTINYINT(ImageLogicalType::ModeCode(mode))});
}

} // namespace duckdb

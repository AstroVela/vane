// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"
#include "image_crop.hpp"
#include "image_operator_contract.hpp"
#include "image_transform.hpp"

#include <algorithm>

using namespace duckdb; // NOLINT

TEST_CASE("Image operator input preserves constant and selected pixel storage", "[image]") {
	for (auto fixed : {false, true}) {
		auto type = fixed ? ImageLogicalType::Create("RGB", 2, 3) : ImageLogicalType::Create();
		uint8_t bytes[18];
		for (idx_t i = 0; i < 18; i++) {
			bytes[i] = uint8_t(i);
		}
		Vector constant(ImageVector::FromPixels(bytes, 18, 3, 2, "RGB", type));
		ImageOperatorInput reader(constant, STANDARD_VECTOR_SIZE);
		ImagePixelView view;
		const_data_ptr_t first = nullptr;
		for (idx_t row = 0; row < STANDARD_VECTOR_SIZE; row++) {
			REQUIRE(reader.Read(row, view));
			REQUIRE(view.layout.width == 3);
			REQUIRE(view.layout.height == 2);
			REQUIRE(view.layout.channels == 3);
			REQUIRE(memcmp(view.data, bytes, 18) == 0);
			if (row == 0) {
				first = view.data;
			}
			REQUIRE(view.data == first);
		}
		REQUIRE(constant.GetVectorType() == VectorType::CONSTANT_VECTOR);
		if (fixed) {
			REQUIRE(ArrayVector::GetTotalSize(constant) == 18);
		} else {
			for (auto &field : StructVector::GetEntries(constant)) {
				REQUIRE(field->GetVectorType() == VectorType::CONSTANT_VECTOR);
			}
		}

		Vector input(type);
		for (idx_t row = 0; row < 4; row++) {
			if (row == 2) {
				input.SetValue(row, Value(type));
				continue;
			}
			auto width = fixed ? idx_t(3) : row + 1;
			auto mode = fixed ? "RGB" : row == 0 ? "L" : row == 1 ? "LA" : "RGBA";
			auto channels = ImageLogicalType::ChannelsForMode(mode);
			duckdb::vector<uint8_t> payload(width * 2 * channels, uint8_t('a' + row));
			ImageVector::WritePixels(input, row, uint32_t(width), 2, mode, payload.data());
		}
		SelectionVector first_selection(6);
		idx_t selected_rows[] = {3, 0, 2, 1, 3, 0};
		for (idx_t row = 0; row < 6; row++) {
			first_selection.set_index(row, selected_rows[row]);
		}
		Vector selected(input, first_selection, 6);
		SelectionVector second_selection(5);
		idx_t positions[] = {4, 2, 1, 3, 0};
		for (idx_t row = 0; row < 5; row++) {
			second_selection.set_index(row, positions[row]);
		}
		selected.Slice(second_selection, 5);
		ImageOperatorInput selected_reader(selected, 5);
		for (idx_t row = 0; row < 5; row++) {
			auto original = selected_rows[positions[row]];
			if (original == 2) {
				REQUIRE_FALSE(selected_reader.Read(row, view));
				continue;
			}
			REQUIRE(selected_reader.Read(row, view));
			REQUIRE(view.layout.width == (fixed ? 3 : original + 1));
			REQUIRE(view.layout.channels == (fixed ? 3 : original + 1));
			for (idx_t i = 0; i < view.layout.Size(); i++) {
				REQUIRE(view.data[i] == 'a' + original);
			}
		}
	}
}

TEST_CASE("Image operator input rejects invalid active pixel windows", "[image]") {
	Vector input(ImageLogicalType::Create());
	duckdb::vector<uint8_t> payload(18, 97);
	ImageVector::WritePixels(input, 0, 3, 2, "RGB", payload.data());
	auto &data = *StructVector::GetEntries(input)[ImageLogicalType::DATA];
	auto entries = FlatVector::GetData<list_entry_t>(data);
	entries[0].offset = NumericLimits<idx_t>::Maximum();
	ImageOperatorInput reader(input, 1);
	ImagePixelView view;
	REQUIRE_THROWS_AS(reader.Read(0, view), InvalidInputException);
}

TEST_CASE("Native Image crop batches tall and strided overlaps", "[image]") {
	constexpr uint32_t tall = 2 * ImageOperatorContract::COPY_BYTES + 7;
	constexpr uint32_t wide = ImageOperatorContract::COPY_BYTES + 7;
	struct CropCase {
		uint32_t width;
		uint32_t height;
		const char *mode;
		ImageCropBox box;
	};
	for (auto &test : duckdb::vector<CropCase> {{1, tall, "L", {0, 0, 1, tall}},
	                                            {1, tall, "L", {0, -1, 1, tall + 2}},
	                                            {3, 300003, "RGBA", {1, -1, 4, 300005}},
	                                            {5, 200003, "RGB", {1, 1, 2, 200001}},
	                                            {wide, 2, "L", {-1, 0, wide + 2, 2}}}) {
		auto channels = ImageLogicalType::ChannelsForMode(test.mode);
		duckdb::vector<uint8_t> pixels(idx_t(test.width) * test.height * channels);
		for (idx_t i = 0; i < pixels.size(); i++) {
			pixels[i] = uint8_t(i % 251 + 1);
		}
		ImagePixelView source {{test.width, test.height, channels, ImageLogicalType::ModeCode(test.mode)},
		                       pixels.data()};
		auto &box = test.box;
		duckdb::vector<uint8_t> actual(idx_t(box.width) * box.height * channels, 0xCC);
		duckdb::vector<uint8_t> expected(actual.size(), 0);
		for (idx_t row = 0; row < box.height; row++) {
			for (idx_t col = 0; col < box.width; col++) {
				auto x = box.x + int64_t(col);
				auto y = box.y + int64_t(row);
				if (x >= 0 && y >= 0 && x < test.width && y < test.height) {
					for (idx_t channel = 0; channel < channels; channel++) {
						expected[(row * box.width + col) * channels + channel] =
						    pixels[(idx_t(y) * test.width + idx_t(x)) * channels + channel];
					}
				}
			}
		}
		idx_t checks = 0;
		CropImagePixels(source, box, actual.data(), actual.size(), [&checks]() { checks++; });
		REQUIRE(memcmp(actual.data(), expected.data(), actual.size()) == 0);
		// A few MiB of pixels must not cause millions of interruption checks
		// simply because rows are narrow. This is independent of machine speed.
		REQUIRE(checks > 0);
		REQUIRE(checks < 32);
	}
}

TEST_CASE("Native Image crop can be interrupted while copying coalesced rows", "[image]") {
	constexpr uint32_t height = 2 * ImageOperatorContract::COPY_BYTES + 7;
	duckdb::vector<uint8_t> pixels(height, 97);
	duckdb::vector<uint8_t> actual(height, 0xCC);
	ImagePixelView source {{1, height, 1, ImageLogicalType::ModeCode("L")}, pixels.data()};
	ImageCropBox box {0, 0, 1, height};
	// Cancel at the next check after copying starts, independently of how
	// many checks the initial zero fill needed.
	REQUIRE_THROWS_AS(CropImagePixels(source, box, actual.data(), actual.size(),
	                                  [&actual]() {
		                                  if (actual[0] == 97) {
			                                  throw InterruptException();
		                                  }
	                                  }),
	                  InterruptException);
	REQUIRE(memcmp(actual.data(), pixels.data(), ImageOperatorContract::COPY_BYTES) == 0);
	REQUIRE(std::all_of(actual.begin() + ImageOperatorContract::COPY_BYTES, actual.end(),
	                    [](uint8_t value) { return value == 0; }));
}

TEST_CASE("Native Image resize filters premultiplied alpha and rounds half up", "[image]") {
	uint8_t pixels[] = {255, 0, 0, 255, 0, 0, 255, 0};
	ImagePixelView source {{2, 1, 4, ImageLogicalType::ModeCode("RGBA")}, pixels};
	ImageLayout output {3, 1, 4, ImageLogicalType::ModeCode("RGBA")};
	uint8_t resized[12] = {};
	ResizeImagePixels(source, output, resized, []() {});
	const uint8_t expected[] = {255, 0, 0, 255, 255, 0, 0, 128, 0, 0, 0, 0};
	REQUIRE(memcmp(resized, expected, sizeof(expected)) == 0);
	uint8_t copied[8] = {};
	ResizeImagePixels(source, source.layout, copied, []() {});
	REQUIRE(memcmp(copied, pixels, sizeof(pixels)) == 0);

	uint8_t half[] = {0, 1};
	source = {{2, 1, 1, ImageLogicalType::ModeCode("L")}, half};
	output = {3, 1, 1, ImageLogicalType::ModeCode("L")};
	ResizeImagePixels(source, output, resized, []() {});
	REQUIRE(resized[0] == 0);
	REQUIRE(resized[1] == 1);
	REQUIRE(resized[2] == 1);
}

TEST_CASE("Native Image conversion drops alpha without compositing", "[image]") {
	uint8_t pixels[] = {255, 0, 0, 0, 0, 255, 0, 128, 0, 0, 250, 255};
	ImagePixelView source {{3, 1, 4, ImageLogicalType::ModeCode("RGBA")}, pixels};
	ImageLayout output {3, 1, 2, ImageLogicalType::ModeCode("LA")};
	uint8_t converted[6] = {};
	ConvertImagePixels(source, output, converted, []() {});
	const uint8_t expected[] = {76, 0, 150, 128, 29, 255};
	REQUIRE(memcmp(converted, expected, sizeof(expected)) == 0);
}

TEST_CASE("Native Image transforms batch narrow rows and can interrupt mid-image", "[image]") {
	constexpr uint32_t height = 1000003;
	duckdb::vector<uint8_t> pixels(height * 3, 97);
	ImagePixelView source {{1, height, 3, ImageLogicalType::ModeCode("RGB")}, pixels.data()};
	for (auto resize : {false, true}) {
		ImageLayout output {resize ? 2u : 1u, height, uint16_t(resize ? 3 : 4),
		                    ImageLogicalType::ModeCode(resize ? "RGB" : "RGBA")};
		duckdb::vector<uint8_t> actual(output.Size(), 0xCC);
		auto run = [&](auto interrupt) {
			if (resize) {
				ResizeImagePixels(source, output, actual.data(), interrupt);
			} else {
				ConvertImagePixels(source, output, actual.data(), interrupt);
			}
		};
		idx_t checks = 0;
		run([&checks]() { checks++; });
		REQUIRE(checks > 1);
		REQUIRE(checks < 150);
		bool correct = true;
		for (idx_t i = 0; i < actual.size(); i++) {
			correct = correct && actual[i] == (!resize && i % 4 == 3 ? 255 : 97);
		}
		REQUIRE(correct);
		std::fill(actual.begin(), actual.end(), 0xCC);
		REQUIRE_THROWS_AS(run([&actual]() {
			                  if (actual[0] == 97) {
				                  throw InterruptException();
			                  }
		                  }),
		                  InterruptException);
		REQUIRE(actual.back() == 0xCC);
	}
}

TEST_CASE("Wide Image storage is lossless and typed transforms retain pixel depth", "[image]") {
	uint16_t pixels[] = {0, 256, 65535, 1000, 32768, 65534};
	for (auto &type : duckdb::vector<LogicalType> {ImageLogicalType::Create(), ImageLogicalType::Create("RGB16"),
	                                               ImageLogicalType::Create("RGB16", 1, 2)}) {
		auto value = ImageVector::FromPixels(const_data_ptr_cast(pixels), 6, 2, 1, "RGB16", type);
		Vector input(value);
		ImageOperatorInput reader(input, 1);
		ImagePixelView view;
		REQUIRE(reader.Read(0, view));
		REQUIRE(view.layout.Bytes() == sizeof(pixels));
		REQUIRE(memcmp(view.data, pixels, sizeof(pixels)) == 0);
		uint16_t cropped[3] = {};
		CropImagePixels(view, {1, 0, 1, 1}, data_ptr_cast(cropped), sizeof(cropped), []() {});
		REQUIRE(memcmp(cropped, pixels + 3, sizeof(cropped)) == 0);
		float converted[6] = {};
		ConvertImagePixels(view, {2, 1, 3, ImageLogicalType::ModeCode("RGB32F")}, data_ptr_cast(converted), []() {});
		REQUIRE(converted[0] == 0);
		REQUIRE(converted[2] == 1);
		REQUIRE(converted[1] == Approx(256.0 / 65535));
		REQUIRE_THROWS_AS(
		    ImageVector::FromPixels(const_data_ptr_cast(converted), 6, 2, 1, "RGB32F", ImageLogicalType::Create("RGB")),
		    InvalidInputException);
	}
}

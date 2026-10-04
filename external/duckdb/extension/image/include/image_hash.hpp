// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "image_hash_contract.hpp"
#include "image_transform.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace duckdb {

//! Perceptual hash version 1. Grayscale uses full-range integer luma,
//! antialiased triangle sampling and rounded UInt8 intermediate samples.
class NativeImageHash {
public:
	NativeImageHash(ClientContext &context, const ImageHashOptions &options) : context(context), options(options) {
	}

	string Compute(const ImagePixelView &image) {
		vector<bool> bits;
		if (options.method == "colorhash") {
			bits = Color(image);
		} else {
			auto width = image.layout.width, height = image.layout.height;
			vector<uint8_t> gray(idx_t(width) * height);
			for (idx_t i = 0; i < gray.size(); i++) {
				if (i % 16384 == 0) {
					Check();
				}
				double red, green, blue;
				RGB(image, i, red, green, blue);
				gray[i] = Byte((299 * red + 587 * green + 114 * blue) / 1000);
			}
			if (options.method == "crop_resistant" && options.segments > 1) {
				if (width < options.segments || height < options.segments) {
					throw InvalidInputException("crop_resistant requires at least one pixel per grid segment");
				}
				for (idx_t y = 0; y < options.segments; y++) {
					for (idx_t x = 0; x < options.segments; x++) {
						auto x0 = x * width / options.segments, x1 = (x + 1) * width / options.segments;
						auto y0 = y * height / options.segments, y1 = (y + 1) * height / options.segments;
						vector<uint8_t> segment((x1 - x0) * (y1 - y0));
						for (idx_t row = y0; row < y1; row++) {
							Check();
							memcpy(segment.data() + (row - y0) * (x1 - x0), gray.data() + row * width + x0, x1 - x0);
						}
						auto part = Gray(segment, x1 - x0, y1 - y0, "phash");
						bits.insert(bits.end(), part.begin(), part.end());
					}
				}
			} else {
				bits = Gray(gray, width, height, options.method == "crop_resistant" ? "phash" : options.method);
			}
		}
		if (bits.size() != options.Bits()) {
			throw InternalException("Image hash violated its bit-count contract");
		}
		string result(options.Bytes(), '\0');
		for (idx_t i = 0; i < bits.size(); i++) {
			if (bits[i]) {
				result[i / 8] = char(uint8_t(result[i / 8]) | uint8_t(1U << (7 - i % 8)));
			}
		}
		return result;
	}

private:
	void Check() {
		ImageOperatorContract::Interrupt(context);
	}
	static uint8_t Byte(double value) {
		return uint8_t(std::floor(MaxValue(0.0, MinValue(255.0, value)) + 0.5));
	}
	static void RGB(const ImagePixelView &image, idx_t pixel, double &red, double &green, double &blue) {
		auto index = pixel * image.layout.channels;
		auto scale = 255 / ImageRange(image.layout.mode);
		red = ImageSample(image.data, image.layout.mode, index) * scale;
		green = image.layout.channels < 3 ? red : ImageSample(image.data, image.layout.mode, index + 1) * scale;
		blue = image.layout.channels < 3 ? red : ImageSample(image.data, image.layout.mode, index + 2) * scale;
	}

	vector<uint8_t> Axis(const vector<uint8_t> &source, idx_t width, idx_t height, bool vertical, idx_t target) {
		auto length = vertical ? height : width;
		auto other = vertical ? width : height;
		if (target * other > ImageOperatorContract::MAX_BYTES) {
			throw OutOfRangeException("Image hash sampling exceeds its scratch limit");
		}
		vector<uint8_t> output(target * other);
		auto ratio = double(length) / target;
		auto support = MaxValue(1.0, ratio);
		for (idx_t i = 0; i < target; i++) {
			Check();
			auto center = (i + 0.5) * ratio;
			auto begin = idx_t(MaxValue(0.0, std::ceil(center - support - 0.5)));
			auto end = idx_t(MinValue(double(length), std::floor(center + support - 0.5) + 1));
			vector<double> sums(other, 0);
			double total = 0;
			for (idx_t position = begin; position < end; position++) {
				auto weight = MaxValue(0.0, 1 - std::abs((position + 0.5 - center) / support));
				total += weight;
				for (idx_t column = 0; column < other; column++) {
					if (column % 16384 == 0) {
						Check();
					}
					auto index = vertical ? position * width + column : column * width + position;
					sums[column] += source[index] * weight;
				}
			}
			for (idx_t column = 0; column < other; column++) {
				output[vertical ? i * width + column : column * target + i] = Byte(sums[column] / total);
			}
		}
		return output;
	}

	vector<uint8_t> Resize(const vector<uint8_t> &source, idx_t width, idx_t height, idx_t target_width,
	                       idx_t target_height) {
		if (width == target_width && height == target_height) {
			return source;
		}
		bool vertical = double(height) / target_height >= double(width) / target_width;
		auto intermediate_size = vertical ? width * target_height : height * target_width;
		// Include the input, intermediate, final buffer and the largest row of
		// double accumulators. No image-sized floating-point scratch is used.
		auto scratch = source.size() + intermediate_size + target_width * target_height +
		               8 * MaxValue(vertical ? width : height, vertical ? target_height : target_width) +
		               ImageOperatorContract::MIB;
		if (scratch > ImageOperatorContract::MAX_BYTES) {
			throw OutOfRangeException("Image hash sampling exceeds its scratch limit");
		}
		if (vertical) {
			auto first = Axis(source, width, height, true, target_height);
			return Axis(first, width, target_height, false, target_width);
		}
		auto first = Axis(source, width, height, false, target_width);
		return Axis(first, target_width, height, true, target_height);
	}

	vector<bool> Gray(const vector<uint8_t> &gray, idx_t width, idx_t height, const string &method) {
		auto size = options.hash_size;
		vector<bool> bits;
		bits.reserve(size * size);
		if (method == "ahash" || method == "dhash" || method == "dhash_vertical") {
			auto w = size + (method == "dhash"), h = size + (method == "dhash_vertical");
			auto samples = Resize(gray, width, height, w, h);
			auto mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
			for (idx_t y = 0; y < size; y++) {
				for (idx_t x = 0; x < size; x++) {
					auto value = samples[y * w + x];
					bits.push_back(method == "dhash"            ? value < samples[y * w + x + 1]
					               : method == "dhash_vertical" ? value < samples[(y + 1) * w + x]
					                                            : value > mean);
				}
			}
			return bits;
		}
		vector<double> coefficients(size * size, 0);
		if (method == "whash") {
			idx_t scale = 1;
			while (scale <= MinValue(width, height) / 2) {
				scale *= 2;
			}
			scale = MaxValue(size, scale);
			auto samples = Resize(gray, width, height, scale, scale);
			auto block = scale / size;
			for (idx_t row = 0; row < scale; row++) {
				Check();
				for (idx_t column = 0; column < scale; column++) {
					coefficients[(row / block) * size + column / block] += samples[row * scale + column];
				}
			}
			for (auto &value : coefficients) {
				value /= double(block * block);
			}
		} else {
			auto n = size * 4;
			auto samples = Resize(gray, width, height, n, n);
			vector<double> basis((size + 1) * n);
			for (idx_t k = 0; k <= size; k++) {
				for (idx_t i = 0; i < n; i++) {
					basis[k * n + i] = std::cos(std::acos(-1.0) * k * (2 * i + 1) / (2 * n));
				}
			}
			vector<double> rows(n * size, 0);
			for (idx_t y = 0; y < n; y++) {
				Check();
				for (idx_t k = 0; k < size; k++) {
					for (idx_t x = 0; x < n; x++) {
						rows[y * size + k] += samples[y * n + x] * basis[(k + (method == "phash_simple")) * n + x];
					}
				}
			}
			for (idx_t y = 0; y < size; y++) {
				Check();
				for (idx_t x = 0; x < size; x++) {
					auto &value = coefficients[y * size + x];
					if (method == "phash_simple") {
						value = rows[y * size + x];
					} else {
						for (idx_t k = 0; k < n; k++) {
							value += rows[k * size + x] * basis[y * n + k];
						}
					}
					value = std::floor(value * 1e6 + 0.5) / 1e6;
				}
			}
		}
		double threshold;
		if (method == "phash") {
			auto ordered = coefficients;
			std::sort(ordered.begin(), ordered.end());
			threshold = (ordered[(ordered.size() - 1) / 2] + ordered[ordered.size() / 2]) / 2;
		} else {
			threshold = std::accumulate(coefficients.begin(), coefficients.end(), 0.0) / coefficients.size();
		}
		for (auto value : coefficients) {
			bits.push_back(value > threshold);
		}
		return bits;
	}

	vector<bool> Color(const ImagePixelView &image) {
		uint64_t counts[14] = {};
		auto pixels = idx_t(image.layout.width) * image.layout.height;
		for (idx_t i = 0; i < pixels; i++) {
			if (i % 16384 == 0) {
				Check();
			}
			double r, g, b;
			RGB(image, i, r, g, b);
			int red = Byte(r), green = Byte(g), blue = Byte(b);
			auto high = MaxValue(red, MaxValue(green, blue)), low = MinValue(red, MinValue(green, blue));
			auto delta = high - low;
			auto saturation = high ? 255 * delta / high : 0;
			auto intensity = (299 * red + 587 * green + 114 * blue + 500) / 1000;
			if (intensity < 32) {
				counts[0]++;
			} else if (saturation < 85) {
				counts[1]++;
			} else {
				auto divisor = 6 * delta;
				auto numerator = high == red     ? (green - blue) * 255
				                 : high == green ? 85 * divisor + (blue - red) * 255
				                                 : 170 * divisor + (red - green) * 255;
				auto hue = int(std::floor(double(numerator) / divisor));
				hue = (hue % 255 + 255) % 255;
				auto bin = MinValue(5, hue * 6 / 255);
				counts[(saturation <= 170 ? 2 : 8) + bin]++;
			}
		}
		auto colors = MaxValue<uint64_t>(1, pixels - counts[0] - counts[1]);
		auto levels = uint64_t(1) << options.binbits;
		vector<bool> bits;
		for (idx_t bin = 0; bin < 14; bin++) {
			auto value = MinValue(levels - 1, counts[bin] * levels / (bin < 2 ? pixels : colors));
			for (idx_t i = 0; i < options.binbits; i++) {
				auto shift = options.binbits - i - 1;
				bits.push_back(((value >> shift) & 1) != 0);
			}
		}
		return bits;
	}

	ClientContext &context;
	const ImageHashOptions &options;
};

} // namespace duckdb

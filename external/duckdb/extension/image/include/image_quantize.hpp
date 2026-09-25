// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "image_codec.hpp"

#include <algorithm>
#include <array>
#include <map>

namespace duckdb {

//! Preserve up to 256 distinct RGB colors exactly. Otherwise split a bounded
//! 5-bit histogram at weighted medians. Ties use R/G/B, then numeric bin order,
//! then the earliest box. Palette means use original 8-bit samples, half up.
class ImageGIFPalette {
public:
	void Encode(ClientContext &context, const ImagePixelView &image, AVFrame &frame) {
		vector<Bin> histogram(32768);
		std::map<uint32_t, uint8_t> exact_colors;
		bool exact = true;
		if (image.layout.channels == 1) {
			for (idx_t i = 0; i < 256; i++) {
				SetColor(frame, i, uint32_t(i) * 0x010101U);
			}
		} else {
			for (idx_t y = 0; y < image.layout.height; y++) {
				MediaInterrupt(context);
				for (idx_t x = 0; x < image.layout.width; x++) {
					auto pixel = image.data + (y * image.layout.width + x) * 3;
					if (exact) {
						exact_colors[Color(pixel)] = 0;
						if (exact_colors.size() > 256) {
							exact = false;
							exact_colors.clear();
						}
					}
					auto &bin = histogram[Index(pixel)];
					bin.count++;
					for (idx_t c = 0; c < 3; c++) {
						bin.sum[c] += pixel[c];
					}
				}
			}
			for (idx_t i = 0; i < 256; i++) {
				SetColor(frame, i, 0);
			}
			if (exact) {
				idx_t index = 0;
				for (auto &entry : exact_colors) {
					entry.second = uint8_t(index);
					SetColor(frame, index++, entry.first);
				}
			} else {
				vector<idx_t> bins;
				for (idx_t i = 0; i < histogram.size(); i++) {
					if (histogram[i].count) {
						bins.push_back(i);
					}
				}
				vector<Box> boxes;
				boxes.push_back(MakeBox(std::move(bins), histogram));
				while (boxes.size() < 256) {
					MediaInterrupt(context);
					idx_t selected = 0;
					for (idx_t i = 1; i < boxes.size(); i++) {
						if (boxes[i].score > boxes[selected].score) {
							selected = i;
						}
					}
					if (boxes[selected].score < 0) {
						break;
					}
					auto box = std::move(boxes[selected]);
					std::sort(box.bins.begin(), box.bins.end(), [&](idx_t a, idx_t b) {
						auto left = Coordinate(a, box.axis), right = Coordinate(b, box.axis);
						return left == right ? a < b : left < right;
					});
					idx_t split = 0;
					uint64_t count = 0;
					while (split < box.bins.size() && count < (box.count + 1) / 2) {
						count += histogram[box.bins[split++]].count;
					}
					split = MaxValue<idx_t>(1, MinValue(split, box.bins.size() - 1));
					vector<idx_t> right(box.bins.begin() + split, box.bins.end());
					box.bins.resize(split);
					boxes[selected] = MakeBox(std::move(box.bins), histogram);
					boxes.push_back(MakeBox(std::move(right), histogram));
				}
				for (idx_t i = 0; i < boxes.size(); i++) {
					auto &box = boxes[i];
					uint64_t sums[3] = {};
					for (auto bin : box.bins) {
						histogram[bin].palette = uint8_t(i);
						for (idx_t c = 0; c < 3; c++) {
							sums[c] += histogram[bin].sum[c];
						}
					}
					uint32_t color = 0;
					for (idx_t c = 0; c < 3; c++) {
						color = (color << 8) | uint32_t((sums[c] + box.count / 2) / box.count);
					}
					SetColor(frame, i, color);
				}
			}
		}
		for (idx_t y = 0; y < image.layout.height; y++) {
			MediaInterrupt(context);
			for (idx_t x = 0; x < image.layout.width; x++) {
				auto pixel = image.data + (y * image.layout.width + x) * image.layout.channels;
				frame.data[0][y * frame.linesize[0] + x] = image.layout.channels == 1 ? pixel[0]
				                                           : exact                    ? exact_colors.at(Color(pixel))
				                                                                      : histogram[Index(pixel)].palette;
			}
		}
	}

private:
	struct Bin {
		uint64_t count = 0;
		std::array<uint64_t, 3> sum {};
		uint8_t palette = 0;
	};
	struct Box {
		vector<idx_t> bins;
		uint64_t count = 0;
		idx_t axis = 0;
		int64_t score = -1;
	};
	static idx_t Coordinate(idx_t bin, idx_t axis) {
		return (bin >> (10 - axis * 5)) & 31;
	}
	static Box MakeBox(vector<idx_t> bins, const vector<Bin> &histogram) {
		Box box;
		idx_t low[3] = {31, 31, 31}, high[3] = {};
		for (auto bin : bins) {
			box.count += histogram[bin].count;
			for (idx_t c = 0; c < 3; c++) {
				low[c] = MinValue(low[c], Coordinate(bin, c));
				high[c] = MaxValue(high[c], Coordinate(bin, c));
			}
		}
		for (idx_t c = 1; c < 3; c++) {
			if (high[c] - low[c] > high[box.axis] - low[box.axis]) {
				box.axis = c;
			}
		}
		box.score = bins.size() > 1 ? int64_t(box.count * (high[box.axis] - low[box.axis])) : -1;
		box.bins = std::move(bins);
		return box;
	}
	static uint32_t Color(const_data_ptr_t pixel) {
		return (uint32_t(pixel[0]) << 16) | (uint32_t(pixel[1]) << 8) | pixel[2];
	}
	static idx_t Index(const_data_ptr_t pixel) {
		return (idx_t(pixel[0] >> 3) << 10) | (idx_t(pixel[1] >> 3) << 5) | (pixel[2] >> 3);
	}
	static void SetColor(AVFrame &frame, idx_t index, uint32_t color) {
		color |= 0xff000000U;
		memcpy(frame.data[1] + index * 4, &color, 4);
	}
};

} // namespace duckdb

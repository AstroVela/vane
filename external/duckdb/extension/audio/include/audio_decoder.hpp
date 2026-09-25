// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "media_reader.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include <sndfile.h>

namespace duckdb {

//! Decode common audio containers with the same library and float64 conversion
//! as Python SoundFile. Every callback stays inside the already-open FILE view
//! and shares the FFmpeg probe's read budget and diagnostic counters.
class NativeSoundFile {
public:
	explicit NativeSoundFile(MediaReader &reader) : reader(reader) {
		SF_VIRTUAL_IO callbacks {Length, Seek, Read, nullptr, Tell};
		handle = sf_open_virtual(&callbacks, SFM_READ, &info, this);
		try {
			Check();
			if (!handle) {
				throw MediaFormatException(string("cannot decode audio: ") + sf_strerror(nullptr));
			}
			if (info.samplerate <= 0 || info.samplerate > 384000 || info.channels <= 0 || info.channels > 64 ||
			    info.frames < 0) {
				throw MediaFormatException("decoded audio requires 1..64 channels and 1..384000 Hz");
			}
			opening = false;
		} catch (...) {
			if (handle) {
				sf_close(handle);
				handle = nullptr;
			}
			throw;
		}
	}

	~NativeSoundFile() {
		if (handle) {
			sf_close(handle);
		}
	}
	NativeSoundFile(const NativeSoundFile &) = delete;
	NativeSoundFile &operator=(const NativeSoundFile &) = delete;

	const SF_INFO &Info() const {
		return info;
	}

	uint64_t ReadFrames(double *output, uint64_t frames) {
		Check();
		auto count = sf_readf_double(handle, output, NumericCast<sf_count_t>(frames));
		Check();
		if (count < 0 || uint64_t(count) > frames || sf_error(handle)) {
			throw MediaFormatException(string("cannot decode audio samples: ") + sf_strerror(handle));
		}
		return uint64_t(count);
	}

	void Close() {
		auto closed = handle;
		handle = nullptr;
		auto error = sf_close(closed);
		Check();
		if (error) {
			throw MediaFormatException(string("cannot close audio decoder: ") + sf_error_number(error));
		}
	}

	static bool Supports(MediaReader &reader) {
		const string format = reader.Format().iformat->name;
		const auto codec = reader.Stream().codecpar->codec_id;
		if (format == "flac") {
			// libsndfile 1.2.2 only decodes 8/16/24-bit native FLAC.
			// Preserve FFmpeg support for other bit depths and Ogg FLAC.
			auto bits = reader.Stream().codecpar->bits_per_raw_sample;
			return bits == 8 || bits == 16 || bits == 24;
		}
		if (format == "mp3") {
			return true;
		}
		if (format == "wav" || format == "aiff") {
			switch (codec) {
			case AV_CODEC_ID_PCM_U8:
			case AV_CODEC_ID_PCM_S8:
			case AV_CODEC_ID_PCM_S16LE:
			case AV_CODEC_ID_PCM_S16BE:
			case AV_CODEC_ID_PCM_S24LE:
			case AV_CODEC_ID_PCM_S24BE:
			case AV_CODEC_ID_PCM_S32LE:
			case AV_CODEC_ID_PCM_S32BE:
			case AV_CODEC_ID_PCM_F32LE:
			case AV_CODEC_ID_PCM_F32BE:
			case AV_CODEC_ID_PCM_F64LE:
			case AV_CODEC_ID_PCM_F64BE:
			case AV_CODEC_ID_PCM_ALAW:
			case AV_CODEC_ID_PCM_MULAW:
				return true;
			default:
				return false;
			}
		}
		return format == "ogg" && (codec == AV_CODEC_ID_VORBIS || codec == AV_CODEC_ID_OPUS);
	}

private:
	void Check() {
		if (error) {
			std::rethrow_exception(error);
		}
		reader.CheckIO(opening);
	}

	static sf_count_t Length(void *opaque) noexcept {
		auto &self = *static_cast<NativeSoundFile *>(opaque);
		try {
			self.Check();
			return NumericCast<sf_count_t>(self.reader.LogicalSize());
		} catch (...) {
			self.error = std::current_exception();
			return -1;
		}
	}

	static sf_count_t Seek(sf_count_t offset, int whence, void *opaque) noexcept {
		auto &self = *static_cast<NativeSoundFile *>(opaque);
		try {
			self.Check();
			uint64_t origin;
			switch (whence) {
			case SEEK_SET:
				origin = 0;
				break;
			case SEEK_CUR:
				origin = self.position;
				break;
			case SEEK_END:
				origin = self.reader.LogicalSize();
				break;
			default:
				return -1;
			}
			if ((offset < 0 && uint64_t(-(offset + 1)) + 1 > origin) ||
			    (offset >= 0 && uint64_t(offset) > self.reader.LogicalSize() - origin)) {
				return -1;
			}
			self.position = offset < 0 ? origin - (uint64_t(-(offset + 1)) + 1) : origin + uint64_t(offset);
			return NumericCast<sf_count_t>(self.position);
		} catch (...) {
			self.error = std::current_exception();
			return -1;
		}
	}

	static sf_count_t Read(void *target, sf_count_t size, void *opaque) noexcept {
		auto &self = *static_cast<NativeSoundFile *>(opaque);
		try {
			self.Check();
			if (size < 0) {
				return 0;
			}
			auto requested = MinValue<uint64_t>(uint64_t(size), self.reader.LogicalSize() - self.position);
			auto count = self.reader.ReadAt(static_cast<data_ptr_t>(target), requested, self.position);
			self.position += count;
			if (count != requested) {
				// A budget-truncated header read is not a malformed file, even
				// when libsndfile stops parsing without another read callback.
				throw OutOfRangeException("native media exceeded its read/probe byte budget");
			}
			return NumericCast<sf_count_t>(count);
		} catch (...) {
			self.error = std::current_exception();
			return 0;
		}
	}

	static sf_count_t Tell(void *opaque) noexcept {
		return Seek(0, SEEK_CUR, opaque);
	}

	MediaReader &reader;
	SNDFILE *handle = nullptr;
	SF_INFO info {};
	uint64_t position = 0;
	std::exception_ptr error;
	bool opening = true;
};

} // namespace duckdb

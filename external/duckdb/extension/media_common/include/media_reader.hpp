// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/execution/expression_executor_state.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/function/table_function.hpp"
#include "file_resolver.hpp"

#include <chrono>
#include <exception>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
}

namespace duckdb {

class ExtensionLoader;

static constexpr uint64_t MEDIA_MIB = 1024 * 1024;
static constexpr uint64_t MEDIA_BATCH_BYTES = 256 * MEDIA_MIB;
static constexpr uint64_t MEDIA_MAX_PIXELS = 100000000;
static constexpr uint64_t MEDIA_MAX_FRAME_BYTES = 512 * MEDIA_MIB;
static constexpr uint64_t MEDIA_METADATA_BYTES = 8 * MEDIA_MIB;

//! Only encoded-content failures may be suppressed by a media on_error policy.
class MediaFormatException : public InvalidInputException {
public:
	explicit MediaFormatException(const string &message) : InvalidInputException("native media: %s", message) {
	}
};

void MediaInterrupt(ClientContext &context);
uint64_t MediaProduct(uint64_t left, uint64_t right, uint64_t limit, const char *description);
uint64_t MediaPositive(const Value &value, const char *name, uint64_t maximum);
void MediaCheck(int code, const char *operation);
void MediaValidateMIME(const FileReference &file, const string &detected);
unique_ptr<FunctionData> BindMediaFile(ClientContext &context, ScalarFunction &function,
                                       vector<unique_ptr<Expression>> &arguments);
ScalarFunction MediaScalar(const string &name, vector<LogicalType> arguments, LogicalType result,
                           scalar_function_t implementation);
LogicalType MediaImageMetadataType();
LogicalType MediaAudioMetadataType();
LogicalType MediaAudioResultType();
LogicalType MediaVideoMetadataType();
LogicalType MediaVideoFrameType();
void RegisterMediaImages(ExtensionLoader &loader);
void RegisterMediaAudio(ExtensionLoader &loader);
void RegisterMediaVideo(ExtensionLoader &loader);

//! One governed logical FILE view. AVIO never receives a URL or an ambient
//! filesystem: all reads and seeks remain inside ResolvedFile's byte window.
class MediaReader {
public:
	MediaReader(ClientContext &context, const FileReference &file, AVMediaType kind, uint64_t input_limit,
	            uint64_t read_limit, uint64_t max_pixels = MEDIA_MAX_PIXELS,
	            uint64_t frame_bytes = MEDIA_MAX_FRAME_BYTES, uint64_t probe_limit = MEDIA_METADATA_BYTES);
	MediaReader(ClientContext &context, const FileReference &file, unique_ptr<ResolvedFile> resolved, AVMediaType kind,
	            uint64_t input_limit, uint64_t read_limit, uint64_t max_pixels, uint64_t frame_bytes,
	            uint64_t probe_limit = MEDIA_METADATA_BYTES);
	~MediaReader();
	MediaReader(const MediaReader &) = delete;
	MediaReader &operator=(const MediaReader &) = delete;

	AVStream &Stream();
	AVFormatContext &Format();
	AVFrame &Frame();
	bool NextFrame();
	void Seek(int64_t timestamp);
	void CheckIO();
	uint64_t BytesRead() const;
	uint64_t FrameBytes() const;

private:
	static int GetBuffer(AVCodecContext *decoder, AVFrame *frame, int flags) noexcept;
	static int Read(void *opaque, uint8_t *target, int size) noexcept;
	static int64_t SeekIO(void *opaque, int64_t offset, int whence) noexcept;
	static int Interrupt(void *opaque) noexcept;
	static int DenyNestedIO(AVFormatContext *format, AVIOContext **io, const char *url, int flags,
	                        AVDictionary **options) noexcept;
	void OpenDecoder();
	void Close() noexcept;

	ClientContext &context;
	unique_ptr<ResolvedFile> file;
	uint64_t position = 0;
	uint64_t bytes_read = 0;
	uint64_t read_limit;
	uint64_t max_pixels;
	uint64_t frame_bytes;
	std::exception_ptr io_error;
	std::chrono::steady_clock::time_point probe_deadline;
	bool probing = true;
	bool source_eof = false;
	bool decoder_flushed = false;
	AVFormatContext *format = nullptr;
	AVIOContext *io = nullptr;
	AVCodecContext *decoder = nullptr;
	AVPacket *packet = nullptr;
	AVFrame *frame = nullptr;
	int stream_index = -1;
};

void MediaConvertPixels(ClientContext &context, const AVFrame &frame, const string &mode, uint32_t width,
                        uint32_t height, data_ptr_t destination);

//! Converted pixels are written into the engine-owned IMAGE vector.
uint64_t MediaWriteImage(ClientContext &context, const AVFrame &frame, const string &mode, uint32_t width,
                         uint32_t height, Vector &result, idx_t row, uint64_t remaining_bytes);

} // namespace duckdb

// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "media_reader.hpp"

#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <limits>

namespace duckdb {

void MediaInterrupt(ClientContext &context) {
	if (context.IsInterrupted()) {
		throw InterruptException();
	}
}

uint64_t MediaProduct(uint64_t left, uint64_t right, uint64_t limit, const char *description) {
	if (right && left > limit / right) {
		throw OutOfRangeException("native media %s exceeds its limit of %llu", description, limit);
	}
	return left * right;
}

uint64_t MediaPositive(const Value &value, const char *name, uint64_t maximum) {
	if (value.IsNull()) {
		throw InvalidInputException("native media %s cannot be NULL", name);
	}
	auto result = value.GetValue<uint64_t>();
	if (!result || result > maximum) {
		throw InvalidInputException("native media %s must be between 1 and %llu", name, maximum);
	}
	return result;
}

void MediaCheck(int code, const char *operation) {
	if (code >= 0) {
		return;
	}
	if (code == AVERROR(ENOMEM)) {
		throw OutOfMemoryException("native media %s ran out of memory", operation);
	}
	char message[AV_ERROR_MAX_STRING_SIZE];
	av_strerror(code, message, sizeof(message));
	throw MediaFormatException(string(operation) + ": " + message);
}

static string CanonicalMIME(string value) {
	value = value.substr(0, value.find(';'));
	StringUtil::Trim(value);
	value = StringUtil::Lower(value);
	if (value == "image/jpg" || value == "image/pjpeg") {
		return "image/jpeg";
	}
	if (value == "image/x-png") {
		return "image/png";
	}
	if (value == "audio/x-wav" || value == "audio/wave" || value == "audio/vnd.wave") {
		return "audio/wav";
	}
	if (value == "audio/x-flac") {
		return "audio/flac";
	}
	if (value == "audio/x-aiff" || value == "audio/aif") {
		return "audio/aiff";
	}
	if (value == "audio/mp3" || value == "audio/x-mp3") {
		return "audio/mpeg";
	}
	if (value == "video/avi") {
		return "video/x-msvideo";
	}
	if (value == "video/x-matroska" || value == "video/mkv") {
		return "video/webm";
	}
	if (value == "audio/x-matroska") {
		return "audio/webm";
	}
	if (value == "video/quicktime" || value == "video/x-m4v") {
		return "video/mp4";
	}
	if (value == "audio/x-m4a") {
		return "audio/mp4";
	}
	return value;
}

void MediaValidateMIME(const FileReference &file, const string &detected) {
	if (!file.has_content_type) {
		return;
	}
	auto declared = CanonicalMIME(file.content_type);
	if (declared == "application/octet-stream" || declared == "binary/octet-stream") {
		return;
	}
	if (declared == "application/ogg" && (detected == "audio/ogg" || detected == "video/ogg")) {
		return;
	}
	auto separator = detected.find('/');
	if (separator != string::npos && declared == detected.substr(0, separator) + "/*") {
		return;
	}
	if (declared != detected) {
		throw MediaFormatException("content_type '" + file.content_type + "' does not match '" + detected + "'");
	}
}

unique_ptr<FunctionData> BindMediaFile(ClientContext &, ScalarFunction &function,
                                       vector<unique_ptr<Expression>> &arguments) {
	FileMediaType media_type;
	if (function.name.find("image") != string::npos) {
		media_type = FileMediaType::IMAGE;
	} else if (function.name.find("audio") != string::npos) {
		media_type = FileMediaType::AUDIO;
	} else {
		media_type = FileMediaType::VIDEO;
	}
	auto type = arguments[0]->return_type;
	if (type.id() == LogicalTypeId::SQLNULL || type.id() == LogicalTypeId::UNKNOWN) {
		type = FileLogicalType::Create(media_type);
	}
	if (!FileLogicalType::IsFile(type) || FileLogicalType::GetMediaType(type) != media_type) {
		throw BinderException("%s requires %s", function.name, FileLogicalType::GetTypeName(media_type));
	}
	function.arguments[0] = type;
	return nullptr;
}

ScalarFunction MediaScalar(const string &name, vector<LogicalType> arguments, LogicalType result,
                           scalar_function_t implementation) {
	ScalarFunction function(
	    "native_" + name, std::move(arguments), std::move(result),
	    [implementation](DataChunk &args, ExpressionState &state, Vector &output) {
		    try {
			    implementation(args, state, output);
		    } catch (...) {
			    MediaInterrupt(state.GetContext());
			    throw;
		    }
	    },
	    BindMediaFile);
	function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	function.SetStability(FunctionStability::VOLATILE);
	function.SetFallible();
	return function;
}

LogicalType MediaImageMetadataType() {
	return LogicalType::STRUCT({{"width", LogicalType::UINTEGER},
	                            {"height", LogicalType::UINTEGER},
	                            {"format", LogicalType::VARCHAR},
	                            {"mode", LogicalType::VARCHAR}});
}

LogicalType MediaAudioMetadataType() {
	return LogicalType::STRUCT({{"sample_rate", LogicalType::BIGINT},
	                            {"channels", LogicalType::BIGINT},
	                            {"frames", LogicalType::BIGINT},
	                            {"duration", LogicalType::DOUBLE},
	                            {"format", LogicalType::VARCHAR},
	                            {"subtype", LogicalType::VARCHAR}});
}

LogicalType MediaAudioResultType() {
	return LogicalType::STRUCT({{"samples", LogicalType::LIST(LogicalType::DOUBLE)},
	                            {"sample_rate", LogicalType::BIGINT},
	                            {"frames", LogicalType::BIGINT},
	                            {"channels", LogicalType::BIGINT}});
}

LogicalType MediaVideoMetadataType() {
	return LogicalType::STRUCT({{"width", LogicalType::UINTEGER},
	                            {"height", LogicalType::UINTEGER},
	                            {"fps", LogicalType::DOUBLE},
	                            {"duration", LogicalType::DOUBLE},
	                            {"frame_count", LogicalType::BIGINT},
	                            {"time_base", LogicalType::STRUCT({{"numerator", LogicalType::BIGINT},
	                                                               {"denominator", LogicalType::BIGINT}})}});
}

LogicalType MediaVideoFrameType() {
	return LogicalType::STRUCT({{"frame_index", LogicalType::BIGINT},
	                            {"frame_time", LogicalType::DOUBLE},
	                            {"frame_time_base_numerator", LogicalType::BIGINT},
	                            {"frame_time_base_denominator", LogicalType::BIGINT},
	                            {"frame_pts", LogicalType::BIGINT},
	                            {"frame_dts", LogicalType::BIGINT},
	                            {"frame_duration", LogicalType::BIGINT},
	                            {"is_key_frame", LogicalType::BOOLEAN},
	                            {"data", ImageLogicalType::Create()}});
}

static string StreamMIME(const AVInputFormat &format, AVMediaType kind) {
	string name(format.name);
	if (name.find("mov") != string::npos) {
		return kind == AVMEDIA_TYPE_AUDIO ? "audio/mp4" : "video/mp4";
	}
	if (name.find("matroska") != string::npos) {
		return kind == AVMEDIA_TYPE_AUDIO ? "audio/webm" : "video/webm";
	}
	if (name == "ogg") {
		return kind == AVMEDIA_TYPE_AUDIO ? "audio/ogg" : "video/ogg";
	}
	if (name == "wav") {
		return "audio/wav";
	}
	if (name == "aiff") {
		return "audio/aiff";
	}
	if (name == "flac") {
		return "audio/flac";
	}
	if (name == "mp3") {
		return "audio/mpeg";
	}
	if (name == "aac") {
		return "audio/aac";
	}
	if (name == "avi") {
		return "video/x-msvideo";
	}
	if (name == "mpegts") {
		return "video/mp2t";
	}
	if (name == "mpeg") {
		return "video/mpeg";
	}
	if (name == "png_pipe") {
		return "image/png";
	}
	if (name == "jpeg_pipe") {
		return "image/jpeg";
	}
	throw MediaFormatException("unsupported container '" + name + "'");
}

MediaReader::MediaReader(ClientContext &context_p, const FileReference &reference, AVMediaType kind,
                         uint64_t input_limit, uint64_t read_limit_p, uint64_t max_pixels_p, uint64_t frame_bytes_p,
                         uint64_t probe_limit, MediaReadProfile *profile_p)
    : MediaReader(context_p, reference, ResolvedFile::Open(context_p, reference), kind, input_limit, read_limit_p,
                  max_pixels_p, frame_bytes_p, probe_limit, nullptr, profile_p) {
}

MediaReader::MediaReader(ClientContext &context_p, const FileReference &reference, unique_ptr<ResolvedFile> resolved,
                         AVMediaType kind, uint64_t input_limit, uint64_t read_limit_p, uint64_t max_pixels_p,
                         uint64_t frame_bytes_p, uint64_t probe_limit, unique_ptr<MediaReadVerifier> verifier_p,
                         MediaReadProfile *profile_p)
    : context(context_p), file(std::move(resolved)), verifier(std::move(verifier_p)), profile(profile_p),
      read_limit(read_limit_p), max_pixels(MinValue<uint64_t>(max_pixels_p, frame_bytes_p / 8)),
      frame_bytes(frame_bytes_p), probe_deadline(std::chrono::steady_clock::now() + std::chrono::seconds(30)) {
	if (!file->LogicalSize()) {
		throw MediaFormatException("empty FILE view");
	}
	if (file->LogicalSize() > input_limit || file->LogicalSize() > uint64_t(INT64_MAX)) {
		throw OutOfRangeException("native media input exceeds max_input_bytes");
	}
	try {
		format = avformat_alloc_context();
		if (!format) {
			throw OutOfMemoryException("Cannot allocate native media format context");
		}
		auto buffer = static_cast<uint8_t *>(av_malloc(64 * 1024));
		if (!buffer) {
			throw OutOfMemoryException("Cannot allocate native media read buffer");
		}
		io = avio_alloc_context(buffer, 64 * 1024, 0, this, Read, nullptr, SeekIO);
		if (!io) {
			av_free(buffer);
			throw OutOfMemoryException("Cannot allocate native media I/O context");
		}
		format->pb = io;
		format->opaque = this;
		format->flags |= AVFMT_FLAG_CUSTOM_IO;
		format->io_open = DenyNestedIO;
		format->interrupt_callback = {Interrupt, this};
		auto probe_bytes = MinValue<uint64_t>(read_limit, probe_limit);
		format->probesize = NumericCast<int64_t>(probe_bytes);
		// FFmpeg requires at least 2048 for format detection. Read still enforces
		// smaller caller budgets and preserves their resource-limit exception.
		format->format_probesize = NumericCast<int>(MaxValue<uint64_t>(probe_bytes, 2048));
		format->max_analyze_duration = AV_TIME_BASE;
		format->max_streams = 16;
		format->max_probe_packets = 256;
		AVDictionary *options = nullptr;
		struct DictionaryGuard {
			AVDictionary **values;
			unsigned count;
			~DictionaryGuard() {
				for (unsigned i = 0; i < count; i++) {
					av_dict_free(&values[i]);
				}
			}
		};
		DictionaryGuard options_guard {&options, 1};
		MediaCheck(av_dict_set(&options, "format_whitelist",
		                       "wav,aiff,flac,mp3,aac,ogg,mov,matroska,webm,avi,mpegts,mpeg,png_pipe,jpeg_pipe", 0),
		           "set container allowlist");
		MediaCheck(av_dict_set(&options, "protocol_whitelist", "", 0), "set protocol policy");
		auto code = avformat_open_input(&format, nullptr, nullptr, &options);
		av_dict_free(&options);
		CheckIO();
		MediaCheck(code, "open container");
		AVDictionary *stream_options[16] = {};
		DictionaryGuard stream_options_guard {stream_options, 16};
		auto stream_count = format->nb_streams;
		if (stream_count > 16) {
			throw OutOfRangeException("native media container exceeds the stream limit");
		}
		for (unsigned index = 0; index < stream_count; index++) {
			MediaCheck(av_dict_set_int(&stream_options[index], "max_pixels", NumericCast<int64_t>(max_pixels), 0),
			           "set probe pixel limit");
			MediaCheck(av_dict_set_int(&stream_options[index], "threads", 1, 0), "set probe thread limit");
			MediaCheck(av_dict_set_int(&stream_options[index], "max_samples", int64_t(frame_bytes / sizeof(double)), 0),
			           "set probe sample limit");
		}
		bool have_parameters = reference.media_type == FileMediaType::IMAGE;
		for (unsigned index = 0; index < stream_count; index++) {
			auto &parameters = *format->streams[index]->codecpar;
			if (parameters.codec_type != kind || parameters.codec_id == AV_CODEC_ID_NONE ||
			    (kind == AVMEDIA_TYPE_VIDEO && (format->streams[index]->disposition & AV_DISPOSITION_ATTACHED_PIC))) {
				continue;
			}
			have_parameters =
			    have_parameters ||
			    (kind == AVMEDIA_TYPE_AUDIO ? parameters.sample_rate > 0 && parameters.ch_layout.nb_channels > 0
			                                : parameters.width > 0 && parameters.height > 0);
		}
		// Container headers often establish the stream contract. Avoid decoding
		// a frame during probing and then decoding the same packet again.
		if (!have_parameters) {
			code = avformat_find_stream_info(format, stream_options);
			CheckIO();
			MediaCheck(code, "inspect container");
		}
		for (unsigned index = 0; index < format->nb_streams; index++) {
			auto stream = format->streams[index];
			if (stream->codecpar->codec_type == kind && !(reference.media_type == FileMediaType::VIDEO &&
			                                              (stream->disposition & AV_DISPOSITION_ATTACHED_PIC))) {
				stream_index = NumericCast<int>(index);
				break;
			}
		}
		if (stream_index < 0) {
			throw MediaFormatException("container does not contain the requested media stream");
		}
		auto mime = StreamMIME(*format->iformat, kind);
		if ((reference.media_type == FileMediaType::VIDEO && !StringUtil::StartsWith(mime, "video/")) ||
		    (reference.media_type == FileMediaType::AUDIO && !StringUtil::StartsWith(mime, "audio/"))) {
			throw MediaFormatException("container does not belong to the requested FILE media type");
		}
		MediaValidateMIME(reference, mime);
		probing = false;
	} catch (...) {
		Close();
		throw;
	}
}

MediaReader::~MediaReader() {
	Close();
}

void MediaReader::Close() noexcept {
	av_frame_free(&frame);
	av_packet_free(&packet);
	avcodec_free_context(&decoder);
	avformat_close_input(&format);
	if (io) {
		av_freep(&io->buffer);
		avio_context_free(&io);
	}
}

int MediaReader::Read(void *opaque, uint8_t *target, int size) noexcept {
	auto &self = *static_cast<MediaReader *>(opaque);
	try {
		self.CheckIO();
		if (size <= 0) {
			return AVERROR(EINVAL);
		}
		if (self.position == self.file->LogicalSize()) {
			return AVERROR_EOF;
		}
		if (self.bytes_read >= self.read_limit) {
			throw OutOfRangeException("native media exceeded its read/probe byte budget");
		}
		auto count = MinValue<uint64_t>(uint64_t(size), self.file->LogicalSize() - self.position);
		count = MinValue<uint64_t>(count, self.read_limit - self.bytes_read);
		{
			MediaProfileTimer timer(self.profile ? &self.profile->seconds : nullptr);
			if (self.verifier) {
				self.verifier->Read(*self.file, target, count, self.position);
			} else {
				self.file->ReadExact(target, count, self.position);
			}
		}
		if (self.profile) {
			self.profile->calls++;
		}
		self.position += count;
		self.bytes_read += count;
		return NumericCast<int>(count);
	} catch (...) {
		self.io_error = std::current_exception();
		return AVERROR_EXTERNAL;
	}
}

int64_t MediaReader::SeekIO(void *opaque, int64_t offset, int whence) noexcept {
	auto &self = *static_cast<MediaReader *>(opaque);
	try {
		self.CheckIO();
		whence &= ~AVSEEK_FORCE;
		if (whence == AVSEEK_SIZE) {
			return NumericCast<int64_t>(self.file->LogicalSize());
		}
		uint64_t origin;
		switch (whence) {
		case SEEK_SET:
			origin = 0;
			break;
		case SEEK_CUR:
			origin = self.position;
			break;
		case SEEK_END:
			origin = self.file->LogicalSize();
			break;
		default:
			return AVERROR(EINVAL);
		}
		if ((offset < 0 && uint64_t(-(offset + 1)) + 1 > origin) ||
		    (offset >= 0 && uint64_t(offset) > self.file->LogicalSize() - origin)) {
			return AVERROR(EINVAL);
		}
		self.position = offset < 0 ? origin - (uint64_t(-(offset + 1)) + 1) : origin + uint64_t(offset);
		return NumericCast<int64_t>(self.position);
	} catch (...) {
		self.io_error = std::current_exception();
		return AVERROR_EXTERNAL;
	}
}

int MediaReader::Interrupt(void *opaque) noexcept {
	auto &self = *static_cast<MediaReader *>(opaque);
	return self.context.IsInterrupted() || bool(self.io_error) ||
	       (self.probing && std::chrono::steady_clock::now() >= self.probe_deadline);
}

int MediaReader::DenyNestedIO(AVFormatContext *format, AVIOContext **, const char *, int, AVDictionary **) noexcept {
	if (!format || !format->opaque) {
		return AVERROR(EACCES);
	}
	auto &self = *static_cast<MediaReader *>(format->opaque);
	try {
		throw PermissionException("container requested an external resource outside its FILE view");
	} catch (...) {
		self.io_error = std::current_exception();
	}
	return AVERROR_EXTERNAL;
}

void MediaReader::CheckIO() {
	MediaInterrupt(context);
	if (io_error) {
		std::rethrow_exception(io_error);
	}
	if (probing && std::chrono::steady_clock::now() >= probe_deadline) {
		throw OutOfRangeException("native media metadata probe exceeded its time budget");
	}
}

AVStream &MediaReader::Stream() {
	return *format->streams[stream_index];
}

AVFormatContext &MediaReader::Format() {
	return *format;
}

AVFrame &MediaReader::Frame() {
	return *frame;
}

uint64_t MediaReader::BytesRead() const {
	return verifier ? verifier->BytesRead() : bytes_read;
}

uint64_t MediaReader::FrameBytes() const {
	return frame_bytes;
}

int MediaReader::GetBuffer(AVCodecContext *decoder, AVFrame *frame, int flags) noexcept {
	auto &self = *static_cast<MediaReader *>(decoder->opaque);
	try {
		self.CheckIO();
		if (frame->width > 0 && frame->height > 0) {
			auto pixels = MediaProduct(frame->width, frame->height, self.max_pixels, "decoded pixels");
			MediaProduct(pixels, 8, self.frame_bytes, "decoded frame bytes");
			int width = frame->width, height = frame->height;
			int alignments[AV_NUM_DATA_POINTERS] = {};
			avcodec_align_dimensions2(decoder, &width, &height, alignments);
			int alignment = 1;
			for (auto value : alignments) {
				alignment = MaxValue(alignment, value);
			}
			auto padded_size = av_image_get_buffer_size(AVPixelFormat(frame->format), width, height, alignment);
			MediaCheck(padded_size, "calculate decoder buffer size");
			MediaProduct(1, uint64_t(padded_size) + 4 * AV_INPUT_BUFFER_PADDING_SIZE, self.frame_bytes,
			             "padded decoder frame bytes");
		}
		if (frame->nb_samples > 0) {
			MediaProduct(frame->nb_samples, uint64_t(frame->ch_layout.nb_channels) * sizeof(double), self.frame_bytes,
			             "decoded audio frame bytes");
		}
		return avcodec_default_get_buffer2(decoder, frame, flags);
	} catch (...) {
		self.io_error = std::current_exception();
		return AVERROR_EXTERNAL;
	}
}

void MediaReader::OpenDecoder() {
	if (decoder) {
		return;
	}
	auto &parameters = *Stream().codecpar;
	if (parameters.width > 0 && parameters.height > 0) {
		MediaProduct(parameters.width, parameters.height, max_pixels, "decoded pixels");
		MediaProduct(uint64_t(parameters.width) * parameters.height, 8, frame_bytes, "decoded frame bytes");
	}
	const auto codec = avcodec_find_decoder(parameters.codec_id);
	if (!codec) {
		throw MediaFormatException("decoder is unavailable for the selected codec");
	}
	decoder = avcodec_alloc_context3(codec);
	packet = av_packet_alloc();
	frame = av_frame_alloc();
	if (!decoder || !packet || !frame) {
		throw OutOfMemoryException("Cannot allocate native media decoder state");
	}
	MediaCheck(avcodec_parameters_to_context(decoder, &parameters), "configure decoder");
	decoder->thread_count = 1;
	decoder->opaque = this;
	decoder->get_buffer2 = GetBuffer;
	decoder->max_pixels = NumericCast<int64_t>(max_pixels);
	decoder->max_samples = NumericCast<int64_t>(frame_bytes / sizeof(double));
	auto code = avcodec_open2(decoder, codec, nullptr);
	CheckIO();
	MediaCheck(code, "open decoder");
}

bool MediaReader::NextFrame() {
	OpenDecoder();
	av_frame_unref(frame);
	for (;;) {
		CheckIO();
		auto code = avcodec_receive_frame(decoder, frame);
		CheckIO();
		if (code >= 0) {
			if (frame->width > 0 && frame->height > 0) {
				MediaProduct(frame->width, frame->height, max_pixels, "decoded pixels");
				MediaProduct(uint64_t(frame->width) * frame->height, 8, frame_bytes, "decoded frame bytes");
			}
			if (frame->nb_samples > 0) {
				MediaProduct(frame->nb_samples, uint64_t(frame->ch_layout.nb_channels) * sizeof(double), frame_bytes,
				             "decoded audio frame bytes");
			}
			return true;
		}
		if (code == AVERROR_EOF) {
			return false;
		}
		if (code != AVERROR(EAGAIN)) {
			MediaCheck(code, "decode frame");
		}
		if (source_eof) {
			if (decoder_flushed) {
				throw MediaFormatException("decoder requested packets after end of stream");
			}
			auto flush_code = avcodec_send_packet(decoder, nullptr);
			CheckIO();
			MediaCheck(flush_code, "flush decoder");
			decoder_flushed = true;
			continue;
		}
		for (;;) {
			MediaInterrupt(context);
			code = av_read_frame(format, packet);
			CheckIO();
			if (code == AVERROR_EOF) {
				source_eof = true;
				break;
			}
			MediaCheck(code, "read packet");
			if (packet->stream_index == stream_index) {
				code = avcodec_send_packet(decoder, packet);
				CheckIO();
				av_packet_unref(packet);
				MediaCheck(code, "submit packet");
				break;
			}
			av_packet_unref(packet);
		}
	}
}

void MediaReader::Seek(int64_t timestamp) {
	OpenDecoder();
	MediaInterrupt(context);
	const auto code = av_seek_frame(format, stream_index, timestamp, AVSEEK_FLAG_BACKWARD);
	CheckIO();
	MediaCheck(code, "seek to keyframe");
	avcodec_flush_buffers(decoder);
	av_packet_unref(packet);
	av_frame_unref(frame);
	source_eof = false;
	decoder_flushed = false;
}

} // namespace duckdb

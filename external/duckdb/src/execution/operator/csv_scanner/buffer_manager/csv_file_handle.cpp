#include "duckdb/execution/operator/csv_scanner/csv_file_handle.hpp"
#include "duckdb/common/exception/binder_exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/compressed_file_system.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/operator/csv_scanner/csv_reader_options.hpp"

namespace duckdb {

CSVFileHandle::CSVFileHandle(ClientContext &context_p, unique_ptr<FileHandle> file_handle_p, const OpenFileInfo &file_p,
                             const CSVReaderOptions &options)
    : compression_type(options.compression), context(context_p), file_handle(std::move(file_handle_p)),
      encoder(context_p, options.encoding, options.buffer_size_option.GetValue()), file(file_p) {
	can_seek = file_handle->CanSeek();
	on_disk_file = file_handle->OnDiskFile();
	file_size = file_handle->GetFileSize();
	is_pipe = file_handle->IsPipe();
	compression_type = file_handle->GetFileCompressionType();
}

unique_ptr<FileHandle> CSVFileHandle::OpenFileHandle(FileSystem &fs, Allocator &allocator, const OpenFileInfo &file,
                                                     FileCompressionType compression) {
	auto file_handle = fs.OpenFile(file, FileFlags::FILE_FLAGS_READ | compression);
	if (file_handle->CanSeek()) {
		file_handle->Reset();
	}
	return file_handle;
}

unique_ptr<CSVFileHandle> CSVFileHandle::OpenFile(ClientContext &context, const OpenFileInfo &file,
                                                  const CSVReaderOptions &options) {
	auto &fs = FileSystem::GetFileSystem(context);
	auto &allocator = BufferAllocator::Get(context);
	auto file_handle = OpenFileHandle(fs, allocator, file, options.compression);
	return make_uniq<CSVFileHandle>(context, std::move(file_handle), file, options);
}

double CSVFileHandle::GetProgress() const {
	return static_cast<double>(file_handle->GetProgress());
}

bool CSVFileHandle::CanSeek() const {
	return can_seek;
}

void CSVFileHandle::Seek(const idx_t position) const {
	if (!can_seek) {
		if (is_pipe) {
			throw InternalException("Trying to seek a piped CSV File.");
		}
		throw InternalException("Trying to seek a compressed CSV File.");
	}
	if (has_scan_range) {
		if (position > scan_read_end - scan_range_start) {
			throw InternalException("Trying to seek past a CSV scan task's byte range");
		}
		file_handle->Seek(scan_range_start + position);
	} else {
		file_handle->Seek(position);
	}
}

bool CSVFileHandle::OnDiskFile() const {
	return on_disk_file;
}

void CSVFileHandle::Reset() {
	if (has_scan_range) {
		file_handle->Seek(scan_range_start);
	} else {
		file_handle->Reset();
	}
	finished = false;
	requested_bytes = 0;
}

void CSVFileHandle::SetScanRange(const idx_t start, const idx_t end, const idx_t overlap) {
	if (compression_type != FileCompressionType::UNCOMPRESSED) {
		throw InvalidInputException("CSV byte-range scans require an uncompressed input file");
	}
	if (!can_seek || is_pipe) {
		throw InvalidInputException("CSV byte-range scans require a seekable input file");
	}
	if (encoder.encoding_name != "utf-8") {
		throw InvalidInputException("CSV byte-range scans require UTF-8 input");
	}
	if (start >= end || end > file_size) {
		throw InvalidInputException("CSV byte range [%llu, %llu) is outside file of size %llu", start, end, file_size);
	}
	idx_t aligned_start = start;
	if (start > 0) {
		char boundary_bytes[2];
		file_handle->Read(context, boundary_bytes, sizeof(boundary_bytes), start - 1);
		if (boundary_bytes[0] == '\r' && boundary_bytes[1] == '\n') {
			// Starting on the LF half of CRLF leaves the strict CSV state machine without the preceding CR.
			// Start after the complete delimiter. The preceding task's overlap owns the boundary row, while this
			// task's normal row synchronization skips that already-owned row.
			aligned_start++;
		}
	}
	if (aligned_start >= end) {
		throw InvalidInputException("CSV byte range [%llu, %llu) cannot be aligned to a complete record delimiter",
		                            start, end);
	}
	has_scan_range = true;
	scan_range_start = aligned_start;
	scan_range_end = end;
	scan_read_end = end > file_size - MinValue<idx_t>(overlap, file_size) ? file_size : end + overlap;
	Reset();
}

bool CSVFileHandle::IsPipe() const {
	return is_pipe;
}

idx_t CSVFileHandle::FileSize() const {
	return has_scan_range ? scan_range_end - scan_range_start : file_size;
}

bool CSVFileHandle::HasScanRange() const {
	return has_scan_range;
}

idx_t CSVFileHandle::ScanRangeStart() const {
	return scan_range_start;
}

bool CSVFileHandle::FinishedReading() const {
	return finished;
}

idx_t CSVFileHandle::Read(void *buffer, idx_t nr_bytes) {
	// We avoid reading past the original size of the file for uncompressed files in utf-8 encoding. This avoids reading
	// the data that is written after opening the file. This can be useful, for example when reading a duckdb log file
	// in csv format while logging is enabled
	if (file_handle->GetFileCompressionType() == FileCompressionType::UNCOMPRESSED && file_handle->CanSeek() &&
	    encoder.encoding_name == "utf-8") {
		const auto read_end = has_scan_range ? scan_read_end : file_size;
		const auto current_position = file_handle->SeekPosition();
		nr_bytes = current_position >= read_end ? 0 : MinValue<idx_t>(nr_bytes, read_end - current_position);
	}

	requested_bytes += nr_bytes;
	// if this is a plain file source OR we can seek we are not caching anything
	idx_t bytes_read = 0;
	if (encoder.encoding_name == "utf-8") {
		bytes_read = static_cast<idx_t>(file_handle->Read(context, buffer, nr_bytes));
	} else {
		bytes_read = encoder.Encode(*file_handle, static_cast<char *>(buffer), nr_bytes);
	}
	if (!finished) {
		finished = bytes_read == 0 || (has_scan_range && file_handle->SeekPosition() >= scan_read_end);
	}
	uncompressed_bytes_read += static_cast<idx_t>(bytes_read);
	return UnsafeNumericCast<idx_t>(bytes_read);
}

string CSVFileHandle::ReadLine() {
	bool carriage_return = false;
	string result;
	char buffer[1];
	while (true) {
		idx_t bytes_read = Read(buffer, 1);
		if (bytes_read == 0) {
			return result;
		}
		if (carriage_return) {
			if (buffer[0] != '\n') {
				if (!file_handle->CanSeek()) {
					throw BinderException(
					    "Carriage return newlines not supported when reading CSV files in which we cannot seek");
				}
				Seek(file_handle->SeekPosition() - scan_range_start - 1);
				return result;
			}
		}
		if (buffer[0] == '\n') {
			return result;
		}
		if (buffer[0] != '\r') {
			result += buffer[0];
		} else {
			carriage_return = true;
		}
	}
}

string CSVFileHandle::GetFilePath() {
	return file.path;
}

} // namespace duckdb

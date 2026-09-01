// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/file.hpp"

#include "duckdb/common/error_data.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "file_value.hpp"
#include "vane_python/pybind11/conversions/pyconnection_default.hpp"
#include "vane_python/pyconnection/pyconnection.hpp"
#include "vane_python/python_objects.hpp"

#include <mutex>
#include <type_traits>
#include <utility>

namespace duckdb {

namespace {

static constexpr uint64_t DEFAULT_IMAGE_METADATA_BYTES = 1024 * 1024;
static constexpr uint64_t DEFAULT_IMAGE_MAX_PIXELS = 100000000;
static constexpr uint64_t DEFAULT_IMAGE_BUFFER_SIZE = 1024 * 1024;
static constexpr uint64_t DEFAULT_IMAGE_MAX_INPUT_BYTES = 256 * 1024 * 1024;
static constexpr uint64_t DEFAULT_IMAGE_MAX_DECODED_BYTES = 512 * 1024 * 1024;
static constexpr uint64_t DEFAULT_AUDIO_METADATA_BYTES = 8 * 1024 * 1024;
static constexpr uint64_t DEFAULT_AUDIO_BUFFER_SIZE = 1024 * 1024;
static constexpr uint64_t DEFAULT_AUDIO_MAX_INPUT_BYTES = 512 * 1024 * 1024;
static constexpr uint64_t DEFAULT_AUDIO_MAX_FRAMES = 100000000;
static constexpr uint64_t DEFAULT_AUDIO_MAX_DECODED_BYTES = 512 * 1024 * 1024;
static constexpr uint64_t DEFAULT_VIDEO_METADATA_BYTES = 8 * 1024 * 1024;

static string PythonTypeName(const py::handle &value) {
	return py::str(py::type::of(value)).cast<string>();
}

static string RequireString(const py::handle &value, const char *field) {
	if (!py::isinstance<py::str>(value)) {
		throw py::type_error(StringUtil::Format("File.%s must be str, not '%s'", field, PythonTypeName(value)));
	}
	return py::cast<string>(value);
}

static std::optional<string> OptionalString(const py::handle &value, const char *field) {
	if (value.is_none()) {
		return std::nullopt;
	}
	return RequireString(value, field);
}

static std::optional<int64_t> OptionalInteger(const py::handle &value, const char *field) {
	if (value.is_none()) {
		return std::nullopt;
	}
	if (py::isinstance<py::bool_>(value) || !py::isinstance<py::int_>(value)) {
		throw py::type_error(StringUtil::Format("File.%s must be int or None, not '%s'", field, PythonTypeName(value)));
	}

	int overflow = 0;
	auto result = PyLong_AsLongLongAndOverflow(value.ptr(), &overflow);
	if (overflow != 0) {
		PyErr_Clear();
		auto message = StringUtil::Format("File.%s must fit in signed 64-bit", field);
		PyErr_SetString(PyExc_OverflowError, message.c_str());
		throw py::error_already_set();
	}
	if (result == -1 && PyErr_Occurred()) {
		throw py::error_already_set();
	}
	return result;
}

static Value OptionalStringValue(const std::optional<string> &value) {
	return value ? Value(*value) : Value(LogicalType::VARCHAR);
}

static Value OptionalIntegerValue(const std::optional<int64_t> &value) {
	return value ? Value::BIGINT(*value) : Value(LogicalType::BIGINT);
}

static std::optional<string> OptionalStringFromReference(bool has_value, const string &value) {
	return has_value ? std::optional<string>(value) : std::nullopt;
}

static std::optional<int64_t> OptionalIntegerFromReference(bool has_value, int64_t value) {
	return has_value ? std::optional<int64_t>(value) : std::nullopt;
}

static FileReference MakeReference(const string &url, const std::optional<string> &content_type,
                                   const std::optional<int64_t> &position, const std::optional<int64_t> &size,
                                   const std::optional<string> &checksum, FileMediaType media_type) {
	return FileReference::FromFields(Value(url), OptionalStringValue(content_type), OptionalIntegerValue(position),
	                                 OptionalIntegerValue(size), OptionalStringValue(checksum), "File", media_type);
}

template <class FILE_TYPE>
static FILE_TYPE FileFromPython(const py::handle &url, const py::handle &content_type, const py::handle &position,
                                const py::handle &size, const py::handle &checksum) {
	auto url_value = RequireString(url, "url");
	auto content_type_value = OptionalString(content_type, "content_type");
	auto position_value = OptionalInteger(position, "position");
	auto size_value = OptionalInteger(size, "size");
	auto checksum_value = OptionalString(checksum, "checksum");
	try {
		return FILE_TYPE(std::move(url_value), std::move(content_type_value), position_value, size_value,
		                 std::move(checksum_value));
	} catch (const InvalidInputException &error) {
		auto error_data = ErrorData(error);
		throw py::value_error(error_data.RawMessage());
	}
}

template <class FILE_TYPE>
static FILE_TYPE FileFromPickleState(const py::tuple &state, const char *class_name) {
	if (state.size() != FileLogicalType::FIELD_COUNT) {
		throw py::value_error(StringUtil::Format("Invalid %s pickle state", class_name));
	}
	return FileFromPython<FILE_TYPE>(state[FileLogicalType::URL], state[FileLogicalType::CONTENT_TYPE],
	                                 state[FileLogicalType::POSITION], state[FileLogicalType::SIZE],
	                                 state[FileLogicalType::CHECKSUM]);
}

template <class FILE_TYPE>
static void BindMediaFileClass(py::handle &m, const char *class_name) {
	auto file = py::class_<FILE_TYPE, PythonFile>(m, class_name, py::module_local(), py::is_final());
	file.def(py::init([](const py::object &url, const py::object &content_type, const py::object &position,
	                     const py::object &size, const py::object &checksum) {
		         return FileFromPython<FILE_TYPE>(url, content_type, position, size, checksum);
	         }),
	         py::arg("url"), py::arg("content_type") = py::none(), py::arg("position") = py::none(),
	         py::arg("size") = py::none(), py::arg("checksum") = py::none());
	file.def(
	    py::pickle([](const FILE_TYPE &value) { return value.State(); },
	               [class_name](const py::tuple &state) { return FileFromPickleState<FILE_TYPE>(state, class_name); }));
	if constexpr (std::is_same_v<FILE_TYPE, PythonImageFile>) {
		file.def(
		    "metadata",
		    [](const FILE_TYPE &value, const py::object &max_bytes, const py::object &max_pixels,
		       shared_ptr<DuckDBPyConnection> connection) {
			    return py::module_::import("vane._image_file")
			        .attr("_image_file_metadata_value")(
			            py::cast(value, py::return_value_policy::copy), py::arg("max_bytes") = max_bytes,
			            py::arg("max_pixels") = max_pixels, py::arg("connection") = std::move(connection));
		    },
		    "Inspect bounded encoded image headers without decoding pixels", py::kw_only(),
		    py::arg("max_bytes") = DEFAULT_IMAGE_METADATA_BYTES, py::arg("max_pixels") = DEFAULT_IMAGE_MAX_PIXELS,
		    py::arg("connection") = py::none());
		file.def(
		    "decode",
		    [](const FILE_TYPE &value, const py::object &mode, const py::object &buffer_size,
		       const py::object &max_input_bytes, const py::object &max_pixels, const py::object &max_decoded_bytes,
		       shared_ptr<DuckDBPyConnection> connection) {
			    return py::module_::import("vane._image_file")
			        .attr("_decode_image_file")(py::cast(value, py::return_value_policy::copy), mode, buffer_size,
			                                    py::arg("max_input_bytes") = max_input_bytes,
			                                    py::arg("max_pixels") = max_pixels,
			                                    py::arg("max_decoded_bytes") = max_decoded_bytes,
			                                    py::arg("connection") = std::move(connection));
		    },
		    "Decode frame zero into a fully loaded, detached Pillow image", py::arg("mode") = py::none(),
		    py::arg("buffer_size") = DEFAULT_IMAGE_BUFFER_SIZE, py::kw_only(),
		    py::arg("max_input_bytes") = DEFAULT_IMAGE_MAX_INPUT_BYTES,
		    py::arg("max_pixels") = DEFAULT_IMAGE_MAX_PIXELS,
		    py::arg("max_decoded_bytes") = DEFAULT_IMAGE_MAX_DECODED_BYTES, py::arg("connection") = py::none());
	} else if constexpr (std::is_same_v<FILE_TYPE, PythonAudioFile>) {
		file.def(
		    "metadata",
		    [](const FILE_TYPE &value, const py::object &max_bytes, shared_ptr<DuckDBPyConnection> connection) {
			    return py::module_::import("vane._audio_file")
			        .attr("_audio_file_metadata_value")(py::cast(value, py::return_value_policy::copy),
			                                            py::arg("max_bytes") = max_bytes,
			                                            py::arg("connection") = std::move(connection));
		    },
		    "Inspect bounded encoded audio metadata without decoding samples", py::kw_only(),
		    py::arg("max_bytes") = DEFAULT_AUDIO_METADATA_BYTES, py::arg("connection") = py::none());
		file.def(
		    "to_numpy",
		    [](const FILE_TYPE &value, const py::object &buffer_size, const py::object &max_input_bytes,
		       const py::object &max_frames, const py::object &max_decoded_bytes,
		       shared_ptr<DuckDBPyConnection> connection) {
			    return py::module_::import("vane._audio_file")
			        .attr("_decode_audio_file")(py::cast(value, py::return_value_policy::copy), buffer_size,
			                                    py::arg("max_input_bytes") = max_input_bytes,
			                                    py::arg("max_frames") = max_frames,
			                                    py::arg("max_decoded_bytes") = max_decoded_bytes,
			                                    py::arg("connection") = std::move(connection));
		    },
		    "Decode audio samples into a detached float64 (frames, channels) NumPy array",
		    py::arg("buffer_size") = DEFAULT_AUDIO_BUFFER_SIZE, py::kw_only(),
		    py::arg("max_input_bytes") = DEFAULT_AUDIO_MAX_INPUT_BYTES,
		    py::arg("max_frames") = DEFAULT_AUDIO_MAX_FRAMES,
		    py::arg("max_decoded_bytes") = DEFAULT_AUDIO_MAX_DECODED_BYTES, py::arg("connection") = py::none());
	} else if constexpr (std::is_same_v<FILE_TYPE, PythonVideoFile>) {
		file.def(
		    "metadata",
		    [](const FILE_TYPE &value, const py::object &max_bytes, shared_ptr<DuckDBPyConnection> connection) {
			    return py::module_::import("vane._video_file")
			        .attr("_video_file_metadata_value")(py::cast(value, py::return_value_policy::copy),
			                                            py::arg("max_bytes") = max_bytes,
			                                            py::arg("connection") = std::move(connection));
		    },
		    "Inspect the first video stream with bounded reads and no frame decoding", py::kw_only(),
		    py::arg("max_bytes") = DEFAULT_VIDEO_METADATA_BYTES, py::arg("connection") = py::none());
	}
}

static py::object ExecuteFileScalar(const PythonFile &file, shared_ptr<DuckDBPyConnection> connection,
                                    const string &query, vector<Value> parameters = {}) {
	if (!connection) {
		connection = DuckDBPyConnection::DefaultConnection();
	}
	parameters.insert(parameters.begin(), file.ToValue());
	Value value;
	ClientProperties client_properties;
	{
		D_ASSERT(py::gil_check());
		py::gil_scoped_release release;
		unique_lock<mutex> lock(connection->py_connection_lock);
		auto &native_connection = connection->con.GetConnection();
		auto pending = native_connection.PendingQuery(query, parameters);
		if (pending->HasError()) {
			pending->ThrowError();
		}
		auto result = DuckDBPyConnection::CompletePendingQuery(*pending);
		if (!result || result->HasError()) {
			if (result) {
				result->ThrowError();
			}
			throw InternalException("FILE metadata query returned no result");
		}
		auto chunk = result->Fetch();
		if (!chunk || chunk->size() != 1 || chunk->ColumnCount() != 1) {
			throw InternalException("FILE metadata query did not return exactly one value");
		}
		value = chunk->GetValue(0, 0);
		client_properties = native_connection.context->GetClientProperties();
	}
	return PythonObject::FromValue(value, value.type(), client_properties);
}

} // namespace

PythonFileMediaType::PythonFileMediaType(FileMediaType media_type_p) : media_type(media_type_p) {
}

PythonFileMediaType PythonFileMediaType::Unknown() {
	return PythonFileMediaType(FileMediaType::UNKNOWN);
}

PythonFileMediaType PythonFileMediaType::Image() {
	return PythonFileMediaType(FileMediaType::IMAGE);
}

PythonFileMediaType PythonFileMediaType::Audio() {
	return PythonFileMediaType(FileMediaType::AUDIO);
}

PythonFileMediaType PythonFileMediaType::Video() {
	return PythonFileMediaType(FileMediaType::VIDEO);
}

FileMediaType PythonFileMediaType::Type() const {
	return media_type;
}

string PythonFileMediaType::Repr() const {
	switch (media_type) {
	case FileMediaType::UNKNOWN:
		return "MediaType.unknown()";
	case FileMediaType::IMAGE:
		return "MediaType.image()";
	case FileMediaType::AUDIO:
		return "MediaType.audio()";
	case FileMediaType::VIDEO:
		return "MediaType.video()";
	default:
		throw InternalException("Unknown FILE media type");
	}
}

bool PythonFileMediaType::Equals(const PythonFileMediaType &other) const {
	return media_type == other.media_type;
}

Py_hash_t PythonFileMediaType::Hash() const {
	return py::hash(py::int_(static_cast<int>(media_type)));
}

PythonFile::PythonFile(string url_p, std::optional<string> content_type_p, std::optional<int64_t> position_p,
                       std::optional<int64_t> size_p, std::optional<string> checksum_p, FileMediaType media_type_p)
    : media_type(media_type_p), url(std::move(url_p)), content_type(std::move(content_type_p)), position(position_p),
      size(size_p), checksum(std::move(checksum_p)) {
	MakeReference(url, content_type, position, size, checksum, media_type);
}

void PythonFile::Initialize(py::handle &m) {
	auto media_type = py::class_<PythonFileMediaType>(m, "MediaType", py::module_local(), py::is_final());
	media_type.def_static("unknown", &PythonFileMediaType::Unknown);
	media_type.def_static("image", &PythonFileMediaType::Image);
	media_type.def_static("audio", &PythonFileMediaType::Audio);
	media_type.def_static("video", &PythonFileMediaType::Video);
	media_type.def("__repr__", &PythonFileMediaType::Repr);
	media_type.def("__eq__", &PythonFileMediaType::Equals, py::arg("other"), py::is_operator());
	media_type.def("__hash__", &PythonFileMediaType::Hash);

	auto file = py::class_<PythonFile>(m, "File", py::module_local());
	file.def(py::init([](const py::object &url, const py::object &content_type, const py::object &position,
	                     const py::object &size, const py::object &checksum) {
		         return PythonFile::FromPython(url, content_type, position, size, checksum);
	         }),
	         py::arg("url"), py::arg("content_type") = py::none(), py::arg("position") = py::none(),
	         py::arg("size") = py::none(), py::arg("checksum") = py::none());
	file.def_property_readonly("url", &PythonFile::Url);
	file.def_property_readonly("content_type", &PythonFile::ContentType);
	file.def_property_readonly("position", &PythonFile::Position);
	file.def_property_readonly("size", &PythonFile::Size);
	file.def_property_readonly("checksum", &PythonFile::Checksum);
	file.def("exists", &PythonFile::Exists, "Return whether this FILE's logical view is accessible", py::kw_only(),
	         py::arg("connection") = py::none());
	file.def("stat", &PythonFile::Stat, "Return the six-field SQL file_stat value", py::kw_only(),
	         py::arg("connection") = py::none());
	file.def("mime_type", &PythonFile::MimeType, "Return the MIME type selected by SQL file_mime_type",
	         py::arg("detect") = "metadata", py::kw_only(), py::arg("connection") = py::none());
	file.def(
	    "open",
	    [](const PythonFile &value, const py::object &buffer_size, shared_ptr<DuckDBPyConnection> connection) {
		    return py::module_::import("vane._file")
		        .attr("_file_open")(py::cast(value, py::return_value_policy::copy), buffer_size,
		                            py::arg("connection") = std::move(connection));
	    },
	    "Open this FILE as a read-only VaneFileReader", py::arg("buffer_size") = py::none(), py::kw_only(),
	    py::arg("connection") = py::none());
	file.def(
	    "to_tempfile",
	    [](const PythonFile &value, const py::object &buffer_size, shared_ptr<DuckDBPyConnection> connection) {
		    return py::module_::import("vane._file")
		        .attr("_file_to_tempfile")(py::cast(value, py::return_value_policy::copy), buffer_size,
		                                   py::arg("connection") = std::move(connection));
	    },
	    "Copy this FILE's logical view into a temporary binary file", py::arg("buffer_size") = 1024 * 1024,
	    py::kw_only(), py::arg("connection") = py::none());
	file.def("__str__", &PythonFile::ToString);
	file.def("__repr__", &PythonFile::Repr);
	file.def("__eq__", &PythonFile::Equals, py::arg("other"), py::is_operator());
	file.def("__ne__", &PythonFile::NotEquals, py::arg("other"), py::is_operator());
	file.def("__hash__", &PythonFile::Hash);
	file.def(py::pickle([](const PythonFile &value) { return value.State(); },
	                    [](const py::tuple &state) { return FileFromPickleState<PythonFile>(state, "File"); }));
	BindMediaFileClass<PythonImageFile>(m, "ImageFile");
	BindMediaFileClass<PythonAudioFile>(m, "AudioFile");
	BindMediaFileClass<PythonVideoFile>(m, "VideoFile");

	// Native media subclasses are registered above; keep the public hierarchy
	// closed so user-defined subclasses cannot add state to governed values.
	reinterpret_cast<PyTypeObject *>(file.ptr())->tp_flags &= ~Py_TPFLAGS_BASETYPE;
}

PythonFile PythonFile::FromPython(const py::handle &url, const py::handle &content_type, const py::handle &position,
                                  const py::handle &size, const py::handle &checksum) {
	return FileFromPython<PythonFile>(url, content_type, position, size, checksum);
}

py::object PythonFile::FromValue(const Value &value) {
	auto reference = FileReference::FromValue(value, "FILE materialization");
	auto content_type = OptionalStringFromReference(reference.has_content_type, reference.content_type);
	auto position = OptionalIntegerFromReference(reference.has_range, reference.position);
	auto size = OptionalIntegerFromReference(reference.has_range, reference.size);
	auto checksum = OptionalStringFromReference(reference.has_checksum, reference.checksum);
	switch (reference.media_type) {
	case FileMediaType::UNKNOWN:
		return py::cast(
		    PythonFile(std::move(reference.url), std::move(content_type), position, size, std::move(checksum)));
	case FileMediaType::IMAGE:
		return py::cast(
		    PythonImageFile(std::move(reference.url), std::move(content_type), position, size, std::move(checksum)));
	case FileMediaType::AUDIO:
		return py::cast(
		    PythonAudioFile(std::move(reference.url), std::move(content_type), position, size, std::move(checksum)));
	case FileMediaType::VIDEO:
		return py::cast(
		    PythonVideoFile(std::move(reference.url), std::move(content_type), position, size, std::move(checksum)));
	default:
		throw InternalException("Unknown FILE media type");
	}
}

Value PythonFile::ToValue() const {
	return MakeReference(url, content_type, position, size, checksum, media_type).ToValue();
}

string PythonFile::ToString() const {
	return url;
}

string PythonFile::Repr() const {
	auto state = State();
	string class_name;
	switch (media_type) {
	case FileMediaType::UNKNOWN:
		class_name = "File";
		break;
	case FileMediaType::IMAGE:
		class_name = "ImageFile";
		break;
	case FileMediaType::AUDIO:
		class_name = "AudioFile";
		break;
	case FileMediaType::VIDEO:
		class_name = "VideoFile";
		break;
	default:
		throw InternalException("Unknown FILE media type");
	}
	return class_name + "(url=" + py::repr(state[FileLogicalType::URL]).cast<string>() +
	       ", content_type=" + py::repr(state[FileLogicalType::CONTENT_TYPE]).cast<string>() +
	       ", position=" + py::repr(state[FileLogicalType::POSITION]).cast<string>() +
	       ", size=" + py::repr(state[FileLogicalType::SIZE]).cast<string>() +
	       ", checksum=" + py::repr(state[FileLogicalType::CHECKSUM]).cast<string>() + ")";
}

bool PythonFile::Equals(const PythonFile &other) const {
	return media_type == other.media_type && url == other.url && content_type == other.content_type &&
	       position == other.position && size == other.size && checksum == other.checksum;
}

bool PythonFile::NotEquals(const PythonFile &other) const {
	return !Equals(other);
}

Py_hash_t PythonFile::Hash() const {
	return py::hash(py::make_tuple(static_cast<int>(media_type), url, content_type, position, size, checksum));
}

py::tuple PythonFile::State() const {
	return py::make_tuple(url, content_type, position, size, checksum);
}

py::object PythonFile::Exists(shared_ptr<DuckDBPyConnection> connection) const {
	return ExecuteFileScalar(*this, std::move(connection), "SELECT file_exists(?)");
}

py::object PythonFile::Stat(shared_ptr<DuckDBPyConnection> connection) const {
	return ExecuteFileScalar(*this, std::move(connection), "SELECT file_stat(?)");
}

py::object PythonFile::MimeType(const string &detect, shared_ptr<DuckDBPyConnection> connection) const {
	if (detect == "metadata") {
		return ExecuteFileScalar(*this, std::move(connection), "SELECT file_mime_type(?)");
	}
	return ExecuteFileScalar(*this, std::move(connection), "SELECT file_mime_type(?, ?)", {Value(detect)});
}

const string &PythonFile::Url() const {
	return url;
}

const std::optional<string> &PythonFile::ContentType() const {
	return content_type;
}

const std::optional<int64_t> &PythonFile::Position() const {
	return position;
}

const std::optional<int64_t> &PythonFile::Size() const {
	return size;
}

const std::optional<string> &PythonFile::Checksum() const {
	return checksum;
}

FileMediaType PythonFile::MediaType() const {
	return media_type;
}

PythonImageFile::PythonImageFile(string url, std::optional<string> content_type, std::optional<int64_t> position,
                                 std::optional<int64_t> size, std::optional<string> checksum)
    : PythonFile(std::move(url), std::move(content_type), position, size, std::move(checksum), FileMediaType::IMAGE) {
}

PythonAudioFile::PythonAudioFile(string url, std::optional<string> content_type, std::optional<int64_t> position,
                                 std::optional<int64_t> size, std::optional<string> checksum)
    : PythonFile(std::move(url), std::move(content_type), position, size, std::move(checksum), FileMediaType::AUDIO) {
}

PythonVideoFile::PythonVideoFile(string url, std::optional<string> content_type, std::optional<int64_t> position,
                                 std::optional<int64_t> size, std::optional<string> checksum)
    : PythonFile(std::move(url), std::move(content_type), position, size, std::move(checksum), FileMediaType::VIDEO) {
}

} // namespace duckdb

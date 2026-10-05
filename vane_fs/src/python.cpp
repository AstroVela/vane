// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include "native_reader.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

namespace py = pybind11;
using namespace vane_fs;

PYBIND11_MODULE(_native, module) {
	module.doc() = "Native SQLite storage and branching for VaneFS";
	RegisterNativeReader(module);
	py::register_exception<Error>(module, "Error");
	for (const auto *name : {"ConflictError", "StalePreviewError", "CapacityError"}) {
		auto qualified = std::string("vane_fs._native.") + name;
		module.attr(name) = py::reinterpret_steal<py::object>(
		    PyErr_NewException(qualified.c_str(), module.attr("Error").ptr(), nullptr));
	}
	py::register_local_exception_translator([](std::exception_ptr exception) {
		try {
			if (exception) {
				std::rethrow_exception(exception);
			}
		} catch (const Error &error) {
			PyObject *type = nullptr;
			switch (error.code) {
			case ErrorCode::Invalid:
			case ErrorCode::Closed:
				type = PyExc_ValueError;
				break;
			case ErrorCode::NotFound:
				type = PyExc_FileNotFoundError;
				break;
			case ErrorCode::Exists:
				type = PyExc_FileExistsError;
				break;
			case ErrorCode::NotDirectory:
				type = PyExc_NotADirectoryError;
				break;
			case ErrorCode::IsDirectory:
				type = PyExc_IsADirectoryError;
				break;
			case ErrorCode::NotEmpty:
				type = PyExc_OSError;
				break;
			case ErrorCode::ReadOnly:
				type = PyExc_PermissionError;
				break;
			case ErrorCode::Busy:
				type = PyExc_BlockingIOError;
				break;
			default:
				break;
			}
			if (type) {
				PyErr_SetString(type, error.what());
				return;
			}
			auto native = py::module_::import("vane_fs._native");
			const char *name = error.code == ErrorCode::Conflict   ? "ConflictError"
			                   : error.code == ErrorCode::Stale    ? "StalePreviewError"
			                   : error.code == ErrorCode::Capacity ? "CapacityError"
			                                                       : "Error";
			PyErr_SetString(native.attr(name).ptr(), error.what());
		}
	});
	py::class_<FileStat>(module, "FileStat")
	    .def_readonly("inode", &FileStat::inode)
	    .def_readonly("is_directory", &FileStat::is_directory)
	    .def_readonly("size", &FileStat::size)
	    .def_readonly("mode", &FileStat::mode)
	    .def_readonly("mtime_ns", &FileStat::mtime_ns)
	    .def_readonly("links", &FileStat::links);
	py::class_<BranchInfo>(module, "BranchInfo")
	    .def_readonly("id", &BranchInfo::id)
	    .def_readonly("name", &BranchInfo::name)
	    .def_readonly("parent_id", &BranchInfo::parent_id)
	    .def_readonly("fork_base", &BranchInfo::fork_base)
	    .def_readonly("state", &BranchInfo::state)
	    .def_readonly("generation", &BranchInfo::generation);
	py::class_<Change>(module, "Change").def_readonly("path", &Change::path).def_readonly("kind", &Change::kind);
	py::class_<MergePreview>(module, "MergePreview")
	    .def_readonly("workspace_id", &MergePreview::workspace_id)
	    .def_readonly("source", &MergePreview::source)
	    .def_readonly("target", &MergePreview::target)
	    .def_readonly("source_generation", &MergePreview::source_generation)
	    .def_readonly("target_generation", &MergePreview::target_generation)
	    .def_readonly("changes", &MergePreview::changes)
	    .def_readonly("conflicts", &MergePreview::conflicts);
	py::class_<CollectionResult>(module, "CollectionResult")
	    .def_readonly("versions", &CollectionResult::versions)
	    .def_readonly("payloads", &CollectionResult::payloads)
	    .def_readonly("snapshots", &CollectionResult::snapshots);
	py::class_<RecoveryResult>(module, "RecoveryResult")
	    .def_readonly("owners", &RecoveryResult::owners)
	    .def_readonly("pins", &RecoveryResult::pins)
	    .def_readonly("mounts", &RecoveryResult::mounts);
	py::class_<Session, std::shared_ptr<Session>>(module, "Session")
	    .def_property_readonly("id", &Session::Id)
	    .def_property_readonly("is_snapshot", &Session::IsSnapshot)
	    .def("close", &Session::Close, py::call_guard<py::gil_scoped_release>())
	    .def(
	        "__enter__", [](Session &self) -> Session & { return self; }, py::return_value_policy::reference_internal)
	    .def("__exit__",
	         [](Session &self, py::object, py::object, py::object) {
		         py::gil_scoped_release release;
		         self.Close();
	         })
	    .def("stat", &Session::Stat, py::arg("path"), py::call_guard<py::gil_scoped_release>())
	    .def("listdir", &Session::ListDirectory, py::arg("path") = "/", py::call_guard<py::gil_scoped_release>())
	    .def(
	        "read",
	        [](Session &self, const std::string &path, int64_t offset, int64_t size) {
		        std::string data;
		        {
			        py::gil_scoped_release release;
			        data = self.Read(path, offset, size);
		        }
		        return py::bytes(data);
	        },
	        py::arg("path"), py::arg("offset") = 0, py::arg("size") = -1)
	    .def("mkdir", &Session::MakeDirectory, py::arg("path"), py::arg("mode") = 0755,
	         py::call_guard<py::gil_scoped_release>())
	    .def(
	        "write_file",
	        [](Session &self, const std::string &path, py::bytes value) {
		        std::string data = value;
		        py::gil_scoped_release release;
		        self.WriteFile(path, data);
	        },
	        py::arg("path"), py::arg("data"))
	    .def(
	        "write",
	        [](Session &self, const std::string &path, py::bytes value, int64_t offset) {
		        std::string data = value;
		        py::gil_scoped_release release;
		        self.Write(path, data, offset);
	        },
	        py::arg("path"), py::arg("data"), py::arg("offset") = 0)
	    .def("truncate", &Session::Truncate, py::arg("path"), py::arg("size"), py::call_guard<py::gil_scoped_release>())
	    .def("rename", &Session::Rename, py::arg("source"), py::arg("target"), py::arg("no_replace") = false,
	         py::call_guard<py::gil_scoped_release>())
	    .def("unlink", &Session::Unlink, py::arg("path"), py::call_guard<py::gil_scoped_release>())
	    .def("rmdir", &Session::RemoveDirectory, py::arg("path"), py::call_guard<py::gil_scoped_release>());
	py::class_<Workspace>(module, "Workspace")
	    .def(py::init([](const std::filesystem::path &path, int timeout_ms) {
		         return std::make_unique<Workspace>(path.u8string(), timeout_ms);
	         }),
	         py::arg("path"), py::arg("timeout_ms") = 5000, py::call_guard<py::gil_scoped_release>())
	    .def_property_readonly("id", &Workspace::Id)
	    .def_static("sqlite_version", &Workspace::SQLiteVersion)
	    .def("close", &Workspace::Close, py::call_guard<py::gil_scoped_release>())
	    .def(
	        "__enter__", [](Workspace &self) -> Workspace & { return self; },
	        py::return_value_policy::reference_internal)
	    .def("__exit__",
	         [](Workspace &self, py::object, py::object, py::object) {
		         py::gil_scoped_release release;
		         self.Close();
	         })
	    .def("branch", &Workspace::GetBranch, py::arg("branch") = "main", py::call_guard<py::gil_scoped_release>())
	    .def("branches", &Workspace::ListBranches, py::call_guard<py::gil_scoped_release>())
	    .def("fork", &Workspace::Fork, py::arg("source"), py::arg("name"), py::arg("terminal") = false,
	         py::call_guard<py::gil_scoped_release>())
	    .def("checkout", &Workspace::Checkout, py::arg("branch") = "main", py::call_guard<py::gil_scoped_release>())
	    .def("snapshot", &Workspace::Snapshot, py::arg("branch") = "main", py::call_guard<py::gil_scoped_release>())
	    .def("open_snapshot", &Workspace::OpenSnapshot, py::arg("snapshot"), py::call_guard<py::gil_scoped_release>())
	    .def("drop_snapshot", &Workspace::DropSnapshot, py::arg("snapshot"), py::call_guard<py::gil_scoped_release>())
	    .def("diff", &Workspace::Diff, py::arg("source_snapshot"), py::arg("target_snapshot"),
	         py::call_guard<py::gil_scoped_release>())
	    .def("preview_merge", &Workspace::PreviewMerge, py::arg("source"), py::arg("target"),
	         py::call_guard<py::gil_scoped_release>())
	    .def("merge", &Workspace::Merge, py::arg("preview"),
	         py::arg("resolutions") = std::map<std::string, std::string> {}, py::call_guard<py::gil_scoped_release>())
	    .def("delete_branch", &Workspace::DeleteBranch, py::arg("branch"), py::arg("recursive") = false,
	         py::call_guard<py::gil_scoped_release>())
	    .def("collect_garbage", &Workspace::CollectGarbage, py::call_guard<py::gil_scoped_release>())
	    .def("recover_owners", &Workspace::RecoverOwners, py::call_guard<py::gil_scoped_release>())
	    .def("acquire_mount", &Workspace::AcquireMount, py::arg("branch") = "main",
	         py::call_guard<py::gil_scoped_release>())
	    .def("release_mount", &Workspace::ReleaseMount, py::arg("branch") = "main",
	         py::call_guard<py::gil_scoped_release>());
}

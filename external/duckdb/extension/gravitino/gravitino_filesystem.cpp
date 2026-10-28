// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "gravitino_catalog.hpp"
#include "duckdb/common/file_opener.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/distributed/client_state.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/copy_info.hpp"
#include <cstring>

namespace duckdb {
class GravitinoFileSystem : public FileSystem {
public:
	string GetName() const override {
		return "GravitinoFileSystem";
	}
	bool CanHandleFile(const string &path) override {
		return StringUtil::StartsWith(path, "gvfs://");
	}
	unique_ptr<FileHandle> OpenFile(const string &path, FileOpenFlags flags, optional_ptr<FileOpener> opener) override {
		if (flags.OpenForWriting() || flags.OpenForAppending() || flags.CreateFileIfNotExists() ||
		    flags.OverwriteExistingFile()) {
			throw NotImplementedException("Gravitino Fileset content writes are unsupported");
		}
		auto &context = Context(opener);
		return FileSystem::GetFileSystem(context).OpenFile(Resolve(context, path), flags);
	}
	bool FileExists(const string &path, optional_ptr<FileOpener> opener) override {
		auto &context = Context(opener);
		return FileSystem::GetFileSystem(context).FileExists(Resolve(context, path));
	}
	bool DirectoryExists(const string &path, optional_ptr<FileOpener> opener) override {
		auto &context = Context(opener);
		return FileSystem::GetFileSystem(context).DirectoryExists(Resolve(context, path));
	}
	bool HasDirectorySemantics(const string &path, optional_ptr<FileOpener> opener) override {
		auto &context = Context(opener);
		return FileSystem::GetFileSystem(context).HasDirectorySemantics(Resolve(context, path));
	}
	bool ListFiles(const string &path, const std::function<void(const string &, bool)> &callback,
	               FileOpener *opener) override {
		auto &context = Context(opener);
		return FileSystem::GetFileSystem(context).ListFiles(Resolve(context, path), callback);
	}
	vector<OpenFileInfo> Glob(const string &path, FileOpener *opener) override {
		auto &context = Context(opener);
		auto parts = ParsePath(path);
		return GravitinoCatalog::Get(context, parts.attachment)
		    .client.Glob(context, parts.schema, parts.fileset, parts.relative);
	}
	string CanonicalizePath(const string &path, optional_ptr<FileOpener> opener) override {
		return Resolve(Context(opener), path);
	}
	FileWriteTarget ResolveWriteTarget(const string &path, const CopyInfo &info,
	                                   optional_ptr<FileOpener> opener) override {
		auto &context = Context(opener);
		auto parts = ParsePath(path);
		auto &catalog = GravitinoCatalog::Get(context, parts.attachment);
		catalog.RequireMutation(context);
		if (info.format != "parquet" && info.format != "csv") {
			throw NotImplementedException("Gravitino Fileset writes support parquet and csv only");
		}
		auto enabled = [&](const char *name) {
			auto found = info.options.find(name);
			return found != info.options.end() &&
			       (found->second.empty() || found->second[0].DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>());
		};
		auto partitions = info.options.find("partition_by");
		const bool partitioned = partitions != info.options.end();
		if (partitioned && (partitions->second.empty() || (partitions->second.size() == 1 &&
		                                                   partitions->second[0].type().id() == LogicalTypeId::LIST &&
		                                                   ListValue::GetChildren(partitions->second[0]).empty()))) {
			throw InvalidInputException("Gravitino PARTITION_BY requires at least one column");
		}
		if (!enabled("append") || (!enabled("per_thread_output") && !partitioned) || enabled("overwrite") ||
		    enabled("overwrite_or_ignore") || enabled("use_tmp_file")) {
			throw NotImplementedException(
			    "Gravitino Fileset writes require directory output with APPEND and PER_THREAD_OUTPUT or PARTITION_BY; "
			    "overwrite and temporary-file publication are unsupported");
		}
		if (info.options.count("filename_pattern") || info.options.count("file_extension")) {
			throw NotImplementedException("Custom Gravitino Fileset output filenames are unsupported");
		}
		if (parts.relative.find_first_of("*?[]{}") != string::npos) {
			throw InvalidInputException("Fileset output paths cannot contain glob patterns");
		}
		FileWriteTarget result;
		string location;
		result.path = catalog.client.Resolve(context, parts.schema, parts.fileset, parts.relative, nullptr, &location);
		if (RunnerClientState::Get(context) == "ray" && !StringUtil::StartsWith(result.path, "s3://")) {
			throw NotImplementedException("Ray Fileset writes require a shared S3 destination; "
			                              "node-local output is unsupported");
		}
		result.requires_rebind = true;
		result.properties["location_name"] = std::move(location);
		return result;
	}

private:
	struct FilesetPath {
		string attachment;
		string schema;
		string fileset;
		string relative;
	};

	static ClientContext &Context(optional_ptr<FileOpener> opener) {
		auto context = FileOpener::TryGetClientContext(opener);
		if (!context) {
			throw InvalidInputException("Gravitino Fileset access requires its attached connection");
		}
		return *context;
	}
	static FilesetPath ParsePath(const string &path) {
		static constexpr const char *PREFIX = "gvfs://fileset/";
		if (!StringUtil::StartsWith(path, PREFIX)) {
			throw InvalidInputException("Expected gvfs://fileset/<attachment>/<schema>/<fileset>/<path>");
		}
		auto parts = StringUtil::Split(path.substr(strlen(PREFIX)), '/');
		if (parts.size() < 3) {
			throw InvalidInputException("Gravitino Fileset URL must identify an attachment, schema and Fileset");
		}
		string relative;
		for (idx_t i = 3; i < parts.size(); i++) {
			if (i != 3) {
				relative += '/';
			}
			relative += parts[i];
		}
		return {parts[0], parts[1], parts[2], relative};
	}
	static string Resolve(ClientContext &context, const string &path) {
		auto parts = ParsePath(path);
		return GravitinoCatalog::Get(context, parts.attachment)
		    .client.Resolve(context, parts.schema, parts.fileset, parts.relative);
	}
};
void RegisterGravitinoFileSystem(ExtensionLoader &loader) {
	FileSystem::GetFileSystem(loader.GetDatabaseInstance()).RegisterSubSystem(make_uniq<GravitinoFileSystem>());
}
} // namespace duckdb

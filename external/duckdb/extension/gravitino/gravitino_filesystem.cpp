// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "gravitino_catalog.hpp"
#include "duckdb/common/file_opener.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
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
		return FileSystem::GetFileSystem(context).Glob(Resolve(context, path), nullptr);
	}
	string CanonicalizePath(const string &path, optional_ptr<FileOpener> opener) override {
		return Resolve(Context(opener), path);
	}

private:
	static ClientContext &Context(optional_ptr<FileOpener> opener) {
		auto context = FileOpener::TryGetClientContext(opener);
		if (!context) {
			throw InvalidInputException("Gravitino Fileset access requires its attached connection");
		}
		return *context;
	}
	static string Resolve(ClientContext &context, const string &path) {
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
		auto &catalog = GravitinoCatalog::Get(context, parts[0]);
		return catalog.client.Resolve(context, parts[1], parts[2], relative);
	}
};
void RegisterGravitinoFileSystem(ExtensionLoader &loader) {
	FileSystem::GetFileSystem(loader.GetDatabaseInstance()).RegisterSubSystem(make_uniq<GravitinoFileSystem>());
}
} // namespace duckdb

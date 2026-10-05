// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#define FUSE_USE_VERSION 31
#include <fuse_lowlevel.h>
#include "vane_fs/workspace.hpp"
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <unistd.h>

using namespace vane_fs;
namespace {
// The mount's exclusive branch lease routes every live mutation through this
// kernel, which invalidates its entries/attributes on mutations. Snapshots are
// immutable. Out-of-band writers would require explicit cache invalidation.
constexpr double METADATA_TIMEOUT = 60.0;

struct Mount {
	Workspace workspace;
	std::shared_ptr<Session> session;
	std::string database;
	bool readonly;
	fuse_session *fuse = nullptr;
	uid_t uid = getuid();
	gid_t gid = getgid();
	std::map<uint64_t, std::vector<std::pair<std::string, FileStat>>> directories;
	uint64_t next_directory = 0;
	Mount(const std::string &path, const std::string &id, bool readonly)
	    : workspace(path), database(path), readonly(readonly) {
		workspace.RecoverOwners();
		session = readonly ? workspace.OpenSnapshot(id) : workspace.Checkout(workspace.AcquireMount(id).id);
		session->OpenDirectory("/");
	}
};
Mount &Context(fuse_req_t request) {
	return *static_cast<Mount *>(fuse_req_userdata(request));
}
int Errno(ErrorCode code) {
	switch (code) {
	case ErrorCode::Invalid:
		return EINVAL;
	case ErrorCode::NotFound:
		return ENOENT;
	case ErrorCode::Exists:
		return EEXIST;
	case ErrorCode::NotDirectory:
		return ENOTDIR;
	case ErrorCode::IsDirectory:
		return EISDIR;
	case ErrorCode::NotEmpty:
		return ENOTEMPTY;
	case ErrorCode::ReadOnly:
		return EROFS;
	case ErrorCode::Busy:
		return EBUSY;
	case ErrorCode::Capacity:
		return ENOSPC;
	case ErrorCode::Closed:
		return EBADF;
	default:
		return EIO;
	}
}
template <class F>
void Guard(fuse_req_t request, F operation) noexcept {
	try {
		operation();
	} catch (const Error &error) {
		fuse_reply_err(request, Errno(error.code));
	} catch (const std::bad_alloc &) {
		fuse_reply_err(request, ENOMEM);
	} catch (const std::exception &error) {
		std::cerr << "VaneFS: " << error.what() << '\n';
		fuse_reply_err(request, EIO);
	} catch (...) {
		fuse_reply_err(request, EIO);
	}
}
void Mutable(Mount &mount) {
	if (mount.readonly)
		throw Error(ErrorCode::ReadOnly, "Snapshot mount is read-only");
}
void Close(Mount &mount, int64_t inode, uint64_t references) noexcept {
	try {
		if (references > uint64_t(INT64_MAX))
			throw Error(ErrorCode::Invalid, "Invalid inode reference count");
		mount.session->CloseFile(inode, int64_t(references));
	} catch (const std::exception &error) {
		// A failed FORGET conservatively retains data until owner cleanup.
		std::cerr << "VaneFS releasing inode: " << error.what() << '\n';
	}
}
struct stat Describe(Mount &mount, const FileStat &file) {
	struct stat result {};
	result.st_ino = file.inode;
	result.st_mode = (file.is_directory ? S_IFDIR : S_IFREG) | mode_t(file.mode);
	result.st_nlink = file.links;
	result.st_uid = mount.uid;
	result.st_gid = mount.gid;
	result.st_size = file.size;
	result.st_blksize = 4096;
	result.st_blocks = file.size / 512 + (file.size % 512 != 0);
	auto seconds = file.mtime_ns / 1000000000, nanoseconds = file.mtime_ns % 1000000000;
	if (nanoseconds < 0) {
		--seconds;
		nanoseconds += 1000000000;
	}
	result.st_mtim = {seconds, nanoseconds};
	result.st_atim = result.st_ctim = result.st_mtim;
	return result;
}
fuse_entry_param Entry(Mount &mount, const FileStat &stat) {
	fuse_entry_param result {};
	result.ino = stat.inode;
	result.generation = 1; // Inode IDs are never reused.
	result.attr = Describe(mount, stat);
	result.entry_timeout = METADATA_TIMEOUT;
	result.attr_timeout = METADATA_TIMEOUT;
	return result;
}
void ReplyEntry(fuse_req_t request, const FileStat &stat) {
	auto &mount = Context(request);
	auto entry = Entry(mount, stat);
	if (fuse_reply_entry(request, &entry) < 0)
		Close(mount, stat.inode, 1);
}
void ConfigureHandle(Mount &mount, fuse_file_info *info, int64_t inode) {
	info->fh = inode;
	info->direct_io = !mount.readonly;
	info->keep_cache = mount.readonly;
}
void InvalidateAttributes(Mount &mount, int64_t inode) {
	// Linux invalidates size/mtime after writes, but our atime aliases mtime.
	// Invalidate all attributes before replying, including for statx(ATIME).
	// An attributes-only notification has no dirty pages to flush or wait for.
	auto error = fuse_lowlevel_notify_inval_inode(mount.fuse, inode, -1, 0);
	if (error && error != -ENOENT)
		throw Error(ErrorCode::Storage, "Could not invalidate inode attributes: " + std::to_string(-error));
}
fuse_lowlevel_ops Operations() {
	fuse_lowlevel_ops ops {};
	ops.init = [](void *, fuse_conn_info *connection) {
		connection->want &= ~(FUSE_CAP_WRITEBACK_CACHE | FUSE_CAP_DONT_MASK | FUSE_CAP_HANDLE_KILLPRIV);
	};
	ops.lookup = [](fuse_req_t req, fuse_ino_t parent, const char *name) {
		Guard(req, [&] { ReplyEntry(req, Context(req).session->LookupInode(parent, name)); });
	};
	ops.forget = [](fuse_req_t req, fuse_ino_t inode, uint64_t count) {
		Close(Context(req), inode, count);
		fuse_reply_none(req);
	};
	ops.forget_multi = [](fuse_req_t req, size_t count, fuse_forget_data *forgets) {
		for (size_t i = 0; i < count; ++i)
			Close(Context(req), forgets[i].ino, forgets[i].nlookup);
		fuse_reply_none(req);
	};
	ops.getattr = [](fuse_req_t req, fuse_ino_t inode, fuse_file_info *) {
		Guard(req, [&] {
			auto stat = Describe(Context(req), Context(req).session->StatInode(inode));
			fuse_reply_attr(req, &stat, METADATA_TIMEOUT);
		});
	};
	ops.open = [](fuse_req_t req, fuse_ino_t inode, fuse_file_info *info) {
		Guard(req, [&] {
			auto &mount = Context(req);
			if ((info->flags & O_ACCMODE) != O_RDONLY || (info->flags & O_TRUNC))
				Mutable(mount);
			mount.session->OpenInode(inode, false, info->flags & O_TRUNC);
			try {
				if (info->flags & O_TRUNC)
					InvalidateAttributes(mount, inode);
			} catch (...) {
				Close(mount, inode, 1);
				throw;
			}
			ConfigureHandle(mount, info, inode);
			if (fuse_reply_open(req, info) < 0)
				Close(mount, inode, 1);
		});
	};
	ops.create = [](fuse_req_t req, fuse_ino_t parent, const char *name, mode_t mode, fuse_file_info *info) {
		Guard(req, [&] {
			auto &mount = Context(req);
			Mutable(mount);
			if ((mode & 07777) & ~0777) {
				fuse_reply_err(req, EOPNOTSUPP);
				return;
			}
			auto stat = mount.session->CreateNode(parent, name, false, mode & 0777, info->flags & O_EXCL,
			                                      info->flags & O_TRUNC, 2);
			auto entry = Entry(mount, stat);
			ConfigureHandle(mount, info, stat.inode);
			if (fuse_reply_create(req, &entry, info) < 0)
				Close(mount, stat.inode, 2);
		});
	};
	ops.read = [](fuse_req_t req, fuse_ino_t, size_t size, off_t offset, fuse_file_info *info) {
		Guard(req, [&] {
			if (size > INT_MAX) {
				fuse_reply_err(req, EINVAL);
				return;
			}
			auto data = Context(req).session->ReadInode(info->fh, offset, int64_t(size));
			fuse_reply_buf(req, data.data(), data.size());
		});
	};
	ops.write = [](fuse_req_t req, fuse_ino_t, const char *buffer, size_t size, off_t offset, fuse_file_info *info) {
		Guard(req, [&] {
			auto &mount = Context(req);
			Mutable(mount);
			mount.session->WriteInode(info->fh, std::string(buffer, size), offset, info->flags & O_APPEND);
			InvalidateAttributes(mount, info->fh);
			fuse_reply_write(req, size);
		});
	};
	ops.release = [](fuse_req_t req, fuse_ino_t, fuse_file_info *info) {
		Guard(req, [&] {
			Context(req).session->CloseFile(info->fh);
			fuse_reply_err(req, 0);
		});
	};
	ops.opendir = [](fuse_req_t req, fuse_ino_t inode, fuse_file_info *info) {
		Guard(req, [&] {
			auto &mount = Context(req);
			mount.session->OpenInode(inode, true);
			try {
				// Cookies index this handle's fixed listing, so namespace
				// mutations cannot shift entries between readdir requests.
				auto entries = mount.session->DirectoryEntries(inode);
				info->fh = ++mount.next_directory;
				mount.directories.emplace(info->fh, std::move(entries));
			} catch (...) {
				Close(mount, inode, 1);
				throw;
			}
			if (fuse_reply_open(req, info) < 0) {
				mount.directories.erase(info->fh);
				Close(mount, inode, 1);
			}
		});
	};
	ops.readdir = [](fuse_req_t req, fuse_ino_t, size_t size, off_t offset, fuse_file_info *info) {
		Guard(req, [&] {
			if (offset < 0 || size > INT_MAX) {
				fuse_reply_err(req, EINVAL);
				return;
			}
			if (size == 0) {
				fuse_reply_buf(req, nullptr, 0);
				return;
			}
			auto &mount = Context(req);
			const auto &entries = mount.directories.at(info->fh);
			std::vector<char> buffer(size);
			size_t used = 0;
			for (size_t i = size_t(offset); i < entries.size(); ++i) {
				auto stat = Describe(mount, entries[i].second);
				auto amount = fuse_add_direntry(req, buffer.data() + used, size - used, entries[i].first.c_str(), &stat,
				                                off_t(i + 1));
				if (amount > size - used)
					break;
				used += amount;
			}
			fuse_reply_buf(req, buffer.data(), used);
		});
	};
	ops.releasedir = [](fuse_req_t req, fuse_ino_t inode, fuse_file_info *info) {
		Guard(req, [&] {
			auto &mount = Context(req);
			mount.directories.erase(info->fh);
			mount.session->CloseFile(inode);
			fuse_reply_err(req, 0);
		});
	};
	ops.mkdir = [](fuse_req_t req, fuse_ino_t parent, const char *name, mode_t mode) {
		Guard(req, [&] {
			Mutable(Context(req));
			if ((mode & 07777) & ~0777) {
				fuse_reply_err(req, EOPNOTSUPP);
				return;
			}
			ReplyEntry(req, Context(req).session->CreateNode(parent, name, true, mode & 0777));
		});
	};
	ops.unlink = [](fuse_req_t req, fuse_ino_t parent, const char *name) {
		Guard(req, [&] {
			Mutable(Context(req));
			Context(req).session->RemoveNode(parent, name, false);
			fuse_reply_err(req, 0);
		});
	};
	ops.rmdir = [](fuse_req_t req, fuse_ino_t parent, const char *name) {
		Guard(req, [&] {
			Mutable(Context(req));
			Context(req).session->RemoveNode(parent, name, true);
			fuse_reply_err(req, 0);
		});
	};
	ops.rename = [](fuse_req_t req, fuse_ino_t parent, const char *name, fuse_ino_t next, const char *target,
	                unsigned flags) {
		Guard(req, [&] {
			Mutable(Context(req));
			if (flags & ~1U) {
				fuse_reply_err(req, EINVAL);
				return;
			}
			Context(req).session->RenameNode(parent, name, next, target, flags & 1U);
			fuse_reply_err(req, 0);
		});
	};
	ops.setattr = [](fuse_req_t req, fuse_ino_t inode, struct stat *attributes, int flags, fuse_file_info *) {
		Guard(req, [&] {
			auto &mount = Context(req);
			Mutable(mount);
			if (((flags & FUSE_SET_ATTR_UID) && attributes->st_uid != mount.uid) ||
			    ((flags & FUSE_SET_ATTR_GID) && attributes->st_gid != mount.gid)) {
				fuse_reply_err(req, EPERM);
				return;
			}
			std::optional<int64_t> mode, size, modified;
			if (flags & FUSE_SET_ATTR_MODE) {
				if ((attributes->st_mode & 07777) & ~0777) {
					fuse_reply_err(req, EOPNOTSUPP);
					return;
				}
				mode = attributes->st_mode & 0777;
			}
			if (flags & FUSE_SET_ATTR_SIZE)
				size = attributes->st_size;
			if (flags & FUSE_SET_ATTR_MTIME_NOW)
				modified = std::chrono::duration_cast<std::chrono::nanoseconds>(
				               std::chrono::system_clock::now().time_since_epoch())
				               .count();
			else if (flags & FUSE_SET_ATTR_MTIME) {
				auto stamp = attributes->st_mtim;
				if (stamp.tv_sec < -9223372035LL || stamp.tv_sec > 9223372035LL || stamp.tv_nsec < 0 ||
				    stamp.tv_nsec >= 1000000000) {
					fuse_reply_err(req, EINVAL);
					return;
				}
				modified = int64_t(stamp.tv_sec) * 1000000000 + stamp.tv_nsec;
			}
			mount.session->SetAttributes("", mode, modified, inode, size);
			auto stat = Describe(mount, mount.session->StatInode(inode));
			fuse_reply_attr(req, &stat, METADATA_TIMEOUT);
		});
	};
	// synchronous=FULL commits finish before write/setattr replies.
	ops.fsync = [](fuse_req_t req, fuse_ino_t inode, int, fuse_file_info *) {
		Guard(req, [&] {
			Context(req).session->StatInode(inode);
			fuse_reply_err(req, 0);
		});
	};
	ops.fsyncdir = ops.fsync;
	ops.flush = [](fuse_req_t req, fuse_ino_t inode, fuse_file_info *) {
		Guard(req, [&] {
			Context(req).session->StatInode(inode);
			fuse_reply_err(req, 0);
		});
	};
	ops.statfs = [](fuse_req_t req, fuse_ino_t) {
		Guard(req, [&] {
			auto &mount = Context(req);
			struct statvfs result {};
			if (::statvfs(mount.database.c_str(), &result) != 0) {
				fuse_reply_err(req, errno);
				return;
			}
			result.f_namemax = 255;
			if (mount.readonly)
				result.f_flag |= ST_RDONLY;
			fuse_reply_statfs(req, &result);
		});
	};
	return ops;
}
struct FuseSession {
	fuse_args args = FUSE_ARGS_INIT(0, nullptr);
	fuse_cmdline_opts options {};
	fuse_session *session = nullptr;
	bool mounted = false, signals = false;
	~FuseSession() {
		if (mounted)
			fuse_session_unmount(session);
		if (signals)
			fuse_remove_signal_handlers(session);
		if (session)
			fuse_session_destroy(session);
		std::free(options.mountpoint);
		fuse_opt_free_args(&args);
	}
};
} // namespace

int main(int argc, char **argv) {
	if ((argc != 5 && !(argc == 6 && std::string(argv[5]) == "--debug")) ||
	    (std::string(argv[2]) != "--branch" && std::string(argv[2]) != "--snapshot")) {
		std::cerr << "Usage: vane-fs-mount DATABASE (--branch NAME_OR_ID | --snapshot ID) EMPTY_MOUNTPOINT "
		             "[--debug]\nRuns in the foreground. Unmount with fusermount3 -u MOUNTPOINT.\n";
		return argc == 2 && std::string(argv[1]) == "--help" ? 0 : 2;
	}
	try {
		auto mountpoint = std::filesystem::canonical(argv[4]), database = std::filesystem::weakly_canonical(argv[1]);
		auto relative = database.lexically_relative(mountpoint);
		if (relative.empty() || *relative.begin() != "..")
			throw std::runtime_error("Database must be outside the mountpoint");
		if (!std::filesystem::is_directory(mountpoint) || !std::filesystem::is_empty(mountpoint))
			throw std::runtime_error("Mountpoint must be an empty directory");
		bool readonly = std::string(argv[2]) == "--snapshot";
		Mount mount(database.string(), argv[3], readonly);
		FuseSession fuse;
		std::vector<std::string> arguments {
		    argv[0], "-f", "-o", readonly ? "ro,default_permissions,nodev,nosuid" : "default_permissions,nodev,nosuid",
		    mountpoint.string()};
		if (argc == 6)
			arguments.push_back("-d");
		for (const auto &argument : arguments)
			if (fuse_opt_add_arg(&fuse.args, argument.c_str()) != 0)
				throw std::bad_alloc();
		if (fuse_parse_cmdline(&fuse.args, &fuse.options) != 0)
			throw std::runtime_error("Invalid FUSE arguments");
		auto operations = Operations();
		fuse.session = fuse_session_new(&fuse.args, &operations, sizeof(operations), &mount);
		if (!fuse.session)
			throw std::runtime_error("Could not create FUSE session");
		mount.fuse = fuse.session;
		if (fuse_set_signal_handlers(fuse.session) != 0)
			throw std::runtime_error("Could not install FUSE signal handlers");
		fuse.signals = true;
		if (fuse_session_mount(fuse.session, fuse.options.mountpoint) != 0)
			throw std::runtime_error("Could not mount VaneFS");
		fuse.mounted = true;
		return fuse_session_loop(fuse.session) == 0 ? 0 : 1;
	} catch (const std::exception &error) {
		std::cerr << "VaneFS: " << error.what() << '\n';
		return 1;
	}
}

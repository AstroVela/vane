// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Appended to the diagnostic timeline after its definitions. This wrapper
// records the original call and preserves its result and errno.
extern "C" int __real_sync_file_range(int, off64_t, off64_t, unsigned);
extern "C" int __wrap_sync_file_range(int fd, off64_t offset, off64_t bytes, unsigned flags) {
	struct stat state {};
	int saved = errno;
	bool have = fstat(fd, &state) == 0;
	errno = saved;
	auto before = timeline::ReadDisk();
	auto start = timeline::Now();
	int rc = __real_sync_file_range(fd, offset, bytes, flags);
	timeline::Record("sync_file_range_write", start, rc, fd, -1, -1, offset, true, have ? state.st_size : -1,
	                 have ? state.st_blocks * 512 : -1, &before, bytes, flags);
	return rc;
}

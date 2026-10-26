// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// Explicit C data interface between independently built native modules. No
// DuckDB, Python, STL objects, exceptions, or allocator ownership cross it.
// The importing module must retain the capsule until every handle is closed.
#define VANE_FS_READER_CAPSULE   "vane_fs.snapshot_reader.v1"
#define VANE_FS_READER_VERSION   1
#define VANE_FS_READER_OK        0
#define VANE_FS_READER_NOT_FOUND 1
#define VANE_FS_READER_INVALID   2
#define VANE_FS_READER_IO        3

typedef struct VaneFSReaderError {
	char message[1024];
} VaneFSReaderError;

typedef struct VaneFSReaderStat {
	int64_t inode;
	int64_t size;
	int64_t mtime_ns;
	int32_t directory;
} VaneFSReaderStat;

typedef struct VaneFSReaderV1 {
	uint32_t version;
	uint32_t struct_size;
	void *context;
	const char *workspace_id;
	// Open and retain one immutable snapshot inode. Directories can be opened
	// for metadata/pinning; reading their bytes fails. Failure leaves *handle NULL.
	int32_t (*open)(void *context, const char *snapshot, const char *path, void **handle, VaneFSReaderStat *stat,
	                VaneFSReaderError *error);
	// Positional reads do not mutate an implicit cursor. EOF can return fewer
	// bytes. Calls on distinct or identical handles may run concurrently.
	int32_t (*read)(void *handle, uint64_t offset, uint64_t size, void *buffer, uint64_t *read_size,
	                VaneFSReaderError *error);
	// Exactly once per opened handle, after its reads have quiesced. Never throws.
	void (*close)(void *handle);
} VaneFSReaderV1;

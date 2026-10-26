// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>
#include <sqlite3.h>

using namespace vane_fs;
namespace fs = std::filesystem;
constexpr int64_t B = 4096;
static void Check(bool value, const char *message) {
	if (!value)
		throw std::runtime_error(message);
}
template <class F>
static void ErrorIs(ErrorCode code, F call) {
	try {
		call();
	} catch (const Error &e) {
		Check(e.code == code, "Unexpected error code");
		return;
	}
	throw std::runtime_error("Expected storage/API failure did not occur");
}
static int64_t SQL(const std::string &path, const std::string &text, bool failure = false) {
	sqlite3 *db = nullptr;
	Check(sqlite3_open(path.c_str(), &db) == SQLITE_OK, "Opening inspection database");
	sqlite3_busy_timeout(db, 5000);
	sqlite3_wal_autocheckpoint(db, 0);
	sqlite3_db_config(db, SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, 1, nullptr);
	int64_t value = 0;
	auto callback = [](void *context, int n, char **values, char **) {
		if (n && values[0])
			*static_cast<int64_t *>(context) = std::stoll(values[0]);
		return 0;
	};
	auto rc = sqlite3_exec(db, text.c_str(), callback, &value, nullptr);
	sqlite3_close(db);
	Check(failure ? rc != SQLITE_OK : rc == SQLITE_OK, "Inspection/fault SQL result differs");
	return value;
}
static std::string Data(int blocks) {
	std::string data(size_t(blocks) * B, 0);
	for (size_t i = 0; i < data.size(); ++i)
		data[i] = char((i * 17 + (i / B) * 29) % 251 + 1);
	return data;
}
static void Shapes(const std::string &path, Durability mode) {
	Workspace w(path, 5000, mode);
	auto main = w.Checkout();
	auto original = Data(128), expected = original;
	main->WriteFile("/file", original);
	Check(SQL(path, "SELECT count(*) FROM block_versions") == 2, "Contiguous writes did not create two extents");
	Check(SQL(path, "SELECT count(*) FROM block_payloads") == 2, "Payload rows were not shared");
	Check(SQL(path, "SELECT sum(count) FROM block_versions") == 128, "Extent coverage differs");
	auto snapshot = w.Snapshot();
	auto frozen = w.OpenSnapshot(snapshot);
	auto patch = Data(2).substr(0, B + 96);
	main->Write("/file", patch, 63 * B - 17);
	expected.replace(63 * B - 17, patch.size(), patch);
	patch.assign(patch.size(), '!'); // Input lifetime must end before the barrier.
	w.Sync();
	Check(main->Read("/file") == expected, "Cross-extent edge overwrite differs");
	Check(main->Read("/file", 62 * B + 3, 4 * B) == expected.substr(62 * B + 3, 4 * B), "Range clipping differs");
	w.CollectGarbage();
	Check(frozen->Read("/file") == original, "Split changed the snapshot");
	frozen->Close();
	w.DropSnapshot(snapshot);
	w.CollectGarbage();
	Check(main->Read("/file") == expected, "GC lost a split range");
	w.Close();
	Workspace reopened(path);
	Check(reopened.Checkout()->Read("/file") == expected, "Reopen changed extent bytes");
}
static void Coordinates(const std::string &path, Durability mode) {
	Workspace w(path, 5000, mode);
	auto main = w.Checkout();
	auto base = Data(128), parent_data = base, child_data = base, grand_data = base;
	main->WriteFile("/file", base);
	auto before = w.Snapshot();
	auto frozen = w.OpenSnapshot(before);
	auto child_id = w.Fork("main", "child").id;
	auto child = w.Checkout(child_id);
	auto grand_id = w.Fork(child_id, "grand").id;
	auto grand = w.Checkout(grand_id);
	main->Write("/file", "MAIN", 11 * B + 1);
	parent_data.replace(11 * B + 1, 4, "MAIN");
	child->Write("/file", "CHILD", 17 * B + 7);
	child_data.replace(17 * B + 7, 5, "CHILD");
	grand->Write("/file", "GRAND", 75 * B + 13);
	grand_data.replace(75 * B + 13, 5, "GRAND");
	w.CollectGarbage();
	Check(main->Read("/file") == parent_data, "Parent coordinate rectangle changed");
	Check(child->Read("/file") == child_data, "Child coordinate rectangle changed");
	Check(grand->Read("/file") == grand_data, "Grandchild coordinate rectangle changed");
	Check(frozen->Read("/file") == base, "Historical coordinate rectangle changed");
	auto preview = w.PreviewMerge(grand_id, child_id);
	Check(preview.conflicts == std::vector<std::string> {"/file"}, "Extent merge failed to report a conflict");
	w.Merge(preview, {{"/file", "source"}});
	Check(child->Read("/file") == grand_data, "Merge lost a shared-payload offset");
	grand->Close();
	w.DeleteBranch(grand_id);
	w.Merge(w.PreviewMerge(child_id, "main"), {{"/file", "source"}});
	child->Close();
	w.DeleteBranch(child_id);
	w.CollectGarbage();
	Check(main->Read("/file") == grand_data, "Second merge changed extent offsets");
	auto after = w.Snapshot();
	auto diff = w.Diff(before, after);
	Check(diff.size() == 1 && diff[0].path == "/file" && diff[0].kind == "modified", "Extent diff differs");
	main->Truncate("/file", 67 * B + 5);
	main->Truncate("/file", 128 * B);
	auto truncated = grand_data.substr(0, 67 * B + 5);
	truncated.resize(128 * B, 0);
	Check(main->Read("/file") == truncated, "Truncate resurrected a suffix");
	Check(frozen->Read("/file") == base, "Truncate changed an older snapshot");
	frozen->Close();
}
static void Holes(const std::string &path, Durability mode) {
	Workspace w(path, 5000, mode);
	auto main = w.Checkout();
	auto original = Data(128);
	for (int i = 0; i < 128; i += 2)
		original.replace(i * B, B, size_t(B), '\0');
	main->WriteFile("/file", original);
	Check(SQL(path, "SELECT count(*) FROM block_payloads") == 1, "Scattered physical batch duplicated payloads");
	Check(SQL(path, "SELECT count(*) FROM block_versions") == 64, "Logical holes were joined together");
	Check(main->Read("/file") == original, "Scattered shared payload reads differ");
	auto snap = w.Snapshot();
	auto frozen = w.OpenSnapshot(snap);
	auto expected = original;
	main->Write("/file", "offset", 99 * B + 31);
	expected.replace(99 * B + 31, 6, "offset");
	main->Write("/file", std::string(B, 0), 3 * B);
	expected.replace(3 * B, B, size_t(B), '\0');
	w.CollectGarbage();
	Check(main->Read("/file") == expected, "Sparse overwrite read the wrong shared block");
	Check(frozen->Read("/file") == original, "Sparse overwrite changed snapshot data");
	frozen->Close();
	w.DropSnapshot(snap);
	w.CollectGarbage();
	Check(main->Read("/file") == expected, "Sparse GC changed live bytes");
}
static void Fragments(const std::string &path, Durability mode) {
	Workspace w(path, 5000, mode);
	auto main = w.Checkout();
	auto original = Data(128), expected = original;
	main->WriteFile("/file", original);
	auto snap = w.Snapshot();
	auto frozen = w.OpenSnapshot(snap);
	for (int i = 0; i < 128; ++i) {
		if (i == 7 || i == 111)
			continue;
		main->Write("/file", std::string(B, 0), i * B);
		expected.replace(i * B, B, size_t(B), '\0');
	}
	w.CollectGarbage();
	Check(frozen->Read("/file") == original, "GC punched snapshot-owned extent fragments");
	frozen->Close();
	w.DropSnapshot(snap);
	w.CollectGarbage();
	struct stat st {};
	Check(!stat((path + ".payload").c_str(), &st) && st.st_blocks * 512 <= 65536,
	      "GC retained unreferenced portions of shared payloads");
	Check(main->Read("/file") == expected, "GC lost the two retained physical slices");
	for (auto block : {7, 111})
		main->Write("/file", std::string(B, 0), block * B);
	w.CollectGarbage();
	Check(fs::file_size(path + ".payload") == B, "No live slices should retain the payload tail");
	main->WriteFile("/new", Data(64));
	w.Sync();
	Check(main->Read("/file") == std::string(128 * B, 0), "Reused allocation resurrected tombstoned bytes");
	Check(main->Read("/new") == Data(64), "Append after physical truncation differs");
}
static void Rollback(const std::string &path, Durability mode) {
	Workspace w(path, 5000, mode);
	auto main = w.Checkout();
	auto original = Data(128);
	main->WriteFile("/file", original);
	w.Sync();
	auto generation = w.GetBranch().generation;
	SQL(path, "CREATE TRIGGER fail_split BEFORE INSERT ON block_versions WHEN new.block=0 AND new.count=17 "
	          "BEGIN SELECT RAISE(ABORT,'split failure'); END");
	ErrorIs(ErrorCode::Storage, [&] { main->Write("/file", std::string(B, 'z'), 17 * B); });
	SQL(path, "DROP TRIGGER fail_split");
	Check(main->Read("/file") == original && w.GetBranch().generation == generation,
	      "Failed rectangle split was visible");
	Check(SQL(path, "SELECT count(*) FROM block_payloads") == 2, "Failed split leaked a SQL payload");
	SQL(path, "CREATE TRIGGER fail_extent BEFORE INSERT ON block_versions WHEN new.count=64 "
	          "BEGIN SELECT RAISE(ABORT,'extent failure'); END");
	auto size = fs::file_size(path + ".payload");
	ErrorIs(ErrorCode::Storage, [&] { main->WriteFile("/failed", Data(64)); });
	Check(fs::file_size(path + ".payload") >= size + 64 * B, "Extent fault did not follow physical append");
	ErrorIs(ErrorCode::NotFound, [&] { main->Stat("/failed"); });
	SQL(path, "DROP TRIGGER fail_extent");
	main->WriteFile("/retry", Data(64));
	w.Sync();
	w.CollectGarbage();
	Check(main->Read("/file") == original && main->Read("/retry") == Data(64), "Extent rollback poisoned retry or GC");
}
static void Corruption(const std::string &path, Durability mode) {
	Workspace w(path, 5000, mode);
	auto main = w.Checkout();
	auto original = Data(64);
	main->WriteFile("/file", original);
	w.Sync();
	Check(SQL(path, "SELECT version FROM format") == 1005, "Extent format is not isolated");
	SQL(path, "UPDATE block_versions SET payload_offset=4096", true);
	SQL(path, "PRAGMA ignore_check_constraints=ON; UPDATE block_versions SET payload_offset=1");
	ErrorIs(ErrorCode::Storage, [&] { main->Read("/file"); });
	ErrorIs(ErrorCode::Storage, [&] { main->Write("/file", "x", 1); });
	SQL(path, "UPDATE block_versions SET payload_offset=0");
	SQL(path, "INSERT INTO block_versions(inode,block,low,high,writer,deleted,payload,count,payload_offset) "
	          "SELECT inode,block+1,low,high,writer,deleted,payload,1,4096 FROM block_versions WHERE block=0");
	ErrorIs(ErrorCode::Storage, [&] { main->Read("/file"); });
	ErrorIs(ErrorCode::Storage, [&] { main->Write("/file", "x", B); });
	SQL(path, "DELETE FROM block_versions WHERE block=1");
	Check(main->Read("/file") == original, "Restored corruption fixture changed bytes");
}
int main(int argc, char **argv) {
	if (argc != 2)
		return 2;
	try {
		for (auto mode : {Durability::Strict, Durability::Fsync}) {
			int index = 0;
			for (auto test : {Shapes, Coordinates, Holes, Fragments, Rollback, Corruption}) {
				auto path = fs::path(argv[1]) / (std::to_string(int(mode)) + "-" + std::to_string(index) + ".sqlite");
				test(path.string(), mode);
				std::cout << "PASS " << int(mode) << ' ' << index++ << '\n';
			}
		}
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return 1;
	}
	return 0;
}

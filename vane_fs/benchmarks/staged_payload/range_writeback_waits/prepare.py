# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build matched range-writeback diagnostics without changing either algorithm."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "prewriteback_waits/prepare.py"
scope = dict(__name__="range_parent", __file__=__file__)
code = PARENT.read_text()
changes = [
    ('"original_plain_failed_checks=4,"', '"original_plain_failed_checks=7,"'),
    ('stats["roots"] == 3', 'stats["roots"] == 4'),
    (
        '                    PARENT,\\n                    HERE / "analyze.py",',
        '                    PARENT,\\n                    HERE.parent / "prewriteback_waits/prepare.py",\\n                    HERE / "analyze.py",',
    ),
]
for before, after in changes:
    assert code.count(before) == 1, before
    code = code.replace(before, after)
exec(compile(code, str(PARENT), "exec"), scope)
TRACE, digest, save, validate = [scope[key] for key in ("TRACE", "digest", "save", "validate")]
original_instrument, original_recorder = scope["instrument"], scope["recorder"]
original_validate = validate
scope["namespace"]["helpers"].WRAPS.append("sync_file_range")


def instrument(source, name):
    text, pairs = original_instrument(source, name)
    if name == "src/fuse.cpp":
        # Identify foreground callbacks in each small-file operation. The init
        # callback has no request context; read/write/fsync already record it.
        for operation in (
            "lookup",
            "forget",
            "forget_multi",
            "getattr",
            "open",
            "create",
            "release",
            "opendir",
            "readdir",
            "releasedir",
            "mkdir",
            "unlink",
            "rmdir",
            "rename",
            "setattr",
            "flush",
            "statfs",
        ):
            before = f'fuse_waits::Root wait_root("{operation}");'
            after = f'fuse_waits::Root wait_root("{operation}", -1, -1, -1, fuse_req_ctx(req)->pid);'
            assert text.count(before) == 1, operation
            text = text.replace(before, after)
            pairs.append((before, after, 1))
    reverse = text
    for before, after, count in reversed(pairs):
        assert reverse.count(after) == count
        reverse = reverse.replace(after, before)
    assert reverse == source
    return text, pairs


def recorder(file):
    text, pairs = original_recorder(file)
    if file.endswith(".hpp"):
        extra = [("\tReadBytes,\n\tCount", "\tReadBytes,\n\tRangeWriteback,\n\tCount")]
    else:
        extra = [
            ('"read_bytes"};', '"read_bytes", "range_writeback"};'),
            (
                "stage == Fsync || stage == Fdatasync ||",
                "stage == Fsync || stage == Fdatasync || stage == RangeWriteback ||",
            ),
            (
                'extern "C" int __real_fsync(int);',
                """extern "C" int __real_sync_file_range(int, off64_t, off64_t, unsigned int);
extern "C" int __wrap_sync_file_range(int fd, off64_t offset, off64_t bytes, unsigned int flags) {
\treturn Call(RangeWriteback, "sync_file_range", fd, offset, bytes, int(flags),
\t            [&] { return __real_sync_file_range(fd, offset, bytes, flags); });
}
extern "C" int __real_fsync(int);""",
            ),
        ]
    for before, after in extra:
        assert text.count(before) == 1, (file, before)
        text = text.replace(before, after)
        pairs.append((before, after))
    return text, pairs


def validate(trace):
    result = original_validate(trace)
    calls = [event for thread in trace["threads"] for event in thread["events"] if event[0] == "sync_file_range"]
    for event in calls:
        assert event[9] >= 4096 and event[10] > 0 and event[11] == 2  # WRITE only
    if any(root[0] == "range_test" for thread in trace["threads"] for root in thread["roots"]):
        assert len(calls) == 2
        assert calls[0][1] == "payload" and calls[0][6:8] == [0, 0]
        assert calls[0][9:12] == [4096, 1024 * 1024, 2]
        assert calls[1][6:8] == [-1, 9]  # EBADF survives recorder calls.
    else:
        assert all(event[1] == "payload" and event[6:8] == [0, 0] for event in calls)
    return dict(result, range_calls=len(calls))


scope["namespace"].update(instrument=instrument, recorder=recorder, validate=validate)

if __name__ == "__main__":
    scope["namespace"]["main"]()

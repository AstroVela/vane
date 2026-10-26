# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build reversible prewriteback and sequential-read diagnostics from frozen sources."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "payload_waits/prepare.py"
namespace = {"__name__": "prewriteback_parent", "__file__": __file__}
code = PARENT.read_text()
before, after = "original_plain_failed_checks=11,", "original_plain_failed_checks=4,"
assert code.count(before) == 1
code = code.replace(before, after)
before = 'inputs = dict(prior["inputs"], **component["binaries"], **component["inputs"])'
after = 'inputs = prior["inputs"] | component["binaries"] | component["inputs"] | source["default_unchanged"]'
assert code.count(before) == 1
code = code.replace(before, after)
before = '                    HERE / "test_hold.cpp",'
after = (
    before
    + '\n                    PARENT,\n                    HERE / "analyze.py",\n                    HERE / "publish.py",'
)
assert code.count(before) == 1
code = code.replace(before, after)
before = '                "src/fuse.cpp",\n            ):'
after = '                "src/fuse.cpp",\n                *(["src/prewriteback.hpp"] if (core / "src/prewriteback.hpp").exists() else []),\n            ):'
assert code.count(before) == 1
code = code.replace(before, after)
before = 'assert stats["roots"] == stats["publication_holds"] == stats["checkpoint_holds"] == 2'
after = 'assert stats["roots"] == 3 and stats["publication_holds"] == stats["checkpoint_holds"] == 2'
assert code.count(before) == 1
code = code.replace(before, after)
namespace["PARENT"] = PARENT
exec(compile(code, str(PARENT), "exec"), namespace)
TRACE, digest, save, validate = [namespace[key] for key in ("TRACE", "digest", "save", "validate")]
original_instrument, original_recorder = namespace["instrument"], namespace["recorder"]


def instrument(source, name):
    text, pairs = original_instrument(source, name)

    def replace(before, after):
        nonlocal text
        assert text.count(before) == 1, (name, before)
        text = text.replace(before, after)
        pairs.append((before, after, 1))

    if name == "src/workspace.cpp":
        before = "std::string ReadBytes(sqlite3 *db, int64_t inode, int64_t offset, int64_t length, const View &view) {"
        replace(before, before + "\n\tfuse_waits::Span read_span(fuse_waits::ReadBytes);")
    if name == "src/prewriteback.hpp":
        replace('#include "payload_file.hpp"', '#include "payload_file.hpp"\n#include "wait_trace.hpp"')
        replace(
            "\t\t\tlock.unlock();",
            '\t\t\tlock.unlock();\n\t\t\tfuse_waits::Root wait_root("prewriteback_worker");',
        )
        replace(
            "\t\t\t\tPayloadFile::Guard lease(*file, false);",
            "\t\t\t\tfuse_waits::Span lease_span(fuse_waits::PayloadGuard);\n"
            "\t\t\t\tPayloadFile::Guard lease(*file, false);\n\t\t\t\tlease_span.Finish();",
        )
    reverse = text
    for before, after, count in reversed(pairs):
        assert reverse.count(after) == count
        reverse = reverse.replace(after, before)
    assert reverse == source
    return text, pairs


def recorder(file):
    text, pairs = original_recorder(file)
    before, after = (
        ("\tCheckpointHold,\n\tCount", "\tCheckpointHold,\n\tReadBytes,\n\tCount")
        if file.endswith(".hpp")
        else ('"checkpoint_hold"};', '"checkpoint_hold", "read_bytes"};')
    )
    assert text.count(before) == 1
    text = text.replace(before, after)
    pairs.append((before, after))
    return text, pairs


namespace.update(instrument=instrument, recorder=recorder)

if __name__ == "__main__":
    namespace["main"]()

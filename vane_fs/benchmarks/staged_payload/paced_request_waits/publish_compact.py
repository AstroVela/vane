# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Keep the frozen publisher while using compact, value-identical JSON indexes."""

import json
import sys
from pathlib import Path

from prepare import digest, save


def compact(path, value):
    text = json.dumps(value, separators=(",", ":")) + "\n"
    assert json.loads(text) == value
    path.write_text(text)


if __name__ == "__main__":
    parent = Path(__file__).resolve().with_name("publish.py")
    out = Path(sys.argv[1]).resolve()
    save(
        out / "publication-format.json",
        dict(
            status="PASS",
            publisher=str(parent),
            publisher_sha256=digest(parent),
            adapter_sha256=digest(Path(__file__)),
            only_change="Compact JSON serialization with exact parsed-value equality; raw gzip bytes unchanged.",
            native_sources_binaries_and_measurements_unchanged=True,
        ),
    )
    namespace = dict(__name__="compact_publication", __file__=str(parent))
    exec(compile(parent.read_text(), str(parent), "exec"), namespace)
    namespace["save"] = compact
    namespace["main"]()

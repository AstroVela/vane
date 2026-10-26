# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reuse the lossless evidence publisher with compact, value-identical indexes."""

import json
from pathlib import Path


def compact(path, value):
    text = json.dumps(value, separators=(",", ":")) + "\n"
    assert json.loads(text) == value
    path.write_text(text)


if __name__ == "__main__":
    parent = Path(__file__).resolve().parent.parent / "paced_request_waits/publish.py"
    namespace = dict(__name__="sql_publication", __file__=__file__)
    exec(compile(parent.read_text(), str(parent), "exec"), namespace)
    namespace["save"] = compact
    namespace["main"]()

# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

from typing_extensions import assert_type

import vane
from vane.ai import embed

text = vane.col("text")
relation = cast(vane.Relation, None)

assert_type(embed(text, dimensions=4), vane.Expression)
assert_type(embed(text=text, dimensions=4), vane.Expression)
assert_type(embed(relation, text, dimensions=4), vane.Relation)
assert_type(embed(rel=relation, text=text, dimensions=4), vane.Relation)
assert_type(relation.embed(text, dimensions=4), vane.Relation)

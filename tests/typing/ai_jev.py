# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

from typing_extensions import assert_type

import vane
from vane.ai import JevOptions, jev

relation = cast(vane.Relation, None)
state = vane.col("text")
questions = {"billing": {"type": "noul", "instructions": "Is this about billing?"}}
options: JevOptions = {"max_concurrency_per_actor": 8, "timeout": 30.0}

assert_type(jev(state, questions=questions, **options), vane.Expression)
assert_type(jev(state=state, questions=questions), vane.Expression)
assert_type(jev(relation, state, questions=questions), vane.Relation)
assert_type(jev(rel=relation, state=state, questions=questions), vane.Relation)
assert_type(relation.jev(state, questions=questions, output_column="judgment"), vane.Relation)

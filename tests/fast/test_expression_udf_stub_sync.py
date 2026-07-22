# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path

import _vane_duckdb


def test_vane_map_batches_runtime_and_stub_signatures_stay_in_sync():
    runtime_doc = _vane_duckdb._VaneUDFMapBatchesExpression.__doc__ or ""
    runtime_parameters = ("gpus", "actor_number", "stateful", "*args")
    assert all(parameter in runtime_doc for parameter in runtime_parameters)
    assert [runtime_doc.index(parameter) for parameter in runtime_parameters] == sorted(
        runtime_doc.index(parameter) for parameter in runtime_parameters
    )

    stub_path = Path(__file__).parents[2] / "_vane_duckdb-stubs" / "__init__.pyi"
    module = ast.parse(stub_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_VaneUDFMapBatchesExpression"
    )
    parameters = [argument.arg for argument in function.args.args]

    assert parameters[-3:] == ["gpus", "actor_number", "stateful"]
    assert function.args.vararg is not None
    assert function.args.vararg.arg == "args"
    actor_number = function.args.args[parameters.index("actor_number")]
    assert ast.unparse(actor_number.annotation) == "int | None"

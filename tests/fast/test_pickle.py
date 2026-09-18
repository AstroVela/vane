# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle

import pyarrow as pa
import pytest

import vane
from vane import pickle as vane_pickle


@pytest.mark.parametrize("protocol", [4, 5])
def test_pickle_preserves_cloudpickle_functions_classes_and_buffer_protocol(protocol):
    class Model:
        def __call__(self, value):
            return value + 1

    buffers = []
    payload = vane_pickle.dumps(
        {
            "function": lambda value: value * 2,
            "model": Model,
            "buffer": pickle.PickleBuffer(bytearray(b"abc")) if protocol == 5 else bytearray(b"abc"),
        },
        protocol=protocol,
        buffer_callback=buffers.append if protocol == 5 else None,
    )
    restored = vane_pickle.loads(payload, buffers=buffers)
    assert restored["function"](3) == 6
    assert restored["model"]()(3) == 4
    assert bytes(restored["buffer"]) == b"abc"
    assert len(buffers) == (1 if protocol == 5 else 0)


def test_pickle_rebuilds_nested_adapters_without_losing_subclass_state():
    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class Model:
        def __init__(self, offset):
            self.offset = offset

        def __call__(self, value):
            return value + self.offset

    model = Model(2)
    adapter = model.actor_class(["value"])

    class CustomAdapter(adapter):
        label = "custom"

        def __call__(self, table):
            return pa.table({"custom": [100] * len(table)})

    payload = vane_pickle.dumps({"adapter": adapter, "custom": CustomAdapter})
    restored = vane_pickle.loads(payload)
    table = pa.table({"value": [3]})
    assert restored["adapter"]()(table).column(0).to_pylist() == [5]
    assert restored["custom"].label == "custom"
    assert restored["custom"]()(table).to_pydict() == {"custom": [100]}

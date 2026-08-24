# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Engine-registry dispatch tests for the native inference operator."""

from __future__ import annotations

import pytest

pa = pytest.importorskip("pyarrow")


def test_unknown_engine_reports_clear_error():
    import vane
    from vane.ai.providers.vllm import _build_native_vllm_options_argument

    envelope = _build_native_vllm_options_argument({})
    envelope["engine"] = "doesnotexist"
    con = vane.connect()
    try:
        con.register("vllm_input", pa.table({"prompt": ["hello"]}))
        with pytest.raises(Exception, match="no executor factory registered for inference engine"):
            con.execute(
                "SELECT vllm(prompt, 'recording-model', ?) AS generated FROM vllm_input",
                [envelope],
            ).fetchall()
    finally:
        con.close()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda envelope: envelope.pop("engine"),
        lambda envelope: envelope.update(engine=""),
        lambda envelope: envelope.update(engine=None),
    ],
    ids=["missing", "empty", "null"],
)
def test_malformed_engine_reports_clear_error(mutate):
    import vane
    from vane.ai.providers.vllm import _build_native_vllm_options_argument

    envelope = _build_native_vllm_options_argument({})
    mutate(envelope)
    con = vane.connect()
    try:
        con.register("vllm_input", pa.table({"prompt": ["hello"]}))
        with pytest.raises(Exception, match="engine"):
            con.execute(
                "SELECT vllm(prompt, 'recording-model', ?) AS generated FROM vllm_input",
                [envelope],
            ).fetchall()
    finally:
        con.close()

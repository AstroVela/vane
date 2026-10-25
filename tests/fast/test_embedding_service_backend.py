# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""The shared service delegates model ownership without importing an SDK."""

import asyncio
import sys

import numpy as np
import pytest

from tests.fast.test_native_multimodal import install_sdk
from vane.ai import _embedding_ipc as ipc
from vane.ai._embedding_content import EmbeddingContent, EmbeddingPart, EmbeddingPartSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai.providers._embedding_service_backend import create_service_backend


@pytest.mark.parametrize("family", ["sglang", "vllm"])
def test_native_backend_owns_one_original_engine(monkeypatch, family):
    for module in ("sglang", "vllm", "mlx", "mlx_vlm"):
        monkeypatch.setitem(sys.modules, module, None)
    backend = create_service_backend(family, "fixture", 2, {"gpus_per_actor": 0})
    assert backend.runtime is None
    assert backend.get_input_spec().supported_parts
    sdk = install_sdk(monkeypatch.setitem)
    row = EmbeddingContent((EmbeddingPart(EmbeddingPartSpec("text"), "document"),))

    async def run():
        await backend.start()
        np.testing.assert_array_equal(await backend.embed(row, ipc.RequestContext()), [3, 4])
        np.testing.assert_array_equal(await backend.embed(row, ipc.RequestContext()), [3, 4])
        with pytest.raises(RuntimeError, match="already started"):
            await backend.start()
        await backend.aclose()
        await backend.aclose()

    asyncio.run(run())
    assert len(sdk.engines) == 1 and sdk.engines[0].closed == 1
    assert len(sdk.calls) == 2


@pytest.mark.parametrize("family", ["http", "unknown", "module.Factory"])
def test_backend_factory_is_closed(family):
    with pytest.raises(EmbeddingConfigurationError, match="family"):
        create_service_backend(family, "fixture", 2, {})

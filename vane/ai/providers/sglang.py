# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""SGLang prompt provider for the native SGLang engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from vane.ai.options import normalize_prompt_options
from vane.ai.protocols import NativeInferencePlan
from vane.ai.provider import Provider

if TYPE_CHECKING:
    from vane.ai.typing import Options


@dataclass
class NativeSGLangPromptPlan(NativeInferencePlan):
    """Serializable configuration for native SGLang query planning."""

    provider_name: str = "sglang"
    model_name: str = "Qwen/Qwen3-1.7B"
    system_message: str | None = None
    on_error: str = "raise"
    return_format: dict[str, Any] | None = None
    sglang_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("SGLang prompt model must be a non-empty string")

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model_name

    def get_options(self) -> Options:
        return dict(self.sglang_options)

    def get_engine(self) -> str:
        return "sglang"

    def build_physical_vllm_options(self) -> dict[str, Any]:
        options = dict(self.sglang_options)
        options["on_error"] = "null" if self.on_error == "ignore" else "raise"
        return options


class SGLangProvider(Provider):
    """Provider backed by a local or remote SGLang engine."""

    DEFAULT_MODEL = "Qwen/Qwen3-1.7B"

    def __init__(self, name: str | None = None):
        self._name = name or "sglang"

    @property
    def name(self) -> str:
        return self._name

    def get_prompter(
        self,
        model: str | None = None,
        system_message: str | None = None,
        return_format: dict[str, Any] | None = None,
        return_raw_response: bool = False,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> NativeSGLangPromptPlan:
        if return_raw_response:
            raise ValueError("Provider 'sglang' does not support return_raw_response")
        prepared = normalize_prompt_options("sglang", options or {}, relation=False)
        model_name = model or self.DEFAULT_MODEL
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("SGLang prompt model must be a non-empty string")
        return NativeSGLangPromptPlan(
            provider_name=self._name,
            model_name=model_name,
            system_message=system_message,
            return_format=return_format,
            sglang_options=prepared,
        )

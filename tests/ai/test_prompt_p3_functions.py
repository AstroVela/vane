# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pytest

import vane
from vane.ai.protocols import PrompterDescriptor
from vane.ai.provider import Provider
from vane.ai.typing import UDFOptions

SCHEMA = {
    "title": "Answer",
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "score": {"type": "integer"},
    },
    "required": ["answer", "score"],
    "additionalProperties": False,
}


class ContractPrompter:
    def __init__(self, *, structured: bool, raw: bool, invalid: bool, extra: bool) -> None:
        self._structured = structured
        self._raw = raw
        self._invalid = invalid
        self._extra = extra

    async def prompt(self, messages: tuple[Any, ...]) -> Any:
        text = str(messages[0])
        if self._raw:
            if self._invalid:
                return "not-json"
            return json.dumps({"id": f"raw-{text}", "output": [{"text": text}]})
        if self._structured:
            if self._invalid:
                return '{"answer":1,"score":"wrong"}'
            result = {"answer": text, "score": len(text)}
            if self._extra:
                result["provider_extra"] = True
            return result
        return f"text:{text}"


@dataclass
class ContractPrompterDescriptor(PrompterDescriptor):
    structured: bool = False
    raw: bool = False
    invalid: bool = False
    extra: bool = False

    def get_provider(self) -> str:
        return "contract_mock"

    def get_model(self) -> str:
        return "invalid" if self.invalid else "contract-model"

    def get_options(self) -> dict[str, Any]:
        return {}

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(actor_number=1, num_gpus=0, batch_size=2, max_retries=0)

    def instantiate(self) -> ContractPrompter:
        return ContractPrompter(
            structured=self.structured,
            raw=self.raw,
            invalid=self.invalid,
            extra=self.extra,
        )


class ContractProvider(Provider):
    @property
    def name(self) -> str:
        return "contract_mock"

    def get_prompter(
        self,
        model: str | None = None,
        system_message: str | None = None,
        return_format: dict[str, Any] | None = None,
        return_raw_response: bool = False,
        **options: Any,
    ) -> PrompterDescriptor:
        return ContractPrompterDescriptor(
            structured=return_format is not None,
            raw=return_raw_response,
            invalid=model == "invalid",
            extra=model == "extra",
        )


def test_structured_output_has_native_struct_type_on_every_python_entry():
    relation = vane.connect().sql("select 'search'::VARCHAR as prompt")
    message = vane.col("prompt")
    provider = ContractProvider()

    results = [
        relation.select(vane.ai.prompt(message, provider=provider, return_format=SCHEMA).alias("response")),
        relation.select(
            vane.ai.prompt(messages=message, provider=provider, return_format=SCHEMA).alias("response")
        ),
        vane.ai.prompt(relation, message, provider=provider, return_format=SCHEMA),
        vane.ai.prompt(rel=relation, messages=message, provider=provider, return_format=SCHEMA),
        relation.prompt(message, provider=provider, return_format=SCHEMA),
    ]

    for result in results:
        assert str(result.types[-1]) == "STRUCT(answer VARCHAR, score BIGINT)"
        assert result.fetchall()[0][-1] == {"answer": "search", "score": 6}


def test_raw_response_wins_public_type_when_combined_with_schema():
    relation = vane.connect().sql("select 'search'::VARCHAR as prompt")
    result = vane.ai.prompt(
        relation,
        vane.col("prompt"),
        provider=ContractProvider(),
        return_format=SCHEMA,
        return_raw_response=True,
    )

    assert str(result.types[-1]) == "VARCHAR"
    raw = result.fetchall()[0][-1]
    assert json.loads(raw) == {"id": "raw-search", "output": [{"text": "search"}]}
    assert "headers" not in raw
    assert "vane" not in raw.casefold()


def test_invalid_structured_output_raises_or_becomes_null_without_fallback():
    relation = vane.connect().sql("select 'search'::VARCHAR as prompt")

    raised = vane.ai.prompt(
        relation,
        vane.col("prompt"),
        provider=ContractProvider(),
        model="invalid",
        return_format=SCHEMA,
    )
    with pytest.raises(Exception, match="must be a string"):
        raised.fetchall()

    ignored = vane.ai.prompt(
        relation,
        vane.col("prompt"),
        provider=ContractProvider(),
        model="invalid",
        return_format=SCHEMA,
        on_error="ignore",
    )
    assert ignored.fetchall()[0][-1] is None


def test_allowed_additional_properties_do_not_break_struct_conversion():
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "score": {"type": "integer"},
            "note": {"type": "string"},
        },
        "required": ["answer", "score"],
    }
    relation = vane.connect().sql("select 'search'::VARCHAR as prompt")

    result = vane.ai.prompt(
        relation,
        vane.col("prompt"),
        provider=ContractProvider(),
        model="extra",
        return_format=schema,
        on_error="ignore",
    )

    assert result.fetchall()[0][-1] == {"answer": "search", "score": 6, "note": None}


def test_invalid_raw_response_raises_or_becomes_null():
    relation = vane.connect().sql("select 'search'::VARCHAR as prompt")

    raised = vane.ai.prompt(
        relation,
        vane.col("prompt"),
        provider=ContractProvider(),
        model="invalid",
        return_raw_response=True,
    )
    with pytest.raises(Exception, match="not valid JSON"):
        raised.fetchall()

    ignored = vane.ai.prompt(
        relation,
        vane.col("prompt"),
        provider=ContractProvider(),
        model="invalid",
        return_raw_response=True,
        on_error="ignore",
    )
    assert ignored.fetchall()[0][-1] is None


def test_pydantic_return_format_matches_json_schema_runtime_result():
    pydantic = pytest.importorskip("pydantic")

    class Answer(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="forbid")

        answer: str
        score: int

    relation = vane.connect().sql("select 'search'::VARCHAR as prompt")
    provider = ContractProvider()

    pydantic_result = vane.ai.prompt(
        relation,
        vane.col("prompt"),
        provider=provider,
        return_format=Answer,
    )
    schema_result = vane.ai.prompt(
        relation,
        vane.col("prompt"),
        provider=provider,
        return_format=Answer.model_json_schema(),
    )

    assert pydantic_result.types[-1] == schema_result.types[-1]
    assert pydantic_result.fetchall() == schema_result.fetchall()


def test_prompt_batch_structured_arrow_boundary_is_json_varchar():
    from vane.ai._schema import compile_return_format
    from vane.ai.functions import _PromptBatch

    descriptor = ContractPrompterDescriptor(structured=True)
    spec = compile_return_format(SCHEMA)
    assert spec is not None
    wrapper = _PromptBatch(
        descriptor,
        ["message_0"],
        "response",
        return_format=spec,
        max_retries=0,
    )
    import asyncio

    loop = asyncio.new_event_loop()
    wrapper.bind_async_runtime(loop.run_until_complete)
    try:
        table = wrapper(pa.table({"message_0": ["search"]}))
    finally:
        wrapper.close()
        loop.close()

    assert table.schema == pa.schema([("response", pa.string())])
    assert json.loads(table.column("response")[0].as_py()) == {"answer": "search", "score": 6}

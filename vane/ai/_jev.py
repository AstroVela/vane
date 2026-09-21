# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Jev judgments over Vane expressions, using the async TypeSafe SDK."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal, overload

import pyarrow as pa
from typing_extensions import Unpack

import vane
from vane._expressions import as_expression
from vane._typing import Expression, Relation
from vane.ai._client_config import _require_key, _setting, copy_client_options
from vane.ai._redaction import unwrap_sensitive_options
from vane.ai.functions import (
    _build_ai_batch_expression,
    _log_substituted_failure,
    _missing_async_runtime,
    _star_excluding_existing_output_column,
    _validate_on_error,
)
from vane.ai.options import JevOptions, _validate_base_url_option
from vane.ai.provider import _safe_provider_execution_error, _translate_missing_provider_dependency
from vane.ai.typing import UDFOptions


def _prepare_questions(questions: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("Jev questions must be a non-empty mapping")
    if any(not isinstance(name, str) or not name for name in questions):
        raise ValueError("Jev question names must be non-empty strings")
    with _translate_missing_provider_dependency("typesafe", "typesafe_sdk"):
        from typesafe_sdk import Choice, Noul, Score

    question_types = {"choice": Choice, "noul": Noul, "score": Score}
    prepared = {}
    for name, question in questions.items():
        if isinstance(question, (Choice, Noul, Score)):
            question = question.model_dump(mode="json")
        if not isinstance(question, Mapping) or question.get("type") not in question_types:
            raise ValueError("Jev questions must be Choice, Noul, Score, or dictionaries with a supported 'type'")
        error = False
        try:
            parsed = question_types[question["type"]].model_validate(dict(question))
            # Copy nested data before capturing it in a distributed expression.
            prepared[name] = json.loads(json.dumps(parsed.model_dump(mode="json"), allow_nan=False))
        except (TypeError, ValueError):
            error = True
        if error:
            raise ValueError("Invalid Jev question; check the TypeSafe instructions and criteria schema") from None
    return prepared


def _prepare_options(
    options: Mapping[str, Any], on_error: Literal["raise", "ignore"]
) -> tuple[UDFOptions, dict[str, Any]]:
    unknown = set(options) - JevOptions.__annotations__.keys()
    if unknown:
        raise TypeError(f"Unsupported Jev option(s): {', '.join(sorted(unknown))}")
    _validate_on_error(on_error)
    values = {"batch_size": 32, "actor_number": 1, "max_concurrency_per_actor": 8, "max_retries": 3}
    for name, default in values.items():
        value = options.get(name, default)
        minimum = 0 if name == "max_retries" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Jev option {name!r} must be an integer >= {minimum}")
        values[name] = value
    backend = options.get("execution_backend")
    if backend is not None and backend not in {"subprocess_task", "subprocess_actor", "ray_task", "ray_actor"}:
        raise ValueError("Invalid Jev execution_backend")
    if backend in {"subprocess_task", "ray_task"} and "actor_number" in options:
        raise ValueError("Jev actor_number requires an actor execution backend")
    base_url = _setting(options.get("base_url"), "TYPESAFE_BASE_URL") or "https://api.typesafe.ai"
    _validate_base_url_option({"base_url": base_url}, api="Jev")
    timeout = options.get("timeout")
    if timeout is not None and (
        isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0
    ):
        raise ValueError("Jev timeout must be a finite positive number or None")
    client_options = {name: options[name] for name in ("base_url", "timeout") if options.get(name) is not None}
    client_options.update(api_key=_setting(None, "TYPESAFE_API_KEY"), base_url=base_url)
    return UDFOptions(**values, num_gpus=0, on_error=on_error), copy_client_options(client_options)


def _serialize_response(response: Any, questions: Mapping[str, Any]) -> str:
    """Check the response against the requested questions before exposing JSON."""
    payload = response.model_dump(mode="json")
    answers = payload["answers"]
    if set(answers) != set(questions):
        raise ValueError("Jev response must contain exactly the requested answers")
    for name, question in questions.items():
        answer = answers[name]
        kind = question["type"]
        if answer["type"] != kind:
            raise ValueError("Jev answer type does not match its question")
        probabilities = [answer["noul"]] if kind == "noul" else list(answer["probabilities"].values())
        if kind != "noul":
            probabilities.append(answer["confidence"])
        if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1 for p in probabilities):
            raise ValueError("Jev probabilities and confidence must be finite numbers between 0 and 1")
        if kind == "choice":
            labels = set(question["criteria"])
            if answer["choice"] not in labels or set(answer["probabilities"]) != labels:
                raise ValueError("Jev choice must use the requested criteria")
        if kind == "score":
            levels = {str(index): criterion for index, criterion in enumerate(question["criteria"])}
            if set(answer["probabilities"]) != levels.keys() or answer["legend"] != levels:
                raise ValueError("Jev score must use the requested levels and criteria")
            score = answer["score"]
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= len(levels) - 1:
                raise ValueError("Jev score must lie within the requested levels")
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def _decode_state(encoded: str | None) -> Any:
    if encoded is None:
        return None
    invalid = False
    try:
        state = json.loads(encoded)
        invalid = state is not None and not isinstance(state, (str, dict, list))
        # JSON columns can contain non-standard NaN/Infinity values. Reject
        # these before the SDK sees them, including inside structured state.
        json.dumps(state, allow_nan=False)
    except (TypeError, ValueError):
        invalid = True
    if invalid:
        raise ValueError("Jev state must be text, a JSON object, a JSON array, or NULL, with finite numbers") from None
    return state


class _JevBatch:
    """One loop-bound SDK client per executor; one System One request per row."""

    def __init__(self, questions: dict[str, Any], model: str, options: dict[str, Any], udf_options: UDFOptions) -> None:
        self._questions = questions
        self._model = model
        self._options = options
        self._udf_options = udf_options
        if udf_options.max_concurrency_per_actor is None:
            raise ValueError("Jev requires an explicit executor concurrency limit")
        self._max_concurrency = udf_options.max_concurrency_per_actor
        self._client: Any = None
        self._run_async: Callable[[Awaitable[Any]], Any] | None = None

    def bind_async_runtime(self, run_async: Callable[[Awaitable[Any]], Any]) -> None:
        self._run_async = run_async

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_client": None, "_run_async": None}

    def close(self) -> None:
        client, self._client = self._client, None
        if client is None or self._run_async is None:
            return
        error = None
        try:
            self._run_async(client.aclose())
        except Exception as exc:
            error = _safe_provider_execution_error("typesafe", self._model, "Jev cleanup", exc)
        if error is not None:
            raise error from None

    def __call__(self, table: pa.Table) -> pa.Table:
        states = []
        for encoded in table.column("state").to_pylist():
            try:
                states.append(_decode_state(encoded))
            except ValueError as exc:
                if self._udf_options.on_error == "raise":
                    raise
                _log_substituted_failure(exc, on_error="ignore")
                states.append(None)
        results: list[str | None] = [None] * len(states)
        if not any(state is not None for state in states):
            return pa.table({"response": pa.array(results, type=pa.string())})
        if self._run_async is None:
            raise _missing_async_runtime()

        async def run_all() -> list[str | None]:
            if self._client is None:
                error = None
                try:
                    with _translate_missing_provider_dependency("typesafe", "typesafe_sdk"):
                        from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

                    client_options = unwrap_sensitive_options(self._options)
                    _require_key(client_options, "TypeSafe")
                    self._client = AsyncTypeSafeClient(
                        model=self._model,
                        retry=RetryPolicy(max_retries=self._udf_options.max_retries),
                        **client_options,
                    )
                except Exception as exc:
                    error = _safe_provider_execution_error("typesafe", self._model, "Jev initialization", exc)
                if error is not None:
                    raise error from None
            semaphore = asyncio.Semaphore(self._max_concurrency)

            async def invoke(state: Any) -> str | None:
                if state is None:
                    return None
                async with semaphore:
                    error = None
                    try:
                        response = await self._client.system_one(state=state, questions=self._questions)
                        return _serialize_response(response, self._questions)
                    except Exception as exc:
                        if self._udf_options.on_error == "ignore":
                            _log_substituted_failure(exc, on_error="ignore")
                            return None
                        error = _safe_provider_execution_error("typesafe", self._model, "Jev execution", exc)
                    raise error from None

            tasks = [asyncio.create_task(invoke(state)) for state in states]
            try:
                return await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        results = self._run_async(run_all())
        return pa.table({"response": pa.array(results, type=pa.string())})


_UNSET: Any = object()


@overload
def jev(
    state: Expression,
    /,
    *,
    questions: Mapping[str, Any],
    model: str = "jev-latest",
    on_error: Literal["raise", "ignore"] = "raise",
    **options: Unpack[JevOptions],
) -> Expression: ...


@overload
def jev(
    *,
    state: Expression,
    questions: Mapping[str, Any],
    model: str = "jev-latest",
    on_error: Literal["raise", "ignore"] = "raise",
    **options: Unpack[JevOptions],
) -> Expression: ...


@overload
def jev(
    rel: Relation,
    /,
    state: Expression,
    *,
    questions: Mapping[str, Any],
    model: str = "jev-latest",
    on_error: Literal["raise", "ignore"] = "raise",
    output_column: str = "response",
    **options: Unpack[JevOptions],
) -> Relation: ...


@overload
def jev(
    *,
    rel: Relation,
    state: Expression,
    questions: Mapping[str, Any],
    model: str = "jev-latest",
    on_error: Literal["raise", "ignore"] = "raise",
    output_column: str = "response",
    **options: Unpack[JevOptions],
) -> Relation: ...


def jev(
    first: Any = _UNSET,
    /,
    state: Any = _UNSET,
    *,
    rel: Any = _UNSET,
    questions: Mapping[str, Any],
    model: str = "jev-latest",
    on_error: Literal["raise", "ignore"] = "raise",
    output_column: Any = _UNSET,
    **options: Unpack[JevOptions],
) -> Expression | Relation:
    """Evaluate text or structured state with Jev and return JSON as VARCHAR.

    ``questions`` accepts TypeSafe ``Choice``, ``Noul``, ``Score`` objects or
    their dictionary forms. Each non-NULL row makes one request containing all
    its questions. Results retain ``answers``, ``model``, and ``usage``; NULL
    input produces NULL output. ``on_error='ignore'`` substitutes NULL for a
    failed row after SDK retries. Initialization errors always propagate.

    Install ``vane-ai[typesafe]`` and configure ``TYPESAFE_API_KEY`` before
    building the expression. Credentials and the endpoint are captured on the
    application, so worker environment changes cannot redirect requests.
    Clients are constructed only during execution, reused
    across actor batches, and closed on the executor's event loop. Concurrency
    is bounded per executor, not globally across queries or workers.

    Examples::

        from typesafe_sdk import Noul
        from vane.ai import jev

        result = source.select(
            jev(
                vane.col("text"),
                questions={"billing": Noul(instructions="Is this about billing?")},
            ).alias("judgment")
        )
    """
    if rel is not _UNSET and first is not _UNSET:
        raise TypeError("jev received both first and rel; pass only one relation argument")
    relation = rel if rel is not _UNSET else first
    if isinstance(relation, Relation):
        if state is _UNSET:
            raise TypeError("jev relation API requires state")
        output_column = "response" if output_column is _UNSET else output_column
        if not isinstance(output_column, str) or not output_column.strip():
            raise ValueError("output_column must be a non-empty string")
    else:
        if rel is not _UNSET:
            raise TypeError("jev rel= must be a Relation")
        if first is not _UNSET and state is not _UNSET:
            raise TypeError("jev received both first and state")
        state = first if first is not _UNSET else state
        if state is _UNSET:
            raise TypeError("jev requires state")
        if output_column is not _UNSET:
            raise TypeError("jev output_column is only supported by the relation API; use Expression.alias()")
        relation = None
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Jev model must be a non-empty string")
    udf_options, client_options = _prepare_options(options, on_error)
    prepared = _prepare_questions(questions)
    wrapper = _JevBatch(prepared, model, client_options, udf_options)
    # Encode on the engine side so JSON, STRUCT, and LIST values arrive as
    # structured state while VARCHAR remains text (even when it looks like JSON).
    encoded_state = vane.FunctionExpression("to_json", as_expression(state)).cast("VARCHAR")
    expression = _build_ai_batch_expression(
        wrapper,
        inputs={"state": encoded_state},
        output_column="response",
        output_type="VARCHAR",
        udf_opts=udf_options,
        name="ai_jev",
        execution_backend=options.get("execution_backend"),
    )
    if relation is None:
        return expression
    star = _star_excluding_existing_output_column(relation, output_column)
    return relation.select(star, expression.alias(output_column))

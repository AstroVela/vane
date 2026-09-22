# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Row-preserving speech transcription on the existing AI execution runtime."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict
from typing import Any, Literal, overload

import pyarrow as pa
from typing_extensions import Unpack

import vane
from vane._typing import Expression, Relation
from vane.ai._media import normalize_media_content_type
from vane.ai._transcription_types import MAX_TRANSCRIPTION_BYTES, TranscriptionInput, validate_transcription
from vane.ai.functions import (
    _build_ai_batch_expression,
    _log_substituted_failure,
    _missing_async_runtime,
    _packed_prompt_messages,
    _resolve_provider,
    _retry_call_async,
    _star_excluding_existing_output_column,
    _validate_on_error,
)
from vane.ai.options import TranscribeOptions
from vane.ai.protocols import Transcriber, TranscriberDescriptor
from vane.ai.provider import Provider, _safe_provider_execution_error
from vane.ai.typing import UDFOptions

_OUTPUT_TYPE = 'STRUCT(text VARCHAR, language VARCHAR, duration DOUBLE, segments STRUCT(start DOUBLE, "end" DOUBLE, text VARCHAR)[])'
_EXECUTION_OPTIONS = {"batch_size", "actor_number", "max_concurrency_per_actor", "execution_backend", "max_retries"}


def _prepare_options(options: Mapping[str, Any], on_error: Literal["raise", "ignore"]) -> UDFOptions:
    unknown = set(options) - TranscribeOptions.__annotations__.keys()
    if unknown:
        raise TypeError(f"Unsupported Transcribe option(s): {', '.join(sorted(unknown))}")
    _validate_on_error(on_error)
    values = {"batch_size": 4, "actor_number": 1, "max_concurrency_per_actor": 4, "max_retries": 3}
    for name, default in values.items():
        value = options.get(name, default)
        if type(value) is not int or value < (0 if name == "max_retries" else 1):
            raise ValueError(f"Invalid Transcribe execution option {name!r}")
        values[name] = value
    backend = options.get("execution_backend")
    if backend is not None and backend not in {"subprocess_task", "subprocess_actor", "ray_task", "ray_actor"}:
        raise ValueError("Invalid Transcribe execution_backend")
    if backend in {"subprocess_task", "ray_task"} and "actor_number" in options:
        raise ValueError("Transcribe actor_number requires an actor execution backend")
    return UDFOptions(**values, num_gpus=0, on_error=on_error)


class _TranscribeBatch:
    def __init__(self, descriptor: TranscriberDescriptor, options: UDFOptions):
        self._descriptor = descriptor
        self._options = options
        self._client: Transcriber | None = None
        self._run_async: Callable[[Awaitable[Any]], Any] | None = None

    def bind_async_runtime(self, run_async: Callable[[Awaitable[Any]], Any]) -> None:
        self._run_async = run_async

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_client": None, "_run_async": None}

    def _error(self, operation: str, error: Exception) -> Exception:
        return _safe_provider_execution_error(
            self._descriptor.get_provider(), self._descriptor.get_model(), f"Transcribe {operation}", error
        )

    def close(self) -> None:
        client, self._client = self._client, None
        if client is None or self._run_async is None:
            return
        error = None
        try:
            self._run_async(client.aclose())
        except Exception as exc:
            error = self._error("cleanup", exc)
        if error is not None:
            raise error from None

    def _decode(self, packed: Any) -> TranscriptionInput | None:
        if packed is None:
            return None
        if not isinstance(packed, dict) or set(packed) != {"message_0"}:
            raise ValueError("Transcribe input did not cross the FILE materialization boundary")
        value = packed["message_0"]
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) != {"content_type", "data", "error"}:
            raise ValueError("Transcribe requires a URL, FILE or AUDIOFILE expression")
        if value["error"] is not None:
            # Native errors are row values so explicit on_error='ignore' can
            # return NULL without sending invalid or partial data to a model.
            code = value["error"]
            if code in {"too_large", "vector_too_large"}:
                raise ValueError("Transcribe FILE exceeds the materialization byte budget")
            raise ValueError("Transcribe FILE cannot be read as nonempty supported audio")
        data, mime = value["data"], value["content_type"]
        if not isinstance(data, bytes) or not 0 < len(data) <= MAX_TRANSCRIPTION_BYTES:
            raise ValueError("Transcribe audio must contain 1 byte to 20 MiB")
        if mime not in self._descriptor.supported_media_mime_types():
            raise ValueError("Transcribe provider does not support the audio MIME type")
        return TranscriptionInput(data, mime)

    def __call__(self, table: pa.Table) -> pa.Table:
        inputs = []
        for value in table.column("audio").to_pylist():
            try:
                inputs.append(self._decode(value))
            except ValueError as exc:
                if self._options.on_error == "raise":
                    raise
                _log_substituted_failure(exc, on_error="ignore")
                inputs.append(None)
        if not any(value is not None for value in inputs):
            return pa.table({"transcription": pa.array([None] * len(inputs), type=pa.string())})
        if self._run_async is None:
            raise _missing_async_runtime()

        async def run_all() -> list[str | None]:
            if self._client is None:
                error = None
                try:
                    self._client = self._descriptor.instantiate()
                except Exception as exc:
                    error = self._error("initialization", exc)
                if error is not None:
                    raise error from None
            client = self._client
            assert client is not None
            semaphore = asyncio.Semaphore(self._options.max_concurrency_per_actor or 1)

            async def invoke(audio: TranscriptionInput | None) -> str | None:
                if audio is None:
                    return None
                async with semaphore:
                    error = None
                    try:
                        result = await _retry_call_async(
                            client.transcribe, audio, max_retries=self._options.max_retries, on_error="raise"
                        )
                        return json.dumps(asdict(validate_transcription(result)), ensure_ascii=False, allow_nan=False)
                    except Exception as exc:
                        if self._options.on_error == "ignore":
                            _log_substituted_failure(exc, on_error="ignore")
                            return None
                        error = self._error("execution", exc)
                    raise error from None

            tasks = [asyncio.create_task(invoke(value)) for value in inputs]
            try:
                return await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        return pa.table({"transcription": pa.array(self._run_async(run_all()), type=pa.string())})


_UNSET: Any = object()


@overload
def transcribe(
    audio: Expression,
    /,
    *,
    provider: str | Provider = "openai",
    model: str | None = None,
    on_error: Literal["raise", "ignore"] = "raise",
    **options: Unpack[TranscribeOptions],
) -> Expression: ...


@overload
def transcribe(
    *,
    audio: Expression,
    provider: str | Provider = "openai",
    model: str | None = None,
    on_error: Literal["raise", "ignore"] = "raise",
    **options: Unpack[TranscribeOptions],
) -> Expression: ...


@overload
def transcribe(
    rel: Relation,
    /,
    audio: Expression,
    *,
    provider: str | Provider = "openai",
    model: str | None = None,
    on_error: Literal["raise", "ignore"] = "raise",
    output_column: str = "transcription",
    **options: Unpack[TranscribeOptions],
) -> Relation: ...


@overload
def transcribe(
    *,
    rel: Relation,
    audio: Expression,
    provider: str | Provider = "openai",
    model: str | None = None,
    on_error: Literal["raise", "ignore"] = "raise",
    output_column: str = "transcription",
    **options: Unpack[TranscribeOptions],
) -> Relation: ...


def transcribe(
    first: Any = _UNSET,
    /,
    audio: Any = _UNSET,
    *,
    rel: Any = _UNSET,
    provider: str | Provider = "openai",
    model: str | None = None,
    on_error: Literal["raise", "ignore"] = "raise",
    output_column: Any = _UNSET,
    **options: Unpack[TranscribeOptions],
) -> Expression | Relation:
    """Transcribe URL/FILE/AUDIOFILE expressions into a typed STRUCT.

    Returns ``text``, ``language``, ``duration`` in seconds and ``segments``
    containing ``start``, ``end`` and ``text``. Segments preserve the model's
    alignment, including overlaps; they are not verified speech boundaries.
    NULL input returns NULL without loading a client. Silence returns an empty
    transcript and segment list. Explicit ``on_error='ignore'`` returns NULL
    for failed rows; provider initialization errors always propagate.

    The initial OpenAI provider requires ``whisper-1`` and ``vane-ai[openai]``.
    Configure credentials before constructing the expression. Client settings
    are captured on the application and reused on workers. FILE bytes cross
    the existing bounded native media transport: 20 MiB per file, 128 MiB per
    materialization vector. Workers need access to referenced FILE resources.
    Output is bounded to 100,000 characters and 4,096 timed segments per row.
    No model substitution, untimed response conversion or inferred timestamps.

    Example::

        from vane.ai import transcribe

        result = recordings.select(
            transcribe(vane.col("audio"), model="whisper-1").alias("speech")
        )
    """
    if rel is not _UNSET and first is not _UNSET:
        raise TypeError("transcribe received both first and rel")
    relation = rel if rel is not _UNSET else first
    if isinstance(relation, Relation):
        if audio is _UNSET:
            raise TypeError("transcribe relation API requires audio")
        output_column = "transcription" if output_column is _UNSET else output_column
        if not isinstance(output_column, str) or not output_column.strip():
            raise ValueError("output_column must be a nonempty string")
    else:
        if rel is not _UNSET:
            raise TypeError("transcribe rel= must be a Relation")
        if first is not _UNSET and audio is not _UNSET:
            raise TypeError("transcribe received both first and audio")
        audio = first if first is not _UNSET else audio
        if output_column is not _UNSET:
            raise TypeError("transcribe output_column requires the relation API; use Expression.alias()")
        relation = None
    if not isinstance(audio, Expression):
        raise TypeError("transcribe requires an audio Expression")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("Transcribe model must be a nonempty string or None")
    udf_options = _prepare_options(options, on_error)
    descriptor = _resolve_provider(provider).get_transcriber(
        model, options={key: value for key, value in options.items() if key not in _EXECUTION_OPTIONS}
    )
    if not isinstance(descriptor, TranscriberDescriptor):
        raise TypeError("Provider.get_transcriber must return a TranscriberDescriptor")
    mime_types = descriptor.supported_media_mime_types()
    if not mime_types or any(normalize_media_content_type(mime) != mime for mime in mime_types):
        raise ValueError("Transcribe descriptor requires normalized supported audio MIME types")
    # AUDIOFILE supplies a bind-time guard and shares the existing FILE byte
    # budget, MIME checks, storage credentials and locator-free model transport.
    packed = _packed_prompt_messages(
        [vane.audio_file(audio)],
        supports_media_inputs=True,
        supported_media_mime_types=tuple(sorted(mime_types)),
        single_message=True,
    )
    expression = (
        _build_ai_batch_expression(
            _TranscribeBatch(descriptor, udf_options),
            inputs={"audio": packed},
            output_column="transcription",
            output_type="VARCHAR",
            udf_opts=udf_options,
            name="ai_transcribe",
            execution_backend=options.get("execution_backend"),
        )
        .cast("JSON")
        .cast(_OUTPUT_TYPE)
    )
    if relation is None:
        return expression
    return relation.select(
        _star_excluding_existing_output_column(relation, output_column), expression.alias(output_column)
    )

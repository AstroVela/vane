# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Public diagnostics reconstructed without copying arbitrary provider text."""

from __future__ import annotations

import re
from typing import Any

_ENGINE_OPTIONS = frozenset(
    {
        "max_model_len",
        "context_length",
        "mem_fraction_static",
        "gpu_memory_utilization",
        "max_num_batched_tokens",
        "max_num_seqs",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "tp_size",
        "pp_size",
        "dp_size",
        "dtype",
        "device",
        "tokenizer",
        "tokenizer_path",
        "trust_remote_code",
        "revision",
        "quantization",
        "attention_backend",
        "is_embedding",
    }
)
_ERROR_TYPES = frozenset(
    {
        "TypeError",
        "ValueError",
        "RuntimeError",
        "AssertionError",
        "ImportError",
        "ModuleNotFoundError",
        "MemoryError",
        "OutOfMemoryError",
        "OutOfMemoryException",
        "TimeoutError",
        "GetTimeoutError",
        "ConnectionError",
        "ConnectionRefusedError",
        "PermissionError",
        "FileNotFoundError",
        "OSError",
        "ActorDiedError",
        "RayActorError",
        "RayTaskError",
        "InvalidDataError",
        "InvalidInputException",
        "ProviderImportError",
        "ProviderCapabilityError",
        "EmbeddingConfigurationError",
        "HTTPStatusError",
        "APIStatusError",
        "APIConnectionError",
        "APITimeoutError",
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
        "BadRequestError",
        "RateLimitError",
        "InternalServerError",
        "ConnectError",
        "ReadError",
        "ReadTimeout",
        "ConnectTimeout",
        "RemoteProtocolError",
        "SSLCertVerificationError",
    }
)
_FRACTION = r"(?:0(?:\.[0-9]{1,6})?|1(?:\.0{1,6})?)"
_UNEXPECTED = re.compile(
    r"(?:[A-Za-z_][A-Za-z_0-9.]{0,127}\(\) )?got an unexpected keyword argument "
    r"'([A-Za-z_][A-Za-z_0-9]{0,127})'"
)
_MEMORY = re.compile(
    r"Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static="
    rf"({_FRACTION})\. Raise --mem-fraction-static above ({_FRACTION})\.?"
)
_SAFE_MEMORY = re.compile(
    rf"insufficient GPU KV-cache memory; mem_fraction_static=({_FRACTION}), required above ({_FRACTION})(?![0-9.])"
)
_PROVIDER_SUMMARY = re.compile(r"upstream error: ([A-Za-z_][A-Za-z_0-9]{0,127})([^\n]{0,1024})")
_STATUS = re.compile(r"(status_code|status|code|errno)=(-?[0-9]{1,6})(?![0-9])")


def _arguments(error: BaseException) -> tuple[Any, ...]:
    try:
        args = BaseException.__getattribute__(error, "args")
    except BaseException:
        return ()
    return args if type(args) is tuple else ()


def _message(error: BaseException) -> str:
    args = _arguments(error)
    if len(args) == 1 and type(args[0]) is str and len(args[0]) <= 65536:
        return args[0]
    return ""


def _type_name(error: BaseException) -> str:
    try:
        name = type.__getattribute__(type(error), "__name__")
    except BaseException:
        return "Exception"
    return name if type(name) is str and len(name) <= 128 and name.isascii() and name.isidentifier() else "Exception"


def technical_detail(error_type: str, message: str, *, transported: bool = False) -> str | None:
    """Only allow closed technical grammars, with no source/model/credential text."""

    def canonical(expected: str) -> bool:
        return message == expected or (transported and message.startswith(expected))

    if error_type == "TypeError":
        match = _UNEXPECTED.fullmatch(message)
        if match:
            option = match[1] if match[1] in _ENGINE_OPTIONS else "<unrecognized option>"
            return f"unexpected engine keyword argument '{option}'; check the selected provider's engine_args"
        for option in (*_ENGINE_OPTIONS, "<unrecognized option>"):
            expected = f"unexpected engine keyword argument '{option}'; check the selected provider's engine_args"
            if canonical(expected):
                return expected
    if error_type == "ValueError":
        match = _MEMORY.fullmatch(message) or (
            _SAFE_MEMORY.match(message) if transported else _SAFE_MEMORY.fullmatch(message)
        )
        if match:
            return f"insufficient GPU KV-cache memory; mem_fraction_static={match[1]}, required above {match[2]}"
    if error_type in {"OutOfMemoryError", "OutOfMemoryException", "MemoryError"}:
        return "memory allocation failed; check model and worker memory budgets"
    if error_type == "RuntimeError":
        if "CUDA out of memory" in message or canonical(
            "CUDA memory allocation failed; check model and worker GPU memory budgets"
        ):
            return "CUDA memory allocation failed; check model and worker GPU memory budgets"
    return None


def _attributes(error: BaseException) -> dict[str, Any]:
    try:
        attrs = BaseException.__getattribute__(error, "__dict__")
    except BaseException:
        return {}
    return attrs if type(attrs) is dict else {}


def _summary(error: BaseException) -> str:
    name = _type_name(error)
    message = _message(error)
    detail = technical_detail(name, message)
    if detail:
        return f"{name}: {detail}"
    # Native query transport may flatten Python errors. Only reconstruct the
    # provider's safe grammar; surrounding SQL, prompts and paths are discarded.
    match = _PROVIDER_SUMMARY.search(message)
    if match:
        name = match[1] if match[1] in _ERROR_TYPES else "ProviderError"
        tail = match[2].strip()
        # Native transport embeds the canonical message in a larger JSON or
        # traceback string. Reconstruct only the recognized prefix and discard
        # its suffix, including any transport syntax or private request text.
        detail = technical_detail(name, tail.removeprefix(": "), transported=True)
        if detail:
            return f"{name}: {detail}"
        status = _STATUS.search(tail)
        return f"{name} ({status[1]}={status[2]})" if status else name
    attrs = _attributes(error)
    for field in ("status_code", "status", "code", "errno"):
        value = attrs.get(field)
        if field == "errno" and isinstance(error, OSError):
            value = OSError.__dict__["errno"].__get__(error)
        if type(value) is int and -999999 <= value <= 999999:
            return f"{name} ({field}={value})"
    return name


def summarize_error(error: BaseException, *, max_chars: int = 512) -> str:
    """Return bounded root-cause and cleanup diagnostics safe for API responses.

    Unknown messages retain their exception type; known technical messages are
    reconstructed from a closed vocabulary. Follows bounded Python/Ray chains
    without importing Ray, inspecting tracebacks or invoking exception __str__.
    """
    if not isinstance(error, BaseException):
        raise TypeError("error must be an exception")
    if type(max_chars) is not int or not 64 <= max_chars <= 4096:
        raise ValueError("max_chars must be between 64 and 4096")
    seen: set[int] = set()
    current = error
    summaries: list[str] = []
    cleanup: list[str] = []
    for _ in range(16):
        if id(current) in seen:
            break
        seen.add(id(current))
        summary = _summary(current)
        if summary not in summaries:
            summaries.append(summary)
        attrs = _attributes(current)
        failures = attrs.get("cleanup_errors", ())
        if type(failures) in (tuple, list):
            for failure in failures[:3]:
                if isinstance(failure, BaseException) and len(cleanup) < 3:
                    cleanup.append(_summary(failure))
        following = next(
            (
                attrs[key]
                for key in ("primary_error", "creation_error", "cause")
                if isinstance(attrs.get(key), BaseException)
            ),
            None,
        )
        if following is None:
            # Ray's ActorDiedError retains a RayTaskError as its constructor
            # argument rather than a cause attribute. Preserve that chain too.
            args = _arguments(current)
            if len(args) == 1 and isinstance(args[0], BaseException):
                following = args[0]
        if following is None:
            for key in ("__cause__", "__context__"):
                candidate = BaseException.__getattribute__(current, key)
                if isinstance(candidate, BaseException):
                    following = candidate
                    break
        if following is None:
            break
        current = following
    result = " <- ".join(reversed(summaries))
    if cleanup:
        suffix = "; cleanup: " + "; ".join(cleanup)
        suffix = suffix[: max_chars // 2]
        result = result[: max_chars - len(suffix)] + suffix
    return result if len(result) <= max_chars else result[: max_chars - 1] + "…"

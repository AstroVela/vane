# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Public diagnostics reconstructed without copying arbitrary provider text."""

from __future__ import annotations

import base64
import binascii
import errno
import json
import re
import sys
from collections.abc import Iterator
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
_CPU_MEMORY = re.compile(
    r"(?:\[enforce fail at [^\]\r\n]{0,512}alloc_cpu\.cpp:[0-9]{1,8}\] err == 0\. )?"
    r"DefaultCPUAllocator: can't allocate memory: you tried to allocate [0-9]{1,20} bytes\. "
    r"Error code [0-9]{1,10} \([^\r\n)]{1,128}\)"
)
_CPU_MEMORY_DETAIL = "CPU memory allocation failed; check model and worker CPU memory budgets"
_CUDA_MEMORY_DETAIL = "CUDA memory allocation failed; check model and worker GPU memory budgets"
_MLX_MEMORY = re.compile(
    r"(?:\[malloc\] Unable to allocate [0-9]{1,20} bytes\."
    r"|\[metal::malloc\] Resource limit \([0-9]{1,20}\) exceeded\."
    r"|\[metal::malloc\] Attempting to allocate [0-9]{1,20} bytes which is greater than"
    r" the maximum allowed buffer size of [0-9]{1,20} bytes\."
    r"|\[METAL\] Command buffer execution failed: Insufficient Memory"
    r"(?: \(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory\))?\.?)"
)
_MLX_MEMORY_DETAIL = "MLX memory allocation failed; check model and worker memory budgets"
_ONNX_MEMORY = re.compile(
    r"\[ONNXRuntimeError\] : (?:1 : FAIL|6 : RUNTIME_EXCEPTION) : "
    r"(?:Non-zero status code returned while running [A-Za-z_][A-Za-z_0-9]{0,127} node\. ?"
    r"Name:'[^\r\n]{0,2048}' Status Message: |Exception during initialization: )"
    r"[^\r\n]{0,2048}bfc_arena\.cc:[0-9]{1,8} "
    r"(?:void\* )?onnxruntime::BFCArena::AllocateRawInternal(?:\([^()\r\n]{0,512}\))? "
    r"Failed to allocate memory for requested buffer of size [0-9]{1,20}"
)
_ONNX_MEMORY_DETAIL = "ONNX Runtime memory allocation failed; check model and worker memory budgets"
_TYPE = re.compile(r"[A-Za-z_][A-Za-z_0-9]{0,127}(?!\w)")
_STATUS_FIELD = r"(?:status_code|status|code|errno)=-?[0-9]{1,6}"
_STATUS_GROUP = re.compile(rf" \({_STATUS_FIELD}(?:, {_STATUS_FIELD}){{0,3}}\)")
_REMOTE_RESULT_COUNTS = re.compile(
    r"vllm actor finished without returning all submitted results: "
    r"actor_idx=[0-9]{1,16} submitted=[0-9]{1,16} received=[0-9]{1,16}(?![0-9])"
)
_DIAGNOSTIC_HEADER = "[vane.diagnostic.v1:"
_MAX_DIAGNOSTIC_ITEMS = 64
_SAFE_ERROR_DETAILS = {
    "distributed Ray UDF payload requires query_id": "RuntimeError",
    "distributed Ray UDF payload requires pre-registered resource_unit_id": "RuntimeError",
    "vllm remote wait exceeded query deadline": "RuntimeError",
    "vllm remote wait has no pending actor wait refs before completion": "RuntimeError",
    **{
        f"{provider} {detail}": "RuntimeError"
        for provider in ("vllm", "sglang")
        for detail in ("returned no outputs", "engine not initialized")
    },
    **{
        f"{provider} API key was not configured on the application before creating the provider/descriptor": "ProviderClientConfigurationError"
        for provider in ("Google", "OpenAI", "TypeSafe")
    },
    "Anthropic api_key or auth_token was not configured on the application before creating the provider/descriptor": "ProviderClientConfigurationError",
    "Google Vertex credentials require an explicit application project": "ProviderClientConfigurationError",
    "Google credentials must be a google.auth.credentials.Credentials object": "ProviderClientConfigurationError",
}


class DiagnosticError(RuntimeError):
    """An explicitly sanitized exception, safe to carry across query workers.

    ``operation`` is an application-owned description, never upstream text.
    Only the bounded summary of ``error`` is retained, not the original object.
    """

    def __init__(self, operation: str, error: BaseException):
        self.summary = summarize_error(error)
        super().__init__(_diagnostic_message(self.summary, f"{operation}; upstream error: {self.summary}"))

    def __reduce__(self) -> tuple[Any, tuple[str, str]]:
        return _restore_diagnostic_error, (self.args[0], self.summary)


def _restore_diagnostic_error(message: str, summary: str) -> DiagnosticError:
    error = DiagnosticError.__new__(DiagnosticError)
    RuntimeError.__init__(error, message)
    error.summary = summary
    return error


def _diagnostic_message(summary: str, message: str) -> str:
    # Put the structured summary before any application-owned description.
    # Model identifiers may themselves contain newlines or diagnostic markers.
    encoded = base64.urlsafe_b64encode(summary.encode()).decode("ascii")
    return f"{_DIAGNOSTIC_HEADER}{encoded}] {message}"


def _encoded_summary(message: str) -> str | None:
    if not message.startswith(_DIAGNOSTIC_HEADER):
        return None
    encoded, separator, _ = message[len(_DIAGNOSTIC_HEADER) :].partition("] ")
    if not separator or not 0 < len(encoded) <= 8192:
        return None
    try:
        summary = base64.b64decode(encoded, altchars=b"-_", validate=True).decode("utf-8")
    except (ValueError, binascii.Error, UnicodeError):
        return None
    if len(summary) > 4096:
        return None
    return _transported_summary(summary)


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
    return name if type(name) is str and _TYPE.fullmatch(name) else "Exception"


def technical_detail(error_type: str, message: str, *, transported: bool = False) -> str | None:
    """Only allow closed technical grammars, with no source/model/credential text."""

    def canonical(expected: str) -> bool:
        return message == expected or (transported and message.startswith(expected))

    for explanation, name in _SAFE_ERROR_DETAILS.items():
        if error_type == name and canonical(explanation):
            return explanation
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
    # RapidOCR keeps the typed ONNX exception in its cause/context chain. Read
    # that allocator error, never the wrapper's arbitrary traceback string.
    if error_type in {"Fail", "RuntimeException"} and (
        _ONNX_MEMORY.fullmatch(message.split("\n", 1)[0]) or canonical(_ONNX_MEMORY_DETAIL)
    ):
        return _ONNX_MEMORY_DETAIL
    if error_type == "RuntimeError":
        if _CPU_MEMORY.fullmatch(message.split("\n", 1)[0]) or canonical(_CPU_MEMORY_DETAIL):
            return _CPU_MEMORY_DETAIL
        # MLX 0.32.4's Metal allocator and command buffers report OOM as
        # RuntimeError. Preserve the resource category with fixed text before the
        # provider message is discarded; never match arbitrary OOM substrings.
        if _MLX_MEMORY.fullmatch(message) or canonical(_MLX_MEMORY_DETAIL):
            return _MLX_MEMORY_DETAIL
        match = _REMOTE_RESULT_COUNTS.match(message) if transported else _REMOTE_RESULT_COUNTS.fullmatch(message)
        if match:
            return match[0]
        # CTranslate2 (used by Faster-Whisper) throws RuntimeError rather than
        # a typed OOM. Preserve its exact CUDA_CHECK allocation failure before
        # sanitization discards the provider message; emit only our fixed text.
        if (
            "CUDA out of memory" in message
            or message == "CUDA failed with error out of memory"
            or canonical(_CUDA_MEMORY_DETAIL)
        ):
            return _CUDA_MEMORY_DETAIL
    return None


def _attributes(error: object) -> dict[str, Any]:
    try:
        attrs = object.__getattribute__(error, "__dict__")
    except BaseException:
        return {}
    return attrs if type(attrs) is dict else {}


def _stored_attribute(value: object, name: str) -> Any:
    attrs = _attributes(value)
    if name in attrs:
        return attrs[name]
    # Some SDK errors declare numeric status as a class attribute. Read stored
    # values without invoking arbitrary SDK properties on the failure path.
    try:
        for cls in type.__getattribute__(type(value), "__mro__")[:16]:
            namespace = type.__getattribute__(cls, "__dict__")
            if name in namespace:
                return namespace[name]
    except BaseException:
        return None
    return None


def _numeric_details(error: BaseException) -> list[str]:
    details = []
    for field in ("status_code", "status", "code", "errno"):
        value = _stored_attribute(error, field)
        if field == "errno" and isinstance(error, OSError):
            value = OSError.__dict__["errno"].__get__(error)
        if field == "status_code" and type(value) is not int:
            # HTTPX keeps status on response, not on HTTPStatusError. Inspect
            # only that numeric field, never request URLs, headers or bodies.
            value = _stored_attribute(_stored_attribute(error, "response"), "status_code")
        if type(value) is int and -999999 <= value <= 999999:
            details.append(f"{field}={value}")
    return details


def _summary_parts(message: str) -> tuple[list[str], list[str]]:
    """Read complete canonical atoms, retaining cause/cleanup boundaries."""
    if name := _SAFE_ERROR_DETAILS.get(message):
        return [f"{name}: {message}"], []
    chains: list[list[str]] = [[]]
    position = 0
    for _ in range(_MAX_DIAGNOSTIC_ITEMS):
        match = _TYPE.match(message, position)
        if match is None:
            break
        # The producer already sanitizes names with this same bounded grammar.
        # Preserve SDK/catalog types without a dependency-specific allowlist.
        name = match[0]
        position = match.end()
        rendered = name
        status = _STATUS_GROUP.match(message, position)
        if status:
            rendered += status[0]
            position = status.end()
        elif message.startswith(": ", position):
            tail = message[position + 2 :]
            detail = technical_detail(name, tail, transported=True)
            if detail and tail.startswith(detail):
                rendered += ": " + detail
                position += 2 + len(detail)
        chains[-1].append(rendered)
        if message.startswith(" <- ", position):
            separator = " <- "
        elif len(chains) < 4 and message.startswith("; cleanup: ", position):
            separator = "; cleanup: "
            chains.append([])
        elif 1 < len(chains) < 4 and message.startswith("; ", position):
            separator = "; "
            chains.append([])
        else:
            break
        position += len(separator)
    return chains[0] or ["ProviderError"], [" <- ".join(chain) for chain in chains[1:] if chain]


def _transported_summary(message: str) -> str:
    primary, cleanup = _summary_parts(message)
    result = " <- ".join(primary)
    return result + "; cleanup: " + "; ".join(cleanup) if cleanup else result


def _is_exception(error: BaseException, module_name: str, *names: str) -> bool:
    # Inspect already loaded transport modules, without importing Ray/native
    # code or treating a user-supplied class name as proof of provenance.
    namespace = _attributes(sys.modules.get(module_name))
    return any(isinstance(cls := namespace.get(name), type) and isinstance(error, cls) for name in names)


def _native_exception_message(message: str, expected_type: str) -> str | None:
    try:
        payload = json.loads(message)
    except (ValueError, RecursionError):
        return None
    if type(payload) is not dict or payload.get("exception_type") != expected_type:
        return None
    nested_message = payload.get("exception_message")
    return nested_message if type(nested_message) is str else None


def _native_summary(message: str) -> str | None:
    """Decode complete native UDF/LLM envelopes, never search arbitrary text."""
    message = message.removeprefix("DuckDBError::ExternalError: ")
    if message.startswith("{"):
        # Terminal INSERT tasks can wrap the FTE failure in the native spawn
        # boundary's INTERNAL envelope. Require that exact envelope; arbitrary
        # internal-error bodies must not acquire diagnostic provenance.
        nested = _native_exception_message(message, "INTERNAL")
        prefix = "[PlanExecutionContext::spawn] task error: DuckDBError::InternalError: "
        if nested is None or not nested.startswith(prefix):
            return None
        message = nested[len(prefix) :]
    for prefix in ("FTE query wait failed: FTE query failed: Execution failed: ", "Execution failed: "):
        if message.startswith(prefix):
            nested = _native_exception_message(
                message[len(prefix) :].partition("\n--- DIAGNOSTIC DUMP ---\n")[0], "Invalid Input"
            )
            if nested is None:
                return None
            message = nested
            break
    message = message.removeprefix("Attempting to execute an unsuccessful or closed pending query result\nError: ")
    message = message.removeprefix("Invalid Input Error: ")
    if message.startswith(("udf async error: ", "streaming UDF async error: ")):
        message = message.split(": ", 1)[1]
        # Initialization can reach either the slot error or the streaming
        # collector first. Both carry the native InvalidInputException JSON.
        if message.startswith("{"):
            nested = _native_exception_message(message, "Invalid Input")
            if nested is None or not nested.startswith("Failed to build udf executor: "):
                return None
            message = nested.removeprefix("Failed to build udf executor: ")
    elif message.startswith("Python exception occurred while executing the UDF: "):
        message = message.removeprefix("Python exception occurred while executing the UDF: ")
    elif message.startswith(
        (
            # Both engines use VLLMPythonExecutor's native bridge.
            "vllm submit failed: ",
            "vllm take_ready_result failed: ",
            "vllm finished_submitting failed: ",
            "vllm all_tasks_finished failed: ",
            "vllm register_wakeup_callback failed: ",
            "vllm wait_for_result failed: ",
            "vllm shutdown failed: ",
        )
    ):
        message = message.split(": ", 1)[1]
    else:
        return None
    # These headers are produced from actual exception types by the transport.
    # A ClientError/RuntimeError body that contains such a header is not one.
    for name in ("DiagnosticError", "ProviderCapabilityError"):
        if message.startswith(name + ": "):
            return _encoded_summary(message[len(name) + 2 :])
    return None


def _native_query_diagnostic(error: BaseException) -> DiagnosticError | None:
    """Capture a native query failure at the producer, before Ray flattens it.

    Call only at a native query operation boundary. A generic RuntimeError or
    RayTaskError is not proof that its message came from the native runtime.
    """
    summary = _native_summary(_message(error))
    if summary is None:
        return None
    message = _diagnostic_message(summary, f"Native query failed; upstream error: {summary}")
    return _restore_diagnostic_error(message, summary)


def _capture_native_datasink_diagnostic(result: dict[str, Any], *, diagnostic: DiagnosticError | None = None) -> None:
    """Capture a returned failure at a native DataSink producer boundary."""
    if result.get("outcome_unknown") is not True and result.get("outcome_aborted") is not True:
        return
    if diagnostic is None and type(detail := result.get("outcome_error")) is str:
        diagnostic = _native_query_diagnostic(RuntimeError(detail))
    if diagnostic is not None:
        result["outcome_diagnostic"] = diagnostic


def _summary(error: BaseException) -> str:
    attributes = _attributes(error)
    if isinstance(error, DiagnosticError) and type(summary := attributes.get("summary")) is str:
        return _transported_summary(summary)
    from vane.ai.provider import ProviderCapabilityError, _SafeProviderError

    if isinstance(error, ProviderCapabilityError) and type(summary := attributes.get("original_error_summary")) is str:
        return _transported_summary(summary)
    if isinstance(error, _SafeProviderError):
        # Internal typed wrappers contain already sanitized text. Raw SDK
        # errors never enter this branch, regardless of their message contents.
        message = _message(error)
        position = message.find(_DIAGNOSTIC_HEADER)
        encoded = _encoded_summary(message[position:]) if position >= 0 else None
        return encoded if encoded is not None else _transported_summary(message)
    name = _type_name(error)
    message = _message(error)
    if _is_exception(error, "vane._native", "Error"):
        transported = _native_summary(message)
        if transported is not None:
            return transported
    detail = technical_detail(name, message)
    if detail:
        return f"{name}: {detail}"
    details = _numeric_details(error)
    return f"{name} ({', '.join(details)})" if details else name


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    """Walk one bounded cause chain, equally for execution and cleanup."""
    seen: set[int] = set()
    current = error
    for _ in range(16):
        if id(current) in seen:
            break
        seen.add(id(current))
        yield current
        attrs = _attributes(current)
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


def _bounded_chain(parts: list[str], max_chars: int) -> list[str]:
    """Keep complete atoms so another transport can still read the chain."""
    result: list[str] = []
    size = 0
    for part in parts:
        next_size = size + (4 if result else 0) + len(part)
        if next_size > max_chars:
            if not result:
                # A very small caller budget can exclude even the root atom.
                # Drop its optional detail, never part of a status or name.
                match = _TYPE.match(part)
                name = match[0] if match else "Exception"
                if len(name) > max_chars:
                    name = "Exception"
                return [name] if len(name) <= max_chars else []
            break
        result.append(part)
        size = next_size
    return result


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
    summaries: list[str] = []
    cleanup: list[str] = []

    def add_cleanup(summary: str) -> None:
        if summary not in cleanup and len(cleanup) < 3:
            cleanup.append(summary)

    # Merge root-to-wrapper so cleanup already captured by a worker stays
    # separate from the cause chain and precedes any later query teardown.
    # Rewrapping must produce one canonical cleanup section, not nest another
    # section inside an exception atom or attribute later errno to the root.
    for current in reversed(list(_exception_chain(error))):
        primary, captured_cleanup = _summary_parts(_summary(current))
        for summary in primary:
            if summary not in summaries:
                summaries.append(summary)
        for summary in captured_cleanup:
            add_cleanup(summary)
        failures = _attributes(current).get("cleanup_errors", ())
        if type(failures) in (tuple, list):
            for failure in failures[:3]:
                if isinstance(failure, BaseException) and len(cleanup) < 3:
                    # At most three cleanup chains of sixteen nodes; do not
                    # recurse into cleanup_errors (which can contain cycles).
                    parts: list[str] = []
                    nested_cleanup: list[str] = []
                    for cause in reversed(list(_exception_chain(failure))):
                        primary, captured_cleanup = _summary_parts(_summary(cause))
                        parts.extend(part for part in primary if part not in parts)
                        nested_cleanup.extend(captured_cleanup)
                    add_cleanup(" <- ".join(parts))
                    for summary in nested_cleanup:
                        add_cleanup(summary)
    # Reserve up to half the budget for cleanup. Both sections must end at
    # complete atoms: slicing inside a detail makes the next parser stop before
    # it reaches the cleanup separator, even when that separator is intact.
    suffix = ""
    remaining_items = _MAX_DIAGNOSTIC_ITEMS
    for summary in cleanup:
        separator = "; cleanup: " if not suffix else "; "
        parts, _ = _summary_parts(summary)
        # Leave at least one slot for the primary cause in the parser's budget.
        rendered = _bounded_chain(parts[: remaining_items - 1], max_chars // 2 - len(suffix) - len(separator))
        if not rendered:
            break
        suffix += separator + " <- ".join(rendered)
        remaining_items -= len(rendered)
    return " <- ".join(_bounded_chain(summaries[:remaining_items], max_chars - len(suffix))) + suffix


def _is_resource_error(error: BaseException) -> bool:
    """Recognize allocation failures through typed chains and safe diagnostics.

    Never search arbitrary exception text for errno or diagnostic markers.
    Keep this independent of torch, Ray and inference SDK imports.
    """
    for cause in _exception_chain(error):
        if isinstance(cause, MemoryError):
            return True
        if isinstance(cause, OSError):
            number = OSError.__dict__["errno"].__get__(cause)
            if type(number) is int and number in {errno.ENOMEM, errno.ENOSPC}:
                return True
    primary, cleanup = _summary_parts(summarize_error(error, max_chars=4096))
    atoms = primary + [atom for chain in cleanup for atom in _summary_parts(chain)[0]]
    for atom in atoms:
        name = _TYPE.match(atom)
        if name is None:
            continue
        if name[0] in {"MemoryError", "OutOfMemoryError", "OutOfMemoryException"}:
            return True
        status = _STATUS_GROUP.match(atom, name.end())
        if status and any(
            field in {f"errno={errno.ENOMEM}", f"errno={errno.ENOSPC}"} for field in status[0][2:-1].split(", ")
        ):
            return True
        detail = atom[name.end() + 2 :] if atom.startswith(": ", name.end()) else ""
        if detail in {
            _CPU_MEMORY_DETAIL,
            _CUDA_MEMORY_DETAIL,
            _MLX_MEMORY_DETAIL,
            _ONNX_MEMORY_DETAIL,
        } or _SAFE_MEMORY.fullmatch(detail):
            return True
    return False


def _raise_if_resource_error(error: BaseException) -> None:
    """Normalize fatal allocation errors before retry or nullable-row handling."""
    if _is_resource_error(error):
        raise MemoryError("Embedding exhausted model or host resources") from None

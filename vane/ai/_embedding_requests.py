# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded, ordered request execution for built-in remote embedders.

The owning UDF executor supplies the event loop. Each worker owns at most
one request (including its retries); successful requests are never replayed
when another request fails. Custom provider protocols remain unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from vane.ai.provider import ProviderCapabilityError, _ProviderResultError

logger = logging.getLogger(__name__)


class _EmbeddingBatchError(_ProviderResultError):
    """Sanitized batch validation failure eligible for smaller input batches.

    Used when response shape or SDK validation prevents attributing vectors
    to input rows. Row-attributable vector failures use _ProviderResultError
    instead, so their successful neighbors do not need another request.
    """


def _is_request_wide_error(error: Exception) -> bool:
    """Recognize structured account/auth errors even when HTTP status is 400.

    SDK error envelopes differ: Google stores ErrorInfo in details, while
    OpenAI-compatible clients expose body/code/type/param. Only inspect these
    structured fields; messages and arbitrary metadata can contain input text.
    """
    labels = {
        "UNAUTHENTICATED",
        "UNAUTHORIZED",
        "PERMISSION_DENIED",
        "PERMISSION_ERROR",
        "FAILED_PRECONDITION",
        "SERVICE_DISABLED",
        "INVALID_API_KEY",
        "INVALID_AUTHENTICATION",
        "INVALID_ORGANIZATION",
        "INVALID_PROJECT",
        "INSUFFICIENT_QUOTA",
        "UNSUPPORTED_COUNTRY_REGION_TERRITORY",
    }
    prefixes = (
        "API_KEY_",
        "AUTHENTICATION_",
        "AUTHORIZATION_",
        "CREDENTIALS_",
        "ACCESS_TOKEN_",
        "BILLING_",
        "ACCOUNT_",
        "PROJECT_",
        "USER_PROJECT_",
        "ORGANIZATION_",
        "CONSUMER_",
        "IAM_",
    )
    fields = ("code", "status", "reason", "type", "param", "field")
    pending = [
        {name: getattr(error, name, None) for name in fields},
        getattr(error, "details", None),
        getattr(error, "body", None),
    ]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, Mapping):
            for name in fields:
                value = item.get(name)
                if not isinstance(value, str):
                    continue
                value = value.strip().replace("-", "_").upper()
                if name in {"param", "field"}:
                    if value in {"API_KEY", "AUTHORIZATION", "CREDENTIALS", "ORGANIZATION", "PROJECT", "BILLING"}:
                        return True
                elif value in labels or value.startswith(prefixes):
                    return True
            pending.extend(item.get(name) for name in ("error", "details", "errors"))
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
    return False


@dataclass
class EmbeddingMetrics:
    requests: int = 0
    retries: int = 0
    failed_inputs: int = 0
    input_tokens: int = 0
    estimated_tokens: int = 0
    request_seconds: float = 0.0
    queue_seconds: float = 0.0


class ManagedTextEmbedder(ABC):
    """Internal opt-in to request-level retry and row error handling."""

    @abstractmethod
    async def embed_text(self, text: list[str]) -> list[Any]:
        """Embed using this adapter's request planner."""
        ...

    def configure_execution(
        self,
        *,
        max_retries: int,
        on_error: str,
        validate: Callable[[Any], Any],
    ) -> None:
        self._request_max_retries = max_retries
        self._request_on_error = on_error
        self._validate_vector = validate

    @property
    def metrics(self) -> EmbeddingMetrics:
        if not hasattr(self, "_embedding_metrics"):
            self._embedding_metrics = EmbeddingMetrics()
        return self._embedding_metrics

    def _decode_response_vectors(self, values: Iterable[Any], decode: Callable[[Any], Any]) -> list[Any]:
        """Keep a malformed SDK vector attributable to its original row.

        Decode errors are represented with a sanitized result error only in
        ignore mode. The request runner handles that marker like any other
        row validation failure, without replaying the successful HTTP call.
        """
        decoded = []
        for value in values:
            try:
                decoded.append(decode(value))
                continue
            except (ValueError, TypeError, OverflowError):
                pass
            error = _ProviderResultError("Provider returned an embedding that cannot be decoded")
            if getattr(self, "_request_on_error", "raise") == "raise":
                raise error from None
            decoded.append(error)
        return decoded

    async def _run_requests(
        self,
        batches: Iterable[tuple[list[int], list[str]]],
        request: Callable[[list[str]], Awaitable[list[Any]]],
        results: list[Any],
    ) -> list[Any]:
        # Imports are deferred to keep provider modules independent of the
        # public functions module during import/descriptor reconstruction.
        from vane.ai.functions import (
            RetryAfterError,
            _is_transient_provider_error,
            _log_substituted_failure,
            _provider_status_code,
            _retry_wait_seconds,
        )

        if not results:
            return results
        queued_at = time.monotonic()
        retries = getattr(self, "_request_max_retries", 0)
        on_error = getattr(self, "_request_on_error", "raise")
        validate = getattr(self, "_validate_vector", lambda value: value)
        iterator = iter(batches)

        async def invoke(indices: list[int], texts: list[str]) -> None:
            failure: Exception | None = None
            for attempt in range(retries + 1):
                wait = None
                started = time.monotonic()
                self.metrics.requests += 1
                try:
                    values = await request(texts)
                    if len(values) != len(texts):
                        raise _EmbeddingBatchError("Embedding request must preserve input row count")
                    for index, value in zip(indices, values, strict=True):
                        try:
                            if isinstance(value, _ProviderResultError):
                                raise value
                            results[index] = validate(value)
                        except _ProviderResultError as exc:
                            if on_error == "raise":
                                raise
                            self.metrics.failed_inputs += 1
                            _log_substituted_failure(exc, on_error="ignore")
                    return
                except Exception as exc:
                    failure = exc
                    if (
                        not isinstance(exc, (ProviderCapabilityError, _ProviderResultError))
                        and not _is_request_wide_error(exc)
                        and _is_transient_provider_error(exc)
                        and attempt < retries
                    ):
                        self.metrics.retries += 1
                        wait = _retry_wait_seconds(exc, attempt)
                        if not isinstance(exc, RetryAfterError):
                            wait *= random.uniform(0.8, 1.2)
                finally:
                    self.metrics.request_seconds += time.monotonic() - started
                if wait is None:
                    break
                # Outside the exception handler: cancellation must not retain
                # a provider exception containing input or credentials.
                await asyncio.sleep(wait)

            assert failure is not None
            if on_error == "raise":
                raise failure
            # Input, payload-size, and batch validation errors can recover after
            # splitting. Never fan out exhausted 429/5xx or structured
            # auth/account errors.
            if (
                len(texts) > 1
                and not isinstance(failure, ProviderCapabilityError)
                and (isinstance(failure, _EmbeddingBatchError) or _provider_status_code(failure) in {400, 413, 422})
                and not _is_request_wide_error(failure)
            ):
                middle = len(texts) // 2
                await invoke(indices[:middle], texts[:middle])
                await invoke(indices[middle:], texts[middle:])
                return
            self.metrics.failed_inputs += len(texts)
            _log_substituted_failure(failure, on_error="ignore")

        async def worker() -> None:
            for indices, texts in iterator:
                self.metrics.queue_seconds += time.monotonic() - queued_at
                await invoke(indices, texts)

        tasks = [
            asyncio.create_task(worker()) for _ in range(min(getattr(self, "_request_concurrency", 1), len(results)))
        ]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            logger.debug("Embedding request metrics: %s", self.metrics)
        return results

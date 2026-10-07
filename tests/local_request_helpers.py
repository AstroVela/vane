# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Inject an operation into model-request lifecycle unit tests."""

from types import SimpleNamespace

_native_operation = SimpleNamespace(execute=lambda *args, **kwargs: None)


def run_request(request, plan, bindings, *, conn, execution_timeout=None):
    def run():
        request._prepare_execution(plan, bindings, conn=conn)
        return _native_operation.execute(conn, plan, cancellation=request._cancellation)

    return request._run_execution(run, execution_timeout=execution_timeout)

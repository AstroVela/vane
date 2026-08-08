# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import platform
import threading
import time

import pytest

import vane


class TestConnectionInterrupt:
    @pytest.mark.xfail(
        condition=platform.system() == "Emscripten",
        reason="threads not allowed on Emscripten",
    )
    def test_connection_interrupt(self):
        conn = vane.connect()

        def interrupt() -> None:
            # Wait for query to start running before interrupting
            time.sleep(0.1)
            conn.interrupt()

        thread = threading.Thread(target=interrupt)
        thread.start()
        with pytest.raises(vane.InterruptException):
            conn.execute("select count(*) from range(100000000000)").fetchall()
        thread.join()

    def test_interrupt_closed_connection(self):
        conn = vane.connect()
        conn.close()
        with pytest.raises(vane.ConnectionException):
            conn.interrupt()

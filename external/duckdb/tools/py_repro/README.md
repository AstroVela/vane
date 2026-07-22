# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT

# Python Thread Repros

This directory contains small pybind11 embedding reproducers for Python event-loop and thread interactions.

Build a repro locally instead of committing the compiled executable:

```bash
c++ -std=c++20 repro_async_thread_fixed.cpp \
  $(python3 -m pybind11 --includes) \
  $(python3-config --embed --ldflags) \
  -o repro_async_thread_fixed
```

Run it from this directory:

```bash
./repro_async_thread_fixed
```

The generated `repro_async_thread_fixed` binary is platform-specific build output and must stay untracked.

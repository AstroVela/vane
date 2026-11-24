# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Record absent setup failures explicitly before freezing a clean replay."""

import sys
from pathlib import Path

from generate import save
from prepare import main

if __name__ == "__main__":
    out = Path(sys.argv[1]).resolve()
    for name in ("initial-preflight.json", "full-pause-preflight.json"):
        if not (out / name).exists():
            save(out / name, dict(status="NOT_NEEDED", inputs={}))
    main()

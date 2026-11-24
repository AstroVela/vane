# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Cache reviewed media sources before bootstrapping the standalone source SDK."""

from __future__ import annotations

import hashlib
import http.client
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

# Avoid SourceForge responses that fail the pinned recipe's checksum check.
# These locations provide the same reviewed bytes; vcpkg verifies them again.
_SOURCE_MIRRORS = {
    "mpg123-1.33.4.tar.bz2": (
        "https://www.mpg123.de/download/mpg123-1.33.4.tar.bz2",
        "https://distfiles.macports.org/mpg123/mpg123-1.33.4.tar.bz2",
    ),
}


def prefetch_media_sources(project: Path, downloads: Path) -> None:
    """Reuse or atomically cache exact reviewed archives from alternate mirrors."""
    reviewed = json.loads((project / "source-licenses.json").read_bytes())
    downloads.mkdir(parents=True, exist_ok=True)
    for name, urls in _SOURCE_MIRRORS.items():
        expected = reviewed[name]["sha512"]
        cached = downloads / name
        if cached.is_file() and hashlib.sha512(cached.read_bytes()).hexdigest() == expected:
            continue
        errors = []
        for url in urls:
            try:
                with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=downloads) as directory:
                    staged = Path(directory) / name
                    with urllib.request.urlopen(url, timeout=60) as response, staged.open("wb") as output:
                        shutil.copyfileobj(response, output)
                    if hashlib.sha512(staged.read_bytes()).hexdigest() != expected:
                        raise ValueError("source SHA512 does not match the reviewed archive")
                    staged.replace(cached)
                break
            except (OSError, ValueError, http.client.HTTPException) as error:
                errors.append(f"{url}: {error}")
        else:
            raise RuntimeError(f"could not cache reviewed media source {name}: " + "; ".join(errors))

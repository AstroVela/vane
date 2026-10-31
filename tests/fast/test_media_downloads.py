# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import http.client
import io
import json
import urllib.error

import pytest

from vane_packaging import media_downloads


@pytest.fixture
def source_cache(tmp_path):
    contents = b"reviewed upstream source archive\n"
    name = "mpg123-1.33.4.tar.bz2"
    project = tmp_path / "project"
    project.mkdir()
    (project / "source-licenses.json").write_text(json.dumps({name: {"sha512": hashlib.sha512(contents).hexdigest()}}))
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    return project, downloads, downloads / name, contents


@pytest.mark.parametrize("cached_contents", [None, b"SourceForge HTML response"])
def test_source_prefetch_replaces_only_with_reviewed_bytes(source_cache, monkeypatch, cached_contents):
    project, downloads, cached, contents = source_cache
    if cached_contents is not None:
        cached.write_bytes(cached_contents)
    calls = []

    def download(url, *, timeout):
        calls.append(url)
        assert timeout > 0
        if cached_contents is None:
            assert not cached.exists()
        else:
            assert cached.read_bytes() == cached_contents
        return io.BytesIO(contents)

    monkeypatch.setattr(media_downloads.urllib.request, "urlopen", download)
    media_downloads.prefetch_media_sources(project, downloads)
    assert cached.read_bytes() == contents
    assert calls == ["https://www.mpg123.de/download/mpg123-1.33.4.tar.bz2"]
    assert list(downloads.iterdir()) == [cached]
    # A valid cache must also work without network access.
    media_downloads.prefetch_media_sources(project, downloads)
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["hash", "network", "truncated"])
def test_source_prefetch_checks_the_fallback_archive(source_cache, monkeypatch, failure):
    project, downloads, cached, contents = source_cache
    calls = []

    def download(url, *, timeout):
        calls.append(url)
        if len(calls) == 1:
            if failure == "network":
                raise urllib.error.URLError("connection reset")
            if failure == "truncated":
                raise http.client.IncompleteRead(b"partial archive", 10)
            return io.BytesIO(b"wrong upstream bytes")
        return io.BytesIO(contents)

    monkeypatch.setattr(media_downloads.urllib.request, "urlopen", download)
    media_downloads.prefetch_media_sources(project, downloads)
    assert cached.read_bytes() == contents
    assert calls == list(media_downloads._SOURCE_MIRRORS[cached.name])
    assert list(downloads.iterdir()) == [cached]


def test_source_prefetch_never_caches_unreviewed_downloads(source_cache, monkeypatch):
    project, downloads, cached, _ = source_cache
    cached.write_bytes(b"old invalid cache")
    monkeypatch.setattr(media_downloads.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"HTML"))
    with pytest.raises(RuntimeError, match="source SHA512 does not match"):
        media_downloads.prefetch_media_sources(project, downloads)
    assert cached.read_bytes() == b"old invalid cache"
    assert list(downloads.iterdir()) == [cached]

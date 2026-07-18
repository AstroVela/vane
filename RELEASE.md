# Release process

Vane releases are immutable source and binary artifacts derived from a reviewed Git commit. A GitHub branch or generated archive is not a release.

## Prepare

1. Open a release pull request that sets the PEP 440 version, updates `CHANGELOG.md`, and records the exact `external/duckdb` commit.
   If the engine source changed, update `OVERRIDE_GIT_DESCRIBE` in
   `pyproject.toml` and the matching record in `SOURCE_PROVENANCE.md`.
2. Confirm that every imported dependency has compatible terms and that `SOURCE_PROVENANCE.md`, `THIRD_PARTY.md`, `LICENSE`, and `NOTICE` are current.
3. Install the pinned vcpkg manifest and run `python scripts/sync_vcpkg_licenses.py --check`.
4. Run formatting, fast tests, the relevant slow/native tests, and the package-artifact workflow.
5. Review security-sensitive changes, known issues, and the supported platform statement.

## Build and inspect

Build from a clean checkout. The DuckDB source is part of the checkout. Release automation must:

- build the sdist first;
- validate it with `scripts/check_release_artifacts.py`;
- build wheels from that exact sdist in clean manylinux environments;
- validate wheel metadata, contents, `RECORD`, and license files;
- install each wheel in a fresh environment and run the Quickstart smoke test;
- produce SHA-256 checksums, a software bill of materials, build provenance, and Sigstore signatures.

Manually inspect the archive file list. TPC-H, TPC-DS, TPC-E tools, local paths, credentials, caches, logs, model weights, and build directories must not be present.

## Stage and publish

1. Publish the candidate to TestPyPI through the protected `testpypi` GitHub environment.
2. Install it by version in a clean machine and run smoke tests without access to the source checkout.
3. Record maintainer approval according to [GOVERNANCE.md](GOVERNANCE.md).
4. Tag the reviewed commit as `vX.Y.Z` and create release notes from `CHANGELOG.md`.
5. Publish to PyPI only through its trusted-publishing workflow and protected `pypi` environment. Do not use a long-lived PyPI token.
6. Attach checksums, SBOM, signatures, and provenance to the GitHub release, then verify that public artifacts reproduce the candidate hashes.

If any artifact is wrong, publish a new version. Never replace an existing release file or move a published tag.

## After release

- Verify installation and metadata from the public index.
- Move the changelog entries to the released version and open a new `Unreleased` section.
- Announce material known issues and security limitations.
- Retain enough logs and provenance to audit how the artifacts were produced.

## One-time repository configuration

Before the first publication, a repository administrator must confirm that
`main` is the GitHub default branch. Protect it: require pull requests, review,
passing CI and code-quality checks, resolved conversations, and block
force-pushes and deletion.

Enable private vulnerability reporting, the dependency graph, Dependabot
alerts, and code scanning. Create protected `testpypi` and `pypi` GitHub
environments with required maintainer approval, then register this repository's
`release.yml` workflow as a trusted publisher on the corresponding package
index. Publishing jobs intentionally contain no API-token fallback.

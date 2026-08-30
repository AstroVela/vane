# Release process

Vane releases are immutable source and binary artifacts derived from one
reviewed commit on `main` or a `release/X.Y` maintenance branch. The GitHub
Release body is the canonical public record of user-visible changes; the
repository does not maintain a rolling changelog or a release-notes template.

## Release invariants

- The package version is a valid, previously unused PEP 440 version. Its tag is
  exactly `v<version>` and points to the reviewed commit tested for release.
- An `X.Y.0` release, including its prereleases, comes from `main`. Patch
  releases and post releases come from the matching `release/X.Y` branch.
- One workflow run builds the sdist and all wheels once. The exact same files
  are promoted through TestPyPI, PyPI, and the GitHub Release.
- A GitHub Release remains a draft until PyPI publication succeeds and all
  checksums, signatures, the SBOM, and provenance are attached.
- A release tag cannot be updated or deleted from the moment it is created,
  including while its GitHub Release is still a draft. Published releases and
  artifacts are never replaced. A bad release is superseded by a new version
  and yanked when necessary.

## Version calculation

Python package versions come from Git through `setuptools-scm`; there is no
manually maintained version in `pyproject.toml`.

- On `main`, development versions count from the latest patch-zero final or
  prerelease tag. Commits after `v0.1.0` are `0.2.0.devN`, commits after
  `v0.2.0rc1` are `0.2.0rc2.devN`, and commits after the final `v0.2.0` start
  `0.3.0.devN`. Restricting the baseline to patch-zero tags prevents a merged
  maintenance or post-release tag from resetting `N`.
- On `release/X.Y`, versions count from the latest tag on that release line and
  use the next patch series: commits after `v0.2.0` are `0.2.1.devN`, and
  commits after `v0.2.1` are `0.2.2.devN`.
- Clean exact tags produce the tag version without a development suffix. The
  release workflow validates the selected tag and supplies that exact version
  to the isolated source build.

Build from a full Git checkout with release tags available, or from the sdist
produced by that checkout. For local feature branches based on a maintenance
line, set `VANE_VERSION_BRANCH=release/X.Y`; pull-request CI infers the same
line from its base branch.

## One-time repository configuration

Repository administrators must:

1. Keep `main` as the default branch and protect it with pull-request review,
   required CI and code-quality checks, resolved conversations, and deletion
   and non-fast-forward update protection. Apply the same protections to every
   active `release/X.Y` branch.
2. Create an active tag ruleset with no bypass actors for `refs/tags/v*`. Allow
   initial tag creation, but restrict deletion and block force pushes so the
   tag cannot move between workflow dispatch and publication. GitHub immutable
   releases provide an additional lock only after the draft is published.
3. Enable private vulnerability reporting, the dependency graph, Dependabot
   alerts and security updates, secret scanning with push protection, and code
   scanning.
4. Configure the `RELEASE_ARTIFACT_CONTENT_RULES` repository secret used by
   trusted artifact validation.
5. Create protected `testpypi` and `pypi` GitHub environments. The `testpypi`
   environment accepts only the protected `main` branch and `v*` tags; the
   `pypi` environment accepts only `v*` tags. Both require maintainer approval
   and disallow administrator bypass.
6. Register `.github/workflows/release.yml` as a trusted publisher for the
   `vane-ai` project on TestPyPI and PyPI, using the matching `testpypi` and
   `pypi` environment names. Publishing intentionally has no API-token fallback.
7. After the draft-first workflow has passed a build-only dry run, enable
   GitHub immutable releases. Draft releases must remain mutable so the
   workflow can attach their assets before publication.

Review these settings before every release rather than assuming the one-time
configuration has remained unchanged.

## Publish a TestPyPI development candidate

Use a development candidate to qualify exact cross-package dependencies before
the next Vane release exists. Dispatch the release workflow from the protected
`main` branch without creating a tag:

```bash
gh workflow run release.yml \
  --repo AstroVela/vane \
  --ref main \
  -f operation=testpypi-dev
```

The workflow accepts only the canonical PEP 440 development version derived
from that exact `main` commit and rejects a version already present on
TestPyPI. It builds, validates, attests, and signs the same complete
distribution set as a release, then publishes and clean-installs it through
the protected `testpypi` environment. It does not publish to PyPI, create a
tag, or create or modify a GitHub Release. A development candidate is
immutable and is never promoted; if it is unsuitable, merge a fix and publish
the new commit's development version.

## Prepare the release pull request

1. Choose a final canonical PEP 440 version with an `X.Y.Z` release segment.
   Do not edit `pyproject.toml`; the release tag is the source of the final
   version.
2. Put the complete proposed GitHub Release notes in the pull-request
   description. Reviewers approve those notes together with the code; do not
   depend on an unreviewed local notes file.
3. Record the exact `external/duckdb` tree ID reported by
   `python scripts/sync_duckdb_source_id.py --print` and the full fork revision
   reported by
   `python scripts/resolve_duckdb_fork_version.py --print-revision` in the pull
   request. Update `DUCKDB_UPSTREAM_VERSION` and `SOURCE_PROVENANCE.md` only
   when the imported upstream baseline, DuckDB version line, or historical
   mapping changes.
4. Confirm that imported dependencies have compatible terms and that
   `SOURCE_PROVENANCE.md`, `THIRD_PARTY.md`, `LICENSE`, and `NOTICE` are current.
5. Install the pinned vcpkg manifest and run
   `python scripts/sync_vcpkg_licenses.py --check`.
6. Run formatting, fast and release tests, relevant slow and native tests, and
   the build-only release workflow. Manually inspect the sdist and wheel file
   lists. TPC-H, TPC-DS, TPC-E tools, local paths, credentials, caches, logs,
   model weights, and build directories must not be present.
7. Triage security findings. A release must not carry an unexplained
   first-party critical or high-severity alert. Record the disposition of
   inherited and third-party findings that affect shipped code.
8. Confirm that the version is absent from both TestPyPI and PyPI, review known
   issues and the supported-platform statement, merge an `X.Y.0` pull request
   into `main` or a patch/post pull request into `release/X.Y`, and record its
   exact commit SHA.

Run a build-only dry run from the reviewed commit with:

```bash
gh workflow run release.yml \
  --repo AstroVela/vane \
  --ref main \
  -f operation=build-only
```

For a maintenance release, replace `main` with the matching `release/X.Y`.

## Create the tag and draft release

1. Confirm the recorded `X.Y.0` release commit is reachable from `main`, or the
   recorded patch/post release commit is reachable from the matching
   `release/X.Y` branch. It must not have changed since the release pull request
   passed its gates.
2. Confirm the active `v*` tag ruleset restricts deletion and force pushes and
   has no bypass actors. Create and push the exact `v<version>` tag at the
   recorded commit, then verify that the remote tag resolves to that commit.
   Never create the tag from an unreviewed working tree.
3. Create a draft GitHub Release for the protected tag. Copy the approved notes
   into its body without changing their meaning, and mark alpha, beta, and
   release-candidate versions as prereleases.
4. Manually dispatch the Release workflow using that tag as the workflow ref
   and `operation=release`.

The workflow rejects a release operation unless the selected ref is the
matching version tag, the tagged commit is reachable from the branch required
by its version line, the GitHub Release exists and is still a draft, and the
version is absent from both package indexes.

## Build, stage, and publish

Release automation must:

- build the sdist first, letting the PEP 517 backend inject
  `DUCKDB_SOURCE_ID` and `DUCKDB_FORK_REVISION`;
- validate it with `scripts/check_release_artifacts.py` and the private content
  rules;
- build all manylinux wheels from that exact sdist in clean environments;
- validate wheel metadata, contents, `RECORD`, license files, and coexistence
  with the official DuckDB package;
- install each wheel in a fresh environment and run the Quickstart smoke test;
- generate SHA-256 checksums, a CycloneDX SBOM, GitHub build provenance, and
  Sigstore signatures;
- publish those distributions to TestPyPI through its protected environment;
- install the indexed candidate in a clean job without a source checkout and
  run the public smoke test;
- wait for explicit approval on the protected `pypi` environment;
- publish the same distributions to PyPI, attach all artifacts to the draft
  GitHub Release, and only then publish the draft.

Do not approve the `pypi` environment until the automated TestPyPI job has
passed and a maintainer has independently installed the exact version in a
clean machine, run the public Quickstart, reviewed artifact contents, and
recorded approval according to `GOVERNANCE.md`.

## Verify the public release

After publication:

1. Install `vane-ai==<version>` from PyPI without access to the source checkout
   and run the Quickstart.
2. Download the GitHub Release assets, run `sha256sum -c SHA256SUMS`, and confirm
   the PyPI files have the same hashes.
3. Verify build provenance for every distribution:

   ```bash
   gh attestation verify <distribution> --repo AstroVela/vane
   ```

4. Confirm the GitHub Release is immutable, the tag points to the recorded
   release commit, PyPI metadata and provenance are correct, and public
   documentation links resolve.
5. Announce material known issues and security limitations, and retain the
   workflow run, approvals, checksums, provenance, and review record for audit.

## Respond to a bad release

If a problem is found before the release tag is created, cancel the workflow
and correct the release pull request. Once a `v*` tag exists, do not move,
delete, or reuse it even if neither package index has received the version;
abandon the draft and supersede it with a new version and tag. For a harmful
PyPI release, yank it, publish a fixed version, and add a clear notice to the
affected GitHub Release. Use the security-advisory process when confidentiality
or coordinated disclosure is required.

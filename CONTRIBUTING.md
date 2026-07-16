# Contributing to the Vane engine fork

Most changes should begin in the parent [Vane repository](https://github.com/AstroVela/vane), where the Python API, package build, and submodule revision are tested together.

For a small engine fix, open a focused pull request. Discuss protocol changes, large refactors, new dependencies, storage-format changes, and broad upstream integrations in an issue first.

## Pull-request expectations

- Explain the root cause, design, compatibility impact, and test evidence.
- Add a focused C++ or SQLLogicTest regression test.
- Record the immutable upstream commit for cherry-picked or adapted code.
- Preserve authorship, MIT notices, and third-party license files.
- Keep generated files, build output, model artifacts, credentials, and local paths out of commits.
- Do not mix a submodule pointer update with unrelated parent-repository work.

Contributions intentionally submitted to this repository are accepted under its MIT license unless explicitly stated otherwise.

## Source license headers

New and modified source files in this fork use `SPDX-License-Identifier: MIT`. Modified upstream files retain the DuckDB Foundation copyright, add the Vane contributors' copyright, and include `Modified by Vane contributors.` Preserve existing third-party headers and do not mechanically relabel vendored dependencies or generated output.

From the parent Vane checkout, validate the applicable source files with:

```bash
python3 scripts/check_source_license_headers.py --repo duckdb
```

## Build and test

```bash
cmake -S . -B build -G Ninja
cmake --build build --parallel
build/test/unittest
```

Run the narrowest relevant test during development, followed by the appropriate full suite before merge. Changes used through Python must also be built and tested from the parent Vane repository.

## Formatting

This fork retains DuckDB's formatting conventions. The parent checkout provides a wrapper:

```bash
scripts/format submodule --changed
```

The formatter requires Black 24+, clang-format 11.0.1, and cmake-format.

## AI-assisted changes

The contributor remains accountable for AI-assisted code. Review every change, verify provenance and license compatibility, avoid confidential inputs, run tests, and disclose material generated code or assets when it helps reviewers assess risk.

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Report vulnerabilities privately according to [SECURITY.md](SECURITY.md).

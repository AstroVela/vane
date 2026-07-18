# Contributing to Vane

Thank you for helping improve Vane. The project welcomes bug reports, design discussions, documentation, tests, and code from contributors of all experience levels.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Security vulnerabilities must be reported privately as described in [SECURITY.md](SECURITY.md).

## Before opening a change

For a small fix, open a pull request directly. For a new public API, protocol change, large dependency, execution-model change, or substantial refactor, open an issue first so scope and compatibility can be discussed.

Search existing issues and pull requests before starting. If an issue exists, comment before investing in a large implementation; an assignment is coordination, not ownership of an idea.

## Pull requests

Keep each pull request focused and explain:

- the user-visible problem and root cause;
- the chosen design and important alternatives;
- tests run and any untested environment;
- compatibility, performance, security, and license impact;
- source and immutable revision of any imported code or generated asset.

Add regression tests for behavior changes. Update public documentation and `CHANGELOG.md` when users need to know about the change. Avoid drive-by formatting or unrelated cleanup.

Contributions intentionally submitted to this repository are accepted under Apache-2.0 unless explicitly stated otherwise. Preserve existing third-party notices and update [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md) and [THIRD_PARTY.md](THIRD_PARTY.md) when importing or modifying inherited code.

## Source license headers

Use short SPDX headers on source files covered by Vane's provenance policy:

- New Vane source uses `SPDX-License-Identifier: Apache-2.0`.
- A file in the parent repository that modifies DuckDB or DuckDB Python client source retains the DuckDB copyright, adds the Vane contributors' copyright, uses `SPDX-License-Identifier: MIT AND Apache-2.0`, and includes `Modified by Vane contributors.`
- New and modified source inside `external/duckdb` uses that repository's MIT license.

Do not replace an existing third-party license header or add a Vane header to unchanged upstream source, vendored dependencies, or generated output. Check the applicable files from the parent checkout with:

```bash
python3 scripts/check_source_license_headers.py
```

## Development checks

Follow [DEVELOPMENT.md](DEVELOPMENT.md). The minimum expected checks are:

```bash
scripts/format root --changed
scripts/run_release_tests.sh
```

C++ changes require an incremental native build. Changes inside
`external/duckdb` must use DuckDB's formatter and relevant engine tests. Keep
engine changes focused within the subtree, and explain any imported baseline or
Vane-specific engine changes in the pull request.

## AI-assisted contributions

AI tools may assist a contribution, but the human contributor remains responsible for every line and claim. Do not submit confidential data to a tool. Review generated changes, verify provenance and license compatibility, run the relevant tests, and disclose material AI-generated code or assets in the pull request when it helps reviewers evaluate risk.

## Review and merge

Maintainers review correctness, scope, maintainability, compatibility, security, provenance, and test evidence. Approval is not guaranteed, and a maintainer may ask that a large change be split. Decisions follow [GOVERNANCE.md](GOVERNANCE.md).

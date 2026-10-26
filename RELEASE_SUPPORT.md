# Releases and support

Use this guide to choose a Vane release and assess its compatibility and
maintenance boundaries. The canonical inventory is
[release-support.json](release-support.json); release notes live in
[GitHub Releases](https://github.com/AstroVela/vane/releases).

## Current release status

Reviewed on **2026-10-26** against published release assets and the release tag.

| Selection | Version | Meaning |
| --- | --- | --- |
| Latest | [0.2.0](https://github.com/AstroVela/vane/releases/tag/v0.2.0) | Most recent published release recorded in the inventory |
| Recommended for evaluation | `0.2.0` | A pinned starting point for workload and environment evaluation |
| Stable / recommended for production | None designated | No release is currently recorded with both qualification evidence and a maintenance commitment |
| Development | `main` | Unreleased changes; pin a commit when evaluating them |

Vane `0.2.0` is an **alpha developer preview**, even though its GitHub release is
not marked as a prerelease. A version number or successful CI run alone does not
establish production readiness. Stable designation requires a recorded
qualification report and an explicit maintenance policy. These labels describe
the published evidence; they do not guarantee suitability for every workload.

## Published platforms

The `0.2.0` release contains these base-package wheels:

| Operating system / architecture | Python | Published wheel tag |
| --- | --- | --- |
| Linux x86-64, glibc 2.28 or newer | CPython 3.10, 3.11, 3.12, 3.13, 3.14 | `manylinux_2_28_x86_64` |

Exact filenames and SHA-256 checksums are recorded in the inventory and can be
compared with the GitHub release assets. No macOS, Windows or Linux arm64 wheel
is recorded for `0.2.0`. Development-branch build or CI support for another
platform does not mean a released wheel is available for it. Published base
wheels also do not qualify a GPU, model, optional extension or external service.

## Extension compatibility

`vane-ai` and optional `vane-extension-*` provider packages have separate
versions and release cycles. Installing the base package does not install the
lakehouse connectors or the native-media provider described in the
[0.2.0 release notes](https://github.com/AstroVela/vane/releases/tag/v0.2.0).

Select a provider qualified for the **exact Vane runtime and extension
descriptor**, including its engine source identity and platform/ABI requirements.
Matching only the DuckDB upstream version, or choosing the newest version of
each package independently, is insufficient. Pin the base and provider wheels
together and use the same versions on the driver and all workers.

The inventory currently records **no qualified provider-wheel combinations**.
This means it does not yet provide the wheel identities and verification
evidence needed for a compatibility recommendation; it does not say that the
providers are unavailable or incompatible. Consult the provider's own release
instructions and verify the exact combination in the target environment.

## Upgrades and recovery

No verified version-to-version upgrade path is currently recorded. In
particular, the inventory makes no promise of rolling upgrades, mixed-version
clusters, or downgrade compatibility.

Distributed plans and task results use a same-version contract. Before an
upgrade, drain in-flight queries, update the driver and all workers together,
and recreate persistent Ray worker actors. This requirement is documented in
the [0.2.0 transport contract](https://github.com/AstroVela/vane/blob/v0.2.0/SECURITY.md#plaintext-flight-transport).

Keep the previous pinned environment and test representative reads, writes and
failure recovery in a separate environment before switching production work.
Restoring a package version does not undo external writes, schema changes or
model/API charges. Reconcile the destination's state before replaying a failed
job; the release inventory does not establish sink-specific replay safety.

## Maintenance and known limitations

For `0.2.0`, no fixed support window, release-branch backport policy or LTS
commitment is announced in this inventory. `maintenance_until: null` means
**no end date has been committed**, not indefinite support. Check the release
notes for fixes included in a published version and the
[security policy](SECURITY.md) for vulnerability reporting and security-fix
coverage. The existing best-effort policy for the latest `0.1.x` prerelease is
unchanged; it does not imply a maintenance commitment for `0.2.0`.

The recorded limitations link to release-specific evidence:

- A Ray cluster is one trusted computing boundary. Plaintext Flight is intended
  for an isolated, trusted cluster; see the
  [0.2.0 security policy](https://github.com/AstroVela/vane/blob/v0.2.0/SECURITY.md#trust-model).
- Mixed-version distributed execution is unsupported; see the transport
  contract linked above.
- Native media and lakehouse extensions are separate provider packages, as
  described in the release notes.

This is not an exhaustive defect list. Review the release notes and
[open bug reports](https://github.com/AstroVela/vane/issues?q=is%3Aissue%20is%3Aopen%20label%3Abug)
when assessing a particular workload. Maintainers update this guide and its
inventory through the [release process](RELEASE.md#release-support-inventory).

# Changelog

All notable user-visible changes are documented here. Vane is currently in alpha, so incompatible changes may occur between prereleases.

## Unreleased

### Added

- Public governance, contribution, security, release, provenance, and third-party documentation.
- Release artifact validation and a reproducible native dependency license bundle.

### Changed

- Moved the forked DuckDB Python API under `vane`, renamed its native module,
  stubs, and ADBC package to Vane-owned names, and made installation and
  uninstallation safe alongside the official `duckdb` distribution.
- Documented and tested the one-time DuckDB reinstall required when upgrading
  an environment containing the conflicting `vane-ai 0.1.0a1` wheel.
- Positioned the current project as the Vane Data developer preview.
- Defined `DuckDBPyRelation.map` exclusively as a row-wise scalar UDF with a
  required `return_type`; batch transforms use `map_batches` with an explicit
  output `schema`. The inherited pandas DataFrame-style DuckDB `map` contract
  is no longer supported.
- Restricted source distributions to the DuckDB components required by Vane.
- Imported the official DuckDB baseline as a squashed Git subtree and retained
  Vane engine customizations as monorepo commits, so normal clones no longer
  require submodule initialization or carry DuckDB's complete commit history.

### Security

- Documented the trust boundaries around Python UDFs, Ray workers, credentials, native parsers, and remote model code.
- Redacted AI provider credentials from descriptor and provider-option `repr`,
  logs, exception formatting, and assertion diffs; plaintext is revealed only at
  provider execution, and SQL continues to reject inline credentials. Option
  mappings held by AI descriptors now store sensitive values wrapped in an
  internal secret type, so code that compared those mappings against plain
  dictionaries must compare revealed values instead.

## 0.1.0a1

Initial public alpha release.

# Changelog

All notable user-visible changes are documented here. Vane is currently in alpha, so incompatible changes may occur between prereleases.

## Unreleased

### Added

- Public governance, contribution, security, release, provenance, and third-party documentation.
- Release artifact validation and a reproducible native dependency license bundle.

### Changed

- Positioned the current project as the Vane Data developer preview.
- Restricted source distributions to the DuckDB components required by Vane.
- Imported the official DuckDB baseline as a squashed Git subtree and retained
  Vane engine customizations as monorepo commits, so normal clones no longer
  require submodule initialization or carry DuckDB's complete commit history.

### Security

- Documented the trust boundaries around Python UDFs, Ray workers, credentials, native parsers, and remote model code.

## 0.1.0a1

First planned public alpha release. Not yet published.

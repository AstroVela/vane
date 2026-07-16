# Vane DuckDB engine fork

This repository is the native engine fork used by [Vane](https://github.com/AstroVela/vane). It is based on [DuckDB](https://github.com/duckdb/duckdb) and adds Vane-specific distributed planning, execution, exchange, fault-tolerance, and Python integration support.

It is not a general-purpose DuckDB distribution. Most users should install and use the parent Vane project rather than build this repository directly.

> [!IMPORTANT]
> This is an independent fork. It is not affiliated with, endorsed by, or maintained by the DuckDB Foundation. For Vane bugs and support, use the Vane issue tracker rather than DuckDB's community channels.

## Relationship to upstream

The fork periodically integrates selected upstream DuckDB changes while maintaining Vane's engine extensions. Pull requests that import upstream work must record the upstream commit and preserve its authorship and license notices.

DuckDB and this engine repository are distributed under the MIT license in [LICENSE](LICENSE). Other code below `third_party/` and selected extension directories retains its own license.

## Build

For an engine-only debug build:

```bash
cmake -S . -B build -G Ninja
cmake --build build --parallel
```

Run a named test or the full native test suite:

```bash
build/test/unittest "test name" -s
build/test/unittest
```

The parent Vane package has additional Arrow Flight, gRPC, Python, and packaging prerequisites. See the parent repository's [development guide](https://github.com/AstroVela/vane/blob/main/DEVELOPMENT.md) for the supported build path.

## Contributing and security

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before participating. Report vulnerabilities through the private process in [SECURITY.md](SECURITY.md).

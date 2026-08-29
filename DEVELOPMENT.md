# Development

Vane contains Python, pybind11, and a modified DuckDB C++ engine. A native build also links Arrow Flight, gRPC, and selected DuckDB extensions.

## Prerequisites

- Linux x86-64 for the currently tested path
- Python 3.10 through 3.14; Python 3.12 is recommended and is the primary development version
- Git with `git subtree` support
- A C++20 compiler, CMake 3.29+, Ninja, and ccache
- vcpkg at the baseline pinned in `vcpkg.json`

The DuckDB engine fork is included directly under `external/duckdb`; a normal
clone contains all source needed for the build.

Bootstrap native dependencies from the repository root:

```bash
bash scripts/bootstrap_vcpkg.sh
```

The helper checks out the exact baseline from `vcpkg.json`, installs into
`vcpkg_installed`, and verifies the committed native-dependency license bundle.
When intentionally changing native dependencies, regenerate the bundle with
`python scripts/sync_vcpkg_licenses.py` and review its diff.

## Incremental package build

Create and activate a virtual environment, then reuse a persistent native build directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

export SKBUILD_BUILD_DIR="$PWD/build/python-release"
export SKBUILD_CMAKE_BUILD_TYPE=Release
python -m pip install . --no-build-isolation -v
```

Do not use `pip install -e`. An editable install can cause Ray workers to invoke the build backend during import and delay actor startup.

Python-only changes do not require a native rebuild, but reinstall the
non-editable package so the test environment receives them. Changes below
`src/vane_py/` or `external/duckdb/src/` require an incremental native build.

## Building a loadable extension artifact

`VANE_LOADABLE_EXTENSIONS` builds selected DuckDB extensions as self-contained
`.duckdb_extension` artifacts without linking them into `vane._native`. The
default is empty, so base Vane builds and wheels do not contain staged optional
extensions. DuckDB's pinned source configuration is preserved for external
extensions such as `httpfs`. For example, build and exercise both the in-tree
`tpch` artifact and the externally sourced `httpfs` artifact:

```bash
export SKBUILD_BUILD_DIR="$PWD/build/python-release"
export SKBUILD_CMAKE_BUILD_TYPE=Release
uv pip install . --no-build-isolation \
  '-Ccmake.define.VANE_LOADABLE_EXTENSIONS=tpch;httpfs'
cmake --build "$SKBUILD_BUILD_DIR" --target vane_loadable_extensions
VANE_TEST_LOADABLE_EXTENSION_PATH=\
"$SKBUILD_BUILD_DIR/vane_extensions/tpch.duckdb_extension" \
VANE_TEST_LOADABLE_HTTPFS_EXTENSION_PATH=\
"$SKBUILD_BUILD_DIR/vane_extensions/httpfs.duckdb_extension" \
  scripts/run_installed_pytest.sh tests/fast/test_loadable_extension_artifacts.py
```

Loadable artifacts require `EXTENSION_STATIC_BUILD=ON`. This keeps each
artifact self-contained and preserves Vane's private `_native` symbol boundary.
The staging directory is configurable with
`VANE_LOADABLE_EXTENSION_OUTPUT_DIRECTORY`.

## Building an optional extension wheel

The base `vane-ai` wheel must stay free of optional `.duckdb_extension`
artifacts. Package one already-staged, release-approved artifact into a
separate platform wheel instead:

```bash
python -I scripts/build_extension_wheel.py \
  --artifact "$SKBUILD_BUILD_DIR/vane_extensions/<extension>.duckdb_extension" \
  --extension-name <extension> \
  --platform-tag manylinux_2_28_x86_64 \
  --trust-identity astrovela/vane \
  --license-expression "Apache-2.0 AND MIT" \
  --license-file LICENSE \
  --license-file NOTICE \
  --license-file LICENSES/DuckDB-MIT.txt \
  --output-directory dist/extensions
```

The generated wheel has an exact `vane-ai` dependency, embeds a versioned
descriptor, and publishes a `vane.dynamic_extension_providers` entry point.
Its public PEP 440 version encodes the Vane release stage followed by the
complete descriptor SHA-256 split across bounded 32-bit numeric components.
This keeps two artifact identities for the same Vane release distinct without
local versions or oversized numeric fields. The generated provider module is
content-addressed as well, so distinct artifacts do not own the same installed
package path. Its compatibility tag is scoped to the active supported CPython
minor as `cpXY-none-<platform>`; it does not advertise cross-interpreter
installability that a single native `vane-ai` wheel cannot satisfy. Pass the
complete transitive closure of dependency wheels in load order with
`--dependency-wheel`; the builder reads and validates each embedded
descriptor, complete descriptor-bound requirements, provider entry point,
licensing, generated core metadata version and WHEEL metadata, native footer
identity, exact owned archive layout, portable path lengths, RECORD, and
platform tag; retains that exact order in the root descriptor; and pins each
corresponding extension wheel by its descriptor-bound exact version. Every
dependency wheel must use the same CPython interpreter tag as the root. It must
also have a platform policy at least as broad as the wheel that depends on it
(for example, a
`manylinux_2_28` wheel cannot depend on a `manylinux_2_39` wheel), advertise the
same exact supported `Requires-Python` range, and stay within the
extension-wheel publication size limit.
Every unique signer used by a dependency wheel must be named explicitly with
a repeated `--dependency-trust-identity`. The supplied allowlist must match
the dependency closure exactly; neither the builder nor clean verifier derives
trust from the wheels being checked.
For Linux wheels, the builder independently inspects every root and dependency
extension ELF header, bounded program and dynamic tables, `DT_NEEDED` library
names, dynamic symbols, GNU version-needed entries, and linked libc
requirements. pyelftools interprets the preflight-bounded dynamic symbol
entries using the same undefined/non-weak semantics as auditwheel. The ELF
machine must match the declared architecture. A manylinux object must satisfy
the exact complete-version symbol allowlist and each `DT_NEEDED` library's
undefined-symbol blacklist from the pinned auditwheel policy; the highest
major/minor glibc requirement is computed separately for display/compatibility
reporting. Unknown versioned-symbol namespaces are rejected, and glibc and
musl artifacts cannot be relabeled as each other.
Every external library must be allowed by the exact
manylinux or musllinux policy; extension wheels cannot rely on private
libraries present only on the publisher host.
The build command uses the installed Vane runtime to create that descriptor, so
run it after installing Vane with the development build procedure above.
Supply every license required by the selected artifact explicitly; the builder
does not infer licenses or reuse the base wheel's metadata. Supply a valid,
corresponding SPDX expression with `--license-expression` as well.
Its platform tag must match the platform embedded in the extension artifact;
the builder rejects a mismatched OS, architecture, or Linux libc-family tag.
For glibc Linux artifacts, use the narrowest truthful `manylinux_*` policy tag
for the build environment. Generic `linux_*` tags are rejected because pip can
install them on musl systems even though DuckDB identifies those artifacts as
glibc-only.
Linux policy version components must use canonical decimal spelling without
leading zeroes. Manylinux policies below glibc 2.5 on x86-64 or glibc 2.17 on
AArch64 are not installable and are rejected. A musllinux root, base wheel, and
every dependency wheel in its graph must use the same exact musl policy. The
root wheel must be built on the exact musl
major/minor baseline named by its tag. The builder records that detected
baseline in canonical, RECORD-bound platform build details; dependency-wheel
and clean verification require those details to match the wheel tag. Clean
verification must itself run on that exact minimum musl runtime before it
installs and loads the graph. The clean verifier also requires the supplied base wheel to
have no build tag or platform-neutral `any` tag, carry WHEEL compatibility tags
that exactly match its filename tags, use the extension wheel's exact CPython
interpreter and ABI tags, declare exactly `Wheel-Version: 1.0` and
`Root-Is-Purelib: false`, advertise the exact supported Python range, use the
generated core metadata version, stay within the publication size limit, pass
the base release-artifact owned layout, RECORD, and complete project dependency
and entry-point metadata checks, and have a platform policy that covers the root
extension wheel's policy for every advertised platform tag. Generic `linux_*`
base tags are rejected because they do not declare a libc family or minimum
version. The verifier independently applies the same ELF architecture, dynamic
dependency, GNU version-needed, and libc checks to the base native module and
every other ELF member. `DT_NEEDED`, `DT_FILTER`, and `DT_AUXILIARY` entries all
use the exact platform external-library policy. Non-policy dependencies are
rejected even if the archive contains a same-named file, so matching filename,
WHEEL, and RECORD tags cannot relabel a binary built for a newer system library
or hide a host-only dependency. RPATH/RUNPATH, audit/config, and no-default-lib
loader settings are rejected because they could redirect an allowlisted name
to a host-specific library. CI
repairs its test-only base wheel with auditwheel before clean
extension verification; release base wheels already come from the configured
manylinux build.

The manylinux allowlists are an unmodified snapshot of auditwheel 6.8.1 from
commit `94e0693e0fcb444c7fe50f09a8a635e791be6174`. The bundled
`vane_packaging/_vendor/auditwheel/manylinux-policy.json` must have SHA-256
`104863eb197685edf6407a51ccde6cbd906be736efb959a991a60d102f1ccf96`.
To update it, copy the policy and license from one immutable auditwheel tag,
update the version, commit, and digest constants in
`vane_packaging/manylinux_policy.py`, review the complete policy diff, and run
the policy parity and Linux extension-wheel tests. Never edit the snapshot to
make a particular artifact pass; choose a truthful tag or update to a reviewed
upstream auditwheel policy instead.

macOS 11 and later tags must use a zero minor version, Arm64 tags require macOS
11 or later, and x86-64 macOS 10 tags are limited to the architecture-specific
tags emitted for 10.4 through 10.16. The builder and verifier inspect each
extension Mach-O's architecture, generic CPU subtype, shared-object file type,
dynamic-library and runtime-search-path load commands, and `LC_BUILD_VERSION`
or `LC_VERSION_MIN_MACOSX` deployment target. Dynamic dependencies must be
canonical system paths below `/usr/lib/` or `/System/Library/`; publisher-local,
`@rpath`, and wheel-relative dynamic-library dependencies are rejected rather
than inferred from the verification host. Any `LC_RPATH` entry must be a
canonical `@loader_path` or `@executable_path` path. The encoded minimum OS must
be covered by the wheel tag. The clean verifier applies the same checks to every
Mach-O member of the supplied base wheel. Legacy FVMLIB, prebound-library, and
dynamic-linker environment commands are rejected. The builder and verifier
also require Windows artifacts to be PE32+ DLLs with a COFF machine matching
the wheel architecture. Bounded section mappings, ordinary imports, and delay
imports are inspected without host DLL resolution. Export address tables are
also bounded and inspected, and forwarded exports are rejected so they cannot
introduce a dependency outside the import tables. Only the fixed system-DLL
policy, fixed UCRT API-set contracts, the exact CPython runtime DLL, and the VC
runtime shipped with CPython are accepted; publisher-local, lookalike API-set,
or path-qualified imports are rejected. Clean verification applies the same
checks to every PE member of the base wheel. The builder and verifier reject
extension wheels above the project's 100 MiB publication limit, any archive
member above a 100 MiB uncompressed limit, or an archive whose total decompressed
contents exceed that limit. The preflight runs before reading
dependency, root, or base wheel members. Before constructing Python's ZIP
reader, every supplied wheel also receives a streaming central-directory
preflight capped at 10,000 members. Archives with comments, spanning, ZIP64 end records, or
internally inconsistent counts and bounds are rejected. Untrusted release and
extension-wheel archive paths are opened once, confirmed to name regular
files, and copied with an explicit byte bound into a private temporary
directory. On POSIX the completed snapshot is made read-only. Raw-content
scanning, ZIP or TAR preflight, and the standard-library semantic parser
consume that same snapshot; clean extension verification also installs the
snapshotted root, dependency, and base wheels.
Clean verification retains at most 1 GiB of snapshot bytes across the complete
root, dependency, and base-wheel set. It validates the root and base snapshots,
then snapshots and validates each dependency in order, so an invalid earlier
artifact stops the verifier before later artifacts consume temporary storage.
The source path is never reopened after the snapshot boundary, so replacing it
cannot change the bytes being approved or installed. Snapshot files and their
private directories are closed and removed on success and on every failure
path. Their METADATA is capped at 1 MiB, 1,024 headers, 10,000 lines, and
bounded line length before the email metadata parser is constructed. The same
bounds apply to dependency, root, base-wheel, and generic release metadata
parsing.
Extension descriptors are capped at 64 KiB and receive bounded JSON nesting,
separator, string, scalar, and dependency-object preflights before a JSON
object is constructed; each packaged graph is limited to 256 dependency
descriptors in addition to its root descriptor.
Extension-wheel
`RECORD` rows are streamed, size-bounded before CSV parsing, and cannot
outnumber the preflight-bounded archive members. The builder applies the same
member cap to its complete generated member set, including `RECORD`, before it
creates either `RECORD` or the archive. On main and release branches,
CI also applies the private release-content rules to the completed extension
wheel before artifact upload.
The generic release-content scanner applies the same individual-member and
aggregate uncompressed limits to wheels and source archives before reading any
member contents. It applies the same wheel member-count preflight before
constructing its ZIP reader and parses raw 512-byte source-archive TAR headers
before constructing `TarFile` metadata. The TAR preflight bounds ordinary
payloads, PAX headers, GNU long-name headers, PAX record counts, and PAX size
overrides before advancing the decompressed stream across each payload. Every
payload alignment block must be zero-filled. After the required two-block TAR
terminator, the preflight consumes the decompressed stream to EOF and permits
only one bounded record of zero padding, so payload padding and concatenated
gzip members cannot hide unreferenced plaintext. Binary private-content rules scan
each decompressed TAR header and bounded PAX or GNU extension-header payload,
including matches split across streaming chunks. They also scan the
bounded raw artifact stream, including ZIP gaps and other bytes not exposed as
archive members; text-only rules retain their archive-member binary filter.
The staged `.duckdb_extension` must also be a regular file within the 100 MiB
uncompressed limit. The builder checks it before descriptor inspection and
performs a second file-descriptor check plus a bounded read before packaging.
License inputs use the same bounded reader, and an existing output wheel is
size-checked before it is compared with the generated wheel in bounded chunks.
The complete generated member set is checked against the total uncompressed
limit before a temporary wheel is created, including a second check after its
RECORD is generated.
Load locally by passing the wheel's provider explicitly to
`DynamicExtensionResolver`:

```python
from importlib import import_module
from importlib.metadata import entry_points

import vane
from vane.extensions import DynamicExtensionResolver

matches = [
    candidate
    for candidate in entry_points(group="vane.dynamic_extension_providers")
    if candidate.name == "<extension>"
]
if len(matches) != 1:
    raise RuntimeError(f"expected one installed provider, found {matches!r}")
installed_extension = import_module(matches[0].module)
local_provider = matches[0].load()()
connection = vane.connect()
DynamicExtensionResolver(
    trusted_identities={local_provider.trust_identity},
    providers=(local_provider,),
).load(connection, installed_extension.descriptor())
```

For an extension with dependencies, resolve each dependency provider through
the same entry-point group and include all of them in `providers` and
`trusted_identities`. Pass the same complete dependency-wheel closure to the
verifier in load order with repeated `--dependency-wheel` arguments.

The installed provider and resolver perform no network lookup at runtime.
Package installation remains standard pip behavior; for an offline deployment,
pre-stage the base wheel and every declared extension wheel and install from
that trusted, hash-locked local wheel set. Python package requirements select a
distribution version, not a wheel filename build tag or content hash. The
verifier rejects base build tags, and the resolver independently fails closed
unless the installed runtime has the descriptor's exact DuckDB SourceID.
Validate a base and extension wheel together in a clean environment with:

```bash
python -I scripts/verify_extension_wheel.py \
  --base-wheel dist/vane_ai-*.whl \
  --extension-wheel dist/extensions/vane_extension_<extension>-*.whl \
  --extension-name <extension> \
  --trust-identity astrovela/vane
```

For a dependency graph, repeat both `--dependency-wheel` in load order and
`--dependency-trust-identity` once for every unique signer in the closure.

The verifier creates its disposable virtual environment through an isolated
interpreter and removes inherited Python runtime controls from every child
process. It sets `PIP_CONFIG_FILE` to the platform null device before creating
the environment, so pip never reads global, user, site, or explicitly selected
configuration files during installation or dependency checking.
The verifier uses DuckDB's default signature policy; it never enables unsigned
extension loading. `tpch` remains an in-tree build/test artifact only: its
source has additional redistribution terms and it must not be published as an
extension wheel. #619 tracks the first release-approved Iceberg wheel.

Applications load a trusted local artifact through
`DynamicExtensionResolver.load()`. After DuckDB accepts the verified cache
snapshot, Vane records the artifact's canonical `DynamicExtensionDescriptor`
on the connection session. Ray snapshots preserve that ordered descriptor
manifest, including each SHA-256 digest and dependency identity, but never a
local artifact path or binary payload.

Every Ray node must install the same platform extension wheels before queries
start. Each wheel supplies one provider entry point under
`vane.dynamic_extension_providers`; its name is the canonical DuckDB extension
name and its callable returns a `LocalExtensionProvider` for the exact
descriptor. Fragment registration prepares the worker's isolated
DatabaseInstance from those providers before task admission. Immediately
before native admission, a worker refreshes the exact database identity (for
example after S3 credential rotation) and prepares a cache miss first; the
worker leases a cursor from that prepared entry before admission, so credential
rotation cannot retire it during handoff. The admitted execution path only uses
that cursor and never issues another dynamic-extension load. Missing, ambiguous, or
mismatched providers fail preparation; workers do not scan directories,
download artifacts, autoinstall, autoload, or enable unsigned loading.
Coordinator extension/home directory and extension repository bootstrap
settings are ignored on workers so verified caches stay node-local.

The real-Ray CI gate signs DuckDB's `loadable_extension_demo` with the
repository's existing mbedTLS test key, packages it with this wheel builder,
and installs that generated platform wheel in every fast-test shard. The base
Python 3.12 wheel used to build the test artifact explicitly enables
`VANE_ENABLE_TEST_EXTENSION_SIGNING_KEY`; the option is off by default and must
never be enabled for release artifacts.

## Native C++ tests

The complete native gate builds DuckDB, distributed exchange, and the test
runner with the same pinned Arrow and C++20 configuration used by CI. The
script starts from a fresh CMake configuration (`cmake --fresh`) to avoid
configuration drift, which triggers a clean rebuild in its build directory:

```bash
scripts/run_native_tests.sh "[distributed]"
```

Run a named engine test or the complete unit suite with the same build:

```bash
scripts/run_native_tests.sh "test name" -s
scripts/run_native_tests.sh
```

The build uses two parallel compile jobs by default to stay within standard CI
runner memory. Override that limit with `VANE_NATIVE_BUILD_JOBS` when the local
machine has more capacity.

Statically linked DuckDB extensions participate in Ray execution through the
explicit scan callback and write provider contracts described in
[DISTRIBUTED_EXTENSIONS.md](DISTRIBUTED_EXTENSIONS.md). Add engine-level
protocol tests and extension-specific normal and fault-tolerant tests when
implementing either contract.

## Python tests

The required release gate covers the supported base installation and does not
need model downloads, cloud credentials, GPUs, or external services:

```bash
scripts/run_release_tests.sh
```

Vane's native extension is private to the installed `vane` package. Test
launchers therefore run outside the checkout and put the installed
site-packages directory before repository support modules. This prevents the
source package from shadowing `vane._native` and ensures tests exercise the
same layout shipped in the wheel. Run an affected test through the same
installed-package wrapper:

```bash
scripts/run_installed_pytest.sh tests/fast/test_udf_process.py
```

The inherited compatibility suites are broader and require the development
dependency group. Run them when changing the corresponding integration:

```bash
scripts/run_fast_tests.sh
scripts/run_installed_pytest.sh tests/slow
scripts/run_installed_pytest.sh tests/ai
```

The fast-test launcher runs non-Ray tests, shared-cluster Ray tests, and
test-owned Ray clusters in separate pytest processes. This keeps the real Ray
runtime out of the long-lived non-Ray pytest process. Fast and release Ray test
clusters let Ray size the object store from the node's available memory by
default. `VANE_TEST_RAY_OBJECT_STORE_BYTES` pins the capacity for a specialized
test; it does not configure production clusters. Tests that call `ray.init()`
directly must be marked `real_ray` and `ray_cluster_owner`.

CI further splits the non-Ray phase across CPU-only jobs. The jobs install the
built wheel, use CPU-only PyTorch, and set hard pytest-process and job deadlines
so the suite fits a standard 4-vCPU, 16-GiB GitHub-hosted runner. Tests marked
`gpu` are excluded there because standard runners do not provide CUDA hardware;
run the default launcher on a GPU host to include them.

Tests that require an externally provisioned service are excluded by default.
Run them explicitly when the required service and credentials are available:

```bash
scripts/run_installed_pytest.sh -m external_service tests/fast
```

Other optional tests may require network access, model weights, GPUs, credentials, or a local Ray setup. Tests must
skip with a clear reason when an optional environment is absent; they must not silently use a maintainer's local
endpoint or credentials.

## Formatting and static checks

```bash
python -m pip install pre-commit
pre-commit install
scripts/format root --changed
pre-commit run --from-ref origin/main --to-ref HEAD
```

Run `pre-commit install` once per clone.

Add `--check` to verify formatting without modifying files. Use `workspace`
when both Vane-owned files and the DuckDB subtree have changed:

```bash
scripts/format workspace --changed --check
```

To check changes relative to a committed ref, including in CI, use:

```bash
scripts/format workspace --from-ref origin/main --check
```

The root formatter deliberately excludes `external/duckdb`. Format DuckDB subtree changes with:

```bash
scripts/format duckdb --changed
```

## Updating the DuckDB subtree

The official engine baseline is imported from `duckdb/duckdb` as a squashed
subtree snapshot. Pull a reviewed upstream revision using the same mode:

```bash
git subtree pull --prefix=external/duckdb --squash \
  https://github.com/duckdb/duckdb.git main
```

The subtree metadata records the exact official DuckDB revision in
`git-subtree-split`. Vane-specific engine changes live as subsequent commits
under `external/duckdb`; review and resolve them when updating the official
baseline. When replaying a change formerly maintained in another repository,
preserve its author and date and record the original commit and upstream parent
as commit trailers. To inspect both engine identities without writing the
checkout, run:

```bash
python scripts/sync_duckdb_source_id.py --print
python scripts/resolve_duckdb_fork_version.py --print-version
```

The first command computes the full Git tree object for `external/duckdb`, including
staged, unstaged, and untracked non-ignored engine files without changing the
real Git index or object store. When Git metadata and a source-distribution
manifest are both absent, as in a `git archive` or GitHub source archive, the
script derives a Git-compatible tree object directly from the materialized
paths, modes, symlinks, and contents. Git expands the constant
`.git_archival.txt` template on export so the fallback preserves the
repository's SHA-1 or SHA-256 object format without a per-change identity file.
Native configuration registers the external tree as a CMake configuration
dependency, so Ninja and Makefile builds refresh configure-time metadata after
timestamp-visible source changes. A lightweight build target also refreshes a
generated header in the CMake binary directory. DuckDB's version object and the
entry points of its default in-tree static extensions force-include that header,
so mode-only changes that leave file timestamps untouched still update every
runtime SourceID on the first incremental build.

The second command reports the user-facing fork version as
`vX.Y.Z-vane.<revision>`. `vX.Y.Z` comes from `DUCKDB_UPSTREAM_VERSION`, and the
ten-character revision is calculated from the last Vane commit that changed
`external/duckdb`. Uncommitted changes within that directory append `-dirty`;
changes elsewhere in the checkout do not. Direct incremental builds refresh
the generated version header on every build, so committing an unchanged dirty
tree also replaces the dirty marker with the new path-changing commit.

A custom `DUCKDB_SOURCE_PATH` has no in-tree baseline to infer. Such builds must
set full `VANE_DUCKDB_SOURCE_ID` and `VANE_DUCKDB_FORK_REVISION` values and an
exact `VANE_DUCKDB_UPSTREAM_VERSION` in `vX.Y.Z` form. Configuration fails when
any of these explicit identities is absent; it never reuses the in-tree base.

The local PEP 517 backend injects full `DUCKDB_SOURCE_ID` and
`DUCKDB_FORK_REVISION` manifests directly into the completed sdist, so
read-only source trees remain supported. The sdist carries both manifests for
subsequent builds without Git metadata, and artifact validation checks them
against the checkout. The manifests are ignored build metadata and must not be
committed, so parallel engine pull requests do not modify shared generated
files. A source archive without Git history must contain the injected fork
revision manifest. Update `SOURCE_PROVENANCE.md` and
`DUCKDB_UPSTREAM_VERSION` only when the imported upstream baseline, DuckDB
version line, or historical mapping changes.

The original upstream history remains in `duckdb/duckdb`. Vane's path history
begins at the squashed snapshot and includes every later Vane engine commit. To
inspect or export that history with DuckDB-rooted paths, split it to a temporary
branch:

```bash
git subtree split --prefix=external/duckdb --ignore-joins -b duckdb-history
git log --stat duckdb-history
```

`--ignore-joins` produces a self-contained compact history containing the
official snapshot and Vane's subsequent commits. To reconnect the split branch
to DuckDB's complete upstream history instead, fetch `duckdb/duckdb` first and
omit `--ignore-joins`; Git uses the recorded `git-subtree-split` revision as the
join point.

## Debugging Ray workers

Set `DUCKDB_DISTRIBUTED_DEBUG=1`. Native debug output uses `DistributedDebugStream()` and appears in Ray worker error logs, normally below `/tmp/ray/session_latest/logs/worker-*.err`. Plain C `stdout` output is not reliably captured by Ray workers.

## Release artifacts

Build and validate an sdist before opening a release pull request:

```bash
python -m build --sdist
python scripts/check_release_artifacts.py dist/*.tar.gz
```

See [RELEASE.md](RELEASE.md) for the complete process.

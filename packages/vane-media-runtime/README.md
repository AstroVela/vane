# Vane media runtime

This independent distribution supplies shared media libraries to Vane's
`native_media` extension, which contains audio, image, and video modules. Importing `vane_media_runtime` does not load
native code. Vane's Python media backend does not require this distribution.

The source distribution includes the corresponding upstream source archives,
the exact vcpkg port trees and patches used to build them, the pinned vcpkg build
scripts, the media feature selection, and the runtime packaging backend.
`source-inventory.json` records every delivered file digest, upstream archive
checksums, and recipe identities. License
notices are checked against `components.json` before producing a wheel.

Versions are generated automatically, using the same Vane-version and digest
encoder as the Iceberg provider wheel. The digest identifies the full Vane Git
commit, working-tree state, and Vane source version. There is no separately
maintained semantic release number. `runtime-version.json` freezes this identity
in the source SDK; its manifest carries it into the wheel. Rebuilding the SDK
without Git preserves the version. Dirty checkouts can produce private fixtures
only. Exact runtime dependency pins select the required build; hash components
do not imply chronological ordering within one Vane version.

## Build a source distribution

From a Vane checkout, first install the media dependencies using the triplet in
this directory. Keep this installation separate from the base engine's static
vcpkg installation. Then export the source SDK:

```bash
python -m build --sdist packages/vane-media-runtime \
  -C vcpkg-root=/path/to/pinned/vcpkg \
  -C installed-root=/path/to/media/installed \
  -C downloads=/path/to/vcpkg/downloads
```

The exporter fails if the actual source archives or exact recipes are missing.
Build tools such as the compiler, NASM, CMake, Ninja, and pkg-config remain build
prerequisites. The pinned vcpkg bootstrap and Meson acquisition scripts may
download build tools; rebuilding does not require a Vane Git checkout.

## Build a runtime wheel

Extract the source distribution and run the wheel build inside that directory.
Use a build environment matching the advertised manylinux baseline. The build
validates every shared object's dependencies and versioned symbols against the
pinned platform policy; choosing an older tag does not make an incompatible
binary portable.

```bash
python -m build --wheel \
  -C "source-archive=/absolute/path/to/vane_media_runtime-<generated-version>.tar.gz" \
  -C platform-tag=manylinux_2_28_x86_64 \
  -C signing-key=/protected/path/to/runtime-signing-key.pem
```

When distributing through Vane's registry or another index, pass
`-C source-url=https://<host>/<release-source-page>` for the actual source
location. This signed reference is informational; verification uses the exact
source archive filename and SHA-256, and the loader never fetches that URL.
The default reference is the matching PyPI release page.

The backend verifies required inputs and the complete file inventory, compares
the extracted inputs with the supplied archive, then builds from a private copy
of those verified archive bytes. Unarchived local files cannot enter the build.
Use a fresh SDK extraction for each build; an existing `build` directory is
rejected so previous installed binaries cannot satisfy a source rebuild.
Binary caches are disabled. It namespaces all media
SONAMEs, repairs their dependency references and RUNPATHs, checks the complete
library graph, and finally hashes and signs the manifest. Do not modify or run
an additional wheel repair tool after signing.

Local fixture builds can use `-C test-only=true` and the repository integration
test key. `-C sdk-prefix=/path/to/media/triplet` additionally permits an existing
development SDK, but only for a fixture. Such wheels carry
`Private :: Do Not Upload` and cannot be released.

## Extension linkage and local replacement

Configure extension builds with `VANE_MEDIA_RUNTIME_SDK` pointing to the media
triplet and `VANE_MEDIA_RUNTIME_DIRECTORY` pointing to the staged runtime package
directory. Keep `EXTENSION_STATIC_BUILD=ON`: DuckDB remains embedded in each
extension, while the media libraries are shared. The build relocates unsigned
extensions and binds their runtime manifest digest before extension signing.

Users replacing a compatible library keep its namespaced filename and SONAME.
The local runtime directory contains `.libs` and an updated
`runtime-manifest.json` with the new file hashes. Before preparing any native
media extension, select it explicitly:

```python
import vane

vane.use_native_media_runtime("/absolute/path/to/my/runtime")
```

The official extension and its signature remain unchanged. The official runtime
package still supplies the authenticated reference manifest; the explicit local
selection authorizes the replacement code for this process. Start a new process
to switch runtimes. The first release supports the official runtime on Ray. Custom runtime selection
is rejected when exporting or preparing a worker snapshot, so a driver cannot
silently run different media code from its workers.


## Loading model

The resolver verifies the signed runtime manifest and every shared library,
then prepares a private directory containing the extension and `.libs`.
DuckDB loads the extension normally; the operating system follows its
`$ORIGIN/.libs` RUNPATH and each library's `$ORIGIN` RUNPATH. Python does not
preload libraries and the native loader does not call back into Python.
A prepared directory can also be loaded directly with SQL `LOAD`, including
in a fresh process, subject to the normal DuckDB ABI and signature checks.

Manifest verification happens during preparation. Direct SQL loading does not
revalidate runtime manifests or library hashes. Keep the complete prepared
directory together and protect it against modification. Versioned Vane SONAMEs
reduce collisions with independently installed codec packages, but do not
provide isolation from arbitrary native code already loaded in a process.

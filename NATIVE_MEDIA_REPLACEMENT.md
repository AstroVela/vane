# Replacing native media libraries

Keep this document with `media-release.json`, the exact `vane-ai` base wheel,
`vane-extension-native-media` provider wheel, `vane-media-runtime` wheel and its
matching `.tar.gz` source SDK. License texts and component notices are inside
the runtime wheel's `.dist-info/licenses` and the SDK's `LICENSES` directories;
the base and provider wheels also carry their own license files. Preserve them
when redistributing the files or incorporating them in an image. The SDK
contains the library sources, patches, pinned recipes and license inventory.

## Verify the delivery

Obtain the release manifest SHA-256 from the publisher's separately authenticated
release record. Use the release tooling from the reviewed Vane source checkout
or base source distribution, with its build dependencies installed:

```bash
python -I scripts/media_release.py verify \
  --directory /path/to/downloaded-delivery \
  --manifest-sha256 <retained-sha256> --trust-identity astrovela/vane
```

Verification checks every delivered file, corresponding sources, licenses,
platform requirements, package compatibility and native signatures, then loads
the provider in a fresh installation. The current delivery tool supports one
Linux x86-64 base/provider/runtime combination per directory. Use a fresh
process and the matching CPython minor and manylinux baseline for acceptance.

## Rebuild and demonstrate replacement

Build prerequisites include a C/C++ compiler, CMake, Ninja, NASM, pkg-config,
autoconf, automake, libtool, Perl, Git, curl, zip/unzip, OpenSSL and patchelf.
Install the Python build dependencies documented in the SDK. The SDK carries
the library source archives; its pinned bootstrap scripts may download build
tools. Binary caches are disabled for the source rebuild.

This acceptance command extracts a fresh SDK, rebuilds its dependencies, changes
SoXR's version function in the C implementation, compiles a replacement, and
executes it through the unchanged signed provider in a new virtual environment:

```bash
python -I scripts/media_release.py rebuild \
  --directory /path/to/downloaded-delivery \
  --manifest-sha256 <retained-sha256> --trust-identity astrovela/vane \
  --output /path/to/new-rebuild-directory --jobs 2
```

The output contains the extracted sources and build, `local-runtime`, and
`rebuild-verification.json`. The receipt binds the release manifest, unchanged
extension SHA-256, effective runtime SHA-256 and observed
`libsoxr-local-rebuild-proof` result. Keep the log and receipt with the release
record. This command needs no publisher private key and does not re-sign or
publish a wheel. It is an acceptance experiment; retain your own source changes
separately when making a useful library modification.

For your own changes, the SDK can also be used without a Vane checkout. Extract
it into a fresh directory, edit the sources/recipes, and run its documented
build entry point to obtain a development SDK:

```bash
python -c 'from pathlib import Path; import backend; backend._build_sdk(Path.cwd())'
```

You can instead compile an individual library from its supplied source archive.
Preserve the ABI required by the extension. Use the SDK's included helper to
relocate your rebuilt library and update its content manifest:

```bash
python scripts/prepare_local_media_runtime.py \
  --runtime /path/to/installed/vane_media_runtime \
  --replacement libsoxr.so.0=/path/to/rebuilt/libsoxr.so \
  --output /path/to/my-runtime
```

`--runtime` is the installed official runtime package directory containing
`runtime-manifest.json` and `.libs`, not a wheel file. The helper preserves the
required filenames/SONAMEs and checks the dependency graph. The exact official
runtime wheel remains installed as the authenticated reference.

## Select a replacement locally

```python
import vane

vane.use_native_media_runtime("/path/to/my-runtime")
con = vane.connect(config={"audio_backend": "native"})
vane.load_installed_extension("native_media", connection=con)
```

Select before preparing any native media extension. Selection trusts your local
library code and leaves the extension's signature policy intact. One process
can select one runtime; start a new process to change it. Direct SQL `LOAD` of
an already prepared directory uses ordinary native loading and does not repeat
manifest verification. Keep that directory intact and protected from edits.

## Deploy the replacement to Ray

Install the exact base, provider and official runtime wheels on the coordinator
and every Ray node. Deploy the complete replacement directory, including its
manifest, to each node. Paths may differ; file contents must match.

Set this variable in **each node's environment before starting Ray**, including
the head node if it can run actors:

```bash
export VANE_NATIVE_MEDIA_RUNTIME=/node/local/path/to/my-runtime
# Start this node with your normal ray start command.
```

In a fresh coordinator process, opt in explicitly before loading media:

```python
import vane

vane.use_native_media_runtime("/coordinator/path/to/my-runtime", allow_distributed=True)
# Load native_media and submit queries with your normal Ray runner configuration.
```

Do not put `VANE_NATIVE_MEDIA_RUNTIME` in a Ray Job or actor `runtime_env`.
It is a node deployment setting and is not copied from the coordinator. Query
snapshots carry only the expected content digest. Every process checks its
independently authorized local runtime before admitting a query, including
when reusing an existing connection. Plans retain that digest when replayed by
a local runner. Missing authorization, changed bytes or
different digests fail before loading. An incompatible query cannot fall back
to official libraries. Restart Ray processes when changing the runtime used by
a deployment. Local selection without `allow_distributed=True` remains local.

## Python wheels included in your own delivery

If an offline wheelhouse or container includes PyAV, SoundFile, python-soxr or
other Python packages, inventory those exact wheels separately:

```bash
python -I scripts/media_release.py inventory-python \
  --wheel /wheelhouse/av-<exact-wheel>.whl \
  --wheel /wheelhouse/soxr-<exact-wheel>.whl \
  --output python-delivery.json
```

Supply every wheel actually redistributed. The report records hashes, declared
licenses, notice files and embedded native binaries without importing packages.
Each entry starts with `review_status: required`: review the selected binary
features, all bundled libraries and their corresponding-source/replacement
requirements. Record that review against the wheel SHA-256 and deliver any
required additional materials. A wrapper's Python license does not establish
the licenses of its bundled codecs. For containers, also retain the final
image digest and inspect OS packages and other files outside these wheels.

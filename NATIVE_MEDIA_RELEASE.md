# Native media publication

The `Native media release` workflow in `.github/workflows/media-release.yml`
publishes one optional package, `vane-extension-native-media`, containing the
`native_media` extension and its dynamically linked libraries. Its first
qualification profile is **CPython 3.12, Linux x86-64, manylinux_2_28**. The
combined wheel is `cp312-none` and pins the exact Vane base version. Its signed
manifest binds the bundled libraries and their corresponding source SDK. Other interpreters and
platforms need their own build and acceptance profiles before publication
support is expanded.

Versions still come from the Git identity and shared provider version encoder.
The base wheel must already be available on PyPI for the exact Vane version
tag being released. A locally compiled base wheel is a build byproduct; the
delivery uses the original published base bytes.

## Repository setup

Configure these before the first `release` run. They are repository/index
settings, not changes made by the workflow or by installing Vane.

- Protect Vane version tags and the main/release branches. Publication requires
  `workflow_dispatch` in `AstroVela/vane`, a protected final version tag such as
  `v0.2.0` or `v0.2.1.post1`, a clean checkout at the dispatched commit, and
  ancestry on the corresponding release branch. The tag must contain this
  workflow and its tools. Development and prerelease versions cannot enter
  the signing job.
- Create `media-production-signing` with required reviewers, prevent self
  approval, and restrict its deployment tags. Store the production key as
  `VANE_EXTENSION_SIGNING_PRIVATE_KEY` only in that environment. Its RSA-2048
  public key must match the built-in `astrovela/vane` key, DER SPKI SHA-256
  `8729fbfbf5276be4b159c0b698c9e4214edd72eaad3e21bcefc03bcb36dffaeb`.
  The integration fixture and TestPyPI development keys cannot be used.
- Create `media-github`, `native-media-testpypi`, and `native-media-pypi`
  as protected environments with
  reviewed tag restrictions. The signing key belongs only to the signing
  environment. An environment name alone does not configure approval or
  tag protection.
- Register these GitHub Trusted Publishers, each with owner `AstroVela`,
  repository `vane`, and workflow `media-release.yml`. No API-token fallback
  is provided.

| Index | Project | GitHub environment |
| --- | --- | --- |
| TestPyPI | `vane-extension-native-media` | `native-media-testpypi` |
| PyPI | `vane-extension-native-media` | `native-media-pypi` |

- Enable [GitHub release immutability](https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/establish-provenance-and-integrity/prevent-release-changes).
  Candidates must be public and immutable before acceptance can start. A
  mutable release fails the workflow before either Python index is updated.
  GitHub [locks the assets and tag](https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases)
  while allowing the prerelease flag and notes to change after qualification.
- Review and protect changes to release tooling. See the
  [Trusted Publishing security model](https://docs.pypi.org/trusted-publishers/security-model/)
  and [publisher registration guide](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

## Build and release

Start with an unsigned build on a reviewed commit:

```bash
gh workflow run media-release.yml --repo AstroVela/vane --ref main \
  -f operation=build-only
```

This exports the complete source SDK, recompiles its libraries with binary
caches disabled, retains the SDK to compile `native_media`, and uploads the
unsigned inputs as Actions artifacts. It needs no signing key or publisher
environment. The runtime archive ends in `.whl.unsigned`; it is not an
installable or publishable wheel. Build-only success does not qualify a
production delivery.

After publishing the matching Vane base release through `release.yml`, dispatch
the media workflow at that same final tag:

```bash
gh workflow run media-release.yml --repo AstroVela/vane --ref v0.2.0 \
  -f operation=release
```

Replace the example tag with the actual release tag. The preflight downloads
and validates the exact base wheel from PyPI, rejects an existing candidate tag
for the same source commit, and freezes the source and base identities. The pipeline then:

1. Exports all corresponding sources and rebuilds the shared libraries in the
   pinned manylinux container. It compiles the unsigned extension against that
   SDK and binds the runtime manifest digest before signing.
2. Enters an independent signing job. System Python with `-I -S` reads only
   the bounded native artifact and manifest. It does not install dependencies,
   compile code, unpack sources, or load native build outputs. The key
   fingerprint, source identity, source URL, empty signature slot and
   extension/runtime binding must match before signing.
3. Bundles the signed runtime into the provider wheel in an unprivileged job.
   The intermediate runtime wheel is a build input and is never uploaded to
   either Python index. Only its signature and RECORD entry may change during
   signing; the native extension's payload must stay identical. Existing source, license, ELF, clean-install
   and native signature validators must pass before exposing a delivery.
4. Attaches all five delivery files to a GitHub draft, then publishes it as an
   immutable prerelease tagged `native-media-<full-Vane-commit>`. The signed
   runtime's source URL points to this release. Draft assets are not anonymously
   downloadable, so acceptance starts only after publication.
5. Calls `media-release-verify.yml` as a mandatory job. It anonymously downloads
   the complete set using the independently retained manifest SHA-256,
   rebuilds from the SDK, modifies and replaces SoXR without a signing key,
   and observes the modified implementation in a fresh installation. Two real
   Ray nodes must accept the matching replacement and reject a node with a
   different runtime. Both tests must pass; skips fail acceptance.
6. Uploads only the combined provider wheel to TestPyPI. It downloads that
   indexed wheel and retrieves the source SDK afresh from the public GitHub
   URL in its signed manifest. Both must match the accepted hashes and sizes;
   clean native verification runs on the minimum supported platform.
7. Publishes the same provider bytes to PyPI using `native-media-pypi`. No
   rebuild, wheel repair, manifest change or re-signing occurs between indexes.
   It downloads the final indexed wheel and public source SDK and verifies them again.
8. Publishes evidence under a second immutable release,
   `native-media-evidence-<full-Vane-commit>`, then marks the original candidate
   qualified. Its assets and tag stay frozen. Evidence includes the pinned
   manifest, download/rebuild logs, modified-SoXR receipt, two-node Ray results,
   both index receipts and the actual wheel inventory. The delivery notes link
   to this evidence. Actions copies are retained for 90 days; public evidence
   remains with the release.

The five delivery files are the base wheel, combined provider wheel, source
SDK, `NATIVE_MEDIA_REPLACEMENT.md`, and `media-release.json`. The manifest keeps
`schema_version: 1` during this early development phase and accepts exactly
the `base`, `provider`, `source`, and `instructions` roles. Regenerate older
dual-package delivery manifests; the former `runtime` role is rejected.
Evidence is separate because the delivery verifier rejects extra files. The
base wheel is not uploaded again to either Python index. The source SDK is
delivered through the same immutable GitHub release that mirrors the provider
wheel. Its URL is also exposed in the wheel's `Project-URL` metadata.

Users install `vane-extension-native-media`; pip installs its exact `vane-ai`
dependency. No `vane-media-runtime` installation or Trusted Publisher is needed.
The combined wheel carries all runtime notices under `.dist-info/licenses/runtime/`
and includes both extension and library licenses in `License-Expression`.
The source SDK retains its internal Git-derived component name and version;
it is a rebuild attachment, not a second installable media dependency.

## Failed runs and recovery

Resume failed jobs while their original Actions artifacts still exist:

```bash
gh run rerun <run-id> --repo AstroVela/vane --failed
```

Every handoff uses immutable Actions artifact IDs and fails on a digest
mismatch. GitHub uploads never overwrite an existing asset. If an index upload
partially succeeded, the job downloads each existing file and requires
identical bytes before uploading only the missing files. Unknown, changed or
yanked files stop publication; `skip-existing` is not used.

Do not rebuild an indexed provider version or replace/delete published source
assets. If source or build changes are required, use a new reviewed source
identity and its matching published Vane base version. Preserve a failed public
candidate and its sources for recipients. Failed acceptance leaves the delivery
as a prerelease and blocks promotion; do not manually bypass the failure.

## License review and redistribution scope

The gate enforces the reviewed dynamic media profile in [COPYLEFT.md](COPYLEFT.md),
the runtime/source notice inventories, matching sources and a working library
replacement path. The SDK contains full upstream archives, including separately
licensed tools and documentation that are not compiled into the runtime. Its
license expression differs from the binary expression. The provider includes
the media notice bundle plus the base engine notices for static dependencies.

The Python inventory describes the **two wheel files actually redistributed
by this workflow**. Metadata and notice hashes remain marked `review_status:
required`; generating an inventory is not an approval of every upstream grant.
Ordinary pip dependencies fetched by recipients are not embedded in this
delivery. Docker images, offline wheelhouses or installation bundles that
redistribute additional wheels must inventory and review those actual bytes
separately. This includes Python media packages such as PyAV, soxr or SoundFile
when included. This workflow does not approve unspecified Python packages or
their bundled codecs.

[Issue #794](https://github.com/AstroVela/vane/issues/794) remains the audit
tracker. PR #811 implemented dynamic runtime separation and replacement;
PR #816 added delivery/source acceptance tools and Ray replacement admission;
this workflow connects those gates to publication. Completion still requires
the first successful production release evidence and review of each actual
redistribution profile. Do not close the audit merely because the automation
has merged or because a keyword scan passes.

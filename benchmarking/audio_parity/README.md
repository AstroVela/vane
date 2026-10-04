# Audio parity audit

The [audit script](../../scripts/audit_audio_parity.py) compares independently
installed Vane and Daft runtimes using identical encoded inputs. It records full
waveforms, metadata, library versions, binary hashes, and errors locally under
`build/audio-parity/`. Generated media and detailed run outputs stay local.

## Output contract

Vane Python and native resample along the time axis and return contiguous
float64 arrays shaped `(frames, channels)`, including mono and empty audio.
Both use SoXR HQ and normalize the complete stream once to
`ceil(actual_decoded_frames * target_rate / actual_source_rate)` using integer
arithmetic. Tail trimming and zero padding count toward output limits.

Native uses libsndfile for PCM/float WAV and AIFF, 8/16/24-bit FLAC, MP3, and
Ogg Vorbis/Opus. Other supported codecs, including Ogg FLAC and 32-bit FLAC,
retain FFmpeg decoding. Common-format metadata uses the same decoder
information as Python SoundFile, including
`WAVEX`/`RF64` container names, PCM bit depth, Opus sample rates, encoder delay,
and tail trimming. Known counts include zero for empty audio; unknown frames
and duration are NULL. Known duration is `frames / sample_rate`.

Metadata stays inside its FILE byte window and read budget. It does not decode
a complete unknown-length waveform to manufacture a frame count. Additional
FFmpeg codecs retain their native format/codec identifiers and return NULL
frames/duration when an exact count is unavailable.

See [NATIVE_MEDIA_EXTENSIONS.md](../../NATIVE_MEDIA_EXTENSIONS.md) for backend
selection, supported formats, diagnostics, limits, and optional dependencies.
`AudioFile.resample(..., connection=...)` remains the Python value method;
native parity must be measured through SQL or Expressions with
`audio_backend='native'`.

## Historical measurements

The [2026-09-08 report](https://github.com/AstroVela/vane/blob/4e12994a2fed5b872a7bdb44df72c1b9c5653cdc/benchmarking/audio_parity/README.md)
records the pinned Vane/Daft versions, corpus, waveform comparisons and limitations.
Use the reproduction steps below to measure a different revision.

## Reproduce

Follow [DEVELOPMENT.md](../../DEVELOPMENT.md) and the dynamic `native_media`
SDK/runtime build instructions in [NATIVE_MEDIA_EXTENSIONS.md](../../NATIVE_MEDIA_EXTENSIONS.md).
Build and stage the shared runtime first, then use its SDK and runtime directory
in the following non-editable build. Keep the staged extension and adjacent
`.libs` directory together:

```bash
export SKBUILD_BUILD_DIR="$PWD/build/python-release"
export SKBUILD_CMAKE_BUILD_TYPE=Release
uv pip install . --no-build-isolation \
  -Ccmake.define.VANE_LOADABLE_EXTENSIONS=native_media \
  -Ccmake.define.VANE_MEDIA_RUNTIME_SDK=/path/to/media/installed/x64-linux-vane-media \
  -Ccmake.define.VANE_MEDIA_RUNTIME_DIRECTORY=/path/to/staged/vane_media_runtime
cmake --build "$SKBUILD_BUILD_DIR" --target vane_loadable_extensions

uv pip install --python .venv/bin/python 'soundfile==0.14.0' 'soxr==1.1.0' 'librosa==0.11.0'
uv venv --python /usr/bin/python3.12 .venv-daft
uv pip install --python .venv-daft/bin/python 'daft[audio]==0.7.24'

.venv/bin/python -I scripts/audit_audio_parity.py generate build/audio-parity
.venv-daft/bin/python -I scripts/audit_audio_parity.py run build/audio-parity --engine daft --label daft
.venv/bin/python -I scripts/audit_audio_parity.py run build/audio-parity --engine vane --label vane-metadata-parity \
  --native-extension build/python-release/vane_extensions/native_media.duckdb_extension
.venv/bin/python -I scripts/audit_audio_parity.py compare build/audio-parity --left vane-metadata-parity --right daft
```

Input generation requires the system `ffmpeg` command. Use a new label to
preserve a previous runtime's arrays and measurements. The comparison JSON
and console summaries include metadata equality, exact waveform equality,
mono-normalized equality, tolerance comparisons, and errors.
Each run retains its input manifest, canonical manifest digest, and verified
copies of the encoded inputs under its label. Comparisons require matching
input identities and complete results, so regenerating the shared corpus
between engine runs cannot silently compare different inputs. Legacy results
without this identity must be rerun with the updated script. The comparison
also records both result-file digests and runtime versions, preserving its
provenance if labels are reused later.
The left label must identify a Vane run and the right label a Daft run.
Reversed or same-engine inputs return a command-line argument error before
accessing engine-specific results; custom labels retain their requested sides.
Each compared array is checked against its recorded SHA-256, shape, and dtype.
Digest verification and NumPy decoding use the same byte snapshot. Damaged
arrays and inconsistent records fail with a data error and retain the traceback;
only unsupported engine order is reported as a command-line argument error.

Regression tests in `test_audio_file.py` and `test_native_audio_parity.py`
add empty/short/fractional/unknown-length inputs, metadata field parity,
WAVEX/RF64, AIFF/FLAC bit depths, Ogg/WebM Opus rate handling, Ogg/32-bit FLAC
fallback, FILE windows, and frame/byte/batch/probe limits. The additional FLAC
fixtures require the system `ffmpeg` command. Native media tests also exercise
shared I/O, cancellation, and execution without Python codec helpers.

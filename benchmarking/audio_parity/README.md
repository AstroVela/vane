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

Native uses libsndfile for PCM/float WAV and AIFF, FLAC, MP3, and Ogg
Vorbis/Opus/FLAC. Other supported codecs retain FFmpeg decoding. Common-format
metadata uses the same decoder information as Python SoundFile, including
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

## Measurements on 2026-09-08

Baseline Vane commit: `45ee1fa9f94b93713a44e1dd7531b854be0a5bb2`.
Daft release: [0.7.24](https://github.com/Eventual-Inc/Daft/releases/tag/v0.7.24),
commit `9c2b73e084356711bd74b2ca629464045737327b`.
Daft main `b01038a27242ef5c1fd3e87e2b49f339f7af992e` had identical audio
implementation and audio tests at the time of the audit.

Linux x86-64, Python 3.12.3, local CPU execution, non-editable Release install.
Vane used SoundFile 0.14.0, NumPy 2.5.2, python-soxr 1.1.0, and librosa 0.11.0
for the reference path. The separate Daft audio environment used SoundFile
0.13.1 and NumPy 2.5.3. Both SoundFile builds used libsndfile 1.2.2. Native
dependencies were libsndfile 1.2.2, SoXR 0.1.3, and FFmpeg 8.1.1. Repeating
Daft with Vane's SoundFile/NumPy versions produced identical Daft outputs.

The corpus contains 28 files and 70 sample-rate combinations, with mono,
stereo, four channels, empty and 1/2/7/31/32/33/65-frame inputs, fractional
output lengths, and a 150001-frame input crossing decoder chunks. Formats
include PCM/float/ALAW/ULAW WAV, AIFF, FLAC, MP3, Ogg Vorbis/Opus, M4A/AAC,
and WebM/Opus. Synthetic inputs include sinusoids, seeded noise, and nonzero
first/last samples. Input SHA-256 is checked before each run.

| Comparison after fixes | Successful shared cases | Complete output equal |
| --- | ---: | ---: |
| Python metadata value / SQL / Expression | 26 | 26 |
| Python / native metadata, all six fields | 26 | 26 |
| Native metadata SQL / Expression | 28 | 28 |
| Python waveform value / SQL / Expression | 66 | 66 |
| Native waveform SQL / Expression | 70 | 70 |
| Python / native waveform | 66 | 64 exact; 66 within `rtol=0, atol=1e-6` |
| Python / librosa with explicit `axis=0` | 66 | 66 |

The two residual waveform differences are one Vorbis file at 48000 and
16000 Hz. Maximum absolute errors are `1.7881393432617188e-7` and
`1.4901161193847656e-7`, respectively. The identity-rate difference is already
present at decoding. The libsndfile builds differ; this audit does not
attribute the difference to a particular compiler option. Shape and dtype
are checked separately and exactly. Matching a common prefix is never
reported as equality of the complete output.

| Input | Baseline native | Fixed native / Python |
| --- | --- | --- |
| 7-frame WAV, 8000 to 16000 Hz | 0 output frames | 14 output frames |
| 1001-frame WAV, 44100 to 16000 Hz | 363 output frames | 364 output frames |
| Vorbis, 48000 to 48000 Hz | 4864 output frames | 4801 output frames |
| Vorbis, 48000 to 16000 Hz | 1621 output frames | 1601 output frames |
| 24-bit FLAC metadata | `flac` / `flac`, NULL frames | `FLAC` / `PCM_24`, 4801 frames |
| Opus metadata duration | 0.10652083333333333 s | 0.10002083333333334 s |

Baseline Python returned SoXR's unnormalized output length. The ceil change
added a single zero tail frame in 17 of the 66 supported combinations;
existing samples remained equal. Baseline native used libswresample's
resampler and FFmpeg decoding; only 25 of the 66 Python/native complete
waveforms were equal.

Daft 0.7.24's audio resampler passes `(frames, channels)` input to librosa
without setting `axis`; its multi-channel resampling operates on the channel
axis. Vane keeps its time-axis behavior. After the Vane length fix, 46 of 66
Python/Daft outputs match when only mono shape is normalized; 20 changed-rate
multi-channel cases still differ. Daft's mono output is one-dimensional.
These results describe this corpus and the listed library builds, not every
platform, remote store, Ray execution path, or possible encoded input.

## Reproduce

Follow [DEVELOPMENT.md](../../DEVELOPMENT.md) and the optional native-audio
dependency/build instructions in [NATIVE_MEDIA_EXTENSIONS.md](../../NATIVE_MEDIA_EXTENSIONS.md).
Use a non-editable installation and matching extension artifact:

```bash
export SKBUILD_BUILD_DIR="$PWD/build/python-release"
export SKBUILD_CMAKE_BUILD_TYPE=Release
uv pip install . --no-build-isolation -Ccmake.define.VANE_LOADABLE_EXTENSIONS=audio
cmake --build "$SKBUILD_BUILD_DIR" --target vane_loadable_extensions

uv pip install --python .venv/bin/python 'soundfile==0.14.0' 'soxr==1.1.0' 'librosa==0.11.0'
uv venv --python /usr/bin/python3.12 .venv-daft
uv pip install --python .venv-daft/bin/python 'daft[audio]==0.7.24'

.venv/bin/python -I scripts/audit_audio_parity.py generate build/audio-parity
.venv-daft/bin/python -I scripts/audit_audio_parity.py run build/audio-parity --engine daft --label daft
.venv/bin/python -I scripts/audit_audio_parity.py run build/audio-parity --engine vane --label vane-metadata-parity \
  --native-extension build/python-release/vane_extensions/audio.duckdb_extension
.venv/bin/python -I scripts/audit_audio_parity.py compare build/audio-parity --left vane-metadata-parity --right daft
```

Input generation requires the system `ffmpeg` command. Use a new label to
preserve a previous runtime's arrays and measurements. The comparison JSON
and console summaries include metadata equality, exact waveform equality,
mono-normalized equality, tolerance comparisons, and errors.

Regression tests in `test_audio_file.py` and `test_native_audio_parity.py`
add empty/short/fractional/unknown-length inputs, metadata field parity,
WAVEX/RF64, AIFF/FLAC bit depths, Ogg/WebM Opus rate handling, FILE windows,
and frame/byte/batch/probe limits. Native media tests also exercise shared
I/O, cancellation, and execution without Python codec helpers.

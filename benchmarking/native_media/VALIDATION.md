# Native image/audio validation and measurement

This guide extends the original local measurements in [README.md](README.md).
Video indexed construction and reuse measurements are in
[../video_seek/README.md](../video_seek/README.md). Each report identifies the
actual runtime and artifact used; old measurements are not relabeled as a
measurement of a newer revision.

## Reproduce inputs and measurements

Build/install the base and selected extensions using
[NATIVE_MEDIA_EXTENSIONS.md](../../NATIVE_MEDIA_EXTENSIONS.md). Generate
synthetic inputs outside Git:

```bash
python -I benchmarking/native_media/generate_inputs.py /tmp/native-media-inputs --matrix
python -I scripts/benchmark_native_media.py audio_resample \
  /tmp/native-media-inputs/audio-large.wav \
  --extension build/python-release/vane_extensions/audio.duckdb_extension \
  --rows 4 --sample-rate 16000 --repetitions 5 --diagnostics \
  --allow-unsigned-development-artifact
```

The matrix contains PNG and JPEG; PCM WAV/AIFF and FLAC at 8, 44.1, 48, and
96 kHz with mono, stereo, six, and eight channels; short, five-second, and
30-second clips; and stereo MP3, Vorbis/Ogg, AAC/ADTS, AAC/MP4, and Opus/WebM.
The latter formats include encoder delay/padding. Python's libsndfile backend
does not support every native container/codec. Run those native-only and
report the unsupported comparison explicitly. Fixture creation uses Pillow,
soundfile, NumPy, and PyAV; these tools are not dependencies of native media
execution. The generator prints input sizes and SHA-256 digests.

Use `image_metadata`, `image_decode`, `audio_metadata`, and `audio_resample`
for the image/audio matrix. `--image-mode` defaults to RGB, making output
dimensions/byte counts comparable across image inputs. Use equal, lower, and
higher target rates for audio, with each setting recorded in the report.
Python uses SoXR HQ; native uses pinned libswresample defaults. Timing ratios
do not establish equal signal fidelity or isolate the interpreter overhead.

Add `--transport http` to serve the same files from a separate local process.
`--http-delay-ms 5` adds five milliseconds to each server request. This is a
controlled latency experiment, not a cloud-storage measurement. `--position`
and `--size` select the same logical FILE window from every supplied input.

`--concurrency 4 --threads 1` runs four independent connections concurrently,
each with `--rows` file references. This measures concurrent query throughput;
it does not imply four-way parallelism inside one scalar expression batch.
Multiple input arguments are visited cyclically. No input/decoded-output
cache is added by the harness. Keep per-batch decoded output within the
operator's existing limits; excessive batches must fail rather than truncate.

Run each `--backend python` and `--backend native` in a fresh process for RSS.
The Python-only command needs no artifact and does not load one. Timed groups
follow one warmup per backend and alternate backend order. Reports retain
every repetition and its aggregate result; changed results within one backend
fail the run. `aggregate_results_match` indicates whether backend aggregates
match; it does not compare pixels or samples, and unmatched output counts
must not be presented as an equivalent-workload speedup.

For Ray, install matching signed providers, then use `--runner ray
--installed-provider --ray-cpus 4`. The command owns a local Ray cluster and
closes its runners before shutdown. No cloud endpoint or credential is needed
for these synthetic local/loopback experiments. Driver CPU and RSS exclude
Worker processes. Ray timing includes distributed query submission, execution,
and aggregate-result transport, after warmup. `extension_descriptors` records
the exact loaded provider identities. `--diagnostics` is a separate local mode
and cannot be combined with Ray.

## Cost accounting

| Metric | Scope |
| --- | --- |
| Query wall time | Complete group of `concurrency` independent queries; binding/execution/aggregate fetch included |
| Process CPU | Benchmark driver, including its local execution threads; excludes HTTP server and Ray Workers |
| Peak RSS | Driver process through timed-query completion; includes imports/loading/warmup; excludes subsequent diagnostics |
| HTTP bytes/requests | Successful response-body writes and requests from a separate server; snapshot waits for active responses to finish |
| Python spool writes | Separate pass intercepting Python `TemporaryFile`; reports cumulative writes and peak live logical spool bytes, not physical disk traffic |
| Audio phase costs | Separate calls to `native_audio_resample_profile`; real decoder, resampler, and bounded output allocation |

HTTP counting detects failed responses rather than silently recording a
successful benchmark. A local input has no HTTP byte count. Audio profiles
add native FILE byte/read counters for local or HTTP inputs. OS caches remain
uncontrolled; no cold-cache claim is made. Python temporary-file accounting
does not intercept native filesystem APIs or engine/Ray spills. The native
image/audio implementations write decoded values into engine buffers and
do not create media spools; zero intercepted Python spool calls alone is not
evidence of zero process-wide disk writes.

The resampler writes samples into its result vector directly. Its conversion
timer therefore includes those writes; allocation time measures reservation
and buffer growth, not a second output-copy stage. FILE-read time overlaps
setup/decode and is reported separately. Profiling adds clocks and returns a
different result, so it explains costs but is not used as the uninstrumented
operator latency. Its waveform workspace is fresh for each batch; allocation
counts can differ from ordinary execution that retains capacity across batches.

## Validation coverage

`test_native_media_extensions.py` checks native dispatch without Python codec
imports/helpers, modes, MIME aliases, bounded headers, windowed reads,
resource/error behavior, and the original local/streaming operators.
`test_native_media_validation.py` adds lossless channel/sample-layout checks,
resampling against analytic signals, additional encoded audio containers,
profile/normal limit enforcement, counted HTTP windows, I/O cancellation,
and end-to-end benchmark accounting. It does not require cross-backend
bitwise or numerical equivalence.

`test_ray_native_media_extensions.py` checks exact installed providers,
concurrent connections with independently selected domains/backends,
repeated queries and FILE windows, native audio profiles, rejected altered
or missing provider identities, and idempotent preparation after rejection.
Video expression/source/index Ray tests remain separate. Broader
multi-provider/multi-Secret-scope credential resolution is tracked by #661;
these native media checks do not establish that stronger credential contract.

Run affected tests before the release suite, keeping Ray in a separate process:

```bash
scripts/run_installed_pytest.sh -m 'not real_ray' \
  tests/fast/test_native_media_extensions.py tests/fast/test_native_media_validation.py
VANE_TEST_NATIVE_MEDIA_PROVIDERS=1 scripts/run_installed_pytest.sh -m real_ray \
  tests/fast/test_ray_native_media_extensions.py
scripts/run_release_tests.sh
```

Artifact-path environment variables and signed-provider setup follow the
native extension guide. Test signatures are for local/CI validation and do
not establish publication of production provider wheels.

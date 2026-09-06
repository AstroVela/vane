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

## Measurements from 2026-09-06

The following measurements use native commit `4f215fc6aa`, Vane
`0.2.0.dev626`, DuckDB `v1.5.5-vane.4f215fc6aa`, and SourceID `454c70f60b`.
The local harness is from `c36da3ba45`; Ray reporting additionally includes
the exact Arrow HUGEINT-to-integer conversion in `d5e97ba432`. The final
four-connection image run uses `6834436e4e`, which starts Workers outside the
source checkout and reports missing partitions explicitly.
The host is an Intel Xeon E5-2686 v4 with 18 physical cores / 36 logical CPUs,
about 62.7 GiB RAM, Linux 6.17.0, glibc 2.39, and local NVMe/ext4 storage.
Runs were sequential after builds/tests ended, without CPU pinning or OS-cache
control. They are shared-host measurements, not a machine-isolated benchmark.

Python versions are 3.12.3, NumPy 2.5.2, Pillow 12.3.0, soundfile 0.14.0
(libsndfile 1.2.2), soxr 1.1.0, PyAV 17.1.0, and Ray 2.58.0. The native
extensions use pinned vcpkg FFmpeg 8.1.1#2; the audio diagnostics report
libavcodec 62.28.101 and libswresample 6.3.101. Local runs load the selected
unsigned development artifact explicitly; Ray uses matching test-signed
installed providers. Artifact hashes and exact input identities accompany
the measurement tables in the CSV files linked below.

### Local operation and format matrix

Wall and driver CPU figures below are medians in milliseconds over five
repetitions, with one connection and one engine thread. All paired runs have
matching aggregate counts. Images decode to RGB at their source resolution;
audio resamples to 16 kHz unless a different target is shown. Native and
Python resampler quality settings differ as described above.
File counts are repeated references to the named fixtures, cycling through
the input list for mixed cases; they are not counts of distinct objects.

| Operation/input | Files per query | Wall Python / native | CPU Python / native |
| --- | ---: | ---: | ---: |
| Metadata, mixed small PNG/RGB JPEG/L JPEG | 96 | 16.82 / 3.45 | 16.80 / 3.44 |
| Metadata, mixed large PNG/RGB JPEG/L JPEG | 12 | 6.05 / 1.85 | 6.04 / 1.85 |
| Decode, mixed small images | 96 | 40.40 / 22.63 | 40.40 / 22.62 |
| Decode, 1280×720 PNG | 8 | 170.06 / 114.17 | 170.06 / 114.15 |
| Decode, 1280×720 RGB JPEG | 8 | 118.86 / 120.33 | 118.84 / 120.29 |
| Decode, 1280×720 L JPEG | 8 | 97.39 / 94.91 | 97.38 / 94.52 |
| Metadata, WAV/AIFF/FLAC/MP3/Ogg | 20 | 10.74 / 9.61 | 10.80 / 9.62 |
| Resample, 0.1 s 8 kHz stereo WAV | 32 | 23.06 / 15.57 | 23.05 / 15.56 |
| Resample, 5 s 48 kHz stereo WAV | 4 | 34.74 / 131.73 | 34.73 / 131.69 |
| Resample, 30 s 48 kHz stereo WAV | 2 | 83.43 / 101.05 | 83.40 / 101.00 |
| Resample, 5 s 44.1 kHz mono WAV | 4 | 19.53 / 67.40 | 19.52 / 67.37 |
| Resample, 5 s 48 kHz six-channel WAV | 2 | 40.11 / 250.23 | 40.11 / 250.18 |
| Resample, 1 s 96 kHz eight-channel WAV | 2 | 21.30 / 224.48 | 21.31 / 224.43 |
| Resample, large WAV → 8 kHz | 4 | 28.39 / 133.54 | 28.37 / 133.53 |
| Resample, large WAV → 48 kHz | 4 | 20.03 / 124.80 | 20.02 / 124.76 |
| Resample, large WAV → 96 kHz | 4 | 70.03 / 149.38 | 70.02 / 149.40 |
| Resample, large FLAC | 4 | 49.83 / 41.07 | 49.81 / 41.03 |
| Resample, large AIFF | 4 | 36.92 / 17.04 | 36.93 / 17.03 |
| Resample, large MP3 | 4 | 73.42 / 48.77 | 73.39 / 48.75 |
| Resample, large Vorbis/Ogg | 4 | 54.26 / 29.31 | 54.25 / 29.30 |

Native-only large AAC/ADTS, AAC/MP4, and Opus/WebM runs take 36.47, 30.73,
and 73.44 ms respectively for four files. They are not paired speedup claims:
the Python audio helper's libsndfile format coverage does not include these
containers. Encoder padding is retained in each input and output count.

### Audio phase diagnosis

An immediately preceding run of the pre-change installed runtime measured
37.75 / 135.95 ms for the same large WAV workload. The current result
34.74 / 131.73 ms reproduces that regression; the small difference between
these separate runs does not demonstrate a resampling optimization.
That baseline reports Vane `0.2.0.dev627`, native revision `eabaf1a6fb`, and
SourceID `d0db5a3a29`; the CSV preserves these identities separately.

The separate diagnostic invocation for four large WAV files records 1.31 ms
of setup, 161.00 ms of decode, 15.53 ms of resampling, and 5.42 ms of buffer
reservation. Its 0.66 ms of FILE reads overlaps setup/decode. These are
instrumented invocation costs, not a decomposition of the 131.73 ms timed
query. The diagnostic retains 5,120,000 output bytes in the waveform batch.
Decode includes demuxing and codec discovery as well as sample decoding.

A separate GDB trace of one uninstrumented WAV query observes 32 calls to
FFmpeg's `probe_codec` from `av_read_frame` inside `MediaReader::NextFrame`.
The corresponding AIFF trace observes none; both produce 80,000 frames.
FFmpeg's [WAV demuxer](https://github.com/FFmpeg/FFmpeg/blob/n8.1.1/libavformat/wavdec.c)
requests secondary probing for PCM16 little-endian streams, and its
[demuxer implementation](https://github.com/FFmpeg/FFmpeg/blob/n8.1.1/libavformat/demux.c)
buffers packets and probes their contents. This identifies a concrete
container-specific cost inside the measured decode phase. The trace does
not separately time every FFmpeg routine or establish the effect of changing
that probing policy.

### HTTP requests and bytes

These are three-repetition medians from a separate loopback server process.
The delayed case adds 5 ms per request. Counters are per complete query;
each row also has one HEAD request per backend. All responses completed
without transport errors, and aggregate counts match.

| Operation | Files | Delay/request | Wall ms Python / native | GETs Python / native | Body bytes Python / native |
| --- | ---: | ---: | ---: | ---: | ---: |
| Large PNG metadata | 20 | 0 ms | 90.01 / 39.33 | 20 / 40 | 20,971,520 / 660 |
| Large PNG metadata | 20 | 5 ms | 214.87 / 259.98 | 20 / 40 | 20,971,520 / 660 |
| Large PNG decode | 4 | 0 ms | 562.37 / 88.63 | 540 / 16 | 11,077,792 / 11,077,876 |
| Large PNG decode | 4 | 5 ms | 3,577.19 / 213.98 | 540 / 16 | 11,077,792 / 11,077,876 |
| Large WAV resample | 4 | 0 ms | 538.98 / 247.52 | 520 / 68 | 3,840,192 / 4,364,288 |
| Large WAV resample | 4 | 5 ms | 3,367.24 / 713.68 | 520 / 68 | 3,840,192 / 4,364,288 |

Fewer bytes and fewer requests have different effects. Native PNG metadata
reads far fewer bytes but issues two requests per file, so the delayed case
is slower. Native WAV resampling issues fewer requests while reading more
bytes. The HTTP cases use whole files; logical windows and their physical
range boundaries are validated by the separate tests.

### Memory, temporary writes, and concurrent local queries

Fresh processes running eight large files give peak driver RSS of
231.7 / 242.9 MiB for Python/native PNG decoding and 224.8 / 240.5 MiB
for WAV resampling. The Python-only processes do not load native media
extensions. These whole-process peaks include different loaded libraries;
eliminating spools does not establish lower RSS.

The separate Python diagnostic pass writes 22,118,400 bytes for eight PNGs
and 10,240,000 bytes for eight WAVs to `TemporaryFile`, with peak live logical
spool sizes of 2,764,800 and 1,280,000 bytes respectively. No spool remains
open at query completion. The native pass makes no Python `TemporaryFile`
calls. This is media-spool accounting, not process-wide filesystem tracing.

Four independent connections, each with one engine thread, process 32 large
PNGs in 222.31 / 128.55 ms and 16 large WAVs in 69.34 / 168.78 ms
(Python/native). Compared with one connection's 8-PNG and 4-WAV runs,
throughput scales by about 3.06× / 3.55× for PNG and 2.00× / 3.12× for WAV.
The native WAV regression persists at concurrency four. This experiment
measures query concurrency rather than parallelism within a scalar batch.

### Ray execution

These local Ray clusters advertise four CPUs. Each query uses one engine
thread; figures are medians over three repetitions after warmup. Native
operators load the exact signed installed provider on the required nodes.
Timing includes distributed query submission/preparation, media execution,
and aggregate transport. It does not isolate Worker preparation from decoding.

| Operation | Files/query | Concurrent queries | Wall ms Python / native | Driver CPU ms Python / native |
| --- | ---: | ---: | ---: | ---: |
| Large WAV resample | 4 | 1 | 472.99 / 2,560.34 | 66.11 / 71.15 |
| Large WAV resample | 4 | 4 | 683.82 / 3,068.11 | 242.60 / 259.70 |
| Large PNG decode | 8 | 1 | 679.72 / 2,643.30 | 67.06 / 77.00 |
| Large PNG decode | 8 | 4 | 884.22 / 3,121.43 | 256.46 / 261.36 |

All reported groups return matching aggregate counts. Native queries are
slower in these Ray workloads even where local native image decoding is
faster. Driver CPU/RSS exclude the Ray query Driver actor and Workers;
neither those figures nor the local audio profile establish distributed
total CPU or memory cost.

One earlier four-connection image run returned no partitions for an aggregate
query during a timed group. It failed and is excluded from the timing table.
A subsequent independent run completed, but that does not establish that the
intermittent empty-result behavior is resolved. This remains validation
evidence for #763, alongside the passing two-connection mixed-backend tests.

### Measurement records and remaining scope

[measurements-20260906.csv](measurements-20260906.csv) retains all wall/driver
CPU samples, output counts, input names, artifact/runtime/harness identities,
HTTP counters, whole-process peak RSS, and separate local diagnostics for
41 successful cases (75 backend records). Semicolons separate repetitions
or cyclic input names; empty fields mean unmeasured or inapplicable.
[inputs-20260906.csv](inputs-20260906.csv) records the 36 generated fixtures'
sizes and hashes. Media payloads and build artifacts are not committed.

`whole_process_peak_rss_bytes` is shared across both backend rows from one
paired process. Only cases named `*-rss-python` / `*-rss-native` support the
fresh-process backend comparison above. Diagnostic times are summed across
the separate invocation's FILE rows. Maximum sample-buffer capacity is the
largest individual batch workspace observed, not the sum across concurrent
connections. Codec version integers use FFmpeg's packed major/minor/micro
representation.

#764 remains broader than these image/audio measurements: total distributed
CPU/memory, wider video/storage coverage, matched resampler quality, and the
intermittent Ray result gap are not established as complete here. #763 also
retains its cancellation/refresh and distributed stability requirements;
#661 owns the stronger credential-governance contract. These reports make
those boundaries visible rather than treating native execution as a universal
performance improvement.

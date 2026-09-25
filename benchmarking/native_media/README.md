# Native extension measurements

`scripts/benchmark_native_media.py` compares explicitly selected Python and
native backends on synthetic media. It measures complete queries, including
binding and execution, rather than isolating interpreter overhead.

## Reproduce

Use the installed environment and extension build procedure in
[NATIVE_MEDIA_EXTENSIONS.md](../../NATIVE_MEDIA_EXTENSIONS.md). Generate the
inputs outside the checkout; no generated media is committed:

```bash
python -I benchmarking/native_media/generate_inputs.py /tmp/vane-media-inputs
python -I scripts/benchmark_native_media.py image_decode \
  /tmp/vane-media-inputs/image-large.png \
  --extension build/python-release/vane_extensions/native_media.duckdb_extension \
  --rows 8 --repetitions 5 --threads 1 \
  --allow-unsigned-development-artifact
```

Use `image_metadata`, `image_decode`, `audio_metadata`, `audio_resample`, or
`video_frames` for the operation. Record input digests, rows, thread counts,
codec versions and artifact identity for each run. Omit the unsigned-development
flag for a signed artifact. `--runner ray --installed-provider` uses signed
providers installed on every node.

Run each backend in a fresh process for peak RSS. Compare output counts and
contents before interpreting timing differences; matching aggregates alone do
not establish pixel or waveform equality. Include index construction when
measuring a one-off indexed video query.

See [VALIDATION.md](VALIDATION.md) for the input matrix, HTTP and concurrency
options, and metric boundaries; [video seeking](../video_seek/README.md) for
indexed access; and [audio parity](../audio_parity/README.md) for waveform checks.

## Historical measurements

The [original report](https://github.com/AstroVela/vane/blob/4e12994a2fed5b872a7bdb44df72c1b9c5653cdc/benchmarking/native_media/README.md)
contains the observations, timing tables and input digests for each recorded environment. Raw records remain in [measurements-20260906.csv](measurements-20260906.csv)
and [inputs-20260906.csv](inputs-20260906.csv). These measurements describe that
revision and workload; native execution is not uniformly faster.

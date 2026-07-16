from multimodal_inference_benchmarks.audio_transcription import vane_main


def test_audio_benchmark_has_no_count_admission_tuning_helper():
    assert not hasattr(vane_main, "_udf_admission_kwargs")

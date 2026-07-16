import pytest


def test_resolve_video_files_sorts_local_directory(tmp_path):
    from multimodal_inference_benchmarks.video_object_detection.video_inputs import resolve_video_files

    (tmp_path / "z.avi").write_bytes(b"")
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "ignore.txt").write_text("x")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.mov").write_bytes(b"")

    assert resolve_video_files(str(tmp_path)) == [
        str(tmp_path / "a.mp4"),
        str(nested / "b.mov"),
        str(tmp_path / "z.avi"),
    ]


def test_resolve_video_files_uses_manifest_order(tmp_path):
    from multimodal_inference_benchmarks.video_object_detection.video_inputs import resolve_video_files

    manifest = tmp_path / "manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                "# generated input order",
                "/data/video_b.avi",
                "",
                "/data/video_a.avi",
            ]
        )
    )

    assert resolve_video_files("/unused", input_manifest=str(manifest)) == [
        "/data/video_b.avi",
        "/data/video_a.avi",
    ]


def test_resolve_video_files_rejects_empty_manifest(tmp_path):
    from multimodal_inference_benchmarks.video_object_detection.video_inputs import resolve_video_files

    manifest = tmp_path / "manifest.txt"
    manifest.write_text("# no files\n\n")

    with pytest.raises(ValueError, match="manifest is empty"):
        resolve_video_files("/unused", input_manifest=str(manifest))


def test_ray_data_read_paths_strip_s3_scheme_for_filesystem_reads():
    from multimodal_inference_benchmarks.video_object_detection.video_inputs import ray_data_read_paths

    assert ray_data_read_paths(["s3://bucket/a.mp4", "/data/b.mp4"]) == [
        "bucket/a.mp4",
        "/data/b.mp4",
    ]


def test_has_s3_video_files_rejects_mixed_local_and_s3_sources():
    from multimodal_inference_benchmarks.video_object_detection.video_inputs import has_s3_video_files

    with pytest.raises(ValueError, match="all S3 or all local"):
        has_s3_video_files(["s3://bucket/a.mp4", "/data/b.mp4"])

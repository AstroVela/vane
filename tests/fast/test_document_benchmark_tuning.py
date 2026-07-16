from multimodal_inference_benchmarks.document_embedding import vane_main


def test_document_cpu_udfs_use_explicit_compute_batch_sizes():
    pdf_kwargs = vane_main._pdf_udf_kwargs()
    chunk_kwargs = vane_main._chunk_udf_kwargs()

    assert pdf_kwargs == {
        "batch_size": vane_main.PDF_EXTRACTION_BATCH_SIZE,
        "streaming_breaker": True,
    }
    assert chunk_kwargs == {
        "batch_size": vane_main.TEXT_CHUNK_BATCH_SIZE,
        "streaming_breaker": True,
    }
    assert 0 < vane_main.PDF_EXTRACTION_BATCH_SIZE < 2048
    assert 0 < vane_main.TEXT_CHUNK_BATCH_SIZE < 2048


def test_document_compute_batch_sizes_are_not_admission_watermarks():
    removed_admission_keys = {
        "max_outstanding_batches",
        "max_ready_rows",
        "max_ready_bytes",
        "max_pending_bytes",
    }

    assert removed_admission_keys.isdisjoint(vane_main._pdf_udf_kwargs())
    assert removed_admission_keys.isdisjoint(vane_main._chunk_udf_kwargs())

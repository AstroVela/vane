# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

import pyarrow as pa
import pymupdf
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import vane

DATA_ROOT = Path("/data/multimodal_inference_benchmarks/digitalcorpora")
INPUT_PATH = Path(os.environ.get("INPUT_PATH", DATA_ROOT / "metadata")).expanduser()
PDF_ROOT = Path(os.environ.get("LOCAL_PDF_ROOT", DATA_ROOT / "pdf_dump")).expanduser()
OUTPUT_DIR = Path(os.environ.get("OUTPUT_PATH", f"/tmp/vane_document_{uuid.uuid4().hex}")).expanduser()
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "10"))
NUM_GPU_NODES = int(os.environ.get("NUM_GPU_NODES", "1"))

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_PDF_PAGES = 100
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 200

if BATCH_SIZE <= 0 or NUM_GPU_NODES <= 0:
    raise ValueError("BATCH_SIZE and NUM_GPU_NODES must be positive")


def _parquet_input(path: Path) -> str:
    return str(path / "**/*.parquet") if path.is_dir() else str(path)


def _local_pdf_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() and path.exists():
        return path
    normalized = value.replace("\\", "/")
    marker = "pdf_dump/"
    relative = normalized.split(marker, 1)[1] if marker in normalized else path.name
    return PDF_ROOT / relative


def _model_path() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_ID, local_files_only=True)


def extract_text_from_pdf(row):
    path = _local_pdf_path(row["uploaded_pdf_path"])
    try:
        with pymupdf.open(path) as document:
            if len(document) > MAX_PDF_PAGES:
                return
            for page in document:
                yield {
                    "uploaded_pdf_path": str(path),
                    "page_number": int(page.number),
                    "page_text": page.get_text(),
                }
    except Exception as exc:
        print(f"Error extracting text from PDF {path}: {exc}", file=sys.stderr, flush=True)
        return


def chunker(row):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    for chunk_id, text in enumerate(splitter.split_text(row["page_text"])):
        yield {
            "uploaded_pdf_path": row["uploaded_pdf_path"],
            "page_number": row["page_number"],
            "chunk_id": chunk_id,
            "chunk": text,
        }


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(_model_path(), device="cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, table):
        chunks = table.column("chunk").to_pylist()
        encoded = self.model.encode(
            chunks,
            show_progress_bar=False,
            convert_to_tensor=True,
        )
        if not isinstance(encoded, torch.Tensor):
            raise TypeError(f"Expected tensor embeddings, got {type(encoded)!r}")
        encoded = encoded.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if tuple(encoded.shape) != (len(chunks), EMBEDDING_DIM):
            raise ValueError(f"Unexpected embedding shape: {tuple(encoded.shape)}")
        values = pa.array(encoded.reshape(-1).numpy(), type=pa.float32())
        embeddings = pa.FixedSizeListArray.from_arrays(values, EMBEDDING_DIM)
        return pa.table(
            {
                "uploaded_pdf_path": table.column("uploaded_pdf_path"),
                "page_number": table.column("page_number"),
                "chunk_id": table.column("chunk_id"),
                "chunk": table.column("chunk"),
                "embedding": embeddings,
            }
        )


def main() -> None:
    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    con = vane.connect()
    try:
        rel = con.read_parquet(_parquet_input(INPUT_PATH))
        rel = rel.filter("ends_with(file_name, '.pdf')").project("uploaded_pdf_path")
        rel = rel.flat_map(
            extract_text_from_pdf,
            schema={
                "uploaded_pdf_path": vane.sqltypes.VARCHAR,
                "page_number": vane.sqltypes.INTEGER,
                "page_text": vane.sqltypes.VARCHAR,
            },
        )
        rel = rel.flat_map(
            chunker,
            schema={
                "uploaded_pdf_path": vane.sqltypes.VARCHAR,
                "page_number": vane.sqltypes.INTEGER,
                "chunk_id": vane.sqltypes.INTEGER,
                "chunk": vane.sqltypes.VARCHAR,
            },
        )
        rel = rel.map_batches(
            Embedder,
            schema={
                "uploaded_pdf_path": vane.sqltypes.VARCHAR,
                "page_number": vane.sqltypes.INTEGER,
                "chunk_id": vane.sqltypes.INTEGER,
                "chunk": vane.sqltypes.VARCHAR,
                "embedding": vane.array_type(vane.sqltypes.FLOAT, EMBEDDING_DIM),
            },
            batch_size=BATCH_SIZE,
            actor_number=NUM_GPU_NODES,
            gpus=1.0,
        )
        rel.write_parquet(str(OUTPUT_DIR / "result.parquet"))
    finally:
        con.close()

    print(f"Runtime: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()

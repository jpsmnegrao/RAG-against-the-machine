"""Tests for the indexer functions - chunk extractor, bm25 scorer and file
exporter."""

import bm25s
import pickle
from pathlib import Path

from src.schemas import IndexedChunk
from src.chunker import chunkinizer
from src.indexer import chunk_collector, bm25_scorer, index_exporter


def test_chunk_collector_collects_supported_files(
        tmp_path: Path, sample_python: str, sample_markdown: str,
        sample_txt: str) -> None:
    """Verify supported files are collected and chunked."""
    (tmp_path / "a.py").write_text(sample_python)
    (tmp_path / "b.md").write_text(sample_markdown)
    (tmp_path / "c.txt").write_text(sample_txt)
    (tmp_path / "d.json").write_text('{"key": "value"}')
    subdir = tmp_path / "nested"
    subdir.mkdir()
    (subdir / "e.py").write_text(sample_python)
    found: list[IndexedChunk] = chunk_collector(tmp_path, 2000)
    assert found is not None
    file_paths = {chunk.minimalsource.file_path for chunk in found}
    assert file_paths == {
        str(tmp_path / "a.py"),
        str(tmp_path / "b.md"),
        str(tmp_path / "c.txt"),
        str(tmp_path / "nested" / "e.py"),
    }
    assert str(tmp_path / "d.json") not in file_paths


def test_chunk_collector_respects_max_chunk_size(
        tmp_path: Path, sample_python: str) -> None:
    """Verify collected chunks respect the configured maximum size."""
    file = tmp_path / "sample.py"
    file.write_text(sample_python)
    chunks = chunk_collector(tmp_path, 50)
    for chunk in chunks:
        assert len(chunk.text) <= 50


def test_bm25_scorer_creates_index(tmp_path: Path, sample_python: str) -> None:
    """Verify a BM25 index is created from multiple chunks."""
    file = tmp_path / "sample.py"
    file.write_text(sample_python)
    chunks = chunkinizer(file, 200)
    index = bm25_scorer(chunks)
    assert isinstance(index, bm25s.BM25)


def test_index_exporter_creates_files(
        tmp_path: Path, sample_python: str) -> None:
    """Verify the index and corpus files are created."""
    file = tmp_path / "sample.py"
    file.write_text(sample_python)
    chunks = chunkinizer(file, 200)
    index = bm25_scorer(chunks)
    index_exporter(index, chunks, tmp_path)
    assert (tmp_path / "index.pkl").is_file()
    assert (tmp_path / "corpus.pkl").is_file()


def test_index_exporter_preserves_corpus(
        tmp_path: Path, sample_python: str) -> None:
    """Verify the exported corpus can be loaded unchanged."""
    src_file = tmp_path / "sample.py"
    src_file.write_text(sample_python)
    chunks = chunkinizer(src_file, 200)
    index = bm25_scorer(chunks)
    index_exporter(index, chunks, tmp_path)
    with open(tmp_path / "corpus.pkl", "rb") as file:
        loaded_chunks = pickle.load(file)
    assert loaded_chunks == chunks


def test_index_exporter_overwrites_existing_files(
        tmp_path: Path, sample_python: str) -> None:
    """Verify existing index and corpus files are overwritten."""
    src_file = tmp_path / "sample.py"
    src_file.write_text(sample_python)
    chunks = chunkinizer(src_file, 200)
    index = bm25_scorer(chunks)
    (tmp_path / "index.pkl").write_bytes(b"old index")
    (tmp_path / "corpus.pkl").write_bytes(b"old corpus")
    index_exporter(index, chunks, tmp_path)
    with open(tmp_path / "index.pkl", "rb") as file:
        loaded_index: bm25s.BM25 = pickle.load(file)
    with open(tmp_path / "corpus.pkl", "rb") as file:
        loaded_chunks: list[IndexedChunk] = pickle.load(file)
    assert isinstance(loaded_index, bm25s.BM25)
    assert loaded_chunks == chunks

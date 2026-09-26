"""Tests for the indexer functions - chunk extractor, bm25 scorer and file
exporter."""

import bm25s
import os
import pickle
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.schemas import IndexedChunk, MinimalSource
from src.chunker import chunkinizer
from src.indexer import chunk_collector, bm25_scorer, index_exporter


def test_bm25_scorer_creates_index(tmp_path: Path, sample_python: str) -> None:
    """Verify a BM25 index is created from multiple chunks."""
    file = tmp_path / "sample.py"
    file.write_text(sample_python)
    chunks = chunkinizer(file, 200)
    index = bm25_scorer(chunks)
    assert isinstance(index, bm25s.BM25)


def test_bm25_scorer_empty_chunks() -> None:
    """Verify BM25 scorer raises an exception for empty chunks."""
    with pytest.raises(Exception):
        bm25_scorer([])


def test_bm25_scorer_large_corpus(sample_python: str) -> None:
    """Verify BM25 scorer handles a large volume of input chunks."""
    source = MinimalSource(
        file_path="test.py", first_character_index=0, last_character_index=10)
    large_chunks = (
        [IndexedChunk(minimalsource=source, text=sample_python * 100)
            for _ in range(50)])
    scorer = bm25_scorer(large_chunks)
    assert scorer is not None


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


def test_chunk_collector_edge_case_content(tmp_path: Path) -> None:
    """Verify chunk collector handles files with whitespace and special
    characters."""
    file = tmp_path / "edge_cases.py"
    file.write_text("   \n\t   # @!#$*&^% \n   ")
    chunks = chunk_collector(tmp_path, 2000)
    assert isinstance(chunks, list)


def test_chunk_collector_empty_directory(tmp_path: Path) -> None:
    """Test chunk_collector raises ValueError on empty directory."""
    with pytest.raises(ValueError, match="No corpus files found"):
        chunk_collector(tmp_path, max_chunk_size=2000)


@patch("src.indexer.extract_corpus_files")
def test_chunk_collector_empty_chunks(
        mock_extract: MagicMock, tmp_path: Path) -> None:
    """Test chunk_collector raises ValueError on empty chunk output."""
    mock_extract.return_value = [tmp_path / "dummy.txt"]
    with patch("src.indexer.chunkinizer", return_value=[]):
        with pytest.raises(ValueError, match="No chunks were successfully "
                                             "extracted"):
            chunk_collector(tmp_path, max_chunk_size=2000)


def test_chunk_collector_zero_or_negative_max_chunk_size(
    tmp_path: Path, sample_python: str
) -> None:
    """Verify chunk_collector rejects non-positive max_chunk_size values."""
    file = tmp_path / "sample.py"
    file.write_text(sample_python)
    for invalid_size in [0, -1, -500]:
        with pytest.raises(ValueError):
            chunk_collector(tmp_path, max_chunk_size=invalid_size)


def test_chunk_collector_invalid_max_chunk_size(
    tmp_path: Path, sample_python: str
) -> None:
    """Check chunk_collector enforces non-int or malformed max_chunk_size."""
    file = tmp_path / "sample.py"
    file.write_text(sample_python)
    for invalid_size in ["2000", None, 3.5]:
        with pytest.raises((TypeError, ValueError)):
            chunk_collector(
                tmp_path,
                max_chunk_size=invalid_size)  # type: ignore[arg-type]


def test_chunk_collector_respects_max_chunk_size(
        tmp_path: Path, sample_python: str) -> None:
    """Verify collected chunks respect the configured maximum size."""
    file = tmp_path / "sample.py"
    file.write_text(sample_python)
    chunks = chunk_collector(tmp_path, 50)
    for chunk in chunks:
        assert len(chunk.text) <= 50


def test_chunk_collector_nonexistent_directory(tmp_path: Path) -> None:
    """Check chunk collector raises FileNotFoundError for missing directory."""
    missing_dir = tmp_path / "definitely_does_not_exist_12345"
    with pytest.raises(FileNotFoundError):
        chunk_collector(missing_dir, 2000)


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


def test_index_exporter_invalid_path(tmp_path: Path) -> None:
    """Verify index exporter raises errors when target path is invalid."""
    blocking_file = tmp_path / "not_a_directory"
    blocking_file.write_text("blocking")
    invalid_path = blocking_file / "subdir"
    with pytest.raises((NotADirectoryError, FileNotFoundError, OSError)):
        index_exporter(None, [], invalid_path)


def test_index_exporter_permission_error(tmp_path: Path) -> None:
    """Verify index exporter handles permission errors during export."""
    if os.geteuid() == 0:
        pytest.skip("Permission checks are meaningless when running as root.")
    restricted_dir = tmp_path / "restricted"
    restricted_dir.mkdir(mode=0o400)
    target_file = restricted_dir / "index.json"
    with pytest.raises((PermissionError, OSError)):
        index_exporter(None, [], target_file)


def test_index_exporter_invalid_arguments() -> None:
    """Verify idx_exporter handles null/malformed model/chunk objs safely."""
    with pytest.raises((AttributeError, TypeError, ValueError)):
        index_exporter(None, [], Path("/tmp/invalid_export_target"))

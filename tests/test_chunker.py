"""Tests for the file chunking and corpus file extraction functions."""


import pytest
from pathlib import Path
from langchain_text_splitters import Language

from src.chunker import file_chunker, chunkinizer, extract_corpus_files
from src.schemas import IndexedChunk


def test_file_chunker_respects_max_chunk_size(
        tmp_path: Path, sample_python: str) -> None:
    """Verify Python chunks do not exceed the requested maximum size."""
    file = tmp_path / "python_sample.py"
    file.write_text(sample_python)
    chunks: list[IndexedChunk] = file_chunker(file, 200, Language.PYTHON)
    for chunk in chunks:
        assert len(chunk.text) <= 200


def test_file_chunker_offsets_match_text(
        tmp_path: Path, sample_python: str) -> None:
    """Verify Python chunk offsets identify the corresponding source text."""
    file = tmp_path / "python_sample.py"
    file.write_text(sample_python)
    chunks: list[IndexedChunk] = file_chunker(file, 200, Language.PYTHON)
    raw_text = file.read_text()
    for chunk in chunks:
        start = chunk.minimalsource.first_character_index
        end = chunk.minimalsource.last_character_index
        assert raw_text[start:end] == chunk.text


def test_file_chunker_markdown_respects_max_chunk_size(
        tmp_path: Path, sample_markdown: str) -> None:
    """Verify Markdown chunks do not exceed the requested maximum size."""
    file = tmp_path / "markdown_sample.md"
    file.write_text(sample_markdown)
    chunks: list[IndexedChunk] = file_chunker(file, 100, Language.MARKDOWN)
    for chunk in chunks:
        assert len(chunk.text) <= 100


def test_markdown_chunker_offsets_match_text(
        tmp_path: Path, sample_markdown: str) -> None:
    """Verify Markdown chunk offsets identify the corresponding source text."""
    file = tmp_path / "markdown_sample.md"
    file.write_text(sample_markdown)
    chunks: list[IndexedChunk] = file_chunker(file, 100, Language.MARKDOWN)
    raw_text = file.read_text()
    for chunk in chunks:
        start = chunk.minimalsource.first_character_index
        end = chunk.minimalsource.last_character_index
        assert raw_text[start:end] == chunk.text


def test_chunkinizer_python_file_path_is_correct(
        tmp_path: Path, sample_python: str) -> None:
    """Verify chunks produced from a Python file report the correct
    file_path."""
    file = tmp_path / "sample.py"
    file.write_text(sample_python)
    chunks = chunkinizer(file, 200)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.minimalsource.file_path == str(file)


def test_chunkinizer_markdown_file_path_is_correct(
        tmp_path: Path, sample_markdown: str) -> None:
    """Verify chunks produced from a Markdown file report the correct
    file_path."""
    file = tmp_path / "sample.md"
    file.write_text(sample_markdown)
    chunks = chunkinizer(file, 100)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.minimalsource.file_path == str(file)


def test_chunkinizer_txt_file_path_is_correct(
        tmp_path: Path, sample_txt: str) -> None:
    """Verify chunks produced from a text file report the correct file_path."""
    file = tmp_path / "sample.txt"
    file.write_text(sample_txt)
    chunks = chunkinizer(file, 100)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.minimalsource.file_path == str(file)


def test_chunkinizer_skips_unsupported_extension(tmp_path: Path) -> None:
    """Verify files with unsupported extensions are ignored."""
    file = tmp_path / "sample.json"
    file.write_text('{"key": "value"}')
    chunks = chunkinizer(file, 100)
    assert chunks == []


def test_extract_corpus_files_finds_expected_files(
    tmp_path: Path, sample_python: str, sample_markdown: str, sample_txt: str,
        ) -> None:
    """Verify corpus extraction finds supported files recursively."""
    (tmp_path / "a.py").write_text(sample_python)
    (tmp_path / "b.md").write_text(sample_markdown)
    (tmp_path / "c.txt").write_text(sample_txt)
    (tmp_path / "d.json").write_text('{"key": "value"}')
    subdir = tmp_path / "nested"
    subdir.mkdir()
    (subdir / "e.py").write_text(sample_python)
    found = extract_corpus_files(tmp_path)
    found_names = {f.name for f in found}
    assert found_names == {"a.py", "b.md", "c.txt", "e.py"}


def test_file_chunker_handles_empty_file(tmp_path: Path) -> None:
    """Verify an empty file produces no chunks."""
    file = tmp_path / "empty.py"
    file.write_text("")
    chunks = file_chunker(file, 200, Language.PYTHON)
    assert chunks == []


def test_chunkinizer_handles_invalid_encoding(tmp_path: Path) -> None:
    """Verify chunkinizer handles files with invalid encoding gracefully."""
    file = tmp_path / "bad_encoding.py"
    file.write_bytes(b"\xff\xfe\xfd")
    chunks = chunkinizer(file, 200)
    assert chunks == []


def test_chunkinizer_handles_nonexistent_file(tmp_path: Path) -> None:
    """Verify chunkinizer handles nonexistent files gracefully."""
    file = tmp_path / "does_not_exist.py"
    chunks = chunkinizer(file, 200)
    assert chunks == []


def test_chunkinizer_handles_invalid_max_chunk(
        tmp_path: Path, sample_python: str) -> None:
    """Verify chunkinizer raises ValueError for non-positive chunk sizes."""
    file = tmp_path / "test.py"
    file.write_text("print('hello')")
    with pytest.raises(ValueError):
        chunkinizer(file, max_chunk=0)


def test_extract_corpus_files_nonexistent_directory(tmp_path: Path) -> None:
    """Verify corpus extraction handles nonexistent directories safely."""
    missing_dir = tmp_path / "nonexistent"
    found = extract_corpus_files(missing_dir)
    assert found == []


def test_chunkinizer_unsupported_extension(tmp_path: Path) -> None:
    """Verify chunker ignores unsupported file extensions."""
    file = tmp_path / "image.png"
    file.write_bytes(b"fake image data")
    chunks = chunkinizer(file, 200)
    assert chunks == []


def test_chunkinizer_empty_file(tmp_path: Path) -> None:
    """Verify chunker handles empty files correctly."""
    file = tmp_path / "empty.txt"
    file.write_text("")
    chunks = chunkinizer(file, 200)
    assert chunks == []


def test_chunkinizer_is_a_directory(tmp_path: Path) -> None:
    """Verify chunker handles directory paths gracefully."""
    sub_dir = tmp_path / "subdir"
    sub_dir.mkdir()
    chunks = chunkinizer(sub_dir, 200)
    assert chunks == []


def test_extract_corpus_files(tmp_path: Path) -> None:
    """Verify corpus extraction filters supported and unsupported files."""
    (tmp_path / "script.py").write_text("print(1)")
    (tmp_path / "notes.md").write_text("# Notes")
    (tmp_path / "readme.txt").write_text("Hello")
    (tmp_path / "data.json").write_text("{}")
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "sub_script.py").write_text("print(2)")
    files = extract_corpus_files(tmp_path)
    relative_files = {f.name for f in files}
    assert relative_files == {"script.py", "notes.md", "readme.txt",
                              "sub_script.py"}
    assert "data.json" not in relative_files

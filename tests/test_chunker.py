"""mambo jambo"""
# Needed fixtures
# A small sample .py file with real structure (a function or two, maybe a class) — either a tiny hand-written fixture file on disk, or a tmp_path/pytest.fixture-created file at test time with a hardcoded string content.
# A small sample .md file with headers/paragraphs.
# A small sample .txt file.
# An unsupported-extension file (e.g. .json) for the "skip" test.
# An empty file (.py or .md).
# A small fixture directory containing a mix of the above for the extract_corpus_files test.

# test_python_chunker_respects_max_chunk_size — chunk a .py fixture, assert every chunk's text length ≤ max_chunk_size.
# test_python_chunker_offsets_match_text — assert raw_text[start:end] == chunk.text for every chunk.
# test_markdown_chunker_respects_max_chunk_size — same as 1, for a .md/.txt fixture.
# test_markdown_chunker_offsets_match_text — same as 2, for a .md/.txt fixture.
# test_chunkinizer_routes_python — .py file → non-empty chunk list.
# test_chunkinizer_routes_markdown — .md file → non-empty chunk list.
# test_chunkinizer_routes_txt — .txt file → non-empty chunk list.
# test_chunkinizer_skips_unsupported_extension — e.g. .json file → [].
# test_extract_corpus_files_finds_expected_files — fixture directory with a mix of .py/.md/.txt/other → returns exactly the relevant ones.
# test_chunker_handles_empty_file — empty .py (and/or .md) file → [] or no crash.

from pathlib import Path
import pytest

from chunker import file_chunker, chunkinizer, extract_corpus_files
from langchain_text_splitters import Language


def test_python_chunker_respects_max_chunk_size(...):
    ...


def test_python_chunker_offsets_match_text(...):
    ...


def test_markdown_chunker_respects_max_chunk_size(...):
    ...


def test_markdown_chunker_offsets_match_text(...):
    ...


def test_chunkinizer_routes_python(...):
    ...


def test_chunkinizer_routes_markdown(...):
    ...


def test_chunkinizer_routes_txt(...):
    ...


def test_chunkinizer_skips_unsupported_extension(...):
    ...


def test_extract_corpus_files_finds_expected_files(...):
    ...


def test_chunker_handles_empty_file(...):
    ...


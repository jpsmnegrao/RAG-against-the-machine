"""mambo jambo"""


from langchain_text_splitters import Language

from src.chunker import file_chunker, chunkinizer, extract_corpus_files
from src.schemas import IndexedChunk


SAMPLE_PYTHON = '''\
def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


class Calculator:
    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        return a / b
'''

SAMPLE_MARKDOWN = '''\
# Title

Some introductory paragraph explaining the topic.

## Section One

Content for section one goes here, with a bit more detail.

## Section Two

Content for section two, also with some detail to pad it out.
'''

SAMPLE_TXT = '''\
This is a plain text file.
It has a few lines of unstructured content.
Nothing fancy, just prose.
'''


def test_python_chunker_respects_max_chunk_size(tmp_path) -> None:
    file = tmp_path / "python_sample.py"
    file.write_text(SAMPLE_PYTHON)
    chunks: list[IndexedChunk] = file_chunker(file, 200, Language.PYTHON)
    for chunk in chunks:
        assert len(chunk.text) <= 200


def test_python_chunker_offsets_match_text(tmp_path) -> None:
    file = tmp_path / "python_sample.py"
    file.write_text(SAMPLE_PYTHON)
    chunks: list[IndexedChunk] = file_chunker(file, 200, Language.PYTHON)
    raw_text = file.read_text()
    for chunk in chunks:
        start = chunk.minimalsource.first_character_index
        end = chunk.minimalsource.last_character_index
        assert raw_text[start:end] == chunk.text


def test_markdown_chunker_respects_max_chunk_size(tmp_path) -> None:
    file = tmp_path / "markdown_sample.md"
    file.write_text(SAMPLE_MARKDOWN)
    chunks: list[IndexedChunk] = file_chunker(file, 100, Language.MARKDOWN)
    for chunk in chunks:
        assert len(chunk.text) <= 100


def test_markdown_chunker_offsets_match_text(tmp_path) -> None:
    file = tmp_path / "markdown_sample.md"
    file.write_text(SAMPLE_MARKDOWN)
    chunks: list[IndexedChunk] = file_chunker(file, 100, Language.MARKDOWN)
    raw_text = file.read_text()
    for chunk in chunks:
        start = chunk.minimalsource.first_character_index
        end = chunk.minimalsource.last_character_index
        assert raw_text[start:end] == chunk.text


def test_chunkinizer_routes_python(tmp_path) -> None:
    file = tmp_path / "sample.py"
    file.write_text(SAMPLE_PYTHON)
    chunks = chunkinizer(file, 200)
    assert len(chunks) > 0


def test_chunkinizer_routes_markdown(tmp_path) -> None:
    file = tmp_path / "sample.md"
    file.write_text(SAMPLE_MARKDOWN)
    chunks = chunkinizer(file, 100)
    assert len(chunks) > 0


def test_chunkinizer_routes_txt(tmp_path) -> None:
    file = tmp_path / "sample.txt"
    file.write_text(SAMPLE_TXT)
    chunks = chunkinizer(file, 100)
    assert len(chunks) > 0


def test_chunkinizer_skips_unsupported_extension(tmp_path) -> None:
    file = tmp_path / "sample.json"
    file.write_text('{"key": "value"}')
    chunks = chunkinizer(file, 100)
    assert chunks == []


def test_extract_corpus_files_finds_expected_files(tmp_path) -> None:
    (tmp_path / "a.py").write_text(SAMPLE_PYTHON)
    (tmp_path / "b.md").write_text(SAMPLE_MARKDOWN)
    (tmp_path / "c.txt").write_text(SAMPLE_TXT)
    (tmp_path / "d.json").write_text('{"key": "value"}')

    subdir = tmp_path / "nested"
    subdir.mkdir()
    (subdir / "e.py").write_text(SAMPLE_PYTHON)

    found = extract_corpus_files(tmp_path)
    found_names = {f.name for f in found}

    assert found_names == {"a.py", "b.md", "c.txt", "e.py"}


def test_chunker_handles_empty_file(tmp_path) -> None:
    file = tmp_path / "empty.py"
    file.write_text("")
    chunks = file_chunker(file, 200, Language.PYTHON)
    assert chunks == []

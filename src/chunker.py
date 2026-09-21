"""Utilities for splitting Python, Markdown, and text files into indexed
chunks."""


from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from pathlib import Path
from src.schemas import IndexedChunk, MinimalSource


def file_chunker(
        file: Path, max_chunk: int, language: Language | None
        ) -> list[IndexedChunk]:
    """Splits a file into indexed text chunks.

    Args:
        file: Path to the file to chunk.
        max_chunk: Maximum number of characters per chunk.
        language: Programming or markup language used for language-aware
            splitting. If None, a generic text splitter is used.

    Returns:
        A list of IndexedChunk objects containing the chunk text and its source
        file and character indices.
    """
    chunks: list[IndexedChunk] = []
    raw_text: str = file.read_text()
    if language is not None:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=max_chunk,
            chunk_overlap=max_chunk // 10,
            keep_separator=True,
            add_start_index=True,
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk,
            chunk_overlap=max_chunk // 10,
            keep_separator=True,
            add_start_index=True,
        )
    raw_chunks: list[Document] = splitter.create_documents([raw_text])
    for document in raw_chunks:
        source: MinimalSource = MinimalSource(
            file_path=str(file),
            first_character_index=document.metadata["start_index"],
            last_character_index=(document.metadata["start_index"] +
                                  len(document.page_content))
        )
        chunk: IndexedChunk = IndexedChunk(
            minimalsource=source, text=document.page_content)
        chunks.append(chunk)
    return chunks


def chunkinizer(file_path: Path, max_chunk: int) -> list[IndexedChunk]:
    """Chunks Python, Markdown, and text files.

    Files with the .py, .md, and .txt extensions are chunked using
    language-specific splitting for Python and Markdown, and generic text
    splitting for text files. Files with other extensions are ignored.

    Args:
        file_path: Path to the file to chunk.
        max_chunk: Maximum number of characters per chunk.

    Returns:
        A list of IndexedChunk objects, or an empty list if the file format is
        not supported.
    """
    if file_path.suffix == ".py":
        language = Language.PYTHON
    elif file_path.suffix == ".md":
        language = Language.MARKDOWN
    elif file_path.suffix == ".txt":
        language = None
    else:
        return []
    return file_chunker(file_path, max_chunk, language)


def extract_corpus_files(raw_data: Path) -> list[Path]:
    """Recursively finds supported corpus files in a directory.

    Args:
        raw_data: Root directory containing the corpus files.

    Returns:
        A list of paths to Python, Markdown, and text files found under the
        given directory.
    """
    corpus_files: list[Path] = []
    for f in raw_data.rglob("*"):
        if f.suffix in {".py", ".md", ".txt"}:
            corpus_files.append(f)
    return corpus_files

"""chunk_python/chunk_markdown/chunk_file"""


from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from pathlib import Path
from schemas import IndexedChunk, MinimalSource


def file_chunker(
        file: Path, max_chunk: int, language: Language | None
        ) -> list[IndexedChunk]:
    """jambo"""
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
    """mambo"""
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
    """mambo"""
    corpus_files: list[Path] = []
    for f in raw_data.rglob("*"):
        if f.suffix in {".py", ".md", ".txt"}:
            corpus_files.append(f)
    return corpus_files

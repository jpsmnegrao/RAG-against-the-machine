"""chunk_python/chunk_markdown/chunk_file"""


# from typing import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from pathlib import Path
from schemas import IndexedChunk, MinimalSource


def python_chunker(file: Path, max_chunk: int) -> list[IndexedChunk]:
    """jambo"""
    chunks: list[IndexedChunk] = []
    raw_text: str = file.read_text()
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
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


def text_chunker(file: Path, max_chunk: int) -> list[IndexedChunk]:
    """jambo"""
    chunks: list[IndexedChunk] = []

    return chunks


def chunkinizer(file_path: Path, max_chunk: int) -> list[IndexedChunk]:
    """mambo"""
    chunks: list[IndexedChunk] = []
    if file_path.suffix == ".py":
        chunks.append(python_chunker(file_path, max_chunk))
    elif file_path.suffix == ".md" or file_path.suffix == ".txt":
        chunks.append(text_chunker(file_path, max_chunk))
    return chunks


def extract_corpus_files(raw_data: Path) -> list[Path]:
    """mambo"""
    corpus_files: list[Path] = []
    for f in raw_data.rglob("*"):
        if f.suffix in {".py", ".md", ".txt"}:
            corpus_files.append(f)
    return corpus_files


if __name__ == "__main__":
    chunks = python_chunker(Path("data/raw/vllm-0.10.1/find_cuda_init.py"), 500)
    for chunk in chunks:
        print(chunk.text)
        print()
        print()
    print(len(extract_corpus_files(Path("data/raw"))))

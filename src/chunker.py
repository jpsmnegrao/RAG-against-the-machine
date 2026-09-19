"""chunk_python/chunk_markdown/chunk_file"""


from pathlib import Path
from schemas import IndexedChunk


def python_chunker(file: Path, max_chunk: int) -> list[IndexedChunk]:
    """jambo"""
    chunks: list[IndexedChunk] = []

    return chunks


def text_chunker(file: Path, max_chunk: int) -> list[IndexedChunk]:
    """jambo"""
    chunks: list[IndexedChunk] = []

    return chunks


def chunkinizer(file_path: Path, max_chunk: int) -> list[IndexedChunk]:
    """mambo"""
    chunks: list[IndexedChunk] = []
    if file_path.suffix == ".py":
        chunks = python_chunker(file_path, max_chunk)
    elif file_path.suffix == ".md" or file_path.suffix == ".txt":
        chunks = text_chunker(file_path, max_chunk)
    return chunks


def extract_corpus_files(raw_data: Path) -> list[Path]:
    """mambo"""
    corpus_files: list[Path] = []
    for f in raw_data.rglob("*"):
        if f.suffix in {".py", ".md", ".txt"}:
            corpus_files.append(f)
    return corpus_files


if __name__ == "__main__":
    print(len(extract_corpus_files(Path("data/raw"))))

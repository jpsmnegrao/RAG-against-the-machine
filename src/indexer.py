"""imports and orchestrates the corpus walk + index building + persistence"""


from pathlib import Path
from bm25s import BM25

from src.schemas import IndexedChunk
from src.chunker import chunkinizer, extract_corpus_files


def bm25_scorer(chunks: list[IndexedChunk]) -> BM25:
    """mambo"""


def chunk_collector(raw_dir: Path, max_chunk_size: int) -> list[IndexedChunk]:
    """mambo"""
    chunks: list[IndexedChunk] = []
    for f in extract_corpus_files(raw_dir):
        chunks.extend(chunkinizer(f, max_chunk_size))
    return chunks


def index_exporter(bm25: BM25, chunks: list[IndexedChunk], path: Path) -> None:
    """mambo"""

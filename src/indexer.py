"""Retrieval-Augmented Generation (RAG) indexing module.

Handles corpus file extraction, text chunking, BM25 index creation, and
artifact persistence to disk.
"""

import bm25s
import pickle
from pathlib import Path
from tqdm import tqdm

from src.schemas import IndexedChunk
from src.chunker import chunkinizer, extract_corpus_files


def bm25_scorer(chunks: list[IndexedChunk]) -> bm25s.BM25:
    """Tokenize text chunks and build a BM25 index scorer.

    Args:
        chunks: List of indexed text chunks.

    Returns:
        The fitted BM25 index retriever instance.
    """
    if not chunks:
        raise ValueError("Cannot build a BM25 index from an empty chunk list.")
    retriever = bm25s.BM25()
    texts = [chunk.text for chunk in chunks]
    with tqdm(total=len(texts), desc="Tokenizing", unit=" chunk") as bar:
        tokenized_corpus = bm25s.tokenize(texts, show_progress=False)
        bar.update(len(texts))
    retriever.index(tokenized_corpus)
    return retriever


def chunk_collector(raw_dir: Path, max_chunk_size: int) -> list[IndexedChunk]:
    """Collect and chunk all raw corpus documents from a directory.

    Args:
        raw_dir: Path to the raw documents directory.
        max_chunk_size: Maximum character count per text chunk.

    Returns:
        A combined list of all generated indexed chunks.
    """
    if not isinstance(max_chunk_size, int) or isinstance(max_chunk_size, bool):
        raise TypeError("max_chunk_size must be an integer")
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be a positive integer")
    if not raw_dir.exists() or not raw_dir.is_dir():
        raise FileNotFoundError("Raw directory not found or is not a directory"
                                f": {raw_dir}")
    files = extract_corpus_files(raw_dir)
    if not files:
        raise ValueError(f"No corpus files found in: {raw_dir}")
    chunks: list[IndexedChunk] = []
    for f in tqdm(files, desc="Chunking", total=len(files), unit=" file"):
        chunks.extend(chunkinizer(f, max_chunk_size))
    if not chunks:
        raise ValueError("No chunks were successfully extracted from: "
                         f"{raw_dir}")
    return chunks


def index_exporter(
        bm25: bm25s.BM25, chunks: list[IndexedChunk], path: Path) -> None:
    """Serialize and save the BM25 index and corpus chunks to disk.

    Args:
        bm25: The trained BM25 index instance.
        chunks: The list of corpus chunks.
        path: Target directory path for saving artifacts.
    """
    if not isinstance(path, Path):
        raise TypeError("Path must be a pathlib.Path instance")
    path.mkdir(parents=True, exist_ok=True)
    if bm25 is None:
        raise ValueError("BM25 model cannot be None")
    if not isinstance(chunks, list):
        raise TypeError("Chunks must be a list")
    with open(path / "index.pkl", "wb") as file:
        pickle.dump(bm25, file)
    with open(path / "corpus.pkl", "wb") as file:
        pickle.dump(chunks, file)
    print(f"Ingestion complete! Indexed {len(chunks)} chunks under {path}/")

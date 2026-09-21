"""imports and orchestrates the corpus walk + index building + persistence"""


import bm25s
import pickle
from pathlib import Path
from tqdm import tqdm

from src.schemas import IndexedChunk
from src.chunker import chunkinizer, extract_corpus_files


def bm25_scorer(chunks: list[IndexedChunk]) -> bm25s.BM25:
    """mambo"""
    retriever = bm25s.BM25()
    tokenized_corpus = [
        bm25s.tokenize([chunk.text], show_progress=False).ids[0]
        for chunk in tqdm(
            chunks, desc="Tokenizing", total=len(chunks))
    ]
    retriever.index(tokenized_corpus)
    return retriever


def chunk_collector(raw_dir: Path, max_chunk_size: int) -> list[IndexedChunk]:
    """mambo"""
    chunks: list[IndexedChunk] = []
    files = extract_corpus_files(raw_dir)
    for f in tqdm(files, desc="Chunking content", total=len(files)):
        chunks.extend(chunkinizer(f, max_chunk_size))
    return chunks


def index_exporter(
        bm25: bm25s.BM25, chunks: list[IndexedChunk], path: Path) -> None:
    """mambo"""
    with open(path / "index.pkl", "wb") as file:
        pickle.dump(bm25, file)
    with open(path / "corpus.pkl", "wb") as file:
        pickle.dump(chunks, file)

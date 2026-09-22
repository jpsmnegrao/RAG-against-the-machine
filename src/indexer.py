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
    texts = [chunk.text for chunk in chunks]
    with tqdm(total=len(texts), desc="Tokenizing", unit=" chunk") as bar:
        tokenized_corpus = bm25s.tokenize(texts, show_progress=False)
        bar.update(len(texts))
    retriever.index(tokenized_corpus)
    return retriever


def chunk_collector(raw_dir: Path, max_chunk_size: int) -> list[IndexedChunk]:
    """mambo"""
    chunks: list[IndexedChunk] = []
    files = extract_corpus_files(raw_dir)
    for f in tqdm(files, desc="Chunking", total=len(files), unit=" file"):
        chunks.extend(chunkinizer(f, max_chunk_size))
    return chunks


def index_exporter(
        bm25: bm25s.BM25, chunks: list[IndexedChunk], path: Path) -> None:
    """mambo"""
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "index.pkl", "wb") as file:
        pickle.dump(bm25, file)
    with open(path / "corpus.pkl", "wb") as file:
        pickle.dump(chunks, file)
    print(f"Ingestion complete! Indexed {len(chunks)} chunks under {path}/")

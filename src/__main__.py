"""Entry point for running the package as a module."""


import fire
from pathlib import Path
from src.indexer import chunk_collector, bm25_scorer, index_exporter


def index(max_chunk_size: int = 2000) -> None:
    """jambo"""
    corpus = chunk_collector(Path("data/raw"), max_chunk_size)
    tokenized_index = bm25_scorer(corpus)
    index_exporter(tokenized_index, corpus, Path("data/processed"))


if __name__ == "__main__":
    fire.Fire({
        "index": index,
    })

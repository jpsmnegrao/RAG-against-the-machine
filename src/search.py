"""Retrieval-Augmented Generation (RAG) search and indexing interface module.

Handles loading search artifacts, executing single-query and batch dataset
searches, and triggering index generation workflows.
"""

import pickle
import bm25s
import json
from tqdm import tqdm
from pathlib import Path
from typing import Any

from src.schemas import MinimalSearchResults, StudentSearchResults
from src.indexer import chunk_collector, bm25_scorer, index_exporter


def load_search_artifacts(
        artifacts_dir: str = "data/processed") -> tuple[Any, list[Any]]:
    """Load the serialized BM25 index and corpus chunks from disk.

    Args:
        artifacts_dir: Directory path containing the saved index artifacts.

    Returns:
        A tuple containing the loaded BM25 index and corpus list.
    """
    artifacts_path = Path(artifacts_dir)
    with open(artifacts_path / "index.pkl", "rb") as f:
        index = pickle.load(f)
    with open(artifacts_path / "corpus.pkl", "rb") as f:
        corpus = pickle.load(f)
    return index, corpus


def index(max_chunk_size: int = 2000) -> None:
    """Execute the corpus indexing pipeline with validation.

    Args:
        max_chunk_size: Maximum character count per generated text chunk.
    """
    if isinstance(max_chunk_size, bool):
        print("Error: Using the flag with no value is a no go (reverts to True"
              " = 1).")
        return
    if not isinstance(max_chunk_size, int):
        print("Max_chunk_size value must be an int.")
        return
    elif max_chunk_size <= 0:
        print("Error: --max_chunk_size requires a positive integer value.")
        return
    try:
        corpus = chunk_collector(Path("data/raw"), max_chunk_size)
        tokenized_index = bm25_scorer(corpus)
        index_exporter(tokenized_index, corpus, Path("data/processed"))
    except (ValueError, FileNotFoundError) as error:
        print(f"Could not index the data given: {error}")


def search(query: str, k: int = 5) -> None:
    """Perform a BM25 similarity search for a single query string.

    Args:
        query: Search query string.
        k: Maximum number of top relevant chunks to retrieve.
    """
    if isinstance(k, bool) or k <= 0 or not isinstance(k, int):
        print("Error: --k requires a positive integer value.")
        return
    try:
        bm25_index, corpus = load_search_artifacts("data/processed/")
    except (FileNotFoundError, pickle.UnpicklingError) as error:
        print(f"Could not load index: {error}")
        return
    k = min(k, len(corpus))
    query_tokens = bm25s.tokenize([query])
    results, scores = bm25_index.retrieve(query_tokens, k=k)
    top_chunks = [corpus[i] for i in results[0]]
    for chunk in top_chunks:
        print(
            chunk.minimalsource.file_path,
            f"[{chunk.minimalsource.first_character_index}:"
            f"{chunk.minimalsource.last_character_index}]")


def search_dataset(dataset_path: Path, k: int, save_directory: Path) -> None:
    """Execute search queries over an entire evaluation dataset.

    Args:
        dataset_path: Path to the JSON dataset file containing questions.
        k: Maximum number of results to retrieve per query.
        save_directory: Directory path where search outputs will be saved.
    """
    dataset_path = Path(dataset_path)
    if isinstance(save_directory, bool):
        raise ValueError("--save_directory requires a path argument.")
    save_directory = Path(save_directory) / dataset_path.name
    if not dataset_path.exists() or not dataset_path.is_file():
        print("Dataset load failed: [Errno 2] No such file or directory: "
              f"'{dataset_path}'")
        return
    try:
        bm25_index, corpus = load_search_artifacts("data/processed/")
    except (FileNotFoundError, pickle.UnpicklingError) as error:
        print(f"Dataset load failed: {error}")
        return
    try:
        with open(dataset_path, encoding="utf-8") as file:
            data = json.load(file)
            queries = data["rag_questions"]
    except (
            FileNotFoundError, json.JSONDecodeError, PermissionError, KeyError
            ) as error:
        print(f"Dataset load failed: {error}")
        return
    results_list = []
    for query in tqdm(queries, desc="Searching", unit=" question"):
        query_tokens = bm25s.tokenize([query["question"]])
        results, scores = bm25_index.retrieve(query_tokens, k=k)
        top_chunks = [corpus[i] for i in results[0]]
        retrieved_sources = [chunk.minimalsource for chunk in top_chunks]
        results_list.append(
            MinimalSearchResults(
                question_id=query["question_id"],
                question=query["question"],
                retrieved_sources=retrieved_sources,
            ))
    student_search_results = (
        StudentSearchResults(search_results=results_list, k=k))
    output_dir = save_directory.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(save_directory, "w", encoding="utf-8") as file:
        json.dump(student_search_results.model_dump(), file, indent=4)
    print(f"Saved student_search_results to {save_directory}")

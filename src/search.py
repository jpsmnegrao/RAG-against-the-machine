"""mambo jambo"""


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
    """jambo"""
    artifacts_path = Path(artifacts_dir)
    try:
        with open(artifacts_path / "index.pkl", "rb") as f:
            index = pickle.load(f)
        with open(artifacts_path / "corpus.pkl", "rb") as f:
            corpus = pickle.load(f)
    except (
            FileNotFoundError, json.JSONDecodeError, PermissionError, KeyError
            ) as error:
        print(f"Dataset load failed: {error}")
        return None, []
    return index, corpus


def index(max_chunk_size: int = 2000) -> None:
    """jambo"""
    try:
        corpus = chunk_collector(Path("data/raw"), max_chunk_size)
        tokenized_index = bm25_scorer(corpus)
        index_exporter(tokenized_index, corpus, Path("data/processed"))
    except ValueError as error:
        print(f"Could not index the data given: {error}")


def search(query: str, k: int = 5) -> None:
    """jambo"""
    bm25_index, corpus = load_search_artifacts("data/processed/")
    query_tokens = bm25s.tokenize([query])
    results, scores = bm25_index.retrieve(query_tokens, k=k)
    top_chunks = [corpus[i] for i in results[0]]
    for chunk in top_chunks:
        print(
            chunk.minimalsource.file_path,
            f"[{chunk.minimalsource.first_character_index}:"
            f"{chunk.minimalsource.last_character_index}]")


def search_dataset(dataset_path: Path, k: int, save_directory: Path) -> None:
    """jambo"""
    dataset_path = Path(dataset_path)
    if isinstance(save_directory, bool):
        raise ValueError("--save_directory requires a path argument.")
    save_directory = Path(save_directory) / dataset_path.name
    bm25_index, corpus = load_search_artifacts("data/processed/")
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

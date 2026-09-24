"""mambo jambo"""


import json
from pathlib import Path
from pydantic import ValidationError
from typing import Any

from src.schemas import StudentSearchResults, RagDataset


def IoU_calculation(s1: dict[str, Any], s2: dict[str, Any]) -> float:
    """jambo"""
    start1, end1 = s1["first_character_index"], s1["last_character_index"]
    start2, end2 = s2["first_character_index"], s2["last_character_index"]
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    if union == 0:
        return 0.0
    return float(intersection / union)


def question_recall(
        ground_truth_srcs: list[dict[str, Any]],
        retrieved: list[dict[str, Any]],
        k: int) -> float:
    """mambo"""
    if not ground_truth_srcs:
        return 0.0
    top_k = retrieved[:k]
    matched = 0
    for src in ground_truth_srcs:
        for retrieved_src in top_k:
            if (src["file_path"] == retrieved_src["file_path"]
                    and IoU_calculation(src, retrieved_src) >= 0.05):
                matched += 1
                break
    return matched / len(ground_truth_srcs)


def evaluate(student_search_results_path: Path, dataset_path: Path) -> None:
    """mambo"""
    try:
        with open(student_search_results_path, "r", encoding="utf-8") as file:
            raw_search_data = json.load(file)
        with open(dataset_path, "r", encoding="utf-8") as file:
            raw_benchmark_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError) as error:
        print(f"Dataset load failed: {error}")
        return
    is_valid = True
    try:
        student_data = StudentSearchResults(**raw_search_data)
    except ValidationError:
        is_valid = False
        student_data = None
    try:
        benchmark_data = RagDataset(**raw_benchmark_data)
    except ValidationError as error:
        print(f"Ground truth dataset failed validation: {error}")
        return
    print(f"Student data is valid: {is_valid}")
    if not is_valid or student_data is None:
        return
    search_results = student_data.search_results
    rag_questions = benchmark_data.rag_questions
    benchmark_lookup: dict[str, Any] = {
        item.question_id: item for item in rag_questions
    }
    total_questions = len(search_results)
    total_with_sources = sum(
        1 for item in rag_questions if getattr(item, "sources", None))
    total_with_student_sources = sum(
        1 for item in search_results if item.retrieved_sources)
    print(f"Total number of questions: {total_questions}")
    print(f"Total number of questions with sources: {total_with_sources}")
    print("Total number of questions with student sources: "
          f"{total_with_student_sources}")
    if total_questions == 0:
        print("No search results were found!")
        return
    ks = [1, 3, 5, 10]
    recalls: dict[int, list[float]] = {k: [] for k in ks}
    evaluated_count = 0
    for item in search_results:
        q_id = item.question_id
        if q_id not in benchmark_lookup:
            continue
        benchmark_item = benchmark_lookup[q_id]
        ground_truth_srcs = [
            source.model_dump()
            for source in getattr(benchmark_item, "sources", [])
        ]
        retrieved = [source.model_dump() for source in item.retrieved_sources]
        evaluated_count += 1
        for k in ks:
            recalls[k].append(question_recall(ground_truth_srcs, retrieved, k))
    print("\nEvaluation Results")
    print("========================================")
    print(f"Questions evaluated: {evaluated_count}")
    for k in ks:
        avg_recall = (
            sum(recalls[k]) / len(recalls[k]) if recalls[k] else 0.0)
        print(f"Recall@{k}: {avg_recall:.3f} ({avg_recall * 100:.1f}%)")
    print({f"recall@{k}": (sum(recalls[k]) / len(recalls[k]) if recalls[k]
           else 0.0) for k in ks})

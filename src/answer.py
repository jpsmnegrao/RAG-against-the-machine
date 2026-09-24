"""mambo jambo"""


import json
import bm25s
import torch
from tqdm import tqdm
from pathlib import Path
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.search import load_search_artifacts
from src.schemas import MinimalAnswer, StudentSearchResultsAndAnswer


def initialize_model() -> tuple[Any, Any]:
    """jambo"""
    model = (
        AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",
                                             device_map="auto"))
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    return model, tokenizer


def retrieve_sources(question: str, k: int) -> list[dict[str, str]]:
    """Retrieve top matching corpus chunks and read their source files."""
    try:
        bm25_index, corpus = load_search_artifacts("data/processed/")
    except (FileNotFoundError, json.JSONDecodeError) as error:
        print(f"Index load failed: {error}")
        return []
    query_tokens = bm25s.tokenize([question])
    results, _ = bm25_index.retrieve(query_tokens, k=k)
    if results is None or len(results) == 0 or len(results[0]) == 0:
        print("[Retrieval Info]: No matching documents found in index.")
        return []
    sources = []
    for idx in results[0]:
        chunk = corpus[idx]
        ms = chunk.minimalsource
        try:
            with open(ms.file_path, "r", encoding="utf-8") as file:
                content = file.read()
            sources.append({
                "file_path": ms.file_path,
                "text": content[
                    ms.first_character_index:ms.last_character_index]
            })
        except (
                FileNotFoundError, PermissionError, UnicodeDecodeError
                ) as error:
            print(f"Source read failed for {ms.file_path}: {error}")
            continue
    if not sources:
        print("[Retrieval Info]: Retrieved documents found, but all source "
              "file reads failed.")
    return sources


def parse_model_output(raw_output: str) -> str:
    """Extract final answer text."""
    if "</think>" in raw_output:
        parts = raw_output.split("</think>")
        answer_text = parts[1].strip()
    else:
        answer_text = raw_output.strip()
    return answer_text


def answer(question: str, k: int) -> None:
    """mambo"""
    sources = retrieve_sources(question, k)
    if not sources:
        print("\nAnswer:\nThe provided context does not contain this "
              "information.")
        return
    context_blocks = [
        f"Source ({s['file_path']}):\n{s['text']}" for s in sources]
    context = "\n\n".join(context_blocks)
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer the question using only the provided context and explicitly "
        "cite the source file path when referencing information."
    )
    model, tokenizer = initialize_model()
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1024)
    raw_answer_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    answer_text = parse_model_output(raw_answer_text)
    print(f"Answer:\n{answer_text}")


def answer_dataset(
        student_search_results_path: Path, save_directory: Path) -> None:
    """jambo"""
    unanswered_questions_path = Path(student_search_results_path)
    save_directory = Path(save_directory) / unanswered_questions_path.name
    model, tokenizer = initialize_model()
    model = torch.compile(model, dynamic=True)
    try:
        with open(student_search_results_path, encoding="utf-8") as file:
            data = json.load(file)
            queries = data["search_results"]
    except (
            FileNotFoundError, json.JSONDecodeError, PermissionError, KeyError
            ) as error:
        print(f"Dataset load failed: {error}")
        return
    answers_list = []
    for item in tqdm(queries, desc="Answering", unit=" question"):
        question_id = item["question_id"]
        question = item["question"]
        retrieved_sources = item["retrieved_sources"]
        context_blocks = []
        for source in retrieved_sources:
            file_path = source["file_path"]
            first_idx = source.get("first_character_index")
            last_idx = source.get("last_character_index")
            if file_path is None or first_idx is None or last_idx is None:
                continue
            try:
                with open(str(file_path), "r", encoding="utf-8") as f:
                    content = f.read()
                    chunk_text = content[first_idx:last_idx]
                    context_blocks.append(f"Source ({file_path}):\n"
                                          f"{chunk_text}")
            except (FileNotFoundError, PermissionError,
                    UnicodeDecodeError) as error:
                print(f"Source read failed for {file_path}: {error}")
        context = "\n\n".join(context_blocks)
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer the question using only the provided context and "
            "explicitly cite the source file path when referencing "
            "information."
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
                enable_thinking=False
            ).to(model.device)  # type: ignore[attr-defined]
            outputs = model.generate(    # type: ignore[attr-defined]
                **inputs, max_new_tokens=256)
        except Exception as e:
            print(f"Error processing query: {e}")
            continue
        raw_answer_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True)
        answer_text = parse_model_output(raw_answer_text)
        answers_list.append(
            MinimalAnswer(
                question_id=question_id,
                question=question,
                retrieved_sources=retrieved_sources,
                answer=answer_text
            ))
    student_search_results_and_answer = StudentSearchResultsAndAnswer(
        search_results=answers_list,
        k=len(queries[0]["retrieved_sources"]) if queries else 0
    )
    output_dir = save_directory.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(save_directory, "w", encoding="utf-8") as file:
        json.dump(student_search_results_and_answer.model_dump(),
                  file, indent=4)
    print(f"Loaded {len(queries)} questions ... Processed {len(answers_list)} "
          f"of {len(queries)}")
    print(f"Saved student_search_results_and_answer to {save_directory}")

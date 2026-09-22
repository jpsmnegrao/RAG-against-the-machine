"""mambo jambo"""


import json
import bm25s
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.search import load_search_artifacts


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
    if (results is None or len(results) == 0
            or getattr(results[0], "size", len(results[0])) == 0):
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
    return sources


def parse_model_output(raw_output: str) -> str:
    """Extract final answer text, printing reasoning trace if present."""
    if "</think>" in raw_output:
        parts = raw_output.split("</think>")
        reasoning = parts[0].replace("<think>", "").strip()
        answer_text = parts[1].strip()
        print(f"[Reasoning Trace]:\n{reasoning}\n")
    else:
        answer_text = raw_output.strip()
    return answer_text


def answer(question: str, k: int) -> None:
    """mambo"""
    sources = retrieve_sources(question, k)
    if not sources:
        print("\nAnswer:\nThe provided context doesn't contain this "
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

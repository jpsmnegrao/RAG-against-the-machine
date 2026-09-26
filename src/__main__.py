"""Command-line interface entry point for the RAG indexing and search pipeline.

Powered by Python Fire to automatically expose module functions as CLI
commands.
"""

import fire

from src.search import index, search, search_dataset
from src.answer import answer, answer_dataset
from src.evaluate import evaluate

if __name__ == "__main__":
    fire.Fire({
        "index": index,
        "search": search,
        "search_dataset": search_dataset,
        "answer": answer,
        "answer_dataset": answer_dataset,
        "evaluate": evaluate
    })

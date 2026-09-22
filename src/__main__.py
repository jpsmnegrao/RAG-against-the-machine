"""Entry point for running the package as a module."""


import fire

from src.search import index, search, search_dataset
from src.answer import answer

if __name__ == "__main__":
    fire.Fire({
        "index": index,
        "search": search,
        "search_dataset": search_dataset,
        "answer": answer
    })

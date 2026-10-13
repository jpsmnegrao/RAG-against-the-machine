.PHONY: install run debug clean lint lint-strict fclean test

export UV_CACHE_DIR := $(CURDIR)/.uv-cache
export UV_PROJECT_ENVIRONMENT := $(CURDIR)/.venv
export HF_HOME := $(CURDIR)/.hf-cache
export TORCH_HOME := $(CURDIR)/.torch-cache
export PYTORCH_CUDA_ALLOC_CONF := expandable_segments:True

all: install

install:
	uv sync
	@echo "=== Installation complete! ==="

run:
	uv run -m src

debug:
	uv run -m pdb -m src

clean:
	@rm -rf .mypy_cache .pytest_cache/ ; \
	find . -type d -name "__pycache__" -exec rm -rf {} + ; \
	echo "Cleanup complete!";

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

# Extra rules:

query:
	uv run -m src search "How to configure OpenAI server?" --k 10

answer:
	uv run -m src answer "How to configure OpenAI server?" --k 10

evaluate:
	uv run -m src evaluate \
		--student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json \
		--dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json	

fclean: clean
	@rm -rf .venv data/processed data/output
	@echo "Removed virtual environment and output files - deep clean complete!"

test:
	uv run pytest


pipeline-run:
	uv run python -m src index --max_chunk_size 2000
	uv run python -m src search_dataset \
		--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
		--k 10 \
		--save_directory data/output/search_results/UnansweredQuestions
	uv run python -m src evaluate \
		--student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
		--dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

pipeline-run-with-answers: pipeline-run
	uv run python -m src answer_dataset \
		--student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
		--save_directory data/output/search_results_and_answer/UnansweredQuestions

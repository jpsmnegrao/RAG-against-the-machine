.PHONY: install run debug clean lint lint-strict fclean test

export HF_HOME := $(CURDIR)/.hf-cache
export TORCH_HOME := $(CURDIR)/.torch-cache

all: install

install:
	uv sync
	@echo "=== Installation complete! ==="

run:
	uv run -m src index --max_chunk_size 2000
	uv run -m src search_dataset \
			--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
			--k 10 \
			--save_directory data/output/search_results/UnansweredQuestions
	uv run -m src search_dataset \
			--dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
			--k 10 \
			--save_directory data/output/search_results/UnansweredQuestions

make answer_dataset:
	uv run -m src answer_dataset \
			--student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
			--save_directory data/output/search_results_and_answer/UnansweredQuestions
	uv run -m src answer_dataset \
			--student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json \
			--save_directory data/output/search_results_and_answer/UnansweredQuestions

debug:
	uv run python -m pdb -m src index --max_chunk_size 2000
	uv run python -m pdb -m src search_dataset \
			--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
			--k 105 \
			--save_directory data/output/search_results/UnansweredQuestions
	uv run python -m pdb -m src answer_dataset \
			--student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
			--save_directory data/output/search_results_and_answer/UnansweredQuestions

clean:
	@rm -rf .mypy_cache .pytest_cache/ ; \
	find . -type d -name "__pycache__" -exec rm -rf {} + ; \
	echo "Cleanup complete!";

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

query:
	uv run -m src search "How to configure OpenAI server?" --k 10

answer:
	uv run -m src answer "How to configure OpenAI server?" --k 10

evaluate:
	uv run -m src evaluate \
		--student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
		--dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json	

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

fclean: clean
	@rm -rf .venv data/processed data/output
	@echo "Removed virtual environment and output files - deep clean complete!"

test:
	uv run pytest

.PHONY: install run debug clean lint lint-strict fclean test

export HF_HOME := $(CURDIR)/.hf-cache
export TORCH_HOME := $(CURDIR)/.torch-cache

all: install

install:
	uv sync
	@echo "=== Installation complete! ==="

run:


debug:


clean:
	@rm -rf .mypy_cache .pytest_cache/ ; \
	find . -type d -name "__pycache__" -exec rm -rf {} + ; \
	echo "Cleanup complete!"; \

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

fclean: clean
	@rm -rf .venv

	@echo "Removed virtual environment and output files - deep clean complete!"

test:
	uv run pytest

"""Tests for the answer functions."""


import pytest
from typing import Any
from pathlib import Path
from src.answer import answer, answer_dataset


class MockModel:
    """Mock model to avoid slow LLM loading during unit tests."""
    def generate(self, *args: Any, **kwargs: Any) -> str:
        return "Mocked answer"


def test_answer_fails_gracefully_without_index(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify answer handles missing index files gracefully."""
    def raise_not_found(_: str) -> None:
        raise FileNotFoundError("Missing")
    monkeypatch.setattr("src.answer.load_search_artifacts", raise_not_found)
    answer("What is Python?", k=5)


def test_answer_empty_query() -> None:
    """Verify answering an empty query executes without crashing."""
    answer("", k=5)


def test_answer_k_zero() -> None:
    """Verify setting k=0 handles retrieval gracefully."""
    answer("What is Python?", k=0)


def test_answer_dataset_nonexistent_path(
        capsys: pytest.CaptureFixture[str]) -> None:
    """Verify answer_dataset handles a missing input path gracefully."""
    answer_dataset(Path("nonexistent_dataset.json"), Path("outputs"))
    captured = capsys.readouterr()
    assert "Dataset load failed" in captured.out


def test_answer_dataset_malformed_json(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify answer_dataset handles malformed JSON without tracebacks."""
    monkeypatch.setattr(
        "src.answer.initialize_model", lambda: (MockModel(), None))
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{ malformed json")
    save_dir = tmp_path / "out"
    answer_dataset(bad_file, save_dir)
    assert not (save_dir / "bad.json").exists()


def test_answer_dataset_missing_source_file(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify answer_dataset handles missing sources cleanly."""
    monkeypatch.setattr(
        "src.answer.initialize_model", lambda: (MockModel(), None))
    dataset_file = tmp_path / "dataset.json"
    content = (
        '{"search_results": [{"question_id": "1", "question": "test", '
        '"retrieved_sources": [{"file_path": "missing.py", '
        '"first_character_index": 0, "last_character_index": 10}]}]}'
    )
    dataset_file.write_text(content)
    save_dir = tmp_path / "out"
    answer_dataset(dataset_file, save_dir)
    assert save_dir.exists()


def test_answer_dataset_generation_failure_does_not_crash_batch(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify model generation errors on items do not crash the overall run."""
    monkeypatch.setattr(
        "src.answer.initialize_model", lambda: (MockModel(), None))
    dataset_file = tmp_path / "dataset.json"
    content = (
        '{"search_results": [{"question_id": "1", "question": "test", '
        '"retrieved_sources": []}]}'
    )
    dataset_file.write_text(content)
    save_dir = tmp_path / "out"
    answer_dataset(dataset_file, save_dir)
    assert save_dir.exists()

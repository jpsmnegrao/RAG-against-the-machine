"""Tests for the search functions."""


import pytest
from pathlib import Path

from src.search import search, search_dataset, index


def test_search_fails_gracefully_without_index(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify search handles missing index files gracefully."""
    def raise_not_found(_: str) -> None:
        raise FileNotFoundError("Missing")
    monkeypatch.setattr("src.search.load_search_artifacts", raise_not_found)
    search("What is Python?", k=5)


def test_search_empty_query(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        sample_python: str) -> None:
    """Verify searching with an empty query does not crash."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "sample.py").write_text(sample_python)
    index(max_chunk_size=200)
    search("", k=5)


def test_search_gibberish_query(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        sample_python: str) -> None:
    """Verify searching with a gibberish query returns without crashing."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "sample.py").write_text(sample_python)
    index(max_chunk_size=200)
    search("xyzabc123nonexistentterm", k=5)


def test_search_k_zero(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        sample_python: str) -> None:
    """Verify searching with k=0 is rejected cleanly."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "sample.py").write_text(sample_python)
    index(max_chunk_size=200)
    search("What is Python?", k=0)


def test_search_negative_k(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        sample_python: str) -> None:
    """Verify searching with a negative k is rejected cleanly."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "sample.py").write_text(sample_python)
    index(max_chunk_size=200)
    search("What is Python?", k=-1)


def test_search_dataset_nonexistent_path(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify search_dataset raises an error for a nonexistent path."""
    search_dataset(Path("nonexistent.json"), k=5, save_directory=tmp_path)
    captured = capsys.readouterr()
    assert "Dataset load failed" in captured.out


def test_search_dataset_fails_gracefully_without_index(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify search_dataset handles missing index files gracefully."""
    def raise_not_found(_: str) -> None:
        raise FileNotFoundError("Missing")
    monkeypatch.setattr("src.search.load_search_artifacts", raise_not_found)
    search_dataset(Path("dataset.json"), k=5, save_directory=tmp_path)


def test_search_dataset_malformed_json(tmp_path: Path) -> None:
    """Verify search_dataset handles malformed JSON without crashing."""
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{ malformed json")
    search_dataset(bad_file, k=5, save_directory=tmp_path)


def test_search_dataset_missing_keys(tmp_path: Path) -> None:
    """Verify search_dataset handles datasets missing expected keys."""
    bad_file = tmp_path / "dataset.json"
    bad_file.write_text('{"wrong_key": []}')
    search_dataset(bad_file, k=5, save_directory=tmp_path)


def test_index_handles_empty_corpus(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """Verify indexing a genuinely empty corpus fails gracefully."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    index(max_chunk_size=2000)
    captured = capsys.readouterr()
    assert "Could not index the data given" in captured.out


def test_index_invalid_max_chunk_size(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """Test that index handles a bool max_chunk_size gracefully."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    index(max_chunk_size=True)
    captured = capsys.readouterr()
    assert "Error" in captured.out


def test_index_boolean_flag_guard(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """Test index intercepts boolean flags without values."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    index(max_chunk_size=True)
    captured = capsys.readouterr()
    assert "Error: Using the flag with no value is a no go" in captured.out


def test_index_non_positive_integer_guard(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """Test index rejects non-positive integers for max_chunk_size."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    index(max_chunk_size=0)
    captured = capsys.readouterr()
    assert ("Error: --max_chunk_size requires a positive integer value"
            in captured.out)

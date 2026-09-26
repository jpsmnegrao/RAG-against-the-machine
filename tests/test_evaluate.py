"""Tests for the answer functions."""


import pytest
from pathlib import Path
from src.evaluate import evaluate, IoU_calculation


def test_evaluate_nonexistent_student_path(tmp_path: Path) -> None:
    """Verify evaluation handles a nonexistent student file gracefully."""
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text('{"questions": []}')
    missing_student = tmp_path / "missing.json"
    evaluate(missing_student, dataset_file)


def test_evaluate_nonexistent_dataset_path(tmp_path: Path) -> None:
    """Verify evaluation handles a nonexistent dataset file gracefully."""
    student_file = tmp_path / "student.json"
    student_file.write_text('{"search_results": []}')
    missing_dataset = tmp_path / "missing.json"
    evaluate(student_file, missing_dataset)


def test_evaluate_malformed_student_json(tmp_path: Path) -> None:
    """Verify evaluation handles malformed student JSON without crashing."""
    student_file = tmp_path / "student.json"
    student_file.write_text("{ malformed student json")
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text('{"questions": []}')
    evaluate(student_file, dataset_file)


def test_evaluate_malformed_dataset_json(tmp_path: Path) -> None:
    """Verify evaluation handles malformed dataset JSON without crashing."""
    student_file = tmp_path / "student.json"
    student_file.write_text('{"search_results": []}')
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text("{ malformed dataset json")
    evaluate(student_file, dataset_file)


def test_evaluate_student_data_invalid_schema(tmp_path: Path) -> None:
    """Verify evaluation handles invalid schema in student data."""
    student_file = tmp_path / "student.json"
    student_file.write_text('{"invalid_key": []}')
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text('{"questions": []}')
    evaluate(student_file, dataset_file)


def test_evaluate_dataset_invalid_schema(tmp_path: Path) -> None:
    """Verify evaluation handles invalid schema in the gold dataset."""
    student_file = tmp_path / "student.json"
    student_file.write_text('{"search_results": [], "k": 5}')
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text('{"invalid_key": []}')
    evaluate(student_file, dataset_file)


def test_evaluate_empty_search_results(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify evaluation handles empty search results list."""
    student_file = tmp_path / "student.json"
    student_file.write_text('{"search_results": [], "k": 5}')
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text('{"rag_questions": []}')
    evaluate(student_file, dataset_file)
    captured = capsys.readouterr()
    assert "No search results were found!" in captured.out


def test_evaluate_no_overlapping_question_ids(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify evaluation handles non-overlapping question IDs."""
    student_file = tmp_path / "student.json"
    student_file.write_text(
        '{"search_results": [{"question_id": "q1", "question": "test", '
        '"retrieved_sources": []}], "k": 5}'
    )
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text(
        '{"rag_questions": [{"question_id": "q2", "question": "test", '
        '"sources": [], "answer": "n/a"}]}'
    )
    evaluate(student_file, dataset_file)
    captured = capsys.readouterr()
    assert "Questions evaluated: 0" in captured.out


def test_iou_calculation_known_values() -> None:
    """Verify IoU calculation returns correct values for known intervals."""
    span_a = {"first_character_index": 0, "last_character_index": 10}
    span_b = {"first_character_index": 5, "last_character_index": 15}
    iou = IoU_calculation(span_a, span_b)
    assert iou == pytest.approx(5 / 15)

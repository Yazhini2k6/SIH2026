"""Tests for the main pipeline CLI program."""

import json
import sys
import pymupdf as fitz
import pytest
from sih2026.main import main, process_document


@pytest.fixture
def test_pdf(tmp_path):
    pdf_path = tmp_path / "test_course.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "UNIT I\nMachine Learning\nSupervised Learning")
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_process_document_success(test_pdf, monkeypatch, tmp_path):
    # Change working directory so output lands in tmp_path/data/output
    monkeypatch.chdir(tmp_path)
    output_json = process_document(test_pdf)

    assert output_json.exists()
    assert output_json.name == "test_course.json"

    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["filename"] == "test_course.pdf"
    assert len(data["pages"]) == 1
    assert data["pages"][0]["page_number"] == 1
    assert "Machine Learning" in data["pages"][0]["text"]
    assert data["pages"][0]["extraction_method"] == "text"


def test_main_no_args(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sih2026.main"])
    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Usage:" in captured.out


def test_main_missing_file(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sih2026.main", "non_existent.pdf"])
    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.err

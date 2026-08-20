"""Tests for PDF text extraction."""

import pymupdf as fitz
import pytest
from sih2026.extraction.exceptions import (
    NoTextExtractedError,
    UnsupportedFileTypeError,
)
from sih2026.extraction.pdf import extract_pdf


@pytest.fixture
def sample_pdf(tmp_path):
    pdf_path = tmp_path / "sample_syllabus.pdf"
    doc = fitz.open()

    # Page 1
    page1 = doc.new_page()
    page1.insert_text((50, 50), "UNIT I\nIntroduction to Computer Science\nBasic Concepts")

    # Page 2
    page2 = doc.new_page()
    page2.insert_text((50, 50), "UNIT II\nData Structures\nArrays and Lists")

    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def empty_text_pdf(tmp_path):
    pdf_path = tmp_path / "empty.pdf"
    doc = fitz.open()
    doc.new_page()  # Page without text
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_extract_pdf_success(sample_pdf):
    doc = extract_pdf(sample_pdf)

    assert doc.filename == "sample_syllabus.pdf"
    assert len(doc.pages) == 2

    assert doc.pages[0].page_number == 1
    assert "UNIT I" in doc.pages[0].text
    assert "Introduction to Computer Science" in doc.pages[0].text
    assert doc.pages[0].extraction_method == "text"

    assert doc.pages[1].page_number == 2
    assert "UNIT II" in doc.pages[1].text


def test_extract_pdf_missing_file():
    with pytest.raises(FileNotFoundError):
        extract_pdf("non_existent_file.pdf")


def test_extract_pdf_unsupported_extension(tmp_path):
    fake_txt = tmp_path / "file.txt"
    fake_txt.write_text("hello")
    with pytest.raises(UnsupportedFileTypeError):
        extract_pdf(fake_txt)


def test_extract_pdf_no_text(empty_text_pdf):
    with pytest.raises(NoTextExtractedError):
        extract_pdf(empty_text_pdf)

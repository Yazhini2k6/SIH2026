"""Tests for Document and DocumentPage models."""

import pytest
from sih2026.models.document import Document, DocumentPage


def test_document_page_creation():
    page = DocumentPage(
        page_number=1, text="UNIT I\nIntroduction", extraction_method="text"
    )
    assert page.page_number == 1
    assert page.text == "UNIT I\nIntroduction"
    assert page.extraction_method == "text"


def test_document_creation():
    page1 = DocumentPage(page_number=1, text="Page 1", extraction_method="text")
    page2 = DocumentPage(page_number=2, text="Page 2", extraction_method="text")
    doc = Document(filename="syllabus.pdf", pages=[page1, page2])

    assert doc.filename == "syllabus.pdf"
    assert len(doc.pages) == 2
    assert doc.pages[0].page_number == 1
    assert doc.pages[1].text == "Page 2"


def test_document_serialization():
    page = DocumentPage(page_number=1, text="Sample Text", extraction_method="text")
    doc = Document(filename="test.pdf", pages=[page])

    dump = doc.model_dump()
    assert dump["filename"] == "test.pdf"
    assert dump["pages"][0]["page_number"] == 1
    assert dump["pages"][0]["text"] == "Sample Text"

"""Tests for conservative text cleaning and local LLM formatting."""

from sih2026.models.document import Document, DocumentPage
from sih2026.processing.cleaner import clean_document, clean_text


def test_clean_text_null_characters():
    dirty = "UNIT I\x00 Introduction to Python\x00"
    cleaned = clean_text(dirty)
    assert "\x00" not in cleaned
    assert cleaned == "UNIT I Introduction to Python"


def test_clean_text_excessive_spaces_and_tabs():
    dirty = "UNIT   I\t\tIntroduction   to   Data   Structures   "
    cleaned = clean_text(dirty)
    assert cleaned == "UNIT I Introduction to Data Structures"


def test_clean_text_excessive_blank_lines():
    dirty = "UNIT I\n\n\n\n\nIntroduction\n\n\nTopics"
    cleaned = clean_text(dirty)
    assert cleaned == "UNIT I\n\nIntroduction\n\nTopics"


def test_clean_text_preserves_line_breaks_and_casing():
    text = "UNIT I: Python Basics\n- Variables & Types\n- Control Flow"
    cleaned = clean_text(text)
    assert cleaned == text


def test_clean_document():
    doc = Document(
        filename="test.pdf",
        pages=[
            DocumentPage(
                page_number=1,
                text="Page   1\x00  text\n\n\n\nmore text",
                extraction_method="text",
            )
        ],
    )
    cleaned_doc = clean_document(doc)
    assert cleaned_doc.filename == "test.pdf"
    assert cleaned_doc.pages[0].text == "Page 1 text\n\nmore text"
    assert cleaned_doc.pages[0].lines == ["Page 1 text", "more text"]
    assert cleaned_doc.pages[0].word_count == 5
    assert cleaned_doc.total_pages == 1
    assert cleaned_doc.total_words == 5
    assert "--- DOCUMENT START: test.pdf ---" in cleaned_doc.llm_input_context
    assert "[PAGE 1]" in cleaned_doc.llm_input_context
    assert "Page 1 text" in cleaned_doc.llm_input_context

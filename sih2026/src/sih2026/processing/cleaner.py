"""Conservative text cleaner module formatted for local LLM consumption."""

import re
from sih2026.models.document import Document, DocumentPage


def clean_text(text: str) -> str:
    """Conservatively cleans raw extracted text.

    Removes null characters, excessive horizontal spaces/tabs, and excessive
    blank lines while keeping meaningful line breaks, punctuation, and wording intact.

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # 1. Remove null characters
    text = text.replace("\x00", "")

    # 2. Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Clean horizontal whitespace line-by-line while preserving line breaks
    cleaned_lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.split("\n")]
    text = "\n".join(cleaned_lines)

    # 4. Collapse 3 or more consecutive blank lines into 2 (1 empty line gap)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def build_llm_input_context(filename: str, pages: list[DocumentPage]) -> str:
    """Formats document pages into a clean, delimited prompt block for local LLMs."""
    blocks = [f"--- DOCUMENT START: {filename} ---"]
    for page in pages:
        blocks.append(f"\n[PAGE {page.page_number}]")
        blocks.append(page.text)
    blocks.append("\n--- DOCUMENT END ---")
    return "\n".join(blocks)


def clean_document(document: Document) -> Document:
    """Returns a new Document with conservatively cleaned page texts and local LLM fields.

    Args:
        document: The input Document instance.

    Returns:
        Document: A new Document with cleaned page text, line arrays, word counts, and LLM context.
    """
    cleaned_pages: list[DocumentPage] = []
    for page in document.pages:
        c_text = clean_text(page.text)
        lines = [line for line in c_text.split("\n") if line.strip()]
        word_count = len(c_text.split())
        cleaned_pages.append(
            DocumentPage(
                page_number=page.page_number,
                text=c_text,
                extraction_method=page.extraction_method,
                word_count=word_count,
                lines=lines,
            )
        )

    total_pages = len(cleaned_pages)
    total_words = sum(p.word_count for p in cleaned_pages)
    llm_context = build_llm_input_context(document.filename, cleaned_pages)

    return Document(
        filename=document.filename,
        total_pages=total_pages,
        total_words=total_words,
        pages=cleaned_pages,
        llm_input_context=llm_context,
    )

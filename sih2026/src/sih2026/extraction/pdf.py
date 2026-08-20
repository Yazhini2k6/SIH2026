"""PDF text extraction module using PyMuPDF."""

from pathlib import Path
import pymupdf as fitz

from sih2026.extraction.exceptions import (
    InvalidDocumentError,
    NoTextExtractedError,
    UnsupportedFileTypeError,
)
from sih2026.models.document import Document, DocumentPage


def extract_pdf(file_path: str | Path) -> Document:
    """Extracts text page-by-page from a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Document: A Document object containing page-level extracted text.

    Raises:
        FileNotFoundError: If the file does not exist or is not a file.
        UnsupportedFileTypeError: If the file extension is not '.pdf'.
        InvalidDocumentError: If the PDF is corrupted or invalid.
        NoTextExtractedError: If no extractable text was found across all pages.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"Path is not a regular file: {path}")

    if path.suffix.lower() != ".pdf":
        raise UnsupportedFileTypeError(
            f"Unsupported file extension '{path.suffix}'. Expected '.pdf'"
        )

    try:
        pdf_doc = fitz.open(path)
    except Exception as exc:
        raise InvalidDocumentError(f"Failed to open PDF file '{path}': {exc}") from exc

    pages: list[DocumentPage] = []
    has_any_text = False

    try:
        if len(pdf_doc) == 0:
            raise InvalidDocumentError(f"PDF file '{path}' contains no pages")

        for page_idx in range(len(pdf_doc)):
            page = pdf_doc[page_idx]
            page_text = page.get_text("text")
            if page_text and page_text.strip():
                has_any_text = True
            pages.append(
                DocumentPage(
                    page_number=page_idx + 1,
                    text=page_text,
                    extraction_method="text",
                )
            )
    finally:
        pdf_doc.close()

    if not has_any_text:
        # Note: OCR stage can be integrated here in future implementation
        raise NoTextExtractedError(
            f"PDF file '{path}' contains no extractable text."
        )

    return Document(filename=path.name, pages=pages)

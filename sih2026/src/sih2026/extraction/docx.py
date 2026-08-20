"""DOCX text extraction module using python-docx."""

from pathlib import Path
import docx
from docx.opc.exceptions import OpcError

from sih2026.extraction.exceptions import (
    InvalidDocumentError,
    NoTextExtractedError,
    UnsupportedFileTypeError,
)
from sih2026.models.document import Document, DocumentPage


def extract_docx(file_path: str | Path) -> Document:
    """Extracts text from a DOCX file while preserving paragraph and table order.

    Args:
        file_path: Path to the DOCX file.

    Returns:
        Document: A Document object containing page-level extracted text.

    Raises:
        FileNotFoundError: If the file does not exist.
        UnsupportedFileTypeError: If the extension is not '.docx'.
        InvalidDocumentError: If the DOCX is corrupted or invalid.
        NoTextExtractedError: If no extractable text was found.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"Path is not a regular file: {path}")

    if path.suffix.lower() != ".docx":
        raise UnsupportedFileTypeError(
            f"Unsupported file extension '{path.suffix}'. Expected '.docx'"
        )

    try:
        doc = docx.Document(path)
    except (OpcError, Exception) as exc:
        raise InvalidDocumentError(
            f"Failed to open DOCX file '{path}': {exc}"
        ) from exc

    content_blocks: list[str] = []

    for element in doc.element.body:
        if element.tag.endswith("p"):
            p = docx.text.paragraph.Paragraph(element, doc)
            content_blocks.append(p.text)
        elif element.tag.endswith("tbl"):
            table = docx.table.Table(element, doc)
            table_lines: list[str] = []
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells]
                row_str = " | ".join(c for c in row_cells if c)
                if row_str:
                    table_lines.append(row_str)
            if table_lines:
                content_blocks.append("\n".join(table_lines))

    full_text = "\n".join(content_blocks)

    if not full_text.strip():
        raise NoTextExtractedError(
            f"DOCX file '{path}' contains no extractable text."
        )

    pages = [
        DocumentPage(
            page_number=1,
            text=full_text,
            extraction_method="docx",
        )
    ]

    return Document(filename=path.name, pages=pages)

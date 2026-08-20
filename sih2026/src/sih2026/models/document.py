"""Document data models using Pydantic, formatted for local LLM consumption."""

from pydantic import BaseModel, Field


class DocumentPage(BaseModel):
    """Represents a single page in a document, structured for AI processing."""

    page_number: int = Field(..., description="1-indexed page number of the document")
    text: str = Field(..., description="Extracted text content of the page")
    extraction_method: str = Field(
        ..., description="Method used to extract text (e.g., 'text', 'docx')"
    )
    word_count: int = Field(
        default=0, description="Word count for local LLM token context estimation"
    )
    lines: list[str] = Field(
        default_factory=list,
        description="Non-empty text lines on this page for structured extraction",
    )


class Document(BaseModel):
    """Represents a full document containing page-level data and LLM prompt context."""

    filename: str = Field(..., description="Name of the document file")
    total_pages: int = Field(
        default=0, description="Total number of pages in the document"
    )
    total_words: int = Field(
        default=0, description="Total word count across all document pages"
    )
    pages: list[DocumentPage] = Field(
        default_factory=list, description="List of document pages"
    )
    llm_input_context: str = Field(
        default="",
        description="Pre-formatted document string ready for direct insertion into local LLM prompts",
    )

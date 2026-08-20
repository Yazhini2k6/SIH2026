"""Custom exceptions for document extraction."""


class DocumentExtractionError(Exception):
    """Base exception for all document extraction errors."""

    pass


class UnsupportedFileTypeError(DocumentExtractionError, ValueError):
    """Raised when an unsupported file format or extension is provided."""

    pass


class InvalidDocumentError(DocumentExtractionError, ValueError):
    """Raised when a document is corrupted or cannot be parsed."""

    pass


class NoTextExtractedError(DocumentExtractionError, ValueError):
    """Raised when a document yields no extractable text."""

    pass

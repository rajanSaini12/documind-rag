"""Domain exceptions. The API layer maps these to HTTP status codes."""


class DocuMindError(Exception):
    """Base class for all DocuMind errors."""


class IngestionError(DocuMindError):
    """The uploaded file could not be parsed or contained no extractable text."""


class DocumentNotFoundError(DocuMindError):
    """No document with the given id exists in the vector store."""


class DocumentExistsError(DocuMindError):
    """The exact same file (same content hash) has already been ingested."""

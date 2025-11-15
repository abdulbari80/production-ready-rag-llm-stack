class RAGBaseException(Exception):
    """
    Base class for all custom exceptions used in the RAG (Retrieval-Augmented Generation) pipeline.

    This class enables consistent exception handling across the system by providing
    a common parent type for all RAG-specific errors.
    """
    pass


class VectorStoreNotInitializedError(RAGBaseException):
    """
    Exception raised when an operation is attempted on a vector store
    that has not been properly initialized or loaded.

    Parameters
    ----------
    message : str, optional
        Descriptive error message indicating the initialization issue.
    """
    def __init__(self, message="Vector store not initialized or loaded."):
        super().__init__(message)


class DocumentLoadError(RAGBaseException):
    """
    Exception raised when the system fails to load, read, or parse documents.

    Parameters
    ----------
    message : str, optional
        Error message describing the document loading or parsing failure.
    """
    def __init__(self, message="Failed to load or parse documents."):
        super().__init__(message)


class EmbeddingModelError(RAGBaseException):
    """
    Exception raised when embedding generation fails due to issues with the
    embedding model, input formatting, or external API communication.

    Parameters
    ----------
    message : str, optional
        The descriptive message associated with the embedding failure.
    """
    def __init__(self, message="Error generating embeddings from model."):
        super().__init__(message)


class RetrievalError(RAGBaseException):
    """
    Exception raised when a retrieval or similarity search operation fails.

    This may occur due to issues with vector search, FAISS index loading,
    or malformed query embeddings.

    Parameters
    ----------
    message : str, optional
        Description of the retrieval-related failure.
    """
    def __init__(self, message="Error during document retrieval."):
        super().__init__(message)

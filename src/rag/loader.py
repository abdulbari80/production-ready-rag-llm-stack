from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from pathlib import Path
from .config import settings
from src.rag.logger import get_logger
from src.rag.exception import DocumentLoadError

logger = get_logger(__name__)

CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap


class TxtLoader:
    """
    Loader for reading plain text (TXT) files and converting them into
    chunked LangChain `Document` objects suitable for downstream processing
    in a Retrieval-Augmented Generation (RAG) pipeline.

    This class handles:
    - Scanning a directory for `.txt` files
    - Loading their contents using LangChain loaders
    - Splitting them into overlapping chunks using a recursive character splitter

    Parameters
    ----------
    chunk_size : int
        Maximum size of each document chunk (in characters).
    chunk_overlap : int
        Number of overlapping characters between consecutive chunks.
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def load_txt(self, dir_path: str) -> List[Document]:
        """
        Load all `.txt` files from a directory and split them into
        overlapping `Document` chunks.

        Parameters
        ----------
        dir_path : str
            Path to the directory containing `.txt` files.

        Returns
        -------
        List[Document]
            A list of processed `Document` chunks ready for embedding or indexing.

        Notes
        -----
        - Uses LangChain's `TextLoader` to read each file.
        - Logs key loading steps.
        - Raises DocumentLoadError on failure.
        """
        documents = []

        try:
            txt_files = list(Path(dir_path).glob("*.txt"))
            if not txt_files:
                logger.warning(f"No TXT files found in directory: {dir_path}")
        except Exception as e:
            logger.error(f"Failed to read directory {dir_path}: {e}")
            raise DocumentLoadError("Unable to scan directory for TXT files.") from e

        for i, file in enumerate(txt_files, start=1):
            try:
                loader = TextLoader(file_path=str(file))
                document = loader.load()
                documents.extend(document)
                logger.info(f"Section {i}: Loaded {len(document)} pages from {file.name}")
            except Exception as e:
                logger.error(f"Error loading file '{file}': {e}")
                raise DocumentLoadError(f"Failed to load file: {file}") from e

        try:
            chunks = self.text_splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"Error splitting documents into chunks: {e}")
            raise DocumentLoadError("Failed during text chunking process.") from e

        return chunks

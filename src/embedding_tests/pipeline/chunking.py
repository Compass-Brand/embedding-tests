"""Text chunking strategies for RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)


class ChunkingStrategy(Enum):
    """Available text chunking strategies."""

    RECURSIVE = "recursive"
    SENTENCE = "sentence"
    TOKEN = "token"


@dataclass
class ChunkMetadata:
    """A text chunk with metadata."""

    text: str
    source_doc_id: str
    chunk_index: int


def chunk_text(
    text: str,
    *,
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    source_doc_id: str = "",
) -> list[ChunkMetadata]:
    """Split text into chunks using the specified strategy."""
    if strategy == ChunkingStrategy.RECURSIVE:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        raw_chunks = splitter.split_text(text)
    elif strategy == ChunkingStrategy.SENTENCE:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[". ", "\n\n", "\n", " ", ""],
        )
        raw_chunks = splitter.split_text(text)
    elif strategy == ChunkingStrategy.TOKEN:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda t: len(t.split()),
        )
        raw_chunks = splitter.split_text(text)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    return [
        ChunkMetadata(text=chunk, source_doc_id=source_doc_id, chunk_index=i)
        for i, chunk in enumerate(raw_chunks)
    ]

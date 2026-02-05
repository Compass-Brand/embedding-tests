"""Text chunking strategies for RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkingStrategy(Enum):
    """Available text chunking strategies."""

    RECURSIVE = "recursive"
    SENTENCE = "sentence"
    TOKEN = "token"  # Uses word count (len(text.split())) as proxy for tokens


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
    """Split text into chunks using the specified strategy.

    For the TOKEN strategy, ``chunk_size`` and ``chunk_overlap`` are measured
    in word counts (using ``str.split()``) rather than character counts.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
        )
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
        # TOKEN uses word count as length function; separators are library defaults
        # (["\n\n", "\n", " ", ""]) since primary control is chunk_size via length_function
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

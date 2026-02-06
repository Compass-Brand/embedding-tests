"""Text chunking strategies for RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkingStrategy(Enum):
    """Available text chunking strategies."""

    RECURSIVE = "recursive"
    SENTENCE = "sentence"
    WORD_COUNT = "word_count"  # Uses word count (len(text.split())) as proxy for tokens


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

    For the SENTENCE strategy, sentence boundaries are approximated using
    ". " as the primary separator rather than NLP-based segmentation.

    For the WORD_COUNT strategy, ``chunk_size`` and ``chunk_overlap`` are measured
    in word counts (using ``str.split()``) rather than character counts.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
        )
    splitter_kwargs: dict[str, object] = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    match strategy:
        case ChunkingStrategy.RECURSIVE:
            splitter_kwargs["separators"] = ["\n\n", "\n", ". ", " ", ""]
        case ChunkingStrategy.SENTENCE:
            splitter_kwargs["separators"] = [". ", "\n\n", "\n", " ", ""]
        case ChunkingStrategy.WORD_COUNT:
            # Word count is a rough proxy for tokens; for precise counting, integrate a model tokenizer
            splitter_kwargs["length_function"] = lambda t: len(t.split())
        case _:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    splitter = RecursiveCharacterTextSplitter(**splitter_kwargs)
    raw_chunks = splitter.split_text(text)

    return [
        ChunkMetadata(text=chunk, source_doc_id=source_doc_id, chunk_index=i)
        for i, chunk in enumerate(raw_chunks)
    ]

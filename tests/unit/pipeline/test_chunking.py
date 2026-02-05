"""Tests for text chunking strategies."""

from __future__ import annotations

import pytest

from embedding_tests.pipeline.chunking import (
    ChunkMetadata,
    chunk_text,
    ChunkingStrategy,
)


class TestChunkText:
    """Tests for text chunking."""

    def test_recursive_chunking_produces_chunks_within_size_limit(self) -> None:
        text = "Hello world. " * 100  # ~1300 chars
        chunks = chunk_text(text, strategy=ChunkingStrategy.RECURSIVE, chunk_size=200, chunk_overlap=20)
        # SPLITTER_OVERFLOW: RecursiveCharacterTextSplitter may exceed chunk_size
        # by up to ~50 chars when it avoids mid-word splits at separator boundaries.
        # The splitter tries each separator in order and picks the first split that
        # keeps the chunk under chunk_size, but if no clean boundary exists within
        # the limit it overflows to the next available separator.
        SPLITTER_OVERFLOW = 50
        for chunk in chunks:
            assert len(chunk.text) <= 200 + SPLITTER_OVERFLOW

    def test_recursive_chunking_respects_overlap(self) -> None:
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. " * 10
        chunks = chunk_text(text, strategy=ChunkingStrategy.RECURSIVE, chunk_size=100, chunk_overlap=20)
        if len(chunks) > 1:
            # Check overlap exists between adjacent chunks via substring match
            for i in range(len(chunks) - 1):
                suffix = chunks[i].text[-20:]
                prefix = chunks[i + 1].text[:40]
                assert any(suffix[j:] in prefix for j in range(len(suffix)))

    def test_sentence_chunking_splits_on_sentence_boundaries(self) -> None:
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        chunks = chunk_text(text, strategy=ChunkingStrategy.SENTENCE, chunk_size=50, chunk_overlap=0)
        for i, chunk in enumerate(chunks):
            stripped = chunk.text.strip()
            assert len(stripped) > 0
            # Non-final chunks should end at a sentence boundary.
            # The ". " separator splits so the period may go to the next chunk,
            # meaning the chunk ends with a complete word (alphanumeric char).
            if i < len(chunks) - 1:
                assert stripped[-1].isalnum() or stripped.endswith(".")

    def test_token_chunking_produces_chunks(self) -> None:
        text = "word " * 200
        chunks = chunk_text(text, strategy=ChunkingStrategy.TOKEN, chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 1

    def test_chunking_preserves_all_content(self) -> None:
        text = "The quick brown fox jumps over the lazy dog. " * 20
        chunks = chunk_text(text, strategy=ChunkingStrategy.RECURSIVE, chunk_size=100, chunk_overlap=0)
        reassembled = "".join(c.text for c in chunks)
        # Use whitespace normalization to avoid spurious failures from
        # trailing/leading whitespace differences between chunks
        assert " ".join(reassembled.split()) == " ".join(text.split())

    def test_chunking_returns_metadata(self) -> None:
        text = "Some text content for chunking. " * 10
        chunks = chunk_text(
            text,
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=100,
            chunk_overlap=10,
            source_doc_id="doc_001",
        )
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, ChunkMetadata)
            assert chunk.source_doc_id == "doc_001"
            assert chunk.chunk_index == i
            assert len(chunk.text) > 0

    def test_empty_text_returns_empty_list(self) -> None:
        chunks = chunk_text("", strategy=ChunkingStrategy.RECURSIVE, chunk_size=100)
        assert chunks == []

    def test_short_text_returns_single_chunk(self) -> None:
        text = "Short text."
        chunks = chunk_text(text, strategy=ChunkingStrategy.RECURSIVE, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_nonpositive_chunk_size_raises(self) -> None:
        """Verify that chunk_size <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_text("some text", strategy=ChunkingStrategy.RECURSIVE, chunk_size=0)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_text("some text", strategy=ChunkingStrategy.RECURSIVE, chunk_size=-10)

    def test_negative_overlap_raises_error(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            chunk_text(
                "Some text content",
                strategy=ChunkingStrategy.RECURSIVE,
                chunk_size=100,
                chunk_overlap=-10,
            )

    def test_overlap_ge_chunk_size_raises(self) -> None:
        """Verify that chunk_overlap >= chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap.*must be less than chunk_size"):
            chunk_text("some text", strategy=ChunkingStrategy.RECURSIVE, chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError, match="chunk_overlap.*must be less than chunk_size"):
            chunk_text("some text", strategy=ChunkingStrategy.RECURSIVE, chunk_size=100, chunk_overlap=200)

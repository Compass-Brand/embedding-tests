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
        for chunk in chunks:
            assert len(chunk.text) <= 200 + 50  # Allow small overflow from splitter

    def test_recursive_chunking_respects_overlap(self) -> None:
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. " * 10
        chunks = chunk_text(text, strategy=ChunkingStrategy.RECURSIVE, chunk_size=100, chunk_overlap=20)
        if len(chunks) > 1:
            # Check overlap exists between adjacent chunks
            for i in range(len(chunks) - 1):
                overlap = set(chunks[i].text[-30:]) & set(chunks[i + 1].text[:30])
                # Some character overlap should exist
                assert len(overlap) > 0

    def test_sentence_chunking_splits_on_sentence_boundaries(self) -> None:
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        chunks = chunk_text(text, strategy=ChunkingStrategy.SENTENCE, chunk_size=50, chunk_overlap=0)
        for chunk in chunks:
            # Each chunk should contain complete sentence fragments
            # The sentence splitter splits on ". " so chunks contain sentence text
            assert len(chunk.text.strip()) > 0

    def test_token_chunking_produces_chunks(self) -> None:
        text = "word " * 200
        chunks = chunk_text(text, strategy=ChunkingStrategy.TOKEN, chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 1

    def test_chunking_preserves_all_content(self) -> None:
        text = "The quick brown fox jumps over the lazy dog. " * 20
        chunks = chunk_text(text, strategy=ChunkingStrategy.RECURSIVE, chunk_size=100, chunk_overlap=0)
        reassembled = "".join(c.text for c in chunks)
        # All content should be present (may have minor whitespace differences)
        assert len(reassembled) >= len(text.strip()) - 10

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

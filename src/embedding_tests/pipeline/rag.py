"""Full RAG pipeline orchestrator."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from embedding_tests.models.base import EmbeddingModel, RerankerModel
from embedding_tests.pipeline.chunking import ChunkingStrategy, chunk_text
from embedding_tests.pipeline.embedding import batch_embed
from embedding_tests.pipeline.reranking import rerank_results
from embedding_tests.pipeline.retrieval import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Results for a single query."""

    query_id: str
    query_text: str
    retrieved_doc_ids: list[str]
    scores: list[float]
    relevant_doc_ids: list[str]


@dataclass
class RagResult:
    """Results of a complete RAG pipeline run."""

    query_results: list[QueryResult]
    total_time_seconds: float
    used_reranker: bool
    num_corpus_chunks: int = 0
    embedding_time_seconds: float = 0.0


class RagPipeline:
    """Orchestrates the full RAG pipeline: chunk -> embed -> index -> query -> retrieve."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        *,
        reranker_model: RerankerModel | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 10,
        reranker_top_k: int = 3,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    ) -> None:
        self._embedding_model = embedding_model
        self._reranker = reranker_model
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._top_k = top_k
        self._reranker_top_k = reranker_top_k
        self._strategy = chunking_strategy

    def run(
        self,
        corpus: list[dict[str, Any]],
        queries: list[dict[str, Any]],
    ) -> RagResult:
        """Run the full RAG pipeline."""
        start = time.perf_counter()

        # Validate corpus documents
        for i, doc in enumerate(corpus):
            if "text" not in doc or "doc_id" not in doc:
                raise ValueError(
                    f"Corpus document at index {i} missing required 'text' or 'doc_id' key"
                )

        # Validate query documents
        for i, q in enumerate(queries):
            if "text" not in q:
                raise ValueError(
                    f"Query document at index {i} missing required 'text' key"
                )

        # 1. Chunk the corpus
        all_chunks: list[dict[str, str]] = []
        for doc in corpus:
            chunks = chunk_text(
                doc["text"],
                strategy=self._strategy,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                source_doc_id=doc["doc_id"],
            )
            for chunk in chunks:
                all_chunks.append({
                    "doc_id": chunk.source_doc_id,
                    "text": chunk.text,
                    "chunk_index": str(chunk.chunk_index),
                })

        # 2. Embed corpus chunks
        chunk_texts = [c["text"] for c in all_chunks]
        embed_result = batch_embed(self._embedding_model, chunk_texts, batch_size=32)

        # 3. Index in vector store
        dim = self._embedding_model.get_embedding_dim()
        store = VectorStore(
            collection_name=f"rag_{id(self)}",
            embedding_dim=dim,
        )
        chunk_ids = [f"{c['doc_id']}_chunk_{c['chunk_index']}" for c in all_chunks]
        store.index(embed_result.embeddings, chunk_ids)

        # Build chunk_lookup once outside the query loop
        chunk_lookup = {
            f"{c['doc_id']}_chunk_{c['chunk_index']}": c
            for c in all_chunks
        }

        # 4. Query and retrieve
        query_results: list[QueryResult] = []
        for q in queries:
            q_embed = batch_embed(
                self._embedding_model, [q["text"]], batch_size=1, is_query=True
            )
            retrieved = store.query(q_embed.embeddings[0], top_k=self._top_k)

            retrieved_doc_ids = [r.doc_id.rsplit("_chunk_", 1)[0] for r in retrieved]
            scores = [r.score for r in retrieved]

            # 5. Optional reranking
            if self._reranker is not None:
                retrieved_docs: list[dict[str, str]] = []
                for r in retrieved:
                    if r.doc_id in chunk_lookup:
                        retrieved_docs.append({
                            "doc_id": chunk_lookup[r.doc_id]["doc_id"],
                            "text": chunk_lookup[r.doc_id]["text"],
                        })
                    else:
                        logger.warning(
                            "Chunk ID '%s' not found in chunk_lookup, skipping",
                            r.doc_id,
                        )
                if retrieved_docs:
                    reranked = rerank_results(
                        q["text"], retrieved_docs, self._reranker,
                        top_k=self._reranker_top_k,
                    )
                    retrieved_doc_ids = [r.doc_id for r in reranked]
                    scores = [r.score for r in reranked]

            query_results.append(QueryResult(
                query_id=q.get("query_id", ""),
                query_text=q["text"],
                retrieved_doc_ids=retrieved_doc_ids,
                scores=scores,
                relevant_doc_ids=q.get("relevant_doc_ids", []),
            ))

        elapsed = time.perf_counter() - start
        store.clear()

        return RagResult(
            query_results=query_results,
            total_time_seconds=elapsed,
            used_reranker=self._reranker is not None,
            num_corpus_chunks=len(all_chunks),
            embedding_time_seconds=embed_result.total_time_seconds,
        )

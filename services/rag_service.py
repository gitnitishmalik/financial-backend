"""
rag_service.py  –  Lightweight RAG pipeline (no external vector DB required)

Flow:
  1. Chunk documents into ~400-token passages
  2. Embed via EmbeddingService (shared all-MiniLM-L6-v2 — loaded only once)
  3. At query time: embed the query, cosine-rank chunks, return top-k

Reusing EmbeddingService avoids loading the 80 MB model twice.
"""

from __future__ import annotations

from typing import List

import numpy as np

from services.embedding_service import EmbeddingService

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 400   # words per chunk (~proxy for tokens)
CHUNK_OVERLAP = 80    # overlap to preserve context at boundaries
TOP_K         = 6     # chunks returned per query


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split *text* into overlapping word-level chunks."""
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# RAG Service
# ---------------------------------------------------------------------------

class RAGService:
    """
    In-memory RAG service backed by the shared EmbeddingService.

    Usage
    -----
    rag = RAGService()
    rag.ingest(texts)             # list of raw document strings
    context = rag.retrieve(query) # top-k relevant passages as one string
    """

    # Shared across all instances — model loads only once per process
    _shared_embedder: "EmbeddingService | None" = None

    def __init__(self) -> None:
        if RAGService._shared_embedder is None:
            RAGService._shared_embedder = EmbeddingService()
        self._embedder: EmbeddingService = RAGService._shared_embedder

        self._chunks: List[str] = []
        self._embeddings: "np.ndarray | None" = None
        self._ready: bool = False

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, texts: List[str]) -> None:
        """Chunk and embed a list of raw document strings."""
        self._chunks = []
        self._embeddings = None
        self._ready = False

        for text in texts:
            if text.strip():
                self._chunks.extend(chunk_text(text))

        if not self._chunks:
            return

        try:
            self._embeddings = self._embedder.embed(self._chunks)
            # Normalise for cosine similarity via dot product
            norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self._embeddings = self._embeddings / norms
            self._ready = True
        except Exception:
            self._embeddings = None

    def ingest_files(self, file_paths: List[str]) -> None:
        """Convenience: extract text from file paths then ingest."""
        from services.analysis_service import extract_text_from_file
        texts = [extract_text_from_file(fp) for fp in file_paths]
        self.ingest([t for t in texts if t])

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = TOP_K) -> str:
        """
        Return the top-k most relevant chunks as a single string.
        Falls back to first top_k chunks if embeddings are unavailable.
        """
        if not self._chunks:
            return ""

        if self._ready and self._embeddings is not None:
            try:
                q_emb = self._embedder.embed([query])[0]
                norm = np.linalg.norm(q_emb)
                if norm > 0:
                    q_emb = q_emb / norm
                scores: np.ndarray = self._embeddings @ q_emb
                top_indices = np.argsort(scores)[::-1][:top_k]
                top_indices_sorted = sorted(top_indices.tolist())
                return "\n\n---\n\n".join(self._chunks[i] for i in top_indices_sorted)
            except Exception:
                pass

        # Fallback: first top_k chunks
        return "\n\n---\n\n".join(self._chunks[:top_k])
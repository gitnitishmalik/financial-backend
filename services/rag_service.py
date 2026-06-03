"""
rag_service.py  –  Lightweight RAG pipeline with on-disk persistence.

Flow:
  1. Chunk documents into ~400-token passages
  2. Embed via EmbeddingService (shared all-MiniLM-L6-v2 — loaded only once)
  3. At query time: embed the query, cosine-rank chunks, return top-k

Persistence:
  - Each corpus is fingerprinted by sha256 of its sorted text contents.
  - Chunks + embeddings are written to .rag_cache/<fingerprint>.npz so
    they survive process restart AND are shared across services in-process
    (two instances with the same corpus load the same npz).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List

import numpy as np

from services.embedding_service import EmbeddingService

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
CACHE_DIR = Path(os.getenv("RAG_CACHE_DIR", ".rag_cache"))
CACHE_DIR.mkdir(exist_ok=True)


def _fingerprint(texts: List[str]) -> str:
    h = hashlib.sha256()
    for t in sorted(t.strip() for t in texts if t and t.strip()):
        h.update(t.encode("utf-8", errors="ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:24]

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
        """Chunk and embed a list of raw document strings.

        Result is persisted to CACHE_DIR keyed by content fingerprint, so a
        restart (or a sibling RAGService instance) skips the expensive
        embed step entirely.
        """
        self._chunks = []
        self._embeddings = None
        self._ready = False

        clean_texts = [t for t in texts if t and t.strip()]
        if not clean_texts:
            return

        # ── Try cache first ──────────────────────────────────────────────
        fp = _fingerprint(clean_texts)
        cache_path = CACHE_DIR / f"{fp}.npz"
        if cache_path.exists():
            try:
                data = np.load(cache_path, allow_pickle=True)
                self._chunks = list(data["chunks"])
                self._embeddings = data["embeddings"]
                self._ready = True
                return
            except Exception:
                pass  # fall through to re-embed

        for text in clean_texts:
            self._chunks.extend(chunk_text(text))

        if not self._chunks:
            return

        try:
            self._embeddings = self._embedder.embed(self._chunks)
            norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self._embeddings = self._embeddings / norms
            self._ready = True
            try:
                np.savez_compressed(
                    cache_path,
                    chunks=np.array(self._chunks, dtype=object),
                    embeddings=self._embeddings,
                )
            except Exception:
                pass  # cache write failure is not fatal
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
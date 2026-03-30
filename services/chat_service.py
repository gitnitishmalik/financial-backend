"""
chat_service.py  –  Streaming chat with RAG-powered document context

Instead of dumping raw document text into every message (wastes tokens),
we use RAGService to retrieve only the relevant passages per user message.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncGenerator, List

from groq import AsyncGroq

from core.config import settings
from services.rag_service import RAGService
from services.analysis_service import extract_text_from_file

SYSTEM_PROMPT = """You are FinAnalyzer AI, a world-class financial analyst assistant powered by Groq.

Your expertise covers:
- Financial statement analysis (income statement, balance sheet, cash flow)
- Investment valuation (DCF, comparables, precedent transactions)
- Risk assessment (market, credit, liquidity, operational, regulatory)
- Portfolio management and asset allocation
- Market dynamics, macroeconomics, and sector analysis

Guidelines:
- Always cite specific numbers from documents when available
- Structure responses clearly with markdown headers and bullet points
- Give direct, actionable insights — no hedging without reason
- Flag risks prominently with severity labels (LOW / MEDIUM / HIGH / CRITICAL)
- Use professional financial terminology accurately
- When uncertain, say so clearly rather than guessing

If document context is provided, base your analysis on it.
For general financial questions, draw on your training knowledge."""


class ChatService:
    """
    Streaming AI chat using Groq's AsyncGroq client.
    Uses RAG to inject only relevant document passages — not the full text.
    Maintains per-session conversation history in memory.
    """

    def __init__(self) -> None:
        self.sessions: dict[str, list] = {}
        # One RAGService per ChatService instance (shares the embedding model)
        self._rag = RAGService()
        self._ingested_paths: list[str] = []   # track what's currently loaded

    def _get_client(self) -> AsyncGroq:
        return AsyncGroq(api_key=settings.GROQ_API_KEY)

    def _ensure_ingested(self, file_paths: List[str]) -> None:
        """Re-ingest only when the file set changes."""
        if sorted(file_paths) != sorted(self._ingested_paths):
            texts = [extract_text_from_file(fp) for fp in file_paths]
            self._rag.ingest([t for t in texts if t])
            self._ingested_paths = list(file_paths)

    def _build_doc_context(self, message: str, file_paths: List[str]) -> str:
        """Retrieve the most relevant passages for *message* via RAG."""
        if not file_paths:
            return ""
        self._ensure_ingested(file_paths)
        context = self._rag.retrieve(message, top_k=5)
        if not context:
            return ""
        return (
            "\n\n[RELEVANT DOCUMENT CONTEXT]\n"
            f"{context}\n"
            "[END DOCUMENT CONTEXT]\n"
        )

    # ------------------------------------------------------------------
    # Public streaming interface
    # ------------------------------------------------------------------

    async def stream_response(
        self,
        message: str,
        file_paths: List[str],
        session_id: str,
    ) -> AsyncGenerator[str, None]:
        """Yield JSON-encoded text chunks for the SSE stream."""

        doc_context = self._build_doc_context(message, file_paths)
        user_content = message + doc_context if doc_context else message

        history = self.sessions.get(session_id, [])
        history.append({"role": "user", "content": user_content})

        # Keep last 16 messages to stay within token limits
        if len(history) > 16:
            history = history[-16:]

        async for chunk in self._stream_groq(history, session_id):
            yield chunk

    async def _stream_groq(
        self, history: list, session_id: str
    ) -> AsyncGenerator[str, None]:
        client       = self._get_client()
        full_response = ""
        messages     = [{"role": "system", "content": SYSTEM_PROMPT}] + history

        try:
            stream = await client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=messages,
                stream=True,
                max_tokens=2048,
                temperature=0.3,
                top_p=0.9,
            )
            async for chunk in stream:
                text = chunk.choices[0].delta.content or ""
                if text:
                    full_response += text
                    yield json.dumps({"text": text})

        except Exception as e:
            error_msg = (
                f"\n\n**Error calling Groq API:** {e}\n\n"
                "Check that your `GROQ_API_KEY` in `.env` is valid."
            )
            yield json.dumps({"text": error_msg})
            full_response = error_msg

        finally:
            if full_response:
                hist = self.sessions.get(session_id, [])
                hist.append({"role": "assistant", "content": full_response})
                if len(hist) > 20:
                    hist = hist[-20:]
                self.sessions[session_id] = hist

    def clear_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
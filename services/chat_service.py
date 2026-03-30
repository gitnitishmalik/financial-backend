import asyncio
from typing import AsyncGenerator, List
from pathlib import Path
import json

from groq import AsyncGroq
import pdfplumber

from core.config import settings


# ── Document text extraction ──────────────────────────────────────

def extract_text(file_path: str) -> str:
    """Extract text from PDF, CSV, or TXT for chat context."""
    path = Path(file_path)
    if not path.exists():
        return ""
    try:
        if path.suffix.lower() == ".pdf":
            text = ""
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages[:15]:          # first 15 pages
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            return text[:12000]
        elif path.suffix.lower() in (".csv", ".txt"):
            return path.read_text(encoding="utf-8", errors="ignore")[:12000]
        elif path.suffix.lower() in (".xlsx", ".xls"):
            import openpyxl
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            rows = []
            for sheet in wb.worksheets:
                for row in sheet.iter_rows(values_only=True):
                    rows.append("\t".join(str(c) if c is not None else "" for c in row))
            return "\n".join(rows)[:12000]
    except Exception as e:
        return f"[Could not read file: {e}]"
    return ""


# ── System prompt ─────────────────────────────────────────────────

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

If documents are provided, base your analysis on them. For general financial questions, draw on your training knowledge."""


# ── Chat service ──────────────────────────────────────────────────

class ChatService:
    """
    Streaming AI chat using Groq's AsyncGroq client.
    Maintains per-session conversation history in memory.
    """

    def __init__(self):
        self.sessions: dict[str, list] = {}

    def _get_client(self) -> AsyncGroq:
        return AsyncGroq(api_key=settings.GROQ_API_KEY)

    def _build_doc_context(self, file_paths: List[str]) -> str:
        """Extract and combine text from all provided documents."""
        if not file_paths:
            return ""
        texts = []
        for fp in file_paths:
            text = extract_text(fp)
            if text:
                filename = Path(fp).name
                texts.append(f"=== Document: {filename} ===\n{text}")
        if not texts:
            return ""
        combined = "\n\n".join(texts)
        # Keep within Groq context window — 8k chars for document context
        return f"\n\n[DOCUMENT CONTEXT — use this to answer]\n{combined[:8000]}\n[END DOCUMENT CONTEXT]\n"

    async def stream_response(
        self,
        message: str,
        file_paths: List[str],
        session_id: str,
    ) -> AsyncGenerator[str, None]:
        """Stream AI response chunks as JSON strings."""

        doc_context = self._build_doc_context(file_paths)

        # Build user message — inject doc context only in the first message of a turn
        user_content = message
        if doc_context:
            user_content = message + doc_context

        # Get or initialise session history
        history = self.sessions.get(session_id, [])
        history.append({"role": "user", "content": user_content})

        # Keep last 16 messages to avoid hitting token limits
        if len(history) > 16:
            history = history[-16:]

        async for chunk in self._stream_groq(history, session_id):
            yield chunk

    async def _stream_groq(
        self, history: list, session_id: str
    ) -> AsyncGenerator[str, None]:
        """Call Groq API with streaming and yield JSON chunks."""
        client = self._get_client()
        full_response = ""

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

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
                delta = chunk.choices[0].delta
                text = delta.content or ""
                if text:
                    full_response += text
                    yield json.dumps({"text": text})

        except Exception as e:
            error_msg = f"\n\n**Error calling Groq API:** {str(e)}\n\nCheck that your `GROQ_API_KEY` in `.env` is valid."
            yield json.dumps({"text": error_msg})
            full_response = error_msg

        finally:
            # Always save the assistant response to session history
            if full_response:
                hist = self.sessions.get(session_id, [])
                hist.append({"role": "assistant", "content": full_response})
                # Trim history
                if len(hist) > 20:
                    hist = hist[-20:]
                self.sessions[session_id] = hist

    def clear_session(self, session_id: str):
        """Clear conversation history for a session."""
        self.sessions.pop(session_id, None)


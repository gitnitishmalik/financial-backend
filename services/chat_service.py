"""
chat_service.py  –  Tool-using streaming chat agent (Groq function calling)

The agent autonomously calls tools mid-conversation:
  search_documents   – RAG over the user's uploaded docs
  market_quote       – live price/market cap/volume for a ticker
  market_history     – OHLC trend for a ticker
  market_news        – recent headlines for a ticker
  peer_comparison    – sector peers for relative valuation
  web_search         – SEC EDGAR filings (10-K/10-Q/8-K)
  calculator         – safe arithmetic evaluator
  sql_query          – cross-document lookups against the analyses DB

Flow:
  1. Loop (max N rounds): call Groq with tools=GROQ_TOOLS, execute any
     tool_calls, append tool messages, continue.
  2. When the model stops calling tools, do a final streaming completion
     (no tools) so the user sees real-time tokens.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, List

import httpx
from groq import AsyncGroq
from sqlalchemy import select

from core.config import settings
from core.database import AsyncSessionLocal, ChatMessage, Memory
from services.analysis_service import extract_text_from_file
from services.market_service import MarketService
from services.rag_service import RAGService
from services.tools import _execute_sql_op, safe_eval

SYSTEM_PROMPT = """You are FinAnalyzer AI, a tool-using financial analyst assistant powered by Groq.

You have these tools — call them whenever they would make your answer more accurate:
- search_documents : search the user's uploaded financial documents (RAG)
- market_quote     : LIVE current price, market cap, volume, daily change
- market_history   : historical OHLC for a ticker (1d / 1w / 1mo / 3mo / 1y / 5y)
- market_news      : recent headlines for a ticker
- peer_comparison  : sector peer comparables (market cap, change_pct)
- web_search       : SEC EDGAR filings beyond Yahoo (10-K, 10-Q, 8-K)
- calculator       : safe numeric expression evaluator
- sql_query        : prior analyses + uploaded documents from the database

Rules:
- Always call the calculator for arithmetic — never compute ratios or percentages in your head.
- Always call market_quote before quoting a current stock price — document figures are stale.
- When the user references uploaded documents, call search_documents to pull exact passages.
- Use peer_comparison whenever you need a sector benchmark for valuation multiples.
- Use web_search when the user asks about recent filings or regulatory events.
- Be decisive. Use professional terminology. Flag risk severity LOW / MEDIUM / HIGH / CRITICAL.
- Output the final answer in clean markdown (headers, bullets, tables where helpful)."""


# ── Groq tool schemas (OpenAI function-calling format) ────────────────────────

GROQ_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the user's uploaded financial documents for relevant passages. "
                "Use whenever the question references uploaded docs or specific figures."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short keyword phrase, e.g. 'revenue 2023', 'operating margin'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "market_quote",
            "description": (
                "Get LIVE current market price, market cap, volume, and daily change for a ticker."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Ticker symbol, e.g. AAPL."},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "market_history",
            "description": "Historical price candles for a ticker over a period.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "period": {
                        "type": "string",
                        "enum": ["1d", "1w", "1mo", "3mo", "1y", "5y"],
                        "description": "Time window — defaults to 1mo.",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "market_news",
            "description": "Recent news headlines for a ticker (up to 8 items).",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "peer_comparison",
            "description": (
                "Sector peer comparables for a ticker — peer tickers, market caps, daily change. "
                "Use to ground sector multiples in real peer data."
            ),
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search SEC EDGAR full-text index for recent filings (10-K, 10-Q, 8-K). "
                "Use for regulatory disclosures and filings beyond Yahoo News."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text query, e.g. 'Apple revenue guidance'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate a numeric expression. ALWAYS use this for arithmetic — "
                "supports + - * / ** % and parentheses."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, e.g. '(450.2 - 380.5) / 380.5 * 100'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sql_query",
            "description": (
                "Query prior analyses and uploaded documents in the database for cross-document context. "
                "Operations: 'recent_analyses', 'high_risk_analyses', 'list_documents'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["recent_analyses", "high_risk_analyses", "list_documents"],
                    },
                    "user_id": {"type": "string"},
                    "risk_threshold": {"type": "number"},
                    "limit": {"type": "integer"},
                },
                "required": ["operation"],
            },
        },
    },
]

MAX_TOOL_ROUNDS = 4
MAX_TOOL_RESULT_CHARS = 2500


HISTORY_TURNS = 16
MAX_MEMORIES = 8


class ChatService:
    """Tool-using streaming chat agent over Groq.

    Conversation history persists to the `chat_messages` table — sessions
    survive process restarts. Per-user `memories` are injected into the
    system prompt every turn.
    """

    def __init__(self) -> None:
        self._rag = RAGService()
        self._market = MarketService()
        self._ingested_paths: list[str] = []

    # ── Persistent history + memory loaders ───────────────────────────────────

    async def _load_history(self, session_id: str, limit: int = HISTORY_TURNS) -> List[Dict[str, str]]:
        async with AsyncSessionLocal() as session:
            q = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.desc())
                .limit(limit)
            )
            rows = (await session.execute(q)).scalars().all()
        rows.reverse()  # back to chronological
        return [{"role": r.role, "content": r.content} for r in rows]

    async def _load_memories(self, user_id: str | None, limit: int = MAX_MEMORIES) -> List[Memory]:
        if not user_id:
            return []
        async with AsyncSessionLocal() as session:
            q = (
                select(Memory)
                .where(Memory.user_id == user_id)
                .order_by(Memory.importance.desc(), Memory.created_at.desc())
                .limit(limit)
            )
            return list((await session.execute(q)).scalars().all())

    async def _save_turn(
        self,
        session_id: str,
        user_id: str | None,
        user_message: str,
        assistant_message: str,
    ) -> None:
        async with AsyncSessionLocal() as session:
            session.add(ChatMessage(
                session_id=session_id, user_id=user_id, role="user", content=user_message,
            ))
            session.add(ChatMessage(
                session_id=session_id, user_id=user_id, role="assistant", content=assistant_message,
            ))
            await session.commit()

    def _memory_preamble(self, memories: List[Memory]) -> str:
        if not memories:
            return ""
        lines = "\n".join(f"  - [{m.key}] {m.content}" for m in memories)
        return (
            "\n\n[Persistent user context — known facts about this user, "
            "respect these when answering]\n" + lines
        )

    # ── Client / RAG setup ────────────────────────────────────────────────────

    def _get_client(self) -> AsyncGroq:
        return AsyncGroq(api_key=settings.GROQ_API_KEY)

    def _ensure_ingested(self, file_paths: List[str]) -> None:
        if sorted(file_paths) != sorted(self._ingested_paths):
            texts = [extract_text_from_file(fp) for fp in file_paths]
            self._rag.ingest([t for t in texts if t])
            self._ingested_paths = list(file_paths)

    # ── Tool executor ─────────────────────────────────────────────────────────

    async def _exec_tool(self, name: str, args: Dict[str, Any]) -> str:
        try:
            if name == "search_documents":
                q = str(args.get("query", "")).strip()
                if not q:
                    return "Missing 'query'."
                return self._rag.retrieve(q, top_k=3) or "No matching passages."

            if name == "calculator":
                expr = str(args.get("expression", "")).strip()
                if not expr:
                    return "Missing 'expression'."
                result = safe_eval(expr)
                return (
                    f"{result:.6f}".rstrip("0").rstrip(".")
                    if isinstance(result, float)
                    else str(result)
                )

            if name == "market_quote":
                ticker = str(args.get("ticker", "")).strip().upper()
                if not ticker:
                    return "Missing 'ticker'."
                return json.dumps(await self._market.get_quote(ticker), default=str)

            if name == "market_history":
                ticker = str(args.get("ticker", "")).strip().upper()
                period = str(args.get("period", "1mo")).strip() or "1mo"
                if not ticker:
                    return "Missing 'ticker'."
                data = await self._market.get_history(ticker, period)
                candles = data.get("candles", [])
                if data.get("error") or not candles:
                    return json.dumps(data, default=str)[:MAX_TOOL_RESULT_CHARS]
                prices = [c["close"] for c in candles if c.get("close")]
                first, last = candles[0], candles[-1]
                change_pct = (
                    round((last["close"] - first["close"]) / first["close"] * 100, 2)
                    if first["close"] else 0
                )
                return json.dumps(
                    {
                        "ticker": ticker,
                        "period": period,
                        "start_date": first["date"],
                        "end_date": last["date"],
                        "start_close": first["close"],
                        "end_close": last["close"],
                        "high": max(prices) if prices else None,
                        "low": min(prices) if prices else None,
                        "change_pct": change_pct,
                        "data_points": len(candles),
                    }
                )

            if name == "market_news":
                ticker = str(args.get("ticker", "")).strip().upper()
                if not ticker:
                    return "Missing 'ticker'."
                news = await self._market.get_news(ticker)
                if not news:
                    return f"No news for {ticker}."
                return "\n".join(
                    f"- {n.get('title', 'untitled')} ({n.get('publisher', 'unknown')}, {n.get('published', '')})"
                    for n in news[:6]
                )

            if name == "peer_comparison":
                ticker = str(args.get("ticker", "")).strip().upper()
                if not ticker:
                    return "Missing 'ticker'."
                data = await self._market.get_peer_comp(ticker)
                return json.dumps(data, default=str)

            if name == "web_search":
                return await self._edgar_search(str(args.get("query", "")))

            if name == "sql_query":
                op = str(args.get("operation", "")).strip().lower()
                if not op:
                    return "Missing 'operation'."
                return await _execute_sql_op(
                    op,
                    str(args.get("user_id", "")).strip(),
                    float(args.get("risk_threshold", 7.0)),
                    min(max(int(args.get("limit", 5) or 5), 1), 20),
                )

            return f"Unknown tool: {name}"
        except Exception as e:
            return f"Tool '{name}' failed: {e}"

    async def _edgar_search(self, query: str) -> str:
        q = query.strip()
        if not q:
            return "Missing 'query'."
        try:
            async with httpx.AsyncClient(timeout=12) as client:
                r = await client.get(
                    "https://efts.sec.gov/LATEST/search-index",
                    params={"q": q, "forms": "10-K,10-Q,8-K"},
                    headers={"User-Agent": "FinAnalyzer research@finanalyzer.app"},
                )
                r.raise_for_status()
                data = r.json()
            hits = data.get("hits", {}).get("hits", [])[:5]
            if not hits:
                return f"No EDGAR filings found for '{q}'."
            lines = []
            for h in hits:
                src = h.get("_source", {})
                names = src.get("display_names") or ["?"]
                lines.append(
                    f"- [{src.get('form', '?')}] {names[0]} "
                    f"filed {src.get('file_date', '?')} "
                    f"(adsh {src.get('adsh', '?')})"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"EDGAR search error: {e}"

    # ── Public streaming interface ────────────────────────────────────────────

    async def stream_response(
        self,
        message: str,
        file_paths: List[str],
        session_id: str,
        user_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Run the tool-calling loop, then stream the final answer as SSE chunks."""

        if file_paths:
            self._ensure_ingested(file_paths)

        # Load DB-backed history + user memories
        prior = await self._load_history(session_id)
        memories = await self._load_memories(user_id)
        history = (prior + [{"role": "user", "content": message}])[-HISTORY_TURNS:]

        system_prompt = SYSTEM_PROMPT + self._memory_preamble(memories)
        messages: List[dict] = [{"role": "system", "content": system_prompt}] + history

        client = self._get_client()
        final_text = ""

        try:
            # ── Phase 1: tool-calling loop (non-streaming) ────────────────────
            for _round in range(MAX_TOOL_ROUNDS):
                response = await client.chat.completions.create(
                    model=settings.GROQ_MODEL,
                    messages=messages,
                    tools=GROQ_TOOLS,
                    tool_choice="auto",
                    max_tokens=1024,
                    temperature=0.2,
                )
                msg = response.choices[0].message
                tool_calls = msg.tool_calls or []

                if not tool_calls:
                    break

                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments or "{}",
                            },
                        }
                        for tc in tool_calls
                    ],
                })

                for tc in tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}

                    yield json.dumps({"tool": name, "args": args})

                    result = await self._exec_tool(name, args)
                    if len(result) > MAX_TOOL_RESULT_CHARS:
                        result = result[:MAX_TOOL_RESULT_CHARS] + "\n...[truncated]"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": result,
                    })

            # ── Phase 2: final streaming pass (no tools) ──────────────────────
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
                    final_text += text
                    yield json.dumps({"text": text})

        except Exception as e:
            err = (
                f"\n\n**Error calling Groq API:** {e}\n\n"
                "Check that your `GROQ_API_KEY` in `.env` is valid."
            )
            yield json.dumps({"text": err})
            final_text = final_text or err

        finally:
            if final_text:
                try:
                    await self._save_turn(session_id, user_id, message, final_text)
                except Exception:
                    pass  # never let DB failure abort streaming

    async def clear_session(self, session_id: str) -> None:
        async with AsyncSessionLocal() as session:
            from sqlalchemy import delete
            await session.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id))
            await session.commit()
